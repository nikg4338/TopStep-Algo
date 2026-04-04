"""
execution/order_manager.py — Trade execution orchestrator.

Enforces the rigid order lifecycle defined in the DevPlan:
  1. Pre-trade check  → circuit_breakers.check_all()
  2. Size calculation  → position_sizer.calculate()
  3. Order placement   → api_client.place_order()
  4. Fill confirmation → poll / WebSocket (timeout → cancel & log)
  5. Exit orders       → stop-loss + profit-target (client-managed bracket)
  6. P&L update        → session state mutation
  7. Trade logging     → full record to logs/

This module owns position awareness: it guarantees the system never
has more than one open position at a time (OE-05) and never places
a duplicate order (REL-03).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz

from config import (
    EOD_CLOSE,
    EXECUTION_MODE,
    INSTRUMENT,
    LOG_DIR,
    ORDER_FILL_TIMEOUT_SEC,
    POINT_VALUE,
    TICK_SIZE,
    TICK_VALUE,
    TIMEZONE,
)
from execution.api_client import OrderResponse, ProjectXClient
from execution.circuit_breakers import BreakerCheckResult, CircuitBreakers
from regime.regime_state import OrderSide, OrderStatus, RegimeState
from risk.position_sizer import PositionSizer

logger = logging.getLogger(__name__)
ET = pytz.timezone(TIMEZONE)


# ── Trade record ────────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    """Immutable record of a completed trade for audit/logging."""

    trade_id: str
    timestamp_entry: str
    timestamp_exit: str | None = None
    regime: str = ""
    strategy: str = ""
    side: str = ""
    qty: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    fill_price: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ""  # "TARGET", "STOP", "TIMEOUT", "EOD_CLOSE", "MANUAL"
    order_id_entry: str = ""
    order_id_stop: str = ""
    order_id_target: str = ""
    metadata: dict = field(default_factory=dict)


# ── Session state (mutable, persisted per-trade) ────────────────────────


@dataclass
class SessionState:
    """Mutable intra-day state that survives process restarts."""

    daily_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    daily_trade_count: int = 0
    current_regime: int = 2  # default CRISIS (safe)
    account_high_water_mark: float = 0.0
    account_balance: float = 0.0
    challenge_status: str = "IN_PROGRESS"
    open_position: dict | None = None  # serialised position snapshot
    pending_exit_orders: list[str] = field(default_factory=list)
    date: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SessionState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Order Manager ───────────────────────────────────────────────────────


class OrderManager:
    """Coordinates the full trade lifecycle.

    Parameters
    ----------
    api : ProjectXClient  — already authenticated.
    breakers : CircuitBreakers
    sizer : PositionSizer
    state : SessionState  — shared mutable session state.
    """

    def __init__(
        self,
        api: ProjectXClient,
        breakers: CircuitBreakers,
        sizer: PositionSizer,
        state: SessionState,
    ) -> None:
        self.api = api
        self.breakers = breakers
        self.sizer = sizer
        self.state = state
        self._trade_log: list[TradeRecord] = []
        self._last_fill_event: dict[str, dict[str, Any]] = {}

    # ── public API ──────────────────────────────────────────────────────

    def submit_entry(
        self,
        side: OrderSide,
        stop_price: float,
        target_price: float,
        entry_price: float,
        strategy: str,
        regime: RegimeState,
        metadata: dict | None = None,
    ) -> TradeRecord | None:
        """Full lifecycle entry attempt.

        Returns a TradeRecord on fill, or None if rejected / timed-out.
        """
        now = datetime.now(ET)

        # ── Step 1: circuit breaker gate ────────────────────────────────
        check = self.breakers.check_all(
            daily_pnl=self.state.daily_pnl,
            cumulative_pnl=self.state.cumulative_pnl,
            account_balance=self.state.account_balance,
            account_high_water_mark=self.state.account_high_water_mark,
            daily_trade_count=self.state.daily_trade_count,
            active_strategy=strategy,
            current_regime=regime,
            now=now,
        )
        if not check.allowed:
            logger.warning(
                "Entry REJECTED by circuit breakers: %s", check.reasons
            )
            return None

        # ── Step 1b: no duplicate position (OE-05) ─────────────────────
        if self.state.open_position is not None:
            logger.warning("Entry REJECTED — existing position open")
            return None

        # Verify with broker
        positions = self.api.get_positions()
        if any(p.qty != 0 for p in positions):
            logger.warning("Entry REJECTED — broker reports open position")
            return None

        # ── Step 2: position sizing ─────────────────────────────────────
        stop_distance = abs(entry_price - stop_price)
        qty = self.sizer.calculate(
            stop_distance_points=stop_distance,
            mll_proximity=check.mll_proximity,
        )

        # ── Step 3: place entry order ───────────────────────────────────
        trade_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{strategy}"
        logger.info(
            "Submitting %s %s %d MES @ ~%.2f | stop=%.2f target=%.2f",
            strategy, side.value, qty, entry_price, stop_price, target_price,
        )

        try:
            entry_resp = self.api.place_order(
                side=side.value, qty=qty, order_type="MARKET"
            )
        except Exception:
            logger.exception("Entry order placement failed")
            return None

        # ── Step 4: confirm fill (poll with timeout) ────────────────────
        filled = self._wait_for_fill(entry_resp.order_id)
        if not filled:
            logger.warning("Entry order %s not filled in %ds — cancelling",
                           entry_resp.order_id, ORDER_FILL_TIMEOUT_SEC)
            self.api.cancel_order(entry_resp.order_id)
            return None

        fill_price = filled.get("filledPrice", entry_price)

        # ── Step 5: place exit orders (client-managed bracket) ──────────
        exit_order_ids = self._place_bracket_exits(
            side=side, qty=qty, stop_price=stop_price, target_price=target_price,
        )

        # ── Step 6: update session state ────────────────────────────────
        self.state.daily_trade_count += 1
        self.state.open_position = {
            "trade_id": trade_id,
            "side": side.value,
            "qty": qty,
            "entry_price": fill_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "strategy": strategy,
        }
        self.state.pending_exit_orders = exit_order_ids

        # ── Step 7: build trade record ──────────────────────────────────
        record = TradeRecord(
            trade_id=trade_id,
            timestamp_entry=now.isoformat(),
            regime=regime.name,
            strategy=strategy,
            side=side.value,
            qty=qty,
            entry_price=entry_price,
            fill_price=fill_price,
            stop_price=stop_price,
            target_price=target_price,
            order_id_entry=entry_resp.order_id,
            order_id_stop=exit_order_ids[0] if len(exit_order_ids) > 0 else "",
            order_id_target=exit_order_ids[1] if len(exit_order_ids) > 1 else "",
            metadata=metadata or {},
        )
        self._trade_log.append(record)
        logger.info("Entry FILLED: %s", trade_id)

        return record

    def record_exit(
        self,
        exit_price: float,
        exit_reason: str,
    ) -> TradeRecord | None:
        """Called when an exit fill is confirmed (stop, target, EOD, etc.).

        Updates P&L, clears open position, and finalises the trade record.
        """
        pos = self.state.open_position
        if pos is None:
            logger.warning("record_exit called with no open position")
            return None

        # Calculate P&L
        side = pos["side"]
        qty = pos["qty"]
        entry = pos["entry_price"]

        if side == OrderSide.BUY.value:
            pnl_points = exit_price - entry
        else:
            pnl_points = entry - exit_price

        pnl_dollars = pnl_points * POINT_VALUE * qty

        # Update state
        self.state.daily_pnl += pnl_dollars
        self.state.cumulative_pnl += pnl_dollars
        self.state.account_balance += pnl_dollars

        if self.state.account_balance > self.state.account_high_water_mark:
            self.state.account_high_water_mark = self.state.account_balance

        # Cancel any remaining exit orders
        for oid in self.state.pending_exit_orders:
            if oid:
                self.api.cancel_order(oid)
        self.state.pending_exit_orders = []
        self.state.open_position = None

        # Update trade record
        now = datetime.now(ET)
        record = self._find_trade(pos["trade_id"])
        if record:
            record.exit_price = exit_price
            record.pnl = pnl_dollars
            record.exit_reason = exit_reason
            record.timestamp_exit = now.isoformat()

        logger.info(
            "Exit %s: P&L $%.2f (%s) — daily $%.2f cumulative $%.2f",
            exit_reason, pnl_dollars, pos["trade_id"],
            self.state.daily_pnl, self.state.cumulative_pnl,
        )
        return record

    def flatten_all(self, reason: str = "EOD_CLOSE") -> None:
        """Emergency / EOD flatten. Close all positions via market orders."""
        self.flatten_all_with_price(reason=reason, exit_price=None)

    def flatten_all_with_price(self, reason: str = "EOD_CLOSE", exit_price: float | None = None) -> None:
        """Emergency / EOD flatten with an optional best-known exit price."""
        logger.info("FLATTEN ALL — reason: %s", reason)

        # Cancel pending exit orders first
        for oid in self.state.pending_exit_orders:
            if oid:
                self.api.cancel_order(oid)
        self.state.pending_exit_orders = []

        # Close via broker
        self.api.close_all_positions()

        # If we had a tracked position, record the exit at unknown price
        if self.state.open_position is not None:
            position = self.state.open_position
            if exit_price is None:
                exit_price = float(position.get("entry_price", 0.0))
            self.record_exit(exit_price=float(exit_price), exit_reason=reason)

    # ── bracket exit management (client_fallback mode) ──────────────────

    def _place_bracket_exits(
        self,
        side: OrderSide,
        qty: int,
        stop_price: float,
        target_price: float,
    ) -> list[str]:
        """Place stop-loss and profit-target as two separate working orders.

        In ``client_fallback`` mode, these are managed independently:
        when one fills, the OrderManager cancels the other (OCO emulation).
        In ``projectx_native`` mode, a single OCO order would be placed instead.
        """
        exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        order_ids: list[str] = []

        # Stop order
        try:
            stop_resp = self.api.place_order(
                side=exit_side.value, qty=qty, order_type="STOP",
                stop_price=stop_price,
            )
            order_ids.append(stop_resp.order_id)
        except Exception:
            logger.exception("Failed to place stop order")
            order_ids.append("")

        # Target (limit) order
        try:
            target_resp = self.api.place_order(
                side=exit_side.value, qty=qty, order_type="LIMIT",
                limit_price=target_price,
            )
            order_ids.append(target_resp.order_id)
        except Exception:
            logger.exception("Failed to place target order")
            order_ids.append("")

        return order_ids

    def handle_exit_fill(self, filled_order_id: str, fill_price: float) -> None:
        """Called by WebSocket listener when an exit order fills.

        Implements client-side OCO: cancels the other exit order.
        """
        pending = self.state.pending_exit_orders
        if filled_order_id not in pending:
            return

        # Cancel the other leg
        for oid in pending:
            if oid and oid != filled_order_id:
                self.api.cancel_order(oid)

        # Determine exit reason
        pos = self.state.open_position
        if pos is None:
            return

        if abs(fill_price - pos["stop_price"]) < abs(fill_price - pos["target_price"]):
            reason = "STOP"
        else:
            reason = "TARGET"

        self.record_exit(exit_price=fill_price, exit_reason=reason)

    def handle_order_update(self, msg: dict[str, Any]) -> None:
        """Handle broker order/fill updates delivered over WebSocket."""
        order_id = self._extract_order_id(msg)
        if not order_id:
            return

        fill_price = self._extract_fill_price(msg)
        status = str(
            msg.get("status")
            or msg.get("orderStatus")
            or msg.get("state")
            or msg.get("event")
            or ""
        ).upper()
        event_type = str(msg.get("type", "")).lower()

        if fill_price is not None or status in {"FILLED", "COMPLETE", "EXECUTED"} or event_type in {"fill", "execution"}:
            self._last_fill_event[order_id] = {
                "orderId": order_id,
                "status": status or "FILLED",
                "filledPrice": fill_price,
                "raw": msg,
            }

        if order_id in self.state.pending_exit_orders and fill_price is None:
            fill_price = self._infer_exit_fill_price(order_id)
            if fill_price is not None and order_id in self._last_fill_event:
                self._last_fill_event[order_id]["filledPrice"] = fill_price

        if order_id in self.state.pending_exit_orders and fill_price is not None:
            self.handle_exit_fill(order_id, fill_price)

    # ── fill polling ────────────────────────────────────────────────────

    def _wait_for_fill(self, order_id: str) -> dict | None:
        """Poll for fill confirmation up to ORDER_FILL_TIMEOUT_SEC.

        Returns fill data dict on success, None on timeout.
        In a real integration the WebSocket event would resolve this faster.
        """
        deadline = time.monotonic() + ORDER_FILL_TIMEOUT_SEC
        while time.monotonic() < deadline:
            try:
                cached_fill = self._last_fill_event.pop(order_id, None)
                if cached_fill is not None:
                    return cached_fill

                orders = self.api.get_open_orders()
                for o in orders:
                    if str(o.get("orderId")) == order_id:
                        if o.get("status", "").upper() in ("FILLED", "COMPLETE"):
                            return o
            except Exception:
                logger.exception("Error polling fill for %s", order_id)
            time.sleep(0.5)
        return None

    def reconcile_with_broker(self) -> dict[str, Any]:
        """Bring local position/order state back in sync with broker reality."""
        positions = self.api.get_positions()
        open_orders = self.api.get_open_orders()

        active_positions = [p for p in positions if abs(self._position_qty(p)) > 0]
        open_order_ids = {
            oid for oid in (self._extract_order_id(o) for o in open_orders) if oid
        }
        result = {
            "broker_positions": len(active_positions),
            "open_orders": len(open_order_ids),
            "cleared_local_position": False,
            "flattened_broker_position": False,
            "cancelled_orphan_orders": [],
        }

        if self.state.open_position is not None and not active_positions:
            self.state.open_position = None
            self.state.pending_exit_orders = []
            result["cleared_local_position"] = True

        if self.state.open_position is None and active_positions:
            self.api.close_all_positions()
            result["flattened_broker_position"] = True

        pending_ids = [oid for oid in self.state.pending_exit_orders if oid]
        for oid in pending_ids:
            if oid not in open_order_ids:
                continue
            if self.state.open_position is None:
                self.api.cancel_order(oid)
                result["cancelled_orphan_orders"].append(oid)

        if self.state.open_position is None:
            self.state.pending_exit_orders = []

        return result

    # ── persistence & logging ───────────────────────────────────────────

    def save_state(self, filepath: str | None = None) -> None:
        """Persist session state to JSON."""
        from config import STATE_FILE

        path = Path(filepath or STATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state.to_dict(), indent=2))

    def load_state(self, filepath: str | None = None) -> None:
        """Reload session state from JSON (for process restart recovery)."""
        from config import STATE_FILE

        path = Path(filepath or STATE_FILE)
        if path.exists():
            data = json.loads(path.read_text())
            self.state = SessionState.from_dict(data)
            logger.info("Session state loaded from %s", path)
        else:
            logger.info("No persisted state found — starting fresh")

    def write_trade_log(self) -> None:
        """Append all trade records from this session to the daily log CSV."""
        if not self._trade_log:
            return

        log_dir = Path(LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now(ET).strftime("%Y-%m-%d")
        log_file = log_dir / f"trades_{today}.json"

        existing: list[dict] = []
        if log_file.exists():
            try:
                existing = json.loads(log_file.read_text())
            except json.JSONDecodeError:
                existing = []

        for rec in self._trade_log:
            existing.append(asdict(rec))

        log_file.write_text(json.dumps(existing, indent=2))
        logger.info("Wrote %d trade records to %s", len(self._trade_log), log_file)

    def daily_summary(self) -> dict:
        """Return a summary dict for end-of-day logging."""
        wins = sum(1 for t in self._trade_log if t.pnl > 0)
        losses = sum(1 for t in self._trade_log if t.pnl < 0)
        return {
            "date": datetime.now(ET).strftime("%Y-%m-%d"),
            "regime": RegimeState(self.state.current_regime).name,
            "total_trades": self.state.daily_trade_count,
            "wins": wins,
            "losses": losses,
            "daily_pnl": round(self.state.daily_pnl, 2),
            "cumulative_pnl": round(self.state.cumulative_pnl, 2),
            "account_balance": round(self.state.account_balance, 2),
            "high_water_mark": round(self.state.account_high_water_mark, 2),
            "breaker_events": len(self.breakers.events),
        }

    # ── helpers ─────────────────────────────────────────────────────────

    def _find_trade(self, trade_id: str) -> TradeRecord | None:
        for t in self._trade_log:
            if t.trade_id == trade_id:
                return t
        return None

    @staticmethod
    def _extract_order_id(msg: dict[str, Any]) -> str:
        return str(
            msg.get("orderId")
            or msg.get("order_id")
            or msg.get("id")
            or msg.get("clientOrderId")
            or ""
        )

    @staticmethod
    def _extract_fill_price(msg: dict[str, Any]) -> float | None:
        raw = msg.get("filledPrice")
        if raw is None:
            raw = msg.get("fillPrice")
        if raw is None:
            raw = msg.get("price")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _position_qty(position: Any) -> int:
        raw = getattr(position, "qty", None)
        if raw is None and isinstance(position, dict):
            raw = position.get("qty", position.get("quantity", 0))
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    def _infer_exit_fill_price(self, order_id: str) -> float | None:
        pos = self.state.open_position or {}
        pending = self.state.pending_exit_orders
        if not pending:
            return None
        if pending and pending[0] == order_id:
            try:
                return float(pos.get("stop_price"))
            except (TypeError, ValueError):
                return None
        if len(pending) > 1 and pending[1] == order_id:
            try:
                return float(pos.get("target_price"))
            except (TypeError, ValueError):
                return None
        return None
