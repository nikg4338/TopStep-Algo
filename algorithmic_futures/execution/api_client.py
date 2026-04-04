"""
execution/api_client.py — ProjectX API client wrapper.

Single point of contact for all TopstepX communication:
  - OAuth token management with auto-refresh
  - REST order placement / cancellation / account queries
  - WebSocket market data subscription with reconnect logic
  - Exponential backoff for transient errors (429, 503)

All raw requests flow through this module — never scatter direct
HTTP calls across the codebase.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import requests

from config import (
    API_BACKOFF_BASE_SEC,
    API_MAX_RETRIES,
    INSTRUMENT,
    WS_RECONNECT_INTERVAL_SEC,
    WS_RECONNECT_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)


# ── Data transfer objects ───────────────────────────────────────────────


@dataclass(frozen=True)
class OrderResponse:
    order_id: str
    status: str
    filled_qty: int = 0
    filled_price: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AccountBalance:
    balance: float
    unrealized_pnl: float
    realized_pnl: float
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Position:
    symbol: str
    side: str
    qty: int
    avg_price: float
    unrealized_pnl: float
    raw: dict = field(default_factory=dict)


# ── Client ──────────────────────────────────────────────────────────────


class ProjectXClient:
    """REST + WebSocket client for the ProjectX API.

    Instantiate once and share across order_manager / session_manager.
    Credentials are loaded from environment variables.
    """

    RETRYABLE_STATUS_CODES = {429, 503}

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        account_id: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("PROJECTX_API_KEY", "")
        self.base_url = (base_url or os.getenv("PROJECTX_BASE_URL", "")).rstrip("/")
        self.account_id = account_id or os.getenv("ACCOUNT_ID", "")

        self._token: str | None = None
        self._token_expiry: datetime | None = None
        self._session = requests.Session()

        # WebSocket state
        self._ws = None
        self._ws_connected = False
        self._ws_callbacks: dict[str, Callable] = {}

    # ── Authentication ──────────────────────────────────────────────────

    def authenticate(self) -> None:
        """Obtain or refresh the OAuth bearer token."""
        logger.info("Authenticating with ProjectX API…")
        payload = {"apiKey": self.api_key}
        resp = self._raw_post("/auth/token", json_body=payload, auth=False)
        self._token = resp.get("token") or resp.get("access_token", "")
        expires_in = resp.get("expires_in", 3600)
        self._token_expiry = datetime.now() + timedelta(seconds=int(expires_in) - 300)
        logger.info("Authenticated — token valid until %s", self._token_expiry)

    def _ensure_auth(self) -> None:
        if self._token is None or (
            self._token_expiry and datetime.now() >= self._token_expiry
        ):
            self.authenticate()

    def _auth_headers(self) -> dict[str, str]:
        self._ensure_auth()
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    # ── Account ─────────────────────────────────────────────────────────

    def get_account_balance(self) -> AccountBalance:
        data = self._get(f"/accounts/{self.account_id}/balance")
        return AccountBalance(
            balance=float(data.get("balance", 0)),
            unrealized_pnl=float(data.get("unrealizedPnl", 0)),
            realized_pnl=float(data.get("realizedPnl", 0)),
            raw=data,
        )

    # ── Orders ──────────────────────────────────────────────────────────

    def place_order(
        self,
        side: str,
        qty: int,
        order_type: str,
        *,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> OrderResponse:
        """Place a single order on MES. Returns an OrderResponse."""
        payload: dict[str, Any] = {
            "accountId": self.account_id,
            "symbol": INSTRUMENT,
            "side": side.upper(),
            "qty": qty,
            "orderType": order_type.upper(),
        }
        if limit_price is not None:
            payload["limitPrice"] = limit_price
        if stop_price is not None:
            payload["stopPrice"] = stop_price

        # Mask payload for logging (no secrets)
        safe_log = {k: v for k, v in payload.items()}
        logger.info("Placing order: %s", safe_log)

        data = self._post("/orders", json_body=payload)
        resp = OrderResponse(
            order_id=str(data.get("orderId", "")),
            status=str(data.get("status", "UNKNOWN")),
            filled_qty=int(data.get("filledQty", 0)),
            filled_price=float(data.get("filledPrice", 0)),
            raw=data,
        )
        logger.info("Order response: id=%s status=%s", resp.order_id, resp.status)
        return resp

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a working order. Returns True on success."""
        logger.info("Cancelling order %s", order_id)
        try:
            self._delete(f"/orders/{order_id}")
            return True
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    def get_open_orders(self) -> list[dict]:
        return self._get(f"/accounts/{self.account_id}/orders") or []

    # ── Positions ───────────────────────────────────────────────────────

    def get_positions(self) -> list[Position]:
        data = self._get(f"/accounts/{self.account_id}/positions") or []
        return [
            Position(
                symbol=p.get("symbol", INSTRUMENT),
                side=p.get("side", ""),
                qty=int(p.get("qty", 0)),
                avg_price=float(p.get("avgPrice", 0)),
                unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                raw=p,
            )
            for p in data
        ]

    def close_all_positions(self) -> bool:
        """Flatten all open MES positions via market orders."""
        positions = self.get_positions()
        if not positions:
            logger.info("No open positions to close")
            return True

        success = True
        for pos in positions:
            close_side = "SELL" if pos.side.upper() == "BUY" else "BUY"
            try:
                self.place_order(close_side, abs(pos.qty), "MARKET")
            except Exception:
                logger.exception("Failed to close position %s", pos)
                success = False
        return success

    # ── WebSocket (market data) ─────────────────────────────────────────

    async def subscribe_market_data(
        self,
        symbol: str,
        on_tick: Callable | None = None,
        on_bar: Callable | None = None,
        on_order_update: Callable | None = None,
    ) -> None:
        """Connect to WebSocket and stream ticks.  Reconnects automatically."""
        import websockets  # type: ignore[import-untyped]

        ws_url = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/ws/market-data"

        self._ensure_auth()

        last_connected = time.monotonic()
        while True:
            try:
                async with websockets.connect(
                    ws_url,
                    additional_headers={"Authorization": f"Bearer {self._token}"},
                ) as ws:
                    self._ws_connected = True
                    last_connected = time.monotonic()
                    logger.info("WebSocket connected for %s", symbol)

                    # Subscribe message
                    await ws.send(json.dumps({"action": "subscribe", "symbol": symbol}))

                    async for raw_msg in ws:
                        msg = json.loads(raw_msg)
                        msg_type = str(msg.get("type", "")).lower()
                        if on_tick and msg_type == "tick":
                            on_tick(msg)
                        if on_bar and msg_type == "bar":
                            on_bar(msg)
                        if on_order_update and msg_type in {"order", "fill", "execution", "order_update"}:
                            on_order_update(msg)

            except Exception as exc:
                self._ws_connected = False
                elapsed = time.monotonic() - last_connected
                logger.warning("WebSocket disconnected (%s) after %.1fs", exc, elapsed)

                if elapsed > WS_RECONNECT_TIMEOUT_SEC:
                    logger.error(
                        "WebSocket down > %ds — triggering emergency flatten",
                        WS_RECONNECT_TIMEOUT_SEC,
                    )
                    self.close_all_positions()
                    raise

                logger.info("Reconnecting in %ds…", WS_RECONNECT_INTERVAL_SEC)
                await asyncio.sleep(WS_RECONNECT_INTERVAL_SEC)

    @property
    def is_ws_connected(self) -> bool:
        return self._ws_connected

    # ── HTTP helpers with retry ─────────────────────────────────────────

    def _get(self, path: str) -> Any:
        return self._request("GET", path)

    def _post(self, path: str, json_body: dict | None = None) -> Any:
        return self._request("POST", path, json_body=json_body)

    def _delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    def _raw_post(self, path: str, json_body: dict | None = None, auth: bool = True) -> Any:
        headers = self._auth_headers() if auth else {"Content-Type": "application/json"}
        resp = self._session.post(f"{self.base_url}{path}", json=json_body, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _request(self, method: str, path: str, json_body: dict | None = None) -> Any:
        """Execute an HTTP request with exponential backoff for transient errors."""
        url = f"{self.base_url}{path}"
        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                resp = self._session.request(
                    method,
                    url,
                    json=json_body,
                    headers=self._auth_headers(),
                    timeout=10,
                )
                if resp.status_code in self.RETRYABLE_STATUS_CODES:
                    delay = API_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                    logger.warning(
                        "HTTP %d on %s %s — retrying in %.1fs (attempt %d/%d)",
                        resp.status_code, method, path, delay, attempt, API_MAX_RETRIES,
                    )
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json() if resp.content else {}
            except requests.exceptions.ConnectionError:
                if attempt < API_MAX_RETRIES:
                    delay = API_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                    logger.warning("Connection error — retrying in %.1fs", delay)
                    time.sleep(delay)
                else:
                    raise
        # Exhausted retries — raise the last response error
        resp.raise_for_status()  # type: ignore[possibly-undefined]
        return {}
