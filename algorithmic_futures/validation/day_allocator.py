"""Day-level strategy allocator for MR baseline + conditional ORB enablement."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DayAllocatorConfig:
    orb_day_pnl_floor: float = -120.0
    orb_max_loss_cluster: int = 2


@dataclass
class DayAllocatorDecision:
    session_id: str
    category: str
    day_pnl_before: float
    max_loss_cluster_before: int
    volatility_ok: bool
    orb_enabled: bool
    reason: str


class DayAllocator:
    """Simple allocator:

    - MR always enabled (implicit in runner)
    - ORB enabled only if:
      1) day PnL > floor
      2) max loss cluster <= threshold
      3) category suggests breakout-supportive regime (trend/event)
    """

    def __init__(self, config: DayAllocatorConfig | None = None) -> None:
        self.config = config or DayAllocatorConfig()
        self.day_pnl: float = 0.0
        self.max_loss_cluster: int = 0

    def decide(self, session_id: str, category: str, orb_requested: bool) -> DayAllocatorDecision:
        volatility_ok = category in {"trend", "event"}
        if not orb_requested:
            return DayAllocatorDecision(
                session_id=session_id,
                category=category,
                day_pnl_before=self.day_pnl,
                max_loss_cluster_before=self.max_loss_cluster,
                volatility_ok=volatility_ok,
                orb_enabled=False,
                reason="ORB_DISABLED_BY_CONFIG",
            )

        if self.day_pnl <= self.config.orb_day_pnl_floor:
            return DayAllocatorDecision(
                session_id=session_id,
                category=category,
                day_pnl_before=self.day_pnl,
                max_loss_cluster_before=self.max_loss_cluster,
                volatility_ok=volatility_ok,
                orb_enabled=False,
                reason="DAY_PNL_FLOOR",
            )

        if self.max_loss_cluster > self.config.orb_max_loss_cluster:
            return DayAllocatorDecision(
                session_id=session_id,
                category=category,
                day_pnl_before=self.day_pnl,
                max_loss_cluster_before=self.max_loss_cluster,
                volatility_ok=volatility_ok,
                orb_enabled=False,
                reason="LOSS_CLUSTER_GUARD",
            )

        if not volatility_ok:
            return DayAllocatorDecision(
                session_id=session_id,
                category=category,
                day_pnl_before=self.day_pnl,
                max_loss_cluster_before=self.max_loss_cluster,
                volatility_ok=volatility_ok,
                orb_enabled=False,
                reason="VOLATILITY_REGIME_GUARD",
            )

        return DayAllocatorDecision(
            session_id=session_id,
            category=category,
            day_pnl_before=self.day_pnl,
            max_loss_cluster_before=self.max_loss_cluster,
            volatility_ok=volatility_ok,
            orb_enabled=True,
            reason="ENABLED",
        )

    def update_from_trades_csv(self, trades_csv: Path) -> None:
        if not trades_csv.is_file():
            return

        pnl_values: list[float] = []
        with trades_csv.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    pnl_values.append(float(row.get("pnl_dollars", 0.0) or 0.0))
                except (TypeError, ValueError):
                    continue

        if not pnl_values:
            return

        self.day_pnl += sum(pnl_values)
        curr = 0
        best = self.max_loss_cluster
        for pnl in pnl_values:
            if pnl <= 0:
                curr += 1
                best = max(best, curr)
            else:
                curr = 0
        self.max_loss_cluster = best
