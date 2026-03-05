"""
regime/regime_state.py — Regime state enums and execution mode types.
"""

from __future__ import annotations

from enum import IntEnum, Enum


class RegimeState(IntEnum):
    """Market regime as classified by the HMM."""
    BALANCED = 0       # Low-vol, ranging → VWAP Mean Reversion
    DIRECTIONAL = 1    # High-vol, trending → ORB Breakout
    CRISIS = 2         # Extreme stress → No trading (cash preservation)


class ChallengeStatus(str, Enum):
    """Overall challenge lifecycle state."""
    IN_PROGRESS = "IN_PROGRESS"
    PASSED = "PASSED"
    FAILED = "FAILED"


class ExecutionMode(str, Enum):
    """Execution abstraction layer mode."""
    PROJECTX_NATIVE = "projectx_native"
    CLIENT_FALLBACK = "client_fallback"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class BreakerType(str, Enum):
    """Circuit breaker identifiers."""
    DAILY_LOSS = "DAILY_LOSS"
    EXTERNAL_LOSS = "EXTERNAL_LOSS"
    DAILY_PROFIT = "DAILY_PROFIT"
    CONSISTENCY_CAP = "CONSISTENCY_CAP"
    MLL_PROXIMITY = "MLL_PROXIMITY"
    TRADE_COUNT = "TRADE_COUNT"
    EOD_TIME_STOP = "EOD_TIME_STOP"
    CRISIS_REGIME = "CRISIS_REGIME"
