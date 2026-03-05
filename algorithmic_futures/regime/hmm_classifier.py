"""
regime/hmm_classifier.py — Hidden Markov Model regime classifier.

Classifies the market into 3 states using a Gaussian HMM trained
on daily features with expanding walk-forward validation.

States are always sorted by realized volatility (ATR percentile mean)
after every refit to ensure economic consistency across label swaps.

Usage:
    from regime.hmm_classifier import RegimeClassifier
    clf = RegimeClassifier()
    clf.fit(feature_df)  # DataFrame with log_return, atr_percentile, vix_term_slope
    regime = clf.predict_next_regime(feature_df)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from config import (
    HMM_COVARIANCE_TYPE,
    HMM_MIN_LOOKBACK_DAYS,
    HMM_N_ITER,
    HMM_N_STATES,
    HMM_STATE_FILE,
)
from regime.regime_state import RegimeState

logger = logging.getLogger(__name__)

FEATURE_COLS = ["log_return", "atr_percentile", "vix_term_slope"]
ATR_COL_IDX = 1  # index of atr_percentile in FEATURE_COLS


class RegimeClassifier:
    """3-state Gaussian HMM with walk-forward expanding-window training."""

    def __init__(self) -> None:
        self.model: GaussianHMM | None = None
        self.state_map: dict[int, int] = {}  # raw HMM state → RegimeState value
        self._last_fit_date: str | None = None

    # ── training ────────────────────────────────────────────────────────

    def fit(self, feature_df: pd.DataFrame) -> None:
        """Fit HMM on an expanding window of daily features.

        Parameters
        ----------
        feature_df : DataFrame
            Must contain columns: log_return, atr_percentile, vix_term_slope.
            Should include *only* data available up to and including
            the current date (no future data).  The expanding window starts
            at HMM_MIN_LOOKBACK_DAYS rows; all rows are used (no rolling cap).
        """
        self._validate_features(feature_df)

        X = feature_df[FEATURE_COLS].values.astype(np.float64)

        if len(X) < HMM_MIN_LOOKBACK_DAYS:
            raise ValueError(
                f"Need at least {HMM_MIN_LOOKBACK_DAYS} rows for HMM training, "
                f"got {len(X)}"
            )

        # Expanding window — use ALL available data up to today
        self.model = GaussianHMM(
            n_components=HMM_N_STATES,
            covariance_type=HMM_COVARIANCE_TYPE,
            n_iter=HMM_N_ITER,
            random_state=42,
        )
        self.model.fit(X)
        self._map_states_to_regimes()

        self._last_fit_date = str(feature_df.index[-1]) if hasattr(feature_df.index, '__len__') else None

        logger.info(
            "HMM fitted on %d bars ending %s | transition matrix:\n%s",
            len(X),
            self._last_fit_date,
            np.array2string(self.model.transmat_, precision=3),
        )

    # ── prediction ──────────────────────────────────────────────────────

    def predict_next_regime(self, feature_df: pd.DataFrame) -> RegimeState:
        """Return predicted regime for the *next* session.

        Uses the last hidden state from the Viterbi path and the
        transition matrix to determine the most-likely next state.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet — call fit() first.")

        self._validate_features(feature_df)
        X = feature_df[FEATURE_COLS].values.astype(np.float64)

        hidden_states = self.model.predict(X)
        current_raw = int(hidden_states[-1])

        # Most probable next state via transition matrix
        next_raw = int(np.argmax(self.model.transmat_[current_raw]))
        regime_int = self.state_map.get(next_raw, 0)

        regime = RegimeState(regime_int)
        logger.info(
            "Current raw state %d → predicted next raw state %d → regime %s",
            current_raw,
            next_raw,
            regime.name,
        )
        return regime

    def predict_current_regime(self, feature_df: pd.DataFrame) -> RegimeState:
        """Return the regime inferred for the *current* (most recent) bar."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet — call fit() first.")

        self._validate_features(feature_df)
        X = feature_df[FEATURE_COLS].values.astype(np.float64)

        hidden_states = self.model.predict(X)
        current_raw = int(hidden_states[-1])
        regime_int = self.state_map.get(current_raw, 0)
        return RegimeState(regime_int)

    # ── state-label stabilisation ───────────────────────────────────────

    def _map_states_to_regimes(self) -> None:
        """Sort HMM states by mean ATR percentile so labels are economically stable.

        Lowest vol → BALANCED (0), mid → DIRECTIONAL (1), highest → CRISIS (2).
        """
        means = self.model.means_  # shape (n_states, n_features)
        vol_order = np.argsort([m[ATR_COL_IDX] for m in means])  # ascending vol
        self.state_map = {
            int(vol_order[0]): RegimeState.BALANCED.value,
            int(vol_order[1]): RegimeState.DIRECTIONAL.value,
            int(vol_order[2]): RegimeState.CRISIS.value,
        }
        logger.info("State map (raw → regime): %s", self.state_map)

    # ── persistence ─────────────────────────────────────────────────────

    def save_regime(self, regime: RegimeState, target_date: str | None = None) -> None:
        """Persist the predicted regime to disk for the session manager to load."""
        path = Path(HMM_STATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "predicted_regime": regime.value,
            "predicted_regime_name": regime.name,
            "target_date": target_date or datetime.now().strftime("%Y-%m-%d"),
            "fit_date": self._last_fit_date,
            "state_map": {str(k): v for k, v in self.state_map.items()},
            "saved_at": datetime.now().isoformat(),
        }

        # Include transition matrix if available
        if self.model is not None:
            payload["transition_matrix"] = self.model.transmat_.tolist()
            payload["state_means"] = self.model.means_.tolist()

        path.write_text(json.dumps(payload, indent=2))
        logger.info("Regime %s saved to %s for date %s", regime.name, path, payload["target_date"])

    @staticmethod
    def load_regime() -> RegimeState:
        """Load the last saved regime from disk."""
        path = Path(HMM_STATE_FILE)
        if not path.exists():
            logger.warning("No HMM state file found at %s — defaulting to CRISIS (no-trade)", path)
            return RegimeState.CRISIS  # safe default

        data = json.loads(path.read_text())
        regime = RegimeState(data["predicted_regime"])
        logger.info("Loaded regime %s (target date %s)", regime.name, data.get("target_date"))
        return regime

    # ── diagnostics ─────────────────────────────────────────────────────

    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostic information for logging / review."""
        if self.model is None:
            return {"status": "not_fitted"}
        return {
            "n_states": HMM_N_STATES,
            "state_map": self.state_map,
            "state_means": self.model.means_.tolist(),
            "transition_matrix": self.model.transmat_.tolist(),
            "last_fit_date": self._last_fit_date,
        }

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _validate_features(df: pd.DataFrame) -> None:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Feature DataFrame missing columns: {missing}")
        if df.empty:
            raise ValueError("Feature DataFrame is empty")
