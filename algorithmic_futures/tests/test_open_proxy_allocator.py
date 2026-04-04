"""Tests for the allocator open-window proxy."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.open_proxy_allocator import OpenProxyConfig, OpenWindowState, decide


def _state(*, bars, or_high, or_low, first_bar_open, atr, post_or_bars=None):
    return OpenWindowState(
        bars=bars,
        or_high=or_high,
        or_low=or_low,
        first_bar_open=first_bar_open,
        atr_at_decision=atr,
        post_or_bars=post_or_bars or [],
    )


def test_open_proxy_routes_orb_on_width_signal():
    state = _state(
        bars=[(100, 102, 99, 101), (101, 104, 100, 103), (103, 106, 102, 105)],
        or_high=106,
        or_low=99,
        first_bar_open=100,
        atr=2.5,
    )
    result = decide(state, OpenProxyConfig(or_width_atr_threshold=2.2, impulse_atr_threshold=10.0))
    assert result.decision == "orb"
    assert result.trigger_width is True
    assert result.trigger_impulse is False


def test_open_proxy_routes_mr_when_no_signal_fires():
    state = _state(
        bars=[(100, 100.5, 99.8, 100.1), (100.1, 100.4, 99.9, 100.0), (100.0, 100.3, 99.8, 100.1)],
        or_high=100.5,
        or_low=99.8,
        first_bar_open=100,
        atr=3.0,
    )
    result = decide(state, OpenProxyConfig(or_width_atr_threshold=2.2, impulse_atr_threshold=0.9))
    assert result.decision == "mr"
    assert result.trigger_width is False
    assert result.trigger_impulse is False
    assert result.trigger_persist is False


def test_open_proxy_can_require_breakout_persistence():
    state = _state(
        bars=[(100, 102, 99, 101), (101, 103, 100, 102), (102, 104, 101, 103)],
        or_high=104,
        or_low=99,
        first_bar_open=100,
        atr=2.0,
        post_or_bars=[(103, 105, 102, 104.5)],
    )
    result = decide(state, OpenProxyConfig(or_width_atr_threshold=2.0, impulse_atr_threshold=0.5, persist_bars=1, require_break=True))
    assert result.decision == "orb"
    assert result.trigger_persist is True
    assert result.breakout_direction == "UP"


def test_open_proxy_has_no_adx_dependency_at_decision_time():
    state = _state(
        bars=[(100, 102, 99, 101), (101, 103, 100, 102), (102, 103, 101, 102.5)],
        or_high=103,
        or_low=99,
        first_bar_open=100,
        atr=2.0,
    )
    result = decide(state, OpenProxyConfig())
    assert hasattr(result, "opening_range_width_atr")
    assert not hasattr(result, "adx")
    assert result.atr_at_decision == 2.0


def test_open_proxy_v1_behavior_unchanged_when_selectivity_disabled():
    state = _state(
        bars=[(100, 102, 99, 101), (101, 104, 100, 103), (103, 106, 102, 105)],
        or_high=106,
        or_low=99,
        first_bar_open=100,
        atr=2.5,
    )
    baseline = decide(state, OpenProxyConfig(or_width_atr_threshold=2.2, impulse_atr_threshold=10.0))
    refined_off = decide(
        state,
        OpenProxyConfig(
            or_width_atr_threshold=2.2,
            impulse_atr_threshold=10.0,
            enable_orb_selectivity_refinement=False,
            orb_selectivity_low_atr_threshold=10.0,
            orb_selectivity_min_persistence_in_low_atr=2,
            orb_selectivity_high_impulse_threshold=2.4,
            orb_selectivity_min_persistence_when_high_impulse=1,
            enable_medium_impulse_weak_persistence_filter=False,
        ),
    )
    assert refined_off.decision == baseline.decision
    assert refined_off.reason == baseline.reason


def test_open_proxy_v1_2_behavior_unchanged_until_v1_3_filter_is_enabled():
    state = _state(
        bars=[(100, 103, 99, 102), (102, 103, 101, 102), (102, 104, 101.5, 103)],
        or_high=104,
        or_low=99,
        first_bar_open=100,
        atr=5.0,
        post_or_bars=[],
    )
    v1_2 = decide(
        state,
        OpenProxyConfig(
            or_width_atr_threshold=10.0,
            impulse_atr_threshold=0.2,
            enable_orb_selectivity_refinement=True,
            orb_selectivity_low_atr_threshold=3.0,
            orb_selectivity_min_persistence_in_low_atr=0,
            orb_selectivity_high_impulse_threshold=1.0,
            orb_selectivity_min_persistence_when_high_impulse=0,
            enable_medium_impulse_weak_persistence_filter=False,
        ),
    )
    v1_2_with_v1_3_off = decide(
        state,
        OpenProxyConfig(
            or_width_atr_threshold=10.0,
            impulse_atr_threshold=0.2,
            enable_orb_selectivity_refinement=True,
            orb_selectivity_low_atr_threshold=3.0,
            orb_selectivity_min_persistence_in_low_atr=0,
            orb_selectivity_high_impulse_threshold=1.0,
            orb_selectivity_min_persistence_when_high_impulse=0,
            enable_medium_impulse_weak_persistence_filter=False,
        ),
    )
    assert v1_2.decision == "orb"
    assert v1_2_with_v1_3_off.decision == v1_2.decision
    assert v1_2_with_v1_3_off.reason == v1_2.reason


def test_open_proxy_selectivity_blocks_low_atr_without_persistence():
    state = _state(
        bars=[(100, 101, 99, 100.5), (100.5, 101.5, 100, 101.0), (101.0, 102.0, 100.75, 101.5)],
        or_high=102.0,
        or_low=99.0,
        first_bar_open=100.0,
        atr=9.5,
        post_or_bars=[],
    )
    result = decide(
        state,
        OpenProxyConfig(
            or_width_atr_threshold=0.2,
            impulse_atr_threshold=0.1,
            enable_orb_selectivity_refinement=True,
            orb_selectivity_low_atr_threshold=10.0,
            orb_selectivity_min_persistence_in_low_atr=1,
            orb_selectivity_high_impulse_threshold=5.0,
            orb_selectivity_min_persistence_when_high_impulse=1,
        ),
    )
    assert result.pre_selectivity_decision == "orb"
    assert result.decision == "mr"
    assert result.selectivity_low_atr_caution is True
    assert result.selectivity_orb_blocked is True


def test_open_proxy_selectivity_blocks_high_impulse_weak_persistence():
    state = _state(
        bars=[(100, 104, 99, 104), (104, 105, 103, 104.5), (104.5, 105, 104, 105)],
        or_high=105,
        or_low=99,
        first_bar_open=100,
        atr=20.0,
        post_or_bars=[],
    )
    result = decide(
        state,
        OpenProxyConfig(
            or_width_atr_threshold=10.0,
            impulse_atr_threshold=0.2,
            enable_orb_selectivity_refinement=True,
            orb_selectivity_low_atr_threshold=5.0,
            orb_selectivity_min_persistence_in_low_atr=2,
            orb_selectivity_high_impulse_threshold=0.2,
            orb_selectivity_min_persistence_when_high_impulse=1,
        ),
    )
    assert result.pre_selectivity_decision == "orb"
    assert result.decision == "mr"
    assert result.selectivity_high_impulse_caution is True


def test_open_proxy_selectivity_allows_strong_persistence():
    state = _state(
        bars=[(100, 102, 99, 101), (101, 103, 100, 102), (102, 104, 101, 103)],
        or_high=104,
        or_low=99,
        first_bar_open=100,
        atr=9.0,
        post_or_bars=[(103, 105, 102, 104.5), (104.5, 106, 104, 105.5)],
    )
    result = decide(
        state,
        OpenProxyConfig(
            or_width_atr_threshold=0.2,
            impulse_atr_threshold=0.2,
            persist_bars=1,
            enable_orb_selectivity_refinement=True,
            orb_selectivity_low_atr_threshold=10.0,
            orb_selectivity_min_persistence_in_low_atr=1,
            orb_selectivity_high_impulse_threshold=0.2,
            orb_selectivity_min_persistence_when_high_impulse=1,
        ),
    )
    assert result.decision == "orb"
    assert result.selectivity_orb_blocked is False


def test_open_proxy_v3_blocks_medium_impulse_weak_persistence():
    state = _state(
        bars=[(100, 103, 99, 102), (102, 103, 101, 102), (102, 104, 101.5, 103)],
        or_high=104,
        or_low=99,
        first_bar_open=100,
        atr=5.0,
        post_or_bars=[],
    )
    result = decide(
        state,
        OpenProxyConfig(
            or_width_atr_threshold=10.0,
            impulse_atr_threshold=0.2,
            enable_orb_selectivity_refinement=True,
            orb_selectivity_low_atr_threshold=3.0,
            orb_selectivity_min_persistence_in_low_atr=0,
            orb_selectivity_high_impulse_threshold=1.0,
            orb_selectivity_min_persistence_when_high_impulse=0,
            enable_medium_impulse_weak_persistence_filter=True,
        ),
    )
    assert result.pre_v3_selectivity_decision == "orb"
    assert result.decision == "mr"
    assert result.selectivity_medium_impulse_weak_persistence_caution is True
    assert result.selectivity_v3_orb_blocked is True
    assert result.selectivity_v3_block_reason == (
        "OPEN_PROXY_RANGE_SELECTIVITY_V3_MEDIUM_IMPULSE_WEAK_PERSISTENCE "
        "impulse_atr=0.60 band=[0.20,1.00) persistence=0"
    )
    assert result.reason == result.selectivity_v3_block_reason


def test_open_proxy_v3_does_not_block_high_impulse_with_persistence():
    state = _state(
        bars=[(100, 104, 99, 104), (104, 105, 103, 104.5), (104.5, 106, 104, 105.5)],
        or_high=106,
        or_low=99,
        first_bar_open=100,
        atr=2.0,
        post_or_bars=[(105.5, 107, 105, 106.5)],
    )
    result = decide(
        state,
        OpenProxyConfig(
            or_width_atr_threshold=10.0,
            impulse_atr_threshold=0.2,
            persist_bars=1,
            enable_orb_selectivity_refinement=True,
            orb_selectivity_low_atr_threshold=1.0,
            orb_selectivity_min_persistence_in_low_atr=0,
            orb_selectivity_high_impulse_threshold=1.5,
            orb_selectivity_min_persistence_when_high_impulse=1,
            enable_medium_impulse_weak_persistence_filter=True,
        ),
    )
    assert result.decision == "orb"


def test_open_proxy_v4_blocks_medium_impulse_decay_band():
    state = _state(
        bars=[(100, 103, 99, 102), (102, 104, 101, 103), (103, 105, 102, 104)],
        or_high=105,
        or_low=99,
        first_bar_open=100,
        atr=9.0,
        post_or_bars=[(104, 105, 103.5, 104.2)],
    )
    result = decide(
        state,
        OpenProxyConfig(
            or_width_atr_threshold=10.0,
            impulse_atr_threshold=0.2,
            enable_orb_selectivity_refinement=True,
            orb_selectivity_low_atr_threshold=5.0,
            orb_selectivity_high_impulse_threshold=2.4,
            enable_medium_impulse_decay_filter=True,
            medium_impulse_min_atr=8.0,
            medium_impulse_max_atr=15.0,
            medium_impulse_min=0.2,
            medium_impulse_max=2.0,
            medium_impulse_min_persistence=2,
        ),
    )
    assert result.pre_v3_selectivity_decision == "orb"
    assert result.decision == "mr"
    assert result.selectivity_medium_impulse_decay_caution is True
    assert "OPEN_PROXY_RANGE_SELECTIVITY_V4_MEDIUM_IMPULSE_DECAY" in result.reason


def test_open_proxy_v4_disabled_keeps_orb_decision():
    state = _state(
        bars=[(100, 103, 99, 102), (102, 104, 101, 103), (103, 105, 102, 104)],
        or_high=105,
        or_low=99,
        first_bar_open=100,
        atr=9.0,
        post_or_bars=[(104, 105, 103.5, 104.2)],
    )
    result = decide(
        state,
        OpenProxyConfig(
            or_width_atr_threshold=10.0,
            impulse_atr_threshold=0.2,
            enable_orb_selectivity_refinement=True,
            orb_selectivity_low_atr_threshold=5.0,
            orb_selectivity_high_impulse_threshold=2.4,
            enable_medium_impulse_decay_filter=False,
        ),
    )
    assert result.decision == "orb"
    assert result.selectivity_medium_impulse_decay_caution is False
    assert result.selectivity_v3_orb_blocked is False
