"""Tests for preset loading and allocator policy normalization."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from run_validation_pack import _load_preset
from validation.candidate_openfix import build_runner_kwargs_from_preset
from validation.preset_utils import normalize_allocator_policy


def test_normalize_allocator_policy_legacy_v2_hyst():
    assert normalize_allocator_policy("ALLOC_V2_HYST") == "v2"


def test_normalize_allocator_policy_open_proxy():
    assert normalize_allocator_policy("open_proxy_v1") == "open_proxy_v1"


def test_normalize_allocator_policy_unknown_raises():
    with pytest.raises(ValueError):
        normalize_allocator_policy("mystery_policy")


def test_load_mainline_baseline_preset_normalizes_allocator():
    preset = _load_preset("mainline_combine_v1")
    assert preset["allocator_policy"] == "v2"
    assert preset["allocator_v2_trend_open_threshold"] == 25.0
    assert preset["allocator_v2_rising_threshold"] == 20.0


def test_load_candidate_alias_preset_uses_open_proxy_settings():
    preset = _load_preset("mainline_combine_v1_1_allocator_openfix")
    assert preset["allocator_policy"] == "open_proxy_v1"
    assert preset["alloc_openproxy_or_width_atr"] == 2.2
    assert preset["alloc_openproxy_impulse_atr"] == 0.9


def test_load_v1_2_orb_selectivity_preset_enables_refinement():
    preset = _load_preset("mainline_combine_v1_2_orb_selectivity")
    assert preset["allocator_policy"] == "open_proxy_v1"
    assert preset["alloc_openproxy_enable_orb_selectivity_refinement"] == "on"
    assert preset["alloc_openproxy_low_atr_threshold"] == 10.0
    assert preset["alloc_openproxy_min_persistence_in_low_atr"] == 2


def test_load_v1_3_orb_selectivity_refine_preset_enables_narrow_filter():
    preset = _load_preset("mainline_combine_v1_3_orb_selectivity_refine")
    assert preset["allocator_policy"] == "open_proxy_v1"
    assert preset["alloc_openproxy_enable_orb_selectivity_refinement"] == "on"
    assert preset["alloc_openproxy_medium_impulse_weak_persistence_filter_enabled"] == "on"


def test_load_v1_2_preset_has_no_v1_3_medium_impulse_override():
    preset = _load_preset("mainline_combine_v1_2_orb_selectivity")
    assert preset["allocator_policy"] == "open_proxy_v1"
    assert preset.get("alloc_openproxy_medium_impulse_weak_persistence_filter_enabled", "off") == "off"


def test_load_v1_4_execution_bridge_preset_enables_new_filters_and_sizing():
    preset = _load_preset("mainline_combine_v1_4_execution_bridge")
    assert preset["allocator_policy"] == "open_proxy_v1"
    assert preset["alloc_openproxy_medium_impulse_decay_filter_enabled"] == "on"
    assert preset["alloc_openproxy_medium_impulse_min_atr"] == 8.0
    assert preset["alloc_openproxy_medium_impulse_max_atr"] == 15.0
    assert preset["dyn_v3_atr_traction_scale_enabled"] == "on"
    assert preset["dyn_v3_consistency_brake_enabled"] == "on"


def test_load_v1_5_signal_quality_preset_carries_signal_quality_overrides():
    preset = _load_preset("mainline_combine_v1_5_signal_quality")
    assert preset["allocator_policy"] == "open_proxy_v1"
    assert preset["mr_sigma_entry"] == 1.3
    assert preset["mr_reclaim_mode"] == "off"
    assert preset["mr_first_outside_enabled"] == "on"
    assert preset["mr_dedupe_enabled"] == "on"
    assert preset["mr_cooldown_bars"] == 1
    assert preset["alloc_openproxy_medium_impulse_min"] == 1.0
    assert preset["alloc_openproxy_medium_impulse_max"] == 1.8


def test_candidate_openfix_builder_uses_v1_5_signal_quality_mr_fields():
    kwargs = build_runner_kwargs_from_preset(
        "mainline_combine_v1_5_signal_quality",
        "artifacts/validation_runs",
    )
    assert kwargs["mr_sigma_entry"] == 1.3
    assert kwargs["mr_reclaim_mode"] == "off"
    assert kwargs["mr_first_outside_enabled"] is True
    assert kwargs["mr_dedupe_enabled"] is True
    assert kwargs["mr_cooldown_bars"] == 1
    assert kwargs["alloc_openproxy_medium_impulse_min"] == 1.0
    assert kwargs["alloc_openproxy_medium_impulse_max"] == 1.8
