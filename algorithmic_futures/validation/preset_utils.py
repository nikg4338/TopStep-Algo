"""Helpers for versioned preset loading and backward-compatible policy mapping.

This module exists to keep frozen presets reproducible while allowing the
runtime to evolve its internal enum names. In particular, the profitable
baseline preset stores legacy allocator labels (for example
``ALLOC_V2_HYST``), while the live replay and validation runners expect the
normalized runtime names (for example ``v2``).
"""

from __future__ import annotations

from typing import Final


ALLOCATOR_POLICY_ALIASES: Final[dict[str, str]] = {
    "none": "none",
    "off": "none",
    "v1": "v1",
    "alloc_v1": "v1",
    "v2": "v2",
    "alloc_v2": "v2",
    "alloc_v2_hyst": "v2",
    "open_proxy_v1": "open_proxy_v1",
}


def normalize_allocator_policy(policy: str | None) -> str:
    """Normalize allocator policy labels to the runtime enum.

    Parameters
    ----------
    policy:
        Raw preset or CLI value.

    Returns
    -------
    str
        One of ``none``, ``v1``, ``v2``, or ``open_proxy_v1``.

    Raises
    ------
    ValueError
        If the supplied label is unknown.
    """
    if policy is None:
        return "none"
    normalized = str(policy).strip().lower()
    if normalized in ALLOCATOR_POLICY_ALIASES:
        return ALLOCATOR_POLICY_ALIASES[normalized]
    raise ValueError(
        "Unknown allocator policy "
        f"'{policy}'. Expected one of: {', '.join(sorted(set(ALLOCATOR_POLICY_ALIASES.values())))}"
    )
