"""Threshold profiles.

``game``: the character goes into a real-time engine (Unreal, Unity, Godot).
``preview``: the character is only looked at (web viewer, turntable); rig
details that matter for animation retargeting are downgraded.
"""

from __future__ import annotations

PROFILES: dict[str, dict] = {
    "game": {
        "max_influences": 4,
        "influences_severity": "warning",
        "finger_incomplete_severity": "warning",
        "toes_missing_severity": "warning",
        "unweighted_finger_severity": "warning",
        "zero_weight_vertices_severity": "error",
        "unnormalized_severity": "warning",
        "weight_sum_tolerance": 0.01,
        "leakage_min_weight": 0.1,
        "leakage_margin": 0.04,  # fraction of height a vertex must be off the midline
        "leakage_error_share": 0.05,
        "hand_single_bone_share": 0.9,
        "hand_single_finger_share": 0.6,
        "height_hard_range": (0.3, 3.0),
        "height_human_range": (1.2, 2.2),
        "feet_offset_fraction": 0.03,
        "scale_key_warning": 0.05,
        "scale_key_error": 0.5,
        "npot_severity": "warning",
        "crack_severity": "warning",
        "pose_other_severity": "warning",
        "node_scale_severity": "warning",
    },
    "preview": {
        "max_influences": None,
        "influences_severity": "info",
        "finger_incomplete_severity": "info",
        "toes_missing_severity": "info",
        "unweighted_finger_severity": "info",
        "zero_weight_vertices_severity": "warning",
        "unnormalized_severity": "info",
        "weight_sum_tolerance": 0.01,
        "leakage_min_weight": 0.1,
        "leakage_margin": 0.04,
        "leakage_error_share": None,
        "hand_single_bone_share": 0.9,
        "hand_single_finger_share": 0.6,
        "height_hard_range": (0.3, 3.0),
        "height_human_range": (1.2, 2.2),
        "feet_offset_fraction": 0.03,
        "scale_key_warning": 0.05,
        "scale_key_error": None,
        "npot_severity": "info",
        "crack_severity": "warning",
        "pose_other_severity": "info",
        "node_scale_severity": "info",
    },
}


def get_profile(name: str) -> dict:
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose from {', '.join(PROFILES)}")
    return dict(PROFILES[name], name=name)
