"""Check 5: size in metres, up axis, feet on the ground, node scales."""

from __future__ import annotations

import numpy as np

from ..context import Context
from ..findings import Finding, Report


def check_scale(ctx: Context, rep: Report) -> None:
    prof, m = ctx.profile, ctx.model
    if len(ctx.points) == 0:
        return
    h = ctx.height
    rep.stats["height_m"] = round(h, 4)
    rep.stats["up_axis"] = ctx.up_label
    lo_hard, hi_hard = prof["height_hard_range"]
    lo_h, hi_h = prof["height_human_range"]
    measured = {"height_m": round(h, 4), "up_axis": ctx.up_label}
    if h < lo_hard or h > hi_hard:
        hint = ""
        if h > hi_hard:
            if 100 * lo_h <= h <= 100 * hi_h:
                hint = " The value fits a human height in centimetres."
            elif 1000 * lo_h <= h <= 1000 * hi_h:
                hint = " The value fits a human height in millimetres."
        elif 0.01 * lo_h <= h <= 0.01 * hi_h:
            hint = " The value fits a human height scaled by 0.01 (centimetre export applied twice)."
        rep.add(Finding(
            "scale.height_implausible", "error", f"Character height {h:.3g} m is outside the plausible range",
            measured=measured, threshold={"min_m": lo_hard, "max_m": hi_hard},
            explanation="glTF units are metres. A wrong unit makes physics, camera, navigation and retargeting "
                        "misbehave." + hint,
            fix="Apply the correct unit scale in the DCC tool and export with scale applied.",
        ))
    elif h < lo_h or h > hi_h:
        rep.add(Finding(
            "scale.height_unusual", "warning", f"Character height {h:.2f} m is outside human size",
            measured=measured, threshold={"min_m": lo_h, "max_m": hi_h},
            explanation="Fine for children, creatures or giants. For an adult human it hints at a unit or scale error.",
            fix="Ignore if intended; otherwise rescale to the expected height.",
        ))
    else:
        rep.add(Finding("scale.height", "info", f"Character height {h:.2f} m", measured=measured,
                        threshold={"min_m": lo_h, "max_m": hi_h}))

    if ctx.up_label != "+Y":
        rep.add(Finding(
            "scale.up_axis", "warning", f"Character stands along {ctx.up_label} instead of +Y",
            measured=ctx.up_label, threshold="+Y",
            explanation="glTF is Y-up. A character standing along another axis was exported with an unapplied "
                        "rotation (often a Z-up source) and imports lying down or upside down.",
            fix="Apply rotation in the DCC tool and export with +Y up.",
        ))

    proj = ctx.points @ ctx.up
    foot = float(proj.min())
    limit = max(0.01, prof["feet_offset_fraction"] * h)
    rep.stats["lowest_point_m"] = round(foot, 4)
    if abs(foot) > limit:
        where = "above" if foot > 0 else "below"
        rep.add(Finding(
            "scale.feet_offset", "warning", f"Lowest point is {abs(foot):.3f} m {where} the ground plane",
            measured={"lowest_point_m": round(foot, 4)}, threshold={"max_abs_m": round(limit, 4)},
            explanation="Engines place the character by its origin. If the feet are not at height 0 it floats or "
                        "sinks into the floor (a centred origin is typical for image-to-3D output).",
            fix="Move the mesh and armature so the soles touch height 0 and apply the transform.",
        ))

    scaled = {}
    relevant = set(m.joint_nodes)
    for p in m.primitives:
        relevant.add(p.node_index)
    ancestors = set()
    for n in relevant:
        a = m.parents[n]
        while a is not None and a not in ancestors:
            ancestors.add(a)
            a = m.parents[a]
    for n in sorted(relevant | ancestors):
        s = np.linalg.norm(m.local[n][:3, :3], axis=0)
        if np.any(np.abs(s - 1.0) > 1e-3):
            scaled[m.node_names[n]] = [round(float(x), 4) for x in s]
    if scaled:
        rep.add(Finding(
            "scale.node_scale", prof["node_scale_severity"], "Nodes in the skeleton chain carry a non-unit scale",
            measured=dict(list(scaled.items())[:10]), threshold=1.0,
            explanation="A scaled armature (for example 0.01 from a centimetre FBX) is a common reason for broken "
                        "retargeting, wrong physics sizes and animations that scale the character.",
            fix="Apply scale to armature and mesh before export so every node has scale 1.",
            details={"count": len(scaled)},
        ))
