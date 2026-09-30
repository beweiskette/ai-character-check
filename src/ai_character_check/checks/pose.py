"""Check 2: rest pose classification (T-pose, A-pose, other)."""

from __future__ import annotations

import math

import numpy as np

from ..context import Context
from ..findings import Finding, Report
from ..humanoid import SIDES

T_LIMIT = 15.0
A_LIMIT = 60.0


def classify_angle(angle: float) -> str:
    if abs(angle) <= T_LIMIT:
        return "T-pose"
    if -A_LIMIT <= angle < -T_LIMIT:
        return "A-pose"
    if angle < -A_LIMIT:
        return "arms-down"
    return "arms-raised"


def arm_angles(ctx: Context) -> dict[str, float]:
    out = {}
    for side in SIDES:
        a = ctx.hm.role(f"{side}_upper_arm")
        b = ctx.hm.role(f"{side}_lower_arm")
        if b is None:
            b = ctx.hm.role(f"{side}_hand")
        if a is None or b is None:
            continue
        d = ctx.pos(b) - ctx.pos(a)
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            continue
        s = float(np.clip(d @ ctx.up / norm, -1.0, 1.0))
        out[side] = math.degrees(math.asin(s))
    return out


def check_pose(ctx: Context, rep: Report) -> None:
    m = ctx.model
    _check_bind_matches_default(ctx, rep)
    angles = arm_angles(ctx)
    if not angles:
        if m.skins:
            rep.add(Finding(
                "pose.unknown", "info", "Rest pose not measured (upper arm bones not found)",
                explanation="The pose check needs upper arm and forearm (or hand) bones.",
            ))
        return
    mean = sum(angles.values()) / len(angles)
    kind = classify_angle(mean)
    rep.stats["rest_pose"] = {"type": kind, "arm_angle_deg": round(mean, 1)}
    measured = {f"{k}_deg": round(v, 1) for k, v in angles.items()}
    measured["classification"] = kind
    threshold = {"t_pose": f"|angle| <= {T_LIMIT:g}", "a_pose": f"{T_LIMIT:g} < below horizontal <= {A_LIMIT:g}"}
    if kind in ("T-pose", "A-pose"):
        rep.add(Finding(
            "pose.rest_pose", "info", f"Rest pose: {kind} (upper arm {mean:+.1f} deg to horizontal)",
            measured=measured, threshold=threshold,
            explanation="Angle of the upper arm in bind pose, negative means below horizontal. "
                        "Retargeting works best when source and target share the same rest pose.",
            fix="Set the retarget pose of the engine (T or A) to match." if kind == "A-pose" else "",
        ))
    else:
        rep.add(Finding(
            "pose.rest_pose", ctx.profile["pose_other_severity"],
            f"Rest pose is neither T nor A ({kind}, {mean:+.1f} deg)", measured=measured, threshold=threshold,
            explanation="Auto-riggers expect arms away from the body. With arms down or raised, the arm and torso "
                        "weights often bleed into each other and retargeting needs a manual pose fix.",
            fix="Regenerate or re-pose the model in T- or A-pose before rigging.",
        ))
    if len(angles) == 2 and abs(angles["left"] - angles["right"]) > T_LIMIT:
        rep.add(Finding(
            "pose.arms_asymmetric", "warning", "Left and right arm differ in rest pose",
            measured=measured, threshold=f"difference <= {T_LIMIT:g} deg",
            explanation="An asymmetric bind pose makes mirrored animations look lopsided.",
            fix="Re-pose the arms symmetrically before rigging.",
        ))


def _check_bind_matches_default(ctx: Context, rep: Report) -> None:
    m = ctx.model
    worst = 0.0
    for skin in m.skins:
        if len(skin.joints) < 2:
            continue
        mats = np.stack([m.world[j] @ skin.inverse_bind[i] for i, j in enumerate(skin.joints)])
        ref = np.median(mats, axis=0)
        scale = max(1e-9, float(np.abs(ref[:3, :3]).max()))
        rot_dev = float(np.abs(mats[:, :3, :3] - ref[:3, :3]).max()) / scale
        t_dev = float(np.abs(mats[:, :3, 3] - ref[:3, 3]).max()) / max(ctx.height, 1e-6)
        worst = max(worst, rot_dev, t_dev)
    if worst > 0.02:
        rep.add(Finding(
            "pose.default_differs_from_bind", "info", "Stored node pose differs from the bind pose",
            measured=round(worst, 3), threshold="<= 0.02",
            explanation="The node transforms in the file are not the pose the mesh was bound in. Measurements in this "
                        "report use the stored pose, which is what an engine shows before any animation plays.",
            fix="Apply the rest pose in the DCC tool before export if the stored pose was not intended.",
        ))
