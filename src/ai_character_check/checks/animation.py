"""Check 6: animation clips."""

from __future__ import annotations

import numpy as np

from ..context import Context
from ..findings import Finding, Report


def _root_node(ctx: Context):
    hm, m = ctx.hm, ctx.model
    if hm.role("hips") is not None:
        return hm.role("hips")
    if hm.spine:
        return hm.spine[0]
    joints = m.joint_nodes
    return min(joints, key=m.depth) if joints else None


def check_animations(ctx: Context, rep: Report) -> None:
    m, prof = ctx.model, ctx.profile
    joints = set(m.joint_nodes)
    skeleton_nodes = set(joints)
    for j in joints:
        a = m.parents[j]
        while a is not None:
            skeleton_nodes.add(a)
            a = m.parents[a]
    root = _root_node(ctx)
    root_chain = set()
    a = root
    while a is not None:
        root_chain.add(a)
        a = m.parents[a]
    clips = []
    for anim in m.animations:
        info = {"name": anim.name, "channels": len(anim.channels)}
        tmin, tmax = np.inf, -np.inf
        nan_keys = 0
        bad_time = []
        missing_targets = set()
        non_skel = set()
        scale_dev = {}
        quat_dev = 0.0
        root_disp = 0.0
        for ch in anim.channels:
            if ch.node is None:
                continue
            if not (0 <= ch.node < len(m.node_names)):
                missing_targets.add(ch.node)
                continue
            name = m.node_names[ch.node]
            if len(ch.times):
                finite_t = ch.times[np.isfinite(ch.times)]
                if len(finite_t):
                    tmin, tmax = min(tmin, finite_t.min()), max(tmax, finite_t.max())
            values = ch.values.reshape(len(ch.values), -1)
            nan_keys += int((~np.isfinite(ch.times)).sum()) + int((~np.isfinite(values)).any(axis=1).sum())
            if len(ch.times) > 1 and np.any(np.diff(ch.times) <= 0):
                bad_time.append(name)
            if joints and ch.node not in skeleton_nodes:
                non_skel.add(name)
            vals = values[np.all(np.isfinite(values), axis=1)]
            if not len(vals):
                continue
            if ch.path == "scale":
                dev = float(np.abs(vals - 1.0).max())
                if dev > prof["scale_key_warning"]:
                    scale_dev[name] = {"min": round(float(vals.min()), 4), "max": round(float(vals.max()), 4)}
            elif ch.path == "rotation" and vals.shape[-1] == 4:
                quat_dev = max(quat_dev, float(np.abs(np.linalg.norm(vals, axis=1) - 1.0).max()))
            elif ch.path == "translation" and ch.node in root_chain and vals.shape[-1] == 3:
                parent = m.parents[ch.node]
                basis = m.world[parent][:3, :3] if parent is not None else np.eye(3)
                d = (vals - vals[0]) @ basis.T
                d = d - np.outer(d @ ctx.up, ctx.up)
                root_disp = max(root_disp, float(np.linalg.norm(d, axis=1).max()))
        length = float(tmax - tmin) if np.isfinite(tmax) and np.isfinite(tmin) else 0.0
        threshold = 0.1 * ctx.height if ctx.height > 0 else 0.1
        info.update({"length_s": round(length, 4), "root_motion": root_disp > threshold,
                     "root_displacement_m": round(root_disp, 4)})
        clips.append(info)
        label = f"'{anim.name}'"
        if nan_keys:
            rep.add(Finding(
                "animation.nan_keys", "error", f"Clip {label} contains NaN or infinite keys",
                measured={"clip": anim.name, "count": nan_keys}, threshold=0,
                explanation="Non-finite keys make bones vanish or explode for the frames that use them; some engines "
                            "reject the whole clip.",
                fix="Re-bake the clip; check the retarget or filter step that produced it.",
            ))
        if bad_time:
            rep.add(Finding(
                "animation.time_not_increasing", "error", f"Clip {label} has key times that do not increase",
                measured={"clip": anim.name, "nodes": sorted(set(bad_time))[:10]}, threshold="strictly increasing",
                explanation="glTF requires strictly increasing key times; importers may reject or reorder the clip.",
                fix="Re-bake the animation with one key per frame.",
            ))
        if missing_targets:
            rep.add(Finding(
                "animation.missing_target", "error", f"Clip {label} animates nodes that do not exist",
                measured={"clip": anim.name, "node_indices": sorted(missing_targets)[:10]}, threshold=0,
                explanation="The clip references bones that are not in this file, typically after deleting bones "
                            "or merging files.",
                fix="Re-export the clip against the current skeleton.",
            ))
        if non_skel:
            rep.add(Finding(
                "animation.non_skeleton_target", "warning", f"Clip {label} animates nodes outside the skeleton",
                measured={"clip": anim.name, "nodes": sorted(non_skel)[:10], "count": len(non_skel)},
                threshold=0,
                explanation="These nodes are not bones of the skin. Engines usually drop such tracks when they import "
                            "the clip for the skeleton, so the motion is lost.",
                fix="Bake the motion onto skeleton bones or remove the extra tracks.",
            ))
        if scale_dev:
            worst = max(max(abs(v["min"] - 1), abs(v["max"] - 1)) for v in scale_dev.values())
            err = prof["scale_key_error"]
            sev = "error" if err is not None and worst > err else "warning"
            rep.add(Finding(
                "animation.scale_keys", sev, f"Clip {label} has scale keys far from 1",
                measured={"clip": anim.name, "bones": dict(list(scale_dev.items())[:10]), "count": len(scale_dev),
                          "max_deviation": round(worst, 4)},
                threshold={"warning_deviation": prof["scale_key_warning"], "error_deviation": err},
                explanation="Scale keys that are not 1 usually come from a file that was normalised or rescaled in a DCC "
                            "tool after rigging. Retargeting copies them and the character grows, shrinks or "
                            "collapses.",
                fix="Apply scale on the armature before baking, or strip scale tracks from the clip.",
            ))
        if quat_dev > 0.01:
            rep.add(Finding(
                "animation.rotation_not_normalized", "warning", f"Clip {label} has non-unit rotation quaternions",
                measured={"clip": anim.name, "max_norm_deviation": round(quat_dev, 4)}, threshold=0.01,
                explanation="glTF requires unit quaternions. Engines normalise them, which changes the interpolation.",
                fix="Re-export the clip; most exporters normalise on bake.",
            ))
        if length <= 0 and anim.channels:
            rep.add(Finding(
                "animation.zero_length", "info", f"Clip {label} has zero length (single pose)",
                measured={"clip": anim.name, "length_s": length}, threshold="> 0 s",
                explanation="A one-key clip is a pose, not an animation. Fine for poses, wrong for loops.",
            ))
    rep.stats["animations"] = clips
    for c in clips:
        rep.add(Finding(
            "animation.clip", "info",
            f"Clip '{c['name']}': {c['length_s']} s, {'root motion' if c['root_motion'] else 'in place'}",
            measured=c,
            threshold={"root_motion_min_displacement_m": round(0.1 * ctx.height, 4) if ctx.height else 0.1},
        ))
