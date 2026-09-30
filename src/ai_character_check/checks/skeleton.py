"""Check 1: humanoid skeleton, fingers and toes."""

from __future__ import annotations

from ..context import Context
from ..findings import Finding, Report
from ..humanoid import FINGERS, SIDES, finger_segments, toe_segments

CORE_ROLES = ["hips", "spine", "neck", "head"] + [
    f"{s}_{r}" for s in SIDES for r in ("upper_arm", "lower_arm", "hand", "upper_leg", "lower_leg", "foot")
]


def check_skeleton(ctx: Context, rep: Report) -> None:
    m, hm, prof = ctx.model, ctx.hm, ctx.profile
    joints = m.joint_nodes
    rep.stats["joints"] = len(joints)
    rep.stats["skins"] = len(m.skins)
    if not m.skins:
        rep.add(Finding(
            "skeleton.no_skin", "error", "No skin: the character has no skeleton binding",
            measured=0, threshold=">= 1 skin",
            explanation="Without a glTF skin the mesh cannot be deformed by bones, so no animation will play on it.",
            fix="Rig the character (auto-rig or manual) and export with skinning enabled.",
        ))
        return
    if len(m.skins) > 1:
        rep.add(Finding(
            "skeleton.multiple_skins", "info", "More than one skin in the file",
            measured=len(m.skins), threshold=1,
            explanation="Several skins usually mean several armatures. Engines import them as separate skeletons.",
            fix="Merge the meshes onto one armature if they belong to the same character.",
        ))

    found = {r: m.node_names[n] for r, n in sorted(hm.roles.items())}
    if hm.role("hips") is None and hm.spine:
        found["hips"] = m.node_names[hm.spine[0]] + " (first spine bone)"
    rep.stats["humanoid_mapping"] = found
    rep.stats["humanoid_mapping_source"] = hm.source

    if not hm.roles and not hm.fingers:
        sample = [m.node_names[n] for n in joints[:8]]
        rep.add(Finding(
            "skeleton.not_humanoid", "error", "No humanoid bone names recognised",
            measured={"joints": len(joints), "sample_names": sample},
            threshold="hips, spine, head, arms, legs recognisable by name",
            explanation="None of the bone names match Mixamo, Unreal, VRM/Unity, rigify, biped or Character Creator "
                        "conventions. Retargeting and the finger/toe checks need a humanoid mapping.",
            fix="Rename the bones to a known convention (for example Mixamo or Unreal) or re-rig with a humanoid template.",
        ))
        return

    missing = [r for r in CORE_ROLES if r not in found]
    if missing:
        rep.add(Finding(
            "skeleton.core_bones_missing", "warning" if len(missing) < 6 else "error",
            "Humanoid core bones not found", measured=missing, threshold="all core bones present",
            explanation="Retargeting maps these bones between skeletons. Missing ones stay in bind pose or break the chain.",
            fix="Check the rig for missing limbs or unusual names; rename bones to one convention.",
        ))

    finger_report: dict[str, dict[str, int]] = {}
    for side in SIDES:
        if hm.role(f"{side}_hand") is None and not hm.finger_bones(side):
            continue
        segs = {f: len(finger_segments(hm, side, f)) for f in FINGERS}
        finger_report[side] = segs
        if sum(segs.values()) == 0:
            rep.add(Finding(
                f"skeleton.hand_no_fingers.{side}", "error", f"{side.capitalize()} hand has no finger bones",
                measured=segs, threshold="5 fingers, >= 2 segments each",
                explanation="Without finger bones the hand is a rigid block: no grip, no pointing, no relaxed hand pose. "
                            "Typical for auto-rigs run without a hand template.",
                fix="Re-run the auto-rigger with finger detection, or add finger chains and weight them.",
            ))
            continue
        absent = [f for f in FINGERS if segs[f] == 0]
        if absent:
            rep.add(Finding(
                f"skeleton.fingers_missing.{side}", prof["finger_incomplete_severity"],
                f"{side.capitalize()} hand is missing fingers", measured={"missing": absent, "segments": segs},
                threshold="5 fingers",
                explanation="Animations authored for five fingers leave the missing ones out; the mesh part stays stiff.",
                fix="Add the missing finger chains or check whether the mesh has fused fingers.",
            ))
        short = {f: n for f, n in segs.items() if 0 < n < 2}
        if short:
            rep.add(Finding(
                f"skeleton.finger_segments_low.{side}", prof["finger_incomplete_severity"],
                f"{side.capitalize()} hand has fingers with a single segment", measured=short,
                threshold=">= 2 segments per finger (thumb included)",
                explanation="One segment per finger cannot curl; fists and grips look like paddles.",
                fix="Use a rig template with 3 segments per finger (thumb at least 2).",
            ))
        two = {f: n for f, n in segs.items() if f != "thumb" and n == 2}
        if two:
            rep.add(Finding(
                f"skeleton.finger_segments_two.{side}", "info",
                f"{side.capitalize()} hand has fingers with 2 instead of 3 segments", measured=two,
                threshold="3 segments for index to little finger",
                explanation="Two segments bend, but retargeted animations for 3-segment hands lose the last joint.",
                fix="Acceptable for most uses; add a third segment if close-up hand animation matters.",
            ))
    rep.stats["finger_segments"] = finger_report

    toe_report = {}
    for side in SIDES:
        if hm.role(f"{side}_foot") is None:
            continue
        n = len(toe_segments(hm, side))
        toe_report[side] = n
        if n == 0:
            rep.add(Finding(
                f"skeleton.toes_missing.{side}", prof["toes_missing_severity"],
                f"{side.capitalize()} foot has no toe bone", measured=0, threshold=">= 1 toe bone",
                explanation="Without a toe (ball) bone the foot cannot roll during walking; the sole stays flat "
                            "and slides or clips through the floor.",
                fix="Add a toe/ball bone at the ball of the foot and weight the front of the foot to it.",
            ))
    rep.stats["toe_bones"] = toe_report
