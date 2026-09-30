"""Check 3: skin weights (normalisation, influences, finger weights, side leakage, hand ownership)."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..context import Context
from ..findings import Finding, Report
from ..humanoid import FINGERS, SIDES, finger_segments

WEIGHTED = 0.01  # a bone "owns" a vertex from this weight on


def _gather(ctx: Context):
    prims = ctx.model.skinned_primitives()
    if not prims:
        return None
    jn = [p.joint_nodes for p in prims]
    k = max(a.shape[1] for a in jn)
    jn = np.concatenate([np.pad(a, ((0, 0), (0, k - a.shape[1])), constant_values=-1) for a in jn])
    w = np.concatenate([np.pad(p.weights, ((0, 0), (0, k - p.weights.shape[1]))) for p in prims])
    pos = np.concatenate([p.world_positions for p in prims])
    # Only vertices that are part of a triangle (or all, for point clouds) count.
    used = []
    for p in prims:
        mask = np.zeros(len(p.positions), dtype=bool)
        tris = p.triangles
        if len(tris):
            mask[tris.ravel()] = True
        else:
            mask[:] = True
        used.append(mask)
    used = np.concatenate(used)
    return jn[used], w[used], pos[used]


def _pct(x: float) -> float:
    return round(100.0 * x, 2)


def check_weights(ctx: Context, rep: Report) -> None:
    data = _gather(ctx)
    if data is None:
        return
    jn, w, pos = data
    prof, hm, m = ctx.profile, ctx.hm, ctx.model
    n = len(w)
    if n == 0:
        return
    w = np.where(np.isfinite(w), w, 0.0)
    total = w.sum(axis=1)

    zero = int((total < 1e-6).sum())
    if zero:
        rep.add(Finding(
            "weights.zero_weight_vertices", prof["zero_weight_vertices_severity"],
            "Vertices without any bone weight", measured={"count": zero, "share_pct": _pct(zero / n)},
            threshold=0,
            explanation="These vertices follow no bone. Depending on the engine they stay at the origin, stay behind "
                        "while the body moves, or get dropped on import.",
            fix="Re-run automatic weights or flood-fill the missing area to its nearest bone.",
        ))
    tol = prof["weight_sum_tolerance"]
    bad_sum = int(((np.abs(total - 1.0) > tol) & (total >= 1e-6)).sum())
    if bad_sum:
        rep.add(Finding(
            "weights.not_normalized", prof["unnormalized_severity"], "Vertex weights do not sum to 1",
            measured={"count": bad_sum, "share_pct": _pct(bad_sum / n),
                      "min_sum": round(float(total[total >= 1e-6].min()), 4),
                      "max_sum": round(float(total.max()), 4)},
            threshold=f"|sum - 1| <= {tol}",
            explanation="glTF requires normalised weights. Engines renormalise on import, which silently shifts the "
                        "deformation compared to the DCC tool.",
            fix="Normalise all weights before export (Blender: Weights > Normalize All).",
        ))

    influences = (w > 1e-4).sum(axis=1)
    max_inf = int(influences.max()) if n else 0
    rep.stats["max_influences"] = max_inf
    limit = prof["max_influences"]
    if limit is not None:
        over = int((influences > limit).sum())
        if over:
            rep.add(Finding(
                "weights.too_many_influences", prof["influences_severity"],
                f"Vertices with more than {limit} bone influences",
                measured={"count": over, "share_pct": _pct(over / n), "max": max_inf}, threshold=limit,
                explanation="Many engines (and mobile targets) keep only the 4 strongest influences. The dropped "
                            "weights change the deformation, most visibly at shoulders and hips.",
                fix=f"Limit influences to {limit} and normalise before export.",
            ))

    # Weighted vertex count per bone.
    mask = (w > WEIGHTED) & (jn >= 0)
    counts = Counter(jn[mask].tolist())

    fingers_all_unweighted = {}
    for side in SIDES:
        segs = []
        for f in FINGERS:
            segs.extend(finger_segments(hm, side, f))
        if not segs:
            continue
        unweighted = [s for s in segs if counts.get(s, 0) == 0]
        fingers_all_unweighted[side] = len(unweighted) == len(segs)
        if len(unweighted) == len(segs):
            rep.add(Finding(
                f"weights.fingers_unweighted.{side}", "error",
                f"{side.capitalize()} finger bones exist but carry no vertices",
                measured={"finger_bones": len(segs), "weighted": 0}, threshold="every finger bone weighted",
                explanation="The skeleton has fingers, but the hand mesh is bound to other bones. Finger animation "
                            "moves invisible bones; the hand will not bend.",
                fix="Re-run weight transfer for the hand, or weight each finger segment to its mesh finger.",
            ))
        elif unweighted:
            rep.add(Finding(
                f"weights.finger_bones_unweighted.{side}", prof["unweighted_finger_severity"],
                f"Some {side} finger bones carry no vertices",
                measured={"bones": [m.node_names[s] for s in unweighted], "count": len(unweighted),
                          "of": len(segs)},
                threshold=0,
                explanation="These finger segments will rotate without moving any geometry.",
                fix="Weight the listed segments, or remove them if the mesh has no matching finger part.",
            ))

    _check_leakage(ctx, rep, jn, w, pos)
    _check_hand_ownership(ctx, rep, jn, w, fingers_all_unweighted)


def _check_leakage(ctx: Context, rep: Report, jn, w, pos) -> None:
    if ctx.lateral is None or ctx.hips is None or ctx.height <= 0:
        return
    prof, hm, m = ctx.profile, ctx.hm, ctx.model
    side_of = {}
    for node, info in hm.bones.items():
        if info.side is not None:
            side_of[node] = 1 if info.side == "left" else -1
    if not side_of:
        return
    offset = (pos - ctx.hips) @ ctx.lateral
    margin = prof["leakage_margin"] * ctx.height
    vert_side = np.where(offset > margin, 1, np.where(offset < -margin, -1, 0))
    lookup = np.vectorize(lambda j: side_of.get(int(j), 0), otypes=[np.int64])
    bone_side = lookup(jn) if jn.size else np.zeros_like(jn)
    min_w = prof["leakage_min_weight"]
    leak_mask = (bone_side == -vert_side[:, None]) & (vert_side[:, None] != 0) & (w >= min_w)
    leak_verts = leak_mask.any(axis=1)
    count = int(leak_verts.sum())
    if not count:
        return
    dominant = jn[np.arange(len(w)), np.argmax(w, axis=1)]
    pairs = Counter()
    idx = np.nonzero(leak_verts)[0]
    for v in idx[:20000]:
        for k in np.nonzero(leak_mask[v])[0]:
            pairs[(m.node_names[int(dominant[v])], m.node_names[int(jn[v, k])])] += 1
    top = [{"vertex_owner": a, "leaks_to": b, "vertices": c} for (a, b), c in pairs.most_common(5)]
    share = count / len(w)
    err = prof["leakage_error_share"]
    sev = "error" if err is not None and share > err else "warning"
    rep.add(Finding(
        "weights.left_right_leakage", sev, "Vertices weighted to bones on the opposite body side",
        measured={"count": count, "share_pct": _pct(share), "top_pairs": top},
        threshold={"min_weight": min_w, "min_distance_from_midline_m": round(margin, 4),
                   "error_share_pct": None if err is None else _pct(err)},
        explanation="Auto-rigs often bind inner thighs, fingers or arms close to the body to the wrong side. When "
                    "one leg or hand moves, parts of the other side get dragged along.",
        fix="Clear the opposite-side weights in the listed regions (mirror weights from the clean side).",
    ))


def _check_hand_ownership(ctx: Context, rep: Report, jn, w, fingers_all_unweighted) -> None:
    hm, m, prof = ctx.hm, ctx.model, ctx.profile
    dominant = jn[np.arange(len(w)), np.argmax(w, axis=1)]
    shares_all = {}
    for side in SIDES:
        hand = hm.role(f"{side}_hand")
        chains = {f: hm.fingers.get((side, f), []) for f in FINGERS}
        finger_nodes = [n for c in chains.values() for n in c]
        if hand is None or not finger_nodes or fingers_all_unweighted.get(side, True):
            continue
        hand_set = set(finger_nodes) | {hand}
        in_hand = np.isin(dominant, list(hand_set))
        total = int(in_hand.sum())
        if total < 8:
            continue
        dom = dominant[in_hand]
        shares = {"hand": float((dom == hand).sum()) / total}
        for f, chain in chains.items():
            shares[f] = float(np.isin(dom, chain).sum()) / total if chain else 0.0
        per_bone = Counter(dom.tolist())
        top_bone, top_count = per_bone.most_common(1)[0]
        top_share = top_count / total
        shares_all[side] = {k: _pct(v) for k, v in shares.items()}
        measured = {"hand_vertices": total, "share_pct": shares_all[side],
                    "top_bone": m.node_names[top_bone], "top_bone_share_pct": _pct(top_share)}
        if top_share > prof["hand_single_bone_share"]:
            rep.add(Finding(
                f"weights.hand_single_bone.{side}", "warning",
                f"One bone owns almost the whole {side} hand", measured=measured,
                threshold={"max_single_bone_share_pct": _pct(prof["hand_single_bone_share"])},
                explanation="If one bone owns nearly all hand vertices, the fingers either do not bend or the whole hand "
                            "follows one finger.",
                fix="Re-weight the hand so each finger segment owns its part of the mesh.",
            ))
            continue
        chain_shares = {f: shares[f] for f in FINGERS}
        top_f = max(chain_shares, key=chain_shares.get)
        if chain_shares[top_f] > prof["hand_single_finger_share"]:
            rep.add(Finding(
                f"weights.hand_finger_dominates.{side}", "warning",
                f"The {top_f} finger chain owns most of the {side} hand", measured=measured,
                threshold={"max_single_finger_share_pct": _pct(prof["hand_single_finger_share"])},
                explanation="Typical for image-to-3D hands with fused or badly separated fingers: the auto-rig binds a "
                            "block of fingers to one chain, which shows up as claw or spread hands when animated.",
                fix="Separate the fingers in the mesh (or rebuild the hand) and re-weight; check the render of the hands.",
            ))
    if shares_all:
        rep.stats["hand_vertex_shares_pct"] = shares_all
