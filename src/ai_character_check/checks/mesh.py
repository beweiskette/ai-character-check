"""Check 4: coincident but unmerged vertices (cracks vs. expected seams)."""

from __future__ import annotations

import numpy as np

from ..context import Context
from ..findings import Finding, Report

REL_EPS = 1e-5  # position tolerance relative to the primitive's bounding box diagonal
UV_EPS = 1e-5
NORMAL_DOT = 0.999


def analyse_primitive(prim) -> dict:
    """Classify coincident vertices of one primitive.

    Every vertex that shares its position with an earlier vertex of the same
    primitive is counted once, in the first matching class:
    ``duplicate`` (same UV and normal: nothing justifies the split, a crack),
    ``uv_seam`` (different UV, same normal: expected),
    ``normal_split`` (same UV, different normal: hard edge or crack),
    ``uv_and_normal_split`` (both differ: seam on a hard edge, usually expected).
    """
    tris = prim.triangles
    res = {"vertices": 0, "triangles": int(len(tris)), "duplicate": 0, "uv_seam": 0,
           "normal_split": 0, "uv_and_normal_split": 0, "crack_edges": 0}
    if len(tris) == 0:
        return res
    used = np.unique(tris.ravel())
    res["vertices"] = int(len(used))
    pos = prim.positions[used]
    finite = np.all(np.isfinite(pos), axis=1)
    used, pos = used[finite], pos[finite]
    if len(pos) < 2:
        return res
    diag = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))
    eps = max(diag * REL_EPS, 1e-9)
    keys = np.round(pos / eps).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    uv = prim.uv0
    nrm = prim.normals
    if nrm is not None:
        lens = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.where(lens > 1e-12, lens, 1.0)

    def same_uv(a, b):
        return uv is None or bool(np.all(np.abs(uv[a] - uv[b]) <= UV_EPS))

    def same_normal(a, b):
        return nrm is None or float(nrm[a] @ nrm[b]) >= NORMAL_DOT

    dup_groups = np.nonzero(counts > 1)[0]
    if len(dup_groups):
        order = np.argsort(inverse, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts)])
        for g in dup_groups:
            members = used[order[starts[g]:starts[g + 1]]]
            for i in range(1, len(members)):
                b = members[i]
                cls = "uv_and_normal_split"
                for a in members[:i]:
                    su, sn = same_uv(a, b), same_normal(a, b)
                    if su and sn:
                        cls = "duplicate"
                        break
                    if sn and cls not in ("duplicate",):
                        cls = "uv_seam"
                    elif su and cls == "uv_and_normal_split":
                        cls = "normal_split"
                res[cls] += 1

    # Crack edges: open edges that lie on top of another open edge with identical
    # UV and normal at both ends (the surface is continuous but not connected).
    weld = np.full(len(prim.positions), -1, dtype=np.int64)
    weld[used] = inverse
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e_sorted = np.sort(e, axis=1)
    _, e_inv, e_cnt = np.unique(e_sorted, axis=0, return_inverse=True, return_counts=True)
    open_edges = e_sorted[e_cnt[e_inv.ravel()] == 1]
    if len(open_edges):
        we = weld[open_edges]
        valid = np.all(we >= 0, axis=1) & (we[:, 0] != we[:, 1])
        open_edges, we = open_edges[valid], we[valid]
        swap = we[:, 0] > we[:, 1]
        we = np.where(swap[:, None], we[:, ::-1], we)
        open_edges = np.where(swap[:, None], open_edges[:, ::-1], open_edges)
        if len(we):
            _, w_inv, w_cnt = np.unique(we, axis=0, return_inverse=True, return_counts=True)
            w_inv = w_inv.ravel()
            e_order = np.argsort(w_inv, kind="stable")
            e_starts = np.concatenate([[0], np.cumsum(w_cnt)])
            for g in np.nonzero(w_cnt > 1)[0]:
                members = open_edges[e_order[e_starts[g]:e_starts[g + 1]]]
                a, b = members[0], members[1]
                if (same_uv(a[0], b[0]) and same_uv(a[1], b[1]) and same_normal(a[0], b[0])
                        and same_normal(a[1], b[1])):
                    res["crack_edges"] += 1
    return res


def check_mesh(ctx: Context, rep: Report) -> None:
    m = ctx.model
    totals = {"vertices": 0, "triangles": 0, "duplicate": 0, "uv_seam": 0, "normal_split": 0,
              "uv_and_normal_split": 0, "crack_edges": 0}
    per_prim = []
    seen = set()
    for p in m.primitives:
        key = (p.mesh_index, p.prim_index)
        if key in seen:
            continue  # the same mesh instanced by several nodes
        seen.add(key)
        r = analyse_primitive(p)
        for k in totals:
            totals[k] += r[k]
        if r["duplicate"] or r["normal_split"] or r["crack_edges"]:
            per_prim.append({"mesh": m.gltf.meshes[p.mesh_index].name or f"mesh_{p.mesh_index}",
                             "primitive": p.prim_index, **{k: r[k] for k in
                                                           ("duplicate", "normal_split", "crack_edges")}})
    rep.stats["vertices"] = totals["vertices"]
    rep.stats["triangles"] = totals["triangles"]
    rep.stats["primitives"] = len(seen)
    nv = max(totals["vertices"], 1)
    if not m.primitives:
        rep.add(Finding("mesh.no_geometry", "error", "The file contains no mesh geometry",
                        measured=0, threshold=">= 1 primitive",
                        explanation="There is nothing to render or skin.",
                        fix="Export the mesh together with the armature."))
        return
    if totals["duplicate"]:
        rep.add(Finding(
            "mesh.unmerged_vertices", ctx.profile["crack_severity"],
            "Coincident vertices that are not merged (same position, UV and normal)",
            measured={"count": totals["duplicate"], "share_pct": round(100 * totals["duplicate"] / nv, 2),
                      "crack_edges": totals["crack_edges"], "by_primitive": per_prim[:10]},
            threshold=0,
            explanation="Nothing in the vertex data justifies the split, so the surface is only visually closed. "
                        "Smoothing, subdivision, decimation or skinning pull the two sides apart and open visible "
                        "cracks. Image-to-3D exports often contain these.",
            fix="Merge by distance with a tiny threshold (Blender: Mesh > Clean Up > Merge by Distance) before "
                "smoothing, decimating or rigging.",
        ))
    if totals["normal_split"]:
        rep.add(Finding(
            "mesh.normal_splits", "info", "Coincident vertices with same UV but different normals",
            measured={"count": totals["normal_split"], "share_pct": round(100 * totals["normal_split"] / nv, 2)},
            threshold="informational",
            explanation="These are either intended hard edges or cracks with broken normals. The tool cannot tell "
                        "which; a crack shows as a dark line in the render.",
            fix="Check the listed meshes visually; merge if the surface should be smooth there.",
        ))
    rep.add(Finding(
        "mesh.seams", "info", "Mesh topology summary",
        measured={k: totals[k] for k in ("vertices", "triangles", "uv_seam", "uv_and_normal_split")},
        explanation="UV seams and hard edges need split vertices; these counts are expected and listed for reference.",
    ))
