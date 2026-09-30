"""Check 7: images, textures and material references."""

from __future__ import annotations

from ..context import Context
from ..findings import Finding, Report


def _is_pot(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def _texture_source(tex):
    if tex.source is not None:
        return tex.source
    for ext in (tex.extensions or {}).values():
        if isinstance(ext, dict) and isinstance(ext.get("source"), int):
            return ext["source"]
    return None


def _material_texture_refs(mat):
    refs = []
    pbr = mat.pbrMetallicRoughness
    if pbr is not None:
        for slot in ("baseColorTexture", "metallicRoughnessTexture"):
            t = getattr(pbr, slot, None)
            if t is not None and t.index is not None:
                refs.append((slot, t.index))
    for slot in ("normalTexture", "occlusionTexture", "emissiveTexture"):
        t = getattr(mat, slot, None)
        if t is not None and t.index is not None:
            refs.append((slot, t.index))
    return refs


def check_textures(ctx: Context, rep: Report) -> None:
    g = ctx.model.gltf
    images = ctx.model.images
    textures = g.textures or []
    missing = []
    for im in images:
        if im.missing_reason:
            missing.append({"image": im.name, "reason": im.missing_reason})
    for ti, tex in enumerate(textures):
        src = _texture_source(tex)
        if src is None or not (0 <= src < len(images)):
            missing.append({"texture": ti, "reason": f"texture points to missing image {src}"})
    for mi, mat in enumerate(g.materials or []):
        for slot, idx in _material_texture_refs(mat):
            if not (0 <= idx < len(textures)):
                missing.append({"material": mat.name or f"material_{mi}", "slot": slot,
                                "reason": f"missing texture {idx}"})
    if missing:
        rep.add(Finding(
            "textures.missing", "error", "Missing or unreadable textures",
            measured={"count": len(missing), "items": missing[:10]}, threshold=0,
            explanation="The engine imports the material without these maps; the character renders untextured or pink.",
            fix="Embed the textures (export as GLB) or ship the image files next to the .gltf with matching names.",
        ))
    sized = [im for im in images if im.width and im.height]
    npot = [{"image": im.name, "size": [im.width, im.height]} for im in sized
            if not (_is_pot(im.width) and _is_pot(im.height))]
    if npot:
        rep.add(Finding(
            "textures.non_power_of_two", ctx.profile["npot_severity"], "Textures with non-power-of-two size",
            measured={"count": len(npot), "items": npot[:10]}, threshold="width and height are powers of two",
            explanation="Older and mobile GPUs and some block compressors need power-of-two sizes; engines then "
                        "resize on import, which blurs the texture or wastes memory.",
            fix="Resize to the nearest power of two (for example 2048 x 2048) before import.",
        ))
    max_res = max((max(im.width, im.height) for im in sized), default=0)
    rep.stats["textures"] = {"images": len(images), "textures": len(textures), "materials": len(g.materials or []),
                             "max_resolution": max_res}
    rep.add(Finding(
        "textures.summary", "info", f"{len(images)} images, largest side {max_res} px",
        measured=rep.stats["textures"],
    ))
