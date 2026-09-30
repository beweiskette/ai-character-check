"""Run all checks on a loaded model."""

from __future__ import annotations

from ..context import build_context
from ..findings import Finding, Report
from ..gltf_io import Model
from ..profiles import get_profile
from .animation import check_animations
from .mesh import check_mesh
from .pose import check_pose
from .scale import check_scale
from .skeleton import check_skeleton
from .textures import check_textures
from .weights import check_weights

ALL_CHECKS = [check_skeleton, check_pose, check_weights, check_mesh, check_scale, check_animations, check_textures]


def blocked_resources_finding(blocked: list[dict]) -> Finding:
    """Error finding for resources that were not read because of where they point."""
    return Finding(
        "resource.external_path_blocked", "error",
        f"{len(blocked)} external resource path(s) outside the model folder were not read",
        measured={"count": len(blocked), "items": blocked[:10]},
        threshold="data: URIs or relative paths inside the model folder",
        explanation="The file points to data outside its own folder: a parent folder, an absolute path, a "
                    "file: or web URI, or a network (UNC) path. Reading such paths can expose unrelated local "
                    "files, and on Windows opening a network path can send your login credentials (NTLM) to "
                    "a remote server. The checker does not read them; the engine or Blender would.",
        fix="Embed the data (export as GLB) or copy the files next to the model and reference them with "
            "relative paths. If you trust the file and the paths are local, rerun with "
            "--allow-external-resources (network paths stay blocked).",
    )


def blocked_report(blocked: list[dict], profile: str, display_name: str) -> Report:
    """Report for a file whose geometry could not be loaded because a buffer was blocked."""
    get_profile(profile)
    rep = Report(display_name, profile)
    rep.add(blocked_resources_finding(blocked))
    return rep


def run_checks(model: Model, profile: str = "game", display_name: str | None = None) -> Report:
    prof = get_profile(profile)
    rep = Report(display_name or model.path, profile, source_note=model.source_note)
    if model.blocked_resources:
        rep.add(blocked_resources_finding(model.blocked_resources))
    for issue in model.issues:
        rep.add(Finding(
            "file.structure", "warning", issue,
            explanation="The file violates the glTF structure rules; importers may reject or repair it differently.",
            fix="Re-export from the DCC tool, or run the Khronos glTF validator for details.",
        ))
    ctx = build_context(model, prof)
    for check in ALL_CHECKS:
        try:
            check(ctx, rep)
        except Exception as exc:  # noqa: BLE001  - one broken check must not hide the others
            rep.add(Finding(
                f"internal.{check.__name__}", "warning", f"Check {check.__name__} failed: {exc!r}",
                explanation="The check crashed on this file. Other results are still valid.",
                fix="Please report the file structure (not the file) to the project.",
            ))
    return rep
