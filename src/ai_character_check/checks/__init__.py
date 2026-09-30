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


def run_checks(model: Model, profile: str = "game", display_name: str | None = None) -> Report:
    prof = get_profile(profile)
    rep = Report(display_name or model.path, profile, source_note=model.source_note)
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
