"""Command-line interface: ``aicc check`` and ``aicc render``.

Exit codes: 0 = no errors, 1 = errors found (or warnings with --fail-on warning),
2 = the file could not be checked (unreadable, Blender missing for FBX, bad arguments).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import contextmanager

from . import __version__
from .blender import (NOT_FOUND_MESSAGE, BlenderFailed, convert_to_glb, find_blender, render)
from .checks import run_checks
from .gltf_io import LoadError, load_model
from .humanoid import build_map
from .output import to_html, to_json, to_text
from .profiles import PROFILES

NEEDS_BLENDER = (".fbx",)


class CliError(Exception):
    pass


@contextmanager
def as_gltf(path: str, blender_arg: str | None):
    """Yield (gltf_path, note). FBX input is converted with Blender into a temp dir."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in NEEDS_BLENDER:
        yield path, None
        return
    blender = find_blender(blender_arg)
    if blender is None:
        raise CliError(f"{ext.upper()[1:]} input needs Blender for conversion. {NOT_FOUND_MESSAGE}")
    with tempfile.TemporaryDirectory(prefix="aicc-") as tmp:
        dst = os.path.join(tmp, "converted.glb")
        try:
            version = convert_to_glb(path, dst, blender)
        except BlenderFailed as exc:
            raise CliError(f"FBX conversion failed: {exc}") from exc
        note = (f"converted from {ext[1:].upper()} with Blender {version}; scale, axis and bone findings reflect "
                "Blender's importer and glTF exporter")
        yield dst, note


def cmd_check(a) -> int:
    try:
        with as_gltf(a.model, a.blender) as (gpath, note):
            model = load_model(gpath)
            model.source_note = note
            rep = run_checks(model, a.profile, display_name=os.path.basename(a.model) if a.relative_name
                             else os.path.abspath(a.model))
    except (LoadError, CliError) as exc:
        print(f"aicc: {exc}", file=sys.stderr)
        return 2
    if a.format == "json":
        out = to_json(rep) + "\n"
    elif a.format == "html":
        out = to_html(rep)
    else:
        out = to_text(rep, verbose=a.verbose)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(out)
    else:
        sys.stdout.write(out)
    if rep.count("error"):
        return 1
    if a.fail_on == "warning" and rep.count("warning"):
        return 1
    return 0


def cmd_render(a) -> int:
    blender = find_blender(a.blender)
    if blender is None:
        print(f"aicc: cannot render. {NOT_FOUND_MESSAGE}", file=sys.stderr)
        return 2
    try:
        with as_gltf(a.model, a.blender) as (gpath, _note):
            model = load_model(gpath)
            hm = build_map(model)
            hands = {s: model.node_names[hm.role(f"{s}_hand")] for s in ("left", "right")
                     if hm.role(f"{s}_hand") is not None}
            clip = model.animations[0].name if model.animations else None
            manifest = render(gpath, a.out, blender, hands, clip, size=a.size, frames=a.frames)
    except (LoadError, CliError, BlenderFailed) as exc:
        print(f"aicc: {exc}", file=sys.stderr)
        return 2
    outs = manifest.get("outputs", {})
    for key, val in outs.items():
        if isinstance(val, str):
            print(f"{key}: {os.path.join(a.out, val)}")
    for note in manifest.get("notes", []):
        print(f"note: {note}")
    return 0 if outs else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aicc", description="Check AI-generated and auto-rigged 3D characters "
                                                         "before importing them into a game engine.")
    p.add_argument("--version", action="version", version=f"ai-character-check {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="run all checks on a .glb/.gltf (or .fbx via Blender)")
    c.add_argument("model")
    c.add_argument("--format", choices=["text", "json", "html"], default="text")
    c.add_argument("--profile", choices=sorted(PROFILES), default="game")
    c.add_argument("--out", help="write the report to this file instead of stdout")
    c.add_argument("--fail-on", choices=["error", "warning"], default="error",
                   help="exit with 1 on errors (default) or on warnings too")
    c.add_argument("--blender", help="Blender executable (for FBX input); default: $BLENDER, PATH, standard folders")
    c.add_argument("--verbose", "-v", action="store_true", help="text format: show info findings and explanations")
    c.add_argument("--relative-name", action="store_true", help="report only the file name, not the full path")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("render", help="render front view, both hands and an animation strip with Blender")
    r.add_argument("model")
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--blender", help="Blender executable; default: $BLENDER, PATH, standard folders")
    r.add_argument("--size", type=int, default=512, help="image size in pixels (square)")
    r.add_argument("--frames", type=int, default=6, help="frames in the animation strip")
    r.set_defaults(func=cmd_render)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
