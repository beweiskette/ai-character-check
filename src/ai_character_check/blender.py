"""Optional Blender integration: FBX conversion and preview renders."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from importlib import resources
from typing import Optional


class BlenderNotFound(Exception):
    pass


class BlenderFailed(Exception):
    pass


NOT_FOUND_MESSAGE = (
    "Blender was not found. Install Blender and either put it on PATH, "
    "set the BLENDER environment variable to the executable, or pass --blender PATH."
)


def _version_key(path: str):
    m = re.search(r"(\d+)\.(\d+)", path)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def find_blender(explicit: Optional[str] = None) -> Optional[str]:
    """Return a Blender executable: explicit path, $BLENDER, PATH, then standard install folders."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    env = os.environ.get("BLENDER")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("blender")
    if found:
        return found
    candidates: list[str] = []
    if sys.platform.startswith("win"):
        for root in {os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles(x86)")}:
            if root:
                candidates += glob.glob(os.path.join(root, "Blender Foundation", "Blender*", "blender.exe"))
    elif sys.platform == "darwin":
        candidates += glob.glob("/Applications/Blender*.app/Contents/MacOS/Blender")
    else:
        candidates += [p for p in ("/snap/bin/blender", "/usr/local/bin/blender", "/opt/blender/blender")
                       if os.path.isfile(p)]
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        return None
    return sorted(candidates, key=_version_key)[-1]


def script_path(name: str) -> str:
    return str(resources.files("ai_character_check").joinpath("blender_scripts", name))


def run_script(blender: str, script: str, script_args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    cmd = [blender, "-b", "--factory-startup", "--python-exit-code", "1", "-P", script, "--", *script_args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                              errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise BlenderFailed(f"Blender timed out after {timeout} s") from exc
    except OSError as exc:
        raise BlenderFailed(f"cannot start Blender: {exc}") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        raise BlenderFailed(f"Blender exited with code {proc.returncode}:\n{tail}")
    return proc


def convert_to_glb(src: str, dst: str, blender: str) -> str:
    """Convert src (FBX) to dst (GLB). Returns the Blender version string."""
    proc = run_script(blender, script_path("convert_to_glb.py"), [os.path.abspath(src), os.path.abspath(dst)])
    if not os.path.isfile(dst):
        raise BlenderFailed("Blender finished but wrote no GLB file")
    m = re.search(r"AICC_BLENDER_VERSION (\S+)", proc.stdout)
    return m.group(1) if m else "unknown"


def render(glb: str, out_dir: str, blender: str, hands: dict, clip: Optional[str], size: int = 512,
           frames: int = 6) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cfg = {"input": os.path.abspath(glb), "out_dir": os.path.abspath(out_dir), "size": size,
           "hands": hands, "clip": clip, "frames": frames}
    cfg_path = os.path.join(out_dir, "render_config.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    try:
        run_script(blender, script_path("render_character.py"), [os.path.abspath(cfg_path)])
    finally:
        try:
            os.remove(cfg_path)
        except OSError:
            pass
    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise BlenderFailed("Blender finished but wrote no manifest")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
