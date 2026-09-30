"""Tests that need a local Blender. Skipped when Blender is not found."""

import os
import subprocess

import pytest

from ai_character_check.blender import find_blender
from ai_character_check.cli import main

BLENDER = find_blender()
pytestmark = pytest.mark.skipif(BLENDER is None, reason="Blender not found (set BLENDER or put it on PATH)")


def test_render_fixture(make, tmp_path, capsys):
    out = tmp_path / "render"
    assert main(["render", make(anim="walk", arm_angle=40), "--out", str(out), "--size", "128"]) == 0
    for name in ("front.png", "hand_left.png", "hand_right.png", "anim_strip.png"):
        p = out / name
        assert p.is_file() and p.stat().st_size > 1000, name
    with open(out / "anim_strip.png", "rb") as fh:
        head = fh.read(24)
    assert int.from_bytes(head[16:20], "big") == 6 * 128


def test_fbx_input_is_converted(make, tmp_path, capsys):
    glb = make(anim="walk", fingers=False)
    fbx = str(tmp_path / "char.fbx")
    expr = ("import bpy,sys; bpy.ops.wm.read_factory_settings(use_empty=True); "
            "bpy.ops.import_scene.gltf(filepath=sys.argv[-2]); "
            "bpy.ops.export_scene.fbx(filepath=sys.argv[-1], add_leaf_bones=False)")
    subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", expr,
                    "--", glb, fbx], check=True, capture_output=True, timeout=300)
    assert os.path.isfile(fbx)
    assert main(["check", fbx, "--format", "json"]) == 1
    out = capsys.readouterr().out
    assert "converted from FBX" in out
    assert "skeleton.hand_no_fingers.left" in out
