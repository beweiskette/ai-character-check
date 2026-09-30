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


TEXTURE_NAME = "outside_texture_0123456789abcdef.png"
NET_HOST = "aicc-test.invalid"  # reserved TLD, never resolves


def _fbx_with_outside_texture(make, tmp_path):
    """FBX in tmp/model/ whose material references tmp/<TEXTURE_NAME> (37 x 23 px) by path."""
    import builder

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tex = tmp_path / TEXTURE_NAME
    tex.write_bytes(builder.png_bytes(37, 23))
    glb = make(fingers=False)
    fbx = str(model_dir / "char.fbx")
    expr = ("import bpy,sys; bpy.ops.wm.read_factory_settings(use_empty=True); "
            "bpy.ops.import_scene.gltf(filepath=sys.argv[-3]); "
            "img = bpy.data.images.load(sys.argv[-1]); "
            "mat = bpy.data.materials.new('Skin'); mat.use_nodes = True; "
            "n = mat.node_tree.nodes.new('ShaderNodeTexImage'); n.image = img; "
            "mat.node_tree.links.new(n.outputs['Color'], mat.node_tree.nodes['Principled BSDF'].inputs['Base Color']); "
            "[o.data.materials.append(mat) for o in bpy.data.objects if o.type == 'MESH']; "
            "bpy.ops.export_scene.fbx(filepath=sys.argv[-2], add_leaf_bones=False, path_mode='ABSOLUTE', "
            "embed_textures=False)")
    subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", expr,
                    "--", glb, fbx, str(tex)], check=True, capture_output=True, timeout=300)
    return fbx


def _texture_sizes(report: dict) -> int:
    return next(f for f in report["findings"] if f["id"] == "textures.summary")["measured"]["max_resolution"]


def test_fbx_texture_outside_folder_is_not_read(make, tmp_path, capsys):
    import json

    fbx = _fbx_with_outside_texture(make, tmp_path)
    main(["check", fbx, "--format", "json"])
    d = json.loads(capsys.readouterr().out)
    f = next(f for f in d["findings"] if f["id"] == "resource.external_path_blocked")
    assert f["severity"] == "error"
    assert f["measured"]["items"][0]["kind"] == "texture"
    assert TEXTURE_NAME in f["measured"]["items"][0]["uri"]
    assert _texture_sizes(d) != 37  # the outside PNG did not end up in the converted model

    # Opt-in: the local texture outside the folder is read.
    main(["check", fbx, "--format", "json", "--allow-external-resources"])
    d = json.loads(capsys.readouterr().out)
    assert "resource.external_path_blocked" not in {f["id"] for f in d["findings"]}
    assert _texture_sizes(d) == 37


def test_fbx_texture_on_network_path_is_blocked(make, tmp_path, capsys):
    import json
    import re
    import struct

    fbx = _fbx_with_outside_texture(make, tmp_path)
    data = bytearray(open(fbx, "rb").read())
    # Rewrite every FBX string property that names the texture into a UNC path of equal length.
    replaced = 0
    for occ in list(re.finditer(re.escape(TEXTURE_NAME.encode()), bytes(data))):
        end = occ.end()
        for s in range(occ.start(), max(5, occ.start() - 600), -1):
            if data[s - 5:s - 4] == b"S" and struct.unpack("<I", data[s - 4:s])[0] == end - s:
                n = end - s
                unc = ("//" + NET_HOST + "/s/").replace("/", chr(92)).encode()
                if n >= len(unc) + 5:
                    data[s:end] = unc + b"x" * (n - len(unc) - 4) + b".png"
                    replaced += 1
                break
    assert replaced >= 1
    # Blender strips leading separators from RelativeFilename, which defuses a UNC path there.
    # Rename that property so the importer falls back to the absolute FileName (now UNC).
    data = bytearray(bytes(data).replace(b"\x10RelativeFilename", b"\x10XelativeFilename"))
    open(fbx, "wb").write(bytes(data))
    for extra in ([], ["--allow-external-resources"]):
        main(["check", fbx, "--format", "json", *extra])
        d = json.loads(capsys.readouterr().out)
        f = next(f for f in d["findings"] if f["id"] == "resource.external_path_blocked")
        assert any(NET_HOST in i["uri"] and "network" in i["reason"] for i in f["measured"]["items"])
