"""Blender script: convert an FBX (or other importable file) to GLB.

Run by ai-character-check as:
    blender -b --factory-startup --python-exit-code 1 -P convert_to_glb.py -- INPUT OUTPUT.glb
"""

import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
src, dst = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
ext = os.path.splitext(src)[1].lower()
if ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=src)
elif ext in (".gltf", ".glb"):
    bpy.ops.import_scene.gltf(filepath=src)
else:
    raise SystemExit(f"unsupported input format: {ext}")
bpy.ops.export_scene.gltf(
    filepath=dst,
    export_format="GLB",
    export_skins=True,
    export_animations=True,
    export_yup=True,
    export_apply=False,
)
if not os.path.exists(dst):
    raise SystemExit("glTF export produced no file")
print("AICC_BLENDER_VERSION", bpy.app.version_string)
