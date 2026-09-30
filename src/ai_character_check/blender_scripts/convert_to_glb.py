"""Blender script: convert an FBX to GLB.

Run by ai-character-check as:
    blender -b --factory-startup --python-exit-code 1 -P convert_to_glb.py -- INPUT.fbx OUTPUT.glb MODE

MODE is "confine" (default) or "allow-external". FBX files name their textures
by path. Blender's importer would open any such path, including parent
folders, absolute paths and network (UNC) paths, and on Windows probing a UNC
path can send the user's NTLM credentials to a remote server. This script
routes every texture path through safe_paths first: only files inside the FBX's
folder are opened (in "allow-external" mode also other local paths). A refused
texture becomes an empty placeholder; embedded texture data still loads.
Refused paths are printed as "AICC_BLOCKED {json}" lines.
"""

import importlib.util
import json
import os
import re
import sys

import bpy
from bpy_extras import image_utils

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
src, dst = argv[0], argv[1]
allow_external = len(argv) > 2 and argv[2] == "allow-external"
base_dir = os.path.dirname(os.path.abspath(src))

_spec = importlib.util.spec_from_file_location(
    "aicc_safe_paths", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "safe_paths.py"))
safe_paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(safe_paths)

_original_load_image = image_utils.load_image
_blocked = []  # (entry, image)


def _placeholder(text):
    name = os.path.basename(text.replace("\\", "/"))
    name = re.sub(r"[^\w.\- ]", "_", name).strip(". ") or "texture"
    image = bpy.data.images.new(name, 128, 128)
    # Point the placeholder at a file inside the model folder, never at the refused path.
    image.filepath = os.path.join(base_dir, name)
    image.source = "FILE"
    return image


def guarded_load_image(imagepath, dirname="", place_holder=False, recursive=False, ncase_cmp=True,
                       convert_callback=None, verbose=False, relpath=None, check_existing=False,
                       force_reload=False):
    text = os.fsdecode(imagepath)
    kwargs = dict(dirname=base_dir, place_holder=place_holder, recursive=False, ncase_cmp=ncase_cmp,
                  convert_callback=convert_callback, verbose=verbose, relpath=relpath,
                  check_existing=check_existing, force_reload=force_reload)
    path, reason = safe_paths.check_path(text, base_dir, allow_external)
    if path is not None:
        return _original_load_image(path, **kwargs)
    # A file with the same name next to the model is a safe and common fallback
    # (FBX files often carry the absolute path from the author's machine).
    name = os.path.basename(text.replace("\\", "/"))
    alt = safe_paths.check_path(name, base_dir, False)[0] if name else None
    if alt is not None and os.path.isfile(alt):
        return _original_load_image(alt, **kwargs)
    image = _placeholder(text)
    _blocked.append(({"kind": "texture", "uri": text if len(text) <= 200 else text[:197] + "...",
                      "reason": reason}, image))
    return image


image_utils.load_image = guarded_load_image

bpy.ops.wm.read_factory_settings(use_empty=True)
ext = os.path.splitext(src)[1].lower()
if ext != ".fbx":
    raise SystemExit(f"unsupported input format: {ext}")
bpy.ops.import_scene.fbx(filepath=src, use_image_search=False)
image_utils.load_image = _original_load_image

index = 0
for entry, image in _blocked:
    if image.packed_file is not None:
        continue  # the FBX embeds the texture, nothing is lost
    entry["index"] = index
    index += 1
    print("AICC_BLOCKED", json.dumps(entry))

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
