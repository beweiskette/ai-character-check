"""Blender script: render a character for visual review.

Run by ai-character-check as:
    blender -b --factory-startup --python-exit-code 1 -P render_character.py -- CONFIG.json

CONFIG.json keys: input (glb path), out_dir, size, hands ({"left": bone, "right": bone}),
clip (animation name or null), frames (int).
Writes PNG files and manifest.json into out_dir.
"""

import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    with open(argv[0], "r", encoding="utf-8") as fh:
        return json.load(fh)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_render(size):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "TEXTURE"
    sc.display.shading.show_cavity = False
    sc.render.resolution_x = size
    sc.render.resolution_y = size
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.new("aicc_world")
    world.color = (0.82, 0.83, 0.85)
    sc.world = world
    cam_data = bpy.data.cameras.new("aicc_cam")
    cam_data.type = "ORTHO"
    cam = bpy.data.objects.new("aicc_cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def mesh_objects():
    # The glTF importer adds a bone display shape (an icosphere) as a mesh object;
    # exclude custom bone shapes from framing and rendering.
    shapes = set()
    for o in bpy.context.scene.objects:
        if o.type == "ARMATURE" and o.pose:
            shapes |= {pb.custom_shape for pb in o.pose.bones if pb.custom_shape is not None}
    for s in shapes:
        s.hide_render = True
    return [o for o in bpy.context.scene.objects if o.type == "MESH" and o not in shapes]


def armature_object():
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    return arms[0] if arms else None


def world_points(objs, max_points=200000):
    deps = bpy.context.evaluated_depsgraph_get()
    chunks = []
    for o in objs:
        eo = o.evaluated_get(deps)
        me = eo.to_mesh()
        n = len(me.vertices)
        if n:
            co = np.empty(n * 3, dtype=np.float64)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            mw = np.array(eo.matrix_world)
            co = co @ mw[:3, :3].T + mw[:3, 3]
            chunks.append(co)
        eo.to_mesh_clear()
    if not chunks:
        return np.zeros((0, 3))
    pts = np.concatenate(chunks)
    if len(pts) > max_points:
        pts = pts[:: len(pts) // max_points + 1]
    return pts


def aim(cam, target, direction, extent, pad=1.15):
    """Place an orthographic camera looking at target from direction."""
    direction = Vector(direction).normalized()
    target = Vector(target)
    cam.location = target + direction * max(extent * 4.0, 1.0)
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.ortho_scale = max(extent * pad, 1e-3)
    cam.data.clip_start = 0.001
    cam.data.clip_end = max(extent * 20.0, 100.0)


def frame_points(cam, pts, direction):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    extent = float(np.max(hi - lo))
    aim(cam, center, direction, extent)


def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return os.path.exists(path)


def hand_points(arm, bone_name, meshes):
    bone = arm.data.bones.get(bone_name)
    if bone is None:
        return None
    names = {bone.name} | {b.name for b in bone.children_recursive}
    deps = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in meshes:
        idx = {vg.index for vg in o.vertex_groups if vg.name in names}
        if not idx:
            continue
        eo = o.evaluated_get(deps)
        me = eo.to_mesh()
        mw = eo.matrix_world
        orig = o.data
        for v in orig.vertices:
            if any(g.group in idx and g.weight > 0.2 for g in v.groups):
                if v.index < len(me.vertices):
                    pts.append(tuple(mw @ me.vertices[v.index].co))
        eo.to_mesh_clear()
    if len(pts) < 4:
        mw = arm.matrix_world
        for n in names:
            b = arm.data.bones[n]
            pts.append(tuple(mw @ b.head_local))
            pts.append(tuple(mw @ b.tail_local))
    return np.array(pts)


def pick_action(clip):
    actions = list(bpy.data.actions)
    if not actions:
        return None
    if clip:
        for a in actions:
            if a.name == clip:
                return a
        for a in actions:
            if a.name.startswith(clip):
                return a
    return actions[0]


def assign_action(arm, action):
    if arm.animation_data is None:
        arm.animation_data_create()
    ad = arm.animation_data
    # Mute NLA tracks the importer may have created so only this clip plays.
    for tr in ad.nla_tracks:
        tr.mute = True
    ad.action = action
    if hasattr(ad, "action_slot") and ad.action_slot is None and hasattr(action, "slots") and len(action.slots):
        ad.action_slot = action.slots[0]


def combine_strip(paths, out_path):
    imgs = []
    for p in paths:
        im = bpy.data.images.load(p)
        w, h = im.size
        px = np.empty(w * h * 4, dtype=np.float32)
        im.pixels.foreach_get(px)
        imgs.append(px.reshape(h, w, 4))
    strip = np.concatenate(imgs, axis=1)
    h, w = strip.shape[:2]
    out = bpy.data.images.new("aicc_strip", width=w, height=h, alpha=True)
    out.pixels.foreach_set(strip.ravel())
    out.filepath_raw = out_path
    out.file_format = "PNG"
    out.save()


def main():
    cfg = args()
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=cfg["input"])
    cam = setup_render(int(cfg.get("size", 512)))
    meshes = mesh_objects()
    arm = armature_object()
    manifest = {"outputs": {}, "notes": [], "blender_version": bpy.app.version_string}
    if not meshes:
        manifest["notes"].append("no mesh objects after import")
        write_manifest(out_dir, manifest)
        return

    # Rest pose for body and hands, with any active animation disabled.
    if arm is not None:
        arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    front = (0.0, -1.0, 0.0)  # glTF +Z forward becomes Blender -Y
    pts = world_points(meshes)
    frame_points(cam, pts, front)
    path = os.path.join(out_dir, "front.png")
    if render_to(path):
        manifest["outputs"]["front"] = "front.png"

    hands = cfg.get("hands") or {}
    for side in ("left", "right"):
        bone = hands.get(side)
        if not bone or arm is None:
            manifest["notes"].append(f"{side} hand: no hand bone known, skipped")
            continue
        hp = hand_points(arm, bone, meshes)
        if hp is None or not len(hp):
            manifest["notes"].append(f"{side} hand: bone {bone} not found in Blender, skipped")
            continue
        lo, hi = hp.min(axis=0), hp.max(axis=0)
        center = (lo + hi) / 2
        extent = float(np.max(hi - lo)) * 1.3
        aim(cam, center, (0.0, -0.6, 1.0), extent)
        name = f"hand_{side}.png"
        if render_to(os.path.join(out_dir, name)):
            manifest["outputs"][f"hand_{side}"] = name

    frames = int(cfg.get("frames", 6))
    action = pick_action(cfg.get("clip")) if arm is not None else None
    if action is None:
        manifest["notes"].append("no animation clip found, strip skipped")
    else:
        arm.data.pose_position = "POSE"
        assign_action(arm, action)
        start, end = action.frame_range
        sc = bpy.context.scene
        times = [start + (end - start) * i / max(frames - 1, 1) for i in range(frames)]
        all_pts = []
        for t in times:
            sc.frame_set(int(math.floor(t)), subframe=t - math.floor(t))
            all_pts.append(world_points(meshes, max_points=20000))
        frame_points(cam, np.concatenate(all_pts), front)
        paths = []
        for i, t in enumerate(times):
            sc.frame_set(int(math.floor(t)), subframe=t - math.floor(t))
            p = os.path.join(out_dir, f"anim_{i:02d}.png")
            if render_to(p):
                paths.append(p)
        if paths:
            combine_strip(paths, os.path.join(out_dir, "anim_strip.png"))
            manifest["outputs"]["anim_strip"] = "anim_strip.png"
            manifest["outputs"]["anim_frames"] = [os.path.basename(p) for p in paths]
            manifest["clip"] = action.name
            manifest["frame_times"] = [round(t, 3) for t in times]
    write_manifest(out_dir, manifest)


def write_manifest(out_dir, manifest):
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


main()
