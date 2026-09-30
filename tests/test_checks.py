import pygltflib
import pytest
from conftest import ids

from ai_character_check.checks import run_checks
from ai_character_check.gltf_io import load_model


def _roles():
    import builder

    return [j[0] for j in builder.skeleton(builder.Spec())]


def _finding(rep, fid):
    return next(f for f in rep.findings if f.id == fid)


def test_clean_character_passes(check):
    rep = check(anim="inplace", texture=(64, 64))
    assert rep.verdict == "pass", [(f.id, f.title) for f in rep.findings if f.severity != "info"]
    assert rep.stats["humanoid_mapping"]["left_hand"] == "mixamorig:LeftHand"
    assert rep.stats["finger_segments"]["left"] == {"thumb": 3, "index": 3, "middle": 3, "ring": 3, "little": 3}
    assert rep.stats["toe_bones"] == {"left": 1, "right": 1}


def test_unreal_naming_is_mapped(check):
    rep = check(naming="unreal")
    assert rep.verdict == "pass"
    assert rep.stats["humanoid_mapping"]["right_upper_leg"] == "thigh_r"
    assert rep.stats["finger_segments"]["right"]["little"] == 3


def test_hand_without_fingers_is_an_error(check):
    rep = check(fingers=False, toes=False)
    assert {"skeleton.hand_no_fingers.left", "skeleton.hand_no_fingers.right"} <= ids(rep, "error")
    assert {"skeleton.toes_missing.left", "skeleton.toes_missing.right"} <= ids(rep, "warning")
    assert rep.verdict == "fail"


def test_toes_missing_is_info_in_preview_profile(check):
    rep = check(profile="preview", toes=False)
    assert "skeleton.toes_missing.left" in ids(rep, "info")


def test_single_segment_fingers(check):
    rep = check(finger_segments=1)
    assert "skeleton.finger_segments_low.left" in ids(rep, "warning")


def test_pose_t_and_a(check):
    t = check()
    a = check(arm_angle=45)
    down = check(arm_angle=80)
    assert t.stats["rest_pose"]["type"] == "T-pose"
    assert a.stats["rest_pose"] == {"type": "A-pose", "arm_angle_deg": -45.0}
    assert down.stats["rest_pose"]["type"] == "arms-down"
    assert "pose.rest_pose" in ids(down, "warning")


def test_finger_bones_without_weights(check):
    rep = check(finger_weights_to_hand=True)
    assert {"weights.fingers_unweighted.left", "weights.fingers_unweighted.right"} <= ids(rep, "error")


def test_claw_hand_heuristic(check):
    rep = check(claw=True)
    assert "weights.hand_finger_dominates.left" in ids(rep, "warning")
    assert _finding(rep, "weights.hand_finger_dominates.left").measured["share_pct"]["middle"] > 60


def test_left_right_leakage(check):
    rep = check(leak=True)
    f = _finding(rep, "weights.left_right_leakage")
    assert f.severity == "warning"
    assert f.measured["count"] > 0
    assert f.measured["top_pairs"][0]["vertex_owner"] == "mixamorig:LeftUpLeg"
    assert f.measured["top_pairs"][0]["leaks_to"] == "mixamorig:RightUpLeg"
    assert "weights.left_right_leakage" not in ids(check())


def test_weight_sum_and_influences(check):
    rep = check(extra_influences=True, unnormalized=True)
    assert "weights.not_normalized" in ids(rep, "warning")
    assert _finding(rep, "weights.too_many_influences").measured["max"] == 5  # 1 main + 4 extra influences
    prev = check(profile="preview", extra_influences=True)
    assert "weights.too_many_influences" not in ids(prev)


def test_duplicated_vertex_is_a_crack(check):
    f = _finding(check(crack=True), "mesh.unmerged_vertices")
    assert f.measured["count"] == 1
    assert f.measured["crack_edges"] >= 1


def test_uv_seam_is_not_a_crack(check):
    rep = check(seam=True)
    assert "mesh.unmerged_vertices" not in ids(rep)
    assert _finding(rep, "mesh.seams").measured["uv_seam"] == 1


def test_height_in_centimetres(check):
    rep = check(unit=100.0)
    f = _finding(rep, "scale.height_implausible")
    assert f.severity == "error"
    assert "centimetres" in f.explanation
    assert rep.stats["height_m"] == pytest.approx(180, abs=2)


def test_scaled_armature(check):
    rep = check(unit=100.0, armature_scale=0.01)
    assert rep.stats["height_m"] == pytest.approx(1.8, abs=0.02)
    assert "scale.node_scale" in ids(rep, "warning")
    assert "scale.height_implausible" not in ids(rep)


def test_z_up_and_floating(check):
    assert "scale.up_axis" in ids(check(z_up=True), "warning")
    f = _finding(check(y_offset=0.5), "scale.feet_offset")
    assert f.measured["lowest_point_m"] == pytest.approx(0.49, abs=0.03)


def test_scale_keys(check):
    f = _finding(check(anim="scale"), "animation.scale_keys")
    assert f.severity == "error"
    assert "mixamorig:LeftArm" in f.measured["bones"]
    prev = check(profile="preview", anim="scale")
    assert _finding(prev, "animation.scale_keys").severity == "warning"


def test_nan_keys(check):
    assert "animation.nan_keys" in ids(check(anim="nan"), "error")


def test_root_motion_and_length(check):
    walk = check(anim="walk").stats["animations"][0]
    still = check(anim="inplace").stats["animations"][0]
    assert walk["root_motion"] is True
    assert walk["root_displacement_m"] == pytest.approx(1.2, abs=1e-3)
    assert walk["length_s"] == pytest.approx(1.0)
    assert still["root_motion"] is False


def test_textures(check):
    assert "textures.non_power_of_two" in ids(check(texture=(100, 50)), "warning")
    assert "textures.missing" in ids(check(missing_texture=True), "error")
    ok = check(texture=(256, 128))
    assert ok.stats["textures"]["max_resolution"] == 256
    assert "textures.non_power_of_two" not in ids(ok)


def test_unknown_bone_names(make):
    path = make(names_override={r: f"Bone.{i:03d}" for i, r in enumerate(_roles())})
    rep = run_checks(load_model(path))
    assert "skeleton.not_humanoid" in ids(rep, "error")


def test_vrm_extension_mapping(make, tmp_path):
    roles = _roles()
    path = make(names_override={r: f"Bone.{i:03d}" for i, r in enumerate(roles)})
    g = pygltflib.GLTF2().load(path)
    vrm_names = {"hips": "hips", "spine1": "spine", "neck": "neck", "head": "head"}
    for side in ("left", "right"):
        for r, v in (("upper_arm", "UpperArm"), ("lower_arm", "LowerArm"), ("hand", "Hand"),
                     ("upper_leg", "UpperLeg"), ("lower_leg", "LowerLeg"), ("foot", "Foot"), ("toe", "Toes")):
            vrm_names[f"{side}_{r}"] = f"{side}{v}"
        for f, v in (("thumb", "Thumb"), ("index", "Index"), ("middle", "Middle"), ("ring", "Ring"),
                     ("little", "Little")):
            seg = ["Metacarpal", "Proximal", "Distal"] if f == "thumb" else ["Proximal", "Intermediate", "Distal"]
            for k in range(3):
                vrm_names[f"{side}_{f}_{k + 1}"] = f"{side}{v}{seg[k]}"
    g.extensions = {"VRMC_vrm": {"specVersion": "1.0", "humanoid": {"humanBones": {
        vrm: {"node": roles.index(role)} for role, vrm in vrm_names.items()}}}}
    out = str(tmp_path / "vrm.glb")
    g.save_binary(out)
    rep = run_checks(load_model(out))
    assert rep.stats["humanoid_mapping_source"] == "vrm-extension"
    assert rep.stats["humanoid_mapping"]["left_hand"] == f"Bone.{roles.index('left_hand'):03d}"
    assert rep.stats["finger_segments"]["right"]["thumb"] == 3
    assert rep.verdict == "pass", [(f.id, f.title) for f in rep.findings if f.severity != "info"]
