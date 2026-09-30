import pytest

from ai_character_check.humanoid import classify


@pytest.mark.parametrize("name,side,role,finger,segment", [
    ("mixamorig:LeftHandIndex1", "left", "finger", "index", 1),
    ("mixamorig:RightHandPinky3", "right", "finger", "little", 3),
    ("mixamorig:LeftArm", "left", "upper_arm", None, None),
    ("mixamorig:LeftForeArm", "left", "lower_arm", None, None),
    ("mixamorig:RightUpLeg", "right", "upper_leg", None, None),
    ("mixamorig:RightLeg", "right", "lower_leg", None, None),
    ("mixamorig:LeftToeBase", "left", "toe", None, None),
    ("mixamorig:Hips", None, "hips", None, None),
    ("thumb_01_l", "left", "finger", "thumb", 1),
    ("index_metacarpal_r", "right", "finger", "index", 0),
    ("upperarm_l", "left", "upper_arm", None, None),
    ("lowerarm_r", "right", "lower_arm", None, None),
    ("calf_l", "left", "lower_leg", None, None),
    ("ball_r", "right", "toe", None, None),
    ("pelvis", None, "hips", None, None),
    ("leftIndexProximal", "left", "finger", "index", 1),
    ("rightThumbDistal", "right", "finger", "thumb", 3),
    ("J_Bip_L_UpperArm", "left", "upper_arm", None, None),
    ("J_Bip_R_Little2", "right", "finger", "little", 2),
    ("J_Bip_C_Hips", None, "hips", None, None),
    ("DEF-f_index.02.L", "left", "finger", "index", 2),
    ("DEF-thumb.01.R", "right", "finger", "thumb", 1),
    ("DEF-upper_arm.L", "left", "upper_arm", None, None),
    ("DEF-shin.R", "right", "lower_leg", None, None),
    ("Bip01 L Finger0", "left", "finger", "thumb", 1),
    ("Bip01 R Finger12", "right", "finger", "index", 3),
    ("Bip01 L UpperArm", "left", "upper_arm", None, None),
    ("CC_Base_L_Mid2", "left", "finger", "middle", 2),
    ("CC_Base_R_Forearm", "right", "lower_arm", None, None),
    ("Left Ring Intermediate", "left", "finger", "ring", 2),
])
def test_classify(name, side, role, finger, segment):
    b = classify(0, name)
    assert (b.side, b.role, b.finger) == (side, role, finger)
    if segment is not None:
        assert b.segment == segment


@pytest.mark.parametrize("name", ["ik_hand_l", "upperarm_twist_01_l", "MCH-hand_ik.L", "pole_target_r"])
def test_helper_bones_get_no_role(name):
    b = classify(0, name)
    assert b.role is None


def test_end_bones_are_tips():
    assert classify(0, "mixamorig:HeadTop_End").tip
    assert classify(0, "Bip01 L Finger0Nub").tip


@pytest.mark.parametrize("name", ["index_01_palm_l", "index_metacarpal_slide_r", "thumb_02_in_l",
                                  "middle_02_side_inn_l", "ring_01_bulge_r", "pinky_02_pip_l"])
def test_unreal5_finger_helpers_are_not_segments(name):
    b = classify(0, name)
    assert b.role is None and b.helper
