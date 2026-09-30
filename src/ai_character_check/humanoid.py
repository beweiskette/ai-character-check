"""Map bone names of common rigging conventions to humanoid roles.

Supported naming schemes (by name pattern, not by exact lists):
Mixamo (``mixamorig:LeftHandIndex1``), Unreal (``hand_l``, ``index_01_l``),
VRM / Unity humanoid (``leftIndexProximal``, ``J_Bip_L_Index1``), Blender
rigify-like (``f_index.01.L``, ``upper_arm.L``), 3ds Max biped
(``Bip01 L Finger1``) and Character Creator (``CC_Base_L_Index1``).
If the file carries a VRM humanoid extension, that mapping wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

FINGERS = ["thumb", "index", "middle", "ring", "little"]
FINGER_TOKENS = {
    "thumb": "thumb", "index": "index", "middle": "middle", "mid": "middle",
    "ring": "ring", "pinky": "little", "pinkie": "little", "little": "little",
}
SIDES = ("left", "right")

NOISE_TOKENS = {
    "mixamorig", "mixamorig1", "mixamorig2", "mixamo", "def", "org", "cc", "base", "j", "bip",
    "bip01", "bip001", "c", "armature", "jnt", "joint", "bn", "b", "rig", "f", "sk", "skel",
}
# Bones with these tokens are helpers and never get a humanoid role.
HELPER_TOKENS = {"ik", "pole", "target", "ctrl", "ctl", "control", "mch", "twist", "roll", "helper",
                 "corrective", "correctiveroot", "vb", "prop", "weapon", "socket", "attach", "share",
                 # Unreal 5 corrective and deformation helpers
                 "half", "side", "pip", "dip", "mcp", "bulge", "inn", "out", "fwd", "bck", "drv", "driver"}
FINGER_WORDS_OK = {"hand", "finger", "fingers", "metacarpal", "proximal", "intermediate", "distal", "phalanx"}
TIP_TOKENS = {"end", "tip", "nub", "top", "tail", "null"}

CENTER_ROLES = {
    "hips": {"hips", "hip", "pelvis"},
    "spine": {"spine", "chest", "upperchest", "torso", "waist", "abdomen", "belly"},
    "neck": {"neck"},
    "head": {"head"},
}
LIMB_ROLES = {
    "shoulder": {"shoulder", "clavicle", "collar", "collarbone"},
    "upper_arm": {"arm", "upperarm", "uparm", "bicep"},
    "lower_arm": {"forearm", "lowerarm", "elbow", "lowarm"},
    "hand": {"hand", "wrist"},
    "upper_leg": {"upleg", "upperleg", "thigh", "legupper"},
    "lower_leg": {"leg", "lowerleg", "calf", "shin", "knee", "lowleg"},
    "foot": {"foot", "ankle"},
}

VRM_ROLE = {
    "hips": "hips", "spine": "spine", "chest": "spine", "upperChest": "spine", "neck": "neck",
    "head": "head",
}
for _s in SIDES:
    VRM_ROLE.update({
        f"{_s}Shoulder": f"{_s}_shoulder", f"{_s}UpperArm": f"{_s}_upper_arm",
        f"{_s}LowerArm": f"{_s}_lower_arm", f"{_s}Hand": f"{_s}_hand",
        f"{_s}UpperLeg": f"{_s}_upper_leg", f"{_s}LowerLeg": f"{_s}_lower_leg",
        f"{_s}Foot": f"{_s}_foot", f"{_s}Toes": f"{_s}_toe",
    })


@dataclass
class BoneInfo:
    node: int
    name: str
    side: Optional[str]  # "left" / "right" / None
    role: Optional[str]  # e.g. "hand", "finger", "toe", "hips"
    finger: Optional[str] = None
    segment: Optional[int] = None
    tip: bool = False
    helper: bool = False


@dataclass
class HumanoidMap:
    bones: dict[int, BoneInfo] = field(default_factory=dict)
    roles: dict[str, int] = field(default_factory=dict)  # "left_hand" -> node
    fingers: dict[tuple[str, str], list[int]] = field(default_factory=dict)  # (side, finger) -> nodes
    toes: dict[str, list[int]] = field(default_factory=dict)  # side -> nodes
    spine: list[int] = field(default_factory=list)
    source: str = "names"

    def role(self, name: str) -> Optional[int]:
        return self.roles.get(name)

    def finger_bones(self, side: str) -> list[int]:
        out: list[int] = []
        for f in FINGERS:
            out.extend(self.fingers.get((side, f), []))
        return out

    def side_of(self, node: int) -> Optional[str]:
        b = self.bones.get(node)
        return b.side if b else None


def tokenize(name: str) -> list[str]:
    s = re.split(r"[:|]", name)[-1]
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def classify(node: int, name: str) -> BoneInfo:
    tokens = tokenize(name)
    side = None
    rest: list[str] = []
    for t in tokens:
        if t in ("left", "l", "lft"):
            side = side or "left"
        elif t in ("right", "r", "rgt", "rt"):
            side = side or "right"
        elif t in NOISE_TOKENS:
            continue
        else:
            rest.append(t)
    info = BoneInfo(node, name, side, None)
    if any(t in HELPER_TOKENS for t in rest):
        info.helper = True
        return info
    info.tip = any(t in TIP_TOKENS for t in rest)
    words = [t for t in rest if not t.isdigit() and t not in TIP_TOKENS]
    numbers = [t for t in rest if t.isdigit()]
    key = "".join(words)

    finger = next((FINGER_TOKENS[w] for w in words if w in FINGER_TOKENS), None)
    if finger is None and words and words[-1] == "finger" and numbers:
        # 3ds Max biped: Finger0 = thumb ... Finger4 = little, Finger01 = thumb segment 2.
        digits = numbers[-1]
        idx = int(digits[0])
        if 0 <= idx <= 4:
            finger = FINGERS[idx]
            info.segment = int(digits[1]) + 1 if len(digits) > 1 else 1
    if finger is None and "finger" in words and len(words) >= 2:
        for w in words:
            if w in FINGER_TOKENS:
                finger = FINGER_TOKENS[w]
    if finger is not None:
        # Any extra word (palm, in, slide, half, ...) marks a helper next to the real chain.
        extras = [w for w in words if w not in FINGER_TOKENS and w not in FINGER_WORDS_OK]
        if extras:
            info.helper = True
            return info
        info.role = "finger"
        info.finger = finger
        if info.segment is None:
            if "metacarpal" in words:
                info.segment = 0
            elif "proximal" in words:
                info.segment = 1
            elif "intermediate" in words:
                info.segment = 2
            elif "distal" in words:
                info.segment = 3
            elif numbers:
                info.segment = int(numbers[-1])
        return info

    if "toe" in key or key in ("ball", "toes", "toebase"):
        info.role = "toe"
        return info
    if side is None:
        for role, keys in CENTER_ROLES.items():
            if key in keys:
                info.role = role
                return info
        if key.startswith("spine") or key.startswith("chest"):
            info.role = "spine"
            return info
    for role, keys in LIMB_ROLES.items():
        if key in keys:
            info.role = role
            return info
    if side is None and key in ("hip",):
        info.role = "hips"
    return info


def _vrm_mapping(gltf) -> dict[str, int]:
    ext = gltf.extensions or {}
    out: dict[str, int] = {}
    vrm1 = ext.get("VRMC_vrm")
    if isinstance(vrm1, dict):
        hb = (vrm1.get("humanoid") or {}).get("humanBones") or {}
        for bone, entry in hb.items():
            if isinstance(entry, dict) and isinstance(entry.get("node"), int):
                out[bone] = entry["node"]
    vrm0 = ext.get("VRM")
    if isinstance(vrm0, dict) and not out:
        for entry in (vrm0.get("humanoid") or {}).get("humanBones") or []:
            if isinstance(entry, dict) and isinstance(entry.get("node"), int) and entry.get("bone"):
                out[entry["bone"]] = entry["node"]
    return out


def build_map(model, nodes: Optional[list[int]] = None) -> HumanoidMap:
    """Classify the given nodes (default: all skin joints) into humanoid roles."""
    if nodes is None:
        nodes = model.joint_nodes
    hm = HumanoidMap()
    for n in nodes:
        hm.bones[n] = classify(n, model.node_names[n])

    # VRM humanoid extension overrides name guessing for the bones it lists.
    vrm = _vrm_mapping(model.gltf)
    if vrm:
        hm.source = "vrm-extension"
        for bone, node in vrm.items():
            if node not in hm.bones:
                continue
            info = hm.bones[node]
            if bone in VRM_ROLE:
                role = VRM_ROLE[bone]
                if "_" in role and role.split("_", 1)[0] in SIDES:
                    info.side, info.role = role.split("_", 1)
                else:
                    info.role = role
            else:
                m = re.match(r"(left|right)(Thumb|Index|Middle|Ring|Little)(Metacarpal|Proximal|Intermediate|Distal)", bone)
                if m:
                    info.side, info.role, info.finger = m.group(1), "finger", m.group(2).lower()
                    info.segment = {"Metacarpal": 0, "Proximal": 1, "Intermediate": 2, "Distal": 3}[m.group(3)]
                    info.tip = False

    # Order by hierarchy depth so the first match of a role is the main bone.
    ordered = sorted(nodes, key=lambda n: (model.depth(n), n))
    for n in ordered:
        b = hm.bones[n]
        if b.helper or b.role is None:
            continue
        if b.tip and b.role not in ("finger", "toe"):
            continue
        if b.role == "finger":
            if b.side is None:
                continue
            hm.fingers.setdefault((b.side, b.finger), []).append(n)
        elif b.role == "toe":
            if b.side is None:
                continue
            hm.toes.setdefault(b.side, []).append(n)
        elif b.role == "spine":
            hm.spine.append(n)
            hm.roles.setdefault("spine", n)
        elif b.side is not None:
            hm.roles.setdefault(f"{b.side}_{b.role}", n)
        elif b.role in CENTER_ROLES:
            hm.roles.setdefault(b.role, n)

    # Mark leaf bones past the third segment as tips (Mixamo "Index4" end bones).
    for key, chain in hm.fingers.items():
        for n in chain:
            b = hm.bones[n]
            is_leaf = not any(c in hm.bones for c in model.children[n])
            if is_leaf and (b.tip or (b.segment is not None and b.segment >= 4)):
                b.tip = True
            elif not is_leaf and b.tip and hm.source != "vrm-extension":
                # A bone with children is not an end bone even if named "top".
                b.tip = False
        chain.sort(key=lambda n: (model.depth(n), n))
    for side, chain in hm.toes.items():
        chain.sort(key=lambda n: (model.depth(n), n))
    hm.spine.sort(key=lambda n: (model.depth(n), n))
    return hm


def finger_segments(hm: HumanoidMap, side: str, finger: str) -> list[int]:
    """Deforming segments of one finger (end/tip bones excluded)."""
    return [n for n in hm.fingers.get((side, finger), []) if not hm.bones[n].tip]


def toe_segments(hm: HumanoidMap, side: str) -> list[int]:
    return [n for n in hm.toes.get(side, []) if not hm.bones[n].tip]
