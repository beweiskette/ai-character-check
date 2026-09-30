"""Build tiny synthetic skinned humanoids as GLB files for the tests.

The character is made of one box per bone segment (8 vertices each, weighted
100 % to its bone). It is 1.8 m tall, faces +Z, stands on y = 0, and its
left side is +X, following the glTF conventions.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pygltflib

FINGERS = ["thumb", "index", "middle", "ring", "little"]
MIXAMO_FINGER = {"thumb": "Thumb", "index": "Index", "middle": "Middle", "ring": "Ring", "little": "Pinky"}
UNREAL_FINGER = {"thumb": "thumb", "index": "index", "middle": "middle", "ring": "ring", "little": "pinky"}


@dataclass
class Spec:
    naming: str = "mixamo"  # "mixamo" | "unreal"
    fingers: bool = True
    finger_segments: int = 3
    toes: bool = True
    arm_angle: float = 0.0  # degrees below horizontal
    finger_weights_to_hand: bool = False
    claw: bool = False  # all finger geometry bound to the middle finger chain
    leak: bool = False  # left thigh vertices partly bound to the right thigh
    crack: bool = False  # one vertex duplicated with identical attributes
    seam: bool = False  # one vertex duplicated with a different UV
    unit: float = 1.0  # multiply all coordinates (100 = centimetres)
    armature_scale: Optional[float] = None  # parent all joints under a scaled node
    z_up: bool = False
    y_offset: float = 0.0
    anim: Optional[str] = None  # None | "walk" | "scale" | "nan" | "inplace"
    sparse_positions: bool = False
    normalized_weights: bool = False
    split_primitives: bool = False
    extra_influences: bool = False
    unnormalized: bool = False
    texture: Optional[tuple] = None  # (width, height) of an embedded PNG
    missing_texture: bool = False
    names_override: dict = field(default_factory=dict)


def _joint_name(spec: Spec, role: str) -> str:
    if role in spec.names_override:
        return spec.names_override[role]
    parts = role.split("_")
    side = parts[0] if parts[0] in ("left", "right") else None
    if spec.naming == "unreal":
        s = {"left": "l", "right": "r"}.get(side)
        center = {"hips": "pelvis", "spine1": "spine_01", "spine2": "spine_02", "spine3": "spine_03",
                  "neck": "neck_01", "head": "head", "head_end": "head_end"}
        if side is None:
            return center[role]
        rest = "_".join(parts[1:])
        limb = {"shoulder": "clavicle", "upper_arm": "upperarm", "lower_arm": "lowerarm", "hand": "hand",
                "upper_leg": "thigh", "lower_leg": "calf", "foot": "foot", "toe": "ball", "toe_end": "ball_end"}
        if rest in limb:
            return f"{limb[rest]}_{s}"
        finger, k = parts[1], int(parts[2])
        if k == 4:
            return f"{UNREAL_FINGER[finger]}_tip_{s}"
        return f"{UNREAL_FINGER[finger]}_0{k}_{s}"
    # Mixamo
    center = {"hips": "Hips", "spine1": "Spine", "spine2": "Spine1", "spine3": "Spine2", "neck": "Neck",
              "head": "Head", "head_end": "HeadTop_End"}
    if side is None:
        return "mixamorig:" + center[role]
    S = side.capitalize()
    rest = "_".join(parts[1:])
    limb = {"shoulder": "Shoulder", "upper_arm": "Arm", "lower_arm": "ForeArm", "hand": "Hand",
            "upper_leg": "UpLeg", "lower_leg": "Leg", "foot": "Foot", "toe": "ToeBase", "toe_end": "Toe_End"}
    if rest in limb:
        return f"mixamorig:{S}{limb[rest]}"
    finger, k = parts[1], int(parts[2])
    return f"mixamorig:{S}Hand{MIXAMO_FINGER[finger]}{k}"


def skeleton(spec: Spec):
    """Return list of (role, parent_role, world_position, box_half_size, box_tail_role)."""
    J = []

    def add(role, parent, pos, half=None, tail=None):
        J.append((role, parent, np.array(pos, dtype=np.float64), half, tail))

    add("hips", None, (0, 0.95, 0), 0.12, "spine1")
    add("spine1", "hips", (0, 1.05, 0), 0.12, "spine2")
    add("spine2", "spine1", (0, 1.2, 0), 0.13, "spine3")
    add("spine3", "spine2", (0, 1.35, 0), 0.14, "neck")
    add("neck", "spine3", (0, 1.5, 0), 0.05, "head")
    add("head", "neck", (0, 1.6, 0), 0.1, "head_end")
    add("head_end", "head", (0, 1.8, 0))
    a = math.radians(spec.arm_angle)
    for side, sx in (("left", 1.0), ("right", -1.0)):
        d = np.array([sx * math.cos(a), -math.sin(a), 0.0])
        sh = np.array([sx * 0.05, 1.45, 0.0])
        arm = np.array([sx * 0.18, 1.45, 0.0])
        fore = arm + d * 0.27
        hand = fore + d * 0.25
        add(f"{side}_shoulder", "spine3", sh, 0.04, f"{side}_upper_arm")
        add(f"{side}_upper_arm", f"{side}_shoulder", arm, 0.05, f"{side}_lower_arm")
        add(f"{side}_lower_arm", f"{side}_upper_arm", fore, 0.045, f"{side}_hand")
        if spec.fingers:
            add(f"{side}_hand", f"{side}_lower_arm", hand, 0.025, f"{side}_middle_1")
            zs = {"thumb": 0.035, "index": 0.018, "middle": 0.0, "ring": -0.018, "little": -0.034}
            for f in FINGERS:
                base = hand + d * (0.03 if f == "thumb" else 0.08) + np.array([0, 0, zs[f]])
                prev = f"{side}_hand"
                nseg = spec.finger_segments
                for k in range(1, nseg + 2):
                    role = f"{side}_{f}_{k if k <= nseg else 4}"
                    pos = base + d * 0.03 * (k - 1)
                    tail = f"{side}_{f}_{k + 1 if k < nseg else 4}" if k <= nseg else None
                    add(role, prev, pos, 0.007 if k <= nseg else None, tail)
                    prev = role
        else:
            add(f"{side}_hand", f"{side}_lower_arm", hand, 0.03, None)
        add(f"{side}_upper_leg", "hips", (sx * 0.1, 0.92, 0), 0.07, f"{side}_lower_leg")
        add(f"{side}_lower_leg", f"{side}_upper_leg", (sx * 0.1, 0.5, 0), 0.055, f"{side}_foot")
        if spec.toes:
            add(f"{side}_foot", f"{side}_lower_leg", (sx * 0.1, 0.08, 0), 0.04, f"{side}_toe")
            add(f"{side}_toe", f"{side}_foot", (sx * 0.1, 0.02, 0.12), 0.02, f"{side}_toe_end")
            add(f"{side}_toe_end", f"{side}_toe", (sx * 0.1, 0.02, 0.2))
        else:
            add(f"{side}_foot", f"{side}_lower_leg", (sx * 0.1, 0.08, 0), 0.04, None)
    return J


def _box(head, tail, half, hand_tail=None):
    if tail is None:
        tail = head + (hand_tail if hand_tail is not None else np.array([0, -0.1, 0]))
    d = tail - head
    L = np.linalg.norm(d)
    d = d / L
    a = head + d * L * 0.05
    b = head + d * L * 0.95
    ref = np.array([0, 1.0, 0]) if abs(d[1]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    corners = []
    for end in (a, b):
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append(end + u * su * half + v * sv * half)
    corners = np.array(corners)
    center = (a + b) / 2
    normals = corners - center
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    tris = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    return corners, normals, tris


def png_bytes(w: int, h: int) -> bytes:
    raw = b"".join(b"\x00" + bytes([200, 120, 80]) * w for _ in range(h))

    def chunk(t, data):
        c = struct.pack(">I", len(data)) + t + data
        return c + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class _Blob:
    def __init__(self):
        self.data = bytearray()
        self.views: list[pygltflib.BufferView] = []
        self.accessors: list[pygltflib.Accessor] = []

    def view(self, raw: bytes) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        self.views.append(pygltflib.BufferView(buffer=0, byteOffset=len(self.data), byteLength=len(raw)))
        self.data.extend(raw)
        return len(self.views) - 1

    def accessor(self, arr: np.ndarray, ctype: int, atype: str, normalized=False, minmax=False,
                 sparse=None) -> int:
        dt = {5126: "<f4", 5121: "u1", 5123: "<u2", 5125: "<u4"}[ctype]
        a = np.ascontiguousarray(arr, dtype=dt)
        v = self.view(a.tobytes())
        acc = pygltflib.Accessor(bufferView=v, componentType=ctype, count=len(a), type=atype,
                                 normalized=normalized or None)
        if minmax:
            acc.min = a.reshape(len(a), -1).min(axis=0).tolist()
            acc.max = a.reshape(len(a), -1).max(axis=0).tolist()
        if sparse is not None:
            idx, vals = sparse
            iv = self.view(np.ascontiguousarray(idx, dtype="<u4").tobytes())
            vv = self.view(np.ascontiguousarray(vals, dtype=dt).tobytes())
            acc.sparse = pygltflib.Sparse(
                count=len(idx),
                indices=pygltflib.AccessorSparseIndices(bufferView=iv, byteOffset=0, componentType=5125),
                values=pygltflib.AccessorSparseValues(bufferView=vv, byteOffset=0),
            )
        self.accessors.append(acc)
        return len(self.accessors) - 1


def build(spec: Spec, path: str) -> str:
    J = skeleton(spec)
    roles = [j[0] for j in J]
    index = {r: i for i, r in enumerate(roles)}
    world_pos = {r: p for r, _, p, _, _ in J}
    R = np.eye(3)
    if spec.z_up:
        # Rotate +Y to +Z (as if a Z-up source was exported without converting axes).
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    off = np.array([0, spec.y_offset, 0])

    def tf(p):
        return (R @ (p + off)) * spec.unit

    # Mesh
    pos_l, nrm_l, uv_l, jnt_l, wgt_l, tri_main, tri_hand = [], [], [], [], [], [], []
    vcount = 0
    for bi, (role, parent, head, half, tail) in enumerate(J):
        if half is None:
            continue
        hand_tail = None
        if role.endswith("_hand") and tail is None:
            sx = 1 if role.startswith("left") else -1
            hand_tail = np.array([sx * 0.15 * math.cos(math.radians(spec.arm_angle)),
                                  -0.15 * math.sin(math.radians(spec.arm_angle)), 0])
        if role.endswith("_foot") and tail is None:
            hand_tail = np.array([0, -0.06, 0.15])
        c, n, t = _box(head, world_pos[tail] if tail else None, half, hand_tail)
        owner = bi
        is_finger = any(f"_{f}_" in role for f in FINGERS)
        side = role.split("_")[0]
        if is_finger and spec.finger_weights_to_hand:
            owner = index[f"{side}_hand"]
        if is_finger and spec.claw:
            k = role.split("_")[-1]
            owner = index[f"{side}_middle_{k}"]
        pos_l.append(np.array([tf(p) for p in c]))
        nrm_l.append(np.array([R @ x for x in n]))
        uv_l.append(np.column_stack([np.linspace(0, 1, 8), np.full(8, bi / len(J))]))
        j = np.zeros((8, 4), dtype=np.int64)
        w = np.zeros((8, 4))
        j[:, 0] = owner
        w[:, 0] = 1.0
        if spec.leak and role == "left_upper_leg":
            j[:, 1] = index["right_upper_leg"]
            w[:, 0] = 0.5
            w[:, 1] = 0.5
        if spec.unnormalized and role == "spine2":
            w[:, 0] = 0.8
        jnt_l.append(j)
        wgt_l.append(w)
        target = tri_hand if spec.split_primitives and (is_finger or role.endswith("_hand")) else tri_main
        target.append(t + vcount)
        vcount += 8
    positions = np.concatenate(pos_l)
    normals = np.concatenate(nrm_l)
    uvs = np.concatenate(uv_l)
    joints = np.concatenate(jnt_l)
    weights = np.concatenate(wgt_l)
    tri_main = np.concatenate(tri_main)

    if spec.crack or spec.seam:
        # Duplicate vertex 1 of the spine2 box and let two of its triangles use the copy.
        box = [r for r, _, _, h, _ in J if h is not None].index("spine2")
        v = box * 8 + 1
        positions = np.vstack([positions, positions[v]])
        normals = np.vstack([normals, normals[v]])
        new_uv = uvs[v].copy()
        if spec.seam:
            new_uv = new_uv + 0.5
        uvs = np.vstack([uvs, new_uv])
        joints = np.vstack([joints, joints[v]])
        weights = np.vstack([weights, weights[v]])
        new = len(positions) - 1
        tris_with = np.nonzero((tri_main == v).any(axis=1))[0]
        for ti in tris_with[:2]:
            tri_main[ti][tri_main[ti] == v] = new

    blob = _Blob()
    sparse = None
    base_positions = positions
    if spec.sparse_positions:
        # Store wrong values for the first 5 vertices in the base view, correct them via sparse.
        base_positions = positions.copy()
        base_positions[:5] += 50.0
        sparse = (np.arange(5), positions[:5])
    a_pos = blob.accessor(base_positions, 5126, "VEC3", minmax=True, sparse=sparse)
    a_nrm = blob.accessor(normals, 5126, "VEC3")
    a_uv = blob.accessor(uvs, 5126, "VEC2")
    a_j = blob.accessor(joints, 5121, "VEC4")
    if spec.extra_influences:
        weights = weights * 0.92
    if spec.normalized_weights:
        wq = np.round(weights * 255).astype(np.int64)
        if not spec.unnormalized:
            wq[:, 0] += 255 - wq.sum(axis=1)
        a_w = blob.accessor(np.clip(wq, 0, 255), 5121, "VEC4", normalized=True)
    else:
        a_w = blob.accessor(weights, 5126, "VEC4")
    attrs = dict(POSITION=a_pos, NORMAL=a_nrm, TEXCOORD_0=a_uv, JOINTS_0=a_j, WEIGHTS_0=a_w)
    if spec.extra_influences:
        # Four more small influences on every vertex (second joint set).
        j1 = np.tile(np.array([index["hips"], index["spine1"], index["spine2"], index["spine3"]]), (len(joints), 1))
        attrs["JOINTS_1"] = blob.accessor(j1, 5121, "VEC4")
        attrs["WEIGHTS_1"] = blob.accessor(np.full((len(joints), 4), 0.02), 5126, "VEC4")
    material = 0 if (spec.texture or spec.missing_texture) else None
    prims = [pygltflib.Primitive(attributes=pygltflib.Attributes(**attrs),
                                 indices=blob.accessor(tri_main.ravel(), 5125, "SCALAR"), material=material)]
    if tri_hand:
        prims.append(pygltflib.Primitive(attributes=pygltflib.Attributes(**attrs),
                                         indices=blob.accessor(np.concatenate(tri_hand).ravel(), 5125, "SCALAR"),
                                         material=material))

    # Nodes: joints use translation relative to parent; world positions in (possibly scaled) units.
    nodes = []
    for role, parent, p, _, _ in J:
        wp = tf(p)
        pp = tf(world_pos[parent]) if parent else np.zeros(3)
        nodes.append(pygltflib.Node(name=_joint_name(spec, role), translation=(wp - pp).tolist()))
    for role, parent, _, _, _ in J:
        if parent:
            nodes[index[parent]].children.append(index[role])
    # Inverse bind matrices: joint world = S * T(world_pos) with S = armature scale.
    s = spec.armature_scale or 1.0
    ibm = []
    for role, _, p, _, _ in J:
        m = np.eye(4)
        m[:3, 3] = -tf(p)
        ibm.append(m.T.ravel())  # column-major
    a_ibm = blob.accessor(np.array(ibm), 5126, "MAT4")
    root_nodes = [index["hips"]]
    if spec.armature_scale is not None:
        nodes.append(pygltflib.Node(name="Armature", children=[index["hips"]], scale=[s, s, s]))
        root_nodes = [len(nodes) - 1]
    nodes.append(pygltflib.Node(name="Body", mesh=0, skin=0))
    mesh_node = len(nodes) - 1
    if spec.armature_scale is not None:
        # The mesh sits under the scaled armature, as FBX exports commonly do.
        nodes[root_nodes[0]].children.append(mesh_node)
        scene_nodes = root_nodes
    else:
        scene_nodes = root_nodes + [mesh_node]

    gltf = pygltflib.GLTF2(
        asset=pygltflib.Asset(version="2.0", generator="ai-character-check test builder"),
        scene=0,
        scenes=[pygltflib.Scene(nodes=scene_nodes)],
        nodes=nodes,
        meshes=[pygltflib.Mesh(name="Body", primitives=prims)],
        skins=[pygltflib.Skin(name="Skeleton", joints=list(range(len(J))), inverseBindMatrices=a_ibm,
                              skeleton=index["hips"])],
    )

    if spec.texture or spec.missing_texture:
        if spec.missing_texture:
            gltf.images = [pygltflib.Image(uri="textures/does_not_exist.png")]
        else:
            iv = blob.view(png_bytes(*spec.texture))
            gltf.images = [pygltflib.Image(bufferView=iv, mimeType="image/png", name="albedo")]
        gltf.textures = [pygltflib.Texture(source=0)]
        gltf.materials = [pygltflib.Material(
            name="Skin", pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=0)))]

    if spec.anim:
        _add_animation(spec, gltf, blob, index, tf)

    gltf.accessors = blob.accessors
    gltf.bufferViews = blob.views
    gltf.buffers = [pygltflib.Buffer(byteLength=len(blob.data))]
    gltf.set_binary_blob(bytes(blob.data))
    gltf.save_binary(path)
    return path


def _add_animation(spec, gltf, blob, index, tf):
    times = np.linspace(0, 1.0, 11)
    samplers, channels = [], []

    def channel(node, path, values, atype):
        i = blob.accessor(times, 5126, "SCALAR", minmax=True)
        o = blob.accessor(values, 5126, atype)
        samplers.append(pygltflib.AnimationSampler(input=i, output=o, interpolation="LINEAR"))
        channels.append(pygltflib.AnimationChannel(
            sampler=len(samplers) - 1, target=pygltflib.AnimationChannelTarget(node=node, path=path)))

    hips = index["hips"]
    hip_t = np.array(gltf.nodes[hips].translation)
    if spec.anim in ("walk", "scale", "nan"):
        # Hips walk forward 1.2 m along +Z.
        vals = np.tile(hip_t, (len(times), 1))
        vals[:, 2] += np.linspace(0, 1.2, len(times)) * spec.unit
        if spec.anim == "nan":
            vals[5, 0] = np.nan
        channel(hips, "translation", vals, "VEC3")
    elif spec.anim == "inplace":
        vals = np.tile(hip_t, (len(times), 1))
        vals[:, 1] += 0.02 * np.sin(np.linspace(0, 2 * np.pi, len(times))) * spec.unit
        channel(hips, "translation", vals, "VEC3")
    # Swing the left upper arm around Z.
    ang = np.radians(np.linspace(0, 40, len(times)))
    q = np.column_stack([np.zeros_like(ang), np.zeros_like(ang), np.sin(ang / 2), np.cos(ang / 2)])
    channel(index["left_upper_arm"], "rotation", q, "VEC4")
    if spec.anim == "scale":
        channel(index["left_upper_arm"], "scale", np.tile([0.01, 0.01, 0.01], (len(times), 1)), "VEC3")
    gltf.animations = [pygltflib.Animation(name="Walk", samplers=samplers, channels=channels)]
