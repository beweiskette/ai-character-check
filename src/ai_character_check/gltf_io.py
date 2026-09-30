"""Load a glTF 2.0 / GLB file into plain numpy arrays.

The loader resolves accessors (strided, sparse, normalized integers), the node
hierarchy, skins with inverse bind matrices, all mesh primitives and all
animation channels. Vertex positions are also skinned into the default pose so
that later checks can measure the character in world space.
"""

from __future__ import annotations

import base64
import os
import struct
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pygltflib


class LoadError(Exception):
    """Raised when a file cannot be read as glTF at all."""


COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
UNSUPPORTED_REQUIRED = {"KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_meshopt_compression"}
TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass
class Skin:
    index: int
    name: str
    joints: list[int]
    inverse_bind: np.ndarray  # (J, 4, 4)


@dataclass
class Primitive:
    mesh_index: int
    prim_index: int
    node_index: int
    mode: int
    positions: np.ndarray  # (N, 3) in mesh space
    normals: Optional[np.ndarray]
    uv0: Optional[np.ndarray]
    joints: Optional[np.ndarray]  # (N, K) skin-local joint indices
    weights: Optional[np.ndarray]  # (N, K) float weights
    indices: Optional[np.ndarray]  # flat index array, or None
    skin_index: Optional[int]
    world_positions: np.ndarray = None  # (N, 3) default pose, world space
    joint_nodes: Optional[np.ndarray] = None  # (N, K) node indices

    @property
    def triangles(self) -> np.ndarray:
        """Triangle vertex indices (T, 3) for TRIANGLES, STRIP and FAN modes."""
        idx = self.indices if self.indices is not None else np.arange(len(self.positions))
        if self.mode == 4:
            n = len(idx) - len(idx) % 3
            return idx[:n].reshape(-1, 3)
        if self.mode == 5 and len(idx) >= 3:
            tris = [(idx[i], idx[i + 1], idx[i + 2]) if i % 2 == 0 else (idx[i + 1], idx[i], idx[i + 2])
                    for i in range(len(idx) - 2)]
            return np.array(tris, dtype=np.int64)
        if self.mode == 6 and len(idx) >= 3:
            return np.array([(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)], dtype=np.int64)
        return np.zeros((0, 3), dtype=np.int64)


@dataclass
class Channel:
    node: Optional[int]
    path: str
    interpolation: str
    times: np.ndarray  # (K,)
    values: np.ndarray  # (K, C) key values (tangents removed for CUBICSPLINE)


@dataclass
class Animation:
    index: int
    name: str
    channels: list[Channel]


@dataclass
class ImageInfo:
    index: int
    name: str
    uri: Optional[str]
    mime: Optional[str]
    data: Optional[bytes]
    missing_reason: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class Model:
    path: str
    gltf: pygltflib.GLTF2
    node_names: list[str]
    parents: list[Optional[int]]
    children: list[list[int]]
    local: np.ndarray  # (n, 4, 4)
    world: np.ndarray  # (n, 4, 4)
    skins: list[Skin]
    primitives: list[Primitive]
    animations: list[Animation]
    images: list[ImageInfo]
    issues: list[str] = field(default_factory=list)
    source_note: Optional[str] = None

    @property
    def joint_nodes(self) -> list[int]:
        seen: list[int] = []
        for s in self.skins:
            for j in s.joints:
                if j not in seen:
                    seen.append(j)
        return seen

    def node_position(self, node: int) -> np.ndarray:
        return self.world[node][:3, 3].copy()

    def depth(self, node: int) -> int:
        d = 0
        while self.parents[node] is not None:
            node = self.parents[node]
            d += 1
        return d

    def skinned_primitives(self) -> list[Primitive]:
        return [p for p in self.primitives if p.skin_index is not None and p.weights is not None]


# --------------------------------------------------------------------------
# Math helpers


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def node_local_matrix(node: pygltflib.Node) -> np.ndarray:
    if node.matrix is not None and len(node.matrix) == 16:
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    t = node.translation if node.translation is not None else [0, 0, 0]
    r = node.rotation if node.rotation is not None else [0, 0, 0, 1]
    s = node.scale if node.scale is not None else [1, 1, 1]
    m[:3, :3] = quat_to_matrix(r) * np.array(s, dtype=np.float64)[None, :]
    m[:3, 3] = t
    return m


# --------------------------------------------------------------------------
# Buffer / accessor access


class _Reader:
    def __init__(self, gltf: pygltflib.GLTF2, base_dir: str):
        self.gltf = gltf
        self.base_dir = base_dir
        self._buffers: dict[int, bytes] = {}

    def buffer(self, index: int) -> bytes:
        if index in self._buffers:
            return self._buffers[index]
        buf = self.gltf.buffers[index]
        uri = buf.uri
        if uri is None:
            data = self.gltf.binary_blob()
            if data is None:
                raise LoadError(f"buffer {index} has no uri and the file has no binary chunk")
        elif uri.startswith("data:"):
            try:
                data = base64.b64decode(uri.split(",", 1)[1])
            except Exception as exc:  # noqa: BLE001
                raise LoadError(f"buffer {index}: invalid data uri ({exc})") from exc
        else:
            path = os.path.join(self.base_dir, urllib.parse.unquote(uri))
            if not os.path.isfile(path):
                raise LoadError(f"buffer {index}: external file not found: {uri}")
            with open(path, "rb") as fh:
                data = fh.read()
        data = bytes(data)
        self._buffers[index] = data
        return data

    def view_bytes(self, view_index: int) -> bytes:
        view = self.gltf.bufferViews[view_index]
        buf = self.buffer(view.buffer)
        start = view.byteOffset or 0
        return buf[start:start + view.byteLength]

    def _read_view(self, view_index: int, offset: int, dtype, count: int, ncomp: int,
                   tight: bool = False) -> np.ndarray:
        view = self.gltf.bufferViews[view_index]
        buf = self.buffer(view.buffer)
        dt = np.dtype(dtype).newbyteorder("<")
        elem = dt.itemsize * ncomp
        stride = view.byteStride if (view.byteStride and not tight) else elem
        start = (view.byteOffset or 0) + (offset or 0)
        if count == 0:
            return np.zeros((0, ncomp), dtype=dt)
        end = start + stride * (count - 1) + elem
        view_end = (view.byteOffset or 0) + view.byteLength
        if end > view_end or end > len(buf):
            raise LoadError(f"accessor data exceeds bufferView {view_index}")
        arr = np.ndarray((count, ncomp), dtype=dt, buffer=buf, offset=start,
                         strides=(stride, dt.itemsize))
        return np.array(arr)

    def accessor(self, index: int, normalize: bool = True) -> np.ndarray:
        acc = self.gltf.accessors[index]
        if acc.componentType not in COMPONENT_DTYPES or acc.type not in TYPE_SIZES:
            raise LoadError(f"accessor {index}: unsupported type {acc.type}/{acc.componentType}")
        dtype = COMPONENT_DTYPES[acc.componentType]
        ncomp = TYPE_SIZES[acc.type]
        if acc.bufferView is None:
            arr = np.zeros((acc.count, ncomp), dtype=dtype)
        else:
            arr = self._read_view(acc.bufferView, acc.byteOffset or 0, dtype, acc.count, ncomp)
        sp = acc.sparse
        if sp is not None and sp.count:
            ind = self._read_view(sp.indices.bufferView, sp.indices.byteOffset or 0,
                                  COMPONENT_DTYPES[sp.indices.componentType], sp.count, 1,
                                  tight=True).ravel().astype(np.int64)
            vals = self._read_view(sp.values.bufferView, sp.values.byteOffset or 0, dtype,
                                   sp.count, ncomp, tight=True)
            if ind.size and ind.max() >= acc.count:
                raise LoadError(f"accessor {index}: sparse index out of range")
            arr = arr.copy()
            arr[ind] = vals
        if normalize and acc.normalized and acc.componentType != 5126:
            f = arr.astype(np.float64)
            ct = acc.componentType
            if ct == 5121:
                f /= 255.0
            elif ct == 5123:
                f /= 65535.0
            elif ct == 5125:
                f /= 4294967295.0
            elif ct == 5120:
                f = np.maximum(f / 127.0, -1.0)
            elif ct == 5122:
                f = np.maximum(f / 32767.0, -1.0)
            return f
        if acc.componentType == 5126:
            return arr.astype(np.float64)
        return arr


# --------------------------------------------------------------------------


def _attr(attrs, name: str) -> Optional[int]:
    return getattr(attrs, name, None)


def _skin_matrices(model_world: np.ndarray, skin: Skin) -> np.ndarray:
    return np.stack([model_world[j] @ skin.inverse_bind[i] for i, j in enumerate(skin.joints)])


def skin_points(points: np.ndarray, joints: np.ndarray, weights: np.ndarray,
                skin_mats: np.ndarray, fallback: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Linear blend skinning of points (N, 3) with per-vertex joints/weights."""
    n = len(points)
    out = np.empty((n, 3))
    jcount = len(skin_mats)
    for s in range(0, n, chunk):
        p = points[s:s + chunk]
        j = np.clip(joints[s:s + chunk].astype(np.int64), 0, max(jcount - 1, 0))
        w = weights[s:s + chunk].astype(np.float64)
        total = w.sum(axis=1)
        safe = np.where(total > 1e-8, total, 1.0)
        wn = w / safe[:, None]
        ph = np.concatenate([p, np.ones((len(p), 1))], axis=1)
        acc = np.zeros((len(p), 3))
        for k in range(j.shape[1]):
            mats = skin_mats[j[:, k]]  # (m, 4, 4)
            acc += wn[:, k:k + 1] * np.einsum("nij,nj->ni", mats, ph)[:, :3]
        zero = total <= 1e-8
        if zero.any():
            acc[zero] = (fallback @ ph[zero].T).T[:, :3]
        out[s:s + chunk] = acc
    return out


def load_model(path: str) -> Model:
    if not os.path.isfile(path):
        raise LoadError(f"file not found: {path}")
    try:
        gltf = pygltflib.GLTF2().load(path)
    except Exception as exc:  # noqa: BLE001
        raise LoadError(f"cannot parse glTF: {exc}") from exc
    if gltf is None:
        raise LoadError("cannot parse glTF")
    unsupported = [e for e in (gltf.extensionsRequired or []) if e in UNSUPPORTED_REQUIRED]
    if unsupported:
        raise LoadError(f"compressed geometry is not supported ({', '.join(unsupported)}); re-export the file "
                        "without mesh compression (for example from Blender)")
    reader = _Reader(gltf, os.path.dirname(os.path.abspath(path)))
    issues: list[str] = []

    nodes = gltf.nodes or []
    n = len(nodes)
    names = [nd.name or f"node_{i}" for i, nd in enumerate(nodes)]
    parents: list[Optional[int]] = [None] * n
    children: list[list[int]] = [[] for _ in range(n)]
    for i, nd in enumerate(nodes):
        for c in nd.children or []:
            if 0 <= c < n:
                if parents[c] is not None:
                    issues.append(f"node {names[c]} has more than one parent")
                parents[c] = i
                children[i].append(c)
            else:
                issues.append(f"node {names[i]} references missing child {c}")
    local = np.stack([node_local_matrix(nd) for nd in nodes]) if n else np.zeros((0, 4, 4))
    world = np.zeros_like(local)
    done = [False] * n

    def resolve(i: int, stack: tuple = ()) -> np.ndarray:
        if done[i]:
            return world[i]
        if i in stack:
            issues.append("node hierarchy contains a cycle")
            return local[i]
        p = parents[i]
        world[i] = local[i] if p is None else resolve(p, stack + (i,)) @ local[i]
        done[i] = True
        return world[i]

    for i in range(n):
        resolve(i)

    skins: list[Skin] = []
    for si, sk in enumerate(gltf.skins or []):
        joints = [j for j in (sk.joints or []) if 0 <= j < n]
        if len(joints) != len(sk.joints or []):
            issues.append(f"skin {si} references missing joint nodes")
        if sk.inverseBindMatrices is not None:
            ibm = reader.accessor(sk.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
            if len(ibm) < len(joints):
                issues.append(f"skin {si}: fewer inverse bind matrices than joints")
                ibm = np.concatenate([ibm, np.tile(np.eye(4), (len(joints) - len(ibm), 1, 1))])
        else:
            ibm = np.tile(np.eye(4), (len(joints), 1, 1))
        skins.append(Skin(si, sk.name or f"skin_{si}", joints, ibm[:len(joints)]))

    # Nodes that instantiate meshes. A mesh can be used by several nodes.
    prims: list[Primitive] = []
    for ni, nd in enumerate(nodes):
        if nd.mesh is None or not (0 <= nd.mesh < len(gltf.meshes or [])):
            continue
        mesh = gltf.meshes[nd.mesh]
        skin_index = nd.skin if nd.skin is not None and 0 <= nd.skin < len(skins) else None
        for pi, pr in enumerate(mesh.primitives or []):
            a = pr.attributes
            pos_idx = _attr(a, "POSITION")
            if pos_idx is None:
                issues.append(f"mesh {mesh.name or nd.mesh} primitive {pi} has no POSITION")
                continue
            positions = reader.accessor(pos_idx)[:, :3]
            normals = reader.accessor(_attr(a, "NORMAL"))[:, :3] if _attr(a, "NORMAL") is not None else None
            uv0 = reader.accessor(_attr(a, "TEXCOORD_0"))[:, :2] if _attr(a, "TEXCOORD_0") is not None else None
            joint_sets, weight_sets = [], []
            k = 0
            while _attr(a, f"JOINTS_{k}") is not None and _attr(a, f"WEIGHTS_{k}") is not None:
                joint_sets.append(reader.accessor(_attr(a, f"JOINTS_{k}")).astype(np.int64))
                weight_sets.append(np.asarray(reader.accessor(_attr(a, f"WEIGHTS_{k}")), dtype=np.float64))
                k += 1
            joints = np.concatenate(joint_sets, axis=1) if joint_sets else None
            weights = np.concatenate(weight_sets, axis=1) if weight_sets else None
            indices = reader.accessor(pr.indices).ravel().astype(np.int64) if pr.indices is not None else None
            if indices is not None and indices.size and indices.max() >= len(positions):
                issues.append(f"mesh {mesh.name or nd.mesh} primitive {pi}: index out of range")
                indices = indices[indices < len(positions)]
            mode = pr.mode if pr.mode is not None else 4
            prim = Primitive(nd.mesh, pi, ni, mode, positions, normals, uv0, joints, weights,
                             indices, skin_index if joints is not None else None)
            if prim.skin_index is not None:
                skin = skins[prim.skin_index]
                if joints.size and joints.max() >= len(skin.joints):
                    issues.append(f"mesh {mesh.name or nd.mesh} primitive {pi}: joint index out of range")
                mats = _skin_matrices(world, skin)
                prim.world_positions = skin_points(positions, joints, weights, mats, world[ni])
                jn = np.array(skin.joints, dtype=np.int64)
                prim.joint_nodes = jn[np.clip(joints, 0, len(jn) - 1)]
            else:
                ph = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
                prim.world_positions = (world[ni] @ ph.T).T[:, :3]
            prims.append(prim)

    anims: list[Animation] = []
    for ai, an in enumerate(gltf.animations or []):
        chans: list[Channel] = []
        for ch in an.channels or []:
            if ch.sampler is None or not (0 <= ch.sampler < len(an.samplers or [])):
                issues.append(f"animation {an.name or ai}: channel with invalid sampler")
                continue
            smp = an.samplers[ch.sampler]
            times = reader.accessor(smp.input).ravel().astype(np.float64)
            values = np.asarray(reader.accessor(smp.output), dtype=np.float64)
            interp = smp.interpolation or "LINEAR"
            if interp == "CUBICSPLINE" and len(values) == 3 * len(times):
                values = values.reshape(len(times), 3, -1)[:, 1, :]
            target = ch.target
            path = target.path if target is not None else None
            if path == "weights" and len(times):
                values = values.reshape(len(times), -1)
            chans.append(Channel(target.node if target is not None else None, path or "", interp,
                                 times, values))
        anims.append(Animation(ai, an.name or f"animation_{ai}", chans))

    images: list[ImageInfo] = []
    for ii, im in enumerate(gltf.images or []):
        info = ImageInfo(ii, im.name or f"image_{ii}", im.uri, im.mimeType, None)
        try:
            if im.bufferView is not None:
                info.data = reader.view_bytes(im.bufferView)
            elif im.uri:
                if im.uri.startswith("data:"):
                    info.data = base64.b64decode(im.uri.split(",", 1)[1])
                else:
                    p = os.path.join(reader.base_dir, urllib.parse.unquote(im.uri))
                    if os.path.isfile(p):
                        with open(p, "rb") as fh:
                            info.data = fh.read()
                    else:
                        info.missing_reason = f"external file not found: {im.uri}"
            else:
                info.missing_reason = "image has neither uri nor bufferView"
        except (LoadError, IndexError, ValueError) as exc:
            info.missing_reason = str(exc)
        if info.data is not None:
            size = image_size(info.data)
            if size is None:
                info.missing_reason = info.missing_reason or "image data could not be decoded (unknown format)"
            else:
                info.width, info.height = size
        images.append(info)

    return Model(os.path.abspath(path), gltf, names, parents, children, local, world, skins,
                 prims, anims, images, issues)


# --------------------------------------------------------------------------
# Image header parsing without an imaging library


def image_size(data: bytes) -> Optional[tuple[int, int]]:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    if len(data) >= 4 and data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return int(w), int(h)
            i += 2 + seg_len
        return None
    if len(data) >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X":
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return w, h
        if chunk == b"VP8L":
            b = data[21:25]
            bits = int.from_bytes(b, "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8 ":
            w, h = struct.unpack("<HH", data[26:30])
            return w & 0x3FFF, h & 0x3FFF
    if len(data) >= 28 and data[:12] == b"\xabKTX 20\xbb\r\n\x1a\n":
        w, h = struct.unpack("<II", data[20:28])
        return int(w), int(h)
    return None
