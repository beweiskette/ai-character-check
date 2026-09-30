"""Shared, derived measurements used by several checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .gltf_io import Model
from .humanoid import HumanoidMap, build_map

AXIS_NAMES = ("X", "Y", "Z")


@dataclass
class Context:
    model: Model
    hm: HumanoidMap
    profile: dict
    points: np.ndarray  # world-space vertices used for size measurements
    up: np.ndarray  # unit vector
    up_label: str
    height: float
    hips: Optional[np.ndarray]
    lateral: Optional[np.ndarray]  # unit vector towards the character's left

    def pos(self, node: Optional[int]) -> Optional[np.ndarray]:
        return None if node is None else self.model.node_position(node)

    def name(self, node: int) -> str:
        return self.model.node_names[node]


def _axis_label(v: np.ndarray) -> str:
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] >= 0 else "-") + AXIS_NAMES[i]


def build_context(model: Model, profile: dict) -> Context:
    hm = build_map(model)
    skinned = model.skinned_primitives()
    prims = skinned or model.primitives
    if prims:
        points = np.concatenate([p.world_positions for p in prims])
        points = points[np.all(np.isfinite(points), axis=1)]
    else:
        points = np.zeros((0, 3))

    def p(role: str) -> Optional[np.ndarray]:
        n = hm.role(role)
        return None if n is None else model.node_position(n)

    head = p("head")
    feet = [x for x in (p("left_foot"), p("right_foot")) if x is not None]
    hips_node = hm.role("hips") if hm.role("hips") is not None else (hm.spine[0] if hm.spine else None)
    hips = None if hips_node is None else model.node_position(hips_node)

    up = None
    low = np.mean(feet, axis=0) if feet else hips
    if head is not None and low is not None:
        v = head - low
        if np.linalg.norm(v) > 1e-9:
            i = int(np.argmax(np.abs(v)))
            up = np.zeros(3)
            up[i] = 1.0 if v[i] >= 0 else -1.0
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
        if len(points):
            ext = points.max(axis=0) - points.min(axis=0)
            i = int(np.argmax(ext))
            up = np.zeros(3)
            up[i] = 1.0

    height = 0.0
    if len(points):
        proj = points @ up
        height = float(proj.max() - proj.min())

    lateral = None
    for a, b in (("left_upper_arm", "right_upper_arm"), ("left_upper_leg", "right_upper_leg"),
                 ("left_hand", "right_hand"), ("left_shoulder", "right_shoulder")):
        pa, pb = p(a), p(b)
        if pa is not None and pb is not None:
            v = pa - pb
            v = v - up * float(v @ up)
            if np.linalg.norm(v) > 1e-9:
                lateral = v / np.linalg.norm(v)
                if hips is None:
                    hips = (pa + pb) / 2
                break

    return Context(model, hm, profile, points, up, _axis_label(up), height, hips, lateral)
