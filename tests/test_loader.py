import struct

import numpy as np
import pygltflib
import pytest

from ai_character_check.gltf_io import LoadError, image_size, load_model


def test_clean_fixture_geometry(make):
    m = load_model(make())
    assert len(m.skins) == 1
    assert len(m.joint_nodes) == 65
    pts = np.concatenate([p.world_positions for p in m.primitives])
    assert pts[:, 1].max() - pts[:, 1].min() == pytest.approx(1.8, abs=0.02)


def test_sparse_accessor_overrides_base_values(make):
    plain = load_model(make())
    sparse = load_model(make(sparse_positions=True))
    np.testing.assert_allclose(sparse.primitives[0].positions, plain.primitives[0].positions, atol=1e-6)


def test_normalized_ubyte_weights_are_decoded(make):
    m = load_model(make(normalized_weights=True, leak=True))
    w = m.primitives[0].weights
    assert w.dtype == np.float64
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
    assert np.any(np.isclose(w, 128 / 255))


def test_multiple_primitives_and_second_joint_set(make):
    m = load_model(make(split_primitives=True, extra_influences=True))
    assert len(m.primitives) == 2
    assert m.primitives[0].joints.shape[1] == 8
    assert m.primitives[0].weights.shape[1] == 8


def test_hierarchy_scale_is_applied(make):
    # Coordinates in centimetres under an armature node with scale 0.01.
    m = load_model(make(unit=100.0, armature_scale=0.01))
    pts = np.concatenate([p.world_positions for p in m.primitives])
    assert pts[:, 1].max() - pts[:, 1].min() == pytest.approx(1.8, abs=0.02)
    hips = m.node_names.index("mixamorig:Hips")
    assert m.node_position(hips)[1] == pytest.approx(0.95, abs=1e-6)


def test_cubic_spline_tangents_are_stripped(make, tmp_path):
    path = make(anim="inplace")
    g = pygltflib.GLTF2().load(path)
    blob = bytearray(g.binary_blob())
    # Replace the first sampler output with a 3x longer CUBICSPLINE output.
    smp = g.animations[0].samplers[0]
    n = g.accessors[smp.input].count
    vals = np.zeros((n, 3, 3), dtype="<f4")
    vals[:, 1, :] = [0, 7, 0]
    while len(blob) % 4:
        blob.append(0)
    g.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=len(blob), byteLength=vals.nbytes))
    blob.extend(vals.tobytes())
    g.accessors.append(pygltflib.Accessor(bufferView=len(g.bufferViews) - 1, componentType=5126,
                                          count=n * 3, type="VEC3"))
    smp.output = len(g.accessors) - 1
    smp.interpolation = "CUBICSPLINE"
    g.buffers[0].byteLength = len(blob)
    g.set_binary_blob(bytes(blob))
    out = str(tmp_path / "cubic.glb")
    g.save_binary(out)
    m = load_model(out)
    ch = m.animations[0].channels[0]
    assert ch.values.shape == (n, 3)
    assert np.all(ch.values[:, 1] == 7)


def test_missing_file_raises():
    with pytest.raises(LoadError):
        load_model("does-not-exist.glb")


def test_garbage_file_raises(tmp_path):
    p = tmp_path / "bad.glb"
    p.write_bytes(b"this is not gltf")
    with pytest.raises(LoadError):
        load_model(str(p))


def test_image_size_png_and_jpeg():
    import builder

    assert image_size(builder.png_bytes(100, 50)) == (100, 50)
    jpeg = b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 4) + b"\x00\x00" + b"\xff\xc0" + struct.pack(
        ">HBHH", 17, 8, 300, 640) + b"\x03" + b"\x00" * 9
    assert image_size(jpeg) == (640, 300)
    assert image_size(b"nope") is None


def test_compressed_geometry_is_rejected(make, tmp_path):
    g = pygltflib.GLTF2().load(make())
    g.extensionsUsed = ["KHR_draco_mesh_compression"]
    g.extensionsRequired = ["KHR_draco_mesh_compression"]
    out = str(tmp_path / "draco.glb")
    g.save_binary(out)
    with pytest.raises(LoadError, match="compressed geometry"):
        load_model(out)
