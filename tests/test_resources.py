"""glTF resource URIs must not reach outside the model folder or onto the network."""

import builtins
import json
import os
import pathlib
import sys

import numpy as np
import pytest

import builder
from ai_character_check.checks import run_checks
from ai_character_check.cli import main
from ai_character_check.gltf_io import ExternalResourceBlocked, load_model

SECRET = b"outside-the-model-folder"
NET_HOST = "aicc-test.invalid"  # reserved TLD, never resolves


def _triangle_bytes() -> bytes:
    return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4").tobytes()


def write_gltf(folder: pathlib.Path, buffer_uri: str | None = None, image_uri: str | None = None) -> str:
    """Write a one-triangle .gltf. The buffer is embedded unless buffer_uri is given."""
    folder.mkdir(parents=True, exist_ok=True)
    geo = _triangle_bytes()
    if buffer_uri is None:
        import base64
        buffer_uri = "data:application/octet-stream;base64," + base64.b64encode(geo).decode()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [0, 0, 0], "max": [1, 1, 0]}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(geo)}],
        "buffers": [{"byteLength": len(geo), "uri": buffer_uri}],
    }
    if image_uri is not None:
        doc["images"] = [{"uri": image_uri}]
        doc["textures"] = [{"source": 0}]
        doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
        doc["meshes"][0]["primitives"][0]["material"] = 0
    path = folder / "model.gltf"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


@pytest.fixture
def layout(tmp_path):
    """tmp/model/ holds the model; tmp/secret.bin and tmp/secret.png lie outside it."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (tmp_path / "secret.bin").write_bytes(_triangle_bytes())
    (tmp_path / "secret.png").write_bytes(builder.png_bytes(37, 23))
    return tmp_path, model_dir


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test if anything touches a UNC or network-style path."""

    def is_net(p) -> bool:
        try:
            s = os.fsdecode(p)
        except TypeError:
            return False
        s = s.replace("\\", "/")
        return s.startswith("//") or NET_HOST in s

    def guard(name, fn):
        def wrapped(p, *a, **kw):
            if is_net(p):
                raise AssertionError(f"{name} touched network path {p!r}")
            return fn(p, *a, **kw)
        return wrapped

    monkeypatch.setattr(builtins, "open", guard("open", builtins.open))
    for mod, name in ((os, "stat"), (os, "lstat"), (os, "readlink"), (os.path, "isfile"), (os.path, "exists"),
                      (os.path, "isdir"), (os.path, "realpath"), (os.path, "islink")):
        monkeypatch.setattr(mod, name, guard(name, getattr(mod, name)))


@pytest.fixture
def reads(monkeypatch):
    """Record every file opened for reading."""
    opened: list[str] = []
    real_open = builtins.open

    def spy(p, mode="r", *a, **kw):
        if isinstance(p, (str, bytes, os.PathLike)) and "r" in mode:
            opened.append(os.path.abspath(os.fsdecode(p)))
        return real_open(p, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", spy)
    return opened


def _blocked_ids(exc_or_model):
    items = exc_or_model.blocked if isinstance(exc_or_model, ExternalResourceBlocked) \
        else exc_or_model.blocked_resources
    return [(b["kind"], b["index"]) for b in items]


# --------------------------------------------------------------------------- buffers


@pytest.mark.parametrize("uri", [
    "../secret.bin",
    "..%2Fsecret.bin",
    "%2E%2E/secret.bin",
    "sub/../../secret.bin",
    "..\\secret.bin",
])
def test_relative_buffer_escape_is_blocked(layout, reads, uri):
    root, model_dir = layout
    path = write_gltf(model_dir, buffer_uri=uri)
    with pytest.raises(ExternalResourceBlocked) as ei:
        load_model(path)
    assert _blocked_ids(ei.value) == [("buffer", 0)]
    assert str(root / "secret.bin") not in reads


def test_absolute_buffer_path_is_blocked(layout, reads):
    root, model_dir = layout
    secret = str(root / "secret.bin")
    for uri in (secret, secret.replace("\\", "/"), pathlib.Path(secret).as_uri()):
        path = write_gltf(model_dir, buffer_uri=uri)
        with pytest.raises(ExternalResourceBlocked):
            load_model(path)
    assert secret not in reads


@pytest.mark.parametrize("uri", [
    f"\\\\{NET_HOST}\\share\\x.bin",
    f"//{NET_HOST}/share/x.bin",
    f"%5C%5C{NET_HOST}%5Cshare%5Cx.bin",
    f"%2F%2F{NET_HOST}/share/x.bin",
    f"file://{NET_HOST}/share/x.bin",
    f"http://{NET_HOST}/x.bin",
    "\\\\?\\UNC\\" + NET_HOST + "\\share\\x.bin",
    "\\\\.\\pipe\\x",
])
def test_network_buffer_path_is_refused_before_any_file_access(tmp_path, no_network, uri):
    path = write_gltf(tmp_path / "model", buffer_uri=uri)
    with pytest.raises(ExternalResourceBlocked) as ei:
        load_model(path)
    assert ei.value.blocked[0]["reason"]


@pytest.mark.parametrize("uri", ["C:secret.bin", "secret.bin:stream", "C%3A/secret.bin", "a/.../b.bin"])
def test_odd_paths_are_blocked(tmp_path, uri):
    path = write_gltf(tmp_path / "model", buffer_uri=uri)
    with pytest.raises(ExternalResourceBlocked):
        load_model(path)


def test_relative_buffer_inside_folder_still_loads(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "data").mkdir(parents=True)
    (model_dir / "data" / "geo file.bin").write_bytes(_triangle_bytes())
    for uri in ("data/geo%20file.bin", "data/../data/geo%20file.bin", "./data/geo file.bin"):
        m = load_model(write_gltf(model_dir, buffer_uri=uri))
        assert m.blocked_resources == []
        assert len(m.primitives[0].positions) == 3


def test_cli_reports_blocked_buffer_as_error_finding(layout, capsys):
    root, model_dir = layout
    path = write_gltf(model_dir, buffer_uri="../secret.bin")
    assert main(["check", path, "--format", "json"]) == 1
    d = json.loads(capsys.readouterr().out)
    assert d["verdict"] == "fail"
    f = next(f for f in d["findings"] if f["id"] == "resource.external_path_blocked")
    assert f["severity"] == "error"
    assert f["measured"]["items"][0]["uri"] == "../secret.bin"


# --------------------------------------------------------------------------- images


def test_image_outside_folder_is_blocked_and_reported(layout, reads):
    root, model_dir = layout
    path = write_gltf(model_dir, image_uri="../secret.png")
    m = load_model(path)
    assert _blocked_ids(m) == [("image", 0)]
    assert m.images[0].data is None
    assert str(root / "secret.png") not in reads
    rep = run_checks(m)
    f = [f for f in rep.findings if f.id == "resource.external_path_blocked"]
    assert len(f) == 1 and f[0].severity == "error"
    # Reported once, not a second time as a missing texture.
    assert "textures.missing" not in {f.id for f in rep.findings}


def test_network_image_is_refused_before_any_file_access(tmp_path, no_network):
    m = load_model(write_gltf(tmp_path / "model", image_uri=f"//{NET_HOST}/share/t.png"))
    assert _blocked_ids(m) == [("image", 0)]


def test_image_inside_folder_loads(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "tex").mkdir(parents=True)
    (model_dir / "tex" / "a.png").write_bytes(builder.png_bytes(64, 32))
    m = load_model(write_gltf(model_dir, image_uri="tex/a.png"))
    assert m.blocked_resources == []
    assert (m.images[0].width, m.images[0].height) == (64, 32)


# --------------------------------------------------------------------------- links


def _make_dir_link(link: pathlib.Path, target: pathlib.Path) -> None:
    if sys.platform.startswith("win"):
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def test_directory_link_leaving_the_folder_is_blocked(layout, reads):
    root, model_dir = layout
    outside = root / "outside"
    outside.mkdir()
    (outside / "geo.bin").write_bytes(_triangle_bytes())
    _make_dir_link(model_dir / "linked", outside)
    path = write_gltf(model_dir, buffer_uri="linked/geo.bin")
    with pytest.raises(ExternalResourceBlocked):
        load_model(path)
    assert str(outside / "geo.bin") not in reads


def test_directory_link_inside_the_folder_is_allowed(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "real").mkdir(parents=True)
    (model_dir / "real" / "geo.bin").write_bytes(_triangle_bytes())
    _make_dir_link(model_dir / "alias", model_dir / "real")
    m = load_model(write_gltf(model_dir, buffer_uri="alias/geo.bin"))
    assert m.blocked_resources == []


def test_file_symlink_leaving_the_folder_is_blocked(layout):
    root, model_dir = layout
    try:
        os.symlink(root / "secret.bin", model_dir / "geo.bin")
    except (OSError, NotImplementedError):
        pytest.skip("creating file symlinks is not permitted here")
    with pytest.raises(ExternalResourceBlocked):
        load_model(write_gltf(model_dir, buffer_uri="geo.bin"))


def test_lexical_dotdot_after_link_is_checked_like_the_os_does(tmp_path):
    # "alias/../x" means model/x to Windows and to Blender (lexical ..), even though
    # "alias" points to model/real/deep. model/x is a link leaving the folder.
    model_dir = tmp_path / "model"
    (model_dir / "real" / "deep").mkdir(parents=True)
    _make_dir_link(model_dir / "alias", model_dir / "real" / "deep")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "geo.bin").write_bytes(_triangle_bytes())
    _make_dir_link(model_dir / "x", outside)
    with pytest.raises(ExternalResourceBlocked):
        load_model(write_gltf(model_dir, buffer_uri="alias/../x/geo.bin"))


# --------------------------------------------------------------------------- opt-in


def test_allow_external_resources_reads_local_files_outside(layout):
    root, model_dir = layout
    m = load_model(write_gltf(model_dir, buffer_uri="../secret.bin", image_uri="../secret.png"),
                   allow_external=True)
    assert m.blocked_resources == []
    assert (m.images[0].width, m.images[0].height) == (37, 23)


def test_allow_external_resources_still_blocks_network(tmp_path, no_network):
    with pytest.raises(ExternalResourceBlocked):
        load_model(write_gltf(tmp_path / "model", buffer_uri=f"//{NET_HOST}/share/x.bin"), allow_external=True)
    m = load_model(write_gltf(tmp_path / "m2", image_uri=f"file://{NET_HOST}/share/t.png"), allow_external=True)
    assert _blocked_ids(m) == [("image", 0)]


def test_cli_flag_allow_external_resources(layout, capsys):
    root, model_dir = layout
    path = write_gltf(model_dir, buffer_uri="../secret.bin")
    # The bare triangle fails the skeleton checks (exit 1), but the buffer was read.
    assert main(["check", path, "--format", "json", "--allow-external-resources"]) == 1
    d = json.loads(capsys.readouterr().out)
    assert "resource.external_path_blocked" not in {f["id"] for f in d["findings"]}
    assert d["stats"]["vertices"] == 3


# --------------------------------------------------------------------------- render


def test_render_refuses_blocked_resources_before_starting_blender(layout, tmp_path, monkeypatch, capsys):
    root, model_dir = layout
    fake_blender = tmp_path / "blender.exe"
    fake_blender.write_bytes(b"")
    import subprocess

    def boom(*a, **kw):
        raise AssertionError("Blender must not be started")

    monkeypatch.setattr(subprocess, "run", boom)
    path = write_gltf(model_dir, image_uri="../secret.png")
    assert main(["render", path, "--out", str(tmp_path / "r"), "--blender", str(fake_blender)]) == 2
    assert "outside the model folder" in capsys.readouterr().err


# --------------------------------------------------------------------------- FBX-style paths


def test_check_path_with_absolute_paths_like_the_fbx_importer_builds(layout):
    from ai_character_check import safe_paths

    root, model_dir = layout
    (model_dir / "tex").mkdir()
    (model_dir / "tex" / "a.png").write_bytes(b"x")
    base = str(model_dir)
    sep = os.sep
    ok, reason = safe_paths.check_path(base + sep + "tex" + sep + "a.png", base)
    assert reason is None and os.path.samefile(ok, model_dir / "tex" / "a.png")
    # The importer joins RelativeFilename onto the folder without normalising it.
    for p in (base + sep + ".." + sep + "secret.png", str(root / "secret.png"),
              base + sep + "tex" + sep + ".." + sep + ".." + sep + "secret.png"):
        assert safe_paths.check_path(p, base)[0] is None
        assert safe_paths.check_path(p, base, allow_external=True)[0] is not None
