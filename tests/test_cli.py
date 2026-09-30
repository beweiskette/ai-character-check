import json

from ai_character_check.cli import main


def test_exit_codes_and_json(make, capsys):
    good = make()
    bad = make(fingers=False)
    assert main(["check", good, "--format", "json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["verdict"] == "pass"
    assert d["summary"]["error"] == 0
    assert main(["check", bad, "--format", "json"]) == 1
    d = json.loads(capsys.readouterr().out)
    assert d["verdict"] == "fail"
    f = next(f for f in d["findings"] if f["id"] == "skeleton.hand_no_fingers.left")
    assert set(f) >= {"id", "severity", "category", "title", "measured", "threshold", "explanation", "fix"}


def test_fail_on_warning(make, capsys):
    path = make(leak=True)
    assert main(["check", path]) == 0
    assert main(["check", path, "--fail-on", "warning"]) == 1
    assert "weights.left_right_leakage" in capsys.readouterr().out


def test_html_output(make, tmp_path):
    out = tmp_path / "report.html"
    assert main(["check", make(crack=True), "--format", "html", "--out", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in html and "mesh.unmerged_vertices" in html


def test_unreadable_file_exit_2(tmp_path, capsys):
    assert main(["check", str(tmp_path / "missing.glb")]) == 2
    assert "file not found" in capsys.readouterr().err


def test_fbx_without_blender_exit_2(tmp_path, capsys):
    fbx = tmp_path / "x.fbx"
    fbx.write_bytes(b"Kaydara FBX Binary  \x00")
    assert main(["check", str(fbx), "--blender", str(tmp_path / "no-blender.exe")]) == 2
    assert "Blender" in capsys.readouterr().err


def test_render_without_blender_exit_2(make, tmp_path, capsys):
    assert main(["render", make(), "--out", str(tmp_path / "r"), "--blender", str(tmp_path / "nope")]) == 2
    assert "Blender was not found" in capsys.readouterr().err
