# ai-character-check

[Deutsch](README.de.md)

`aicc` checks a 3D character (glTF/GLB, or FBX through Blender) before it goes into a game engine and prints the result in a form an AI agent can act on: a verdict, an exit code, and one finding per problem with the measured value, the threshold, an explanation and a suggested fix.

It targets the failures that image-to-3D generators and auto-riggers produce most often: hands without finger bones, finger bones that carry no vertices, weights that leak to the other side of the body, unmerged vertices that open into cracks after smoothing, characters in centimetres or lying on their back, and animation clips with scale keys left over from a rescale in a DCC tool.

Other tools cover parts of this. Animation linters look at clips only, and general model validators check file structure without knowing what a humanoid needs. `aicc` checks a humanoid character end to end and writes a report that an agent can parse.

## Install

The package is not on PyPI yet. Install from a clone:

```bash
git clone https://github.com/beweiskette/ai-character-check.git
cd ai-character-check
python -m venv .venv
.venv/bin/pip install -e .          # Windows: .venv\Scripts\pip install -e .
```

Requirements: Python 3.10 or newer, `numpy` and `pygltflib`. Blender is optional (tested with 5.2; older versions are untested) and only needed for FBX input and for `aicc render`.

## Quick start

```bash
aicc check character.glb                  # text report, exit code 1 on errors
aicc check character.glb --format json    # machine-readable report
aicc check character.glb --format html --out report.html
aicc check character.glb --profile preview
aicc check character.fbx --blender /path/to/blender
aicc render character.glb --out renders/
```

Exit codes: `0` no errors, `1` errors found (with `--fail-on warning` also on warnings), `2` the file could not be checked (unreadable, compressed geometry, Blender missing for FBX).

## Example output

For a synthetic test character without finger bones, with one leaking thigh, one duplicated vertex and a clip that scales the upper arm to 0.01:

```text
$ aicc check character.glb --relative-name
ai-character-check 0.1.0  file: character.glb  profile: game
verdict: FAIL  (3 errors, 2 warnings, 5 info)
height m 1.8028  up +Y  joints 25  vertices 177  triangles 264  max influences 2  pose A-pose (-45.0 deg)

[ERROR] skeleton.hand_no_fingers.left: Left hand has no finger bones
        measured:  {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "little": 0}
        threshold: 5 fingers, >= 2 segments each
        fix: Re-run the auto-rigger with finger detection, or add finger chains and weight them.
[ERROR] skeleton.hand_no_fingers.right: Right hand has no finger bones
        ...
[ERROR] animation.scale_keys: Clip 'Walk' has scale keys far from 1
        measured:  {"clip": "Walk", "bones": {"mixamorig:LeftArm": {"min": 0.01, "max": 0.01}}, "count": 1, "max_deviation": 0.99}
        threshold: {"warning_deviation": 0.05, "error_deviation": 0.5}
        fix: Apply scale on the armature before baking, or strip scale tracks from the clip.
[WARN ] weights.left_right_leakage: Vertices weighted to bones on the opposite body side
        measured:  {"count": 4, "share_pct": 2.26, "top_pairs": [{"vertex_owner": "mixamorig:LeftUpLeg", "leaks_to": "mixamorig:RightUpLeg", "vertices": 4}]}
        threshold: {"min_weight": 0.1, "min_distance_from_midline_m": 0.0721, "error_share_pct": 5.0}
        fix: Clear the opposite-side weights in the listed regions (mirror weights from the clean side).
[WARN ] mesh.unmerged_vertices: Coincident vertices that are not merged (same position, UV and normal)
        measured:  {"count": 1, "share_pct": 0.56, "crack_edges": 2, ...}
        threshold: 0
        fix: Merge by distance with a tiny threshold (Blender: Mesh > Clean Up > Merge by Distance) before smoothing, decimating or rigging.
(5 info findings hidden, use --verbose or --format json)
```

The JSON report has this shape (shortened):

```json
{
  "tool": "ai-character-check",
  "version": "0.1.0",
  "file": "character.glb",
  "profile": "game",
  "verdict": "fail",
  "summary": {"error": 3, "warning": 2, "info": 5},
  "stats": {
    "joints": 25, "height_m": 1.8028, "up_axis": "+Y",
    "rest_pose": {"type": "A-pose", "arm_angle_deg": -45.0},
    "humanoid_mapping": {"left_hand": "mixamorig:LeftHand", "...": "..."},
    "finger_segments": {"left": {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "little": 0}},
    "animations": [{"name": "Walk", "length_s": 1.0, "root_motion": true, "root_displacement_m": 1.2}]
  },
  "findings": [
    {
      "id": "skeleton.hand_no_fingers.left",
      "severity": "error",
      "category": "skeleton",
      "title": "Left hand has no finger bones",
      "measured": {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "little": 0},
      "threshold": "5 fingers, >= 2 segments each",
      "explanation": "Without finger bones the hand is a rigid block ...",
      "fix": "Re-run the auto-rigger with finger detection, or add finger chains and weight them."
    }
  ]
}
```

`verdict` is `pass`, `pass_with_warnings` or `fail`. Finding ids are stable; the part after the second dot (`.left`, `.right`) names the side.

## What is checked

| Area | Finding ids | What it measures |
|---|---|---|
| Skeleton | `skeleton.*` | Humanoid bones recognised by name (Mixamo, Unreal, VRM/Unity, rigify, 3ds Max biped, Character Creator; a VRM humanoid extension wins over names). Fingers per hand and segments per finger, toe bones per foot. A hand without any finger bone is an error. |
| Rest pose | `pose.*` | Angle of the upper arm to the horizontal: T-pose within 15 degrees, A-pose 15 to 60 degrees below, otherwise "arms-down" or "arms-raised". Asymmetric arms. Stored node pose that differs from the bind pose. |
| Skin weights | `weights.*` | Vertices without weights, weights not summing to 1, more than 4 influences (game profile), finger bones with no weighted vertices, vertices weighted to bones of the opposite body side, and a hand heuristic: share of hand vertices owned by the hand bone and each finger chain, flagged when one bone or one chain owns almost everything. |
| Mesh | `mesh.*` | Vertices at the same position (within 1e-5 of the mesh size) that are not merged, split into: identical attributes (crack), different UV only (seam, expected), same UV but different normal (hard edge or crack, reported as ambiguous), both different. Also counts open edges lying on top of each other with identical attributes. |
| Scale | `scale.*` | Height in metres of the skinned mesh in the stored pose (error outside 0.3 to 3.0 m, warning outside 1.2 to 2.2 m), up axis from head and feet bones, lowest point relative to 0, non-unit scale on skeleton nodes. |
| Animation | `animation.*` | Per clip: NaN or infinite keys, key times that do not increase, targets outside the file or outside the skeleton, scale keys far from 1, non-unit quaternions, root motion (horizontal hip displacement above 10 % of the height), clip length. |
| Textures | `textures.*` | Missing or unreadable images, broken texture and material references, non-power-of-two sizes, image count and largest side. |

Profiles: `game` (default) is strict about influences, finger and toe completeness, texture sizes and animation scale keys. `preview` downgrades those to info or warning for characters that are only viewed.

## Renders

`aicc render model.glb --out dir` runs Blender headless with a script shipped in the package and writes:

- `front.png`: the whole body from the front, in rest pose
- `hand_left.png`, `hand_right.png`: close-ups of both hands, framed on the vertices weighted to the hand and finger bones
- `anim_strip.png` and `anim_00.png` to `anim_05.png`: six frames spread over the first animation clip
- `manifest.json`: file list, Blender version, clip name and frame times

It uses the Workbench engine at 512 x 512 pixels (`--size`, `--frames` change that). Blender is found through `--blender PATH`, the `BLENDER` environment variable, `PATH`, and the standard install folders. Without Blender the command prints how to provide it and exits with 2.

FBX input for `check` and `render` goes through the same Blender: the file is converted to GLB in a temporary folder and then checked. The report then carries a `source_note`, because scale, axes and bone names are what Blender's FBX importer made of the file.

## Use from Claude Code or Codex

The tool is a plain CLI, so an agent can call it directly. Add an instruction to `CLAUDE.md` (Claude Code) or `AGENTS.md` (Codex):

```markdown
After generating, rigging or converting a character model, run
`aicc check <file> --format json` and read the findings.
Fix every finding with severity "error" before importing the model into the engine.
For findings about hands or weights, run `aicc render <file> --out renders/` and look at the hand images.
```

To make it a hard gate in Claude Code, a `Stop` hook can refuse to finish while any character in a folder fails. Put this in `.claude/settings.json` (POSIX shell; adjust the folder):

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "for f in assets/characters/*.glb; do [ -e \"$f\" ] || continue; aicc check \"$f\" >/dev/null || { aicc check \"$f\" >&2; exit 2; }; done"
          }
        ]
      }
    ]
  }
}
```

Exit code 2 from a Stop hook keeps Claude working and passes the report on stderr back to it. If a finding cannot be fixed, the hook keeps blocking; use `--profile preview` or `--fail-on error` deliberately, or remove the hook.

## Limitations

- Bone roles come from names. A rig with generic names (`Bone.001`) gets `skeleton.not_humanoid` and the finger, toe, pose and side checks are skipped. There is no geometric bone detection yet.
- All measurements use the node pose stored in the file. If it differs from the bind pose, `pose.default_differs_from_bind` says so.
- Left/right leakage relies on side markers in bone names and on a lateral axis taken from the arm or leg bones. Vertices within 4 % of the height from the midline are not judged, so leakage between the inner thighs close to the crotch can go unnoticed.
- The hand ownership check is a heuristic. A mitten hand or a character with gloves can trigger it on purpose-built geometry.
- Crack detection works within one primitive. Seams between separate primitives or meshes (often material borders) are not examined. Same-UV, different-normal splits are reported as ambiguous because a hard edge and a crack look the same in the data.
- Morph targets are ignored. Meshes compressed with Draco or meshopt are rejected with exit code 2.
- Animation targets are node indices in glTF, so "bone does not exist" only catches references outside the file or outside the skeleton. Comparing a clip against a different skeleton by name is not implemented.
- Textures are checked by their header (PNG, JPEG, WebP, KTX2). The image content is not inspected.
- FBX support and renders depend on Blender. Both were tested with Blender 5.2 on Windows only.
- The automated tests use synthetic box characters generated in the test code. The tool was also run by hand on a few real auto-rigged characters, which are not part of the repository.

## Development

```bash
pip install -e ".[test]"
pytest
```

The two Blender tests run only when Blender is found and are skipped otherwise.

## License

MIT, see [LICENSE](LICENSE).
