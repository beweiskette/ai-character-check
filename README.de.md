# ai-character-check

[English](README.md)

`aicc` prüft eine 3D-Figur (glTF/GLB, FBX über Blender), bevor sie in eine Game-Engine kommt, und gibt das Ergebnis so aus, dass ein KI-Agent damit arbeiten kann: ein Urteil, ein Exitcode und pro Problem ein Befund mit Messwert, Schwelle, Erklärung und Vorschlag zur Behebung.

Das Werkzeug zielt auf die Fehler, die Bild-zu-3D-Generatoren und Auto-Rigger am häufigsten liefern: Hände ohne Fingerknochen, Fingerknochen ohne Gewichte, Gewichte auf der falschen Körperseite, nicht verschmolzene Vertices, die nach dem Glätten als Risse aufgehen, Figuren in Zentimetern oder auf dem Rücken liegend, und Animationen mit Skalierungsschlüsseln aus einer Umskalierung im DCC-Werkzeug.

Andere Werkzeuge decken Teile davon ab. Animations-Linter prüfen nur Clips, allgemeine Modellprüfer prüfen die Dateistruktur, wissen aber nicht, was eine menschliche Figur braucht. `aicc` prüft die ganze Figur und schreibt einen Bericht, den ein Agent auswerten kann.

## Installation

Das Paket ist noch nicht auf PyPI. Installation aus einem Klon:

```bash
git clone https://github.com/beweiskette/ai-character-check.git
cd ai-character-check
python -m venv .venv
.venv/bin/pip install -e .          # Windows: .venv\Scripts\pip install -e .
```

Voraussetzungen: Python 3.10 oder neuer, `numpy` und `pygltflib`. Blender ist freiwillig (geprüft mit 5.2, ältere Versionen nicht geprüft) und nur für FBX-Dateien und für `aicc render` nötig.

## Schnellstart

```bash
aicc check figur.glb                  # Textbericht, Exitcode 1 bei Fehlern
aicc check figur.glb --format json    # maschinenlesbarer Bericht
aicc check figur.glb --format html --out bericht.html
aicc check figur.glb --profile preview
aicc check figur.fbx --blender /pfad/zu/blender
aicc render figur.glb --out bilder/
```

Exitcodes: `0` keine Fehler, `1` Fehler gefunden (mit `--fail-on warning` auch bei Warnungen), `2` die Datei liess sich nicht prüfen (nicht lesbar, komprimierte Geometrie, Blender fehlt für FBX).

## Beispielausgabe

Für eine synthetische Testfigur ohne Fingerknochen, mit einem Oberschenkel, dessen Gewichte auf die andere Seite reichen, einem doppelten Vertex und einem Clip, der den Oberarm auf 0,01 skaliert:

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

Die Befunde selbst sind auf Englisch, damit Agenten und Skripte mit festen Texten und Kennungen arbeiten können. Der JSON-Bericht ist so aufgebaut (gekürzt):

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

`verdict` ist `pass`, `pass_with_warnings` oder `fail`. Die Kennungen der Befunde bleiben stabil; der Teil nach dem zweiten Punkt (`.left`, `.right`) nennt die Körperseite.

## Was geprüft wird

| Bereich | Kennungen | Was gemessen wird |
|---|---|---|
| Skelett | `skeleton.*` | Menschliche Knochen, erkannt am Namen (Mixamo, Unreal, VRM/Unity, Rigify, 3ds-Max-Biped, Character Creator; eine VRM-Humanoid-Erweiterung hat Vorrang vor den Namen). Finger pro Hand, Glieder pro Finger, Zehenknochen pro Fuss. Eine Hand ganz ohne Fingerknochen ist ein Fehler. |
| Ruhepose | `pose.*` | Winkel des Oberarms zur Waagrechten: T-Pose bis 15 Grad, A-Pose 15 bis 60 Grad darunter, sonst «arms-down» oder «arms-raised». Ungleiche Arme. Eine gespeicherte Knotenpose, die von der Bindepose abweicht. |
| Skin-Gewichte | `weights.*` | Vertices ohne Gewicht, Gewichtssummen ungleich 1, mehr als 4 Einflüsse (Profil game), Fingerknochen ohne gewichtete Vertices, Vertices mit Gewicht auf Knochen der anderen Körperseite, und eine Handprüfung: welcher Anteil der Hand-Vertices dem Handknochen und jeder Fingerkette gehört; gemeldet, wenn ein Knochen oder eine Kette fast alles besitzt. |
| Mesh | `mesh.*` | Vertices an derselben Stelle (innerhalb 1e-5 der Meshgrösse), die nicht verschmolzen sind, aufgeteilt in: gleiche Attribute (Riss), nur andere UV (Naht, erwartet), gleiche UV mit anderer Normale (harte Kante oder Riss, als unklar gemeldet), beides anders. Zusätzlich offene Kanten, die mit gleichen Attributen aufeinanderliegen. |
| Massstab | `scale.*` | Höhe des geskinnten Meshs in Metern in der gespeicherten Pose (Fehler ausserhalb 0,3 bis 3,0 m, Warnung ausserhalb 1,2 bis 2,2 m), Hochachse aus Kopf- und Fussknochen, tiefster Punkt gegenüber 0, Skalierung ungleich 1 auf Skelettknoten. |
| Animation | `animation.*` | Pro Clip: NaN- oder unendliche Schlüssel, Zeiten, die nicht ansteigen, Ziele ausserhalb der Datei oder ausserhalb des Skeletts, Skalierungsschlüssel weit weg von 1, nicht normierte Quaternionen, Root Motion (waagrechte Hüftverschiebung über 10 % der Höhe), Cliplänge. |
| Texturen | `textures.*` | Fehlende oder unlesbare Bilder, kaputte Textur- und Materialverweise, Grössen, die keine Zweierpotenz sind, Anzahl Bilder und grösste Seitenlänge. |

Profile: `game` (Standard) ist streng bei Einflüssen, Fingern und Zehen, Texturgrössen und Skalierungsschlüsseln. `preview` stuft diese Punkte für Figuren, die nur angeschaut werden, auf Info oder Warnung herunter.

## Bilder

`aicc render modell.glb --out ordner` startet Blender ohne Fenster mit einem Skript aus dem Paket und schreibt:

- `front.png`: die ganze Figur von vorne, in Ruhepose
- `hand_left.png`, `hand_right.png`: Nahaufnahmen beider Hände, ausgerichtet auf die Vertices, die an Hand- und Fingerknochen hängen
- `anim_strip.png` und `anim_00.png` bis `anim_05.png`: sechs Bilder über den ersten Animationsclip verteilt
- `manifest.json`: Dateiliste, Blender-Version, Clipname und Bildzeiten

Gerendert wird mit Workbench in 512 x 512 Pixeln (`--size` und `--frames` ändern das). Blender wird über `--blender PFAD`, die Umgebungsvariable `BLENDER`, `PATH` und die üblichen Installationsordner gesucht. Ohne Blender sagt der Befehl, wie man es angibt, und endet mit Exitcode 2.

FBX-Dateien laufen bei `check` und `render` über dasselbe Blender: Die Datei wird in einem temporären Ordner nach GLB umgewandelt und dann geprüft. Der Bericht trägt dann eine `source_note`, weil Massstab, Achsen und Knochennamen das sind, was der FBX-Import von Blender aus der Datei gemacht hat.

## Einbindung in Claude Code oder Codex

Das Werkzeug ist ein gewöhnliches Kommandozeilenprogramm, ein Agent kann es direkt aufrufen. Eine Anweisung in `CLAUDE.md` (Claude Code) oder `AGENTS.md` (Codex) genügt:

```markdown
After generating, rigging or converting a character model, run
`aicc check <file> --format json` and read the findings.
Fix every finding with severity "error" before importing the model into the engine.
For findings about hands or weights, run `aicc render <file> --out renders/` and look at the hand images.
```

Als feste Sperre in Claude Code kann ein `Stop`-Hook das Beenden verweigern, solange eine Figur in einem Ordner durchfällt. In `.claude/settings.json` (POSIX-Shell; Ordner anpassen):

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

Exitcode 2 aus einem Stop-Hook lässt Claude weiterarbeiten und gibt den Bericht über stderr an Claude zurück. Lässt sich ein Befund nicht beheben, blockiert der Hook weiter; dann bewusst `--profile preview` setzen oder den Hook entfernen.

## Grenzen

- Die Rollen der Knochen kommen aus den Namen. Ein Rig mit Namen wie `Bone.001` bekommt `skeleton.not_humanoid`, und die Prüfungen für Finger, Zehen, Pose und Körperseiten entfallen. Eine Erkennung aus der Geometrie gibt es noch nicht.
- Alle Messungen nutzen die in der Datei gespeicherte Knotenpose. Weicht sie von der Bindepose ab, meldet das `pose.default_differs_from_bind`.
- Die Seitenprüfung stützt sich auf Seitenkennzeichen in den Knochennamen und auf eine Querachse aus Arm- oder Beinknochen. Vertices, die weniger als 4 % der Höhe von der Mittellinie entfernt liegen, werden nicht beurteilt. Übergriffe zwischen den Innenseiten der Oberschenkel nahe am Schritt können deshalb durchrutschen.
- Die Handprüfung ist eine Heuristik. Fäustlinge oder Handschuhe können sie auslösen, obwohl die Geometrie so gewollt ist.
- Risse werden innerhalb eines Primitivs gesucht. Nähte zwischen getrennten Primitiven oder Meshes (oft Materialgrenzen) werden nicht geprüft. Gleiche UV mit anderer Normale wird als unklar gemeldet, weil harte Kante und Riss in den Daten gleich aussehen.
- Morph-Targets werden ignoriert. Mit Draco oder meshopt komprimierte Meshes werden mit Exitcode 2 abgelehnt.
- Animationsziele sind in glTF Knotennummern. «Knochen existiert nicht» erkennt deshalb nur Verweise ausserhalb der Datei oder ausserhalb des Skeletts. Ein Vergleich eines Clips mit einem anderen Skelett über die Namen fehlt noch.
- Texturen werden über ihren Dateikopf geprüft (PNG, JPEG, WebP, KTX2), der Bildinhalt nicht.
- FBX und Bilder hängen von Blender ab. Beides ist nur mit Blender 5.2 unter Windows geprüft.
- Die automatischen Tests nutzen synthetische Figuren aus Quadern, die im Testcode erzeugt werden. Von Hand lief das Werkzeug zusätzlich auf einigen echten, automatisch geriggten Figuren, die nicht im Repository liegen.

## Entwicklung

```bash
pip install -e ".[test]"
pytest
```

Die beiden Blender-Tests laufen nur, wenn Blender gefunden wird, sonst werden sie übersprungen.

## Lizenz

MIT, siehe [LICENSE](LICENSE).
