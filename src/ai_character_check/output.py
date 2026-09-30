"""Text, JSON and HTML renderers for a report."""

from __future__ import annotations

import html
import json

from .findings import Report

LABEL = {"error": "ERROR", "warning": "WARN ", "info": "info "}


def _compact(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False)
        return s if len(s) <= 160 else s[:157] + "..."
    return str(v)


def to_json(rep: Report) -> str:
    return json.dumps(rep.to_dict(), indent=2, ensure_ascii=False, default=_default)


def _default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def to_text(rep: Report, verbose: bool = False) -> str:
    d = json.loads(to_json(rep))
    s = d["summary"]
    lines = [
        f"ai-character-check {d['version']}  file: {d['file']}  profile: {d['profile']}",
        f"verdict: {d['verdict'].upper()}  ({s['error']} errors, {s['warning']} warnings, {s['info']} info)",
    ]
    if d.get("source_note"):
        lines.append(f"note: {d['source_note']}")
    st = d["stats"]
    facts = []
    for key, label in (("height_m", "height m"), ("up_axis", "up"), ("joints", "joints"), ("vertices", "vertices"),
                       ("triangles", "triangles"), ("max_influences", "max influences")):
        if key in st:
            facts.append(f"{label} {st[key]}")
    if "rest_pose" in st:
        facts.append(f"pose {st['rest_pose']['type']} ({st['rest_pose']['arm_angle_deg']} deg)")
    if facts:
        lines.append("  ".join(facts))
    lines.append("")
    for f in d["findings"]:
        if f["severity"] == "info" and not verbose:
            continue
        lines.append(f"[{LABEL[f['severity']]}] {f['id']}: {f['title']}")
        if f["measured"] is not None:
            lines.append(f"        measured:  {_compact(f['measured'])}")
        if f["threshold"] is not None:
            lines.append(f"        threshold: {_compact(f['threshold'])}")
        if verbose and f["explanation"]:
            lines.append(f"        why: {f['explanation']}")
        if f["fix"]:
            lines.append(f"        fix: {f['fix']}")
    infos = sum(1 for f in d["findings"] if f["severity"] == "info")
    if infos and not verbose:
        lines.append(f"({infos} info findings hidden, use --verbose or --format json)")
    return "\n".join(lines) + "\n"


def to_html(rep: Report, images: list[str] | None = None) -> str:
    d = json.loads(to_json(rep))
    e = html.escape
    rows = []
    for f in d["findings"]:
        rows.append(
            f"<tr class='{e(f['severity'])}'><td><span class='sev'>{e(f['severity'])}</span></td>"
            f"<td><code>{e(f['id'])}</code><div class='t'>{e(f['title'])}</div>"
            f"<div class='x'>{e(f['explanation'])}</div>"
            + (f"<div class='fix'>Fix: {e(f['fix'])}</div>" if f["fix"] else "")
            + f"</td><td><pre>{e(json.dumps(f['measured'], indent=1, ensure_ascii=False))}</pre></td>"
            f"<td><pre>{e(json.dumps(f['threshold'], indent=1, ensure_ascii=False))}</pre></td></tr>"
        )
    gallery = ""
    if images:
        gallery = "<h2>Renders</h2><div class='gal'>" + "".join(
            f"<figure><img src='{e(p)}' alt='{e(p)}'><figcaption>{e(p)}</figcaption></figure>" for p in images
        ) + "</div>"
    s = d["summary"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Character check</title>
<style>
:root {{ --bg:#fbfbfa; --fg:#1d1d1b; --muted:#5f5f5a; --line:#dcdcd6; --err:#b3261e; --warn:#8a5a00; --info:#3d5a80; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ececea; --muted:#a3a39d; --line:#34342f;
  --err:#ff8a80; --warn:#ffcc66; --info:#9ec5ff; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:1100px; margin:0 auto; }}
h1 {{ font-size:20px; margin:0 0 4px; }} .meta {{ color:var(--muted); margin-bottom:16px; word-break:break-all; }}
.verdict {{ font-weight:700; }} .fail {{ color:var(--err); }} .pass_with_warnings {{ color:var(--warn); }}
.pass {{ color:var(--info); }}
table {{ border-collapse:collapse; width:100%; }} td {{ border-top:1px solid var(--line); padding:8px 6px; vertical-align:top; }}
pre {{ margin:0; font-size:12px; white-space:pre-wrap; word-break:break-word; max-width:280px; }}
.sev {{ font-size:12px; font-weight:700; text-transform:uppercase; }}
tr.error .sev {{ color:var(--err); }} tr.warning .sev {{ color:var(--warn); }} tr.info .sev {{ color:var(--info); }}
.t {{ font-weight:600; }} .x {{ color:var(--muted); font-size:13px; }} .fix {{ font-size:13px; margin-top:4px; }}
.gal {{ display:flex; flex-wrap:wrap; gap:12px; }} .gal img {{ max-width:100%; width:320px; border:1px solid var(--line); }}
figure {{ margin:0; }} figcaption {{ font-size:12px; color:var(--muted); }}
@media (max-width:700px) {{ td:nth-child(4) {{ display:none; }} }}
</style></head><body><main>
<h1>Character check</h1>
<div class="meta">{e(d['file'])} &middot; profile {e(d['profile'])} &middot; ai-character-check {e(d['version'])}</div>
<p class="verdict {e(d['verdict'])}">Verdict: {e(d['verdict'])} ({s['error']} errors, {s['warning']} warnings, {s['info']} info)</p>
<table><tbody>{''.join(rows)}</tbody></table>
{gallery}
<h2>Stats</h2><pre style="max-width:none">{e(json.dumps(d['stats'], indent=2, ensure_ascii=False))}</pre>
</main></body></html>
"""
