"""Finding and report data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SEVERITIES = ("error", "warning", "info")


@dataclass
class Finding:
    id: str
    severity: str
    title: str
    measured: Any = None
    threshold: Any = None
    explanation: str = ""
    fix: str = ""
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "severity": self.severity,
            "category": self.id.split(".", 1)[0],
            "title": self.title,
            "measured": self.measured,
            "threshold": self.threshold,
            "explanation": self.explanation,
            "fix": self.fix,
        }
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class Report:
    file: str
    profile: str
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    source_note: Optional[str] = None

    def add(self, f: Finding) -> None:
        self.findings.append(f)

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    @property
    def verdict(self) -> str:
        if self.count("error"):
            return "fail"
        if self.count("warning"):
            return "pass_with_warnings"
        return "pass"

    def sorted_findings(self) -> list[Finding]:
        order = {s: i for i, s in enumerate(SEVERITIES)}
        return sorted(self.findings, key=lambda f: order.get(f.severity, 9))

    def to_dict(self) -> dict:
        from . import __version__

        d = {
            "tool": "ai-character-check",
            "version": __version__,
            "file": self.file,
            "profile": self.profile,
            "verdict": self.verdict,
            "summary": {s: self.count(s) for s in SEVERITIES},
            "stats": self.stats,
            "findings": [f.to_dict() for f in self.sorted_findings()],
        }
        if self.source_note:
            d["source_note"] = self.source_note
        return d
