import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import builder  # noqa: E402

from ai_character_check.checks import run_checks  # noqa: E402
from ai_character_check.gltf_io import load_model  # noqa: E402


@pytest.fixture
def make(tmp_path):
    """Build a fixture GLB from Spec keyword arguments and return its path."""
    counter = {"n": 0}

    def _make(**kw):
        counter["n"] += 1
        path = str(tmp_path / f"fixture_{counter['n']}.glb")
        return builder.build(builder.Spec(**kw), path)

    return _make


@pytest.fixture
def check(make):
    """Build a fixture and run all checks; returns the report."""

    def _check(profile="game", **kw):
        return run_checks(load_model(make(**kw)), profile)

    return _check


def ids(report, severity=None):
    return {f.id for f in report.findings if severity is None or f.severity == severity}
