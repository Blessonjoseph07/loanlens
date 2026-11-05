"""Shared fixtures for the feature tests (the core tests keep their own)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loanlens.data import FEATURES          # noqa: E402
from loanlens.predict import Predictor      # noqa: E402


@pytest.fixture(scope="session")
def shared_predictor():
    return Predictor()


@pytest.fixture(scope="session")
def samples():
    """The three real test-set applicants saved by train.py (approved, rejected, borderline)."""
    raw = json.loads((ROOT / "metrics.json").read_text())["sample_applicants"]
    return {name.split()[0].lower(): {f: s[f] for f in FEATURES} for name, s in raw.items()}
