from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest = importlib.import_module("pytest")


@pytest.fixture(autouse=True)
def force_offline_test_environment(monkeypatch) -> None:
    """Keep the automated test suite deterministic and offline-safe."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("PULSE_SERVER", raising=False)


@pytest.fixture
def repo_fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"
