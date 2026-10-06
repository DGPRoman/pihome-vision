from __future__ import annotations

import os
from pathlib import Path

import pytest

#: A camera address whose credentials no output may contain.
CAMERA_URL = "rtsp://viewer:hunter2-not-real@192.168.1.50:554/stream2"
CAMERA_PASSWORD = "hunter2-not-real"
HUB_URL = "https://203.0.113.10"
HUB_KEY = "k" * 48

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "vision.example.yaml"


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test sees the developer's environment, or a .env in the checkout."""
    for name in list(os.environ):
        if name.startswith("PIHOME_VISION_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A complete, valid environment, pointing at a copy of the example config."""
    config = tmp_path / "vision.yaml"
    config.write_text(EXAMPLE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("PIHOME_VISION_CAMERA_URL", CAMERA_URL)
    monkeypatch.setenv("PIHOME_VISION_HUB_URL", HUB_URL)
    monkeypatch.setenv("PIHOME_VISION_HUB_KEY", HUB_KEY)
    monkeypatch.setenv("PIHOME_VISION_CONFIG_PATH", str(config))
    return config
