from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from pihome_vision import __version__, detect
from pihome_vision.__main__ import EXIT_CONFIGURATION_ERROR, main
from pihome_vision.detect import Detection
from tests.conftest import CAMERA_PASSWORD, HUB_KEY


def test_version_names_the_installed_package(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["--version"])

    assert exited.value.code == 0
    assert capsys.readouterr().out.strip() == f"pihome-vision {__version__}"


def test_no_arguments_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out.startswith("usage: pihome-vision")


@pytest.mark.usefixtures("environment")
def test_validate_says_what_the_configuration_describes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate"]) == 0

    out = capsys.readouterr().out
    assert "camera   rtsp://***@192.168.1.50:554/stream2" in out
    assert "trigger  gate/driveway: zone" in out
    assert "light    gate-light <- driveway, wicket (off after 120 s, after dark)" in out
    assert CAMERA_PASSWORD not in out
    assert HUB_KEY not in out


def test_a_bad_environment_exits_2_without_the_password(
    environment: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PIHOME_VISION_CAMERA_URL", f"ftp://viewer:{CAMERA_PASSWORD}@192.168.1.50/")
    monkeypatch.delenv("PIHOME_VISION_HUB_KEY")

    with pytest.raises(SystemExit) as exited:
        main(["validate"])

    assert exited.value.code == EXIT_CONFIGURATION_ERROR
    captured = capsys.readouterr()
    assert "PIHOME_VISION_CAMERA_URL" in captured.err
    assert "PIHOME_VISION_HUB_KEY" in captured.err
    assert CAMERA_PASSWORD not in captured.err + captured.out


def test_a_bad_config_file_exits_2_naming_the_field(
    environment: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environment.write_text(
        environment.read_text(encoding="utf-8").replace("fps: 5", "fps: 0"), encoding="utf-8"
    )

    with pytest.raises(SystemExit) as exited:
        main(["validate"])

    assert exited.value.code == EXIT_CONFIGURATION_ERROR
    assert "cameras.0.fps" in capsys.readouterr().err


def test_a_dotenv_in_the_working_directory_is_read(
    environment: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PIHOME_VISION_HUB_KEY")
    Path(".env").write_text(f"PIHOME_VISION_HUB_KEY={HUB_KEY}\n", encoding="utf-8")

    assert main(["validate"]) == 0


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "yard.png"
    cv2.imwrite(str(path), np.zeros((90, 160, 3), dtype=np.uint8))
    return path


class _StubDetector:
    def __init__(self, path: Path, **_: object) -> None:
        self.path = path

    def detect(self, frame: npt.NDArray[np.uint8]) -> list[Detection]:
        assert frame.shape == (90, 160, 3)
        return [
            Detection(10, 20, 30, 40, 0.61, "vehicle"),
            Detection(100, 10, 20, 60, 0.88, "person"),
        ]


def test_detect_lists_what_the_model_found(
    environment: Path,
    image: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(detect, "Detector", _StubDetector)

    assert main(["detect", str(image)]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"{image}: 160x90, ")
    assert lines[0].endswith(", 2 found")
    assert lines[1:] == ["  person   0.88 at 110,40", "  vehicle  0.61 at 25,40"]


@pytest.mark.usefixtures("environment")
def test_detect_without_a_model_exits_2(image: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["detect", str(image)]) == EXIT_CONFIGURATION_ERROR
    assert "docs/models.md" in capsys.readouterr().err


@pytest.mark.usefixtures("environment")
def test_detect_needs_no_secrets(
    image: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("CAMERA_URL", "HUB_URL", "HUB_KEY"):
        monkeypatch.delenv(f"PIHOME_VISION_{name}")
    monkeypatch.setattr(detect, "Detector", _StubDetector)

    assert main(["detect", str(image)]) == 0


@pytest.mark.usefixtures("environment")
def test_detect_says_when_the_file_is_not_an_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("not a picture", encoding="utf-8")

    assert main(["detect", str(text)]) == 1
    assert "not an image" in capsys.readouterr().err
