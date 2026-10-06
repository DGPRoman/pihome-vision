from __future__ import annotations

from pathlib import Path

import pytest

from pihome_vision import __version__
from pihome_vision.__main__ import EXIT_CONFIGURATION_ERROR, main
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
