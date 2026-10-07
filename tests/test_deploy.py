"""Guards that keep ``deploy/`` agreeing with the code it installs and starts.

The suite can neither start a unit nor run the installer, so what is checked here is
where those two files restate a fact that lives in Python: the exit status not to
retry, the command that runs the service, the variable that says where vision.yaml
is, how long stopping and a step can take, and the files the installer copies. Each is easy to
change on one side only, and the failure would surface on the machine the service
runs on rather than in CI.
"""

from __future__ import annotations

import re
import tomllib
from configparser import ConfigParser
from pathlib import Path

import pytest

from pihome_vision import __main__, camera, hub, service
from pihome_vision.settings import ENV_PREFIX, Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIT_PATH = REPO_ROOT / "deploy" / "pihome-vision.service"
INSTALLER_PATH = REPO_ROOT / "deploy" / "install.sh"


class _UnitParser(ConfigParser):
    """``ConfigParser`` in systemd's dialect."""

    def optionxform(self, optionstr: str) -> str:
        """Preserve case: systemd keys are ``CamelCase`` and case-sensitive."""
        return optionstr


@pytest.fixture(scope="module")
def service_section() -> dict[str, str]:
    """The unit's ``[Service]`` section. A key that repeats has its last value."""
    # systemd lets a key repeat, and does not %-interpolate values the way
    # configparser assumes, so both of those defaults are turned off.
    parser = _UnitParser(strict=False, interpolation=None)
    parser.read_string(UNIT_PATH.read_text(encoding="utf-8"))
    return dict(parser["Service"])


@pytest.fixture(scope="module")
def script() -> str:
    return INSTALLER_PATH.read_text(encoding="utf-8")


class TestUnitMatchesTheCode:
    def test_a_configuration_it_cannot_use_is_not_retried(
        self, service_section: dict[str, str]
    ) -> None:
        statuses = service_section["RestartPreventExitStatus"].split()

        assert str(__main__.EXIT_CONFIGURATION_ERROR) in statuses
        assert str(service.EXIT_CONFIGURATION_ERROR) in statuses

    def test_it_runs_the_service_through_a_console_script(
        self, service_section: dict[str, str]
    ) -> None:
        program, *arguments = service_section["ExecStart"].split()
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        assert Path(program).name in pyproject["project"]["scripts"]
        assert arguments == ["run"]
        assert __main__.build_parser().parse_args(arguments).handler is __main__._run

    def test_vision_yaml_is_the_credential_it_loads(self, service_section: dict[str, str]) -> None:
        name, _, source = service_section["LoadCredential"].partition(":")
        variable = f"{ENV_PREFIX}config_path".upper()

        assert "config_path" in Settings.model_fields
        assert f"{variable}=%d/{name}" in service_section["Environment"].split()
        assert Path(source).name == name

    def test_stopping_has_time_to_switch_a_light_off_and_close_the_camera(
        self, service_section: dict[str, str]
    ) -> None:
        """A light is read and then switched, and ffmpeg is given its grace after."""
        timeout = re.fullmatch(r"(\d+)s", service_section["TimeoutStopSec"])
        needed = service.STOP_GRACE + 2 * hub.TIMEOUT + camera._EXIT_GRACE

        assert timeout is not None
        assert int(timeout.group(1)) > needed

    def test_the_watchdog_waits_out_a_step_with_no_frame_and_a_slow_model(
        self, service_section: dict[str, str]
    ) -> None:
        """Silence for half of WatchdogSec means stuck, so that must be longer than the
        longest step: waiting for a frame, then a model run on a small machine."""
        watchdog = re.fullmatch(r"(\d+)s", service_section["WatchdogSec"])

        assert service_section["Type"] == "notify"
        assert watchdog is not None
        assert int(watchdog.group(1)) / 2 > service.FRAME_WAIT + 5

    def test_the_watchdog_stops_it_as_systemctl_stop_does(
        self, service_section: dict[str, str]
    ) -> None:
        """With SIGTERM, which switches the lights off; SIGABRT would leave them on."""
        assert service_section["WatchdogSignal"] == "SIGTERM"

    def test_sigterm_goes_to_the_service_alone(self, service_section: dict[str, str]) -> None:
        """ffmpeg is closed by the service once the lights are off, not by systemd."""
        assert service_section["KillMode"] == "mixed"

    def test_it_runs_as_an_account_of_its_own(self, service_section: dict[str, str]) -> None:
        assert service_section["DynamicUser"] == "yes"
        assert "User" not in service_section


class TestInstallerMatchesTheCode:
    def test_every_example_it_copies_exists(self, script: str) -> None:
        referenced = re.findall(r'\$REPO_ROOT/((?:config/|\.env)[^"\s]*)', script)

        assert sorted(referenced) == [".env.example", "config/vision.example.yaml"]
        for relative in referenced:
            assert (REPO_ROOT / relative).is_file()

    def test_it_installs_the_headless_lockfile(self, script: str) -> None:
        lockfile = re.search(r"^readonly LOCKFILE=(\S+)$", script, re.MULTILINE)

        assert lockfile is not None
        assert lockfile.group(1) == "requirements/headless.txt"
        assert (REPO_ROOT / lockfile.group(1)).is_file()

    def test_the_line_it_drops_from_the_example_is_there(self, script: str) -> None:
        """The unit says where vision.yaml is; the environment file must not."""
        dropped = re.search(r"grep -v '([^']+)'", script)
        example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

        assert dropped is not None
        matching = [line for line in example.splitlines() if re.match(dropped.group(1), line)]
        assert matching == [f"# {ENV_PREFIX}CONFIG_PATH=config/vision.yaml"]

    def test_it_reads_every_path_from_the_unit(self, script: str) -> None:
        for key in ("EnvironmentFile", "WorkingDirectory", "ExecStart", "LoadCredential"):
            assert f"unit_value {key}" in script
        assert "/etc/pihome-vision" not in script
        assert "/opt/pihome-vision" not in script
