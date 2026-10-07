from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from pihome_vision.systemd import notify, watchdog_seconds


@pytest.fixture
def listening(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[socket.socket]:
    """A socket where systemd would listen, named as systemd names it."""
    path = tmp_path / "notify"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.bind(str(path))
        sock.settimeout(1)
        monkeypatch.setenv("NOTIFY_SOCKET", str(path))
        yield sock


def test_a_state_goes_to_the_socket_systemd_names(listening: socket.socket) -> None:
    notify("READY=1")

    assert listening.recv(64) == b"READY=1"


def test_an_abstract_socket_is_reached_by_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    name = f"pihome-vision-test-{os.getpid()}"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.bind(f"\0{name}")
        sock.settimeout(1)
        monkeypatch.setenv("NOTIFY_SOCKET", f"@{name}")

        notify("WATCHDOG=1")

        assert sock.recv(64) == b"WATCHDOG=1"


def test_without_systemd_nothing_is_sent() -> None:
    notify("READY=1")


def test_a_socket_that_is_not_there_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOTIFY_SOCKET", str(tmp_path / "gone"))

    notify("READY=1")


@pytest.mark.parametrize(
    ("usec", "pid", "seconds"),
    [
        (None, None, None),
        ("30000000", None, 30.0),
        ("30000000", "self", 30.0),
        ("30000000", "other", None),
        ("0", None, None),
        ("thirty", None, None),
    ],
)
def test_the_watchdog_is_only_this_processs_to_answer(
    monkeypatch: pytest.MonkeyPatch, usec: str | None, pid: str | None, seconds: float | None
) -> None:
    if usec is not None:
        monkeypatch.setenv("WATCHDOG_USEC", usec)
    if pid is not None:
        monkeypatch.setenv("WATCHDOG_PID", str(os.getpid() if pid == "self" else os.getpid() + 1))

    assert watchdog_seconds() == seconds
