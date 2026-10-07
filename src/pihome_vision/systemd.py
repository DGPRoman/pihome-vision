"""Telling systemd how the service is doing, as ``Type=notify`` and ``WatchdogSec=`` ask.

The protocol is a datagram of ``KEY=value`` lines to the socket systemd names in
``NOTIFY_SOCKET``, so no library is needed. Run any other way, there is no socket, and
nothing is sent.
"""

from __future__ import annotations

import logging
import os
import socket

_log = logging.getLogger(__name__)


def notify(state: str) -> None:
    """Send systemd ``state``, ``READY=1`` say, if it is listening."""
    address = os.environ.get("NOTIFY_SOCKET", "")
    if not address:
        return
    if address.startswith("@"):
        address = "\0" + address[1:]  # an abstract socket
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC) as sock:
            sock.connect(address)
            sock.sendall(state.encode())
    except OSError as exc:
        # Nothing to be done here: if it matters, the watchdog will say so.
        _log.debug("could not tell systemd %s: %s", state, exc)


def watchdog_seconds() -> float | None:
    """How often systemd wants to hear that the service is alive, or ``None`` if it
    does not."""
    pid = os.environ.get("WATCHDOG_PID", "")
    if pid and pid != str(os.getpid()):
        return None  # meant for another process: a child that inherited it
    try:
        usec = int(os.environ.get("WATCHDOG_USEC", ""))
    except ValueError:
        return None
    return usec / 1_000_000 if usec > 0 else None
