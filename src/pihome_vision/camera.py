"""Frames from a camera, read through ffmpeg.

ffmpeg does the connecting and the decoding, which covers every camera worth having:
RTSP, HTTP (MJPEG and the like) and a local webcam. It writes YUV4MPEG to a pipe, a
format that states the frame size in its header, so one connection is enough: nothing
has to probe the camera first.

:class:`Stream` is one run of ffmpeg, from connecting until the first thing that goes
wrong. :class:`Camera` keeps one running in a thread, starts another when it fails, and
holds on to the newest frame only, so a detector that falls behind skips frames rather
than working through a backlog.

The camera's address carries its password. It goes to ffmpeg and nowhere else: every
message here is built from :func:`~pihome_vision.redact.mask_url` and
:func:`~pihome_vision.redact.scrub`.
"""

from __future__ import annotations

import enum
import logging
import os
import re
import select
import subprocess
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import IO, Final, Self

import cv2
import numpy as np

from pihome_vision.detect import Frame
from pihome_vision.redact import mask_url, scrub

_log = logging.getLogger(__name__)

#: How ffmpeg is started. A module attribute rather than a constant so the tests can put
#: a stand-in in its place.
FFMPEG: Sequence[str] = ("ffmpeg",)

#: How long connecting may take before the camera counts as not answering. Without a
#: limit of its own an unanswered TCP connection takes two minutes to fail.
OPEN_TIMEOUT: Final = 10.0

#: How long a connected stream may go without a frame before it is restarted.
STALL_TIMEOUT: Final = 10.0

#: The longest wait between attempts to reconnect. Each failure doubles the wait from
#: one second, up to this.
MAX_BACKOFF: Final = 30.0

#: Lines of ffmpeg's error output kept to explain why it stopped.
_STDERR_LINES: Final = 20

#: A YUV4MPEG header is one short line. Anything longer is not one.
_MAX_HEADER: Final = 1024

_READ_CHUNK: Final = 1 << 20

#: How often a wait for ffmpeg looks up to see whether it has been cancelled.
_CANCEL_POLL: Final = 0.25

#: How long ffmpeg has to exit once it has closed its output.
_EXIT_GRACE: Final = 5.0


class Failure(enum.Enum):
    """Why a stream could not be opened, or stopped. The value says it to a person."""

    REFUSED = "the camera refused the connection; check the address and the port"
    TIMEOUT = "no answer from the camera; check that it is on and reachable from here"
    UNAUTHORIZED = "the camera turned down the user name or password"
    NOT_FOUND = "the camera has no stream at that path"
    UNRESOLVED = "the camera's host name does not resolve"
    UNREACHABLE = "there is no route from this machine to the camera"
    NO_DEVICE = "there is no such local camera"
    NO_FFMPEG = "ffmpeg is not installed; on Debian or Ubuntu, apt install ffmpeg"
    STALLED = "the camera stopped sending frames"
    UNREADABLE = "ffmpeg sent something other than the video it was asked for"
    ENDED = "the stream ended"


#: What ffmpeg says for each failure, checked in this order. Status codes are matched as
#: whole words because ffmpeg prefixes its lines with addresses like ``@ 0x5a4041``.
_SYMPTOMS: Final[tuple[tuple[Failure, re.Pattern[str]], ...]] = tuple(
    (failure, re.compile(pattern, re.IGNORECASE))
    for failure, pattern in (
        (Failure.UNAUTHORIZED, r"\b40[13]\b|unauthorized|forbidden"),
        (Failure.NOT_FOUND, r"\b404\b"),
        (Failure.REFUSED, r"connection refused"),
        (Failure.TIMEOUT, r"timed out"),
        (Failure.UNRESOLVED, r"resolve hostname|name or service not known|name resolution"),
        (Failure.UNREACHABLE, r"no route to host|unreachable"),
        (Failure.NO_DEVICE, r"cannot open video device"),
    )
)


def classify(stderr: str) -> Failure:
    """The likeliest reason for ffmpeg giving up, from what it wrote to stderr."""
    for failure, symptom in _SYMPTOMS:
        if symptom.search(stderr):
            return failure
    return Failure.ENDED


@dataclass(frozen=True, slots=True)
class Timeouts:
    """How long to wait for a camera, in seconds."""

    #: For a connection, before the camera counts as not answering.
    open: float = OPEN_TIMEOUT
    #: For the next frame, before a connected stream counts as stalled.
    stall: float = STALL_TIMEOUT
    #: Between attempts to reconnect, at most.
    backoff: float = MAX_BACKOFF


class StreamError(Exception):
    """A stream could not be opened, or has stopped."""

    def __init__(self, failure: Failure, detail: str = "") -> None:
        self.failure = failure
        self.detail = detail
        super().__init__(f"{failure.value} ({detail})" if detail else failure.value)


def ffmpeg_arguments(source: str, *, fps: float | None = None) -> list[str]:
    """ffmpeg's arguments for reading ``source``, without the executable."""
    arguments = ["-hide_banner", "-nostdin", "-loglevel", "error"]
    if source.startswith("cam:"):
        arguments += ["-f", "v4l2", "-i", f"/dev/video{source.removeprefix('cam:')}"]
    else:
        if source.startswith(("rtsp://", "rtsps://")):
            # UDP loses packets on a busy network, and a lost packet is a smeared frame.
            arguments += ["-rtsp_transport", "tcp"]
            # Hand each frame on as soon as it arrives. By default ffmpeg buffers its
            # input, which on a camera was measured at about 0.4 s: time somebody
            # spends in the dark before anything here has seen them. RTSP only: over
            # MPEG-TS, which some HTTP cameras send, it leaves ffmpeg decoding nothing.
            arguments += ["-fflags", "nobuffer"]
        arguments += ["-flags", "low_delay", "-i", source]
    # Even sides, because 4:2:0 shares one colour sample between four pixels and the
    # conversion to BGR needs whole blocks. Cropping a pixel costs nothing.
    filters = ["crop=trunc(iw/2)*2:trunc(ih/2)*2"]
    if fps is not None:
        filters.insert(0, f"fps={fps:g}")
    return [
        *arguments,
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        ",".join(filters),
        "-pix_fmt",
        "yuv420p",
        "-f",
        "yuv4mpegpipe",
        "pipe:1",
    ]


def to_bgr(raw: bytes, width: int, height: int) -> Frame:
    """A 4:2:0 frame from ffmpeg as the BGR image the detector takes."""
    planes = np.frombuffer(raw, dtype=np.uint8).reshape(height * 3 // 2, width)
    return np.asarray(cv2.cvtColor(planes, cv2.COLOR_YUV2BGR_I420), dtype=np.uint8)


class Stream:
    """One run of ffmpeg: connected on construction, then read frame by frame.

    Every way of failing, from ffmpeg not being installed to the camera going quiet,
    comes out as a :class:`StreamError`. Setting ``cancel`` ends a wait for ffmpeg,
    whether for the connection or for a frame, within a quarter of a second.
    """

    def __init__(
        self,
        source: str,
        *,
        fps: float | None = None,
        command: Sequence[str] | None = None,
        timeouts: Timeouts = Timeouts(),  # noqa: B008  # frozen, so one instance is safe to share
        cancel: threading.Event | None = None,
    ) -> None:
        self._source = source
        self._cancel = cancel or threading.Event()
        self._stall_timeout = timeouts.stall
        self._buffer = bytearray()
        self._stderr: deque[str] = deque(maxlen=_STDERR_LINES)
        try:
            self._process = subprocess.Popen(  # noqa: S603  # the arguments are a list, never a shell
                [*(command or FFMPEG), *ffmpeg_arguments(source, fps=fps)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise StreamError(Failure.NO_FFMPEG) from None
        except OSError as exc:
            raise StreamError(Failure.ENDED, f"ffmpeg would not start: {exc.strerror}") from None
        assert self._process.stdout is not None  # noqa: S101  # PIPE was asked for
        assert self._process.stderr is not None  # noqa: S101
        self._stdout = self._process.stdout
        # Drained all along, because a pipe nobody reads fills up and stalls ffmpeg.
        self._drain = threading.Thread(
            target=self._collect, args=(self._process.stderr,), daemon=True
        )
        self._drain.start()
        try:
            self.width, self.height, self.rate = self._read_header(timeouts.open)
        except BaseException:
            self.close()
            raise
        self._frame_size = self.width * self.height * 3 // 2

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read(self) -> bytes:
        """The next frame, in ffmpeg's 4:2:0 layout; :func:`to_bgr` converts it."""
        if self._cancel.is_set():
            # Before anything already read: a frame handed on now would go unwatched.
            raise StreamError(Failure.ENDED, "stopped")
        deadline = time.monotonic() + self._stall_timeout
        marker = self._take_line(deadline, Failure.STALLED)
        if not marker.startswith(b"FRAME"):
            raise StreamError(Failure.UNREADABLE, "no frame marker")
        return self._take(self._frame_size, deadline, Failure.STALLED)

    def close(self) -> None:
        """Stop ffmpeg and release everything it held."""
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._drain.join()
        self._stdout.close()
        if self._process.stderr is not None:
            self._process.stderr.close()

    def _collect(self, stderr: IO[bytes]) -> None:
        for line in stderr:
            text = line.decode(errors="replace").strip()
            if text:
                self._stderr.append(scrub(text, self._source))

    def _read_header(self, timeout: float) -> tuple[int, int, float | None]:
        line = self._take_line(time.monotonic() + timeout, Failure.TIMEOUT)
        fields = line.split()
        if not fields or fields[0] != b"YUV4MPEG2":
            raise StreamError(Failure.UNREADABLE, "no YUV4MPEG header")
        values = {field[:1]: field[1:].decode(errors="replace") for field in fields[1:]}
        try:
            width, height = int(values[b"W"]), int(values[b"H"])
        except (KeyError, ValueError):
            raise StreamError(Failure.UNREADABLE, "no frame size in the header") from None
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise StreamError(Failure.UNREADABLE, f"a {width}x{height} frame")
        if not values.get(b"C", "420").startswith("420"):
            raise StreamError(Failure.UNREADABLE, f"colour layout {values[b'C']}")
        return width, height, _rate(values.get(b"F"))

    def _take_line(self, deadline: float, failure: Failure) -> bytes:
        while (end := self._buffer.find(b"\n")) < 0:
            if len(self._buffer) > _MAX_HEADER:
                raise StreamError(Failure.UNREADABLE, "a line with no end")
            self._fill(deadline, failure)
        line = bytes(self._buffer[:end])
        del self._buffer[: end + 1]
        return line

    def _take(self, size: int, deadline: float, failure: Failure) -> bytes:
        while len(self._buffer) < size:
            self._fill(deadline, failure)
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def _fill(self, deadline: float, failure: Failure) -> None:
        # Looked at before every read, and not only while waiting for one: a camera
        # that never pauses would otherwise never be stopped, and a frame that
        # trickles in a few bytes at a time never counted as stalled.
        while True:
            if self._cancel.is_set():
                raise StreamError(Failure.ENDED, "stopped")
            if time.monotonic() >= deadline:
                raise StreamError(failure)
            if select.select([self._stdout], [], [], _CANCEL_POLL)[0]:
                break
        chunk = os.read(self._stdout.fileno(), _READ_CHUNK)
        if not chunk:
            raise self._ended()
        self._buffer += chunk

    def _ended(self) -> StreamError:
        """Why ffmpeg closed its output, from what it said on the way out."""
        try:
            self._process.wait(_EXIT_GRACE)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._drain.join()
        said = "\n".join(self._stderr)
        last = self._stderr[-1] if self._stderr else f"ffmpeg exited {self._process.returncode}"
        return StreamError(classify(said), last)


def _rate(field: str | None) -> float | None:
    """A YUV4MPEG ``F`` field, ``25:1`` or ``30000:1001``, in frames a second."""
    if field is None:
        return None
    numerator, _, denominator = field.partition(":")
    try:
        rate = int(numerator) / int(denominator)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


class Camera:
    """Keeps a camera's newest frame to hand, reconnecting whenever its stream fails.

    A thread reads the stream for as long as :meth:`start` to :meth:`stop`. Frames are
    numbered as they arrive, and :meth:`next_frame` hands out the newest one after a
    given number, so a reader that falls behind skips to the present.
    """

    def __init__(
        self,
        source: str,
        *,
        fps: float | None = None,
        command: Sequence[str] | None = None,
        timeouts: Timeouts = Timeouts(),  # noqa: B008  # frozen, so one instance is safe to share
    ) -> None:
        self._source = source
        self._name = mask_url(source)
        self._fps = fps
        self._command = command
        self._timeouts = timeouts
        self._stopping = threading.Event()
        self._changed = threading.Condition()
        self._latest: tuple[int, bytes, int, int] | None = None
        self._number = 0
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop reading, and wait for ffmpeg to be gone."""
        self._stopping.set()
        with self._changed:
            self._changed.notify_all()
        self._thread.join()

    @property
    def frames_received(self) -> int:
        """Frames read since :meth:`start`, across every reconnection. The newest frame
        has this number."""
        with self._changed:
            return self._number

    def next_frame(self, after: int, timeout: float) -> tuple[int, Frame] | None:
        """The newest frame numbered above ``after``, with its number, or ``None`` if
        none arrives within ``timeout`` seconds."""
        with self._changed:
            arrived = self._changed.wait_for(
                lambda: (
                    (self._latest is not None and self._latest[0] > after)
                    or self._stopping.is_set()
                ),
                timeout,
            )
            if not arrived or self._latest is None or self._latest[0] <= after:
                return None
            number, raw, width, height = self._latest
        # Converted here rather than as frames arrive, so skipped frames cost nothing.
        return number, to_bgr(raw, width, height)

    def _run(self) -> None:
        first_backoff = min(1.0, self._timeouts.backoff)
        backoff = first_backoff
        reported: Failure | None = None
        while not self._stopping.is_set():
            try:
                with Stream(
                    self._source,
                    fps=self._fps,
                    command=self._command,
                    timeouts=self._timeouts,
                    cancel=self._stopping,
                ) as stream:
                    _log.info("camera %s: %dx%d", self._name, stream.width, stream.height)
                    while True:
                        raw = stream.read()
                        with self._changed:
                            self._number += 1
                            self._latest = (self._number, raw, stream.width, stream.height)
                            self._changed.notify_all()
                        backoff, reported = first_backoff, None
            except StreamError as exc:
                if self._stopping.is_set():
                    break
                # The same failure every half a minute for an hour is one line, not 120.
                # Compared by kind: ffmpeg's own words differ from one run to the next.
                if exc.failure is not reported:
                    _log.warning(
                        "camera %s: %s; trying again, every %.0f s at most",
                        self._name,
                        exc,
                        self._timeouts.backoff,
                    )
                    reported = exc.failure
                self._stopping.wait(backoff)
                backoff = min(backoff * 2, self._timeouts.backoff)
