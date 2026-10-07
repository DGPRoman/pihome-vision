"""The ``run`` service: a camera's frames through the model, the tracker and the
triggers, to the lights.

One :class:`Pipeline` per camera, each in a thread of its own. The camera reads frames
in its own thread too, and the pipeline always takes the newest, so a detector slower
than the camera skips frames rather than falling behind. :func:`serve` runs them until
SIGTERM or SIGINT, then switches off every light the service switched on. Under
systemd it says when it is ready, and keeps saying it is alive while the pipeline moves.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable
from typing import Final, Protocol

import cv2
import numpy as np

from pihome_vision import systemd
from pihome_vision.config import Camera
from pihome_vision.detect import Detection, Frame, ModelError
from pihome_vision.lights import Lights
from pihome_vision.triggers import Watcher

#: Seconds to wait for a frame before letting time pass without one. A camera that has
#: gone quiet must not hold a zone, and so a light, on for ever.
FRAME_WAIT: Final = 1.0

#: The longest a still picture may go without being detected on, in seconds, however
#: little it changes.
MAX_SKIP: Final = 5.0

#: Seconds between log lines saying how the pipeline is doing.
STATS_SECONDS: Final = 60.0

#: How long stopping waits for the pipeline to finish its step, in seconds. A step takes
#: at most :data:`FRAME_WAIT` and one model run; one that takes longer is stuck.
STOP_GRACE: Final = 5.0

#: How often the main thread looks in on the pipeline, in seconds, unless systemd's
#: watchdog asks for more.
_TICK: Final = 0.5

#: What a frame is shrunk to, grey, to tell whether it has changed. Small enough that
#: sensor noise averages out and comparing costs nothing next to the model.
_THUMBNAIL: Final = (160, 90)

#: Exit status for a configuration the service cannot run with, as the CLI's.
EXIT_CONFIGURATION_ERROR: Final = 2

_log = logging.getLogger(__name__)


class Frames(Protocol):
    """Where a pipeline's frames come from: a :class:`~pihome_vision.camera.Camera`."""

    @property
    def frames_received(self) -> int: ...

    def next_frame(self, after: int, timeout: float) -> tuple[int, Frame] | None: ...


class Detects(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...


class Stillness:
    """Whether a frame has changed enough since the last one detected on to be worth
    detecting on again.

    Compared with the last frame the model saw rather than the one before, so a scene
    that changes slowly still crosses the threshold in the end.
    """

    def __init__(self, threshold: float, *, max_skip: float = MAX_SKIP) -> None:
        self._threshold = threshold
        self._max_skip = max_skip
        self._reference: Frame | None = None
        self._at = 0.0

    def changed(self, frame: Frame, now: float) -> bool:
        if self._threshold <= 0:
            return True
        thumbnail = cv2.resize(frame, _THUMBNAIL, interpolation=cv2.INTER_AREA)
        small = np.asarray(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        if (
            self._reference is None
            or now - self._at >= self._max_skip
            or float(cv2.absdiff(small, self._reference).mean()) >= self._threshold
        ):
            self._reference, self._at = small, now
            return True
        return False


class Pipeline:
    """One camera: each frame through the model, the tracker and its triggers, and
    what changed on to the lights."""

    def __init__(
        self,
        camera: Camera,
        frames: Frames,
        detector: Detects,
        lights: Lights,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera = camera
        self._frames = frames
        self._detector = detector
        self._lights = lights
        self._clock = clock
        self._watcher = Watcher(camera)
        self._stillness = Stillness(camera.motion_threshold)
        self._last = 0
        self._size = (1, 1)
        self._previous: list[Detection] = []
        self._stats_at = clock()
        self._received = 0
        self._examined = 0
        self._detected = 0
        self._inference = 0.0
        #: When the last step ended, on the monotonic clock whatever ``clock`` is: what
        #: tells a pipeline that is moving from one that is stuck.
        self.stepped_at = time.monotonic()

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.step()
            self.stepped_at = time.monotonic()

    def step(self) -> None:
        """Take the next frame, or wait :data:`FRAME_WAIT` for one, and act on it."""
        got = self._frames.next_frame(self._last, FRAME_WAIT)
        now = self._clock()
        if got is None:
            # Nothing seen: tracks coast and expire, and zones come clear in time.
            detections: list[Detection] = []
        else:
            self._last, frame = got
            height, width = frame.shape[:2]
            self._size = (width, height)
            self._examined += 1
            if self._stillness.changed(frame, now):
                started = time.perf_counter()
                self._previous = self._detector.detect(frame)
                self._inference += time.perf_counter() - started
                self._detected += 1
            # A still picture has what it had when the model last looked.
            detections = self._previous
        events = self._watcher.update(detections, *self._size, now)
        for event in events:
            _log.info("%s/%s: %s", self.camera.id, event.trigger, event.change)
        self._lights.update(events, self._watcher.active, now)
        if now - self._stats_at >= STATS_SECONDS:
            self._report(now)

    def _report(self, now: float) -> None:
        elapsed = now - self._stats_at
        received = self._frames.frames_received
        took = f" at {self._inference / self._detected * 1000:.0f} ms" if self._detected else ""
        _log.info(
            "%s: %.1f frames/s from the camera, %.1f examined, %.1f detected on%s; "
            "active: %s; lit: %s",
            self.camera.id,
            (received - self._received) / elapsed,
            self._examined / elapsed,
            self._detected / elapsed,
            took,
            ", ".join(sorted(self._watcher.active)) or "none",
            ", ".join(sorted(self._lights.lit)) or "none",
        )
        self._stats_at, self._received = now, received
        self._examined = self._detected = 0
        self._inference = 0.0


class _Source(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


def serve(pipeline: Pipeline, camera: _Source, lights: Lights) -> int:
    """Run ``pipeline`` until SIGTERM or SIGINT, and return the exit status.

    On the way out the lights are switched off before the camera is closed, which can
    take a few seconds, so nothing waits on ffmpeg to leave a light dark. A pipeline
    that does not stop within :data:`STOP_GRACE` is left behind, and the status is 1.
    """
    stop = threading.Event()
    failures: list[Exception] = []
    watchdog = systemd.watchdog_seconds()
    # At least twice in each of the watchdog's intervals, with time to spare.
    tick = _TICK if watchdog is None else min(_TICK, watchdog / 4)
    stuck = False

    def work() -> None:
        try:
            pipeline.run(stop)
        except Exception as exc:  # reported below, from the main thread
            failures.append(exc)
            stop.set()

    def stopping(signum: int, _: object) -> None:
        _log.info("%s: stopping", signal.Signals(signum).name)
        stop.set()

    previous = {sig: signal.signal(sig, stopping) for sig in (signal.SIGTERM, signal.SIGINT)}
    thread = threading.Thread(target=work, name=f"pipeline {pipeline.camera.id}", daemon=True)
    try:
        lights.start()
        camera.start()
        thread.start()
        _log.info("watching camera %s", pipeline.camera.id)
        systemd.notify("READY=1")
        while not stop.wait(tick):
            # Only while the pipeline moves: systemd restarts one stuck for good.
            if watchdog is not None and time.monotonic() - pipeline.stepped_at < watchdog / 2:
                systemd.notify("WATCHDOG=1")
    finally:
        stop.set()
        systemd.notify("STOPPING=1")
        if thread.is_alive():
            thread.join(STOP_GRACE)
            stuck = thread.is_alive()
        if stuck:
            _log.error(
                "the pipeline has not moved for %.0f s; stopping without it",
                time.monotonic() - pipeline.stepped_at,
            )
        lights.stop()
        camera.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    for exc in failures:
        if isinstance(exc, ModelError):
            _log.error("%s", exc)
            return EXIT_CONFIGURATION_ERROR
        _log.error("the pipeline failed", exc_info=exc)
        return 1
    if stuck:
        return 1
    _log.info("stopped")
    return 0
