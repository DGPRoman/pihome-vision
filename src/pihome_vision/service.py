"""The ``run`` service: a camera's frames through the model, the tracker and the
triggers, to the lights.

One :class:`Pipeline` per camera, each in a thread of its own. The camera reads frames
in its own thread too, and the pipeline always takes the newest, so a detector slower
than the camera skips frames rather than falling behind. :func:`serve` runs them until
SIGTERM or SIGINT, then switches off every light the service switched on. Under
systemd it says when it is ready, and keeps saying it is alive while the pipeline moves.

A camera that is ``only_after_dark`` is closed by day, and the model is not run. The
pipeline asks the sun about it every :data:`SUN_CHECK`, and in between goes on as it
would for a camera that has gone quiet, so that at sunrise its zones come clear and its
lights go off in the usual time.
"""

from __future__ import annotations

import logging
import math
import signal
import threading
import time
from collections.abc import Callable
from typing import Final, Protocol

import cv2
import numpy as np

from pihome_vision import systemd
from pihome_vision.camera import Picture
from pihome_vision.config import EXIT_CONFIGURATION_ERROR, Camera
from pihome_vision.detect import Detection, Frame, ModelError
from pihome_vision.lights import Darkness, Lights
from pihome_vision.triggers import Watcher

#: Seconds to wait for a frame before letting time pass without one. A camera that has
#: gone quiet must not hold a zone, and so a light, on for ever.
FRAME_WAIT: Final = 1.0

#: The longest a still picture may go without being detected on, in seconds, however
#: little it changes.
MAX_SKIP: Final = 5.0

#: Seconds between log lines saying how the pipeline is doing.
STATS_SECONDS: Final = 60.0

#: How often a camera watched only after dark asks whether it is, in seconds. Twilight
#: lasts half an hour after sunset, so a minute late to notice it is in good time.
SUN_CHECK: Final = 60.0

#: How long stopping waits for the pipeline to finish its step, in seconds. A step takes
#: at most :data:`FRAME_WAIT` and one model run; one that takes longer is stuck.
STOP_GRACE: Final = 5.0

#: How often the main thread looks in on the pipeline, in seconds, unless systemd's
#: watchdog asks for more.
_TICK: Final = 0.5

#: What a frame's brightness is shrunk to, to tell whether it has changed. Small enough
#: that sensor noise averages out and comparing costs nothing next to the model.
_THUMBNAIL: Final = (160, 90)

#: Grey levels in one step of brightness: :meth:`Picture.brightness` spans 16 to 235,
#: and ``motion_threshold`` is in levels of grey from 0 to 255.
_GREY_PER_STEP: Final = 255 / 219

_log = logging.getLogger(__name__)


class Frames(Protocol):
    """Where a pipeline's frames come from: a :class:`~pihome_vision.camera.Camera`."""

    @property
    def frames_received(self) -> int: ...

    def next_frame(self, after: int, timeout: float) -> tuple[int, Picture] | None: ...


class Source(Frames, Protocol):
    """Frames to be had only between :meth:`start` and :meth:`stop`, which can be
    called again in turn: a :class:`~pihome_vision.camera.Camera`."""

    def start(self) -> None: ...

    def stop(self) -> None: ...


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

    def changed(self, picture: Picture, now: float) -> bool:
        if self._threshold <= 0:
            return True
        # From the brightness alone, a third of the work of shrinking the colour picture.
        thumbnail = cv2.resize(picture.brightness(), _THUMBNAIL, interpolation=cv2.INTER_AREA)
        small = np.asarray(thumbnail, dtype=np.uint8)
        if (
            self._reference is None
            or now - self._at >= self._max_skip
            or float(cv2.absdiff(small, self._reference).mean()) * _GREY_PER_STEP >= self._threshold
        ):
            self._reference, self._at = small, now
            return True
        return False


class Pipeline:
    """One camera: each frame through the model, the tracker and its triggers, and
    what changed on to the lights.

    :meth:`run` opens the camera, and for one that is ``only_after_dark`` closes it at
    sunrise and opens it again at sunset, by ``darkness``. :meth:`close` closes it for
    good.
    """

    def __init__(  # noqa: PLR0913 - the four parts it joins, and two ways of telling time
        self,
        camera: Camera,
        frames: Source,
        detector: Detects,
        lights: Lights,
        *,
        darkness: Darkness | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera = camera
        self._frames = frames
        self._detector = detector
        self._lights = lights
        self._darkness = darkness
        self._clock = clock
        #: Whether the camera is open: None until :meth:`run` has first looked.
        self._watching: bool | None = None
        self._looked_at = -math.inf
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
            if self._watch():
                self.step()
            else:
                # Nothing to see with the camera closed: time passes as it would
                # waiting for a frame, and stopping need not wait for it.
                stop.wait(FRAME_WAIT)
                self._follow([], self._clock())
            self.stepped_at = time.monotonic()

    def close(self) -> None:
        """Close the camera, and wait for ffmpeg to be gone."""
        self._frames.stop()

    def step(self) -> None:
        """Take the next frame, or wait :data:`FRAME_WAIT` for one, and act on it."""
        got = self._frames.next_frame(self._last, FRAME_WAIT)
        now = self._clock()
        if got is None:
            # Nothing seen: tracks coast and expire, and zones come clear in time.
            detections: list[Detection] = []
        else:
            self._last, picture = got
            self._size = (picture.width, picture.height)
            self._examined += 1
            if self._stillness.changed(picture, now):
                frame = picture.bgr()
                started = time.perf_counter()
                self._previous = self._detector.detect(frame)
                self._inference += time.perf_counter() - started
                self._detected += 1
            # A still picture has what it had when the model last looked.
            detections = self._previous
        self._follow(detections, now)
        if now - self._stats_at >= STATS_SECONDS:
            self._report(now)

    def _watch(self) -> bool:
        """Whether the camera is to be watched now, opening or closing it to match."""
        now = self._clock()
        if self._watching is not None and now - self._looked_at < SUN_CHECK:
            return self._watching
        self._looked_at = now
        wanted = not self.camera.only_after_dark or (
            self._darkness is not None and self._darkness.is_dark()
        )
        if wanted is self._watching:
            return wanted
        if wanted:
            self._frames.start()
            # Counted from now, so the first figures after dusk are not spread over the day.
            self._count_from(now, self._frames.frames_received)
        elif self._watching:
            self._frames.stop()
        if self.camera.only_after_dark:
            if wanted:
                _log.info("%s: dark; watching", self.camera.id)
            else:
                _log.info("%s: daylight; not watching until dark", self.camera.id)
        self._watching = wanted
        return wanted

    def _follow(self, detections: list[Detection], now: float) -> None:
        """One frame's ``detections``, or none, through the triggers to the lights."""
        events = self._watcher.update(detections, *self._size, now)
        for event in events:
            _log.info("%s/%s: %s", self.camera.id, event.trigger, event.change)
        self._lights.update(events, self._watcher.active, now)

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
        self._count_from(now, received)

    def _count_from(self, now: float, received: int) -> None:
        self._stats_at, self._received = now, received
        self._examined = self._detected = 0
        self._inference = 0.0


def serve(pipeline: Pipeline, lights: Lights) -> int:
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
        thread.start()
        after_dark = " after dark" if pipeline.camera.only_after_dark else ""
        _log.info("watching camera %s%s", pipeline.camera.id, after_dark)
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
        pipeline.close()
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
