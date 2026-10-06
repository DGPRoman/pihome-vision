"""The two windows: ``edit``, for drawing zones and lines, and ``preview``, for watching
them work on the live picture.

These need the OpenCV build with windows (the ``gui`` extra). The service does not, and
never imports this module.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from typing import Final

import cv2

from pihome_vision.config import Camera, Trigger
from pihome_vision.detect import Frame
from pihome_vision.overlay import Scene, draw
from pihome_vision.service import Detects, Frames
from pihome_vision.sketch import ESCAPE, Sketch, fraction
from pihome_vision.triggers import Watcher

WINDOW: Final = "pihome-vision"

#: How long a crossed line stays highlighted, in seconds.
CROSSED_SECONDS: Final = 1.0

#: How long a window waits for a key between redraws, in milliseconds.
_KEY_WAIT: Final = 30

#: The largest a window opens, in screen pixels. A bigger frame is shown smaller, and
#: the window can be resized from there.
_LARGEST: Final = (1280, 800)


class NoWindowError(Exception):
    """No window can be opened here. The message says why."""


def _open() -> None:
    # Qt, which the OpenCV wheels draw windows with on Linux, ends the whole process
    # when there is no display rather than raising anything.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        msg = "there is no display to open a window on; run this from a desktop session"
        raise NoWindowError(msg)
    try:
        # No toolbar: its zoom and pan would get in the way of clicking points.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    except cv2.error:
        msg = (
            "this OpenCV has no windows: install requirements/gui.txt in place of"
            " requirements/headless.txt"
        )
        raise NoWindowError(msg) from None


def _fit(width: int, height: int) -> None:
    """Size the window to a ``width`` by ``height`` frame, or as near as fits.

    A resizable window otherwise opens at a size of its own, whatever it shows.
    """
    scale = min(1.0, _LARGEST[0] / width, _LARGEST[1] / height)
    cv2.resizeWindow(WINDOW, round(width * scale), round(height * scale))


def _closed() -> bool:
    """Whether the window has been closed from its title bar."""
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1


def _key() -> int | None:
    key = cv2.waitKey(_KEY_WAIT)
    return None if key < 0 else key & 0xFF


def edit(frame: Frame, triggers: list[Trigger]) -> list[Trigger]:
    """Let somebody draw over ``frame``, starting from ``triggers``, until they finish
    or close the window. Returns what they drew."""
    sketch = Sketch(triggers)
    height, width = frame.shape[:2]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            sketch.click(fraction(x, y, width, height))

    _open()
    try:
        _fit(width, height)
        cv2.setMouseCallback(WINDOW, on_mouse)
        while not sketch.done:
            scene = Scene(sketch.triggers, sketch=sketch.points, status=sketch.status)
            shown = draw(frame, scene)
            cv2.imshow(WINDOW, shown)
            if (key := _key()) is not None:
                sketch.key(key)
            if _closed():
                break
    finally:
        cv2.destroyWindow(WINDOW)
    return sketch.triggers


def preview(
    camera: Camera,
    frames: Frames,
    detector: Detects,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Show the camera's live picture with everything the service would see in it,
    until ``q`` or Esc is pressed or the window is closed. Nothing is switched."""
    watcher = Watcher(camera)
    crossed: dict[str, float] = {}
    last = 0
    shown_any = False
    _open()
    try:
        while True:
            got = frames.next_frame(last, 0.0)
            if got is not None:
                last, frame = got
                started = time.perf_counter()
                detections = detector.detect(frame)
                took = time.perf_counter() - started
                now = clock()
                height, width = frame.shape[:2]
                if not shown_any:
                    _fit(width, height)
                for event in watcher.update(detections, width, height, now):
                    if event.change == "crossed":
                        crossed[event.trigger] = now
                lit = {line for line, at in crossed.items() if now - at < CROSSED_SECONDS}
                active = ", ".join(sorted(watcher.active)) or "none"
                status = f"{took * 1000:.0f} ms;  active: {active};  q: quit"
                cv2.imshow(
                    WINDOW,
                    draw(
                        frame,
                        Scene(
                            camera.triggers,
                            active=watcher.active,
                            crossed=lit,
                            detections=detections,
                            tracks=watcher.tracks,
                            status=status,
                        ),
                    ),
                )
                shown_any = True
            key = _key()
            if key in (ord("q"), ESCAPE) or (shown_any and _closed()):
                return
    finally:
        cv2.destroyWindow(WINDOW)
