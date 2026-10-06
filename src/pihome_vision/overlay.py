"""Zones, lines, detections and tracks drawn over a frame, for people to look at.

Only :mod:`pihome_vision.gui` shows the result, but the drawing needs no window, so it
is tested with the same headless OpenCV the service runs on.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from pihome_vision.config import LineTrigger, Point, Trigger, ZoneTrigger
from pihome_vision.detect import Detection, Frame
from pihome_vision.track import Track

# Colours are BGR, as OpenCV takes them.
ZONE: Final = (0, 200, 255)
ZONE_ACTIVE: Final = (0, 220, 0)
LINE: Final = (255, 160, 0)
LINE_CROSSED: Final = (0, 0, 255)
DETECTION: Final = (255, 255, 255)
TRACK: Final = (255, 0, 255)
SKETCH: Final = (0, 255, 255)
TEXT_BACKGROUND: Final = (0, 0, 0)

#: How strongly an active zone is filled in, from 0 to 1.
_FILL: Final = 0.3

#: The frame width at which lines are one pixel thick and text is at half its size.
#: Wider frames get thicker lines and bigger text, so that they can still be seen in a
#: window smaller than the frame.
_REFERENCE_WIDTH: Final = 480

_FONT: Final = cv2.FONT_HERSHEY_SIMPLEX


def _pixel(point: Point, width: int, height: int) -> tuple[int, int]:
    return round(point[0] * width), round(point[1] * height)


class _Pen:
    """Line thickness and text size to suit a frame's size, so a 4K picture shrunk
    into a window is as readable as a small one."""

    def __init__(self, image: Frame) -> None:
        self.image = image
        self.height, self.width = image.shape[:2]
        scale = max(self.width / _REFERENCE_WIDTH, 1.0)
        self.thickness = max(round(scale), 1)
        self.text_scale = 0.5 * scale

    def pixel(self, point: Point) -> tuple[int, int]:
        return _pixel(point, self.width, self.height)

    def label(self, text: str, at: tuple[int, int], colour: tuple[int, int, int]) -> None:
        (width, height), baseline = cv2.getTextSize(text, _FONT, self.text_scale, self.thickness)
        x = min(max(at[0], 0), max(self.width - width, 0))
        y = min(max(at[1], height + baseline), self.height)
        cv2.rectangle(
            self.image, (x, y - height - baseline), (x + width, y), TEXT_BACKGROUND, cv2.FILLED
        )
        cv2.putText(
            self.image,
            text,
            (x, y - baseline),
            _FONT,
            self.text_scale,
            colour,
            self.thickness,
            cv2.LINE_AA,
        )


@dataclass(frozen=True, slots=True)
class Scene:
    """What to draw over a frame."""

    triggers: Sequence[Trigger]
    #: The zones occupied now.
    active: Collection[str] = ()
    #: The lines crossed a moment ago.
    crossed: Collection[str] = ()
    detections: Sequence[Detection] = ()
    tracks: Sequence[Track] = ()
    #: A shape still being drawn.
    sketch: Sequence[Point] = ()
    #: A line of text along the bottom.
    status: str = ""


def draw(frame: Frame, scene: Scene) -> Frame:
    """A copy of ``frame`` with ``scene`` drawn over it."""
    image = np.ascontiguousarray(frame.copy())
    pen = _Pen(image)
    zones = [t for t in scene.triggers if isinstance(t, ZoneTrigger)]
    if filled := [zone for zone in zones if zone.id in scene.active]:
        shade = image.copy()
        for zone in filled:
            points = np.array([pen.pixel(p) for p in zone.polygon], dtype=np.int32)
            cv2.fillPoly(shade, [points], ZONE_ACTIVE)
        cv2.addWeighted(shade, _FILL, image, 1 - _FILL, 0, dst=image)
    for zone in zones:
        colour = ZONE_ACTIVE if zone.id in scene.active else ZONE
        points = np.array([pen.pixel(p) for p in zone.polygon], dtype=np.int32)
        cv2.polylines(image, [points], isClosed=True, color=colour, thickness=pen.thickness)
        pen.label(zone.id, pen.pixel(zone.polygon[0]), colour)
    for line in (t for t in scene.triggers if isinstance(t, LineTrigger)):
        _line(pen, line, crossed=line.id in scene.crossed)

    for detection in scene.detections:
        top_left = (round(detection.x), round(detection.y))
        bottom_right = (
            round(detection.x + detection.width),
            round(detection.y + detection.height),
        )
        cv2.rectangle(image, top_left, bottom_right, DETECTION, pen.thickness)
        pen.label(f"{detection.category} {detection.confidence:.2f}", top_left, DETECTION)
    for track in scene.tracks:
        at = pen.pixel(track.position)
        cv2.circle(image, at, 3 * pen.thickness, TRACK, cv2.FILLED)
        pen.label(str(track.id), (at[0] + 4 * pen.thickness, at[1]), TRACK)

    if scene.sketch:
        points = np.array([pen.pixel(p) for p in scene.sketch], dtype=np.int32)
        cv2.polylines(image, [points], isClosed=False, color=SKETCH, thickness=pen.thickness)
        for point in points:
            cv2.circle(image, (int(point[0]), int(point[1])), 3 * pen.thickness, SKETCH, -1)
    if scene.status:
        pen.label(scene.status, (0, pen.height), (255, 255, 255))
    return image


def _line(pen: _Pen, line: LineTrigger, *, crossed: bool) -> None:
    """A line, with an arrow to the side that counts when only one way across does."""
    colour = LINE_CROSSED if crossed else LINE
    start, end = (pen.pixel(p) for p in line.points)
    cv2.line(pen.image, start, end, colour, 2 * pen.thickness)
    cv2.circle(pen.image, start, 3 * pen.thickness, colour, cv2.FILLED)
    pen.label(line.id, start, colour)
    if line.direction == "any":
        return
    # The right-hand side, standing on the first point looking towards the second:
    # with y growing downwards, (dx, dy) turned to (-dy, dx).
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    side = 1 if line.direction == "left_to_right" else -1
    reach = 20 * pen.thickness
    middle = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    tip = (
        round(middle[0] - side * dy / length * reach),
        round(middle[1] + side * dx / length * reach),
    )
    cv2.arrowedLine(pen.image, middle, tip, colour, pen.thickness, tipLength=0.4)
