"""When each zone starts and stops being occupied, and when each line is crossed.

:class:`Watcher` takes one camera's detections, frame by frame, and says what changed:
a zone that became active, a zone that came clear, a line that somebody crossed.
Deciding what a light does about it is the next step's job, not this one's.

Any part of somebody counts, not only where they stand: an arm over a zone's edge is
in the zone, and a shoulder touching a line has crossed it.

A zone is a state. It becomes active once something it watches for has been in it for
``min_seconds``, and clear once it has been empty for ``clear_seconds``. A line is an
instant: reaching it is one event, with nothing to end, and the next takes leaving it
first.

A track the detector has lost for a moment stays where it was last seen (see
:mod:`pihome_vision.track`). That keeps a zone occupied, but it cannot make one
active: only something seen in the current frame can, or one false detection would
sit in the zone until its track expired and light the yard.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pihome_vision.config import Camera, LineTrigger, ZoneTrigger
from pihome_vision.detect import Detection
from pihome_vision.track import Box, Position, Track, Tracker

#: How far clear of a line, in fractions of the frame, something on it has to be before
#: it counts as having left. A box wobbles by a few pixels from frame to frame, and a
#: person standing at the line must not reach it on every wobble.
LINE_MARGIN: Final = 0.02

Change = Literal["active", "clear", "crossed"]


@dataclass(frozen=True, slots=True)
class Event:
    """A trigger changed: a zone became ``active`` or ``clear``, or a line was
    ``crossed``."""

    trigger: str
    change: Change
    at: float


def inside(point: Position, polygon: Sequence[Position]) -> bool:
    """Whether ``point`` is inside ``polygon``, by counting the edges a ray crosses."""
    x, y = point
    result = False
    for (x1, y1), (x2, y2) in zip(polygon, [*polygon[1:], polygon[0]], strict=True):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            result = not result
    return result


def offset(start: Position, end: Position, point: Position) -> float:
    """How far ``point`` is to the right of the line from ``start`` towards ``end``,
    seen standing on ``start``. Negative is to the left.

    Frame coordinates grow downwards, which is what makes a positive cross product
    the right-hand side here rather than the left.
    """
    (x1, y1), (x2, y2), (x, y) = start, end, point
    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return cross / math.dist(start, end)


def _orientation(a: Position, b: Position, c: Position) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _within(box: Box, point: Position) -> bool:
    left, top, right, bottom = box
    return left <= point[0] <= right and top <= point[1] <= bottom


def intersect(a: Position, b: Position, c: Position, d: Position) -> bool:
    """Whether the segments from ``a`` to ``b`` and from ``c`` to ``d`` meet, ends and
    overlaps included."""
    d1, d2 = _orientation(c, d, a), _orientation(c, d, b)
    d3, d4 = _orientation(a, b, c), _orientation(a, b, d)
    if ((d1 > 0) != (d2 > 0) and d1 != 0 and d2 != 0) and (
        (d3 > 0) != (d4 > 0) and d3 != 0 and d4 != 0
    ):
        return True

    def on(p: Position, q: Position, r: Position) -> bool:
        return _orientation(p, q, r) == 0 and _within(
            (min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1])), r
        )

    return on(c, d, a) or on(c, d, b) or on(a, b, c) or on(a, b, d)


def _edges(box: Box) -> list[tuple[Position, Position]]:
    left, top, right, bottom = box
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    return list(zip(corners, [*corners[1:], corners[0]], strict=True))


def overlaps(box: Box, polygon: Sequence[Position]) -> bool:
    """Whether any of ``box`` is in ``polygon``."""
    left, top, right, bottom = box
    if any(inside(corner, polygon) for corner in [(left, top), (right, bottom)]):
        return True
    if any(_within(box, vertex) for vertex in polygon):
        return True
    sides = list(zip(polygon, [*polygon[1:], polygon[0]], strict=True))
    return any(intersect(*edge, *side) for edge in _edges(box) for side in sides)


def touches(box: Box, start: Position, end: Position) -> bool:
    """Whether any of ``box`` is on the line from ``start`` to ``end``."""
    if _within(box, start) or _within(box, end):
        return True
    return any(intersect(*edge, start, end) for edge in _edges(box))


def _grown(box: Box, by: float) -> Box:
    left, top, right, bottom = box
    return left - by, top - by, right + by, bottom + by


def _centre(box: Box) -> Position:
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def _spanning(a: Box, b: Box) -> Box:
    """The box both fit in: roughly what one swept through on its way to the other."""
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


class ZoneWatch:
    """Whether one zone is occupied."""

    def __init__(self, trigger: ZoneTrigger) -> None:
        self.trigger = trigger
        self.active = False
        self._entered: dict[int, float] = {}
        self._occupied_at = 0.0

    def update(self, tracks: Iterable[Track], now: float) -> Event | None:
        here = {
            track.id
            for track in tracks
            if track.category in self.trigger.classes
            and overlaps(track.extent, self.trigger.polygon)
        }
        # Time inside counts from when each one arrived, and starts again if it leaves.
        self._entered = {i: self._entered.get(i, now) for i in here}
        if here:
            self._occupied_at = now
        if not self.active:
            seen = {track.id for track in tracks if track.seen_at == now}
            if any(
                now - since >= self.trigger.min_seconds
                for i, since in self._entered.items()
                if i in seen
            ):
                self.active = True
                return Event(self.trigger.id, "active", now)
        elif now - self._occupied_at >= self.trigger.clear_seconds:
            self.active = False
            return Event(self.trigger.id, "clear", now)
        return None


@dataclass(frozen=True, slots=True)
class _Near:
    """Where one track was relative to a line, as of the last frame."""

    #: Touching the line, or not yet clear of it again since it did.
    on: bool
    #: The side it was last clearly on, 1 right and -1 left, if it has been on one.
    side: int | None
    box: Box


class LineWatch:
    """Each time something reaches a line."""

    def __init__(self, trigger: LineTrigger) -> None:
        self.trigger = trigger
        self._near: dict[int, _Near] = {}

    def update(self, tracks: Iterable[Track], now: float) -> list[Event]:
        start, end = self.trigger.points
        events: list[Event] = []
        near: dict[int, _Near] = {}
        for track in tracks:
            if track.category not in self.trigger.classes:
                continue
            box = track.extent
            last = self._near.get(track.id)
            # Once on the line, it has to get clear of it to leave, or a box wobbling
            # at the line would reach it again on every wobble.
            reach = LINE_MARGIN if last is not None and last.on else 0.0
            if touches(_grown(box, reach), start, end):
                if last is None or not last.on:
                    came_from = None if last is None else last.side
                    if self._counts(came_from):
                        events.append(Event(self.trigger.id, "crossed", now))
                near[track.id] = _Near(on=True, side=None if last is None else last.side, box=box)
                continue
            side = 1 if offset(start, end, _centre(box)) > 0 else -1
            # Between two frames it may have gone right over the line.
            if (
                last is not None
                and not last.on
                and last.side is not None
                and last.side != side
                and touches(_spanning(last.box, box), start, end)
                and self._counts(last.side)
            ):
                events.append(Event(self.trigger.id, "crossed", now))
            near[track.id] = _Near(on=False, side=side, box=box)
        self._near = near
        return events

    def _counts(self, came_from: int | None) -> bool:
        """Whether reaching the line from ``came_from`` counts: from the left is
        ``left_to_right``, and from nowhere known counts only for ``any``."""
        match self.trigger.direction:
            case "left_to_right":
                return came_from == -1
            case "right_to_left":
                return came_from == 1
            case "any":
                return True


class Watcher:
    """Every trigger on one camera, fed one frame's detections at a time."""

    def __init__(self, camera: Camera, tracker: Tracker | None = None) -> None:
        self._tracker = tracker or Tracker()
        self._zones = [ZoneWatch(t) for t in camera.triggers if isinstance(t, ZoneTrigger)]
        self._lines = [LineWatch(t) for t in camera.triggers if isinstance(t, LineTrigger)]

    @property
    def active(self) -> frozenset[str]:
        """The zones occupied right now."""
        return frozenset(zone.trigger.id for zone in self._zones if zone.active)

    @property
    def tracks(self) -> list[Track]:
        """Everything followed as of the last frame, for drawing."""
        return self._tracker.tracks

    def update(
        self, detections: Sequence[Detection], width: int, height: int, now: float
    ) -> list[Event]:
        """What changed with this frame, which arrived at ``now``, in seconds on any
        clock that only goes forward."""
        tracks = self._tracker.update(detections, width, height, now)
        events = [event for zone in self._zones if (event := zone.update(tracks, now))]
        for line in self._lines:
            events += line.update(tracks, now)
        return events
