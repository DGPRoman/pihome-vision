"""When each zone starts and stops being occupied, and when each line is crossed.

:class:`Watcher` takes one camera's detections, frame by frame, and says what changed:
a zone that became active, a zone that came clear, a line that somebody crossed.
Deciding what a light does about it is the next step's job, not this one's.

A zone is a state. It becomes active once something it watches for has been inside for
``min_seconds``, and clear once it has been empty for ``clear_seconds``. A line is an
instant: crossing it is one event, with nothing to end.

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
from pihome_vision.track import Position, Track, Tracker

#: How far past a line, in fractions of the frame, an object has to be before it counts
#: as being on that side. A box wobbles by a few pixels from frame to frame, and a
#: person standing on the line must not cross it on every wobble.
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


def _reaches(start: Position, end: Position, a: Position, b: Position) -> bool:
    """Whether the step from ``a`` to ``b``, which goes from one side of the line to the
    other, passes between its two ends rather than beyond one of them."""
    return offset(a, b, start) * offset(a, b, end) <= 0


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
            and inside(track.position, self.trigger.polygon)
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


class LineWatch:
    """Crossings of one line."""

    def __init__(self, trigger: LineTrigger) -> None:
        self.trigger = trigger
        # For each track, the side it was last clearly on (1 right, -1 left) and where.
        self._sides: dict[int, tuple[int, Position]] = {}

    def update(self, tracks: Iterable[Track], now: float) -> list[Event]:
        start, end = self.trigger.points
        events: list[Event] = []
        sides: dict[int, tuple[int, Position]] = {}
        for track in tracks:
            if track.category not in self.trigger.classes:
                continue
            last = self._sides.get(track.id)
            distance = offset(start, end, track.position)
            if abs(distance) < LINE_MARGIN:
                # On the line, or close enough to wobble across it: wait and see.
                if last is not None:
                    sides[track.id] = last
                continue
            side = 1 if distance > 0 else -1
            sides[track.id] = (side, track.position)
            if (
                last is not None
                and last[0] != side
                and _reaches(start, end, last[1], track.position)
                and self._counts(side)
            ):
                events.append(Event(self.trigger.id, "crossed", now))
        self._sides = sides
        return events

    def _counts(self, arrived_on: int) -> bool:
        match self.trigger.direction:
            case "left_to_right":
                return arrived_on > 0
            case "right_to_left":
                return arrived_on < 0
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
