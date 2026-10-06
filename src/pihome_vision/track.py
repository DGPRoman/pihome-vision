"""Following each person and vehicle from one frame to the next.

The detector sees boxes, one frame at a time. A trigger needs more: that the person by
the gate now is the one who was on the pavement a moment ago, and how long they have
been in the driveway. :class:`Tracker` gives each object an id that it keeps from frame
to frame, by matching every box to the nearest track of the same kind.

A track is followed by where it stands, the middle of the bottom edge of its box, and
keeps the box itself for the triggers: any part of somebody counts as being in a zone
or on a line. Both are in fractions of the frame, which keeps a camera's triggers where
they were when its resolution changes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from pihome_vision.config import ObjectClass
from pihome_vision.detect import Detection

#: ``x, y`` in fractions of the frame, from the top left.
Position = tuple[float, float]

#: ``left, top, right, bottom`` in fractions of the frame.
Box = tuple[float, float, float, float]

#: How far, in fractions of the frame, a box may be from where a track was expected to
#: be and still be matched to it. Further than that, it is somebody else.
MAX_STEP: Final = 0.15

#: How long a track outlives its last match, in seconds. A detector misses a person
#: for a frame or two now and then, and they should not come back as a stranger.
MAX_AGE: Final = 1.5

#: How much of each new step goes into a track's velocity; the rest is its history.
SMOOTHING: Final = 0.5

#: How far inside the frame's edge an object cut off by it stands, in fractions.
EDGE: Final = 0.001


@dataclass(slots=True)
class Track:
    """One object followed across frames."""

    id: int
    category: ObjectClass
    position: Position
    #: When it was last matched. Until :data:`MAX_AGE` after that, it stays where it was.
    seen_at: float
    #: Smoothed, in fractions of the frame a second.
    velocity: tuple[float, float] = (0.0, 0.0)
    #: The whole of it as last seen. None for a track made without one, which is then
    #: no bigger than its position.
    box: Box | None = None

    @property
    def extent(self) -> Box:
        """Its box, or its position as a box with no size."""
        if self.box is not None:
            return self.box
        x, y = self.position
        return x, y, x, y

    def expected_at(self, now: float) -> Position:
        """Where it should be by ``now``, if it carried on as it was going."""
        elapsed = now - self.seen_at
        return (
            self.position[0] + self.velocity[0] * elapsed,
            self.position[1] + self.velocity[1] * elapsed,
        )


def footing(detection: Detection, width: int, height: int) -> Position:
    """Where ``detection`` stands, in fractions of a ``width`` by ``height`` frame.

    A box the frame cuts off, somebody close enough that their feet are out of view,
    stands just inside the frame's edge rather than on it: a point on a zone's edge is
    outside it, and a zone drawn to the edge of the frame should hold them.
    """
    x = (detection.x + detection.width / 2) / width
    y = (detection.y + detection.height) / height
    return min(max(x, EDGE), 1.0 - EDGE), min(max(y, EDGE), 1.0 - EDGE)


def outline(detection: Detection, width: int, height: int) -> Box:
    """``detection``'s box in fractions of a ``width`` by ``height`` frame, cut to the
    frame."""

    def fit(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    return (
        fit(detection.x / width),
        fit(detection.y / height),
        fit((detection.x + detection.width) / width),
        fit((detection.y + detection.height) / height),
    )


class Tracker:
    """Matches each frame's detections to the tracks from the frames before.

    Nothing here reads a clock: every update is told the time, so a test can play a
    minute in a millisecond.
    """

    def __init__(self, *, max_step: float = MAX_STEP, max_age: float = MAX_AGE) -> None:
        self._max_step = max_step
        self._max_age = max_age
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        """Every track as of the last update."""
        return list(self._tracks.values())

    def update(
        self, detections: Sequence[Detection], width: int, height: int, now: float
    ) -> list[Track]:
        """Every track after this frame: matched ones moved, new ones started, and
        unmatched ones left where they were until :data:`MAX_AGE` has passed."""
        positions = [footing(detection, width, height) for detection in detections]
        # Closest pairs first, so two people walking side by side are not swapped
        # because one of them happened to be detected first.
        candidates = sorted(
            (distance, track_id, index)
            for track_id, track in self._tracks.items()
            for index, (detection, position) in enumerate(zip(detections, positions, strict=True))
            if detection.category == track.category
            and (distance := math.dist(track.expected_at(now), position)) <= self._max_step
        )
        matched: set[int] = set()
        placed: set[int] = set()
        for _, track_id, index in candidates:
            if track_id in matched or index in placed:
                continue
            track = self._tracks[track_id]
            self._move(track, positions[index], now)
            track.box = outline(detections[index], width, height)
            matched.add(track_id)
            placed.add(index)

        for index, detection in enumerate(detections):
            if index not in placed:
                track = Track(
                    self._next_id,
                    detection.category,
                    positions[index],
                    now,
                    box=outline(detection, width, height),
                )
                self._tracks[track.id] = track
                self._next_id += 1

        for track_id in [i for i, t in self._tracks.items() if now - t.seen_at > self._max_age]:
            del self._tracks[track_id]
        return list(self._tracks.values())

    @staticmethod
    def _move(track: Track, position: Position, now: float) -> None:
        elapsed = now - track.seen_at
        if elapsed > 0:
            step = (
                (position[0] - track.position[0]) / elapsed,
                (position[1] - track.position[1]) / elapsed,
            )
            track.velocity = (
                SMOOTHING * step[0] + (1 - SMOOTHING) * track.velocity[0],
                SMOOTHING * step[1] + (1 - SMOOTHING) * track.velocity[1],
            )
        track.position = position
        track.seen_at = now
