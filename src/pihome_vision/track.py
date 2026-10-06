"""Following each person and vehicle from one frame to the next.

The detector sees boxes, one frame at a time. A trigger needs more: that the person by
the gate now is the one who was on the pavement a moment ago, and how long they have
been in the driveway. :class:`Tracker` gives each object an id that it keeps from frame
to frame, by matching every box to the nearest track of the same kind.

A position is where an object stands: the middle of the bottom edge of its box, in
fractions of the frame. Zones and lines are drawn on the ground, and the middle of a
person's box is a metre above it. Fractions keep a camera's triggers where they were
when its resolution changes.
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

#: How far, in fractions of the frame, a box may be from where a track was expected to
#: be and still be matched to it. Further than that, it is somebody else.
MAX_STEP: Final = 0.15

#: How long a track outlives its last match, in seconds. A detector misses a person
#: for a frame or two now and then, and they should not come back as a stranger.
MAX_AGE: Final = 1.5

#: How much of each new step goes into a track's velocity; the rest is its history.
SMOOTHING: Final = 0.5


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

    def expected_at(self, now: float) -> Position:
        """Where it should be by ``now``, if it carried on as it was going."""
        elapsed = now - self.seen_at
        return (
            self.position[0] + self.velocity[0] * elapsed,
            self.position[1] + self.velocity[1] * elapsed,
        )


def footing(detection: Detection, width: int, height: int) -> Position:
    """Where ``detection`` stands, in fractions of a ``width`` by ``height`` frame."""
    x = (detection.x + detection.width / 2) / width
    y = (detection.y + detection.height) / height
    return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)


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
            self._move(self._tracks[track_id], positions[index], now)
            matched.add(track_id)
            placed.add(index)

        for index, detection in enumerate(detections):
            if index not in placed:
                track = Track(self._next_id, detection.category, positions[index], now)
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
