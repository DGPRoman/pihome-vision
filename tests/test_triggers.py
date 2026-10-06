from __future__ import annotations

from collections.abc import Iterable, Sequence

import pytest

from pihome_vision.config import Camera, ObjectClass
from pihome_vision.detect import Detection
from pihome_vision.triggers import LINE_MARGIN, Event, Watcher, inside, offset

WIDTH, HEIGHT = 1000, 500

#: Five frames a second, as a camera is set up by default.
STEP = 0.2

#: A box in the middle of the frame, ``x`` 0.2 to 0.6 and ``y`` 0.2 to 0.8.
SQUARE = [[0.2, 0.2], [0.6, 0.2], [0.6, 0.8], [0.2, 0.8]]

#: Upright at ``x`` 0.5, from ``y`` 0.8 up to 0.2. Seen from its first point looking at
#: its second, which is up the frame, left and right are the frame's own.
UPRIGHT = [[0.5, 0.8], [0.5, 0.2]]

#: A walk from left to right across UPRIGHT, a twentieth of the frame a step, and the
#: same walk back. Each is clear of the line on the far side at the frame after the
#: step across it: the fifth going, the fourth coming back.
ACROSS = [0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65]
BACK = ACROSS[::-1]
ACROSS_AT, BACK_AT = 4, 3


def standing_at(x: float, y: float = 0.5, category: ObjectClass = "person") -> Detection:
    """A box whose bottom middle is at ``x, y`` in fractions of the frame."""
    return Detection(x * WIDTH - 20, y * HEIGHT - 80, 40, 80, 0.9, category)


def watcher(*triggers: dict[str, object]) -> Watcher:
    return Watcher(Camera.model_validate({"id": "gate", "triggers": list(triggers)}))


def zone(**overrides: object) -> dict[str, object]:
    return {
        "kind": "zone",
        "id": "drive",
        "polygon": SQUARE,
        "classes": ["person"],
        "min_seconds": 1,
        "clear_seconds": 5,
    } | overrides


def line(**overrides: object) -> dict[str, object]:
    return {"kind": "line", "id": "gate", "points": UPRIGHT, "classes": ["person"]} | overrides


def play(
    watching: Watcher, frames: Iterable[Sequence[Detection]], start: float = 0.0
) -> list[Event]:
    """Every event from ``frames``, one each STEP from ``start``."""
    events: list[Event] = []
    for number, detections in enumerate(frames):
        events += watching.update(detections, WIDTH, HEIGHT, start + number * STEP)
    return events


def walk(
    xs: Iterable[float], y: float = 0.5, category: ObjectClass = "person"
) -> list[list[Detection]]:
    return [[standing_at(x, y, category)] for x in xs]


def still(x: float, frames: int, y: float = 0.5) -> list[list[Detection]]:
    return walk([x] * frames, y)


def nobody(frames: int) -> list[list[Detection]]:
    return [[] for _ in range(frames)]


def at(frame: int) -> float:
    return frame * STEP


class TestGeometry:
    def test_inside_a_square(self) -> None:
        square = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.8), (0.2, 0.8)]

        assert inside((0.4, 0.5), square)
        assert not inside((0.7, 0.5), square)
        assert not inside((0.4, 0.9), square)

    def test_inside_the_notch_of_an_l_is_outside(self) -> None:
        shape = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]

        assert inside((0.25, 0.25), shape)
        assert inside((0.75, 0.75), shape)
        assert not inside((0.75, 0.25), shape)

    def test_right_is_positive_seen_from_the_first_point(self) -> None:
        up = ((0.5, 0.8), (0.5, 0.2))

        assert offset(*up, (0.7, 0.5)) == pytest.approx(0.2)
        assert offset(*up, (0.3, 0.5)) == pytest.approx(-0.2)


class TestZone:
    def test_walks_in_stays_and_walks_out(self) -> None:
        frames = [
            *walk([0.05, 0.10, 0.15, 0.25]),  # in at frame 3
            *still(0.30, 15),  # frames 4 to 18
            *walk([0.40, 0.50, 0.65, 0.75, 0.85]),  # last inside at frame 20
            *nobody(40),
        ]

        events = play(watcher(zone()), frames)

        # Active once inside for a second: entered at frame 3, so frame 8.
        # Clear five seconds after the last frame inside, 20: frame 45.
        assert events == [Event("drive", "active", at(8)), Event("drive", "clear", at(45))]

    def test_a_moment_inside_is_not_enough(self) -> None:
        frames = walk([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])

        assert play(watcher(zone()), frames) == []

    def test_one_false_detection_does_not_make_it_active(self) -> None:
        """Its track lingers in the zone for a while, unseen, and that is not enough."""
        frames = [[standing_at(0.3)], *nobody(20)]

        assert play(watcher(zone()), frames) == []

    def test_with_no_minimum_one_frame_is_enough(self) -> None:
        events = play(watcher(zone(min_seconds=0)), walk([0.1, 0.3]))

        assert events == [Event("drive", "active", at(1))]

    def test_a_person_the_detector_misses_for_a_moment_does_not_clear_it(self) -> None:
        frames = [*still(0.3, 10), *nobody(5), *still(0.3, 10)]

        events = play(watcher(zone(clear_seconds=0.5)), frames)

        assert [event.change for event in events] == ["active"]

    def test_someone_else_arriving_keeps_it_active(self) -> None:
        frames = [
            *still(0.3, 10),
            *[[standing_at(0.3), standing_at(0.25, 0.7)] for _ in range(2)],
            *still(0.25, 30, y=0.7),  # the first has gone, the second stays six seconds
        ]

        events = play(watcher(zone()), frames)

        assert [event.change for event in events] == ["active"]

    def test_a_class_it_does_not_watch_for_is_ignored(self) -> None:
        frames = [[standing_at(0.3, category="vehicle")] for _ in range(20)]

        assert play(watcher(zone()), frames) == []

    def test_it_says_which_zones_are_active(self) -> None:
        watching = watcher(zone(), zone(id="lawn", polygon=[[0.7, 0.2], [0.9, 0.2], [0.9, 0.8]]))

        play(watching, still(0.3, 10))

        assert watching.active == {"drive"}


class TestLine:
    @pytest.mark.parametrize(
        ("direction", "xs", "frame"),
        [
            ("any", ACROSS, ACROSS_AT),
            ("any", BACK, BACK_AT),
            ("left_to_right", ACROSS, ACROSS_AT),
            ("left_to_right", BACK, None),
            ("right_to_left", BACK, BACK_AT),
            ("right_to_left", ACROSS, None),
        ],
    )
    def test_a_crossing_counts_in_the_chosen_direction(
        self, direction: str, xs: list[float], frame: int | None
    ) -> None:
        events = play(watcher(line(direction=direction)), walk(xs))

        assert events == ([] if frame is None else [Event("gate", "crossed", at(frame))])

    def test_wobbling_on_the_line_crosses_it_once(self) -> None:
        wobble = [0.5 - LINE_MARGIN / 2, 0.5 + LINE_MARGIN / 2] * 10
        frames = walk([*ACROSS[:3], *wobble, *ACROSS[-3:]])

        events = play(watcher(line()), frames)

        assert [event.change for event in events] == ["crossed"]

    def test_stepping_onto_the_line_and_back_is_not_a_crossing(self) -> None:
        frames = walk([0.30, 0.35, 0.40, 0.45, 0.50, 0.505, 0.45, 0.40])

        assert play(watcher(line()), frames) == []

    def test_there_and_back_is_two_crossings(self) -> None:
        frames = walk([*ACROSS, *BACK[1:]])

        assert len(play(watcher(line()), frames)) == 2

    def test_passing_beyond_its_end_is_not_a_crossing(self) -> None:
        frames = walk(ACROSS, y=0.9)

        assert play(watcher(line()), frames) == []

    def test_a_class_it_does_not_watch_for_is_ignored(self) -> None:
        frames = walk(ACROSS, category="vehicle")

        assert play(watcher(line()), frames) == []

    def test_a_crossing_while_the_detector_blinked_still_counts(self) -> None:
        frames = [*walk(ACROSS[:4]), *nobody(2), *walk(ACROSS[-2:])]

        events = play(watcher(line()), frames)

        assert events == [Event("gate", "crossed", at(6))]


def test_a_camera_watches_its_zones_and_lines_together() -> None:
    watching = watcher(zone(polygon=[[0.0, 0.0], [0.45, 0.0], [0.45, 1.0], [0.0, 1.0]]), line())

    events = play(watching, [*still(0.3, 10), *walk(ACROSS[1:])])

    assert [(event.trigger, event.change) for event in events] == [
        ("drive", "active"),
        ("gate", "crossed"),
    ]
