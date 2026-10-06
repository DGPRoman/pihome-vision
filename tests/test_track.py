from __future__ import annotations

import itertools

import pytest

from pihome_vision.config import ObjectClass
from pihome_vision.detect import Detection
from pihome_vision.track import EDGE, MAX_AGE, MAX_STEP, Track, Tracker, footing

WIDTH, HEIGHT = 1000, 500


def standing_at(x: float, y: float, category: ObjectClass = "person") -> Detection:
    """A box whose bottom middle is at ``x, y`` in fractions of a WIDTH by HEIGHT frame."""
    return Detection(x * WIDTH - 20, y * HEIGHT - 80, 40, 80, 0.9, category)


def test_a_position_is_where_the_box_meets_the_ground() -> None:
    assert footing(Detection(100, 50, 40, 80, 0.9, "person"), WIDTH, HEIGHT) == (0.12, 0.26)


def test_a_position_is_the_same_at_any_resolution() -> None:
    small = footing(Detection(100, 50, 40, 80, 0.9, "person"), 1000, 500)
    large = footing(Detection(200, 100, 80, 160, 0.9, "person"), 2000, 1000)

    assert small == large


def test_a_box_hanging_off_the_frame_stands_just_inside_its_edge() -> None:
    assert footing(Detection(-30, 450, 40, 80, 0.9, "person"), WIDTH, HEIGHT) == (
        EDGE,
        1.0 - EDGE,
    )


def test_a_track_keeps_its_box_cut_to_the_frame() -> None:
    tracker = Tracker()

    (first,) = tracker.update([Detection(100, 400, 40, 80, 0.9, "person")], WIDTH, HEIGHT, 0.0)
    assert first.box == (0.1, 0.8, 0.14, 0.96)

    # Walking towards the camera, their feet go out of view.
    (moved,) = tracker.update([Detection(110, 450, 40, 80, 0.9, "person")], WIDTH, HEIGHT, 0.1)
    assert moved.id == first.id
    assert moved.box == (0.11, 0.9, 0.15, 1.0)
    assert moved.extent == moved.box


def test_a_track_made_without_a_box_is_its_position() -> None:
    track = Track(1, "person", (0.3, 0.4), 0.0)

    assert track.extent == (0.3, 0.4, 0.3, 0.4)


def test_an_object_keeps_its_id_as_it_moves() -> None:
    tracker = Tracker()

    ids = {
        track.id
        for step in range(10)
        for track in tracker.update([standing_at(0.1 + step * 0.03, 0.5)], WIDTH, HEIGHT, step / 5)
    }

    assert len(ids) == 1


def test_two_people_side_by_side_keep_their_own_ids() -> None:
    tracker = Tracker()
    first = tracker.update([standing_at(0.40, 0.5), standing_at(0.48, 0.5)], WIDTH, HEIGHT, 0.0)
    left, right = sorted(first, key=lambda t: t.position[0])

    # Listed the other way round, and each a little further along.
    later = tracker.update([standing_at(0.50, 0.5), standing_at(0.42, 0.5)], WIDTH, HEIGHT, 0.2)

    by_id = {track.id: track.position[0] for track in later}
    assert by_id[left.id] == pytest.approx(0.42)
    assert by_id[right.id] == pytest.approx(0.50)


def test_a_person_and_a_car_in_one_place_are_two_tracks() -> None:
    tracker = Tracker()
    tracker.update([standing_at(0.5, 0.5, "person")], WIDTH, HEIGHT, 0.0)

    tracks = tracker.update([standing_at(0.5, 0.5, "vehicle")], WIDTH, HEIGHT, 0.2)

    assert {track.category for track in tracks} == {"person", "vehicle"}


def test_a_box_too_far_away_is_somebody_else() -> None:
    tracker = Tracker()
    (first,) = tracker.update([standing_at(0.1, 0.5)], WIDTH, HEIGHT, 0.0)

    tracks = tracker.update([standing_at(0.9, 0.5)], WIDTH, HEIGHT, 0.2)

    assert first.id not in {track.id for track in tracks if track.position[0] > 0.5}


def test_a_track_missed_for_a_moment_stays_where_it_was() -> None:
    tracker = Tracker()
    (first,) = tracker.update([standing_at(0.3, 0.5)], WIDTH, HEIGHT, 0.0)

    (still,) = tracker.update([], WIDTH, HEIGHT, MAX_AGE - 0.1)
    (back,) = tracker.update([standing_at(0.31, 0.5)], WIDTH, HEIGHT, MAX_AGE)

    assert still.id == back.id == first.id
    assert still.position == first.position


def test_a_track_missed_for_longer_is_forgotten() -> None:
    tracker = Tracker()
    (first,) = tracker.update([standing_at(0.3, 0.5)], WIDTH, HEIGHT, 0.0)

    assert tracker.update([], WIDTH, HEIGHT, MAX_AGE + 0.1) == []
    (later,) = tracker.update([standing_at(0.3, 0.5)], WIDTH, HEIGHT, MAX_AGE + 0.2)
    assert later.id != first.id


def test_velocity_is_in_frame_fractions_a_second() -> None:
    tracker = Tracker()
    for step in range(20):
        tracks = tracker.update([standing_at(0.1 + step * 0.02, 0.5)], WIDTH, HEIGHT, step / 5)

    (walker,) = tracks
    assert walker.velocity == pytest.approx((0.1, 0.0), abs=1e-6)


def test_a_car_speeding_up_is_followed_by_where_it_was_heading() -> None:
    """Its last steps are longer than MAX_STEP, and it is still one track."""
    tracker = Tracker()
    along = [0.05, 0.15, 0.27, 0.42, 0.60, 0.80]
    assert max(b - a for a, b in itertools.pairwise(along)) > MAX_STEP

    ids = {
        track.id
        for step, x in enumerate(along)
        for track in tracker.update([standing_at(x, 0.8, "vehicle")], WIDTH, HEIGHT, step / 5)
    }

    assert len(ids) == 1
