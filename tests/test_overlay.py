from __future__ import annotations

import numpy as np
import pytest

from pihome_vision.config import LineTrigger, ZoneTrigger
from pihome_vision.detect import Detection, Frame
from pihome_vision.overlay import (
    DETECTION,
    LINE,
    LINE_CROSSED,
    SKETCH,
    TRACK,
    ZONE,
    ZONE_ACTIVE,
    Scene,
    draw,
)
from pihome_vision.track import Track

WIDTH, HEIGHT = 320, 240

ZONE_TRIGGER = ZoneTrigger.model_validate(
    {
        "kind": "zone",
        "id": "yard",
        "polygon": [[0.5, 0.5], [0.9, 0.5], [0.9, 0.9], [0.5, 0.9]],
        "classes": ["person"],
    }
)


def line(direction: str = "any") -> LineTrigger:
    """Down the middle, from the top to the bottom."""
    return LineTrigger.model_validate(
        {
            "kind": "line",
            "id": "gate",
            "points": [[0.25, 0.1], [0.25, 0.9]],
            "direction": direction,
            "classes": ["person"],
        }
    )


def grey() -> Frame:
    return np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)


def colour_at(image: Frame, x: float, y: float) -> tuple[int, int, int]:
    b, g, r = image[round(y * HEIGHT), round(x * WIDTH)]
    return int(b), int(g), int(r)


def test_the_frame_itself_is_left_alone() -> None:
    frame = grey()

    drawn = draw(frame, Scene([ZONE_TRIGGER], active={"yard"}, status="hello"))

    assert (frame == 128).all()
    assert drawn.shape == frame.shape
    assert not (drawn == 128).all()


def test_a_zone_is_outlined_until_it_is_active_and_then_filled() -> None:
    idle = draw(grey(), Scene([ZONE_TRIGGER]))
    active = draw(grey(), Scene([ZONE_TRIGGER], active={"yard"}))

    assert colour_at(idle, 0.9, 0.7) == ZONE
    assert colour_at(idle, 0.7, 0.7) == (128, 128, 128)
    assert colour_at(active, 0.9, 0.7) == ZONE_ACTIVE
    b, g, r = colour_at(active, 0.7, 0.7)
    assert g > 128 > b
    assert r < 128


def test_a_crossed_line_changes_colour() -> None:
    assert colour_at(draw(grey(), Scene([line()])), 0.25, 0.5) == LINE
    assert colour_at(draw(grey(), Scene([line()], crossed={"gate"})), 0.25, 0.5) == LINE_CROSSED


@pytest.mark.parametrize(("direction", "arrow_x"), [("left_to_right", 0.2), ("right_to_left", 0.3)])
def test_a_one_way_line_points_to_the_side_it_counts_on(direction: str, arrow_x: float) -> None:
    """Going down the frame, the right-hand side is the left of the picture."""
    plain = draw(grey(), Scene([line()]))
    one_way = draw(grey(), Scene([line(direction)]))

    other_x = 0.5 - arrow_x
    assert colour_at(one_way, arrow_x, 0.5) == LINE
    assert colour_at(plain, arrow_x, 0.5) == (128, 128, 128)
    assert colour_at(one_way, other_x, 0.5) == (128, 128, 128)


def test_detections_tracks_and_a_half_drawn_shape_are_shown() -> None:
    scene = Scene(
        [],
        detections=[Detection(10, 20, 50, 100, 0.9, "person")],
        tracks=[Track(7, "person", (0.8, 0.2), 0.0)],
        sketch=[(0.5, 0.95), (0.9, 0.95)],
    )

    drawn = draw(grey(), scene)

    assert tuple(int(c) for c in drawn[60, 10]) == DETECTION
    assert colour_at(drawn, 0.8, 0.2) == TRACK
    assert colour_at(drawn, 0.7, 0.95) == SKETCH


def test_labels_at_the_edges_stay_inside_the_frame() -> None:
    corner = ZoneTrigger.model_validate(
        {
            "kind": "zone",
            "id": "a-zone-with-quite-a-long-name",
            "polygon": [[1.0, 1.0], [0.9, 1.0], [1.0, 0.9]],
            "classes": ["person"],
        }
    )

    drawn = draw(np.full((24, 32, 3), 128, dtype=np.uint8), Scene([corner], status="x" * 80))

    assert drawn.shape == (24, 32, 3)


def test_lines_are_thicker_on_a_bigger_frame() -> None:
    small = draw(grey(), Scene([line()]))
    big = draw(np.full((HEIGHT * 4, WIDTH * 4, 3), 128, dtype=np.uint8), Scene([line()]))

    def width(image: Frame) -> int:
        row = image[image.shape[0] // 2]
        return int((row != 128).any(axis=1).sum())

    assert width(big) > width(small)
