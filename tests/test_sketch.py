from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pihome_vision.config import LineTrigger, VisionConfig, ZoneTrigger, load_config
from pihome_vision.sketch import BACKSPACE, ENTER, ESCAPE, Sketch, dangling, fraction, fragment
from tests.conftest import EXAMPLE_CONFIG

SQUARE = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]


def drawn(sketch: Sketch, points: list[tuple[float, float]], kind: str, name: str = "") -> None:
    for point in points:
        sketch.click(point)
    sketch.key(ord(kind))
    for character in name:
        sketch.key(ord(character))
    sketch.key(ENTER[1])


def test_a_pixel_becomes_a_rounded_fraction_inside_the_frame() -> None:
    assert fraction(640, 120, 1280, 720) == (0.5, 0.167)
    assert fraction(-3, 800, 1280, 720) == (0.0, 1.0)


def test_a_click_close_to_an_edge_is_on_it() -> None:
    """A zone meant to reach the edge would otherwise stop a few pixels short of it."""
    assert fraction(1270, 5, 1280, 720) == (1.0, 0.0)
    assert fraction(20, 712, 1280, 720) == (0.0, 1.0)
    assert fraction(1240, 30, 1280, 720) == (0.969, 0.042)


def test_points_closed_with_z_and_named_are_a_zone() -> None:
    sketch = Sketch()

    drawn(sketch, SQUARE, "z", "drive way")

    (zone,) = sketch.triggers
    assert isinstance(zone, ZoneTrigger)
    assert zone.id == "drive-way"
    assert zone.polygon == SQUARE
    assert zone.classes == {"person", "vehicle"}
    assert sketch.points == []


def test_two_points_and_l_are_a_line() -> None:
    sketch = Sketch()

    drawn(sketch, [(0.7, 0.2), (0.7, 0.9)], "l", "Wicket")

    (line,) = sketch.triggers
    assert isinstance(line, LineTrigger)
    assert line.id == "wicket"
    assert line.points == ((0.7, 0.2), (0.7, 0.9))


def test_a_name_left_empty_is_numbered() -> None:
    sketch = Sketch()

    drawn(sketch, SQUARE, "z")
    drawn(sketch, SQUARE, "z")
    drawn(sketch, [(0.1, 0.1), (0.2, 0.2)], "l")

    assert [t.id for t in sketch.triggers] == ["zone-1", "zone-2", "line-1"]


@pytest.mark.parametrize(
    ("points", "kind", "says"),
    [
        (SQUARE[:2], "z", "a zone needs three points or more"),
        ([(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)], "z", "not a zone: has no area"),
        (SQUARE[:3], "l", "a line is two points"),
        ([(0.4, 0.4), (0.4, 0.4)], "l", "not a line: both ends are the same point"),
    ],
)
def test_a_shape_the_configuration_would_refuse_is_not_finished(
    points: list[tuple[float, float]], kind: str, says: str
) -> None:
    sketch = Sketch()
    for point in points:
        sketch.click(point)

    sketch.key(ord(kind))

    assert sketch.naming is None
    assert sketch.status.startswith(says)
    assert sketch.points == points


def test_a_name_is_a_slug_whatever_is_typed() -> None:
    sketch = Sketch()
    for point in SQUARE:
        sketch.click(point)
    sketch.key(ord("z"))

    for character in "-Front_ DOOR!--":
        sketch.key(ord(character))

    assert sketch.name == "front-door-"
    sketch.key(ENTER[0])
    assert sketch.triggers[0].id == "front-door"


def test_a_name_already_taken_is_asked_for_again() -> None:
    sketch = Sketch()
    drawn(sketch, SQUARE, "z", "yard")

    drawn(sketch, SQUARE, "z", "yard")

    assert len(sketch.triggers) == 1
    assert sketch.naming == "zone"
    assert sketch.status == "there is a trigger called yard already"


def test_backspace_takes_back_a_letter_a_point_and_then_a_shape() -> None:
    sketch = Sketch()
    drawn(sketch, SQUARE, "z", "yard")
    sketch.click((0.9, 0.9))
    sketch.key(ord("l"))  # one point is not a line

    sketch.key(BACKSPACE[0])
    assert sketch.points == []
    sketch.key(BACKSPACE[1])
    assert sketch.triggers == []
    assert sketch.status == "removed zone yard"


def test_escape_while_naming_goes_back_to_the_points() -> None:
    sketch = Sketch()
    for point in SQUARE:
        sketch.click(point)
    sketch.key(ord("z"))
    sketch.key(ord("x"))

    sketch.key(ESCAPE)
    sketch.click((0.6, 0.6))  # clicks count again

    assert sketch.naming is None
    assert sketch.points == [*SQUARE, (0.6, 0.6)]
    assert sketch.triggers == []


def test_clicks_while_naming_are_ignored() -> None:
    sketch = Sketch()
    for point in SQUARE:
        sketch.click(point)
    sketch.key(ord("z"))

    sketch.click((0.6, 0.6))
    sketch.key(ENTER[0])

    (zone,) = sketch.triggers
    assert isinstance(zone, ZoneTrigger)
    assert zone.polygon == SQUARE


def test_q_finishes_unless_a_shape_is_half_drawn() -> None:
    sketch = Sketch()
    sketch.click((0.1, 0.1))

    sketch.key(ord("q"))
    assert not sketch.done
    assert sketch.status.startswith("z or l to finish this shape")

    sketch.key(BACKSPACE[0])
    sketch.key(ESCAPE)
    assert sketch.done


def test_the_status_line_says_what_to_do_next() -> None:
    sketch = Sketch()
    assert sketch.status.startswith("click to add points;  backspace: remove the last shape")
    sketch.click((0.1, 0.1))
    assert "z: zone;  l: line" in sketch.status
    for point in SQUARE[1:]:
        sketch.click(point)
    sketch.key(ord("z"))
    sketch.key(ord("a"))
    assert sketch.status.startswith("name the zone: a_   (enter for a,")


@pytest.fixture
def config() -> VisionConfig:
    return load_config(EXAMPLE_CONFIG)


def test_the_fragment_replaces_the_cameras_section_as_it_is(
    config: VisionConfig, tmp_path: Path
) -> None:
    """What ``edit`` prints is pasted over ``cameras:`` and loads without changes."""
    (camera,) = config.cameras
    sketch = Sketch(camera.triggers)
    drawn(sketch, SQUARE, "z", "porch")
    drawn(sketch, [(0.2, 0.9), (0.8, 0.9)], "l", "pavement")

    printed = fragment(camera, sketch.triggers)

    document = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    document |= yaml.safe_load(printed)
    edited = tmp_path / "vision.yaml"
    edited.write_text(yaml.safe_dump(document), encoding="utf-8")
    (loaded,) = load_config(edited).cameras
    assert loaded == camera.model_copy(update={"triggers": sketch.triggers})


def test_the_fragment_reads_like_the_example(config: VisionConfig) -> None:
    (camera,) = config.cameras
    (zone, line) = camera.triggers
    zone = zone.model_copy(update={"classes": frozenset({"vehicle", "person"})})

    lines = fragment(camera, [zone, line]).splitlines()

    assert lines[:4] == ["cameras:", "  - id: gate", "    fps: 5.0", "    motion_threshold: 0.0"]
    assert "        polygon:" in lines
    assert "          - [0.1, 0.55]" in lines
    assert "        classes: [person, vehicle]" in lines
    assert "        direction: any" in lines


def test_lights_left_naming_a_removed_trigger_are_listed(config: VisionConfig) -> None:
    (camera,) = config.cameras
    driveway, _ = camera.triggers

    assert dangling(config, [driveway]) == ["gate-light: wicket"]
    assert dangling(config, camera.triggers) == []
