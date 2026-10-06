from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from pihome_vision.config import (
    ConfigError,
    LineTrigger,
    VisionConfig,
    ZoneTrigger,
    load_config,
    render_config_error,
)
from tests.conftest import EXAMPLE_CONFIG


@pytest.fixture
def document() -> dict[str, Any]:
    """The example configuration, as a mapping a test can break one way."""
    loaded: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return copy.deepcopy(loaded)


def refusal(document: dict[str, Any]) -> str:
    with pytest.raises(ValidationError) as caught:
        VisionConfig.model_validate(document)
    return render_config_error(caught.value)


def test_the_example_loads() -> None:
    config = load_config(EXAMPLE_CONFIG)

    triggers = config.cameras[0].triggers
    assert [type(trigger) for trigger in triggers] == [ZoneTrigger, LineTrigger]
    assert config.lights[0].triggers == ["driveway", "wicket"]


def test_defaults_fill_what_is_left_out(document: dict[str, Any]) -> None:
    del document["location"]
    document["lights"] = [{"relay": "gate-light", "triggers": ["wicket"]}]
    document["cameras"][0]["triggers"][0] = {
        "kind": "zone",
        "id": "driveway",
        "polygon": [[0, 0], [1, 0], [1, 1]],
        "classes": ["person"],
    }

    config = VisionConfig.model_validate(document)

    zone = config.cameras[0].triggers[0]
    assert isinstance(zone, ZoneTrigger)
    assert (zone.min_seconds, zone.clear_seconds) == (1.0, 5.0)
    assert config.lights[0].off_after_seconds == 120.0
    assert config.lights[0].only_after_dark is False


def test_an_unknown_key_is_refused_with_its_place(document: dict[str, Any]) -> None:
    document["cameras"][0]["triggers"][1]["colour"] = "red"

    assert "cameras.0.triggers.1.line.colour" in refusal(document)


@pytest.mark.parametrize(
    ("polygon", "reason"),
    [
        ([[0, 0], [1, 1]], "at least 3"),
        ([[0, 0], [0.5, 0.5], [1, 1]], "no area"),
        ([[0, 0], [1.2, 0], [1, 1]], "less than or equal to 1"),
        ([[0, 0], [-0.1, 0], [1, 1]], "greater than or equal to 0"),
    ],
)
def test_a_zone_that_cannot_be_drawn_is_refused(
    document: dict[str, Any], polygon: list[list[float]], reason: str
) -> None:
    document["cameras"][0]["triggers"][0]["polygon"] = polygon

    assert reason in refusal(document)


def test_a_line_with_no_length_is_refused(document: dict[str, Any]) -> None:
    document["cameras"][0]["triggers"][1]["points"] = [[0.5, 0.5], [0.5, 0.5]]

    assert "same point" in refusal(document)


@pytest.mark.parametrize("classes", [["dog"], []])
def test_classes_must_be_ones_it_can_detect(document: dict[str, Any], classes: list[str]) -> None:
    document["cameras"][0]["triggers"][0]["classes"] = classes

    assert "classes" in refusal(document)


def test_a_direction_must_be_one_of_the_three(document: dict[str, Any]) -> None:
    document["cameras"][0]["triggers"][1]["direction"] = "inwards"

    assert "direction" in refusal(document)


def test_trigger_ids_are_unique(document: dict[str, Any]) -> None:
    document["cameras"][0]["triggers"][1]["id"] = "driveway"

    assert "'driveway' is defined more than once" in refusal(document)


def test_a_light_must_name_triggers_that_exist(document: dict[str, Any]) -> None:
    document["lights"][0]["triggers"] = ["driveway", "porch"]

    assert "'porch', which no camera has" in refusal(document)


def test_a_light_names_each_trigger_once(document: dict[str, Any]) -> None:
    document["lights"][0]["triggers"] = ["driveway", "driveway"]

    assert "more than once" in refusal(document)


def test_one_relay_has_one_light(document: dict[str, Any]) -> None:
    document["lights"].append({"relay": "gate-light", "triggers": ["wicket"]})

    assert "'gate-light' has more than one light" in refusal(document)


def test_after_dark_needs_a_location(document: dict[str, Any]) -> None:
    del document["location"]

    assert "needs a location" in refusal(document)


def test_a_time_zone_must_exist(document: dict[str, Any]) -> None:
    document["location"]["timezone"] = "Europe/Atlantis"

    assert "not a time zone" in refusal(document)


def test_a_second_camera_is_refused_for_now(document: dict[str, Any]) -> None:
    second = copy.deepcopy(document["cameras"][0])
    second["id"] = "yard"
    for trigger in second["triggers"]:
        trigger["id"] = f"yard-{trigger['id']}"
    document["cameras"].append(second)

    assert "at most 1" in refusal(document)


@pytest.mark.parametrize("relay", ["Gate light", "gate_light", "-gate", ""])
def test_a_relay_id_must_be_one_the_hub_could_have(document: dict[str, Any], relay: str) -> None:
    document["lights"][0]["relay"] = relay

    assert "lights.0.relay" in refusal(document)


@pytest.mark.parametrize("size", [100, 650, 2048])
def test_the_model_input_must_be_a_size_it_can_take(document: dict[str, Any], size: int) -> None:
    document["model"]["input_size"] = size

    assert "model.input_size" in refusal(document)


def test_a_missing_file_says_where_to_start(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"vision\.example\.yaml"):
        load_config(tmp_path / "vision.yaml")


@pytest.mark.parametrize(("text", "reason"), [("cameras: [", "not valid YAML"), ("- 1", "mapping")])
def test_a_file_that_is_not_a_mapping_is_refused(tmp_path: Path, text: str, reason: str) -> None:
    path = tmp_path / "vision.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match=reason):
        load_config(path)


def test_a_wrong_file_names_itself_and_every_problem(tmp_path: Path) -> None:
    path = tmp_path / "vision.yaml"
    path.write_text("model: {path: m.onnx, confidence: 2}\ncameras: []\nlights: []\n")

    with pytest.raises(ConfigError) as caught:
        load_config(path)

    message = str(caught.value)
    assert str(path) in message
    for where in ("model.confidence", "cameras", "lights"):
        assert f"  {where}:" in message
