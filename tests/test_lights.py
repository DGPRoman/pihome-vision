from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import pytest

from pihome_vision.config import Light
from pihome_vision.hub import Hub
from pihome_vision.lights import (
    MAX_RETRY_WAIT,
    OFF_PATIENCE,
    ON_WINDOW,
    THROTTLE_PAUSE,
    Lights,
    Rule,
)
from pihome_vision.triggers import Event
from tests.conftest import HUB_KEY
from tests.fake_hub import FakeHub

RELAY = "gate-light"
OFF_AFTER = 60.0


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Sky:
    def __init__(self, *, dark: bool) -> None:
        self.dark = dark

    def is_dark(self) -> bool:
        return self.dark


def light(**overrides: object) -> Light:
    fields: dict[str, object] = {
        "relay": RELAY,
        "triggers": ["drive", "gate"],
        "off_after_seconds": OFF_AFTER,
    }
    return Light.model_validate(fields | overrides)


class TestRule:
    """When a light is wanted on, with no hub involved."""

    def test_on_while_a_zone_is_active_and_for_a_while_after(self) -> None:
        rule = Rule(light())

        assert rule.update([Event("drive", "active", 0.0)], frozenset({"drive"}), 0.0, None)
        assert rule.update([], frozenset({"drive"}), 500.0, None)
        assert rule.update([Event("drive", "clear", 600.0)], frozenset(), 600.0, None)
        assert rule.update([], frozenset(), 600.0 + OFF_AFTER - 0.1, None)
        assert not rule.update([], frozenset(), 600.0 + OFF_AFTER, None)

    def test_a_crossing_lights_it_for_a_while(self) -> None:
        rule = Rule(light())

        assert rule.update([Event("gate", "crossed", 10.0)], frozenset(), 10.0, None)
        assert rule.update([], frozenset(), 10.0 + OFF_AFTER - 0.1, None)
        assert not rule.update([], frozenset(), 10.0 + OFF_AFTER, None)

    def test_another_crossing_keeps_it_on_longer(self) -> None:
        rule = Rule(light())
        rule.update([Event("gate", "crossed", 10.0)], frozenset(), 10.0, None)
        rule.update([Event("gate", "crossed", 40.0)], frozenset(), 40.0, None)

        assert rule.update([], frozenset(), 40.0 + OFF_AFTER - 0.1, None)
        assert not rule.update([], frozenset(), 40.0 + OFF_AFTER, None)

    def test_a_crossing_during_a_zone_keeps_it_on_after_the_zone(self) -> None:
        rule = Rule(light())
        rule.update([Event("drive", "active", 0.0)], frozenset({"drive"}), 0.0, None)
        rule.update([Event("drive", "clear", 20.0)], frozenset(), 20.0, None)
        rule.update([Event("gate", "crossed", 50.0)], frozenset(), 50.0, None)

        assert rule.update([], frozenset(), 20.0 + OFF_AFTER + 1, None)
        assert not rule.update([], frozenset(), 50.0 + OFF_AFTER, None)

    def test_another_lights_triggers_are_not_its_own(self) -> None:
        rule = Rule(light())

        assert not rule.update([Event("lawn", "crossed", 0.0)], frozenset({"porch"}), 0.0, None)

    def test_only_after_dark_it_waits_for_the_dark(self) -> None:
        rule = Rule(light(only_after_dark=True))
        sky = Sky(dark=False)

        assert not rule.update([Event("drive", "active", 0.0)], frozenset({"drive"}), 0.0, sky)
        sky.dark = True
        assert rule.update([], frozenset({"drive"}), 1.0, sky)

    def test_only_after_dark_it_stays_on_past_dawn_until_the_triggers_end(self) -> None:
        rule = Rule(light(only_after_dark=True))
        sky = Sky(dark=True)
        rule.update([Event("drive", "active", 0.0)], frozenset({"drive"}), 0.0, sky)

        sky.dark = False
        assert rule.update([], frozenset({"drive"}), 1.0, sky)
        rule.update([Event("drive", "clear", 2.0)], frozenset(), 2.0, sky)
        assert not rule.update([], frozenset(), 2.0 + OFF_AFTER, sky)


class Scene:
    """Lights for one relay, a fake clock, and a fake hub behind a real client."""

    def __init__(self, hub: FakeHub, *, key: str = HUB_KEY, **overrides: object) -> None:
        self.hub = hub
        self.clock = Clock()
        self.lights = Lights([light(**overrides)], Hub(hub.origin, key), clock=self.clock)

    def zone(self, change: str) -> None:
        now = self.clock.now
        active = frozenset({"drive"}) if change == "active" else frozenset()
        self.lights.update([Event("drive", change, now)], active, now)  # type: ignore[arg-type]
        self.lights.pump()

    def later(self, seconds: float, active: Sequence[str] = ()) -> float | None:
        self.clock.now += seconds
        self.lights.update([], frozenset(active), self.clock.now)
        return self.lights.pump()

    def comes_and_goes(self) -> None:
        self.zone("active")
        self.later(5, ["drive"])
        self.zone("clear")
        self.later(OFF_AFTER)


def test_switches_on_and_off_reading_the_relay_first(hub: FakeHub) -> None:
    scene = Scene(hub)

    scene.zone("active")
    assert hub.relays[RELAY] is True
    assert scene.lights.lit == {RELAY}

    scene.later(5, ["drive"])
    scene.zone("clear")
    assert hub.relays[RELAY] is True
    scene.later(OFF_AFTER)

    assert [r.method for r in hub.requests] == ["GET", "PUT", "GET", "PUT"]
    assert hub.puts == [(RELAY, True), (RELAY, False)]
    assert scene.lights.lit == set()


def test_a_light_already_on_is_left_alone(hub: FakeHub) -> None:
    hub.relays[RELAY] = True
    scene = Scene(hub)

    scene.comes_and_goes()

    assert hub.puts == []
    assert hub.relays[RELAY] is True


def test_a_light_switched_off_by_hand_is_not_switched_again(hub: FakeHub) -> None:
    scene = Scene(hub)
    scene.zone("active")

    hub.relays[RELAY] = False
    scene.later(5, ["drive"])
    scene.later(5, ["drive"])
    scene.zone("clear")
    scene.later(OFF_AFTER)

    assert hub.puts == [(RELAY, True)]
    assert hub.relays[RELAY] is False


def test_it_lights_again_next_time(hub: FakeHub) -> None:
    scene = Scene(hub)

    scene.comes_and_goes()
    scene.comes_and_goes()

    assert hub.puts == [(RELAY, True), (RELAY, False)] * 2


def test_a_wrong_key_is_tried_exactly_once(hub: FakeHub, caplog: pytest.LogCaptureFixture) -> None:
    scene = Scene(hub, key="w" * 48)

    with caplog.at_level(logging.ERROR):
        scene.comes_and_goes()
        scene.comes_and_goes()
        scene.later(3600)

    assert len(hub.requests) == 1
    assert "PIHOME_VISION_HUB_KEY" in caplog.text
    assert "w" * 48 not in caplog.text


def test_a_relay_the_hub_does_not_have_is_left_alone(hub: FakeHub) -> None:
    del hub.relays[RELAY]
    scene = Scene(hub)

    scene.comes_and_goes()
    scene.comes_and_goes()

    assert len(hub.requests) == 1


def test_a_429_pauses_everything(hub: FakeHub) -> None:
    scene = Scene(hub)
    scene.zone("active")
    scene.zone("clear")
    hub.statuses.append(429)

    scene.later(OFF_AFTER)
    sent = len(hub.requests)
    wait = scene.later(THROTTLE_PAUSE - 1)

    assert len(hub.requests) == sent
    assert wait == pytest.approx(1)
    scene.later(1)
    assert hub.relays[RELAY] is False


def test_switching_on_is_retried_while_it_would_still_help(hub: FakeHub) -> None:
    scene = Scene(hub)
    hub.statuses.extend([503, 503])

    scene.zone("active")
    while hub.relays[RELAY] is False and scene.clock.now < 1000 + ON_WINDOW:
        scene.later(scene.lights.pump() or 0.1, ["drive"])

    assert hub.relays[RELAY] is True


def test_switching_on_too_late_is_given_up(hub: FakeHub) -> None:
    scene = Scene(hub)
    hub.statuses.extend([503] * 100)

    scene.zone("active")
    while (wait := scene.lights.pump()) is not None:
        scene.later(wait, ["drive"])
    hub.statuses.clear()
    scene.later(1, ["drive"])

    assert scene.clock.now - 1000 > ON_WINDOW
    assert hub.puts == []
    assert hub.relays[RELAY] is False


def test_switching_off_is_retried_with_longer_and_longer_waits(hub: FakeHub) -> None:
    scene = Scene(hub)
    scene.zone("active")
    scene.zone("clear")
    hub.statuses.extend([503] * 10)

    waits = []
    scene.clock.now += OFF_AFTER
    scene.lights.update([], frozenset(), scene.clock.now)
    while hub.relays[RELAY] is True:
        wait = scene.lights.pump()
        assert wait is not None
        waits.append(wait)
        scene.later(wait)

    assert waits == sorted(waits)
    assert max(waits) == MAX_RETRY_WAIT
    assert scene.lights.lit == set()


def test_switching_off_is_given_up_in_the_end(hub: FakeHub) -> None:
    scene = Scene(hub)
    scene.zone("active")
    scene.zone("clear")
    hub.statuses.extend([503] * 1000)

    scene.later(OFF_AFTER)
    while (wait := scene.lights.pump()) is not None:
        scene.later(wait)

    assert scene.clock.now - 1000 - OFF_AFTER > OFF_PATIENCE
    assert scene.lights.lit == set()
    assert hub.relays[RELAY] is True


def test_wanting_it_on_again_cancels_switching_it_off(hub: FakeHub) -> None:
    scene = Scene(hub)
    scene.zone("active")
    scene.zone("clear")
    hub.statuses.extend([503, 503])
    scene.later(OFF_AFTER)

    scene.zone("active")
    scene.later(MAX_RETRY_WAIT, ["drive"])

    assert hub.puts == [(RELAY, True)]
    assert scene.lights.lit == {RELAY}


def test_stopping_switches_off_what_it_lit(hub: FakeHub) -> None:
    scene = Scene(hub)
    scene.zone("active")

    scene.lights.stop()

    assert hub.relays[RELAY] is False


def test_stopping_leaves_what_it_did_not_light(hub: FakeHub) -> None:
    hub.relays[RELAY] = True
    scene = Scene(hub)
    scene.zone("active")

    scene.lights.stop()

    assert hub.relays[RELAY] is True
    assert hub.puts == []


def test_its_thread_switches_as_the_camera_decides(hub: FakeHub) -> None:
    lights = Lights([light()], Hub(hub.origin, HUB_KEY))
    lights.start()
    try:
        now = time.monotonic()
        lights.update([Event("drive", "active", now)], frozenset({"drive"}), now)
        deadline = time.monotonic() + 5
        while hub.relays[RELAY] is False and time.monotonic() < deadline:
            time.sleep(0.01)
        assert hub.relays[RELAY] is True
    finally:
        lights.stop()

    assert hub.relays[RELAY] is False
