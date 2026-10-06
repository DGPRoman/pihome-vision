"""The hub client and the lights against a real hub, which scripts/contract-test.sh
starts with the mock relay backend and the relays in tests/contract/relays.yaml.

Skipped without one: the rest of the suite proves the same behaviour against a stand-in,
and this proves the stand-in answers as the hub does.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from pihome_vision.config import Light
from pihome_vision.hub import Hub, HubError, Refusal
from pihome_vision.lights import Lights
from pihome_vision.triggers import Event

ORIGIN = os.environ.get("PIHOME_CONTRACT_HUB", "")
KEY = os.environ.get("PIHOME_CONTRACT_RELAY_KEY", "")

if not ORIGIN or not KEY:
    pytest.skip(
        "no hub to run against: scripts/contract-test.sh starts one", allow_module_level=True
    )

RELAY = "gate-light"


class Counting:
    """A hub client that counts what it is asked."""

    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.requests = 0

    def read(self, relay: str) -> bool:
        self.requests += 1
        return self.hub.read(relay)

    def switch(self, relay: str, *, on: bool) -> bool:
        self.requests += 1
        return self.hub.switch(relay, on=on)


@pytest.fixture
def hub() -> Iterator[Hub]:
    """The real hub, with every relay off before and after."""
    client = Hub(ORIGIN, KEY)
    for relay in ("gate-light", "porch-light"):
        client.switch(relay, on=False)
    yield client
    for relay in ("gate-light", "porch-light"):
        client.switch(relay, on=False)


def come_and_go(lights: Lights) -> None:
    lights.update([Event("drive", "active", 0.0)], frozenset({"drive"}), 0.0)
    lights.pump()
    lights.update([Event("drive", "clear", 1.0)], frozenset(), 1.0)
    lights.update([], frozenset(), 1.0 + 60.0)
    lights.pump()


def lights_for(relays: Hub | Counting) -> Lights:
    light = Light(relay=RELAY, triggers=["drive"], off_after_seconds=60.0)
    return Lights([light], relays)


def test_reads_and_switches_a_relay(hub: Hub) -> None:
    assert hub.read(RELAY) is False
    assert hub.switch(RELAY, on=True) is True
    assert hub.read(RELAY) is True
    assert hub.read("porch-light") is False


def test_a_relay_the_hub_does_not_have(hub: Hub) -> None:
    with pytest.raises(HubError) as caught:
        hub.read("no-such-light")

    assert caught.value.refusal is Refusal.NO_SUCH_RELAY


def test_a_wrong_key(hub: Hub) -> None:
    with pytest.raises(HubError) as caught:
        Hub(ORIGIN, "w" * 48).read(RELAY)

    assert caught.value.refusal is Refusal.UNAUTHORIZED


def test_a_light_comes_on_and_goes_off(hub: Hub) -> None:
    lights = lights_for(hub)

    lights.update([Event("drive", "active", 0.0)], frozenset({"drive"}), 0.0)
    lights.pump()
    assert hub.read(RELAY) is True

    lights.update([Event("drive", "clear", 1.0)], frozenset(), 1.0)
    lights.update([], frozenset(), 1.0 + 60.0)
    lights.pump()
    assert hub.read(RELAY) is False


def test_a_light_already_on_is_left_on(hub: Hub) -> None:
    hub.switch(RELAY, on=True)

    come_and_go(lights_for(hub))

    assert hub.read(RELAY) is True


def test_a_wrong_key_is_tried_once(hub: Hub) -> None:
    counting = Counting(Hub(ORIGIN, "w" * 48))
    lights = lights_for(counting)

    come_and_go(lights)
    come_and_go(lights)

    assert counting.requests == 1
    assert hub.read(RELAY) is False


def test_stopping_switches_off_what_it_lit(hub: Hub) -> None:
    lights = lights_for(hub)
    lights.update([Event("drive", "active", 0.0)], frozenset({"drive"}), 0.0)
    lights.pump()

    lights.stop()

    assert hub.read(RELAY) is False
