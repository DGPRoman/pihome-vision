from __future__ import annotations

import socket

import pytest

from pihome_vision.hub import Hub, HubError, Refusal
from tests import fake_hub
from tests.conftest import HUB_KEY
from tests.fake_hub import FakeHub


def test_reads_a_relay_with_the_key(hub: FakeHub) -> None:
    hub.relays["gate-light"] = True

    assert hub.client().read("gate-light") is True
    (request,) = hub.requests
    assert (request.method, request.path, request.key) == ("GET", "/v1/relays/gate-light", HUB_KEY)


def test_switches_a_relay_and_says_what_it_is_now(hub: FakeHub) -> None:
    assert hub.client().switch("gate-light", on=True) is True

    assert hub.puts == [("gate-light", True)]
    assert hub.relays["gate-light"] is True


def test_one_connection_carries_request_after_request(hub: FakeHub) -> None:
    """Switching a light on is a read and a switch: neither waits for a new connection."""
    client = hub.client()

    client.read("gate-light")
    client.switch("gate-light", on=True)
    client.read("gate-light")

    assert len({request.peer for request in hub.requests}) == 1


def test_a_connection_the_hub_has_closed_is_replaced(hub: FakeHub) -> None:
    client = hub.client()
    hub.hang_up = True
    client.read("gate-light")

    assert client.switch("gate-light", on=True) is True

    first, second = hub.requests
    assert first.peer != second.peer
    assert hub.puts == [("gate-light", True)]


def test_a_hub_too_slow_on_a_kept_connection_is_not_asked_again_at_once(hub: FakeHub) -> None:
    client = hub.client(timeout=0.2)
    client.read("gate-light")
    hub.released.clear()

    with pytest.raises(HubError) as caught:
        client.switch("gate-light", on=True)

    assert caught.value.refusal is Refusal.UNAVAILABLE
    assert len(hub.requests) == 2
    hub.released.set()
    client.read("gate-light")
    assert hub.requests[2].peer != hub.requests[1].peer


@pytest.mark.parametrize(
    ("origin", "address"),
    [
        ("http://[fd00::20]", ("fd00::20", 80)),
        ("https://[fd00::20]", ("fd00::20", 443)),
        ("https://[fd00::20]:8443", ("fd00::20", 8443)),
    ],
)
def test_an_ipv6_origin_is_reached_on_its_port(
    origin: str, address: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not on the last group of the address, which http.client would take for one."""
    tried: list[tuple[str, int]] = []

    def refuse(to: tuple[str, int], *args: object, **kwargs: object) -> socket.socket:
        tried.append(to)
        raise ConnectionRefusedError

    monkeypatch.setattr(socket, "create_connection", refuse)

    with Hub(origin, HUB_KEY) as client, pytest.raises(HubError):
        client.read("gate-light")

    assert tried == [address]


@pytest.mark.parametrize(
    ("status", "refusal"),
    [
        (401, Refusal.UNAUTHORIZED),
        (404, Refusal.NO_SUCH_RELAY),
        (429, Refusal.THROTTLED),
        (500, Refusal.UNAVAILABLE),
        (503, Refusal.UNAVAILABLE),
        (400, Refusal.REJECTED),
        (403, Refusal.REJECTED),
        (422, Refusal.REJECTED),
    ],
)
def test_each_status_is_a_refusal(hub: FakeHub, status: int, refusal: Refusal) -> None:
    hub.statuses.append(status)

    with pytest.raises(HubError) as caught:
        hub.client().read("gate-light")

    assert caught.value.refusal is refusal
    assert HUB_KEY not in str(caught.value)


def test_a_redirect_is_not_followed(hub: FakeHub) -> None:
    """Following it would send the key to wherever it points."""
    elsewhere = FakeHub(HUB_KEY)
    server = fake_hub.serve(elsewhere)
    hub.statuses.append(307)
    hub.headers["Location"] = f"{elsewhere.origin}/v1/relays/gate-light"
    try:
        with pytest.raises(HubError, match="redirect") as caught:
            hub.client().read("gate-light")
    finally:
        server.shutdown()
        server.server_close()

    assert caught.value.refusal is Refusal.REJECTED
    assert elsewhere.requests == []


def test_a_proxy_in_the_environment_is_not_used(
    hub: FakeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("http_proxy", "http://192.0.2.1:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://192.0.2.1:3128")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)

    assert hub.client(timeout=2).read("gate-light") is False


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"on": "yes"}', b'{"id": "gate-light"}'])
def test_an_answer_without_a_state_is_refused(hub: FakeHub, body: bytes) -> None:
    hub.body = body

    with pytest.raises(HubError) as caught:
        hub.client().read("gate-light")

    assert caught.value.refusal is Refusal.REJECTED


def test_a_hub_that_is_not_there_is_unavailable() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    with pytest.raises(HubError) as caught:
        Hub(f"http://127.0.0.1:{port}", HUB_KEY).read("gate-light")

    assert caught.value.refusal is Refusal.UNAVAILABLE


def test_a_hub_that_does_not_answer_is_unavailable(hub: FakeHub) -> None:
    hub.released.clear()

    with pytest.raises(HubError) as caught:
        hub.client(timeout=0.2).read("gate-light")

    assert caught.value.refusal is Refusal.UNAVAILABLE


def test_a_relay_name_cannot_reach_another_route(hub: FakeHub) -> None:
    with pytest.raises(HubError):
        hub.client().read("../../health")

    assert hub.requests[0].path == "/v1/relays/..%2F..%2Fhealth"
