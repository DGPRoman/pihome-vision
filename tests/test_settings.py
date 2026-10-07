from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from pihome_vision.settings import CameraSettings, Settings, render_settings_error
from tests.conftest import CAMERA_PASSWORD, CAMERA_URL, HUB_KEY, HUB_URL


def build(**overrides: str) -> Settings:
    values: dict[str, Any] = {"camera_url": CAMERA_URL, "hub_url": HUB_URL, "hub_key": HUB_KEY}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]  # a pydantic-settings init option


def refusal(**overrides: str) -> str:
    with pytest.raises(ValidationError) as caught:
        build(**overrides)
    return render_settings_error(caught.value)


def test_a_complete_environment_loads() -> None:
    settings = build()

    assert settings.camera_url.get_secret_value() == CAMERA_URL
    assert settings.hub_url == HUB_URL


def test_the_camera_password_is_not_in_the_repr() -> None:
    assert CAMERA_PASSWORD not in repr(build())
    assert CAMERA_PASSWORD not in str(build())


@pytest.mark.parametrize("url", ["cam:0", "http://192.168.1.50/mjpeg", "rtsps://192.168.1.50/s"])
def test_other_camera_sources_are_accepted(url: str) -> None:
    assert build(camera_url=url).camera_url.get_secret_value() == url


@pytest.mark.parametrize(
    "url",
    [
        f"ftp://viewer:{CAMERA_PASSWORD}@192.168.1.50/stream",
        f"viewer:{CAMERA_PASSWORD}@192.168.1.50/stream",
        f"rtsp://viewer:{CAMERA_PASSWORD}@/stream",
        f"rtsp://viewer:{CAMERA_PASSWORD}@[192.168.1.50/stream",
    ],
)
def test_a_refused_camera_url_is_not_repeated_back(url: str) -> None:
    message = refusal(camera_url=url)

    assert "PIHOME_VISION_CAMERA_URL" in message
    assert CAMERA_PASSWORD not in message


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.20:5002",
        "http://10.0.0.5",
        "http://100.64.0.5:5002",
        "http://127.0.0.1:5002",
        "http://localhost:5002",
        "http://[fd00::20]:5002",
        "https://203.0.113.10/",
    ],
)
def test_hub_urls_that_keep_the_key_safe_are_accepted(url: str) -> None:
    assert build(hub_url=url).hub_url == url.rstrip("/")


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://203.0.113.10", "in the clear"),
        ("http://hub.example", "in the clear"),
        ("https://203.0.113.10/v1", "origin"),
        ("https://203.0.113.10?x=1", "origin"),
        ("ws://203.0.113.10", "http or https"),
        ("https://", "http or https"),
        ("https://203.0.113.10:port", "not a URL that can be read"),
    ],
)
def test_hub_urls_that_would_not_work_or_would_leak_are_refused(url: str, reason: str) -> None:
    assert reason in refusal(hub_url=url)


def test_a_key_pasted_into_the_hub_url_is_refused_without_repeating_it() -> None:
    message = refusal(hub_url=f"https://user:{HUB_KEY}@203.0.113.10")

    assert "credentials" in message
    assert HUB_KEY not in message


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        ("short", "at least 32"),
        ("replace-me-with-the-hubs-relay-key", "example configuration"),
        ("x" * 20 + "CHANGEME" + "x" * 20, "example configuration"),
    ],
)
def test_weak_or_example_keys_are_refused(key: str, reason: str) -> None:
    message = refusal(hub_key=key)

    assert "PIHOME_VISION_HUB_KEY" in message
    assert reason in message


def test_every_missing_variable_is_named() -> None:
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None)  # type: ignore[call-arg]

    message = render_settings_error(caught.value)
    for name in ("CAMERA_URL", "HUB_URL", "HUB_KEY"):
        assert f"PIHOME_VISION_{name}" in message


def camera_only(**given: str) -> CameraSettings:
    values: dict[str, Any] = dict(given)
    return CameraSettings(_env_file=None, **values)  # type: ignore[call-arg]  # a pydantic-settings init option


def test_the_camera_alone_needs_only_the_camera() -> None:
    assert camera_only(camera_url=CAMERA_URL).camera_url.get_secret_value() == CAMERA_URL
    with pytest.raises(ValidationError) as caught:
        camera_only()

    message = render_settings_error(caught.value)
    assert "PIHOME_VISION_CAMERA_URL" in message
    assert "HUB" not in message


def test_the_camera_alone_refuses_the_same_addresses() -> None:
    with pytest.raises(ValidationError) as caught:
        camera_only(camera_url=f"ftp://viewer:{CAMERA_PASSWORD}@192.168.1.50/")

    message = render_settings_error(caught.value)
    assert "PIHOME_VISION_CAMERA_URL" in message
    assert CAMERA_PASSWORD not in message
