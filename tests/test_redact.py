from __future__ import annotations

import pytest

from pihome_vision.redact import MASK, mask_url, scrub
from tests.conftest import CAMERA_PASSWORD, CAMERA_URL


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("rtsp://user:secret@192.168.1.50:554/stream2", f"rtsp://{MASK}@192.168.1.50:554/stream2"),
        ("rtsp://user@192.168.1.50/stream2", f"rtsp://{MASK}@192.168.1.50/stream2"),
        ("rtsp://:secret@192.168.1.50/stream2", f"rtsp://{MASK}@192.168.1.50/stream2"),
        ("rtsp://user:p%40ss@192.168.1.50/s?x=1", f"rtsp://{MASK}@192.168.1.50/s?x=1"),
        ("rtsp://user:secret@[fd00::20]:554/s", f"rtsp://{MASK}@[fd00::20]:554/s"),
    ],
)
def test_credentials_are_replaced(url: str, expected: str) -> None:
    assert mask_url(url) == expected


@pytest.mark.parametrize(
    "url",
    ["rtsp://192.168.1.50/stream2", "https://203.0.113.10", "cam:0"],
)
def test_a_url_without_credentials_is_unchanged(url: str) -> None:
    assert mask_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        # A password holding an @ that was not percent-encoded.
        "rtsp://user:se@cret@192.168.1.50/stream2",
        # A port that is not a number makes the parser give up.
        "rtsp://user:secret@192.168.1.50:port/stream2",
        # No scheme at all.
        "user:secret@192.168.1.50/stream2",
    ],
)
def test_nothing_of_a_password_survives_a_malformed_url(url: str) -> None:
    assert "secret" not in mask_url(url)
    assert "cret" not in mask_url(url)


def test_scrub_masks_a_url_that_ffmpeg_repeats() -> None:
    said = f"Error opening input file {CAMERA_URL}."

    assert (
        scrub(said, CAMERA_URL)
        == f"Error opening input file rtsp://{MASK}@192.168.1.50:554/stream2."
    )


def test_scrub_masks_the_password_on_its_own() -> None:
    said = f"[rtsp @ 0x5a40] bad password {CAMERA_PASSWORD} for viewer"

    assert CAMERA_PASSWORD not in scrub(said, CAMERA_URL)


def test_scrub_masks_a_percent_encoded_password_in_both_spellings() -> None:
    url = "rtsp://viewer:p%40ss-word@192.168.1.50/stream2"

    assert "p@ss-word" not in scrub("tried p@ss-word", url)
    assert "p%40ss-word" not in scrub("tried p%40ss-word", url)


def test_scrub_masks_any_url_with_credentials_even_when_the_address_will_not_parse() -> None:
    said = "Error opening input file rtsp://viewer:secret@192.168.1.50:port/stream2."

    assert "secret" not in scrub(said, "rtsp://viewer:secret@192.168.1.50:port/stream2")


def test_scrub_leaves_text_without_credentials_alone() -> None:
    said = "[tcp @ 0x5a97] Connection to tcp://192.168.1.50:554 failed: Connection refused"

    assert scrub(said, CAMERA_URL) == said


AT_IN_PASSWORD = "rtsp://viewer:Summer@2024x@192.168.1.50:554/stream2"


@pytest.mark.parametrize(
    "said",
    [
        "Error opening input file {url}.",
        "[in#0 @ 0x615f] Impossible to open '{url}'",
        "{url}: Connection refused",
    ],
)
def test_scrub_leaves_nothing_of_a_password_that_holds_an_at(said: str) -> None:
    scrubbed = scrub(said.format(url=AT_IN_PASSWORD), AT_IN_PASSWORD)

    assert "Summer" not in scrubbed
    assert "2024x" not in scrubbed
    assert f"rtsp://{MASK}@192.168.1.50:554/stream2" in scrubbed
