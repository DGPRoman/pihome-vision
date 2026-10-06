from __future__ import annotations

import pytest

from pihome_vision.redact import MASK, mask_url


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
