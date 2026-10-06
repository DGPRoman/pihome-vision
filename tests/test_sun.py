from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from pihome_vision.config import Location
from pihome_vision.sun import Sun

#: The middle of a city, as in config/vision.example.yaml.
KYIV = Location(latitude=50.4501, longitude=30.5234, timezone="Europe/Kyiv")
#: Far enough north for the sun not to set in summer, nor rise in winter.
SVALBARD = Location(latitude=78.22, longitude=15.65, timezone="Arctic/Longyearbyen")


def sun_at(location: Location, month: int, day: int, hour: int, minute: int = 0) -> Sun:
    moment = datetime(2026, month, day, hour, minute, tzinfo=ZoneInfo(location.timezone))
    return Sun(location, clock=lambda: moment)


@pytest.mark.parametrize(
    ("when", "dark"),
    [
        ((6, 21, 3, 0), True),
        ((6, 21, 12, 0), False),
        ((6, 21, 23, 0), True),
        ((12, 21, 7, 30), True),
        ((12, 21, 12, 0), False),
        ((12, 21, 17, 0), True),
    ],
)
def test_dark_before_sunrise_and_after_sunset(when: tuple[int, ...], dark: bool) -> None:
    assert sun_at(KYIV, *when).is_dark() is dark


def test_light_all_night_in_the_midnight_sun() -> None:
    assert sun_at(SVALBARD, 6, 21, 0, 30).is_dark() is False


def test_dark_all_day_in_the_polar_night() -> None:
    assert sun_at(SVALBARD, 12, 21, 12, 0).is_dark() is True


def test_a_new_day_has_its_own_sunrise() -> None:
    moments = iter(
        [
            datetime(2026, 6, 21, 12, 0, tzinfo=ZoneInfo(KYIV.timezone)),
            datetime(2026, 6, 22, 2, 0, tzinfo=ZoneInfo(KYIV.timezone)),
        ]
    )
    sun = Sun(KYIV, clock=lambda: next(moments))

    assert sun.is_dark() is False
    assert sun.is_dark() is True
