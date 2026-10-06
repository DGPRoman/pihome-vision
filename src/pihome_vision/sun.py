"""Whether it is dark where the house is, for lights that only come on at night.

The same question the hub answers for its own rules, answered the same way: dark is
before sunrise or after sunset, local time, from :mod:`astral`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from astral import Observer
from astral.sun import elevation, noon, sunrise, sunset

from pihome_vision.config import Location

_log = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Sun:
    """Answers :meth:`is_dark` at one place.

    Sunrise and sunset move by minutes a day, so they are worked out once a local
    date. Far enough north or south there are dates with neither, and the whole of
    such a date is day or night by where the sun is at noon.
    """

    def __init__(self, location: Location, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._zone = ZoneInfo(location.timezone)
        self._observer = Observer(location.latitude, location.longitude)
        self._clock = clock
        self._day: date | None = None
        #: Sunrise and sunset on that date, or None on a date with neither.
        self._times: tuple[datetime, datetime] | None = None
        #: On a date with neither, whether it is dark all day.
        self._dark_all_day = False

    def is_dark(self) -> bool:
        now = self._clock().astimezone(self._zone)
        if now.date() != self._day:
            self._work_out(now.date())
        if self._times is None:
            return self._dark_all_day
        rise, set_ = self._times
        return now < rise or now >= set_

    def _work_out(self, day: date) -> None:
        try:
            # Asked for one at a time: astral.sun.sun() also works out dusk, which
            # has no answer for weeks around midsummer in the far north, on dates
            # that have a sunrise and a sunset.
            self._times = (
                sunrise(self._observer, date=day, tzinfo=self._zone),
                sunset(self._observer, date=day, tzinfo=self._zone),
            )
        except ValueError:
            # The sun does not reach the horizon on this date: astral says so with
            # the ValueError of an arccosine out of its domain.
            self._times = None
            highest = elevation(self._observer, noon(self._observer, date=day, tzinfo=self._zone))
            self._dark_all_day = highest < 0.0
            _log.info(
                "the sun does not rise or set on %s: dark all day is %s", day, self._dark_all_day
            )
        self._day = day
