"""What the lights do about what the camera sees.

A light is wanted on while any of its zones is active, and for ``off_after_seconds``
after the last of them comes clear or one of its lines is crossed. A light that is
``only_after_dark`` is not switched on in daylight.

Switching is the hub's job, and :class:`Lights` asks it from a thread of its own so
that a slow hub never holds up the camera. Two rules keep it from fighting the
people in the house:

* **It only switches off what it switched on.** Before switching a light on it
  reads the relay, and if somebody has it on already, it leaves it alone, then and
  when its triggers end. Before switching off, it reads the relay again: a light
  somebody has switched off in the meantime stays off, and is not switched again
  until the triggers end and start again.
* **A late "on" is worse than none.** Lighting an empty yard a minute after the car
  has gone is a bug, so switching on is retried for :data:`ON_WINDOW` and then
  given up. Switching off is retried for :data:`OFF_PATIENCE`.

A request that gets no answer may still have been done: the hub can switch the relay
and the answer be lost on the way back. So an "on" that went unanswered counts as
this program's if the relay turns out to be on, rather than as somebody else's.

A 401 stops every request until the program restarts: the hub counts failed keys
per address, and trying a wrong key again would lock out every client behind the
same router. A 404 or another refusal stops that light only, and a 429 pauses all
of them for :data:`THROTTLE_PAUSE`.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from pihome_vision.config import Light
from pihome_vision.hub import HubError, Refusal, Relays
from pihome_vision.triggers import Event

#: Seconds after a light was wanted on that switching it on is still worth trying.
ON_WINDOW: Final = 10.0
#: Seconds to keep trying to switch off a light this program switched on.
OFF_PATIENCE: Final = 300.0
#: Seconds before the first retry; each one after waits twice as long, up to
#: :data:`MAX_RETRY_WAIT`.
FIRST_RETRY_WAIT: Final = 0.5
MAX_RETRY_WAIT: Final = 30.0
#: Seconds to send nothing after the hub answers 429.
THROTTLE_PAUSE: Final = 60.0

_log = logging.getLogger(__name__)


class Darkness(Protocol):
    def is_dark(self) -> bool: ...


class Rule:
    """Whether one light is wanted on, frame by frame."""

    def __init__(self, light: Light) -> None:
        self.light = light
        self._triggers = frozenset(light.triggers)
        #: Until when a crossed line or a zone that came clear keeps it on.
        self._until = -math.inf
        self.on = False

    def update(
        self, events: Iterable[Event], active: frozenset[str], now: float, dark: Darkness | None
    ) -> bool:
        """Whether the light is wanted on at ``now``, given this frame's ``events`` and
        the zones ``active`` after it."""
        for event in events:
            if event.trigger in self._triggers and event.change != "active":
                self._until = max(self._until, event.at + self.light.off_after_seconds)
        triggered = bool(active & self._triggers) or now < self._until
        if not triggered:
            self.on = False
        elif not self.on:
            # Asked only now, so a light that came on at dusk stays on past dawn
            # until its triggers end, and one triggered all afternoon comes on at
            # sunset if they still have not.
            self.on = not self.light.only_after_dark or (dark is not None and dark.is_dark())
        return self.on


@dataclass(slots=True)
class _Switch:
    """One relay, as the thread that talks to the hub sees it."""

    relay: str
    #: Switched on by this program, and not switched off since.
    lit: bool = False
    #: Refused for good: no relay by that name, or a request it will never accept.
    disabled: bool = False
    #: The state to put it in, or None when there is nothing to do.
    wanted: bool | None = None
    #: When it was wanted, on the clock :class:`Lights` was given.
    since: float = 0.0
    #: When to try next.
    due: float = 0.0
    wait: float = FIRST_RETRY_WAIT
    #: Whether the hub being unavailable has been logged since the last success.
    warned: bool = False
    #: The state last asked of the hub that did not say whether it was done (no
    #: answer, or a failure on its side), which the relay may or may not be in now.
    #: None once a request has been answered.
    unanswered: bool | None = None

    @property
    def maybe_lit(self) -> bool:
        """Lit by this program, or perhaps lit by an "on" that went unanswered."""
        return self.lit or self.unanswered is True


class Lights:
    """Every light, switched through ``relays`` as the camera's triggers come and go.

    :meth:`update` is called with every frame's events, and only decides. A thread
    started by :meth:`start` does the switching; :meth:`stop` ends it and switches off
    whatever is still lit by this program.
    """

    def __init__(
        self,
        lights: Sequence[Light],
        relays: Relays,
        *,
        darkness: Darkness | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rules = [Rule(light) for light in lights]
        self._switches = {light.relay: _Switch(light.relay) for light in lights}
        self._relays = relays
        self._darkness = darkness
        self._clock = clock
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._dirty = False
        self._stopping = False
        #: Why nothing is sent any more, once the key has been turned down.
        self._halted: HubError | None = None
        self._paused_until = -math.inf
        self._thread = threading.Thread(target=self._run, name="lights", daemon=True)

    @property
    def lit(self) -> frozenset[str]:
        """The relays this program has switched on and not yet off."""
        with self._lock:
            return frozenset(s.relay for s in self._switches.values() if s.lit)

    def update(self, events: Sequence[Event], active: frozenset[str], now: float) -> None:
        """Decide with one frame's ``events`` and the zones ``active`` after it.

        Call it on every frame, events or not: a light goes off when time passes.
        """
        changed: list[tuple[str, bool]] = []
        for rule in self._rules:
            before = rule.on
            if rule.update(events, active, now, self._darkness) != before:
                changed.append((rule.light.relay, rule.on))
        if not changed:
            return
        with self._changed:
            at = self._clock()
            for relay, wanted in changed:
                switch = self._switches[relay]
                switch.wanted, switch.since, switch.due = wanted, at, at
                switch.wait = FIRST_RETRY_WAIT
            self._dirty = True
            self._changed.notify()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """End the thread, then switch off every light this program left on.

        Each is read first, as always, so one somebody has switched off meanwhile is
        not switched again.
        """
        with self._changed:
            self._stopping = True
            self._changed.notify()
        if self._thread.is_alive():
            self._thread.join()
        for switch in self._switches.values():
            if switch.maybe_lit and not switch.disabled and self._halted is None:
                try:
                    switch.lit = self._switch_off(switch)
                except HubError as exc:
                    _log.error("could not switch %s off on the way out: %s", switch.relay, exc)

    def pump(self) -> float | None:
        """Try every switch that is due, and say how many seconds until the next one
        is, or None if none is waiting."""
        while (task := self._next_due()) is not None:
            switch, wanted, since = task
            try:
                lit = self._apply(switch, wanted=wanted)
            except HubError as exc:
                self._refused(switch, since, exc)
                continue
            with self._lock:
                switch.lit = lit
                switch.warned = False
                # Unless it was wanted otherwise while the hub was being asked.
                if switch.since == since:
                    switch.wanted = None
        with self._lock:
            waiting = [s.due for s in self._switches.values() if self._waiting(s)]
            if not waiting:
                return None
            return max(0.0, max(min(waiting), self._paused_until) - self._clock())

    def _waiting(self, switch: _Switch) -> bool:
        return switch.wanted is not None and not switch.disabled and self._halted is None

    def _next_due(self) -> tuple[_Switch, bool, float] | None:
        with self._lock:
            now = self._clock()
            if now < self._paused_until:
                return None
            for switch in self._switches.values():
                if not self._waiting(switch) or switch.due > now:
                    continue
                assert switch.wanted is not None  # noqa: S101 - checked by _waiting
                if switch.wanted and now - switch.since > ON_WINDOW:
                    _log.warning(
                        "gave up switching %s on: %.0f s late, it would light nobody",
                        switch.relay,
                        now - switch.since,
                    )
                    switch.wanted = None
                    continue
                if not switch.wanted and not switch.maybe_lit:
                    # Somebody else's light, or one already off.
                    switch.wanted = None
                    continue
                if not switch.wanted and now - switch.since > OFF_PATIENCE:
                    _log.error(
                        "gave up switching %s off after %.0f s; it may still be on",
                        switch.relay,
                        now - switch.since,
                    )
                    switch.wanted = None
                    switch.lit = False
                    switch.unanswered = None
                    continue
                return switch, switch.wanted, switch.since
        return None

    def _apply(self, switch: _Switch, *, wanted: bool) -> bool:
        """Put ``switch`` in the ``wanted`` state if that is this program's to do, and
        say whether it is lit by this program afterwards."""
        if not wanted:
            return self._switch_off(switch)
        if self._relays.read(switch.relay):
            if switch.unanswered is True:
                _log.info("%s is on: the request that went unanswered got there", switch.relay)
                switch.unanswered = None
                return True
            if not switch.lit:
                _log.info("%s is on already; leaving it to whoever switched it on", switch.relay)
            return switch.lit
        lit = self._send(switch, on=True)
        _log.info("switched %s on", switch.relay)
        return lit

    def _switch_off(self, switch: _Switch) -> bool:
        if self._relays.read(switch.relay):
            self._send(switch, on=False)
            _log.info("switched %s off", switch.relay)
        else:
            if switch.unanswered is False:
                _log.info("%s is off: the request that went unanswered got there", switch.relay)
            else:
                _log.info("%s is off already: somebody switched it off", switch.relay)
            switch.unanswered = None
        return False

    def _send(self, switch: _Switch, *, on: bool) -> bool:
        """Ask the hub to switch ``switch`` and say whether it is on, remembering a
        request that got no answer: the hub may well have done it anyway."""
        try:
            done = self._relays.switch(switch.relay, on=on)
        except HubError as exc:
            if exc.refusal is Refusal.UNAVAILABLE:
                switch.unanswered = on
            raise
        switch.unanswered = None
        return done

    def _refused(self, switch: _Switch, since: float, exc: HubError) -> None:
        with self._lock:
            now = self._clock()
            match exc.refusal:
                case Refusal.UNAUTHORIZED:
                    self._halted = exc
                    _log.error(
                        "%s; sending the hub nothing more until restarted, so that a wrong "
                        "key does not get this address locked out. Check PIHOME_VISION_HUB_KEY",
                        exc,
                    )
                case Refusal.NO_SUCH_RELAY | Refusal.REJECTED:
                    switch.disabled = True
                    _log.error(
                        "%s: %s; that light is left alone until restarted", switch.relay, exc
                    )
                case Refusal.THROTTLED:
                    self._paused_until = now + THROTTLE_PAUSE
                    _log.warning("%s; trying again in %.0f s", exc, THROTTLE_PAUSE)
                case Refusal.UNAVAILABLE:
                    if switch.since == since:
                        switch.due = now + switch.wait
                        switch.wait = min(switch.wait * 2, MAX_RETRY_WAIT)
                    if not switch.warned:
                        switch.warned = True
                        _log.warning("could not switch %s: %s; trying again", switch.relay, exc)

    def _run(self) -> None:
        while True:
            wait = self.pump()
            with self._changed:
                self._changed.wait_for(lambda: self._dirty or self._stopping, timeout=wait)
                if self._stopping:
                    return
                self._dirty = False
