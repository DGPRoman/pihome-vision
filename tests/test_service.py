from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from pihome_vision import __main__, camera, detect, service
from pihome_vision.camera import Picture
from pihome_vision.config import Camera
from pihome_vision.detect import Detection, Frame, ModelError
from pihome_vision.lights import Lights
from pihome_vision.service import MAX_SKIP, STATS_SECONDS, Pipeline, Stillness
from tests.conftest import CAMERA_URL, HUB_KEY, Plan
from tests.fake_hub import FakeHub

RELAY = "gate-light"

#: The whole frame, so that anything detected is in it.
EVERYWHERE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def picture(value: int, width: int = 64, height: int = 48) -> Frame:
    return np.full((height, width, 3), value, dtype=np.uint8)


def seen(value: int) -> Picture:
    """:func:`picture` as the camera hands it out."""
    return Picture.from_bgr(picture(value))


#: Somebody standing in the middle of any frame at least 4 by 2.
SOMEBODY = Detection(1, 0, 2, 1, 0.9, "person")


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Frames:
    """Frames handed out one a call, each a step of the clock apart, then none."""

    def __init__(self, clock: Clock, frames: list[Frame], step: float = 0.2) -> None:
        self.clock = clock
        self.frames = frames
        self.step = step
        self.frames_received = 0

    def next_frame(self, after: int, timeout: float) -> tuple[int, Picture] | None:
        if self.frames_received >= len(self.frames):
            self.clock.now += timeout
            return None
        self.clock.now += self.step
        self.frames_received += 1
        return self.frames_received, Picture.from_bgr(self.frames[self.frames_received - 1])


class Counting:
    def __init__(self, found: list[Detection]) -> None:
        self.found = found
        self.calls = 0

    def detect(self, frame: Frame) -> list[Detection]:
        self.calls += 1
        return self.found


class NoRelays:
    def read(self, relay: str) -> bool:
        raise AssertionError

    def switch(self, relay: str, *, on: bool) -> bool:
        raise AssertionError


def gate(**overrides: object) -> Camera:
    zone: dict[str, object] = {"kind": "zone", "id": "yard", "polygon": EVERYWHERE}
    zone |= {"classes": ["person"], "min_seconds": 0, "clear_seconds": 1}
    return Camera.model_validate({"id": "gate", "triggers": [zone]} | overrides)


def pipeline(
    clock: Clock, frames: list[Frame], detector: Counting, **overrides: object
) -> tuple[Pipeline, Frames]:
    source = Frames(clock, frames)
    lights = Lights([], NoRelays())
    return Pipeline(gate(**overrides), source, detector, lights, clock=clock), source


class TestStillness:
    def test_with_no_threshold_every_frame_is_detected_on(self) -> None:
        still = Stillness(0)

        assert all(still.changed(seen(50), t) for t in range(5))

    def test_a_still_picture_is_skipped(self) -> None:
        still = Stillness(2)

        assert still.changed(seen(50), 0.0)
        assert not still.changed(seen(50), 0.2)
        assert not still.changed(seen(51), 0.4)

    def test_a_change_is_detected_on(self) -> None:
        still = Stillness(2)
        still.changed(seen(50), 0.0)

        assert still.changed(seen(60), 0.2)

    @pytest.mark.parametrize(("threshold", "changed"), [(18, True), (22, False)])
    def test_the_threshold_is_in_levels_of_grey(self, threshold: float, changed: bool) -> None:
        """20 levels of grey are 17 steps of the camera's brightness, which runs 16-235."""
        still = Stillness(threshold)
        still.changed(seen(50), 0.0)

        assert still.changed(seen(70), 0.2) is changed

    def test_a_slow_change_adds_up(self) -> None:
        """Each frame one level brighter than the last, which alone is not enough."""
        still = Stillness(2)
        still.changed(seen(50), 0.0)

        assert [still.changed(seen(50 + n), n * 0.2) for n in range(1, 4)] == [
            False,
            True,
            False,
        ]

    def test_however_still_it_is_looked_at_now_and_then(self) -> None:
        still = Stillness(2)
        still.changed(seen(50), 0.0)

        assert not still.changed(seen(50), MAX_SKIP - 0.1)
        assert still.changed(seen(50), MAX_SKIP)


class TestPipeline:
    def test_every_frame_goes_to_the_model_by_default(self) -> None:
        clock, detector = Clock(), Counting([])
        run, _ = pipeline(clock, [picture(50)] * 5, detector)

        for _ in range(5):
            run.step()

        assert detector.calls == 5

    def test_a_still_scene_keeps_what_was_found_in_it(self) -> None:
        """A parked car is still parked while the picture does not change."""
        clock, detector = Clock(), Counting([SOMEBODY])
        run, _ = pipeline(clock, [picture(50)] * 20, detector, motion_threshold=2)

        for _ in range(20):
            run.step()

        assert detector.calls == 1
        assert run._watcher.active == {"yard"}

    def test_a_camera_gone_quiet_clears_its_zones(self, caplog: pytest.LogCaptureFixture) -> None:
        clock, detector = Clock(), Counting([SOMEBODY])
        run, _ = pipeline(clock, [picture(50)] * 3, detector)

        with caplog.at_level(logging.INFO, logger="pihome_vision.service"):
            for _ in range(6):
                run.step()

        assert run._watcher.active == set()
        assert "gate/yard: active" in caplog.text
        assert "gate/yard: clear" in caplog.text

    def test_it_says_how_it_is_doing_now_and_then(self, caplog: pytest.LogCaptureFixture) -> None:
        clock, detector = Clock(), Counting([SOMEBODY])
        frames = [picture(50)] * int(STATS_SECONDS / 0.2 + 1)
        run, _ = pipeline(clock, frames, detector)

        with caplog.at_level(logging.INFO, logger="pihome_vision.service"):
            for _ in frames:
                run.step()

        (line,) = [r.message for r in caplog.records if "frames/s" in r.message]
        assert line.startswith("gate: 5.0 frames/s from the camera, 5.0 examined, 5.0 detected")
        assert line.endswith("active: yard; lit: none")


# The whole service, through the command line: the stand-in ffmpeg as the camera, a
# model that sees somebody in the frames of a chosen brightness, and a stand-in hub.


def brightness(frame: Frame) -> int:
    return int(frame[0, 0, 0])


class SeesSomebody:
    """Somebody is in the frames the stand-in ffmpeg writes from about the 40th to
    about the 120th: their brightness is between 30 and 120."""

    def __init__(self, path: Path, **_: object) -> None:
        pass

    def detect(self, frame: Frame) -> list[Detection]:
        return [SOMEBODY] if 30 <= brightness(frame) <= 120 else []


class FailsLater(SeesSomebody):
    """Runs on the first frame, and fails as OpenCV short of memory would on the next."""

    def __init__(self, path: Path, **_: object) -> None:
        self.ran = False

    def detect(self, frame: Frame) -> list[Detection]:
        if self.ran:
            msg = "Insufficient memory"
            raise cv2.error(msg)
        self.ran = True
        return []


class Broken(SeesSomebody):
    def detect(self, frame: Frame) -> list[Detection]:
        msg = "the model would not run on a 640x640 input"
        raise ModelError(msg)


@pytest.fixture
def started(
    hub: FakeHub, plan: Plan, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., None]:
    """The environment and configuration for one light that follows one zone, and a
    camera playing the given run."""

    def start(run: dict[str, object], detector: type | None = SeesSomebody) -> None:
        config = tmp_path / "vision.yaml"
        config.write_text(
            f"""
model:
  path: {tmp_path / "model.onnx"}
cameras:
  - id: gate
    triggers:
      - kind: zone
        id: yard
        polygon: {EVERYWHERE}
        classes: [person]
        min_seconds: 0
        clear_seconds: 0.1
lights:
  - relay: {RELAY}
    triggers: [yard]
    off_after_seconds: 0.3
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("PIHOME_VISION_CAMERA_URL", CAMERA_URL)
        monkeypatch.setenv("PIHOME_VISION_HUB_URL", hub.origin)
        monkeypatch.setenv("PIHOME_VISION_HUB_KEY", HUB_KEY)
        monkeypatch.setenv("PIHOME_VISION_CONFIG_PATH", str(config))
        monkeypatch.setattr(camera, "FFMPEG", plan(run))
        # Once the camera has nothing more, each step waits this long for a frame
        # before zones come clear and lights go off: a second of it, as in service,
        # would be most of what these tests take.
        monkeypatch.setattr(service, "FRAME_WAIT", 0.05)
        if detector is not None:
            monkeypatch.setattr(detect, "Detector", detector)

    return start


#: When the last SIGTERM was sent, on the monotonic clock.
SENT: list[float] = []


@pytest.fixture
def terminate() -> Iterator[Callable[[Callable[[], bool]], None]]:
    """Sends this process SIGTERM once a condition holds, or after ten seconds."""
    threads: list[threading.Thread] = []

    def when(condition: Callable[[], bool]) -> None:
        def wait() -> None:
            deadline = time.monotonic() + 10
            while not condition() and time.monotonic() < deadline:
                time.sleep(0.01)
            SENT.append(time.monotonic())
            os.kill(os.getpid(), signal.SIGTERM)

        threads.append(threading.Thread(target=wait, daemon=True))
        threads[-1].start()

    yield when
    for thread in threads:
        thread.join()


def test_somebody_walking_through_switches_the_light_on_once_and_off_once(
    started: Callable[..., None],
    terminate: Callable[[Callable[[], bool]], None],
    hub: FakeHub,
) -> None:
    started({"frames": 140, "interval": 0.005, "then": "hang"})
    terminate(lambda: len(hub.puts) >= 2)

    assert __main__.main(["run"]) == 0
    assert hub.puts == [(RELAY, True), (RELAY, False)]


def test_stopping_switches_its_light_off(
    started: Callable[..., None],
    terminate: Callable[[Callable[[], bool]], None],
    hub: FakeHub,
) -> None:
    """Somebody stands in the yard for good, and the service is stopped."""
    started({"frames": 100, "interval": 0.01, "then": "hang"})
    terminate(lambda: hub.relays[RELAY])

    assert __main__.main(["run"]) == 0
    assert time.monotonic() - SENT[-1] < 3
    assert hub.puts == [(RELAY, True), (RELAY, False)]
    assert hub.relays[RELAY] is False


def test_a_light_somebody_else_switched_on_is_left_on(
    started: Callable[..., None],
    terminate: Callable[[Callable[[], bool]], None],
    hub: FakeHub,
) -> None:
    hub.relays[RELAY] = True
    started({"frames": 100, "interval": 0.01, "then": "hang"})
    terminate(lambda: len(hub.requests) >= 1)

    assert __main__.main(["run"]) == 0
    assert hub.puts == []
    assert hub.relays[RELAY] is True


def test_a_model_that_will_not_run_exits_2(
    started: Callable[..., None],
    terminate: Callable[[Callable[[], bool]], None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    started({"frames": 100, "interval": 0.01, "then": "hang"}, detector=Broken)

    with caplog.at_level(logging.ERROR):
        assert __main__.main(["run"]) == 2

    assert "would not run" in caplog.text


def test_a_model_that_fails_after_running_exits_1_to_be_restarted(
    started: Callable[..., None], caplog: pytest.LogCaptureFixture
) -> None:
    """Not the configuration: a service manager should try again."""
    started({"frames": 100, "interval": 0.01, "then": "hang"}, detector=FailsLater)

    with caplog.at_level(logging.ERROR):
        assert __main__.main(["run"]) == 1

    assert "the pipeline failed" in caplog.text
    assert "Insufficient memory" in caplog.text


def test_a_missing_model_exits_2_before_starting(
    started: Callable[..., None], hub: FakeHub, capsys: pytest.CaptureFixture[str]
) -> None:
    started({"frames": 100}, detector=None)

    assert __main__.main(["run"]) == 2
    assert "no model" in capsys.readouterr().err
    assert hub.requests == []


class Systemd:
    """Listens where the service is told systemd does, and keeps what it hears and when."""

    def __init__(self, name: str) -> None:
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.bind(f"\0{name}")
        self._socket.settimeout(0.02)
        self._closed = threading.Event()
        self.heard: list[tuple[float, str]] = []
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self) -> None:
        while not self._closed.is_set():
            try:
                said = self._socket.recv(256)
            except TimeoutError:
                continue
            self.heard.append((time.monotonic(), said.decode()))

    @property
    def states(self) -> list[str]:
        return [state for _, state in self.heard]

    def alive(self, after: float = -1.0, before: float = float("inf")) -> int:
        """How many times it heard the service was alive between two moments."""
        return sum(1 for at, state in self.heard if state == "WATCHDOG=1" and after < at < before)

    def hear(self, state: str) -> None:
        deadline = time.monotonic() + 5
        while state not in self.states and time.monotonic() < deadline:
            time.sleep(0.01)

    def close(self) -> None:
        self._closed.set()
        self._thread.join()
        self._socket.close()


#: How often the service is asked to say it is alive, in seconds. It looks in on the
#: pipeline four times in that, and vouches for it if it moved in the last half.
WATCHDOG = 0.4


@pytest.fixture
def systemd(monkeypatch: pytest.MonkeyPatch) -> Iterator[Systemd]:
    listener = Systemd(f"pihome-vision-test-{os.getpid()}")
    monkeypatch.setenv("NOTIFY_SOCKET", f"@pihome-vision-test-{os.getpid()}")
    monkeypatch.setenv("WATCHDOG_USEC", str(int(WATCHDOG * 1_000_000)))
    yield listener
    listener.close()


def test_systemd_hears_it_is_ready_then_alive_then_stopping(
    started: Callable[..., None],
    terminate: Callable[[Callable[[], bool]], None],
    systemd: Systemd,
) -> None:
    started({"frames": 100, "interval": 0.01, "then": "hang"})
    terminate(lambda: systemd.alive() >= 3)

    assert __main__.main(["run"]) == 0
    systemd.hear("STOPPING=1")
    assert systemd.states[0] == "READY=1"
    assert systemd.states[-1] == "STOPPING=1"
    assert set(systemd.states[1:-1]) == {"WATCHDOG=1"}


#: When the stand-in model below hung, on the monotonic clock, and what lets it go.
HUNG: list[float] = []
UNSTICK = threading.Event()


class HangsOnTheSecondLook(SeesSomebody):
    """Sees somebody once, and the next time hangs, as a model run that never returns
    would, until let go."""

    def __init__(self, path: Path, **_: object) -> None:
        self.seen = 0

    def detect(self, frame: Frame) -> list[Detection]:
        found = super().detect(frame)
        if found:
            self.seen += 1
        if self.seen == 2:  # the second look
            HUNG.append(time.monotonic())
            UNSTICK.wait()
        return found


def test_a_hung_pipeline_is_not_vouched_for_and_stopping_still_darkens_its_light(
    started: Callable[..., None],
    terminate: Callable[[Callable[[], bool]], None],
    hub: FakeHub,
    systemd: Systemd,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    HUNG.clear()
    UNSTICK.clear()
    monkeypatch.setattr(service, "STOP_GRACE", 0.2)
    started({"frames": 100, "interval": 0.01, "then": "hang"}, detector=HangsOnTheSecondLook)
    # Long enough after hanging for the last word to have gone, and as long again.
    terminate(lambda: bool(HUNG) and time.monotonic() - HUNG[0] > WATCHDOG * 1.5)

    try:
        assert __main__.main(["run"]) == 1
    finally:
        UNSTICK.set()
    assert systemd.alive(before=HUNG[0]) >= 1
    assert systemd.alive(after=HUNG[0] + WATCHDOG * 0.75) == 0
    assert hub.puts == [(RELAY, True), (RELAY, False)]
