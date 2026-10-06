"""The windows and the commands that open them, with OpenCV's window functions replaced
by a scripted screen, so the same tests run on the headless build and need no display."""

from __future__ import annotations

import stat
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from pihome_vision import __main__, camera, detect, gui
from pihome_vision.config import Camera, load_config
from pihome_vision.detect import Detection, Frame
from pihome_vision.sketch import ENTER, ESCAPE
from tests.conftest import CAMERA_PASSWORD, EXAMPLE_CONFIG, Plan

#: A click at a pixel, a key, the window closed from its title bar, or nothing until
#: a condition holds.
Event = tuple[str, int, int] | tuple[str, int] | tuple[str] | tuple[str, Callable[[], bool]]


class Screen:
    """Stands in for OpenCV's window functions, playing a script of events."""

    def __init__(self, *events: Event) -> None:
        self.events: deque[Event] = deque(events)
        self.shown: list[Frame] = []
        self.on_mouse: Callable[..., None] | None = None
        self.open = False
        self.waits = 0
        self.size = (0, 0)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISPLAY", ":99")
        monkeypatch.setattr(cv2, "namedWindow", self.named_window)
        monkeypatch.setattr(cv2, "setMouseCallback", self.set_mouse_callback)
        monkeypatch.setattr(cv2, "imshow", self.imshow)
        monkeypatch.setattr(cv2, "waitKey", self.wait_key)
        monkeypatch.setattr(cv2, "getWindowProperty", self.get_window_property)
        monkeypatch.setattr(cv2, "destroyWindow", self.destroy_window)
        monkeypatch.setattr(cv2, "resizeWindow", self.resize_window)

    def named_window(self, name: str, flags: int) -> None:
        assert name == gui.WINDOW
        self.open = True

    def resize_window(self, name: str, width: int, height: int) -> None:
        self.size = (width, height)

    def set_mouse_callback(self, name: str, callback: Callable[..., None]) -> None:
        self.on_mouse = callback

    def imshow(self, name: str, image: Frame) -> None:
        assert self.open
        self.shown.append(image)

    def wait_key(self, delay: int) -> int:
        self.waits += 1
        assert self.waits < 1000, "the window never finished"
        time.sleep(delay / 10_000)
        if not self.events:
            return -1
        event = self.events[0]
        match event:
            case ("click", int(x), int(y)):
                assert self.on_mouse is not None
                self.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
            case ("key", int(code)):
                self.events.popleft()
                return code
            case ("close",):
                self.open = False
            case ("until", condition) if callable(condition):
                if not condition():
                    return -1
        self.events.popleft()
        return -1

    def get_window_property(self, name: str, prop: int) -> float:
        assert prop == cv2.WND_PROP_VISIBLE
        return 1.0 if self.open else 0.0

    def destroy_window(self, name: str) -> None:
        self.open = False


def keys(text: str) -> list[Event]:
    return [("key", ord(character)) for character in text]


def frame(width: int = 200, height: int = 100) -> Frame:
    return np.full((height, width, 3), 90, dtype=np.uint8)


class TestEdit:
    def test_what_is_drawn_comes_back_with_what_was_there(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        screen = Screen(
            *[("click", x, y) for x, y in [(20, 10), (100, 10), (100, 50)]],
            *keys("zporch"),
            ("key", ENTER[1]),
            ("key", ord("q")),
        )
        screen.install(monkeypatch)
        (existing,) = load_config(EXAMPLE_CONFIG).cameras[0].triggers[:1]

        drawn = gui.edit(frame(), [existing])

        assert [t.id for t in drawn] == ["driveway", "porch"]
        assert drawn[1].kind == "zone"
        assert drawn[1].polygon == [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)]
        assert not screen.open
        assert screen.size == (200, 100)
        # The last picture shown has the new zone on it, and says what to do next.
        assert not (screen.shown[-1] == frame()).all()

    def test_a_big_frame_opens_a_window_that_fits_a_screen(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        screen = Screen(("key", ord("q")))
        screen.install(monkeypatch)

        gui.edit(frame(3840, 2160), [])

        assert screen.size == (1280, 720)

    def test_closing_the_window_finishes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = Screen(("click", 20, 10), ("close",))
        screen.install(monkeypatch)

        assert gui.edit(frame(), []) == []

    def test_with_no_display_it_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        Screen().install(monkeypatch)
        monkeypatch.delenv("DISPLAY")
        monkeypatch.setattr(sys, "platform", "linux")

        with pytest.raises(gui.NoWindowError, match="no display"):
            gui.edit(frame(), [])

    def test_the_headless_build_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        Screen().install(monkeypatch)

        def no_windows(name: str, flags: int) -> None:
            raise cv2.error("The function is not implemented")

        monkeypatch.setattr(cv2, "namedWindow", no_windows)

        with pytest.raises(gui.NoWindowError, match=r"requirements/gui\.txt"):
            gui.edit(frame(), [])


class Frames:
    def __init__(self, frames: list[Frame]) -> None:
        self.frames = frames
        self.frames_received = 0

    def next_frame(self, after: int, timeout: float) -> tuple[int, Frame] | None:
        if self.frames_received >= len(self.frames):
            return None
        self.frames_received += 1
        return self.frames_received, self.frames[self.frames_received - 1]


class Walker:
    """Somebody walking from the left of the frame to the right, a step a frame."""

    def __init__(self, *_: object, **__: object) -> None:
        self.calls = 0

    def detect(self, image: Frame) -> list[Detection]:
        self.calls += 1
        height, width = image.shape[:2]
        x = width * (0.05 * self.calls)
        return [Detection(x, height * 0.4, width * 0.05, height * 0.5, 0.8, "person")]


def gate() -> Camera:
    return Camera.model_validate(
        {
            "id": "gate",
            "triggers": [
                {
                    "kind": "line",
                    "id": "wicket",
                    "points": [[0.5, 0.0], [0.5, 1.0]],
                    "classes": ["person"],
                }
            ],
        }
    )


class TestPreview:
    def test_each_frame_is_shown_with_what_was_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = Screen(*[("wait",)] * 25, ("key", ord("q")))
        screen.install(monkeypatch)
        clock = iter(range(1000))

        gui.preview(gate(), Frames([frame()] * 20), Walker(), clock=lambda: next(clock) / 10)

        assert len(screen.shown) == 20
        assert not screen.open

    def test_a_crossed_line_is_lit_for_a_moment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = Screen(*[("wait",)] * 35, ("key", ESCAPE))
        screen.install(monkeypatch)
        clock = iter(range(1000))

        gui.preview(gate(), Frames([frame()] * 30), Walker(), clock=lambda: next(clock) / 10)

        # The walker crosses the middle around the tenth frame. Red is drawn then,
        # and not a second later.
        line_at = (50, 100)
        red = [bool(shown[line_at][2] == 255 and shown[line_at][0] == 0) for shown in screen.shown]
        assert any(red)
        assert not red[-1]

    def test_closing_the_window_finishes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        screen = Screen(("wait",), ("close",))
        screen.install(monkeypatch)

        gui.preview(gate(), Frames([frame()] * 100), Walker())

        assert len(screen.shown) <= 3


# The commands.


@pytest.fixture
def picture(tmp_path: Path) -> Path:
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), frame())
    return path


@pytest.mark.usefixtures("environment")
def test_edit_prints_the_cameras_section_for_vision_yaml(
    picture: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    Screen(
        ("click", 20, 90),
        ("click", 180, 90),
        *keys("lpavement"),
        ("key", ENTER[0]),
        ("key", ord("q")),
    ).install(monkeypatch)

    assert __main__.main(["edit", "--image", str(picture)]) == 0

    captured = capsys.readouterr()
    assert captured.out.startswith("# The cameras section for ")
    (watched,) = yaml.safe_load(captured.out)["cameras"]
    assert [t["id"] for t in watched["triggers"]] == ["driveway", "wicket", "pavement"]
    assert watched["triggers"][2]["points"] == [[0.1, 0.9], [0.9, 0.9]]
    assert captured.err == ""


@pytest.mark.usefixtures("environment")
def test_edit_warns_of_a_light_left_without_its_trigger(
    picture: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    Screen(("key", 8), ("key", ord("q"))).install(monkeypatch)

    assert __main__.main(["edit", "--image", str(picture)]) == 0

    captured = capsys.readouterr()
    assert "wicket" not in captured.out
    assert captured.err == "pihome-vision: light gate-light: wicket is no longer drawn\n"


@pytest.mark.usefixtures("environment")
def test_edit_with_nothing_drawn_prints_nothing(
    picture: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    Screen(("key", 8), ("key", 8), ("key", ord("q"))).install(monkeypatch)

    assert __main__.main(["edit", "--image", str(picture)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "a camera needs a trigger" in captured.err


@pytest.mark.usefixtures("environment")
def test_edit_without_windows_exits_1(
    picture: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    assert __main__.main(["edit", "--image", str(picture)]) == 1
    assert "no display" in capsys.readouterr().err


@pytest.mark.usefixtures("environment")
def test_edit_draws_over_a_frame_from_the_camera(
    plan: Plan, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(camera, "FFMPEG", plan({"frames": 3}))
    screen = Screen(("key", ord("q")))
    screen.install(monkeypatch)

    assert __main__.main(["edit"]) == 0

    assert screen.shown[0].shape == (2, 4, 3)
    assert CAMERA_PASSWORD not in capsys.readouterr().out


def test_edit_needs_a_configuration(
    picture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PIHOME_VISION_CONFIG_PATH", str(tmp_path / "missing.yaml"))

    assert __main__.main(["edit", "--image", str(picture)]) == 2


@pytest.mark.usefixtures("environment")
def test_preview_shows_the_camera_until_q(plan: Plan, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera, "FFMPEG", plan({"frames": 50, "interval": 0.01, "then": "hang"}))
    monkeypatch.setattr(detect, "Detector", Walker)
    screen = Screen()
    screen.events.extend([("until", lambda: len(screen.shown) >= 3), ("key", ord("q"))])
    screen.install(monkeypatch)

    assert __main__.main(["preview"]) == 0
    assert len(screen.shown) >= 3


@pytest.mark.usefixtures("environment")
def test_preview_without_a_model_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert __main__.main(["preview"]) == 2
    assert "docs/models.md" in capsys.readouterr().err


@pytest.mark.usefixtures("environment")
def test_snapshot_saves_a_frame_only_its_owner_can_read(
    plan: Plan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(camera, "FFMPEG", plan({"frames": 3}))
    path = tmp_path / "gate.png"

    assert __main__.main(["snapshot", str(path)]) == 0

    assert capsys.readouterr().out == f"{path}: 4x2\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    saved = cv2.imread(str(path))
    assert saved is not None
    assert saved.shape == (2, 4, 3)


@pytest.mark.usefixtures("environment")
def test_snapshot_takes_the_last_frame_of_its_first_second(
    plan: Plan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(__main__, "SNAPSHOT_SECONDS", 0.2)
    monkeypatch.setattr(camera, "FFMPEG", plan({"frames": 100, "interval": 0.01, "then": "hang"}))
    path = tmp_path / "gate.png"

    assert __main__.main(["snapshot", str(path)]) == 0

    saved = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert saved is not None
    # Frame n is filled with n. How many arrive in the time depends on the machine,
    # but the first, which may be half drawn, is not the one kept.
    assert int(saved[0, 0]) > 1


@pytest.mark.usefixtures("environment")
def test_snapshot_refuses_a_format_it_cannot_write(
    plan: Plan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(camera, "FFMPEG", plan({"frames": 3}))

    assert __main__.main(["snapshot", str(tmp_path / "gate.txt")]) == 1
    assert "cannot save a picture as .txt" in capsys.readouterr().err
    assert not (tmp_path / "gate.txt").exists()


@pytest.mark.usefixtures("environment")
def test_snapshot_says_why_the_camera_could_not_be_read(
    plan: Plan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        camera,
        "FFMPEG",
        plan({"header": False, "stderr": "{url}: Connection refused", "status": 1}),
    )

    assert __main__.main(["snapshot", str(tmp_path / "gate.png")]) == 1

    err = capsys.readouterr().err
    assert "the camera refused the connection" in err
    assert CAMERA_PASSWORD not in err
