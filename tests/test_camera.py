from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pihome_vision import camera
from pihome_vision.camera import (
    Camera,
    Failure,
    Stream,
    StreamError,
    Timeouts,
    classify,
    ffmpeg_arguments,
    to_bgr,
)
from tests import fake_ffmpeg
from tests.conftest import CAMERA_PASSWORD, CAMERA_URL, Plan

#: Short enough that a test about waiting does not wait long.
QUICK = Timeouts(open=0.5, stall=0.5, backoff=0.05)


class TestArguments:
    def test_rtsp_goes_over_tcp(self) -> None:
        arguments = ffmpeg_arguments(CAMERA_URL)

        assert arguments[arguments.index("-rtsp_transport") + 1] == "tcp"
        assert arguments.index("-rtsp_transport") < arguments.index("-i")
        assert arguments[arguments.index("-i") + 1] == CAMERA_URL

    def test_http_is_given_as_it_is(self) -> None:
        arguments = ffmpeg_arguments("http://192.168.1.50/video.mjpg")

        assert "-rtsp_transport" not in arguments
        assert arguments[arguments.index("-i") + 1] == "http://192.168.1.50/video.mjpg"

    def test_rtsp_is_not_buffered(self) -> None:
        arguments = ffmpeg_arguments(CAMERA_URL)
        given = arguments.index("-i")

        assert arguments[arguments.index("-fflags") + 1] == "nobuffer"
        assert arguments[arguments.index("-flags") + 1] == "low_delay"
        assert arguments.index("-fflags") < given
        assert arguments.index("-flags") < given

    def test_http_keeps_its_buffer(self) -> None:
        """Without one, ffmpeg decodes nothing from MPEG-TS, which some cameras send."""
        arguments = ffmpeg_arguments("http://192.168.1.50/video.ts")

        assert "-fflags" not in arguments
        assert arguments[arguments.index("-flags") + 1] == "low_delay"

    @pytest.mark.parametrize("source", [CAMERA_URL, "https://192.168.1.50/video.mjpg"])
    def test_a_network_camera_is_timed_by_arrival_not_its_own_clock(self, source: str) -> None:
        arguments = ffmpeg_arguments(source, fps=10)
        stamped = arguments.index("-use_wallclock_as_timestamps")

        assert arguments[stamped + 1] == "1"
        assert stamped < arguments.index("-i")

    @pytest.mark.parametrize("source", ["cam:0", "/var/lib/clips/gate.y4m"])
    def test_a_local_camera_or_a_file_keeps_its_own_timing(self, source: str) -> None:
        assert "-use_wallclock_as_timestamps" not in ffmpeg_arguments(source, fps=10)

    def test_a_local_camera_is_its_video_device(self) -> None:
        arguments = ffmpeg_arguments("cam:2")

        assert arguments[arguments.index("-i") - 2 : arguments.index("-i") + 2] == [
            "-f",
            "v4l2",
            "-i",
            "/dev/video2",
        ]

    def test_frames_are_yuv4mpeg_on_stdout(self) -> None:
        arguments = ffmpeg_arguments(CAMERA_URL)

        assert arguments[-3:] == ["-f", "yuv4mpegpipe", "pipe:1"]
        assert "-nostdin" in arguments

    def test_a_rate_selects_frames_and_passes_them_on_as_they_are(self) -> None:
        arguments = ffmpeg_arguments(CAMERA_URL, fps=2.5)

        assert arguments[arguments.index("-vf") + 1].startswith("select='")
        assert "floor(t*2.5)" in arguments[arguments.index("-vf") + 1]
        assert arguments[arguments.index("-fps_mode") + 1] == "passthrough"
        assert arguments.index("-fps_mode") > arguments.index("-i")

    def test_without_a_rate_every_frame_comes_through(self) -> None:
        arguments = ffmpeg_arguments(CAMERA_URL)

        assert "select" not in arguments[arguments.index("-vf") + 1]
        assert "-fps_mode" not in arguments


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs a real ffmpeg")
def test_a_rate_takes_the_first_frame_in_each_slot(tmp_path: Path) -> None:
    """At 10 a second from 25: one frame in each tenth of a second, each once."""
    clip = tmp_path / "numbered.y4m"
    # Two seconds at 25 frames a second, each as bright as its number.
    numbered = "color=black:size=64x64:rate=25,format=yuv420p,geq=lum='N':cb=128:cr=128"
    making = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", numbered]
    subprocess.run([*making, "-t", "2", str(clip)], check=True)  # noqa: S603 - fixed

    with Stream(str(clip), fps=10, timeouts=QUICK) as stream:
        numbers = [stream.read()[0] for _ in range(20)]

    assert numbers == [0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48]


@pytest.mark.parametrize(
    ("said", "failure"),
    [
        (
            "[in#0 @ 0x5794] Error opening input: Server returned 401 Unauthorized"
            " (authorization failed)",
            Failure.UNAUTHORIZED,
        ),
        ("[http @ 0x5a1] HTTP error 403 Forbidden", Failure.UNAUTHORIZED),
        ("[in#0 @ 0x570c] Error opening input: Server returned 404 Not Found", Failure.NOT_FOUND),
        (
            "[tcp @ 0x5a972fdb4040] Connection to tcp://192.168.1.50:554?timeout=0 failed:"
            " Connection refused",
            Failure.REFUSED,
        ),
        (
            "[tcp @ 0x5710] Connection to tcp://192.168.1.50:554 failed: Connection timed out",
            Failure.TIMEOUT,
        ),
        (
            "[tcp @ 0x5b99] Failed to resolve hostname camera.invalid: Name or service not known",
            Failure.UNRESOLVED,
        ),
        ("[tcp @ 0x5b99] Connection failed: No route to host", Failure.UNREACHABLE),
        (
            "[in#0 @ 0x5ec4] Cannot open video device /dev/video9: No such file or directory",
            Failure.NO_DEVICE,
        ),
        ("Error opening input: Invalid data found when processing input", Failure.ENDED),
    ],
)
def test_ffmpeg_errors_are_explained(said: str, failure: Failure) -> None:
    assert classify(said) is failure


def test_a_status_code_inside_an_address_is_not_one() -> None:
    assert classify("[rtsp @ 0x5a404] Invalid data found when processing input") is Failure.ENDED


class TestStream:
    def test_it_reads_the_size_and_rate_then_frames_in_order(self, plan: Plan) -> None:
        command = plan({"frames": 3})

        with Stream(CAMERA_URL, command=command, timeouts=QUICK) as stream:
            assert (stream.width, stream.height) == (fake_ffmpeg.WIDTH, fake_ffmpeg.HEIGHT)
            assert stream.rate == 25
            assert [stream.read() for _ in range(3)] == [fake_ffmpeg.frame(n) for n in (1, 2, 3)]

    def test_a_frame_converts_to_a_bgr_image(self) -> None:
        image = to_bgr(fake_ffmpeg.frame(200), fake_ffmpeg.WIDTH, fake_ffmpeg.HEIGHT)

        assert image.shape == (fake_ffmpeg.HEIGHT, fake_ffmpeg.WIDTH, 3)
        assert image.dtype == np.uint8
        assert image.flags.writeable

    def test_a_refusal_is_explained_without_the_password(self, plan: Plan) -> None:
        command = plan(
            {
                "header": False,
                "stderr": "Error opening input file {url}.\n"
                "Error opening input files: Server returned 401 Unauthorized",
                "status": 1,
            }
        )

        with pytest.raises(StreamError) as raised:
            Stream(CAMERA_URL, command=command, timeouts=QUICK)

        assert raised.value.failure is Failure.UNAUTHORIZED
        assert CAMERA_PASSWORD not in str(raised.value)

    def test_the_last_thing_ffmpeg_said_is_kept_without_the_password(self, plan: Plan) -> None:
        command = plan({"header": False, "stderr": "Error opening input file {url}.", "status": 1})

        with pytest.raises(StreamError) as raised:
            Stream(CAMERA_URL, command=command, timeouts=QUICK)

        assert "rtsp://***@192.168.1.50:554/stream2" in raised.value.detail
        assert CAMERA_PASSWORD not in raised.value.detail

    def test_a_camera_that_never_answers_times_out(self, plan: Plan) -> None:
        command = plan({"header": False, "then": "hang"})
        started = time.monotonic()

        with pytest.raises(StreamError) as raised:
            Stream(CAMERA_URL, command=command, timeouts=QUICK)

        assert raised.value.failure is Failure.TIMEOUT
        assert time.monotonic() - started < QUICK.open + 2

    def test_a_stream_that_goes_quiet_has_stalled(self, plan: Plan) -> None:
        command = plan({"frames": 1, "then": "hang"})

        with Stream(CAMERA_URL, command=command, timeouts=QUICK) as stream:
            stream.read()
            with pytest.raises(StreamError) as raised:
                stream.read()

        assert raised.value.failure is Failure.STALLED

    def test_a_stream_that_closes_has_ended(self, plan: Plan) -> None:
        command = plan({"frames": 1})

        with Stream(CAMERA_URL, command=command, timeouts=QUICK) as stream:
            stream.read()
            with pytest.raises(StreamError) as raised:
                stream.read()

        assert raised.value.failure is Failure.ENDED

    def test_output_that_is_not_video_is_refused(self, plan: Plan) -> None:
        command = plan({"garbage": True, "then": "hang"})

        with pytest.raises(StreamError) as raised:
            Stream(CAMERA_URL, command=command, timeouts=QUICK)

        assert raised.value.failure is Failure.UNREADABLE

    def test_no_ffmpeg_says_how_to_install_it(self, tmp_path: Path) -> None:
        with pytest.raises(StreamError) as raised:
            Stream(CAMERA_URL, command=[str(tmp_path / "ffmpeg")])

        assert raised.value.failure is Failure.NO_FFMPEG
        assert "apt install ffmpeg" in str(raised.value)

    def test_cancelling_ends_reading_even_with_frames_waiting(self, plan: Plan) -> None:
        """A camera that never pauses is still stopped."""
        command = plan({"frames": 100_000, "interval": 0.001})
        cancel = threading.Event()

        with Stream(CAMERA_URL, command=command, timeouts=QUICK, cancel=cancel) as stream:
            stream.read()
            cancel.set()
            with pytest.raises(StreamError, match="stopped"):
                stream.read()

    def test_cancelling_ends_a_wait_for_the_camera(self, plan: Plan) -> None:
        command = plan({"header": False, "then": "hang"})
        cancel = threading.Event()
        threading.Timer(0.1, cancel.set).start()
        started = time.monotonic()

        with pytest.raises(StreamError):
            Stream(CAMERA_URL, command=command, timeouts=Timeouts(open=30), cancel=cancel)

        assert time.monotonic() - started < 2


class TestCamera:
    def _frames_until(self, source: Camera, last: int) -> list[int]:
        seen: list[int] = []
        number = 0
        deadline = time.monotonic() + 10
        while number < last and time.monotonic() < deadline:
            got = source.next_frame(number, timeout=1)
            if got is not None:
                number = got[0]
                seen.append(number)
        return seen

    def test_it_comes_back_after_the_stream_dies(
        self, plan: Plan, caplog: pytest.LogCaptureFixture
    ) -> None:
        command = plan(
            {"frames": 3, "interval": 0.02, "stderr": "Connection reset by peer", "status": 1},
            {"header": False, "stderr": "Connection refused", "status": 1},
            {"frames": 3, "interval": 0.02, "then": "hang"},
        )
        source = Camera(CAMERA_URL, command=command, timeouts=QUICK)
        caplog.set_level(logging.INFO, logger=camera.__name__)

        source.start()
        try:
            seen = self._frames_until(source, 6)
        finally:
            source.stop()

        assert seen[-1] == 6
        assert "refused the connection" in caplog.text
        assert "rtsp://***@192.168.1.50:554/stream2" in caplog.text
        assert CAMERA_PASSWORD not in caplog.text

    def test_a_slow_reader_is_handed_the_newest_frame(self, plan: Plan) -> None:
        command = plan({"frames": 10, "then": "hang"})
        source = Camera(CAMERA_URL, command=command, timeouts=Timeouts(stall=30))

        source.start()
        try:
            first = source.next_frame(0, timeout=5)
            assert first is not None
            # The other nine arrive while this reader is busy.
            deadline = time.monotonic() + 5
            while source.frames_received < 10 and time.monotonic() < deadline:
                time.sleep(0.01)
            newest = source.next_frame(first[0], timeout=5)
        finally:
            source.stop()

        assert newest is not None
        number, image = newest
        assert number == 10
        expected = to_bgr(fake_ffmpeg.frame(10), fake_ffmpeg.WIDTH, fake_ffmpeg.HEIGHT)
        assert np.array_equal(image, expected)

    def test_with_nothing_new_it_waits_then_says_so(self, plan: Plan) -> None:
        command = plan({"frames": 1, "then": "hang"})
        source = Camera(CAMERA_URL, command=command, timeouts=Timeouts(stall=30))

        source.start()
        try:
            first = source.next_frame(0, timeout=5)
            assert first is not None
            assert source.next_frame(first[0], timeout=0.1) is None
        finally:
            source.stop()

    def test_stopping_does_not_wait_for_a_camera_that_is_not_answering(self, plan: Plan) -> None:
        command = plan({"header": False, "then": "hang"})
        source = Camera(CAMERA_URL, command=command, timeouts=Timeouts(open=30))
        source.start()
        time.sleep(0.2)
        started = time.monotonic()

        source.stop()

        assert time.monotonic() - started < 2

    def test_stopping_does_not_wait_for_a_camera_that_never_pauses(self, plan: Plan) -> None:
        command = plan({"frames": 100_000, "interval": 0.001})
        source = Camera(CAMERA_URL, command=command, timeouts=QUICK)
        source.start()
        assert source.next_frame(0, timeout=5) is not None
        started = time.monotonic()

        source.stop()

        assert time.monotonic() - started < 2

    def test_the_same_failure_is_logged_once(
        self, plan: Plan, caplog: pytest.LogCaptureFixture
    ) -> None:
        command = plan({"header": False, "stderr": "Connection refused", "status": 1})
        source = Camera(CAMERA_URL, command=command, timeouts=QUICK)
        caplog.set_level(logging.WARNING, logger=camera.__name__)

        source.start()
        time.sleep(0.6)
        source.stop()

        refusals = [r for r in caplog.records if "refused the connection" in r.getMessage()]
        assert len(refusals) == 1
