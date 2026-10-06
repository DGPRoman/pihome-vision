"""A stand-in for ffmpeg that plays a scripted run, so no test needs a camera.

    python fake_ffmpeg.py PLAN [ffmpeg's arguments...]

PLAN is a JSON file holding a list of runs. Each start plays the next run, and the last
one again once the list is used up. A run is an object with any of:

``frames``
    How many frames to write. The luma of frame ``n``, counting from 1 across the
    whole plan, is filled with ``n``, so a test can tell which frame it was handed.
``interval``
    Seconds between frames.
``header``
    ``false`` to write nothing at all, as a camera that never answers.
``garbage``
    ``true`` to write something that is not YUV4MPEG.
``stderr``
    Written to stderr before exiting. ``{url}`` becomes the ``-i`` argument, which is
    what ffmpeg does with the address it was given.
``then``
    ``"exit"`` (the default) or ``"hang"``, to stop writing but stay running.
``status``
    The exit status.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WIDTH = 4
HEIGHT = 2


def frame(number: int) -> bytes:
    """What the fake writes as frame ``number``: 4:2:0, luma ``number``, grey chroma."""
    luma = WIDTH * HEIGHT
    return bytes([number % 256]) * luma + bytes([128]) * (luma // 2)


def main() -> int:
    plan_path = Path(sys.argv[1])
    arguments = sys.argv[2:]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    counter = plan_path.with_suffix(".count")
    started = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(started + 1))
    run = plan[min(started, len(plan) - 1)]
    numbered = plan_path.with_suffix(".numbered")
    number = int(numbered.read_text()) if numbered.exists() else 0

    out = sys.stdout.buffer
    if run.get("garbage"):
        out.write(b"this is not video\n")
    elif run.get("header", True):
        out.write(f"YUV4MPEG2 W{WIDTH} H{HEIGHT} F25:1 Ip A1:1 C420jpeg\n".encode())
        out.flush()
        for _ in range(run.get("frames", 0)):
            number += 1
            numbered.write_text(str(number))
            out.write(b"FRAME\n" + frame(number))
            out.flush()
            time.sleep(run.get("interval", 0))
    out.flush()

    if "stderr" in run:
        url = arguments[arguments.index("-i") + 1]
        sys.stderr.write(run["stderr"].replace("{url}", url) + "\n")
    if run.get("then") == "hang":
        time.sleep(60)
    return int(run.get("status", 0))


if __name__ == "__main__":
    raise SystemExit(main())
