"""The ``pihome-vision`` command."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError

from pihome_vision import __version__
from pihome_vision.config import ConfigError, DetectionModel, VisionConfig, load_config
from pihome_vision.redact import mask_url
from pihome_vision.settings import Settings, config_path_from_environment, render_settings_error

#: Started with a configuration it cannot use. A service manager should not retry:
#: nothing will have changed by the next attempt.
EXIT_CONFIGURATION_ERROR: Final = 2

#: How long ``check`` watches the camera to measure how fast frames arrive.
CHECK_SECONDS = 3.0

if TYPE_CHECKING:
    from pihome_vision.detect import Detection, Frame


def _load() -> tuple[Settings, VisionConfig]:
    """Both halves of the configuration, or exit 2 saying what is wrong."""
    try:
        settings = Settings()  # type: ignore[call-arg]  # pydantic-settings reads the environment
    except ValidationError as exc:
        sys.stderr.write(render_settings_error(exc) + "\n")
        raise SystemExit(EXIT_CONFIGURATION_ERROR) from None
    try:
        config = load_config(settings.config_path)
    except ConfigError as exc:
        sys.stderr.write(f"pihome-vision: {exc}\n")
        raise SystemExit(EXIT_CONFIGURATION_ERROR) from None
    return settings, config


def _validate(_: argparse.Namespace) -> int:
    settings, config = _load()
    out = sys.stdout
    out.write(f"camera   {mask_url(settings.camera_url.get_secret_value())}\n")
    out.write(f"hub      {settings.hub_url}\n")
    out.write(f"model    {config.model.path}\n")
    for camera in config.cameras:
        for trigger in camera.triggers:
            out.write(f"trigger  {camera.id}/{trigger.id}: {trigger.kind}\n")
    for light in config.lights:
        dark = ", after dark" if light.only_after_dark else ""
        out.write(
            f"light    {light.relay} <- {', '.join(light.triggers)}"
            f" (off after {light.off_after_seconds:g} s{dark})\n"
        )
    return 0


def _run_model(model: DetectionModel, frame: Frame) -> tuple[list[Detection], float]:
    """What the model finds in ``frame``, and how many seconds that took.

    Raises :class:`~pihome_vision.detect.ModelError` for a model that cannot run.
    """
    from pihome_vision.detect import Detector  # noqa: PLC0415

    detector = Detector(
        model.path,
        input_size=model.input_size,
        confidence=model.confidence,
        sha256=model.sha256,
    )
    detector.detect(frame)  # the first run includes setting the network up
    started = time.perf_counter()
    found = detector.detect(frame)
    return sorted(found, key=lambda d: -d.confidence), time.perf_counter() - started


def _detect(args: argparse.Namespace) -> int:
    # Imported here: OpenCV takes a moment to load, and validate has no use for it.
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from pihome_vision.detect import ModelError  # noqa: PLC0415

    try:
        model = load_config(config_path_from_environment()).model
    except ConfigError as exc:
        sys.stderr.write(f"pihome-vision: {exc}\n")
        return EXIT_CONFIGURATION_ERROR
    image = cv2.imread(str(args.image))
    if image is None:
        sys.stderr.write(f"pihome-vision: {args.image} is not an image OpenCV can read\n")
        return 1
    frame = np.asarray(image, dtype=np.uint8)
    try:
        found, elapsed = _run_model(model, frame)
    except ModelError as exc:
        sys.stderr.write(f"pihome-vision: {exc}\n")
        return EXIT_CONFIGURATION_ERROR
    height, width = frame.shape[:2]
    out = sys.stdout
    out.write(f"{args.image}: {width}x{height}, {elapsed * 1000:.0f} ms, {len(found)} found\n")
    for item in found:
        x, y = item.centre
        out.write(f"  {item.category:<8} {item.confidence:.2f} at {x:.0f},{y:.0f}\n")
    return 0


def _check(_: argparse.Namespace) -> int:
    from pihome_vision import camera  # noqa: PLC0415
    from pihome_vision.detect import ModelError  # noqa: PLC0415

    settings, config = _load()
    url = settings.camera_url.get_secret_value()
    out = sys.stdout
    out.write(f"camera   {mask_url(url)}\n")
    out.flush()  # connecting can take a while; say to what in the meantime
    started = time.monotonic()
    try:
        with camera.Stream(url) as stream:
            connected = time.monotonic() - started
            raw = stream.read()
            first, frames = time.monotonic(), 0
            while (elapsed := time.monotonic() - first) < CHECK_SECONDS:
                raw = stream.read()
                frames += 1
    except camera.StreamError as exc:
        sys.stderr.write(f"pihome-vision: cannot read the camera: {exc}\n")
        return 1
    says = f" (the stream says {stream.rate:.4g})" if stream.rate else ""
    out.write(
        f"  connected in {connected:.1f} s; {stream.width}x{stream.height}"
        f" at {frames / elapsed:.1f} frames/s{says}\n"
    )

    out.write(f"model    {config.model.path}\n")
    try:
        found, took = _run_model(config.model, camera.to_bgr(raw, stream.width, stream.height))
    except ModelError as exc:
        sys.stderr.write(f"pihome-vision: {exc}\n")
        return EXIT_CONFIGURATION_ERROR
    seen = ", ".join(f"{item.category} {item.confidence:.2f}" for item in found)
    out.write(f"  {took * 1000:.0f} ms a frame; found {seen or 'nothing'}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pihome-vision",
        description="Watch a camera for people and vehicles and switch the hub's lights.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(title="commands", metavar="COMMAND")
    validate = commands.add_parser(
        "validate",
        help="check the environment and vision.yaml, and say what they describe",
    )
    validate.set_defaults(handler=_validate)
    detect = commands.add_parser(
        "detect",
        help="run the model in vision.yaml over one image file, to try a model out",
    )
    detect.add_argument("image", type=Path, help="a photo, in any format OpenCV reads")
    detect.set_defaults(handler=_detect)
    check = commands.add_parser(
        "check",
        help="connect to the camera, measure its frame rate and run the model on one frame",
    )
    check.set_defaults(handler=_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
