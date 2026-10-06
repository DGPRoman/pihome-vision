"""The ``pihome-vision`` command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Final

from pydantic import ValidationError

from pihome_vision import __version__
from pihome_vision.config import ConfigError, VisionConfig, load_config
from pihome_vision.redact import mask_url
from pihome_vision.settings import Settings, render_settings_error

#: Started with a configuration it cannot use. A service manager should not retry:
#: nothing will have changed by the next attempt.
EXIT_CONFIGURATION_ERROR: Final = 2


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
