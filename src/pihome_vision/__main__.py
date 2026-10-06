"""The ``pihome-vision`` command."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pihome_vision import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pihome-vision",
        description="Watch a camera for people and vehicles and report them to pihome-hub.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
