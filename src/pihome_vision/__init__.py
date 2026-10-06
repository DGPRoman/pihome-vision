"""pihome-vision — watches a camera and reports zones and lines to pihome-hub."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pihome-vision")
except PackageNotFoundError:  # pragma: no cover - only hit when running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
