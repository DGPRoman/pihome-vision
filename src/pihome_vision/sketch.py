"""Drawing zones and lines over a frame, and writing them out for ``vision.yaml``.

:class:`Sketch` is what ``pihome-vision edit`` does with each click and key press,
with no window involved, so every step of it can be tested without a display. The
window itself is :mod:`pihome_vision.gui`'s.

Click points onto the frame, then press ``z`` to close them into a zone or ``l`` to
make the two of them a line, and type its name. Backspace takes back the last point,
or with none, the last shape. ``q`` finishes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final, Literal

import yaml
from pydantic import ValidationError

from pihome_vision.config import (
    Camera,
    LineTrigger,
    Point,
    Trigger,
    VisionConfig,
    ZoneTrigger,
)

#: Decimal places a point is kept to: a pixel or two on any camera there is.
PRECISION: Final = 3

#: Key codes as ``cv2.waitKey`` gives them.
ENTER: Final = (10, 13)
ESCAPE: Final = 27
BACKSPACE: Final = (8, 127)

_NAME_CHARACTER = re.compile(r"[a-z0-9-]")
_NAME_LENGTH: Final = 64
_ZONE_POINTS: Final = 3
_LINE_POINTS: Final = 2

Kind = Literal["zone", "line"]

#: What a new shape watches for until somebody edits the YAML to say otherwise.
_CLASSES: Final = frozenset({"person", "vehicle"})


def fraction(x: float, y: float, width: int, height: int) -> Point:
    """A pixel of a ``width`` by ``height`` frame as fractions of it, rounded to
    :data:`PRECISION` places and kept inside the frame."""
    return (
        round(min(max(x / width, 0.0), 1.0), PRECISION),
        round(min(max(y / height, 0.0), 1.0), PRECISION),
    )


class Sketch:
    """The shapes on a camera, as somebody adds and takes them away."""

    def __init__(self, triggers: Sequence[Trigger] = ()) -> None:
        self.triggers: list[Trigger] = list(triggers)
        #: The shape being drawn, before it is a zone or a line.
        self.points: list[Point] = []
        #: What the finished shape is to be while its name is typed, otherwise None.
        self.naming: Kind | None = None
        self.name = ""
        #: Said once, about the last key, until the next one.
        self.message = ""
        self.done = False

    @property
    def status(self) -> str:
        """One line saying what has happened or what to do next."""
        if self.message:
            return self.message
        if self.naming is not None:
            return (
                f"name the {self.naming}: {self.name}_   "
                f"(enter for {self.name or self._default_name(self.naming)}, esc to go back)"
            )
        if not self.points:
            return "click to add points;  backspace: remove the last shape;  q: finish"
        return "click to add points;  z: zone;  l: line;  backspace: remove a point"

    def click(self, point: Point) -> None:
        if self.naming is None:
            self.points.append(point)
            self.message = ""

    def key(self, code: int) -> None:
        self.message = ""
        if self.naming is not None:
            self._name_key(code)
            return
        if code in BACKSPACE:
            if self.points:
                self.points.pop()
            elif self.triggers:
                removed = self.triggers.pop()
                self.message = f"removed {removed.kind} {removed.id}"
        elif code == ord("z"):
            self._finish("zone")
        elif code == ord("l"):
            self._finish("line")
        elif code in (ord("q"), ESCAPE):
            if self.points:
                self.message = "z or l to finish this shape, or backspace its points away"
            else:
                self.done = True

    def _finish(self, kind: Kind) -> None:
        if kind == "zone" and len(self.points) < _ZONE_POINTS:
            self.message = "a zone needs three points or more"
            return
        if kind == "line" and len(self.points) != _LINE_POINTS:
            self.message = "a line is two points"
            return
        # The rest is checked by the configuration's own rules, before a name is asked
        # for rather than after.
        try:
            self._shape(kind, self._default_name(kind))
        except ValidationError as exc:
            reason = exc.errors()[0]["msg"].removeprefix("Value error, ")
            self.message = f"not a {kind}: {reason}"
            return
        self.naming, self.name = kind, ""

    def _name_key(self, code: int) -> None:
        assert self.naming is not None  # noqa: S101  # only called while naming
        if code in ENTER:
            name = self.name.rstrip("-") or self._default_name(self.naming)
            if any(trigger.id == name for trigger in self.triggers):
                self.message = f"there is a trigger called {name} already"
                return
            self.triggers.append(self._shape(self.naming, name))
            self.points, self.naming = [], None
        elif code == ESCAPE:
            self.naming = None
        elif code in BACKSPACE:
            self.name = self.name[:-1]
        elif 0 < code < 128 and len(self.name) < _NAME_LENGTH:  # noqa: PLR2004  # ASCII
            typed = chr(code).lower().replace(" ", "-").replace("_", "-")
            # A slug: no leading hyphen, and no two in a row.
            if _NAME_CHARACTER.fullmatch(typed) and not (
                typed == "-" and (not self.name or self.name.endswith("-"))
            ):
                self.name += typed

    def _shape(self, kind: Kind, name: str) -> Trigger:
        if kind == "zone":
            return ZoneTrigger.model_validate(
                {"kind": "zone", "id": name, "polygon": self.points, "classes": _CLASSES}
            )
        return LineTrigger.model_validate(
            {"kind": "line", "id": name, "points": self.points, "classes": _CLASSES}
        )

    def _default_name(self, kind: Kind) -> str:
        taken = {trigger.id for trigger in self.triggers}
        number = 1
        while f"{kind}-{number}" in taken:
            number += 1
        return f"{kind}-{number}"


class _Dumper(yaml.SafeDumper):
    """Indents a list under its key, as ``vision.example.yaml`` is written."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow, indentless=False)


def fragment(camera: Camera, triggers: Sequence[Trigger]) -> str:
    """The ``cameras:`` section of ``vision.yaml``, with ``camera``'s triggers replaced."""
    entry = camera.model_dump(mode="json", exclude={"triggers"})
    entry["triggers"] = []
    for trigger in triggers:
        dumped = trigger.model_dump(mode="json")
        dumped["classes"] = sorted(dumped["classes"])
        entry["triggers"].append(dumped)
    return yaml.dump(
        {"cameras": [entry]},
        Dumper=_Dumper,
        sort_keys=False,
        default_flow_style=None,
        width=100,
    )


def dangling(config: VisionConfig, triggers: Sequence[Trigger]) -> list[str]:
    """Each light's trigger that is no longer drawn, as ``relay: trigger``."""
    drawn = {trigger.id for trigger in triggers}
    return [
        f"{light.relay}: {name}"
        for light in config.lights
        for name in light.triggers
        if name not in drawn
    ]
