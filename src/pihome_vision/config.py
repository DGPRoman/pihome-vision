"""``vision.yaml``: the camera, what to watch for in it, and which lights follow.

None of this is secret, but all of it describes one particular house — the zones are
drawn over somebody's driveway and the location is where they live — so the real file
is git-ignored and only ``config/vision.example.yaml`` is tracked.

Geometry is in fractions of the frame, 0 to 1 from the top left corner, so a camera
that changes resolution keeps its zones and lines where they were drawn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

#: The same shape the hub requires of its relay ids, so a light's ``relay`` can name
#: any relay the hub has.
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=64)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Point = tuple[Fraction, Fraction]

#: What a trigger can be set to notice. ``vehicle`` is a bicycle, car, motorcycle,
#: bus or truck.
ObjectClass = Literal["person", "vehicle"]

#: Which way across a line counts. Left and right are as seen standing on the first
#: point and looking towards the second.
Direction = Literal["any", "left_to_right", "right_to_left"]

#: Cameras watched at once. One for now; the configuration is a list already so that
#: a second is a new entry rather than a new format.
MAX_CAMERAS: Final = 1

#: Twice the area, in frame fractions squared, below which a polygon is a line.
_MIN_DOUBLED_AREA: Final = 1e-9


class ConfigError(Exception):
    """``vision.yaml`` is missing, unreadable or wrong. The message says which."""


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Location(_Strict):
    """Where the house is, for sunrise and sunset. Only needed for ``only_after_dark``."""

    latitude: Annotated[float, Field(ge=-90.0, le=90.0)]
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)]
    timezone: str = Field(min_length=1, examples=["Europe/Kyiv"])

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            msg = f"{value!r} is not a time zone this machine knows"
            raise ValueError(msg) from None
        return value


class DetectionModel(_Strict):
    """The ONNX file that finds people and vehicles, and how to run it."""

    path: Path
    #: The file's SHA-256, to refuse a model that is not the one expected.
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    #: The square the frame is scaled into. Smaller is faster and misses more far away.
    input_size: Annotated[int, Field(ge=160, le=1280, multiple_of=32)] = 640
    #: Detections less certain than this are ignored.
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.35


class ZoneTrigger(_Strict):
    """Active while something of a chosen class is inside a polygon."""

    kind: Literal["zone"]
    id: Slug
    polygon: Annotated[list[Point], Field(min_length=3)]
    classes: Annotated[frozenset[ObjectClass], Field(min_length=1)]
    #: How long something must stay inside before the zone counts as occupied, so
    #: one frame's false detection does not light the yard.
    min_seconds: Annotated[float, Field(ge=0.0, le=60.0)] = 1.0
    #: How long the zone must be empty before it counts as clear, so a person the
    #: detector loses for a moment does not end it.
    clear_seconds: Annotated[float, Field(gt=0.0, le=600.0)] = 5.0

    @field_validator("polygon")
    @classmethod
    def _has_area(cls, polygon: list[Point]) -> list[Point]:
        # The shoelace formula: zero when every point lies on one line.
        doubled = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(polygon, [*polygon[1:], polygon[0]], strict=True)
        )
        if abs(doubled) < _MIN_DOUBLED_AREA:
            msg = "has no area: its points lie on one line"
            raise ValueError(msg)
        return polygon


class LineTrigger(_Strict):
    """Fires at the moment something of a chosen class crosses a line."""

    kind: Literal["line"]
    id: Slug
    points: tuple[Point, Point]
    direction: Direction = "any"
    classes: Annotated[frozenset[ObjectClass], Field(min_length=1)]

    @field_validator("points")
    @classmethod
    def _has_length(cls, points: tuple[Point, Point]) -> tuple[Point, Point]:
        if points[0] == points[1]:
            msg = "both ends are the same point"
            raise ValueError(msg)
        return points


Trigger = Annotated[ZoneTrigger | LineTrigger, Field(discriminator="kind")]


class Camera(_Strict):
    id: Slug
    #: Frames examined per second. The camera may send more; the rest are skipped.
    fps: Annotated[float, Field(gt=0.0, le=30.0)] = 5.0
    #: Skip detection while the picture has changed less than this since the last
    #: detection, as a mean difference of grey levels (0-255). 0 always detects.
    motion_threshold: Annotated[float, Field(ge=0.0, le=255.0)] = 0.0
    triggers: Annotated[list[Trigger], Field(min_length=1)]


class Light(_Strict):
    """A relay on the hub, and the triggers that switch it on."""

    relay: Slug
    triggers: Annotated[list[Slug], Field(min_length=1)]
    #: How long the light stays on after the last of its triggers ends, or after a
    #: line is crossed.
    off_after_seconds: Annotated[float, Field(gt=0.0, le=86_400.0)] = 120.0
    #: Only switch on between sunset and sunrise at ``location``.
    only_after_dark: StrictBool = False

    @field_validator("triggers")
    @classmethod
    def _distinct(cls, triggers: list[str]) -> list[str]:
        if len(set(triggers)) != len(triggers):
            msg = "names a trigger more than once"
            raise ValueError(msg)
        return triggers


class VisionConfig(_Strict):
    location: Location | None = None
    model: DetectionModel
    cameras: Annotated[list[Camera], Field(min_length=1, max_length=MAX_CAMERAS)]
    lights: Annotated[list[Light], Field(min_length=1)]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        seen: set[str] = set()
        for camera in self.cameras:
            for trigger in camera.triggers:
                if trigger.id in seen:
                    msg = f"trigger {trigger.id!r} is defined more than once"
                    raise ValueError(msg)
                seen.add(trigger.id)

        relays: set[str] = set()
        for light in self.lights:
            # Two lights on one relay would each switch it off under the other.
            if light.relay in relays:
                msg = f"relay {light.relay!r} has more than one light; list its triggers in one"
                raise ValueError(msg)
            relays.add(light.relay)
            for trigger_id in light.triggers:
                if trigger_id not in seen:
                    msg = f"light {light.relay!r} names trigger {trigger_id!r}, which no camera has"
                    raise ValueError(msg)
            if light.only_after_dark and self.location is None:
                msg = f"light {light.relay!r} is only_after_dark, which needs a location"
                raise ValueError(msg)
        return self


def render_config_error(exc: ValidationError) -> str:
    """One line per problem, each naming where in the file it is."""
    lines = []
    for error in exc.errors(include_input=False, include_url=False):
        where = ".".join(str(part) for part in error["loc"])
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"  {where}: {message}" if where else f"  {message}")
    return "\n".join(lines)


def load_config(path: Path) -> VisionConfig:
    """Read and check ``vision.yaml``."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        msg = f"{path} does not exist; start from config/vision.example.yaml"
        raise ConfigError(msg) from None
    except OSError as exc:
        msg = f"could not read {path}: {exc.strerror}"
        raise ConfigError(msg) from None

    try:
        document: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"{path} is not valid YAML: {exc}"
        raise ConfigError(msg) from None
    if not isinstance(document, dict):
        msg = f"{path} must be a mapping with model, cameras and lights"
        raise ConfigError(msg)

    try:
        return VisionConfig.model_validate(document)
    except ValidationError as exc:
        msg = f"{path} is invalid:\n{render_config_error(exc)}"
        raise ConfigError(msg) from None
