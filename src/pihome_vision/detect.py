"""Finding people and vehicles in a frame, with a YOLO-format ONNX model.

The model runs through OpenCV's DNN module, so nothing beyond OpenCV is needed to run
it. Two output heads are understood, which covers the models Ultralytics exports:

* NMS-free, ``(1, N, 6)``: rows of ``x1, y1, x2, y2, confidence, class``, already
  filtered for overlaps (YOLO26 and later).
* Classic, ``(1, 4 + C, N)``: a centre box and one score per class for each of N
  candidates, which still need non-maximum suppression (YOLO11 and earlier).

Class numbers are COCO's. No model is shipped with this project: see docs/models.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt

from pihome_vision.config import DetectionModel, ObjectClass

Frame = npt.NDArray[np.uint8]

#: COCO class numbers for what a trigger can watch for. Everything else a model finds,
#: a cat, a bench, a kite, is dropped as soon as it is decoded.
CATEGORIES: Final[dict[int, ObjectClass]] = {
    0: "person",
    1: "vehicle",  # bicycle
    2: "vehicle",  # car
    3: "vehicle",  # motorcycle
    5: "vehicle",  # bus
    7: "vehicle",  # truck
}

#: Classes in a COCO model, which is what a classic head's width is checked against.
COCO_CLASSES: Final = 80

#: How much two boxes of one class may overlap before the weaker is taken for a
#: second look at the same object.
NMS_IOU: Final = 0.45

#: The most threads a model runs on unless the configuration says otherwise. Past
#: about this many a 640 model runs no faster, and each thread more only burns CPU:
#: on a 6-core, 12-thread machine, 8 threads took 35 ms a frame and 12 took 36, on
#: 0.28 and 0.37 seconds of CPU time.
MAX_THREADS: Final = 8

#: The grey YOLO pads a letterboxed frame with, as it was trained.
_PAD: Final = 114

#: Columns in a row of an NMS-free head.
_E2E_COLUMNS: Final = 6


class ModelError(Exception):
    """The model file is missing, not the one expected, or not a model this can run."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One object in one frame. The box is in the frame's own pixels."""

    x: float
    y: float
    width: float
    height: float
    confidence: float
    category: ObjectClass

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


@dataclass(frozen=True, slots=True)
class Letterbox:
    """How a frame was fitted into the model's square, to undo it afterwards."""

    scale: float
    left: int
    top: int

    def to_frame(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> tuple[float, float, float, float]:
        """Corners in the square → ``x, y, width, height`` in the original frame."""
        x = (x1 - self.left) / self.scale
        y = (y1 - self.top) / self.scale
        return x, y, (x2 - x1) / self.scale, (y2 - y1) / self.scale


def letterbox(frame: Frame, size: int) -> tuple[Frame, Letterbox]:
    """Scale ``frame`` to fit a ``size`` square, keeping its shape, and pad the rest."""
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), _PAD, dtype=np.uint8)
    top, left = (size - new_height) // 2, (size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas, Letterbox(scale, left, top)


def decode(output: npt.NDArray[np.float32], fit: Letterbox, confidence: float) -> list[Detection]:
    """Turn a model's raw output into the people and vehicles in it.

    Raises :class:`ModelError` for an output of neither shape this understands.
    """
    rows = output[0] if output.ndim == 3 else output  # noqa: PLR2004 - a batch of one
    if rows.ndim != 2:  # noqa: PLR2004 - a table
        msg = f"the model's output has shape {output.shape}, which is not a YOLO head"
        raise ModelError(msg)
    if rows.shape[1] == _E2E_COLUMNS:
        return _decode_end_to_end(rows, fit, confidence)
    if rows.shape[0] == 4 + COCO_CLASSES:
        return _decode_classic(rows.T, fit, confidence)
    msg = (
        f"the model's output has shape {output.shape}: neither an NMS-free head "
        f"(1, N, 6) nor a COCO head (1, {4 + COCO_CLASSES}, N)"
    )
    raise ModelError(msg)


def _decode_end_to_end(
    rows: npt.NDArray[np.float32], fit: Letterbox, confidence: float
) -> list[Detection]:
    found = []
    for x1, y1, x2, y2, score, class_id in rows:
        category = CATEGORIES.get(int(class_id))
        if score < confidence or category is None:
            continue
        found.append(Detection(*fit.to_frame(x1, y1, x2, y2), float(score), category))
    return found


def _decode_classic(
    rows: npt.NDArray[np.float32], fit: Letterbox, confidence: float
) -> list[Detection]:
    boxes, scores = rows[:, :4], rows[:, 4:]
    class_ids = scores.argmax(axis=1)
    best = scores[np.arange(len(scores)), class_ids]
    wanted = np.isin(class_ids, list(CATEGORIES)) & (best >= confidence)
    boxes, best, class_ids = boxes[wanted], best[wanted], class_ids[wanted]
    if len(boxes) == 0:
        return []

    centre_x, centre_y, width, height = boxes.T
    corners = np.stack(
        [centre_x - width / 2, centre_y - height / 2, centre_x + width / 2, centre_y + height / 2],
        axis=1,
    )
    found = []
    # Per class, so a person standing in front of a car does not suppress the car.
    for class_id in np.unique(class_ids):
        mine = np.flatnonzero(class_ids == class_id)
        xywh = [
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            for x1, y1, x2, y2 in corners[mine]
        ]
        kept = cv2.dnn.NMSBoxes(xywh, best[mine].tolist(), confidence, NMS_IOU)
        category = CATEGORIES[int(class_id)]
        for index in np.asarray(kept, dtype=np.int64).flatten():
            x1, y1, x2, y2 = corners[mine[index]]
            found.append(
                Detection(*fit.to_frame(x1, y1, x2, y2), float(best[mine[index]]), category)
            )
    return found


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model:
        for block in iter(lambda: model.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Detector:
    """One loaded model. Not to be shared between threads: give each camera its own.

    OpenCV runs every model in the process on one pool of ``threads``, so the last
    detector made decides how many there are. Unset, it is as many as there are CPUs
    for this process, up to :data:`MAX_THREADS`.
    """

    def __init__(
        self,
        path: Path,
        *,
        input_size: int,
        confidence: float,
        sha256: str | None = None,
        threads: int | None = None,
    ) -> None:
        if not path.is_file():
            msg = f"there is no model at {path}; docs/models.md says where to get one"
            raise ModelError(msg)
        if sha256 is not None and (actual := file_sha256(path)) != sha256:
            msg = f"{path} has SHA-256 {actual}, not the {sha256} the configuration expects"
            raise ModelError(msg)
        try:
            self._net = cv2.dnn.readNetFromONNX(str(path))
        except cv2.error as exc:
            msg = f"OpenCV cannot load {path} as an ONNX model: {_first_line(exc)}"
            raise ModelError(msg) from None
        cv2.setNumThreads(
            threads if threads is not None else min(MAX_THREADS, cv2.getNumberOfCPUs())
        )
        self.input_size = input_size
        self.confidence = confidence
        #: Whether the model has run on a frame. Until it has, a failure is most likely
        #: the configuration's; after, it is not, and restarting may well cure it.
        self._ran = False

    def detect(self, frame: Frame) -> list[Detection]:
        """The people and vehicles in ``frame``, a BGR image as OpenCV reads one.

        Raises :class:`ModelError` if the model will not run on its first frame, and
        whatever OpenCV raised for a failure after that.
        """
        square, fit = letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(square, 1 / 255.0, swapRB=True, crop=False)
        self._net.setInput(blob)
        try:
            output = self._net.forward()
        except cv2.error as exc:
            if self._ran:
                # Not the configuration, which has worked: running short of memory,
                # say. Reported as a failure, so a service manager restarts it.
                raise
            # By far the likeliest cause: the input size is fixed when a model is
            # exported, and the configuration names another.
            msg = (
                f"the model would not run on a {self.input_size}x{self.input_size} input; "
                f"input_size must be the size it was exported at ({_first_line(exc)})"
            )
            raise ModelError(msg) from None
        found = decode(np.asarray(output, dtype=np.float32), fit, self.confidence)
        self._ran = True
        return found


def load(model: DetectionModel) -> Detector:
    """The model the configuration describes, loaded to run as it says.

    Raises :class:`ModelError` as :class:`Detector` does.
    """
    return Detector(
        model.path,
        input_size=model.input_size,
        confidence=model.confidence,
        sha256=model.sha256,
        threads=model.threads,
    )


def _first_line(exc: cv2.error) -> str:
    text = str(exc).strip()
    return text.splitlines()[-1] if text else type(exc).__name__
