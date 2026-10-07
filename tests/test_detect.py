from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from pihome_vision.config import DetectionModel
from pihome_vision.detect import (
    COCO_CLASSES,
    DEFAULT_INPUT_SIZE,
    MAX_THREADS,
    Detection,
    Detector,
    Letterbox,
    ModelError,
    decode,
    exported_size,
    file_sha256,
    letterbox,
    load,
)

PERSON, BICYCLE, CAR, CAT, TRUCK, DOG = 0, 1, 2, 15, 7, 16

#: A frame that fills the square exactly, so boxes decode to the numbers written.
IDENTITY = Letterbox(scale=1.0, left=0, top=0)

#: A model exported from YOLO, if there is one here to try the reader on.
EXPORTED = Path(__file__).parents[1] / "models" / "detector.onnx"


def end_to_end(*rows: tuple[float, float, float, float, float, int]) -> npt.NDArray[np.float32]:
    return np.array([rows], dtype=np.float32)


def classic(
    *candidates: tuple[float, float, float, float, int, float],
) -> npt.NDArray[np.float32]:
    """``(1, 84, N)`` from candidates of centre x, centre y, width, height, class, score."""
    output = np.zeros((1, 4 + COCO_CLASSES, len(candidates)), dtype=np.float32)
    for column, (cx, cy, w, h, class_id, score) in enumerate(candidates):
        output[0, :4, column] = (cx, cy, w, h)
        output[0, 4 + class_id, column] = score
    return output


def _varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append(value & 0x7F | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def field(number: int, value: int | bytes) -> bytes:
    """One protobuf field: an integer, or the bytes of a string or message."""
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    return _varint(number << 3 | 2) + _varint(len(value)) + value


def onnx_model(*dims: int | str) -> bytes:
    """An ONNX model, as far as its size goes: one input of shape ``dims``, a number
    or, for a dimension of any size, a name. Around it are the fields of a real export
    that a reader steps over: a node, weights, an output and metadata."""
    shape = b"".join(
        field(1, field(1, dim) if isinstance(dim, int) else field(2, dim.encode())) for dim in dims
    )
    image = field(1, b"images") + field(2, field(1, field(1, 1) + field(2, shape)))
    # dims, name and raw_data, then float_data and double_data written unpacked.
    weights = field(1, 64) + field(8, b"model.0.conv.weight") + field(9, bytes(256))
    weights += _varint(4 << 3 | 5) + bytes(4) + _varint(10 << 3 | 1) + bytes(8)
    graph = (
        field(1, field(1, b"images") + field(2, b"output0") + field(4, b"Conv"))
        + field(2, b"main_graph")
        + field(5, weights)
        + field(11, image)
        + field(12, field(1, b"output0"))
    )
    return (
        field(1, 8)  # ir_version
        + field(2, b"pytorch")
        + field(8, field(2, 12))  # opset 12
        + field(7, graph)
        + field(14, field(1, b"imgsz") + field(2, b"[640, 640]"))
    )


def boxes(found: list[Detection]) -> list[tuple[str, float, float, float, float]]:
    return sorted(
        (d.category, round(d.x), round(d.y), round(d.width), round(d.height)) for d in found
    )


class TestLetterbox:
    @pytest.mark.parametrize(
        ("height", "width", "expected"),
        [
            (720, 1280, Letterbox(scale=0.5, left=0, top=140)),
            (1280, 720, Letterbox(scale=0.5, left=140, top=0)),
            (320, 320, Letterbox(scale=2.0, left=0, top=0)),
        ],
    )
    def test_the_frame_is_fitted_and_centred(
        self, height: int, width: int, expected: Letterbox
    ) -> None:
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        square, fit = letterbox(frame, 640)

        assert square.shape == (640, 640, 3)
        assert fit == expected

    def test_padding_is_the_grey_the_model_was_trained_on(self) -> None:
        square, _ = letterbox(np.zeros((720, 1280, 3), dtype=np.uint8), 640)

        assert square[0, 0].tolist() == [114, 114, 114]
        assert square[320, 320].tolist() == [0, 0, 0]

    @pytest.mark.parametrize(("height", "width"), [(720, 1280), (1280, 720), (500, 500)])
    def test_a_box_maps_back_to_where_it_was(self, height: int, width: int) -> None:
        _, fit = letterbox(np.zeros((height, width, 3), dtype=np.uint8), 640)
        x, y, w, h = 100.0, 120.0, 80.0, 60.0
        corners = (
            x * fit.scale + fit.left,
            y * fit.scale + fit.top,
            (x + w) * fit.scale + fit.left,
            (y + h) * fit.scale + fit.top,
        )

        assert fit.to_frame(*corners) == pytest.approx((x, y, w, h))


class TestEndToEndHead:
    def test_people_and_vehicles_above_the_threshold_are_kept(self) -> None:
        output = end_to_end(
            (10, 20, 50, 120, 0.9, PERSON),
            (100, 100, 300, 200, 0.6, TRUCK),
            (200, 200, 240, 260, 0.5, BICYCLE),
        )

        found = decode(output, IDENTITY, confidence=0.35)

        assert boxes(found) == [
            ("person", 10, 20, 40, 100),
            ("vehicle", 100, 100, 200, 100),
            ("vehicle", 200, 200, 40, 60),
        ]
        assert [d.confidence for d in found] == pytest.approx([0.9, 0.6, 0.5])

    def test_other_classes_and_weak_detections_are_dropped(self) -> None:
        output = end_to_end(
            (10, 20, 50, 120, 0.95, CAT),
            (10, 20, 50, 120, 0.95, DOG),
            (100, 100, 300, 200, 0.2, CAR),
            (0, 0, 0, 0, 0.0, PERSON),
        )

        assert decode(output, IDENTITY, confidence=0.35) == []

    def test_boxes_come_back_in_frame_pixels(self) -> None:
        fit = Letterbox(scale=0.5, left=0, top=140)
        output = end_to_end((50, 150, 100, 200, 0.9, PERSON))

        (found,) = decode(output, fit, confidence=0.35)

        assert (found.x, found.y, found.width, found.height) == (100, 20, 100, 100)
        assert found.centre == (150, 70)


class TestClassicHead:
    def test_overlapping_boxes_of_one_object_become_one(self) -> None:
        output = classic(
            (100, 100, 40, 100, PERSON, 0.9),
            (102, 101, 40, 100, PERSON, 0.7),
            (400, 300, 120, 80, CAR, 0.8),
        )

        found = decode(output, IDENTITY, confidence=0.35)

        assert boxes(found) == [("person", 80, 50, 40, 100), ("vehicle", 340, 260, 120, 80)]
        assert max(d.confidence for d in found if d.category == "person") == pytest.approx(0.9)

    def test_a_person_in_front_of_a_car_does_not_hide_it(self) -> None:
        output = classic((200, 200, 100, 100, PERSON, 0.9), (200, 200, 100, 100, CAR, 0.8))

        assert {d.category for d in decode(output, IDENTITY, confidence=0.35)} == {
            "person",
            "vehicle",
        }

    def test_other_classes_and_weak_detections_are_dropped(self) -> None:
        output = classic((100, 100, 40, 40, DOG, 0.9), (300, 300, 40, 40, CAR, 0.1))

        assert decode(output, IDENTITY, confidence=0.35) == []

    def test_an_empty_frame_is_fine(self) -> None:
        output = np.zeros((1, 4 + COCO_CLASSES, 8400), dtype=np.float32)

        assert decode(output, IDENTITY, confidence=0.35) == []


@pytest.mark.parametrize("shape", [(1, 300, 7), (1, 20, 8400), (8400,), (1, 2, 3, 4)])
def test_an_output_of_another_shape_is_refused(shape: tuple[int, ...]) -> None:
    with pytest.raises(ModelError, match=r"YOLO|head"):
        decode(np.zeros(shape, dtype=np.float32), IDENTITY, confidence=0.35)


class TestLoading:
    def test_a_missing_model_says_where_to_get_one(self, tmp_path: Path) -> None:
        with pytest.raises(ModelError, match=r"docs/models\.md"):
            Detector(tmp_path / "detector.onnx", input_size=640, confidence=0.35)

    def test_a_model_that_is_not_the_expected_one_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "detector.onnx"
        path.write_bytes(b"not the model you were looking for")

        with pytest.raises(ModelError, match=r"not the 0+ the configuration expects"):
            Detector(path, input_size=640, confidence=0.35, sha256="0" * 64)

    def test_a_file_that_is_not_onnx_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "detector.onnx"
        path.write_bytes(b"\x00garbage" * 64)

        with pytest.raises(ModelError, match="cannot load"):
            Detector(path, input_size=640, confidence=0.35, sha256=file_sha256(path))


class FakeNet:
    """Stands in for a loaded network: runs ``works`` times, then fails as OpenCV does."""

    def __init__(self, works: int) -> None:
        self.works = works

    def setInput(self, blob: object) -> None:  # noqa: N802 - OpenCV's name
        pass

    def forward(self) -> npt.NDArray[np.float32]:
        if self.works == 0:
            msg = "Insufficient memory"
            raise cv2.error(msg)
        self.works -= 1
        return end_to_end((10, 10, 50, 90, 0.9, PERSON))


@pytest.fixture
def fake_model(tmp_path: Path) -> Path:
    path = tmp_path / "detector.onnx"
    path.write_bytes(b"a model")
    return path


class TestFailing:
    FRAME = np.zeros((640, 640, 3), dtype=np.uint8)

    def detector(self, path: Path, monkeypatch: pytest.MonkeyPatch, works: int) -> Detector:
        net = FakeNet(works)
        monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *_: net)
        return Detector(path, input_size=640, confidence=0.35)

    def test_a_model_that_will_not_run_at_all_is_the_configurations_fault(
        self, fake_model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        detector = self.detector(fake_model, monkeypatch, works=0)

        with pytest.raises(ModelError, match="input_size must be the size it was exported at"):
            detector.detect(self.FRAME)

    def test_so_is_one_that_says_its_size_and_will_not_run_at_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "detector.onnx"
        path.write_bytes(onnx_model(1, 3, 640, 640))
        detector = self.detector(path, monkeypatch, works=0)

        with pytest.raises(ModelError, match=r"640x640 input, the size it was exported at \("):
            detector.detect(self.FRAME)

    def test_a_failure_after_it_has_run_is_not(
        self, fake_model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        detector = self.detector(fake_model, monkeypatch, works=1)

        assert len(detector.detect(self.FRAME)) == 1
        with pytest.raises(cv2.error, match="Insufficient memory") as raised:
            detector.detect(self.FRAME)
        assert not isinstance(raised.value, ModelError)


class TestThreads:
    @pytest.fixture
    def pool(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """The sizes OpenCV's thread pool is set to, on a machine with 12 CPUs."""
        sizes: list[int] = []
        monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *_: FakeNet(works=1))
        monkeypatch.setattr(cv2, "getNumberOfCPUs", lambda: 12)
        monkeypatch.setattr(cv2, "setNumThreads", sizes.append)
        return sizes

    def test_unless_told_it_runs_on_every_cpu_up_to_a_few(
        self, fake_model: Path, pool: list[int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        Detector(fake_model, input_size=640, confidence=0.35)
        monkeypatch.setattr(cv2, "getNumberOfCPUs", lambda: 4)
        Detector(fake_model, input_size=640, confidence=0.35)

        assert pool == [MAX_THREADS, 4]

    def test_the_configuration_says_how_many(self, fake_model: Path, pool: list[int]) -> None:
        model = DetectionModel(path=fake_model, threads=12)

        load(model)

        assert pool == [12]


class TestInputSize:
    @pytest.fixture
    def model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Path]:
        """Writes a model with an input of the given shape, which loads as one that runs."""
        monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *_: FakeNet(works=1))

        def write(*dims: int | str) -> Path:
            path = tmp_path / "detector.onnx"
            path.write_bytes(onnx_model(*dims))
            return path

        return write

    @pytest.mark.parametrize(
        ("dims", "size"),
        [
            ((1, 3, 640, 640), (640, 640)),
            ((1, 3, 1024, 1024), (1024, 1024)),
            ((1, 3, 384, 640), (384, 640)),
            (("batch", 3, "height", "width"), None),
            ((1, 3, 640), None),
        ],
    )
    def test_the_model_file_says_what_size_it_takes(
        self, model: Callable[..., Path], dims: tuple[int | str, ...], size: tuple[int, int]
    ) -> None:
        assert exported_size(model(*dims)) == size

    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"not a model at all",
            b"\xff" * 16,
            onnx_model(1, 3, 640, 640)[:100],
        ],
    )
    def test_a_file_that_is_not_one_says_nothing(self, tmp_path: Path, data: bytes) -> None:
        path = tmp_path / "detector.onnx"
        path.write_bytes(data)

        assert exported_size(path) is None

    @pytest.mark.skipif(not EXPORTED.is_file(), reason="no model in models/")
    def test_so_does_one_exported_from_yolo(self) -> None:
        size = exported_size(EXPORTED)

        assert size is not None
        assert size[0] == size[1]
        assert size[0] % 32 == 0

    def test_left_unset_it_is_the_models_own(self, model: Callable[..., Path]) -> None:
        assert load(DetectionModel(path=model(1, 3, 960, 960))).input_size == 960

    def test_set_it_must_be_the_models_own(self, model: Callable[..., Path]) -> None:
        path = model(1, 3, 960, 960)

        assert Detector(path, input_size=960, confidence=0.35).input_size == 960
        with pytest.raises(ModelError, match=r"exported to take 960x960 images, not the 640"):
            Detector(path, input_size=640, confidence=0.35)

    def test_a_model_that_is_not_square_is_refused(self, model: Callable[..., Path]) -> None:
        with pytest.raises(ModelError, match=r"takes 640x384 images"):
            Detector(model(1, 3, 384, 640), confidence=0.35)

    @pytest.mark.parametrize(("configured", "size"), [(None, DEFAULT_INPUT_SIZE), (800, 800)])
    def test_a_model_that_takes_any_size_is_run_at_the_one_configured(
        self, model: Callable[..., Path], configured: int | None, size: int
    ) -> None:
        path = model("batch", 3, "height", "width")

        assert Detector(path, input_size=configured, confidence=0.35).input_size == size
