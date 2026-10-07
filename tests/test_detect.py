from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime
import pytest
from onnxruntime.capi import onnxruntime_pybind11_state

from pihome_vision.config import DetectionModel, Engine
from pihome_vision.detect import (
    COCO,
    DEFAULT_INPUT_SIZE,
    MAX_THREADS,
    Classes,
    Detection,
    Detector,
    Letterbox,
    ModelError,
    decode,
    exported_size,
    file_sha256,
    letterbox,
    load,
    model_classes,
)

PERSON, BICYCLE, CAR, CAT, TRUCK, DOG = 0, 1, 2, 15, 7, 16

#: A frame that fills the square exactly, so boxes decode to the numbers written.
IDENTITY = Letterbox(scale=1.0, left=0, top=0)

#: A model exported from YOLO, if there is one here to try the reader on.
EXPORTED = Path(__file__).parents[1] / "models" / "detector.onnx"


def end_to_end(*rows: tuple[float, float, float, float, float, int]) -> npt.NDArray[np.float32]:
    return np.array([rows], dtype=np.float32)


def classic(
    *candidates: tuple[float, float, float, float, int, float], classes: int = COCO.count
) -> npt.NDArray[np.float32]:
    """``(1, 4 + classes, N)`` from candidates of centre x, centre y, width, height,
    class, score."""
    output = np.zeros((1, 4 + classes, len(candidates)), dtype=np.float32)
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


def onnx_model(*dims: int | str, names: str | None = None) -> bytes:
    """An ONNX model, as far as its size and classes go: one input of shape ``dims``, a
    number or, for a dimension of any size, a name, and ``names`` in its metadata if
    given. Around them are the fields of a real export that a reader steps over: a node,
    weights, an output and other metadata."""
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
    model = (
        field(1, 8)  # ir_version
        + field(2, b"pytorch")
        + field(8, field(2, 12))  # opset 12
        + field(7, graph)
        + field(14, field(1, b"imgsz") + field(2, b"[640, 640]"))
    )
    if names is not None:
        model += field(14, field(1, b"names") + field(2, names.encode()))
    return model + field(14, field(1, b"task") + field(2, b"detect"))


def constant_model(
    height: int, width: int, *rows: tuple[float, ...], names: str | None = None
) -> bytes:
    """A real ONNX model, small enough to write here, that both engines run: it takes
    images of ``height`` and ``width`` and, whatever is in them, gives an NMS-free
    head of ``rows``. ``names`` are its classes, as Ultralytics writes them."""

    def value(name: bytes, dims: tuple[int, ...]) -> bytes:
        shape = b"".join(field(1, field(1, dim)) for dim in dims)
        return field(1, name) + field(2, field(1, field(1, 1) + field(2, shape)))

    output = np.array([rows], dtype=np.float32)
    # dims, data_type float and raw_data, as the value of a Constant node.
    tensor = b"".join(field(1, dim) for dim in output.shape)
    tensor += field(2, 1) + field(9, output.tobytes())
    constant = field(1, b"value") + field(20, 4) + field(5, tensor)
    graph = (
        field(1, field(2, b"output0") + field(4, b"Constant") + field(5, constant))
        + field(2, b"main_graph")
        + field(11, value(b"images", (1, 3, height, width)))
        + field(12, value(b"output0", output.shape))
    )
    model = field(1, 8) + field(8, field(2, 13)) + field(7, graph)  # ir_version, opset 13
    if names is not None:
        model += field(14, field(1, b"names") + field(2, names.encode()))
    return model


def boxes(found: list[Detection]) -> list[tuple[str, float, float, float, float]]:
    return sorted(
        (d.category, round(d.x), round(d.y), round(d.width), round(d.height)) for d in found
    )


class TestLetterbox:
    @pytest.mark.parametrize(
        ("height", "width", "size", "expected"),
        [
            (720, 1280, (640, 640), Letterbox(scale=0.5, left=0, top=140)),
            (1280, 720, (640, 640), Letterbox(scale=0.5, left=140, top=0)),
            (320, 320, (640, 640), Letterbox(scale=2.0, left=0, top=0)),
            (1440, 2560, (640, 384), Letterbox(scale=0.25, left=0, top=12)),
            (1280, 720, (640, 384), Letterbox(scale=0.3, left=212, top=0)),
        ],
    )
    def test_the_frame_is_fitted_and_centred(
        self, height: int, width: int, size: tuple[int, int], expected: Letterbox
    ) -> None:
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        fitted, fit = letterbox(frame, size)

        assert fitted.shape == (size[1], size[0], 3)
        assert fit == expected

    def test_padding_is_the_grey_the_model_was_trained_on(self) -> None:
        square, _ = letterbox(np.zeros((720, 1280, 3), dtype=np.uint8), (640, 640))

        assert square[0, 0].tolist() == [114, 114, 114]
        assert square[320, 320].tolist() == [0, 0, 0]

    @pytest.mark.parametrize(("height", "width"), [(720, 1280), (1280, 720), (500, 500)])
    def test_a_box_maps_back_to_where_it_was(self, height: int, width: int) -> None:
        _, fit = letterbox(np.zeros((height, width, 3), dtype=np.uint8), (640, 384))
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

    def test_class_numbers_are_the_models_own(self) -> None:
        output = end_to_end((10, 20, 50, 120, 0.9, 0), (100, 100, 300, 200, 0.8, 1))

        found = decode(output, IDENTITY, 0.35, Classes(2, {1: "person"}))

        assert boxes(found) == [("person", 100, 100, 200, 100)]

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
        output = np.zeros((1, 4 + COCO.count, 8400), dtype=np.float32)

        assert decode(output, IDENTITY, confidence=0.35) == []

    def test_a_model_of_its_own_classes_has_a_head_as_wide_as_they_are(self) -> None:
        output = classic(
            (100, 100, 40, 100, 1, 0.9), (400, 300, 120, 80, 0, 0.8), (50, 50, 20, 20, 2, 0.9),
            classes=3,
        )  # fmt: skip

        found = decode(output, IDENTITY, 0.35, Classes(3, {0: "vehicle", 1: "person"}))

        assert boxes(found) == [("person", 80, 50, 40, 100), ("vehicle", 340, 260, 120, 80)]

    def test_of_only_the_classes_used_a_person_is_not_outscored(self) -> None:
        # The same box: a cat to a model of COCO, a person to one that has no cats.
        everything = classic((100, 100, 40, 40, PERSON, 0.4), (100, 100, 40, 40, CAT, 0.6))
        everything[0, 4 + CAT, 1] = 0.0
        everything[0, 4 + CAT, 0] = 0.6
        cut = everything[:, [0, 1, 2, 3, 4 + PERSON, 4 + CAR], :]

        assert decode(everything, IDENTITY, 0.35) == []
        assert boxes(decode(cut, IDENTITY, 0.35, Classes(2, {0: "person", 1: "vehicle"}))) == [
            ("person", 80, 80, 40, 40)
        ]

    def test_a_head_not_as_wide_as_the_models_classes_is_refused(self) -> None:
        output = classic((100, 100, 40, 100, PERSON, 0.9))

        with pytest.raises(ModelError, match=r"classic head of its 2 classes \(1, 6, N\)"):
            decode(output, IDENTITY, 0.35, Classes(2, {0: "person"}))


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

    def test_a_file_that_is_not_onnx_is_refused(self, tmp_path: Path, engine: Engine) -> None:
        path = tmp_path / "detector.onnx"
        path.write_bytes(b"\x00garbage" * 64)

        with pytest.raises(ModelError, match=r"(ONNX Runtime|OpenCV) cannot load") as raised:
            Detector(path, engine=engine, input_size=640, confidence=0.35, sha256=file_sha256(path))
        assert "[ONNXRuntimeError]" not in str(raised.value)


class TestEngines:
    def test_onnx_runtime_unless_the_configuration_says_otherwise(self) -> None:
        assert DetectionModel(path=Path("detector.onnx")).engine == "onnxruntime"

    def test_each_runs_a_real_model(self, tmp_path: Path, engine: Engine) -> None:
        path = tmp_path / "detector.onnx"
        path.write_bytes(constant_model(384, 640, (10, 10, 50, 90, 0.9, PERSON)))

        found = load(DetectionModel(path=path, engine=engine)).detect(
            np.zeros((1440, 2560, 3), np.uint8)
        )

        assert boxes(found) == [("person", 40, -8, 160, 320)]

    def test_each_runs_a_model_of_its_own_classes(self, tmp_path: Path, engine: Engine) -> None:
        path = tmp_path / "detector.onnx"
        rows = (10, 10, 50, 90, 0.9, 0), (100, 100, 140, 180, 0.8, 1)
        path.write_bytes(constant_model(384, 640, *rows, names="{0: 'car', 1: 'person'}"))

        found = load(DetectionModel(path=path, engine=engine)).detect(
            np.zeros((1440, 2560, 3), np.uint8)
        )

        assert [d.category for d in found] == ["vehicle", "person"]


class TestClasses:
    def write(self, tmp_path: Path, names: str | None) -> Path:
        path = tmp_path / "detector.onnx"
        path.write_bytes(onnx_model(1, 3, 384, 640, names=names))
        return path

    @pytest.mark.parametrize(
        ("names", "classes"),
        [
            ("{0: 'person', 1: 'vehicle'}", Classes(2, {0: "person", 1: "vehicle"})),
            ("{0: 'car', 1: 'dog', 2: 'person'}", Classes(3, {0: "vehicle", 2: "person"})),
            ("{0: 'Person', 1: 'Truck'}", Classes(2, {0: "person", 1: "vehicle"})),
            (
                "{0: 'bicycle', 1: 'motorcycle', 2: 'bus', 3: \"it's a kite\"}",
                Classes(4, {0: "vehicle", 1: "vehicle", 2: "vehicle"}),
            ),
        ],
    )
    def test_the_model_file_names_them(self, tmp_path: Path, names: str, classes: Classes) -> None:
        assert model_classes(self.write(tmp_path, names)) == classes

    def test_a_file_that_names_none_is_taken_for_coco(self, tmp_path: Path) -> None:
        assert model_classes(self.write(tmp_path, None)) == COCO

    @pytest.mark.parametrize(
        "names",
        [
            "",
            "not python",
            "{}",
            "['person', 'car']",
            "{1: 'person', 2: 'car'}",
            "{0: 'person', 1: 2}",
            "__import__('os').getcwd()",
            "{0: 'person'" * 1000,
        ],
    )
    def test_names_that_are_not_a_list_of_classes_are_taken_for_none(
        self, tmp_path: Path, names: str
    ) -> None:
        assert model_classes(self.write(tmp_path, names)) == COCO

    @pytest.mark.parametrize("data", [b"", b"not a model at all", b"\xff" * 16])
    def test_so_is_a_file_that_is_not_a_model(self, tmp_path: Path, data: bytes) -> None:
        path = tmp_path / "detector.onnx"
        path.write_bytes(data)

        assert model_classes(path) == COCO

    def test_a_model_of_nothing_a_trigger_watches_for_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake(monkeypatch, "onnxruntime")
        path = self.write(tmp_path, "{0: 'cat', 1: 'dog'}")

        with pytest.raises(ModelError, match=r"model of cat, dog: neither a person nor a vehicle"):
            Detector(path, confidence=0.35)

    @pytest.mark.skipif(not EXPORTED.is_file(), reason="no model in models/")
    def test_one_exported_from_yolo_names_cocos(self) -> None:
        assert model_classes(EXPORTED) == COCO


class FakeModel:
    """Stands in for a model as either engine loads it: runs ``works`` times, then
    fails as that engine does."""

    def __init__(self, works: int, error: Exception) -> None:
        self.works = works
        self.error = error
        #: The shape of each input it was given.
        self.inputs: list[tuple[int, ...]] = []
        #: The threads it was loaded to run on.
        self.threads: int | None = None

    def _run(self, blob: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.inputs.append(blob.shape)
        if self.works == 0:
            raise self.error
        self.works -= 1
        return end_to_end((10, 10, 50, 90, 0.9, PERSON))

    # As OpenCV's network.

    def setInput(self, blob: npt.NDArray[np.float32]) -> None:  # noqa: N802 - OpenCV's name
        self._blob = blob

    def forward(self) -> npt.NDArray[np.float32]:
        return self._run(self._blob)

    # As an ONNX Runtime session.

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="images")]

    def run(
        self, _outputs: None, feeds: dict[str, npt.NDArray[np.float32]]
    ) -> list[npt.NDArray[np.float32]]:
        return [self._run(feeds["images"])]


def fake(monkeypatch: pytest.MonkeyPatch, engine: Engine, works: int = 1) -> FakeModel:
    """Has ``engine`` load every model as a :class:`FakeModel`, and returns it."""
    if engine == "opencv":
        model = FakeModel(works, cv2.error("Insufficient memory"))
        monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *_: model)
        monkeypatch.setattr(cv2, "setNumThreads", lambda count: setattr(model, "threads", count))
        return model

    model = FakeModel(
        works,
        onnxruntime_pybind11_state.RuntimeException(
            "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Insufficient memory"
        ),
    )

    def session(_path: str, options: onnxruntime.SessionOptions, providers: list[str]) -> FakeModel:
        assert providers == ["CPUExecutionProvider"]
        model.threads = options.intra_op_num_threads
        return model

    monkeypatch.setattr(onnxruntime, "InferenceSession", session)
    return model


@pytest.fixture(params=["onnxruntime", "opencv"])
def engine(request: pytest.FixtureRequest) -> Engine:
    name: Engine = request.param
    return name


@pytest.fixture
def fake_model(tmp_path: Path) -> Path:
    path = tmp_path / "detector.onnx"
    path.write_bytes(b"a model")
    return path


class TestFailing:
    FRAME = np.zeros((640, 640, 3), dtype=np.uint8)

    def test_a_model_that_will_not_run_at_all_is_the_configurations_fault(
        self, fake_model: Path, monkeypatch: pytest.MonkeyPatch, engine: Engine
    ) -> None:
        fake(monkeypatch, engine, works=0)
        detector = Detector(fake_model, engine=engine, input_size=640, confidence=0.35)

        with pytest.raises(
            ModelError,
            match=r"input_size must be the size it was exported at \(Insufficient memory\)",
        ):
            detector.detect(self.FRAME)

    def test_so_is_one_that_says_its_size_and_will_not_run_at_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: Engine
    ) -> None:
        path = tmp_path / "detector.onnx"
        path.write_bytes(onnx_model(1, 3, 640, 640))
        fake(monkeypatch, engine, works=0)
        detector = Detector(path, engine=engine, confidence=0.35)

        with pytest.raises(ModelError, match=r"640x640 input, the size it was exported at \("):
            detector.detect(self.FRAME)

    def test_a_failure_after_it_has_run_is_not(
        self, fake_model: Path, monkeypatch: pytest.MonkeyPatch, engine: Engine
    ) -> None:
        model = fake(monkeypatch, engine, works=1)
        detector = Detector(fake_model, engine=engine, input_size=640, confidence=0.35)

        assert len(detector.detect(self.FRAME)) == 1
        with pytest.raises(type(model.error), match="Insufficient memory") as raised:
            detector.detect(self.FRAME)
        assert not isinstance(raised.value, ModelError)


class TestThreads:
    def test_unless_told_it_runs_on_every_cpu_up_to_a_few(
        self, fake_model: Path, monkeypatch: pytest.MonkeyPatch, engine: Engine
    ) -> None:
        model = fake(monkeypatch, engine)
        threads = []
        for cpus in (12, 2):
            monkeypatch.setattr(cv2, "getNumberOfCPUs", lambda cpus=cpus: cpus)
            Detector(fake_model, engine=engine, input_size=640, confidence=0.35)
            threads.append(model.threads)

        assert threads == [MAX_THREADS[engine], 2]

    def test_the_configuration_says_how_many(
        self, fake_model: Path, monkeypatch: pytest.MonkeyPatch, engine: Engine
    ) -> None:
        model = fake(monkeypatch, engine)

        load(DetectionModel(path=fake_model, engine=engine, threads=12))

        assert model.threads == 12


class TestInputSize:
    @pytest.fixture
    def model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Path]:
        """Writes a model with an input of the given shape, which loads as one that runs."""
        fake(monkeypatch, "onnxruntime")

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
        assert size[0] % 32 == 0
        assert size[1] % 32 == 0

    @pytest.mark.parametrize(
        ("dims", "size"), [((1, 3, 960, 960), (960, 960)), ((1, 3, 384, 640), (640, 384))]
    )
    def test_left_unset_it_is_the_models_own(
        self, model: Callable[..., Path], dims: tuple[int, ...], size: tuple[int, int]
    ) -> None:
        assert load(DetectionModel(path=model(*dims))).input_size == size

    def test_set_it_must_be_the_models_own(self, model: Callable[..., Path]) -> None:
        path = model(1, 3, 960, 960)

        assert Detector(path, input_size=960, confidence=0.35).input_size == (960, 960)
        with pytest.raises(ModelError, match=r"exported to take 960x960 images, not the 640x640"):
            Detector(path, input_size=640, confidence=0.35)

    def test_a_model_that_is_not_square_cannot_be_given_a_size(
        self, model: Callable[..., Path]
    ) -> None:
        with pytest.raises(ModelError, match=r"exported to take 640x384 images, not the 640x640"):
            Detector(model(1, 3, 384, 640), input_size=640, confidence=0.35)

    def test_a_wide_frame_goes_to_a_wide_model_and_its_boxes_come_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        net = fake(monkeypatch, "onnxruntime")
        path = tmp_path / "detector.onnx"
        path.write_bytes(onnx_model(1, 3, 384, 640))

        # The net finds a person at (10, 10)-(50, 90) of its 640x384 input, which a
        # 2560x1440 frame fills at a quarter of its size, below 12 rows of padding.
        found = Detector(path, confidence=0.35).detect(np.zeros((1440, 2560, 3), np.uint8))

        assert net.inputs == [(1, 3, 384, 640)]
        assert boxes(found) == [("person", 40, -8, 160, 320)]

    @pytest.mark.parametrize(
        ("configured", "size"),
        [(None, (DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)), (800, (800, 800))],
    )
    def test_a_model_that_takes_any_size_is_run_at_the_one_configured(
        self, model: Callable[..., Path], configured: int | None, size: tuple[int, int]
    ) -> None:
        path = model("batch", 3, "height", "width")

        assert Detector(path, input_size=configured, confidence=0.35).input_size == size
