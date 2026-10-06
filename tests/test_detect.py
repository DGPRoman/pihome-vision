from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pihome_vision.detect import (
    COCO_CLASSES,
    Detection,
    Detector,
    Letterbox,
    ModelError,
    decode,
    file_sha256,
    letterbox,
)

PERSON, BICYCLE, CAR, CAT, TRUCK, DOG = 0, 1, 2, 15, 7, 16

#: A frame that fills the square exactly, so boxes decode to the numbers written.
IDENTITY = Letterbox(scale=1.0, left=0, top=0)


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
