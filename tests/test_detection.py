"""Detection module tests, without GPU or real YOLO weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import supervision as sv
import torch
from ultralytics.engine.results import Results

from handball_ai.config import HANDBALL_CLASSES
from handball_ai.detection import (
    Detector,
    build_class_map,
    detect_video,
    normalize_class_name,
    to_handball_classes,
)
from tests.conftest import make_fake_video

# Alphabetical order of the Roboflow exports: ball, goalkeeper, players, referees.
ROBOFLOW_NAMES = {0: "ball", 1: "goalkeeper", 2: "players", 3: "referees"}


class FakeYOLO:
    """Mimics `ultralytics.YOLO`: one detection of each class per frame."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.names = ROBOFLOW_NAMES
        self.calls: list[dict[str, Any]] = []

    def predict(self, frames: list[np.ndarray], **options: Any) -> list[Results]:
        self.calls.append(options)
        boxes = torch.tensor(
            [[10, 10, 20, 20, 0.9, cid] for cid in self.names], dtype=torch.float32
        )
        return [Results(frame, path="", names=self.names, boxes=boxes) for frame in frames]


@pytest.fixture
def weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Dummy weights file, loaded by FakeYOLO."""
    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    path = tmp_path / "handball.pt"
    path.write_bytes(b"")
    return path


def _detections(class_ids: list[int]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.tile(np.array([0, 0, 1, 1], dtype=np.float32), (len(class_ids), 1)),
        confidence=np.full(len(class_ids), 0.9, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------


def test_normalize_class_name() -> None:
    """Case, separators and simple plural are normalized."""
    assert normalize_class_name("Players") == "player"
    assert normalize_class_name("referees") == "referee"
    assert normalize_class_name("Goal_Keeper") == "goal keeper"
    assert normalize_class_name("ball") == "ball"


def test_build_class_map_roboflow_order() -> None:
    """The alphabetical Roboflow order is mapped to HANDBALL_CLASSES by name."""
    assert build_class_map(ROBOFLOW_NAMES) == {
        0: HANDBALL_CLASSES["ball"],
        1: HANDBALL_CLASSES["goalkeeper"],
        2: HANDBALL_CLASSES["players"],
        3: HANDBALL_CLASSES["referees"],
    }


def test_build_class_map_exclusion() -> None:
    """An excluded class, singular or plural, disappears."""
    class_map = build_class_map(ROBOFLOW_NAMES, exclude=["Ball", "referee"])
    assert class_map == {1: HANDBALL_CLASSES["goalkeeper"], 2: HANDBALL_CLASSES["players"]}


def test_build_class_map_ignores_non_handball_classes() -> None:
    """A non handball class (crowd) is ignored."""
    assert build_class_map({0: "players", 1: "crowd"}) == {0: HANDBALL_CLASSES["players"]}


@pytest.mark.parametrize(
    ("names", "exclude", "message"),
    [
        (ROBOFLOW_NAMES, ["crowd"], "Unknown excluded classes: crowd"),
        ({0: "person", 1: "car"}, [], "no handball class"),
        ({0: "players"}, ["players"], "No class left"),
    ],
    ids=["unknown-exclusion", "coco-model", "all-excluded"],
)
def test_build_class_map_errors(names: dict[int, str], exclude: list[str], message: str) -> None:
    """Invalid configurations: explicit ValueError."""
    with pytest.raises(ValueError, match=message):
        build_class_map(names, exclude)


def test_to_handball_classes_converts_and_filters() -> None:
    """Ids are converted; classes outside the map are dropped."""
    out = to_handball_classes(_detections([2, 0, 9]), {0: 3, 2: 0})
    assert out.class_id.tolist() == [0, 3]


def test_to_handball_classes_empty() -> None:
    """Empty detections go through without error."""
    assert len(to_handball_classes(sv.Detections.empty(), {0: 0})) == 0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def test_detector_missing_weights(tmp_path: Path) -> None:
    """Missing weights file: FileNotFoundError before any loading."""
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        Detector(tmp_path / "missing.pt")


def test_detector_invalid_confidence(weights: Path) -> None:
    """Confidence outside [0, 1]: ValueError."""
    with pytest.raises(ValueError, match="Confidence"):
        Detector(weights, confidence=1.5)


def test_detector_detect(weights: Path) -> None:
    """detect converts the classes and passes threshold and kept classes to YOLO."""
    detector = Detector(weights, confidence=0.5, exclude=["ball"])
    frames = [np.zeros((32, 32, 3), dtype=np.uint8)] * 2

    results = detector.detect(frames)

    assert len(results) == 2
    expected = sorted(HANDBALL_CLASSES[name] for name in ("goalkeeper", "players", "referees"))
    assert sorted(results[0].class_id.tolist()) == expected
    options = detector.model.calls[0]
    assert options["conf"] == 0.5
    assert options["classes"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# detect_video
# ---------------------------------------------------------------------------


class StubDetector:
    """Records the batch sizes and returns empty detections."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def detect(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        self.batches.append(len(frames))
        return [sv.Detections.empty() for _ in frames]


def test_detect_video_samples_in_batches(tmp_path: Path) -> None:
    """subsample, batch_size, start_frame and max_frames are honoured."""
    video = tmp_path / "clip.mp4"
    make_fake_video(video, nb_frames=30, fps=25.0, width=64, height=48)
    stub = StubDetector()

    results = dict(detect_video(
        video, stub, subsample=3, batch_size=4, start_frame=5, max_frames=20,
    ))

    assert list(results) == [0, 3, 6, 9, 12, 15, 18]
    assert stub.batches == [4, 3]


def test_detect_video_is_lazy(tmp_path: Path) -> None:
    """The generator only detects the first batch when stopped at the first item."""
    video = tmp_path / "clip.mp4"
    make_fake_video(video, nb_frames=20, fps=25.0, width=64, height=48)
    stub = StubDetector()

    generator = detect_video(video, stub, batch_size=4)
    index, _ = next(generator)
    generator.close()

    assert index == 0
    assert stub.batches == [4]


@pytest.mark.parametrize("params", [{"subsample": 0}, {"batch_size": 0}])
def test_detect_video_invalid_parameters(tmp_path: Path, params: dict[str, int]) -> None:
    """subsample or batch_size < 1: ValueError on the first item."""
    with pytest.raises(ValueError):
        next(detect_video(tmp_path / "x.mp4", StubDetector(), **params))
