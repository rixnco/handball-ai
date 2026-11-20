"""Handball object detection with a fine-tuned Ultralytics YOLO model.

Detections are returned as `supervision.Detections` whose `class_id` follows
`CLASSES_HANDBALL`, whatever the class order of the training dataset.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import supervision as sv

from pivot_ai.config import CLASSES_HANDBALL
from pivot_ai.video import positionner_capture

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE = 0.35


class DetectorLike(Protocol):
    """Interface used by the pipeline, implemented by `Detector` and test stubs."""

    def detect(self, frames: list[np.ndarray]) -> list[sv.Detections]: ...


def normalize_class_name(name: str) -> str:
    """Normalize a class name: case, separators and simple plural."""
    normalized = name.strip().lower().replace("_", " ").replace("-", " ")
    if len(normalized) > 1 and normalized.endswith("s"):
        normalized = normalized[:-1]
    return normalized


def build_class_map(model_names: dict[int, str], exclude: Iterable[str] = ()) -> dict[int, int]:
    """Map the model class ids to `CLASSES_HANDBALL` ids by name.

    Args:
        model_names: model class names, typically `model.names` from Ultralytics
        exclude: class names to ignore

    Returns:
        Model class id -> handball class id, for kept handball classes only.

    Raises:
        ValueError: unknown excluded class, model without handball classes,
            or no class left after exclusion.
    """
    targets = {normalize_class_name(name): cid for name, cid in CLASSES_HANDBALL.items()}
    names = {int(cid): normalize_class_name(name) for cid, name in model_names.items()}
    excluded = {normalize_class_name(name) for name in exclude}

    unknown = excluded.difference(names.values())
    if unknown:
        raise ValueError(
            f"Unknown excluded classes: {', '.join(sorted(unknown))}. "
            f"Model classes: {', '.join(model_names.values())}."
        )
    handball = {cid: targets[name] for cid, name in names.items() if name in targets}
    if not handball:
        raise ValueError(
            f"The model has no handball class ({', '.join(CLASSES_HANDBALL)}). "
            f"Model classes: {', '.join(model_names.values())}."
        )
    ignored = sorted(model_names[cid] for cid in names if cid not in handball)
    if ignored:
        logger.warning("Model classes ignored (not handball): %s", ", ".join(ignored))

    class_map = {cid: target for cid, target in handball.items() if names[cid] not in excluded}
    if not class_map:
        raise ValueError("No class left to detect after exclusion.")
    return class_map


def to_handball_classes(detections: sv.Detections, class_map: dict[int, int]) -> sv.Detections:
    """Keep the mapped classes and convert their ids to `CLASSES_HANDBALL`."""
    if detections.class_id is None or len(detections) == 0:
        return detections
    detections = detections[np.isin(detections.class_id, list(class_map))]
    detections.class_id = np.array([class_map[int(c)] for c in detections.class_id], dtype=int)
    return detections


class Detector:
    """YOLO detector fine-tuned on handball.

    Usage:
        detector = Detector("models/handball_yolo26m.pt")
        detections = detector.detect([frame])
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        device: str | None = None,
        exclude: Iterable[str] = (),
    ) -> None:
        """Load the weights and map their classes to `CLASSES_HANDBALL`.

        Args:
            weights: fine-tuned YOLO weights file (.pt)
            confidence: minimum confidence, between 0 and 1
            device: "cuda", "cpu", "mps"... None lets Ultralytics choose
            exclude: model class names to ignore

        Raises:
            FileNotFoundError: missing weights file
            ValueError: invalid confidence or class configuration
        """
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights}")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

        from ultralytics import YOLO  # lazy import: torch is slow to load

        self.model = YOLO(str(weights))
        self.confidence = confidence
        self.device = device
        self.class_map = build_class_map(self.model.names, exclude)
        logger.info(
            "Model %s loaded, classes: %s",
            weights,
            {self.model.names[cid]: target for cid, target in self.class_map.items()},
        )

    def _options(self) -> dict:
        return {
            "conf": self.confidence,
            "classes": list(self.class_map),
            "device": self.device,
            "verbose": False,
        }

    def detect(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        """Detect objects on BGR frames, processed as a single batch."""
        results = self.model.predict(frames, **self._options())
        return [
            to_handball_classes(sv.Detections.from_ultralytics(r), self.class_map)
            for r in results
        ]

    def track(self, frame: np.ndarray, tracker_config: str | Path) -> sv.Detections:
        """Detect and track on one frame with the Ultralytics tracker (stateful)."""
        results = self.model.track(
            frame, persist=True, tracker=str(tracker_config), **self._options()
        )
        return to_handball_classes(sv.Detections.from_ultralytics(results[0]), self.class_map)


def detect_video(
    video_path: str | Path,
    detector: DetectorLike,
    *,
    subsample: int = 1,
    batch_size: int = 8,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> Iterator[tuple[int, sv.Detections]]:
    """Read the video sequentially and yield the detections of sampled frames.

    Args:
        video_path: video file
        detector: object exposing `detect(list[ndarray])`
        subsample: keep 1 frame out of N
        batch_size: number of frames per detection call, to fit GPU memory
        start_frame: first frame index in the source
        max_frames: maximum number of frames read from start_frame

    Yields:
        (frame index relative to start_frame, detections), in video order.

    Raises:
        ValueError: invalid subsample or batch_size
        RuntimeError: the video cannot be opened
    """
    if subsample < 1:
        raise ValueError(f"subsample must be >= 1, got {subsample}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    if max_frames is not None:
        total = min(total, max_frames)

    frames: list[np.ndarray] = []
    indices: list[int] = []
    nb_detected = 0
    try:
        positionner_capture(cap, start_frame)
        for index in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if index % subsample:
                continue
            frames.append(frame)
            indices.append(index)
            if len(frames) == batch_size:
                yield from zip(indices, detector.detect(frames), strict=True)
                nb_detected += len(frames)
                frames, indices = [], []
        if frames:
            yield from zip(indices, detector.detect(frames), strict=True)
            nb_detected += len(frames)
    finally:
        cap.release()

    logger.info(
        "Detection done: %d frames analysed out of %d (subsample=%d, batch_size=%d)",
        nb_detected,
        total,
        subsample,
        batch_size,
    )
