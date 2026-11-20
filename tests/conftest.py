"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv


def make_fake_video(
    path: Path,
    nb_frames: int = 50,
    fps: float = 25.0,
    width: int = 320,
    height: int = 240,
) -> None:
    """Write a small valid MP4 to test the pipelines end to end."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot create the test video")
    for i in range(nb_frames):
        frame = np.full((height, width, 3), (i * 3) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def make_detections(
    bboxes: list[tuple[float, float, float, float]],
    class_ids: list[int],
    confidences: list[float] | None = None,
) -> sv.Detections:
    """Build an sv.Detections without tracker_id (tracker input)."""
    n = len(bboxes)
    assert n == len(class_ids)
    if n == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(bboxes, dtype=np.float32),
        confidence=np.array(confidences or [1.0] * n, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )
