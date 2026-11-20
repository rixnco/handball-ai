"""Multi-object tracking with the Ultralytics trackers, with class stabilization.

Tracking assigns a stable ID to objects across frames. Both backends go
through `model.track`:

- `bytetrack`: motion only, fast.
- `botsort`: motion + appearance re-identification (ReID), keeps the IDs on
  contact and occlusion, slower.

The class predicted by the detector may change from one frame to the next for
the same tracker_id; it is stabilized by a majority vote over the whole clip.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import supervision as sv

from handball_ai.video import seek_capture

logger = logging.getLogger(__name__)

TRACKERS_DIR = Path(__file__).parent / "trackers"
# Tracker configurations shipped with the package (Ultralytics YAML schema).
TRACKER_CONFIGS: dict[str, Path] = {
    "bytetrack": TRACKERS_DIR / "bytetrack.yaml",
    "botsort": TRACKERS_DIR / "botsort_reid.yaml",
}


class TrackingDetector(Protocol):
    """Interface used by `track_video`, implemented by `Detector` and test stubs."""

    def track(self, frame: np.ndarray, tracker_config: str | Path) -> sv.Detections: ...

    def reset_tracking(self) -> None: ...


def majority_class(tracked_detections: dict[int, sv.Detections]) -> dict[int, int]:
    """Return tracker_id -> most frequent class over the clip."""
    observed: dict[int, Counter] = defaultdict(Counter)
    for dets in tracked_detections.values():
        if dets.tracker_id is None:
            continue
        for tid, cid in zip(dets.tracker_id, dets.class_id, strict=True):
            observed[int(tid)][int(cid)] += 1
    return {tid: int(counter.most_common(1)[0][0]) for tid, counter in observed.items()}


def track_video(
    video_path: str | Path,
    detector: TrackingDetector,
    *,
    tracker: str = "bytetrack",
    subsample: int = 1,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> tuple[dict[int, sv.Detections], dict[int, int]]:
    """Detect and track the objects of a video, then stabilize their class.

    The tracker state is reset before the first frame, so that IDs never
    carry over from a previous video. Only one frame out of `subsample` is fed
    to the tracker; `track_buffer` in the tracker YAML counts those frames.

    Args:
        video_path: video file
        detector: object exposing `track` and `reset_tracking`
        tracker: "bytetrack" or "botsort"
        subsample: 1 frame out of N fed to the tracker
        start_frame: first frame index in the source
        max_frames: maximum number of frames read from start_frame

    Returns:
        (tracked_detections, final_class)
        - tracked_detections: frame index relative to start_frame -> Detections
          with tracker_id, for the sampled frames
        - final_class: tracker_id -> stabilized class (majority vote)

    Raises:
        ValueError: unknown tracker or invalid subsample
        RuntimeError: the video cannot be opened
    """
    if tracker not in TRACKER_CONFIGS:
        raise ValueError(
            f"Unknown tracker: {tracker!r} (expected one of {', '.join(TRACKER_CONFIGS)})"
        )
    if subsample < 1:
        raise ValueError(f"subsample must be >= 1, got {subsample}")
    config = TRACKER_CONFIGS[tracker]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    if max_frames is not None:
        total = min(total, max_frames)

    tracked_detections: dict[int, sv.Detections] = {}
    detector.reset_tracking()
    try:
        seek_capture(cap, start_frame)
        for index in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if index % subsample == 0:
                tracked_detections[index] = detector.track(frame, config)
    finally:
        cap.release()

    final_class = majority_class(tracked_detections)
    logger.info(
        "Tracking done (%s): %d trackers over %d analysed frames (subsample=%d)",
        tracker,
        len(final_class),
        len(tracked_detections),
        subsample,
    )
    return tracked_detections, final_class
