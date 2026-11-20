"""Multi-object tracking with ByteTrack or BoT-SORT, with class stabilization.

Tracking assigns a stable ID to objects across frames. The class predicted by
the detector may change from one frame to the next for the same tracker_id;
it is stabilized by a majority vote over the whole clip.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import supervision as sv

from handball_ai.video import seek_capture

if TYPE_CHECKING:
    from handball_ai.detection import Detector

logger = logging.getLogger(__name__)

# BoT-SORT + ReID configuration shipped with the package.
BOTSORT_CONFIG_PATH = Path(__file__).parent / "trackers" / "botsort_reid.yaml"


def _majority_class(tracked_detections: dict[int, sv.Detections]) -> dict[int, int]:
    """Return tracker_id -> most frequent class over the clip."""
    observed: dict[int, Counter] = defaultdict(Counter)
    for dets in tracked_detections.values():
        if dets.tracker_id is None:
            continue
        for tid, cid in zip(dets.tracker_id, dets.class_id, strict=True):
            observed[int(tid)][int(cid)] += 1
    return {tid: int(counter.most_common(1)[0][0]) for tid, counter in observed.items()}


def track_detections(
    detections_by_frame: dict[int, sv.Detections],
    fps: float,
    subsample: int = 1,
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    minimum_matching_threshold: float = 0.8,
    minimum_consecutive_frames: int = 2,
) -> tuple[dict[int, sv.Detections], dict[int, int]]:
    """Apply ByteTrack and stabilize the class per tracker_id.

    The frame rate given to ByteTrack accounts for the subsample: only one
    frame out of `subsample` is fed, so the motion model must use the
    effective frame rate `fps / subsample` (otherwise the Kalman prediction
    is off by a factor `subsample`, which fragments the tracking).

    `minimum_consecutive_frames > 1` removes ghost tracks (isolated detections
    on 1-2 frames that used to get a short-lived tracker_id).

    Args:
        detections_by_frame: detector output, frame index -> Detections
        fps: real frame rate of the clip
        subsample: 1 frame out of N actually fed to the tracker
        track_activation_threshold: min confidence to start a track
        lost_track_buffer: number of (effective) frames a lost track survives,
            to re-match a player after occlusion without changing its ID
        minimum_matching_threshold: IoU association threshold
        minimum_consecutive_frames: consecutive frames required before a
            track is valid (>= 2 removes 1-frame ghosts)

    Returns:
        (tracked_detections, final_class)
        - tracked_detections: frame index -> Detections with tracker_id
        - final_class: tracker_id -> stabilized class (majority vote)
    """
    effective_frame_rate = max(1, int(round(fps / max(1, subsample))))
    tracker = sv.ByteTrack(
        frame_rate=effective_frame_rate,
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold,
        minimum_consecutive_frames=minimum_consecutive_frames,
    )
    tracked_detections = {
        fi: tracker.update_with_detections(detections_by_frame[fi])
        for fi in sorted(detections_by_frame)
    }
    final_class = _majority_class(tracked_detections)

    logger.info(
        "Tracking done: %d stable trackers over %d frames",
        len(final_class),
        len(tracked_detections),
    )
    return tracked_detections, final_class


def track_video_botsort(
    video_path: str | Path,
    detector: Detector,
    subsample: int = 1,
    config_path: str | Path | None = None,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> tuple[dict[int, sv.Detections], dict[int, int]]:
    """Detect AND track with BoT-SORT + ReID (Ultralytics `model.track`).

    Unlike ByteTrack (motion only), BoT-SORT computes an appearance fingerprint
    per player and re-identifies it after occlusion, which strongly reduces ID
    jumps on contact. Detection and tracking are coupled in `model.track`
    (stateful, persist=True), so this function replaces both detect_video and
    track_detections.

    Classes are converted to HANDBALL_CLASSES by the detector.

    Args:
        video_path: video file
        detector: handball Detector
        subsample: 1 frame out of N fed to the tracker
        config_path: BoT-SORT yaml (default: shipped ReID configuration)
        max_frames: maximum number of frames read from start_frame
        start_frame: first frame index in the source

    Returns:
        (tracked_detections, final_class), same format as track_detections.

    Raises:
        RuntimeError: the video cannot be opened
    """
    config = str(config_path or BOTSORT_CONFIG_PATH)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    if max_frames is not None:
        total = min(total, max_frames)

    tracked_detections: dict[int, sv.Detections] = {}
    try:
        seek_capture(cap, start_frame)
        for fi in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if fi % subsample == 0:
                tracked_detections[fi] = detector.track(frame, config)
    finally:
        cap.release()

    final_class = _majority_class(tracked_detections)
    logger.info(
        "BoT-SORT tracking done: %d trackers over %d analysed frames (ReID)",
        len(final_class),
        len(tracked_detections),
    )
    return tracked_detections, final_class
