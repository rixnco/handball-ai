"""Tracking tests: ghost removal, ID stability, effective frame rate."""

from __future__ import annotations

import numpy as np
import supervision as sv

from handball_ai.tracking import track_detections


def _det(x: float, y: float, class_id: int = 0) -> sv.Detections:
    """A 20x40 px detection with its top left corner at (x, y)."""
    return sv.Detections(
        xyxy=np.array([[x, y, x + 20, y + 40]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([class_id], dtype=int),
    )


def _moving_object(nb_frames: int, x0: float, dx: float) -> dict[int, sv.Detections]:
    """An object moving steadily, present on every frame."""
    return {fi: _det(x0 + fi * dx, 100.0) for fi in range(nb_frames)}


def test_persistent_object_gets_a_stable_id() -> None:
    """An object present on many frames gets a single tracker_id."""
    seq = _moving_object(nb_frames=20, x0=100.0, dx=2.0)
    _dets, final_class = track_detections(seq, fps=30.0, subsample=1)
    assert len(final_class) == 1
    assert next(iter(final_class.values())) == 0


def test_one_frame_ghost_removed() -> None:
    """A detection isolated on 1 frame creates no tracker (minimum_consecutive_frames)."""
    seq: dict[int, sv.Detections] = {}
    # Stable object over 15 frames (around x=100, moving slowly)
    for fi in range(15):
        seq[fi] = _det(100.0 + fi * 2, 100.0)
    # Ghost: on frame 7, add a second isolated detection far away
    x_obj = 100.0 + 7 * 2
    seq[7] = sv.Detections(
        xyxy=np.array(
            [[x_obj, 100.0, x_obj + 20, 140.0], [500.0, 300.0, 520.0, 340.0]],
            dtype=np.float32,
        ),
        confidence=np.array([0.9, 0.9], dtype=np.float32),
        class_id=np.array([0, 0], dtype=int),
    )
    _dets, final_class = track_detections(
        seq, fps=30.0, subsample=1, minimum_consecutive_frames=3
    )
    assert len(final_class) == 1


def test_effective_frame_rate_accounts_for_subsample() -> None:
    """The frame rate given to ByteTrack accounts for the subsample.

    Checked indirectly: with subsample=3 and fps=30, the tracker works
    (effective frame rate = 10) and tracks the object as a single ID.
    """
    seq = _moving_object(nb_frames=15, x0=100.0, dx=3.0)
    _dets, final_class = track_detections(seq, fps=30.0, subsample=3)
    assert len(final_class) == 1


def test_minimum_consecutive_frames_can_be_disabled() -> None:
    """With minimum_consecutive_frames=1, an ID is assigned on the first frame."""
    seq = _moving_object(nb_frames=5, x0=100.0, dx=1.0)
    tracked, _final_class = track_detections(
        seq, fps=30.0, subsample=1, minimum_consecutive_frames=1
    )
    first = tracked[0]
    assert first.tracker_id is not None
    assert len(first) == 1


def test_two_objects_two_ids() -> None:
    """Two distinct persistent objects -> two trackers."""
    seq: dict[int, sv.Detections] = {}
    for fi in range(15):
        seq[fi] = sv.Detections(
            xyxy=np.array(
                [[100 + fi * 2, 100, 120 + fi * 2, 140],
                 [400 - fi * 2, 100, 420 - fi * 2, 140]],
                dtype=np.float32,
            ),
            confidence=np.array([0.9, 0.9], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
        )
    _dets, final_class = track_detections(seq, fps=30.0, subsample=1)
    assert len(final_class) == 2
