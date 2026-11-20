"""Access to the frames of the source video."""

from __future__ import annotations

import cv2


def seek_capture(cap: cv2.VideoCapture, start_frame: int) -> None:
    """Perform a single random access before sequential reading."""
    if start_frame < 0:
        raise ValueError("The start index must be positive or zero")
    if start_frame and not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
        raise RuntimeError(f"Cannot seek to frame {start_frame}")
