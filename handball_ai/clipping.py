"""Automatic split of a match into individual actions.

Simple heuristic approach, without ML, based on domain rules:
- An action = sequence with >= 6 players on the court
- Start: transition from 0-2 players to 6+ players
- End: back to < 4 players for more than 3 s

To be improved later with an ML event classification model.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import supervision as sv

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """An action detected in the match."""

    start_frame: int
    end_frame: int
    duration_s: float
    attacking_team: int | None  # 0, 1 or None when unknown
    mean_nb_players: float


def _count_players_per_frame(
    tracked_detections: dict[int, sv.Detections],
    final_class: dict[int, int],
    player_class_id: int,
) -> dict[int, int]:
    """Count the field players per frame, using the stabilized class.

    Args:
        tracked_detections: frame index -> Detections (with tracker_id)
        final_class: tracker_id -> stabilized class
        player_class_id: id of the field player class

    Returns:
        frame index -> number of field players
    """
    counts: dict[int, int] = {}
    for fi, dets in tracked_detections.items():
        if dets.tracker_id is None:
            counts[fi] = 0
            continue
        counts[fi] = sum(
            1 for tid in dets.tracker_id if final_class.get(int(tid)) == player_class_id
        )
    return counts


def detect_actions(
    tracked_detections: dict[int, sv.Detections],
    final_class: dict[int, int],
    player_class_id: int,
    fps: float,
    start_threshold: int = 6,
    end_threshold: int = 4,
    min_action_duration_s: float = 3.0,
    max_pause_s: float = 3.0,
) -> list[Action]:
    """Detect the actions of a match with a simple heuristic.

    Two-state machine (waiting, action) going through the frames in increasing
    order. Waiting -> action when the number of players >= start_threshold.
    Action -> waiting when the pause (players < end_threshold) lasts longer
    than max_pause_s. Actions shorter than min_action_duration_s are dropped.

    Args:
        tracked_detections: frame index -> Detections
        final_class: tracker_id -> stabilized class
        player_class_id: id of the field player class
        fps: frame rate
        start_threshold: min number of players to start an action
        end_threshold: number of players under which the game is paused
        min_action_duration_s: minimum duration of a valid action
        max_pause_s: maximum pause allowed inside an action

    Returns:
        list of Action sorted by start_frame.

    Raises:
        ValueError: fps <= 0 or inconsistent thresholds
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if end_threshold > start_threshold:
        raise ValueError("end_threshold cannot be greater than start_threshold")

    counts = _count_players_per_frame(tracked_detections, final_class, player_class_id)
    if not counts:
        return []

    sorted_frames = sorted(counts)
    action_start: int | None = None
    last_presence: int | None = None
    actions: list[Action] = []

    def close(end_frame: int) -> None:
        """Close the current action and keep it when it is long enough.

        mean_nb_players is computed over [action_start, end_frame], using only
        the frames actually analysed.
        """
        nonlocal action_start, last_presence
        assert action_start is not None
        duration_s = (end_frame - action_start) / fps
        if duration_s >= min_action_duration_s:
            window = [counts[f] for f in sorted_frames if action_start <= f <= end_frame]
            actions.append(
                Action(
                    start_frame=action_start,
                    end_frame=end_frame,
                    duration_s=float(duration_s),
                    attacking_team=None,
                    mean_nb_players=sum(window) / len(window),
                )
            )
        action_start = None
        last_presence = None

    for fi in sorted_frames:
        n = counts[fi]
        if action_start is None:
            if n >= start_threshold:
                action_start = fi
                last_presence = fi
        elif n >= end_threshold:
            last_presence = fi
        else:
            assert last_presence is not None
            if (fi - last_presence) / fps > max_pause_s:
                close(last_presence)

    # Close the running action at the end of the clip.
    if action_start is not None and last_presence is not None:
        close(last_presence)

    logger.info(
        "Action detection: %d actions kept over %d analysed frames",
        len(actions),
        len(sorted_frames),
    )
    return actions


def _check_ffmpeg() -> None:
    """Raise when ffmpeg is not available."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not found. Install it: "
            "apt-get install ffmpeg (Linux), brew install ffmpeg (Mac) "
            "or winget install ffmpeg (Windows)."
        )


def _read_fps(video_path: Path) -> float:
    """Read the frame rate of a video with OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Invalid fps ({fps}) for video: {video_path}")
    return fps


def cut_video_clips(
    source_video: str | Path,
    actions: list[Action],
    output_dir: str | Path,
    prefix: str = "action",
) -> list[Path]:
    """Cut the detected actions into separate MP4 files with ffmpeg.

    Uses -c copy to avoid re-encoding (fast, quality preserved). Bounds do not
    necessarily fall on keyframes: ffmpeg may round to the nearest keyframe.

    Args:
        source_video: full match video
        actions: actions to extract
        output_dir: output folder (created when missing)
        prefix: file name prefix (default "action")

    Returns:
        paths of the generated clips (may be shorter than `actions` when some
        cuts fail: warning + skip).

    Raises:
        FileNotFoundError: ffmpeg not installed or missing source
    """
    _check_ffmpeg()
    source = Path(source_video)
    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fps = _read_fps(source)
    clips: list[Path] = []

    for idx, action in enumerate(actions):
        start_s = action.start_frame / fps
        clip_path = output / f"{prefix}_{idx:03d}_{action.duration_s:.1f}s.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            str(source),
            "-t",
            f"{action.duration_s:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(clip_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Failed to cut action %d (%.1fs): %s",
                idx,
                action.duration_s,
                exc.stderr.decode("utf-8", errors="replace")[:200],
            )
            continue
        clips.append(clip_path)
        logger.info(
            "Clip written: %s (start=%.2fs, duration=%.2fs)",
            clip_path.name,
            start_s,
            action.duration_s,
        )

    logger.info("Clipping done: %d/%d clips written to %s", len(clips), len(actions), output)
    return clips
