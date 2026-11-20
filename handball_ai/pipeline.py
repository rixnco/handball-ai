"""Pipeline orchestrator: chains detection, tracking, teams, stats and clipping."""

from __future__ import annotations

import logging
import math
from collections.abc import Generator
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl
import supervision as sv

from handball_ai.clipping import Action, cut_video_clips, detect_actions
from handball_ai.config import HANDBALL_CLASSES, CourtConfig
from handball_ai.homography import CourtPoint, Homography, calibrate_homography
from handball_ai.metrics import summarize_tracking_quality
from handball_ai.radar import draw_radar
from handball_ai.stats import (
    PlayerPosition,
    compute_defensive_block_width,
    compute_player_stats,
)
from handball_ai.teams import classify_teams
from handball_ai.tracking import TrackingDetector, track_video
from handball_ai.video import seek_capture

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Display colours per role (BGR, consistent with radar.draw_legend)
# ---------------------------------------------------------------------------

COLOR_HOME: tuple[int, int, int] = (0, 255, 0)        # green
COLOR_AWAY: tuple[int, int, int] = (0, 0, 255)        # red
COLOR_GK: tuple[int, int, int] = (0, 215, 255)        # yellow-orange
COLOR_REF: tuple[int, int, int] = (255, 0, 255)       # magenta
COLOR_UNKNOWN: tuple[int, int, int] = (200, 200, 200)  # grey


@dataclass
class PipelineResult:
    """Full result of a clip or match processing.

    Attributes:
        source_video: input video
        radar_video: annotated video with the radar when generated, else None
        player_stats: Polars DataFrame aggregated per tracker
        positions_by_tracker: raw court positions (m) per tracker_id,
            useful for heatmaps and detailed spatial analyses.
        detected_actions: actions returned by detect_actions
        action_clips: paths of the ffmpeg action clips
        metadata: information dict (fps, nb_trackers, etc.)
    """

    source_video: Path
    radar_video: Path | None
    player_stats: pl.DataFrame
    positions_by_tracker: dict[int, list[PlayerPosition]]
    detected_actions: list[Action]
    action_clips: list[Path]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Projection of the detections onto the court
# ---------------------------------------------------------------------------


def project_detections_to_court(
    tracked_detections: dict[int, sv.Detections],
    final_class: dict[int, int],
    homography: Homography,
    court_config: CourtConfig | None = None,
    player_class_id: int = HANDBALL_CLASSES["players"],
    filter_zone: bool = True,
) -> dict[int, list[PlayerPosition]]:
    """Project the bboxes (bottom centre) onto the court (metres).

    Only trackers whose stabilized class == player_class_id are kept.
    Positions outside `is_in_zone` are dropped when `filter_zone=True`
    (removes benches and staff).

    Args:
        tracked_detections: frame index -> Detections (with tracker_id)
        final_class: tracker_id -> stabilized class
        homography: pixel -> metres projection
        court_config: court configuration (default: CourtConfig())
        player_class_id: id of the field player class
        filter_zone: apply the `is_in_zone` filter

    Returns:
        tracker_id -> list of (non interpolated) PlayerPosition, sorted by frame.
    """
    config = court_config or CourtConfig()
    positions_by_tracker: dict[int, list[PlayerPosition]] = {}

    for fi in sorted(tracked_detections):
        dets = tracked_detections[fi]
        if dets.tracker_id is None:
            continue
        for i in range(len(dets)):
            tid = int(dets.tracker_id[i])
            if final_class.get(tid) != player_class_id:
                continue
            x_m, y_m = homography.project_bbox_feet(dets.xyxy[i])
            if filter_zone and not config.is_in_zone((x_m, y_m)):
                continue
            positions_by_tracker.setdefault(tid, []).append(
                PlayerPosition(frame_idx=int(fi), x_m=float(x_m), y_m=float(y_m))
            )

    logger.info("Court projection: %d trackers kept", len(positions_by_tracker))
    return positions_by_tracker


# ---------------------------------------------------------------------------
# Track filtering and interpolation (interpolation is for display only)
# ---------------------------------------------------------------------------


def filter_short_tracks(
    positions_by_tracker: dict[int, list[PlayerPosition]],
    min_frames: int,
) -> dict[int, list[PlayerPosition]]:
    """Drop the trackers seen on too few real frames (fragments).

    ByteTrack (without appearance re-identification) splits occluded players
    into many short IDs. This filter removes the long tail of fragments to
    keep only substantial trajectories, usable for the statistics.

    Args:
        positions_by_tracker: real positions per tracker_id
        min_frames: minimum number of real positions to keep a tracker
            (<= 1 disables the filter)

    Returns:
        filtered tracker_id -> positions.
    """
    if min_frames <= 1:
        return positions_by_tracker
    kept = {
        tid: positions
        for tid, positions in positions_by_tracker.items()
        if sum(1 for p in positions if not p.interpolated) >= min_frames
    }
    logger.info(
        "Fragment filter: %d trackers kept out of %d (min_frames=%d)",
        len(kept),
        len(positions_by_tracker),
        min_frames,
    )
    return kept


def interpolate_positions(
    positions_by_tracker: dict[int, list[PlayerPosition]],
    max_gap_frames: int = 10,
) -> dict[int, list[PlayerPosition]]:
    """Linearly interpolate the missing frames between consecutive appearances.

    Interpolates ONLY when the gap between two appearances is <= max_gap_frames.
    This covers the frames skipped by the subsample without extrapolating over
    long disappearances (occlusion or leaving the field of view).

    Generated positions carry `interpolated=True` so that the statistics can
    ignore them.

    Args:
        positions_by_tracker: real (non interpolated) positions
        max_gap_frames: maximum gap interpolated (inclusive)

    Returns:
        tracker_id -> extended list (real + interpolated), sorted by frame.
    """
    result: dict[int, list[PlayerPosition]] = {}
    nb_interpolated = 0

    for tid, positions in positions_by_tracker.items():
        positions = sorted(positions, key=lambda p: p.frame_idx)
        if not positions:
            result[tid] = []
            continue

        extended: list[PlayerPosition] = [positions[0]]
        for prev, nxt in zip(positions, positions[1:], strict=False):
            gap = nxt.frame_idx - prev.frame_idx
            if 1 < gap <= max_gap_frames:
                for k in range(1, gap):
                    alpha = k / gap
                    extended.append(
                        PlayerPosition(
                            frame_idx=prev.frame_idx + k,
                            x_m=prev.x_m + alpha * (nxt.x_m - prev.x_m),
                            y_m=prev.y_m + alpha * (nxt.y_m - prev.y_m),
                            interpolated=True,
                        )
                    )
                    nb_interpolated += 1
            extended.append(nxt)
        result[tid] = extended

    logger.info(
        "Interpolation: %d positions interpolated for %d trackers",
        nb_interpolated,
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Tracker colour from its role and team
# ---------------------------------------------------------------------------


def tracker_color(
    tracker_id: int,
    final_class: dict[int, int],
    final_team: dict[int, int],
    home_cluster: int,
) -> tuple[int, int, int]:
    """Return the BGR colour of a tracker."""
    cls = final_class.get(tracker_id)
    if cls == HANDBALL_CLASSES["goalkeeper"]:
        return COLOR_GK
    if cls == HANDBALL_CLASSES["referees"]:
        return COLOR_REF
    if cls == HANDBALL_CLASSES["players"]:
        team = final_team.get(tracker_id)
        if team is None:
            return COLOR_UNKNOWN
        return COLOR_HOME if team == home_cluster else COLOR_AWAY
    return COLOR_UNKNOWN


# ---------------------------------------------------------------------------
# Video with the radar below the source
# ---------------------------------------------------------------------------


def generate_radar_video(
    source_video: str | Path,
    tracked_detections: dict[int, sv.Detections],
    interpolated_positions: dict[int, list[PlayerPosition]],
    final_class: dict[int, int],
    final_team: dict[int, int],
    home_cluster: int,
    output_path: str | Path,
    court_config: CourtConfig | None = None,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> Path:
    """Generate a video with the annotated source above the 2D radar.

    Output resolution: broadcast_w x (broadcast_h + radar_h), mp4v encoding.
    Broadcast bboxes are annotated with the team colour and the tracker_id.
    The radar uses the interpolated positions (smooth rendering).

    Args:
        source_video: original video
        tracked_detections: detections + tracker_id per sampled frame
        interpolated_positions: court positions (with interpolation)
        final_class: tracker_id -> stabilized class
        final_team: tracker_id -> team 0 or 1
        home_cluster: index of the home team cluster
        output_path: output MP4 path
        court_config: court configuration (default: CourtConfig())
        max_frames: maximum number of frames written
        start_frame: first frame index in the source

    Returns:
        Path of the generated file.

    Raises:
        RuntimeError: the source or the writer cannot be opened
    """
    config = court_config or CourtConfig()
    source = Path(source_video)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    if max_frames is not None:
        nb_frames = min(nb_frames, max_frames)
    if fps <= 0 or nb_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: fps={fps}, n={nb_frames}")

    if config.broadcast_w != config.radar_w:
        cap.release()
        raise RuntimeError("The video and the radar must have the same width")
    output_w = config.broadcast_w
    output_h = config.broadcast_h + config.radar_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter: {output_path}")

    # Reverse index: frame -> list of (tid, x_m, y_m) for the radar
    positions_by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for tid, positions in interpolated_positions.items():
        for p in positions:
            positions_by_frame.setdefault(p.frame_idx, []).append((tid, p.x_m, p.y_m))

    try:
        seek_capture(cap, start_frame)
        for fi in range(nb_frames):
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = min(config.broadcast_w / w, config.broadcast_h / h)
            width = max(1, round(w * scale))
            height = max(1, round(h * scale))
            x = (config.broadcast_w - width) // 2
            y = (config.broadcast_h - height) // 2
            video_panel = np.zeros((config.broadcast_h, config.broadcast_w, 3), dtype=np.uint8)
            broadcast = video_panel[y:y + height, x:x + width]
            cv2.resize(frame, (width, height), dst=broadcast)

            # Annotate the boxes (sampled frames only)
            dets = tracked_detections.get(fi)
            if dets is not None and dets.tracker_id is not None:
                sx = width / w
                sy = height / h
                for i in range(len(dets)):
                    tid = int(dets.tracker_id[i])
                    color = tracker_color(tid, final_class, final_team, home_cluster)
                    x1, y1, x2, y2 = dets.xyxy[i]
                    p1 = (int(x1 * sx), int(y1 * sy))
                    p2 = (int(x2 * sx), int(y2 * sy))
                    cv2.rectangle(broadcast, p1, p2, color, 2)
                    cv2.putText(
                        broadcast,
                        str(tid),
                        (p1[0], max(0, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            # Radar: positions of the current frame (interpolated included)
            radar_positions = [
                (x_m, y_m, tracker_color(tid, final_class, final_team, home_cluster), "", tid)
                for tid, x_m, y_m in positions_by_frame.get(fi, [])
            ]
            radar = draw_radar(radar_positions, config, show_teams=bool(final_team))
            writer.write(np.vstack([video_panel, radar]))
    finally:
        cap.release()
        writer.release()

    logger.info("Radar video written: %s (%dx%d @ %.1f fps)", output_path, output_w, output_h, fps)
    return output_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _read_video_metadata(path: Path) -> tuple[float, int]:
    """Read the fps and number of frames of a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or nb <= 0:
        raise RuntimeError(f"Invalid video metadata: fps={fps}, n={nb}")
    return fps, nb


def _read_indexed_frames(
    path: Path, indices: list[int], start_frame: int = 0,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield the requested frames one by one, in video order."""
    if not indices:
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    wanted = {int(i) for i in indices}
    try:
        seek_capture(cap, start_frame)
        for fi in range(max(wanted) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if fi in wanted:
                yield fi, frame
    finally:
        cap.release()


def process_match(
    video_path: str | Path,
    court_mapping: dict[str, CourtPoint],
    output_dir: str | Path,
    detector: TrackingDetector,
    subsample: int = 2,
    with_radar_video: bool = True,
    with_clips: bool = True,
    court_config: CourtConfig | None = None,
    min_track_frames: int = 0,
    tracker: str = "bytetrack",
    duration: float | None = None,
    with_teams: bool = True,
    with_stats: bool = True,
    start: float = 0.0,
    correct_fisheye: bool = True,
) -> PipelineResult:
    """Full pipeline: video -> stats + action clips + radar video.

    Args:
        video_path: source video
        court_mapping: points used to calibrate the homography
        output_dir: folder for the statistics, radar video and clips
        detector: handball detector (`Detector`, or a stub in tests)
        subsample: 1 frame out of N fed to the tracker (2 by default)
        with_radar_video: generate the annotated video with the radar
        with_clips: cut the detected actions with ffmpeg
        court_config: court configuration (default: CourtConfig())
        min_track_frames: drop the trackers seen on fewer than N real frames
            (fragments). 0 disables the filter. ~8-12 is recommended in
            production to keep only usable trajectories.
        tracker: tracking backend. "bytetrack" (default, fast, motion only) or
            "botsort" (appearance ReID: keeps the IDs on contact/occlusion,
            slower).
        duration: maximum duration in seconds from start
        with_teams: classify the teams by jersey colour
        with_stats: compute and write the player statistics
        start: start position in seconds in the source video
        correct_fisheye: correct the projection when points A to F are available

    Returns:
        PipelineResult with all the produced artefacts.

    Raises:
        FileNotFoundError: the source video does not exist
        ValueError: invalid start, duration or tracker
    """
    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = court_config or CourtConfig()

    # 1. Homography
    homography = calibrate_homography(court_mapping, correct_fisheye=correct_fisheye)
    logger.info(
        "Homography calibrated: %d points, method=%s", homography.nb_points, homography.method
    )

    # 2. Video metadata
    fps, nb_source_frames = _read_video_metadata(source)
    if not math.isfinite(start) or not 0 <= start < nb_source_frames / fps:
        raise ValueError("The start must be inside the video")
    start_frame = min(round(start * fps), nb_source_frames - 1)
    nb_frames_total = nb_source_frames - start_frame
    frame_limit = None
    if duration is not None:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("The duration must be finite and strictly positive")
        if duration < nb_frames_total / fps:
            frame_limit = math.ceil(duration * fps)
            nb_frames_total = frame_limit
    logger.info(
        "Video: %d frames at %.1f fps, from frame %d", nb_frames_total, fps, start_frame
    )

    # 3-4. Detection + tracking with the chosen backend
    tracked_detections, final_class = track_video(
        source,
        detector,
        tracker=tracker,
        subsample=subsample,
        start_frame=start_frame,
        max_frames=frame_limit,
    )

    # 5. Teams (needs the sampled frames to read the colours)
    final_team: dict[int, int] = {}
    home_cluster = 0
    if with_teams:
        logger.info("Team classification: reading jersey colours")
        with closing(
            _read_indexed_frames(source, list(tracked_detections), start_frame)
        ) as frames:
            final_team, home_cluster = classify_teams(
                tracked_detections,
                frames,
                final_class,
                player_class_id=HANDBALL_CLASSES["players"],
            )

    # 6. Court projection
    positions_by_tracker = project_detections_to_court(
        tracked_detections, final_class, homography, court_config=config
    )

    # 6b. Tracking quality KPIs (on positions BEFORE filtering: that is where
    # fragmentation and ID switches live)
    nb_trackers_before_filter = len(positions_by_tracker)
    quality = summarize_tracking_quality(positions_by_tracker) if with_stats else None

    # 6c. Short fragment filter (tail of short-lived tracks)
    positions_by_tracker = filter_short_tracks(positions_by_tracker, min_frames=min_track_frames)

    # 7. Interpolation (radar rendering only)
    max_gap = max(2, subsample * 5)
    interpolated = interpolate_positions(positions_by_tracker, max_gap_frames=max_gap)

    # 8. Player statistics (real positions only)
    stats_df = pl.DataFrame()
    if with_stats:
        stats_df = compute_player_stats(positions_by_tracker, fps=fps)
        csv_path = output / "player_stats.csv"
        parquet_path = output / "player_stats.parquet"
        stats_df.write_csv(csv_path)
        stats_df.write_parquet(parquet_path)
        logger.info("Statistics saved: %s / %s", csv_path, parquet_path)

        # Defensive block width for both teams (when teams are classified)
        if final_team:
            for team in (0, 1):
                block_df = compute_defensive_block_width(
                    positions_by_tracker, final_team, defending_team=team, fps=fps
                )
                suffix = "home" if team == home_cluster else "away"
                block_df.write_csv(output / f"defensive_block_width_{suffix}.csv")

    # 9. Radar video
    radar_video: Path | None = None
    if with_radar_video:
        radar_video = output / "match_radar.mp4"
        generate_radar_video(
            source,
            tracked_detections,
            interpolated,
            final_class,
            final_team,
            home_cluster,
            radar_video,
            court_config=config,
            max_frames=frame_limit,
            start_frame=start_frame,
        )

    # 10. Action clips (on raw detections, not interpolated ones)
    actions: list[Action] = []
    clips: list[Path] = []
    if with_clips:
        actions = detect_actions(
            tracked_detections,
            final_class,
            player_class_id=HANDBALL_CLASSES["players"],
            fps=fps,
        )
        if actions:
            source_actions = [
                replace(
                    action,
                    start_frame=action.start_frame + start_frame,
                    end_frame=action.end_frame + start_frame,
                )
                for action in actions
            ]
            try:
                clips = cut_video_clips(source, source_actions, output / "actions")
            except FileNotFoundError as exc:
                logger.warning("Action clips skipped (ffmpeg missing): %s", exc)

    metadata: dict[str, Any] = {
        "fps": fps,
        "nb_frames_total": nb_frames_total,
        "source_start_frame": start_frame,
        "subsample": subsample,
        "max_interpolation_gap": max_gap,
        "homography_method": homography.method,
        "homography_nb_points": homography.nb_points,
        "home_cluster": home_cluster,
        "tracker": tracker,
        "nb_trackers_total": len(final_class),
        "nb_classified_players": len(final_team),
        "min_track_frames": min_track_frames,
        "nb_players_before_filter": nb_trackers_before_filter,
        "nb_players_stats": len(positions_by_tracker),
        "nb_detected_actions": len(actions),
    }
    if quality is not None:
        metadata.update({
            "nb_id_switches": quality.nb_id_switches,
            "nb_estimated_players": quality.nb_estimated_players,
            "mean_track_length": round(quality.mean_track_length, 1),
        })
    logger.info("Pipeline done. Metadata: %s", metadata)

    return PipelineResult(
        source_video=source,
        radar_video=radar_video,
        player_stats=stats_df,
        positions_by_tracker=positions_by_tracker,
        detected_actions=actions,
        action_clips=clips,
        metadata=metadata,
    )
