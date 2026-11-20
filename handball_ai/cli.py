"""handball-ai CLI. Usage: `handball-ai process --video X.mp4 --output folder/ --model W.pt`."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

from handball_ai.config import configure_logging, court_mapping_path
from handball_ai.detection import DEFAULT_CONFIDENCE, Detector
from handball_ai.homography import CourtPoint
from handball_ai.pipeline import PipelineResult, process_match

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Court mapping loading from JSON
# ---------------------------------------------------------------------------


def _load_court_mapping(path: Path) -> dict[str, CourtPoint]:
    """Load pixel <-> court points from a JSON file.

    Expected format:
        {
          "A": {"pixel": [x, y], "court_m": [X, Y]},
          "B": {"pixel": [...], "court_m": [...]},
          ...
        }
    At least 4 points are needed to calibrate a homography.

    Args:
        path: JSON file

    Returns:
        point name -> CourtPoint (with tuples for pixel and court_m)

    Raises:
        FileNotFoundError: missing file
        ValueError: invalid format (< 4 points or wrong structure)
        json.JSONDecodeError: malformed JSON
    """
    if not path.exists():
        raise FileNotFoundError(f"Court mapping file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: Any = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Court mapping: the root must be a JSON object, got {type(raw).__name__}")

    # Keys starting with "_" are comments or metadata, ignored.
    raw_points = {k: v for k, v in raw.items() if not k.startswith("_")}
    if len(raw_points) < 4:
        raise ValueError(f"Court mapping: at least 4 points required, got {len(raw_points)}")

    court_mapping: dict[str, CourtPoint] = {}
    for name, data in raw_points.items():
        if not isinstance(data, dict) or "pixel" not in data or "court_m" not in data:
            raise ValueError(f"Point '{name}': must be an object with 'pixel' and 'court_m'")
        pixel = data["pixel"]
        court = data["court_m"]
        if not (isinstance(pixel, list) and len(pixel) == 2):
            raise ValueError(f"Point '{name}': 'pixel' must be a list of 2 numbers")
        if not (isinstance(court, list) and len(court) == 2):
            raise ValueError(f"Point '{name}': 'court_m' must be a list of 2 numbers")
        court_mapping[name] = {
            "pixel": (int(pixel[0]), int(pixel[1])),
            "court_m": (float(court[0]), float(court[1])),
        }

    return court_mapping


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------


def _log_summary(
    result: PipelineResult, output_dir: Path, *, teams: bool, stats: bool
) -> None:
    """Log a summary of the produced artefacts."""
    meta = result.metadata
    logger.info("=" * 60)
    logger.info("Pipeline done. Artefacts in: %s", output_dir)
    logger.info("-" * 60)
    logger.info("Source video: %s", result.source_video)
    if stats:
        logger.info("Player stats: %s", output_dir / "player_stats.csv")
        logger.info("              %s", output_dir / "player_stats.parquet")
    logger.info("Total trackers: %d", meta.get("nb_trackers_total", 0))
    if teams:
        logger.info("Players classified in a team: %d", meta.get("nb_classified_players", 0))
    if stats:
        logger.info("--- Tracking quality (%s) ---", meta.get("tracker", "?"))
        logger.info("ID switches (suspected): %d", meta.get("nb_id_switches", 0))
        logger.info("Estimated players (after joining): %d", meta.get("nb_estimated_players", 0))
    logger.info("Usable players (after filter): %d", meta.get("nb_players_stats", 0))
    if result.radar_video is not None:
        logger.info("Radar video: %s", result.radar_video)
    logger.info("Detected actions: %d", len(result.detected_actions))
    logger.info("Action clips: %d", len(result.action_clips))
    for clip in result.action_clips:
        logger.info("  - %s", clip)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# `handball-ai process` sub-command
# ---------------------------------------------------------------------------


def process_command(args: argparse.Namespace) -> int:
    """`handball-ai process` command."""
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return 1

    mapping_path = Path(args.court_mapping) if args.court_mapping else court_mapping_path(video_path)
    try:
        court_mapping = _load_court_mapping(mapping_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Court mapping error: %s", exc)
        return 1

    try:
        detector = Detector(args.model, confidence=args.conf, exclude=args.exclude or ())
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Model error: %s", exc)
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting pipeline: video=%s, court mapping=%d points, subsample=%d",
        video_path,
        len(court_mapping),
        args.subsample,
    )

    try:
        result = process_match(
            video_path=video_path,
            court_mapping=court_mapping,
            output_dir=output_dir,
            detector=detector,
            subsample=args.subsample,
            with_radar_video=not args.no_radar_video,
            with_clips=not args.no_clips,
            min_track_frames=args.min_track_frames,
            tracker=args.tracker,
            duration=args.duration,
            start=args.start,
            with_teams=args.teams,
            with_stats=args.stats,
            correct_fisheye=not args.no_fisheye,
        )
    except Exception:
        logger.exception("Pipeline failed")
        return 1

    _log_summary(result, output_dir, teams=args.teams, stats=args.stats)
    return 0


# ---------------------------------------------------------------------------
# Parser and entry point
# ---------------------------------------------------------------------------


def _start_timestamp(value: str) -> float:
    if not re.fullmatch(r"[0-9]{2}:[0-5][0-9]:[0-5][0-9][:\.][0-9]{2}", value):
        raise argparse.ArgumentTypeError("use HH:MM:SS:cc (hours, minutes, seconds, hundredths)")
    hours, minutes, seconds = map(int, value[:8].split(":"))
    return hours * 3600 + minutes * 60 + seconds + int(value[9:]) / 100


def _positive_duration(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("the duration must be finite and strictly positive")
    return seconds


def _confidence_threshold(value: str) -> float:
    threshold = float(value)
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("the threshold must be between 0 and 1")
    return threshold


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (separate from main to ease testing)."""
    parser = argparse.ArgumentParser(
        prog="handball-ai",
        description="Handball scouting pipeline: detection, tracking, stats, action clips.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="Process a video and generate stats and clips.")
    p.add_argument("--video", required=True, help="MP4 file to process")
    p.add_argument("--output", required=True, help="Output folder")
    p.add_argument(
        "--start", type=_start_timestamp, default=0.0, metavar="HH:MM:SS:cc",
        help="Processing start, with hundredths of a second (default: 00:00:00:00).",
    )
    p.add_argument(
        "--duration", type=_positive_duration, metavar="SECONDS",
        help="Processing duration from the chosen start (default: until the end).",
    )
    p.add_argument(
        "--court-mapping",
        help="Court mapping JSON file (default: <video_name>.court_mapping.json next to the video)",
    )
    p.add_argument(
        "--model",
        type=Path,
        required=True,
        help="YOLO weights trained on handball (e.g. models/handball_yolo26m.pt)",
    )
    p.add_argument(
        "--exclude",
        action="append",
        metavar="CLASS",
        help="Exclude a model class by name. Repeatable.",
    )
    p.add_argument(
        "--conf",
        type=_confidence_threshold,
        default=DEFAULT_CONFIDENCE,
        metavar="THRESHOLD",
        help=f"YOLO confidence threshold between 0 and 1 (default {DEFAULT_CONFIDENCE}).",
    )
    p.add_argument("--subsample", type=int, default=2, help="1 frame out of N (default 2)")
    p.add_argument(
        "--min-track-frames",
        type=int,
        default=0,
        help="Drop the trackers seen on fewer than N frames (fragments). 0 = no filter.",
    )
    p.add_argument(
        "--tracker",
        choices=["bytetrack", "botsort"],
        default="bytetrack",
        help="Tracking backend: bytetrack (fast) or botsort (ReID, stable IDs on contact).",
    )
    p.add_argument(
        "--no-radar-video",
        action="store_true",
        help="Disable the video with the radar below the source.",
    )
    p.add_argument(
        "--teams", action="store_true",
        help="Enable team classification by jersey colour.",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Enable the computation and export of the statistics.",
    )
    p.add_argument(
        "--no-fisheye",
        action="store_true",
        help="Disable the radial correction estimated from points A to F.",
    )
    p.add_argument(
        "--no-clips",
        action="store_true",
        help="Do not cut the detected actions into clips.",
    )
    p.set_defaults(func=process_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    configure_logging()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
