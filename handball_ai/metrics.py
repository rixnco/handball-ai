"""Tracking quality metrics: ID switches and estimated number of players.

Judges a tracker (ByteTrack vs BoT-SORT) objectively without watching the
video. Idea: a poorly tracked player shows up as several short tracks. When a
track ends and a new one starts right after, very close by, it is most likely
the same player -> an "ID switch". Joining these chains gives an estimate of
the real number of players.

Purely geometric (court positions in metres), testable without GPU.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from handball_ai.stats import PlayerPosition

logger = logging.getLogger(__name__)


@dataclass
class TrackingQualitySummary:
    """Figures summarizing the quality of a tracking."""

    nb_tracks: int
    nb_id_switches: int
    nb_estimated_players: int
    mean_track_length: float


@dataclass
class _TrackBounds:
    """First and last real position of a track."""

    start_frame: int
    start_x: float
    start_y: float
    end_frame: int
    end_x: float
    end_y: float
    nb_real: int


def _track_bounds(positions: list[PlayerPosition]) -> _TrackBounds | None:
    """Extract the time and space bounds of a track (real positions only)."""
    real = sorted((p for p in positions if not p.interpolated), key=lambda p: p.frame_idx)
    if not real:
        return None
    first, last = real[0], real[-1]
    return _TrackBounds(
        start_frame=first.frame_idx,
        start_x=first.x_m,
        start_y=first.y_m,
        end_frame=last.frame_idx,
        end_x=last.x_m,
        end_y=last.y_m,
        nb_real=len(real),
    )


def detect_id_switches(
    positions_by_tracker: dict[int, list[PlayerPosition]],
    max_distance_m: float = 3.0,
    max_frames: int = 30,
) -> list[tuple[int, int]]:
    """Detect ID switches (track A ends -> track B starts very close by).

    A switch (A, B) is counted when B starts after the end of A, within
    max_frames, at most max_distance_m away from the last point of A.
    Each track has at most one predecessor and one successor (simple chains):
    each track end is matched with the closest switch not already taken.

    Args:
        positions_by_tracker: real positions per tracker_id (before filtering)
        max_distance_m: max distance (m) between the end of A and the start of B
        max_frames: max delay (source frames) between the end of A and the start of B

    Returns:
        list of (tracker_a, tracker_b) pairs = detected switches.
    """
    bounds = {
        tid: b
        for tid, pos in positions_by_tracker.items()
        if (b := _track_bounds(pos)) is not None
    }

    # Process the track ends in chronological order.
    ends = sorted(bounds.items(), key=lambda kv: kv[1].end_frame)
    taken_starts: set[int] = set()
    switches: list[tuple[int, int]] = []

    for tid_a, ba in ends:
        best: int | None = None
        best_dist: float | None = None
        for tid_b, bb in bounds.items():
            if tid_b == tid_a or tid_b in taken_starts:
                continue
            dt = bb.start_frame - ba.end_frame
            if dt <= 0 or dt > max_frames:
                continue
            dist = math.hypot(bb.start_x - ba.end_x, bb.start_y - ba.end_y)
            if dist > max_distance_m:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = tid_b
        if best is not None:
            switches.append((tid_a, best))
            taken_starts.add(best)

    return switches


def estimate_nb_players(
    positions_by_tracker: dict[int, list[PlayerPosition]],
    switches: list[tuple[int, int]] | None = None,
    max_distance_m: float = 3.0,
    max_frames: int = 30,
) -> int:
    """Estimate the real number of players by joining switch chains.

    Union-find: each switch (A, B) merges A and B into the same player.
    The number of connected components = estimated number of players.

    Args:
        positions_by_tracker: real positions per tracker_id
        switches: switches already computed (recomputed otherwise)
        max_distance_m: passed to detect_id_switches when switches is None
        max_frames: passed to detect_id_switches when switches is None

    Returns:
        estimated number of players (connected components).
    """
    tids = list(positions_by_tracker)
    parent = {t: t for t in tids}

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb

    if switches is None:
        switches = detect_id_switches(positions_by_tracker, max_distance_m, max_frames)
    for a, b in switches:
        if a in parent and b in parent:
            union(a, b)

    return len({root(t) for t in tids})


def summarize_tracking_quality(
    positions_by_tracker: dict[int, list[PlayerPosition]],
    max_distance_m: float = 3.0,
    max_frames: int = 30,
) -> TrackingQualitySummary:
    """Gather the tracking quality KPIs.

    Args:
        positions_by_tracker: real positions per tracker_id (before filtering)
        max_distance_m: max distance for an ID switch
        max_frames: max delay (source frames) for an ID switch

    Returns:
        TrackingQualitySummary (nb_tracks, nb_id_switches, nb_estimated_players,
        mean_track_length).
    """
    switches = detect_id_switches(positions_by_tracker, max_distance_m, max_frames)
    lengths = [
        sum(1 for p in pos if not p.interpolated) for pos in positions_by_tracker.values()
    ]
    summary = TrackingQualitySummary(
        nb_tracks=len(positions_by_tracker),
        nb_id_switches=len(switches),
        nb_estimated_players=estimate_nb_players(positions_by_tracker, switches=switches),
        mean_track_length=sum(lengths) / len(lengths) if lengths else 0.0,
    )
    logger.info(
        "Tracking quality: %d tracks, %d ID switches, ~%d estimated players "
        "(mean length %.1f frames)",
        summary.nb_tracks,
        summary.nb_id_switches,
        summary.nb_estimated_players,
        summary.mean_track_length,
    )
    return summary
