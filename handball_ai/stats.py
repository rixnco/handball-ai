"""Per-player statistics: distance covered, speed, heatmap, area of play.

WARNING: never compute statistics on interpolated positions, only on frames
actually detected. The `interpolated` flag of the positions allows filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


@dataclass
class PlayerPosition:
    """One player position at a given frame."""

    frame_idx: int
    x_m: float
    y_m: float
    interpolated: bool = False


# Columns of compute_player_stats, in guaranteed order.
_STATS_SCHEMA: dict[str, type[pl.DataType]] = {
    "tracker_id": pl.Int64,
    "distance_m": pl.Float64,
    "mean_speed_m_s": pl.Float64,
    "max_speed_m_s": pl.Float64,
    "total_time_s": pl.Float64,
    "x_min": pl.Float64,
    "x_max": pl.Float64,
    "y_min": pl.Float64,
    "y_max": pl.Float64,
    "centroid_x": pl.Float64,
    "centroid_y": pl.Float64,
    "nb_detected_frames": pl.Int64,
}

_BLOCK_SCHEMA: dict[str, type[pl.DataType]] = {
    "frame_idx": pl.Int64,
    "time_s": pl.Float64,
    "width_y_m": pl.Float64,
    "mean_x": pl.Float64,
    "nb_visible_defenders": pl.Int64,
}


# Realistic top speed for a sprinting human (~43 km/h, 100 m world record).
# Above that: tracker artefact (ID jumping onto another player).
MAX_REALISTIC_SPEED_M_S: float = 12.0


def _single_tracker_stats(
    tracker_id: int,
    positions: list[PlayerPosition],
    fps: float,
    ignore_interpolated: bool,
    max_speed_m_s: float | None,
) -> dict[str, float | int]:
    """Compute the aggregated statistics of a single tracker."""
    if ignore_interpolated:
        positions = [p for p in positions if not p.interpolated]

    # Sort by frame_idx (positions may arrive out of order).
    positions = sorted(positions, key=lambda p: p.frame_idx)
    nb = len(positions)

    if nb == 0:
        empty = dict.fromkeys(_STATS_SCHEMA, 0.0)
        return empty | {"tracker_id": tracker_id, "nb_detected_frames": 0}

    xs = np.array([p.x_m for p in positions], dtype=np.float64)
    ys = np.array([p.y_m for p in positions], dtype=np.float64)
    idxs = np.array([p.frame_idx for p in positions], dtype=np.float64)

    stats: dict[str, float | int] = {
        "tracker_id": tracker_id,
        "distance_m": 0.0,
        "mean_speed_m_s": 0.0,
        "max_speed_m_s": 0.0,
        "total_time_s": 0.0,
        "x_min": float(xs.min()),
        "x_max": float(xs.max()),
        "y_min": float(ys.min()),
        "y_max": float(ys.max()),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "nb_detected_frames": nb,
    }
    if nb == 1:
        # No distance with a single point.
        return stats

    # Euclidean distances between consecutive points.
    distances = np.hypot(np.diff(xs), np.diff(ys))
    total_distance = float(distances.sum())

    # Time between consecutive points (handles subsample gaps).
    dt = np.diff(idxs) / fps
    # dt is in theory > 0 since frames are sorted and unique per tracker.
    safe_dt = np.where(dt > 0, dt, 1.0 / fps)
    speeds = distances / safe_dt

    total_time_s = float((idxs[-1] - idxs[0]) / fps)

    # Cap the max speed to drop tracker outliers (ID jumping onto another
    # player between two sampled frames). The 100 m human record is
    # ~12.4 m/s; anything above is an artefact.
    if max_speed_m_s is not None:
        plausible = speeds[speeds <= max_speed_m_s]
        max_speed = float(plausible.max()) if len(plausible) else float(max_speed_m_s)
    else:
        max_speed = float(speeds.max())

    stats.update(
        distance_m=total_distance,
        mean_speed_m_s=total_distance / total_time_s if total_time_s > 0 else 0.0,
        max_speed_m_s=max_speed,
        total_time_s=total_time_s,
    )
    return stats


def compute_player_stats(
    positions_by_tracker: dict[int, list[PlayerPosition]],
    fps: float,
    ignore_interpolated: bool = True,
    max_speed_m_s: float | None = MAX_REALISTIC_SPEED_M_S,
) -> pl.DataFrame:
    """Compute the aggregated statistics per tracker_id.

    Args:
        positions_by_tracker: tracker_id -> list of PlayerPosition
        fps: clip frame rate
        ignore_interpolated: exclude interpolated positions from the computation
        max_speed_m_s: realistic top speed used to drop tracker outliers in
            max_speed_m_s. None disables the cap.

    Returns:
        Polars DataFrame with columns:
        tracker_id, distance_m, mean_speed_m_s, max_speed_m_s, total_time_s,
        x_min, x_max, y_min, y_max, centroid_x, centroid_y, nb_detected_frames.

    Raises:
        ValueError: fps <= 0
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    if not positions_by_tracker:
        return pl.DataFrame(schema=_STATS_SCHEMA)

    rows = [
        _single_tracker_stats(tid, positions, fps, ignore_interpolated, max_speed_m_s)
        for tid, positions in positions_by_tracker.items()
    ]
    df = pl.DataFrame(rows).select(list(_STATS_SCHEMA))
    logger.info("Statistics computed for %d trackers", df.height)
    return df


def generate_player_heatmap(
    positions: list[PlayerPosition],
    court_length: float = 40.0,
    court_width: float = 20.0,
    resolution: float = 0.5,
    smoothing_sigma: float | None = 1.0,
    ignore_interpolated: bool = True,
) -> np.ndarray:
    """Generate a 2D presence heatmap of a player on the court.

    Args:
        positions: player positions
        court_length: length in metres (default 40)
        court_width: width in metres (default 20)
        resolution: cell size in metres (default 0.5 m)
        smoothing_sigma: Gaussian filter sigma (None = no smoothing)
        ignore_interpolated: exclude interpolated positions

    Returns:
        2D array of shape (n_y, n_x) with n_y = width/res, n_x = length/res.
        Normalized density (sum = 1), zeros when there is no valid position.

    Raises:
        ValueError: invalid dimensions or resolution
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")
    if court_length <= 0 or court_width <= 0:
        raise ValueError("court dimensions must be > 0")

    n_x = int(round(court_length / resolution))
    n_y = int(round(court_width / resolution))
    grid = np.zeros((n_y, n_x), dtype=np.float64)

    if ignore_interpolated:
        positions = [p for p in positions if not p.interpolated]

    if not positions:
        return grid

    # Indexing: axis 0 = y (width), axis 1 = x (length).
    for p in positions:
        # Clip to handle off-court positions (projection noise).
        ix = int(np.clip(p.x_m / resolution, 0, n_x - 1))
        iy = int(np.clip(p.y_m / resolution, 0, n_y - 1))
        grid[iy, ix] += 1.0

    if smoothing_sigma is not None and smoothing_sigma > 0:
        grid = gaussian_filter(grid, sigma=smoothing_sigma, mode="constant")

    total = grid.sum()
    if total > 0:
        grid /= total

    return grid


def compute_defensive_block_width(
    positions_by_tracker: dict[int, list[PlayerPosition]],
    final_team: dict[int, int],
    defending_team: int,
    fps: float,
    ignore_interpolated: bool = True,
) -> pl.DataFrame:
    """Compute the width of the defensive block over time.

    For each frame with at least one visible defender, computes the lateral
    width (y_max - y_min) of the block and its mean longitudinal position.

    Args:
        positions_by_tracker: positions per tracker_id
        final_team: tracker_id -> team
        defending_team: index of the defending team
        fps: frame rate
        ignore_interpolated: exclude interpolated positions

    Returns:
        Polars DataFrame sorted by frame_idx with columns:
        frame_idx, time_s, width_y_m, mean_x, nb_visible_defenders.

    Raises:
        ValueError: fps <= 0
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    # Frame -> list of (x_m, y_m) of the visible defenders.
    by_frame: dict[int, list[tuple[float, float]]] = {}
    for tid, positions in positions_by_tracker.items():
        if final_team.get(tid) != defending_team:
            continue
        for p in positions:
            if ignore_interpolated and p.interpolated:
                continue
            by_frame.setdefault(p.frame_idx, []).append((p.x_m, p.y_m))

    if not by_frame:
        return pl.DataFrame(schema=_BLOCK_SCHEMA)

    rows = []
    for fi in sorted(by_frame):
        pts = by_frame[fi]
        xs = np.array([x for x, _ in pts], dtype=np.float64)
        ys = np.array([y for _, y in pts], dtype=np.float64)
        rows.append(
            {
                "frame_idx": int(fi),
                "time_s": float(fi / fps),
                "width_y_m": float(ys.max() - ys.min()) if len(ys) > 1 else 0.0,
                "mean_x": float(xs.mean()),
                "nb_visible_defenders": len(pts),
            }
        )

    df = pl.DataFrame(rows).select(list(_BLOCK_SCHEMA))
    logger.info("Defensive block width: %d frames computed", df.height)
    return df
