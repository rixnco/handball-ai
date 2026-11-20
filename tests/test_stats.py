"""Unit tests for handball_ai.stats."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from handball_ai.stats import (
    PlayerPosition,
    compute_defensive_block_width,
    compute_player_stats,
    generate_player_heatmap,
)


def test_player_position_construction() -> None:
    """PlayerPosition builds correctly."""
    p = PlayerPosition(frame_idx=10, x_m=15.5, y_m=8.2)
    assert p.frame_idx == 10
    assert p.x_m == 15.5
    assert p.y_m == 8.2
    assert not p.interpolated


def test_distance_straight_path() -> None:
    """Straight path: 26 positions, 0.4 m/frame in x, total distance = 10 m."""
    positions = [PlayerPosition(frame_idx=i, x_m=i * 0.4, y_m=10.0) for i in range(26)]
    df = compute_player_stats({42: positions}, fps=25.0)

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["tracker_id"] == 42
    assert math.isclose(row["distance_m"], 10.0, abs_tol=1e-6)
    assert row["nb_detected_frames"] == 26
    assert math.isclose(row["x_min"], 0.0, abs_tol=1e-6)
    assert math.isclose(row["x_max"], 10.0, abs_tol=1e-6)


def test_mean_speed_constant_path() -> None:
    """Constant 5 m/s: 0.2 m/frame at 25 fps, mean speed == 5.0."""
    positions = [PlayerPosition(frame_idx=i, x_m=i * 0.2, y_m=5.0) for i in range(50)]
    row = compute_player_stats({1: positions}, fps=25.0).row(0, named=True)
    assert math.isclose(row["mean_speed_m_s"], 5.0, abs_tol=1e-3)
    assert math.isclose(row["max_speed_m_s"], 5.0, abs_tol=1e-3)


def test_stats_ignore_interpolated() -> None:
    """With ignore_interpolated=True, interpolated positions are excluded."""
    # 3 real positions: (0,0), (1,0), (2,0) -> distance = 2 m
    # 1 interpolated position at (10,0) that must NOT count.
    positions = [
        PlayerPosition(frame_idx=0, x_m=0.0, y_m=0.0, interpolated=False),
        PlayerPosition(frame_idx=1, x_m=1.0, y_m=0.0, interpolated=False),
        PlayerPosition(frame_idx=2, x_m=10.0, y_m=0.0, interpolated=True),
        PlayerPosition(frame_idx=3, x_m=2.0, y_m=0.0, interpolated=False),
    ]
    row = compute_player_stats({7: positions}, fps=25.0, ignore_interpolated=True).row(
        0, named=True
    )
    assert math.isclose(row["distance_m"], 2.0, abs_tol=1e-6)
    assert row["nb_detected_frames"] == 3
    assert math.isclose(row["x_max"], 2.0, abs_tol=1e-6)

    # Including interpolated positions, distance = 1 + 9 + 8 = 18 m
    row2 = compute_player_stats({7: positions}, fps=25.0, ignore_interpolated=False).row(
        0, named=True
    )
    assert math.isclose(row2["distance_m"], 18.0, abs_tol=1e-6)
    assert row2["nb_detected_frames"] == 4


def test_stats_several_trackers() -> None:
    """Several trackers: one row per tracker, columns in order."""
    positions = {
        1: [PlayerPosition(0, 0.0, 0.0), PlayerPosition(1, 3.0, 4.0)],  # 5 m
        2: [PlayerPosition(0, 0.0, 0.0), PlayerPosition(1, 0.0, 1.0)],  # 1 m
    }
    df = compute_player_stats(positions, fps=25.0)
    assert df.height == 2
    assert df.columns[0] == "tracker_id"
    by_tid = {row["tracker_id"]: row for row in df.iter_rows(named=True)}
    assert math.isclose(by_tid[1]["distance_m"], 5.0, abs_tol=1e-6)
    assert math.isclose(by_tid[2]["distance_m"], 1.0, abs_tol=1e-6)


def test_stats_single_position() -> None:
    """A single position: zero distance, bounds on that point."""
    row = compute_player_stats({3: [PlayerPosition(0, 4.0, 2.0)]}, fps=25.0).row(0, named=True)
    assert row["distance_m"] == 0.0
    assert row["centroid_x"] == 4.0
    assert row["nb_detected_frames"] == 1


def test_stats_only_interpolated_positions() -> None:
    """Only interpolated positions: zero row, no crash."""
    positions = [PlayerPosition(0, 4.0, 2.0, interpolated=True)]
    row = compute_player_stats({3: positions}, fps=25.0).row(0, named=True)
    assert row["tracker_id"] == 3
    assert row["nb_detected_frames"] == 0
    assert row["distance_m"] == 0.0


def test_empty_stats_returns_df_with_schema() -> None:
    """Empty dict: empty DataFrame with the right schema."""
    df = compute_player_stats({}, fps=25.0)
    assert df.height == 0
    assert "tracker_id" in df.columns
    assert "distance_m" in df.columns


def test_stats_invalid_fps_raises() -> None:
    """fps <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="fps"):
        compute_player_stats({1: [PlayerPosition(0, 0, 0)]}, fps=0.0)


def test_heatmap_dimensions_and_normalization() -> None:
    """Heatmap: right dimensions and sum normalized to 1."""
    positions = [PlayerPosition(frame_idx=i, x_m=float(i), y_m=10.0) for i in range(10)]
    heatmap = generate_player_heatmap(positions, resolution=1.0, smoothing_sigma=None)
    # 40 m / 1 m = 40 cells in x; 20 m / 1 m = 20 cells in y
    assert heatmap.shape == (20, 40)
    assert math.isclose(heatmap.sum(), 1.0, abs_tol=1e-9)


def test_heatmap_single_position_concentration() -> None:
    """A single position: all the density in its cell (no smoothing)."""
    positions = [PlayerPosition(frame_idx=0, x_m=20.0, y_m=10.0)]
    heatmap = generate_player_heatmap(positions, resolution=1.0, smoothing_sigma=None)
    assert math.isclose(heatmap[10, 20], 1.0, abs_tol=1e-9)
    assert math.isclose(heatmap.sum(), 1.0, abs_tol=1e-9)


def test_empty_heatmap_returns_zeros() -> None:
    """No position: heatmap of zeros (no NaN)."""
    heatmap = generate_player_heatmap([], resolution=1.0)
    assert heatmap.shape == (20, 40)
    assert np.all(heatmap == 0.0)


def test_heatmap_ignores_interpolated() -> None:
    """Interpolated positions do not contribute by default."""
    positions = [
        PlayerPosition(frame_idx=0, x_m=5.0, y_m=5.0, interpolated=False),
        PlayerPosition(frame_idx=1, x_m=30.0, y_m=15.0, interpolated=True),
    ]
    heatmap = generate_player_heatmap(positions, resolution=1.0, smoothing_sigma=None)
    assert math.isclose(heatmap[5, 5], 1.0, abs_tol=1e-9)
    assert math.isclose(heatmap[15, 30], 0.0, abs_tol=1e-9)


def test_defensive_block_width_4_aligned_defenders() -> None:
    """4 defenders at y=[5,8,11,14] on the same frame: width = 9 m, count = 4."""
    positions_by_tracker = {
        10: [PlayerPosition(0, 10.0, 5.0)],
        11: [PlayerPosition(0, 12.0, 8.0)],
        12: [PlayerPosition(0, 14.0, 11.0)],
        13: [PlayerPosition(0, 11.0, 14.0)],
        20: [PlayerPosition(0, 30.0, 10.0)],  # opposing attacker, ignored
    }
    final_team = {10: 0, 11: 0, 12: 0, 13: 0, 20: 1}
    df = compute_defensive_block_width(positions_by_tracker, final_team, defending_team=0, fps=25.0)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["nb_visible_defenders"] == 4
    assert math.isclose(row["width_y_m"], 9.0, abs_tol=1e-6)
    assert math.isclose(row["mean_x"], (10 + 12 + 14 + 11) / 4, abs_tol=1e-6)
    assert math.isclose(row["time_s"], 0.0, abs_tol=1e-9)


def test_defensive_block_width_over_time() -> None:
    """Several frames: sorted by frame_idx with a consistent time_s."""
    positions_by_tracker = {
        1: [PlayerPosition(0, 20.0, 5.0), PlayerPosition(25, 20.0, 6.0)],
        2: [PlayerPosition(0, 20.0, 15.0), PlayerPosition(25, 20.0, 14.0)],
    }
    df = compute_defensive_block_width(
        positions_by_tracker, {1: 0, 2: 0}, defending_team=0, fps=25.0
    )
    assert df.height == 2
    assert df["frame_idx"].to_list() == [0, 25]
    assert math.isclose(df["time_s"].to_list()[1], 1.0, abs_tol=1e-9)
    # frame 0: width = 15-5 = 10; frame 25: width = 14-6 = 8
    assert math.isclose(df["width_y_m"].to_list()[0], 10.0, abs_tol=1e-6)
    assert math.isclose(df["width_y_m"].to_list()[1], 8.0, abs_tol=1e-6)


def test_max_speed_outlier_capped_by_default() -> None:
    """A tracker jump (~20 m in 1 frame) is > 100 m/s: capped at 12 m/s by default."""
    positions = [
        PlayerPosition(frame_idx=0, x_m=0.0, y_m=0.0),
        PlayerPosition(frame_idx=1, x_m=0.5, y_m=0.0),   # ~12.5 m/s, just above the cap
        PlayerPosition(frame_idx=2, x_m=20.0, y_m=0.0),  # tracker jump: 487.5 m/s
        PlayerPosition(frame_idx=3, x_m=20.4, y_m=0.0),  # 10 m/s
    ]
    row = compute_player_stats({1: positions}, fps=25.0).row(0, named=True)
    # The 487 m/s jump is excluded, the kept max is 10 m/s
    assert row["max_speed_m_s"] <= 12.0
    assert row["max_speed_m_s"] == pytest.approx(10.0, abs=1e-3)


def test_max_speed_outlier_without_cap() -> None:
    """With max_speed_m_s=None, outliers go through."""
    positions = [
        PlayerPosition(frame_idx=0, x_m=0.0, y_m=0.0),
        PlayerPosition(frame_idx=1, x_m=100.0, y_m=0.0),  # 2500 m/s
    ]
    row = compute_player_stats({1: positions}, fps=25.0, max_speed_m_s=None).row(0, named=True)
    assert row["max_speed_m_s"] > 1000.0


def test_empty_defensive_block_width() -> None:
    """No defender in the requested team: empty DataFrame with the right schema."""
    df = compute_defensive_block_width(
        {1: [PlayerPosition(0, 0, 0)]}, {1: 0}, defending_team=1, fps=25.0
    )
    assert df.height == 0
    assert isinstance(df, pl.DataFrame)
    assert "width_y_m" in df.columns
