"""Tracking quality metrics tests (ID switches, estimated number of players)."""

from __future__ import annotations

from handball_ai.metrics import (
    detect_id_switches,
    estimate_nb_players,
    summarize_tracking_quality,
)
from handball_ai.stats import PlayerPosition


def _track(frames: range, x: float, y: float, dx: float = 0.0) -> list[PlayerPosition]:
    """A track: real positions on the given frames, around (x, y)."""
    return [PlayerPosition(frame_idx=fi, x_m=x + i * dx, y_m=y) for i, fi in enumerate(frames)]


def test_simple_switch_detected() -> None:
    """Track A ends at (20,10) f=10; B starts at (20.5,10) f=14 -> 1 switch."""
    positions = {
        1: _track(range(0, 11, 2), 20.0, 10.0),
        2: _track(range(14, 30, 2), 20.5, 10.0),
    }
    assert detect_id_switches(positions, max_distance_m=3.0, max_frames=30) == [(1, 2)]


def test_distant_tracks_no_switch() -> None:
    """Two tracks far apart in space: no switch."""
    positions = {
        1: _track(range(0, 11, 2), 5.0, 5.0),
        2: _track(range(14, 30, 2), 35.0, 15.0),
    }
    assert detect_id_switches(positions, max_distance_m=3.0, max_frames=30) == []


def test_gap_too_long_no_switch() -> None:
    """Close in space but the delay is too long: no switch."""
    positions = {
        1: _track(range(0, 11, 2), 20.0, 10.0),     # ends at frame 10
        2: _track(range(100, 120, 2), 20.5, 10.0),  # starts at frame 100 (gap 90)
    }
    assert detect_id_switches(positions, max_distance_m=3.0, max_frames=30) == []


def test_chain_of_three_is_one_player() -> None:
    """A -> B -> C joined: 2 switches, 1 estimated player over 3 tracks."""
    positions = {
        1: _track(range(0, 11, 2), 20.0, 10.0),
        2: _track(range(14, 25, 2), 20.5, 10.0),
        3: _track(range(28, 40, 2), 21.0, 10.0),
    }
    switches = detect_id_switches(positions, max_distance_m=3.0, max_frames=30)
    assert len(switches) == 2
    assert estimate_nb_players(positions, switches=switches) == 1


def test_two_distinct_players() -> None:
    """Two players that never join: 2 estimated players."""
    positions = {
        1: _track(range(0, 40, 2), 10.0, 5.0),
        2: _track(range(0, 40, 2), 30.0, 15.0),
    }
    assert estimate_nb_players(positions) == 2


def test_full_quality_summary() -> None:
    """The summary aggregates tracks, switches, estimated players and mean length."""
    positions = {
        1: _track(range(0, 11, 2), 20.0, 10.0),    # 6 real positions
        2: _track(range(14, 25, 2), 20.5, 10.0),   # 6 positions, continues track 1
        3: _track(range(0, 40, 2), 5.0, 5.0),      # distinct player
    }
    summary = summarize_tracking_quality(positions)
    assert summary.nb_tracks == 3
    assert summary.nb_id_switches == 1  # 1 -> 2
    assert summary.nb_estimated_players == 2  # {1,2} joined + {3}
    assert summary.mean_track_length > 0


def test_summary_ignores_interpolated_positions() -> None:
    """The track length only counts real positions."""
    positions = {
        1: [
            PlayerPosition(0, 10.0, 5.0, interpolated=False),
            PlayerPosition(1, 10.5, 5.0, interpolated=True),
            PlayerPosition(2, 11.0, 5.0, interpolated=False),
        ],
    }
    assert summarize_tracking_quality(positions).mean_track_length == 2.0


def test_empty_dict() -> None:
    """No track: zero summary, no crash."""
    summary = summarize_tracking_quality({})
    assert summary.nb_tracks == 0
    assert summary.nb_id_switches == 0
    assert summary.nb_estimated_players == 0
    assert summary.mean_track_length == 0.0
