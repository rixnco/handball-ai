"""Unit tests for handball_ai.config."""

from __future__ import annotations

from pathlib import Path

from handball_ai.config import (
    CLASS_NAMES,
    COURT_POINTS_M,
    HANDBALL_CLASSES,
    CourtConfig,
    court_mapping_path,
)


def test_class_names_consistent() -> None:
    """The class inversion is a bijection."""
    for name, idx in HANDBALL_CLASSES.items():
        assert CLASS_NAMES[idx] == name


def test_court_pixel_dimensions() -> None:
    """Pixel dimensions are consistent with px_per_m."""
    config = CourtConfig()
    assert config.court_px_w == 40 * 22
    assert config.court_px_h == 20 * 22


def test_court_m2px_corners() -> None:
    """The corners map to the right pixels."""
    config = CourtConfig()
    x0, y0 = config.m2px(0, 0)
    x1, y1 = config.m2px(40, 20)
    assert x0 == config.origin_x
    assert y0 == config.origin_y + config.court_px_h
    assert x1 == config.origin_x + config.court_px_w
    assert y1 == config.origin_y


def test_court_is_in_zone() -> None:
    """The zone filter follows the asymmetric margins."""
    config = CourtConfig()
    assert config.is_in_zone((20.0, 10.0))  # centre
    assert config.is_in_zone((0.0, 5.0))    # X edge, inside the Y zone
    assert not config.is_in_zone((20.0, 1.0))   # too low
    assert not config.is_in_zone((20.0, 19.0))  # too high


def test_court_tolerates_wingers_near_the_four_corners() -> None:
    """Wingers close to the corners stay in the zone, within the tolerance."""
    config = CourtConfig()
    for x in (1.0, 39.0):
        for y in (0.5, 19.5):
            assert config.is_in_zone((x, y))

    assert config.is_in_zone((-0.5, -0.5))
    assert config.is_in_zone((40.5, 20.5))
    assert not config.is_in_zone((-0.6, 0.5))
    assert not config.is_in_zone((0.5, 20.6))
    assert not config.is_in_zone((5.0, 0.5))


def test_court_points_inside_court() -> None:
    """Reference points lie inside the 40x20 rectangle."""
    for name, (x, y) in COURT_POINTS_M.items():
        assert 0 <= x <= 40, f"Point {name} off court in X: {x}"
        assert 0 <= y <= 20, f"Point {name} off court in Y: {y}"


def test_court_mapping_path_next_to_video() -> None:
    """The court mapping sits next to its video."""
    assert court_mapping_path(Path("clips/match.mp4")) == Path("clips/match.mp4.court_mapping.json")
