"""Unit tests for handball_ai.homography."""

from __future__ import annotations

import numpy as np
import pytest

from handball_ai.homography import calibrate_homography


def test_calibrate_homography_4_points_grid() -> None:
    """With 4 points on a regular grid, the projection is correct."""
    court_mapping = {
        "A": {"pixel": (100, 100), "court_m": (0.0, 0.0)},
        "B": {"pixel": (1000, 100), "court_m": (40.0, 0.0)},
        "C": {"pixel": (1000, 600), "court_m": (40.0, 20.0)},
        "D": {"pixel": (100, 600), "court_m": (0.0, 20.0)},
    }
    H = calibrate_homography(court_mapping)
    assert H.nb_points == 4
    assert H.method == "perspective_4pts"

    # The centre of the pixel rectangle projects onto the centre of the court
    x_m, y_m = H.project_point((550.0, 350.0))
    assert abs(x_m - 20.0) < 0.5
    assert abs(y_m - 10.0) < 0.5


def test_calibrate_homography_fewer_than_4_points_raises() -> None:
    """Fewer than 4 points raises ValueError."""
    court_mapping = {
        "A": {"pixel": (0, 0), "court_m": (0.0, 0.0)},
        "B": {"pixel": (100, 0), "court_m": (1.0, 0.0)},
        "C": {"pixel": (100, 100), "court_m": (1.0, 1.0)},
    }
    with pytest.raises(ValueError, match="at least 4 points"):
        calibrate_homography(court_mapping)


def test_calibrate_homography_7_points_ransac() -> None:
    """With more than 4 points, RANSAC is used."""
    court_mapping = {
        f"P{i}": {"pixel": (100 + i * 50, 100 + i * 30), "court_m": (float(i * 2), float(i))}
        for i in range(7)
    }
    H = calibrate_homography(court_mapping)
    assert H.nb_points == 7
    assert H.method in ("ransac", "least_squares")


def test_project_bbox_feet() -> None:
    """The bbox projection uses the bottom centre."""
    court_mapping = {
        "A": {"pixel": (0, 0), "court_m": (0.0, 0.0)},
        "B": {"pixel": (1000, 0), "court_m": (40.0, 0.0)},
        "C": {"pixel": (1000, 500), "court_m": (40.0, 20.0)},
        "D": {"pixel": (0, 500), "court_m": (0.0, 20.0)},
    }
    H = calibrate_homography(court_mapping)
    bbox = np.array([100.0, 100.0, 200.0, 400.0])
    x_m, y_m = H.project_bbox_feet(bbox)
    # The bottom centre is (150, 400)
    assert 5.0 < x_m < 7.0
    assert 15.0 < y_m < 17.0


def test_fisheye_correction_straightens_side_lines() -> None:
    """With A to F, the correction maps every point to its court position."""
    pixels = {
        "A": (100, 100), "B": (900, 100), "C": (900, 500), "D": (100, 500),
        "E": (500, 80), "F": (500, 520),
    }
    court = {
        "A": (0.0, 0.0), "B": (40.0, 0.0), "C": (40.0, 20.0), "D": (0.0, 20.0),
        "E": (20.0, 0.0), "F": (20.0, 20.0),
    }
    court_mapping = {
        name: {"pixel": pixel, "court_m": court[name]} for name, pixel in pixels.items()
    }

    homography = calibrate_homography(court_mapping, correct_fisheye=True)
    assert homography.method == "fisheye_4pts"
    for name, pixel in pixels.items():
        assert homography.project_point(pixel) == pytest.approx(court[name], abs=0.01)

    assert calibrate_homography(court_mapping).radial_correction is None
