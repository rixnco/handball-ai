"""Homography: projection of pixel coordinates onto the court, in metres.

A homography is a 3x3 projective transform mapping the image plane onto the
court plane. It needs at least 4 points (pixel, court_m).
With 4 points: `getPerspectiveTransform`. With more: `findHomography` + RANSAC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict

import cv2
import numpy as np

from handball_ai.config import COURT_POINTS_M

logger = logging.getLogger(__name__)


class CourtPoint(TypedDict):
    """One pixel <-> court point used to calibrate the homography."""

    pixel: tuple[int, int]
    court_m: tuple[float, float]


@dataclass
class RadialCorrection:
    """Single-coefficient correction around the centre of the four corners."""

    center: np.ndarray
    radius: float
    coefficient: float

    def correct(self, pixels: np.ndarray) -> np.ndarray:
        offsets = pixels - self.center
        squared_radius = np.sum((offsets / self.radius) ** 2, axis=-1, keepdims=True)
        # Beyond the calibrated radius, extend with a constant factor.
        factor = 1 + self.coefficient * np.minimum(squared_radius, 1)
        return self.center + offsets * factor


@dataclass
class Homography:
    """Calibrated homography: 3x3 matrix and projection methods."""

    matrix: np.ndarray
    nb_points: int
    method: str
    radial_correction: RadialCorrection | None = None

    def project_point(self, pixel_point: tuple[float, float]) -> tuple[float, float]:
        """Project a pixel point onto the court (metres)."""
        pt = np.array([[pixel_point]], dtype=np.float32)
        if self.radial_correction is not None:
            pt = self.radial_correction.correct(pt).astype(np.float32)
        proj = cv2.perspectiveTransform(pt, self.matrix)
        return float(proj[0][0][0]), float(proj[0][0][1])

    def project_bbox_feet(self, bbox: np.ndarray) -> tuple[float, float]:
        """Project the bottom centre of a bbox (feet position) onto the court."""
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_bottom = float(bbox[3])
        return self.project_point((x_center, y_bottom))


def _calibrate_with_fisheye(court_mapping: dict[str, CourtPoint]) -> Homography:
    from scipy.optimize import minimize_scalar

    pixels = np.array([court_mapping[name]["pixel"] for name in "ABCDEF"], dtype=np.float64)
    corners = pixels[:4]
    center = corners.mean(axis=0)
    radius = float(np.linalg.norm(corners - center, axis=1).max())
    if not np.isfinite(pixels).all() or not cv2.isContourConvex(corners.astype(np.float32)):
        raise ValueError("Corners A, B, C and D must form a convex quadrilateral.")

    def alignment_error(coefficient: float) -> float:
        a, b, c, d, e, f = RadialCorrection(center, radius, coefficient).correct(pixels)
        # Squared perpendicular distance from E to AB and from F to CD, in pixels.
        error = 0.0
        for start, end, middle in ((a, b, e), (c, d, f)):
            dx, dy = end - start
            mx, my = middle - start
            error += (dx * my - dy * mx) ** 2 / (dx * dx + dy * dy)
        return float(error)

    # The lower bound keeps the correction increasing with the radius.
    result = minimize_scalar(alignment_error, bounds=(-0.3, 0.5), method="bounded")
    initial_error = alignment_error(0)
    coefficient = float(result.x) if result.success and result.fun < initial_error else 0.0
    correction = RadialCorrection(center, radius, coefficient)
    corrected_corners = correction.correct(corners).astype(np.float32)
    rectangle = np.array([COURT_POINTS_M[name] for name in "ABCD"], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corrected_corners, rectangle)
    if (
        not cv2.isContourConvex(corrected_corners)
        or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) < 3
    ):
        raise ValueError("Degenerate fisheye correction. Check points A to F.")
    logger.info(
        "Fisheye correction: k=%.4f, RMS error on E and F from %.1f to %.1f px",
        coefficient, (initial_error / 2) ** 0.5, (alignment_error(coefficient) / 2) ** 0.5,
    )
    return Homography(matrix, len(pixels), "fisheye_4pts", correction)


def calibrate_homography(
    court_mapping: dict[str, CourtPoint], *, correct_fisheye: bool = False
) -> Homography:
    """Calibrate a homography from pixel <-> court points.

    Args:
        court_mapping: point name -> {"pixel": (x, y), "court_m": (X, Y)}
        correct_fisheye: estimate the radial correction when points A to F are present

    Returns:
        Calibrated homography

    Raises:
        ValueError: fewer than 4 points or calibration failure
    """
    if len(court_mapping) < 4:
        raise ValueError(f"Homography: at least 4 points required, got {len(court_mapping)}")

    if correct_fisheye:
        missing = sorted(set("ABCDEF") - court_mapping.keys())
        if not missing:
            return _calibrate_with_fisheye(court_mapping)
        logger.info("Fisheye correction skipped, missing points: %s", ", ".join(missing))

    pixel_pts = np.array([p["pixel"] for p in court_mapping.values()], dtype=np.float32)
    court_pts = np.array([p["court_m"] for p in court_mapping.values()], dtype=np.float32)

    if len(court_mapping) == 4:
        H = cv2.getPerspectiveTransform(pixel_pts, court_pts)
        if H is None:
            raise ValueError("getPerspectiveTransform failed, collinear points?")
        return Homography(matrix=H, nb_points=4, method="perspective_4pts")

    # More than 4 points: RANSAC first, least-squares fallback
    H, _ = cv2.findHomography(pixel_pts, court_pts, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if H is not None:
        return Homography(matrix=H, nb_points=len(court_mapping), method="ransac")

    logger.warning("Degenerate RANSAC, least-squares fallback")
    H, _ = cv2.findHomography(pixel_pts, court_pts, method=0)
    if H is None:
        raise ValueError("Homography calibration failed even with the fallback")

    return Homography(matrix=H, nb_points=len(court_mapping), method="least_squares")
