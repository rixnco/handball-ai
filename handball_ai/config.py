"""Central configuration: constants, court parameters and reference points."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# Handball detection classes
# ----------------------------------------------------------------------------

HANDBALL_CLASSES: dict[str, int] = {
    "players": 0,
    "goalkeeper": 1,
    "referees": 2,
    "ball": 3,
}

CLASS_NAMES: dict[int, str] = {v: k for k, v in HANDBALL_CLASSES.items()}


# ----------------------------------------------------------------------------
# Handball court: 40 m x 20 m
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class CourtConfig:
    """Court geometry and radar rendering configuration."""

    # Real dimensions (metres)
    length_m: float = 40.0
    width_m: float = 20.0

    # Radar rendering (pixels)
    radar_w: int = 1920
    radar_h: int = 540
    broadcast_w: int = 1920
    broadcast_h: int = 540
    px_per_m: int = 22

    # Asymmetric filter (drops benches and staff on the sides)
    margin_x_m: float = 0.0
    margin_y_m: float = 2.0
    corner_depth_m: float = 4.0
    corner_tolerance_m: float = 0.5

    @property
    def court_px_w(self) -> int:
        return round(self.length_m * self.px_per_m)

    @property
    def court_px_h(self) -> int:
        return int(self.width_m * self.px_per_m)

    @property
    def origin_x(self) -> int:
        return (self.radar_w - self.court_px_w) // 2

    @property
    def origin_y(self) -> int:
        return (self.radar_h - self.court_px_h) // 2

    def m2px(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Convert court coordinates (metres) to radar pixels."""
        x_px = self.origin_x + round(x_m * self.px_per_m)
        y_px = self.origin_y + int(round((self.width_m - y_m) * self.px_per_m))
        return x_px, y_px

    def is_in_zone(self, point_m: tuple[float, float]) -> bool:
        """Check that a point lies in the usable area of the court."""
        x, y = point_m
        in_zone = (
            self.margin_x_m <= x <= self.length_m - self.margin_x_m
            and self.margin_y_m <= y <= self.width_m - self.margin_y_m
        )
        # Lift the inner margin near the corners and absorb small projection
        # errors of the foot point beyond the lines.
        near_corner = (
            -self.corner_tolerance_m <= min(x, self.length_m - x) <= self.corner_depth_m
            and -self.corner_tolerance_m <= min(y, self.width_m - y) <= self.margin_y_m
        )
        return in_zone or near_corner


# ----------------------------------------------------------------------------
# Court reference points (for the homography)
# ----------------------------------------------------------------------------

def court_mapping_path(video: str | Path) -> Path:
    """Return the path of the court mapping stored next to its video."""
    path = Path(video)
    return path.with_name(f"{path.name}.court_mapping.json")


COURT_POINTS_M: dict[str, tuple[float, float]] = {
    # Corners
    "A": (0.0, 0.0),
    "B": (40.0, 0.0),
    "C": (40.0, 20.0),
    "D": (0.0, 20.0),
    # Halfway line
    "E": (20.0, 0.0),
    "F": (20.0, 20.0),
    # 6 m lines crossing the goal lines
    "G": (0.0, 4.0),
    "H": (0.0, 16.0),
    "I": (40.0, 4.0),
    "J": (40.0, 16.0),
    # Tips of the 6 m areas (on the centre line Y=10)
    "K": (6.0, 10.0),
    "L": (34.0, 10.0),
    # Goal posts
    "M_prime": (0.0, 8.5),
    "N_prime": (0.0, 11.5),
    "M": (40.0, 8.5),
    "N": (40.0, 11.5),
    # Tips of the 9 m lines
    "O_prime": (9.0, 10.0),
    "O": (31.0, 10.0),
}


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    """Configure the project-wide logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
