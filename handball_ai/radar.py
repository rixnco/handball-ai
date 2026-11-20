"""2D radar rendering: handball court with player positions."""

from __future__ import annotations

import math

import cv2
import numpy as np

from handball_ai.config import CourtConfig


def draw_empty_court(config: CourtConfig) -> np.ndarray:
    """Draw an empty handball court (lines, goal areas, 9 m lines)."""
    radar = np.full((config.radar_h, config.radar_w, 3), 35, dtype=np.uint8)

    # Court surface (beige)
    cv2.rectangle(radar, config.m2px(0, 0), config.m2px(40, 20), (200, 160, 110), -1)
    # Border
    cv2.rectangle(radar, config.m2px(0, 0), config.m2px(40, 20), (255, 255, 255), 2)
    # Halfway line
    cv2.line(radar, config.m2px(20, 0), config.m2px(20, 20), (255, 255, 255), 1)

    # Each line joins two arcs centred on the posts with a 3 m segment.
    white = (255, 255, 255)
    for goal_x, direction, orientation in ((0, 1, 0), (40, -1, 180)):
        top_post = config.m2px(goal_x, 11.5)
        bottom_post = config.m2px(goal_x, 8.5)
        for radius, thickness in ((6, 2), (9, 1)):
            line_x = goal_x + direction * radius
            cv2.line(
                radar, config.m2px(line_x, 11.5), config.m2px(line_x, 8.5),
                white, thickness, cv2.LINE_AA,
            )
            # The 9 m radius reaches the side line before the end of the quarter circle.
            span = math.degrees(math.asin(min(1.0, 8.5 / radius)))
            axes = (radius * config.px_per_m,) * 2
            for post, half_plane in ((top_post, -direction), (bottom_post, direction)):
                angle = orientation + half_plane * span
                start, end = sorted((orientation, angle))
                cv2.ellipse(radar, post, axes, 0, start, end, white, thickness, cv2.LINE_AA)

    # Goal lines (red)
    cv2.line(radar, config.m2px(0, 8.5), config.m2px(0, 11.5), (0, 0, 220), 4)
    cv2.line(radar, config.m2px(40, 8.5), config.m2px(40, 11.5), (0, 0, 220), 4)

    return radar


def draw_legend(radar: np.ndarray, show_teams: bool = True) -> np.ndarray:
    """Add the colour legend in the top left corner."""
    if show_teams:
        cv2.putText(radar, "HOME", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(radar, "AWAY", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(radar, "GK", (90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(radar, "REF", (90, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 0, 255), 2, cv2.LINE_AA)
    return radar


def draw_radar(
    positions: list[tuple[float, float, tuple[int, int, int], str, int]],
    config: CourtConfig | None = None,
    *,
    show_teams: bool = True,
) -> np.ndarray:
    """Draw a radar with player positions.

    Args:
        positions: list of (x_m, y_m, bgr_color, tag, tracker_id) tuples
        config: court configuration (default: CourtConfig())
        show_teams: show the HOME / AWAY legend

    Returns:
        BGR radar image
    """
    config = config or CourtConfig()
    radar = draw_empty_court(config)
    radar = draw_legend(radar, show_teams)

    for x_m, y_m, color, _tag, tid in positions:
        x_c = float(np.clip(x_m, 0, config.length_m))
        y_c = float(np.clip(y_m, 0, config.width_m))
        cx, cy = config.m2px(x_c, y_c)
        cv2.circle(radar, (cx, cy), 9, color, -1)
        cv2.circle(radar, (cx, cy), 9, (255, 255, 255), 2)
        cv2.putText(radar, str(tid), (cx + 11, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return radar
