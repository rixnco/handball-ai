"""Team classification with KMeans on the torso colour.

Possible improvements:
- Move the colour from BGR to HSV or Lab for more robustness
- Allow a manual selection of the home cluster with a ROI on the first frame
- Handle goalkeepers, who often wear a different jersey
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
import supervision as sv
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def extract_torso_color(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Extract the median torso colour of a player inside a bbox.

    Uses a horizontal band in the upper middle of the bbox (25%-50% of the height),
    over 80% of the width, centred.

    Args:
        frame: BGR image
        bbox: array [x1, y1, x2, y2]

    Returns:
        median [B, G, R] colour (zeros for an invalid bbox)
    """
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return np.zeros(3, dtype=np.float64)

    cx = (x1 + x2) // 2
    half_w = max(1, int(w * 0.40))
    y_top = y1 + int(h * 0.25)
    y_bottom = y1 + int(h * 0.50)
    crop = frame[y_top:y_bottom, cx - half_w : cx + half_w]

    if crop.size == 0:
        return np.zeros(3, dtype=np.float64)

    return np.median(crop.reshape(-1, 3), axis=0).astype(np.float64)


def classify_teams(
    tracked_detections: dict[int, sv.Detections],
    frames_by_index: dict[int, np.ndarray] | Iterable[tuple[int, np.ndarray]],
    final_class: dict[int, int],
    player_class_id: int,
) -> tuple[dict[int, int], int]:
    """Assign each tracker_id to team 0 or 1 with KMeans on the torso colour.

    Args:
        tracked_detections: frame index -> Detections (with tracker_id)
        frames_by_index: indexed BGR frames, as a dict or a stream
        final_class: tracker_id -> stabilized class
        player_class_id: id of the "field player" class

    Returns:
        (final_team, home_cluster)
        - final_team: tracker_id -> 0 or 1
        - home_cluster: index of the cluster treated as the home team (the darkest)
    """
    colors: list[np.ndarray] = []
    color_trackers: list[int] = []

    if isinstance(frames_by_index, dict):
        frames = (
            (fi, frames_by_index[fi]) for fi in tracked_detections if fi in frames_by_index
        )
    else:
        frames = iter(frames_by_index)

    for fi, frame in frames:
        dets = tracked_detections[fi]
        if dets.tracker_id is None:
            continue
        for i in range(len(dets)):
            tid = int(dets.tracker_id[i])
            if final_class.get(tid) != player_class_id:
                continue
            colors.append(extract_torso_color(frame, dets.xyxy[i]))
            color_trackers.append(tid)

    if len(colors) < 2:
        logger.warning("Too few colour samples for KMeans, fallback")
        return {}, 0

    X = np.array(colors, dtype=np.float64)
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # Convention: home team = cluster with the darkest mean BGR
    means = [km.cluster_centers_[i].mean() for i in range(2)]
    home_cluster = int(np.argmin(means))

    # Majority vote per tracker_id
    votes: dict[int, Counter] = defaultdict(Counter)
    for tid, label in zip(color_trackers, labels, strict=True):
        votes[tid][int(label)] += 1

    final_team = {tid: int(counter.most_common(1)[0][0]) for tid, counter in votes.items()}

    logger.info(
        "Team classification: %d players classified, home_cluster=%d",
        len(final_team),
        home_cluster,
    )
    return final_team, home_cluster
