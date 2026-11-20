"""Homographie : projection des coordonnees pixel vers le terrain en metres.

Une homographie est une transformation projective 3x3 qui mappe le plan image
au plan terrain. Necessite minimum 4 correspondances (pixel, terrain_m).
Avec 4 points : `getPerspectiveTransform`. Avec plus : `findHomography` + RANSAC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict

import cv2
import numpy as np

from pivot_ai.config import POINTS_TERRAIN_M

logger = logging.getLogger(__name__)


class Correspondance(TypedDict):
    """Une correspondance pixel <-> terrain pour calibrer l'homographie."""

    pixel: tuple[int, int]
    terrain_m: tuple[float, float]


@dataclass
class CorrectionRadiale:
    """Correction à un coefficient, autour du centre des quatre coins."""

    centre: np.ndarray
    rayon: float
    coefficient: float

    def corriger(self, pixels: np.ndarray) -> np.ndarray:
        ecarts = pixels - self.centre
        rayon_carre = np.sum((ecarts / self.rayon) ** 2, axis=-1, keepdims=True)
        # Au-delà du rayon calibré, prolonger avec un facteur constant.
        facteur = 1 + self.coefficient * np.minimum(rayon_carre, 1)
        return self.centre + ecarts * facteur


@dataclass
class Homographie:
    """Homographie calibree, encapsule la matrice 3x3 et les methodes de projection."""

    matrice: np.ndarray
    nb_points: int
    methode: str
    correction_radiale: CorrectionRadiale | None = None

    def projeter_point(self, point_pixel: tuple[float, float]) -> tuple[float, float]:
        """Projette un point pixel vers le terrain (metres)."""
        pt = np.array([[point_pixel]], dtype=np.float32)
        if self.correction_radiale is not None:
            pt = self.correction_radiale.corriger(pt).astype(np.float32)
        proj = cv2.perspectiveTransform(pt, self.matrice)
        return float(proj[0][0][0]), float(proj[0][0][1])

    def projeter_bbox_pieds(self, bbox: np.ndarray) -> tuple[float, float]:
        """Projette le centre-bas d'une bbox (position des pieds) vers le terrain."""
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_bottom = float(bbox[3])
        return self.projeter_point((x_center, y_bottom))


def _calibrer_avec_fisheye(
    correspondances: dict[str, Correspondance],
) -> Homographie:
    from scipy.optimize import minimize_scalar

    pixels = np.array([correspondances[nom]["pixel"] for nom in "ABCDEF"], dtype=np.float64)
    coins = pixels[:4]
    centre = coins.mean(axis=0)
    rayon = float(np.linalg.norm(coins - centre, axis=1).max())
    if not np.isfinite(pixels).all() or not cv2.isContourConvex(coins.astype(np.float32)):
        raise ValueError("Les coins A, B, C et D doivent former un quadrilatère convexe.")

    def erreur_alignement(coefficient: float) -> float:
        a, b, c, d, e, f = CorrectionRadiale(centre, rayon, coefficient).corriger(pixels)
        # Distance perpendiculaire carrée de E à AB et de F à CD, en pixels.
        erreur = 0.0
        for debut, fin, milieu in ((a, b, e), (c, d, f)):
            dx, dy = fin - debut
            mx, my = milieu - debut
            erreur += (dx * my - dy * mx) ** 2 / (dx * dx + dy * dy)
        return float(erreur)

    # La borne basse garde la correction croissante en fonction du rayon.
    resultat = minimize_scalar(erreur_alignement, bounds=(-0.3, 0.5), method="bounded")
    erreur_initiale = erreur_alignement(0)
    coefficient = float(resultat.x) if resultat.success and resultat.fun < erreur_initiale else 0.0
    correction = CorrectionRadiale(centre, rayon, coefficient)
    coins_corriges = correction.corriger(coins).astype(np.float32)
    rectangle = np.array([POINTS_TERRAIN_M[nom] for nom in "ABCD"], dtype=np.float32)
    matrice = cv2.getPerspectiveTransform(coins_corriges, rectangle)
    if (
        not cv2.isContourConvex(coins_corriges)
        or not np.isfinite(matrice).all()
        or np.linalg.matrix_rank(matrice) < 3
    ):
        raise ValueError("Correction fisheye dégénérée. Vérifie les repères A à F.")
    logger.info(
        "Correction fisheye : k=%.4f, écart RMS sur E et F de %.1f à %.1f px",
        coefficient, (erreur_initiale / 2) ** 0.5, (erreur_alignement(coefficient) / 2) ** 0.5,
    )
    return Homographie(matrice, len(pixels), "fisheye_4pts", correction)


def calibrer_homographie(
    correspondances: dict[str, Correspondance], *, corriger_fisheye: bool = False
) -> Homographie:
    """Calibre une homographie a partir de correspondances pixel <-> terrain.

    Args:
        correspondances: dict nom_point -> {"pixel": (x,y), "terrain_m": (X,Y)}
        corriger_fisheye: estime la correction si les six repères A à F sont présents

    Returns:
        Homographie calibree

    Raises:
        ValueError: si moins de 4 points ou si la calibration echoue
    """
    if len(correspondances) < 4:
        raise ValueError(
            f"Homographie : minimum 4 points requis, recu {len(correspondances)}"
        )

    if corriger_fisheye:
        manquants = sorted(set("ABCDEF") - correspondances.keys())
        if not manquants:
            return _calibrer_avec_fisheye(correspondances)
        logger.info("Correction fisheye ignorée, repères manquants : %s", ", ".join(manquants))

    pts_pix = np.array(
        [c["pixel"] for c in correspondances.values()], dtype=np.float32
    )
    pts_ter = np.array(
        [c["terrain_m"] for c in correspondances.values()], dtype=np.float32
    )

    if len(correspondances) == 4:
        H = cv2.getPerspectiveTransform(pts_pix, pts_ter)
        if H is None:
            raise ValueError("getPerspectiveTransform a echoue, points colineaires ?")
        return Homographie(matrice=H, nb_points=4, methode="perspective_4pts")

    # >4 points : RANSAC d'abord, fallback least-squares
    H, _ = cv2.findHomography(
        pts_pix, pts_ter, method=cv2.RANSAC, ransacReprojThreshold=2.0
    )
    if H is not None:
        return Homographie(
            matrice=H, nb_points=len(correspondances), methode="ransac"
        )

    logger.warning("RANSAC degenere, fallback least-squares")
    H, _ = cv2.findHomography(pts_pix, pts_ter, method=0)
    if H is None:
        raise ValueError("Calibration homographie KO meme en fallback")

    return Homographie(
        matrice=H, nb_points=len(correspondances), methode="least_squares"
    )
