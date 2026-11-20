"""Accès aux images de la vidéo source."""

from __future__ import annotations

import cv2


def positionner_capture(cap: cv2.VideoCapture, frame_debut: int) -> None:
    """Effectue un seul accès aléatoire avant la lecture séquentielle."""
    if frame_debut < 0:
        raise ValueError("L'index de début doit être positif ou nul")
    if frame_debut and not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_debut):
        raise RuntimeError(f"Impossible de se positionner à l'image {frame_debut}")
