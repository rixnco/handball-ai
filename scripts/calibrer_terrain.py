"""Créer les correspondances d’homographie en cliquant sur une image du match."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def lire_image(source: Path, numero: int | None):
    """Lire une image fixe ou extraire une frame de la vidéo."""
    import cv2

    if numero is None:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is not None:
            return image, source.name

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Impossible d’ouvrir {source}.")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        index = (total // 2 if total > 0 else 0) if numero is None else numero
        if index < 0 or (total > 0 and index >= total):
            raise ValueError(f"Frame {index} hors de la vidéo ({total} frames).")
        if index and not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError(f"Impossible d’atteindre la frame {index}.")
        ok, image = capture.read()
        if not ok:
            raise ValueError(f"Impossible de lire la frame {index}.")
        return image, f"{source.name} · frame {index}"
    finally:
        capture.release()


def enregistrer(points: dict[str, tuple[int, int]], sortie: Path, force: bool) -> None:
    """Valider les points puis écrire le JSON attendu par le pipeline."""
    import numpy as np

    from pivot_ai.config import POINTS_TERRAIN_M
    from pivot_ai.homographie import calibrer_homographie

    if len(set(points.values())) != len(points):
        raise ValueError("Chaque repère doit correspondre à un pixel distinct.")
    correspondances = {
        nom: {"pixel": pixel, "terrain_m": POINTS_TERRAIN_M[nom]}
        for nom, pixel in points.items()
    }
    matrice = calibrer_homographie(correspondances).matrice
    if not np.isfinite(matrice).all() or np.linalg.matrix_rank(matrice) < 3:
        raise ValueError("Points dégénérés. Répartis les repères sur la surface du terrain.")

    sortie.parent.mkdir(parents=True, exist_ok=True)
    contenu = json.dumps(correspondances, indent=2, ensure_ascii=False) + "\n"
    with sortie.open("w" if force else "x", encoding="utf-8") as fichier:
        fichier.write(contenu)


def ouvrir_editeur(image, titre: str, sortie: Path, force: bool, marge: int | None = None) -> bool:
    """Associer les repères du schéma aux pixels de l’image."""
    try:
        import tkinter as tk
    except ImportError as erreur:
        raise ValueError("Tkinter est requis pour l’éditeur. Sous Debian ou Ubuntu, installe python3-tk.") from erreur
    from tkinter import ttk

    from PIL import Image, ImageTk

    from pivot_ai.config import POINTS_TERRAIN_M

    hauteur, largeur = image.shape[:2]
    marge_x = max(1, round(largeur * 0.1)) if marge is None else marge
    marge_y = max(1, round(hauteur * 0.1)) if marge is None else marge
    rgb = Image.fromarray(image[:, :, ::-1])

    try:
        fenetre = tk.Tk()
    except tk.TclError as erreur:
        raise ValueError(f"Impossible d’ouvrir l’éditeur Tk. Vérifie l’affichage graphique ({erreur}).") from erreur
    fenetre.title("Calibrer le terrain")
    fenetre.geometry("1200x760")
    fenetre.minsize(620, 460)
    fenetre.columnconfigure(0, weight=1)
    fenetre.rowconfigure(1, weight=1)

    ttk.Label(fenetre, text=titre, padding=(12, 6)).grid(row=0, sticky="w")
    photo = tk.Canvas(fenetre, background="#252525", highlightthickness=0)
    photo.grid(row=1, sticky="nsew")
    ttk.Label(
        fenetre, text="Molette : zoom   ·   Clic droit + glisser : déplacer   ·   Clic gauche : placer",
        padding=(12, 4),
    ).grid(row=2, sticky="w")

    terrain = tk.Canvas(fenetre, height=145, background="#f1f4f5", highlightthickness=0)
    terrain.grid(row=3, sticky="ew")
    commandes = ttk.Frame(fenetre, padding=(12, 8))
    commandes.grid(row=4, sticky="ew")
    commandes.columnconfigure(0, weight=1)
    statut = tk.StringVar()
    ttk.Label(commandes, textvariable=statut, wraplength=440).grid(row=0, column=0, sticky="w")

    points: dict[str, tuple[int, int]] = {}
    historique: list[tuple[str, tuple[int, int] | None]] = []
    selection = None
    sauvegarde = False
    echelle = 1.0
    origine_x = origine_y = 0.0
    vue_manipulee = False
    rendu = None
    ancre = None

    def afficher_photo() -> None:
        nonlocal rendu
        inverse = 1 / echelle
        vue = rgb.transform(
            (photo.winfo_width(), photo.winfo_height()),
            Image.Transform.AFFINE,
            (inverse, 0, -origine_x * inverse, 0, inverse, -origine_y * inverse),
            resample=Image.Resampling.BILINEAR if echelle < 1 else Image.Resampling.NEAREST,
            fillcolor=(53, 53, 53),
        )
        rendu = ImageTk.PhotoImage(vue, master=photo)
        photo.delete("all")
        photo.create_image(0, 0, image=rendu, anchor="nw")
        for nom, (x, y) in points.items():
            px = origine_x + (x + 0.5) * echelle
            py = origine_y + (y + 0.5) * echelle
            photo.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#8bff7b", outline="black")
            photo.create_text(px + 8, py - 8, text=nom, anchor="sw", fill="white")

    def ajuster(_evenement=None) -> None:
        nonlocal echelle, origine_x, origine_y, vue_manipulee
        w, h = photo.winfo_width(), photo.winfo_height()
        echelle = min(w / (largeur + 2 * marge_x), h / (hauteur + 2 * marge_y))
        origine_x = (w - largeur * echelle) / 2
        origine_y = (h - hauteur * echelle) / 2
        vue_manipulee = False
        afficher_photo()

    def redimensionner(_evenement) -> None:
        if vue_manipulee:
            afficher_photo()
        else:
            ajuster()

    def zoomer(evenement) -> None:
        nonlocal echelle, origine_x, origine_y, vue_manipulee
        agrandir = evenement.num == 4 or evenement.delta > 0
        facteur = 1.25 if agrandir else 0.8
        echelle_min = min(
            photo.winfo_width() / (largeur + 2 * marge_x),
            photo.winfo_height() / (hauteur + 2 * marge_y),
        ) / 2
        nouvelle = max(echelle_min, min(max(24.0, echelle_min), echelle * facteur))
        ratio = nouvelle / echelle
        x, y = evenement.x + 0.5, evenement.y + 0.5
        origine_x = x - (x - origine_x) * ratio
        origine_y = y - (y - origine_y) * ratio
        echelle = nouvelle
        vue_manipulee = True
        afficher_photo()

    def commencer_deplacement(evenement) -> None:
        nonlocal ancre
        ancre = evenement.x, evenement.y, origine_x, origine_y

    def deplacer(evenement) -> None:
        nonlocal origine_x, origine_y, vue_manipulee
        if ancre is not None:
            x, y, ox, oy = ancre
            origine_x = ox + evenement.x - x
            origine_y = oy + evenement.y - y
            vue_manipulee = True
            afficher_photo()

    def actualiser(message: str = "") -> None:
        terrain.delete("all")
        centre = terrain.winfo_width() / 2
        echelle_terrain = min(5.5, (terrain.winfo_width() - 40) / 40)
        centre_y = 72
        gauche = centre - 20 * echelle_terrain
        droite = centre + 20 * echelle_terrain
        haut = centre_y - 10 * echelle_terrain
        bas = centre_y + 10 * echelle_terrain
        terrain.create_rectangle(gauche, haut, droite, bas, fill="#dceced", outline="#466671")
        terrain.create_line(centre, haut, centre, bas, fill="#466671")
        for nom, (x, y) in POINTS_TERRAIN_M.items():
            px, py = gauche + x * echelle_terrain, bas - y * echelle_terrain
            couleur = "#da7318" if nom == selection else "#168044" if nom in points else "#14628c"
            tag = f"repere:{nom}"
            terrain.create_oval(px - 5, py - 5, px + 5, py + 5, fill=couleur, outline="", tags=tag)
            if x == 0:
                terrain.create_text(px - 8, py, text=nom, anchor="e", tags=tag)
            elif x == 40:
                terrain.create_text(px + 8, py, text=nom, anchor="w", tags=tag)
            else:
                decalage_y = 10 if y == 0 or nom in {"K", "L"} else -10
                terrain.create_text(px, py + decalage_y, text=nom, tags=tag)
            terrain.tag_bind(tag, "<Button-1>", lambda _evt, nom=nom: choisir(nom))

        instruction = f"Place {selection} dans l’image ou la marge." if selection else "Choisis un repère ci-dessus."
        statut.set(f"{len(points)} point(s), minimum 4. {message or instruction}")

    def choisir(nom: str) -> None:
        nonlocal selection
        selection = nom
        actualiser()

    def placer(evenement) -> None:
        nonlocal selection
        if selection is None:
            return
        x = math.floor((evenement.x + 0.5 - origine_x) / echelle)
        y = math.floor((evenement.y + 0.5 - origine_y) / echelle)
        if -marge_x <= x < largeur + marge_x and -marge_y <= y < hauteur + marge_y:
            historique.append((selection, points.get(selection)))
            points[selection] = (x, y)
            selection = None
            afficher_photo()
            actualiser()

    def annuler() -> None:
        nonlocal selection
        if historique:
            selection, precedent = historique.pop()
            if precedent is None:
                del points[selection]
            else:
                points[selection] = precedent
            afficher_photo()
            actualiser()

    def sauvegarder() -> None:
        nonlocal sauvegarde
        try:
            enregistrer(points, sortie, force)
        except (ValueError, OSError) as erreur:
            actualiser(str(erreur))
            return
        sauvegarde = True
        print(f"{len(points)} correspondances enregistrées dans {sortie}")
        fenetre.destroy()

    ttk.Button(commandes, text="Ajuster l’image", command=ajuster).grid(row=0, column=1, padx=4)
    ttk.Button(commandes, text="Annuler le point", command=annuler).grid(row=0, column=2, padx=4)
    ttk.Button(commandes, text="Enregistrer", command=sauvegarder).grid(row=0, column=3, padx=4)
    photo.bind("<Configure>", redimensionner)
    for signal in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        photo.bind(signal, zoomer)
    for bouton in (2, 3):
        photo.bind(f"<ButtonPress-{bouton}>", commencer_deplacement)
        photo.bind(f"<B{bouton}-Motion>", deplacer)
    photo.bind("<Button-1>", placer)
    terrain.bind("<Configure>", lambda _evt: actualiser())
    fenetre.mainloop()
    return sauvegarde


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="vidéo du match ou image de calibration")
    parser.add_argument("--frame", type=int, help="index de la frame, à partir de 0 (défaut : milieu)")
    parser.add_argument("--marge", type=int, help="marge cliquable en pixels (défaut : 10 %% par axe)")
    parser.add_argument("--video-reference", type=Path, help="vidéo à associer si la source est une image")
    parser.add_argument("--sortie", type=Path, help="fichier JSON à créer (par défaut, à côté de la vidéo)")
    parser.add_argument("--force", action="store_true", help="autoriser le remplacement du fichier JSON")
    args = parser.parse_args()

    if args.marge is not None and args.marge < 0:
        parser.error("--marge doit être positive ou nulle.")
    if not args.source.is_file():
        parser.error(f"Fichier introuvable : {args.source}")
    if args.video_reference is not None and not args.video_reference.is_file():
        parser.error(f"Vidéo de référence introuvable : {args.video_reference}")

    from pivot_ai.config import chemin_correspondances_video

    video = args.video_reference or args.source
    sortie = args.sortie or chemin_correspondances_video(video)
    if sortie.resolve() in {args.source.resolve(), video.resolve()}:
        parser.error("La sortie JSON doit être différente du fichier source.")
    if sortie.exists() and not args.force:
        parser.error(f"{sortie} existe déjà. Utilise --force ou choisis une autre --sortie.")

    try:
        image, titre = lire_image(args.source, args.frame)
        if ouvrir_editeur(image, titre, sortie, args.force, marge=args.marge):
            return 0
    except (ValueError, OSError, ImportError) as erreur:
        parser.exit(1, f"{erreur}\n")

    print("Aucune correspondance enregistrée.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
