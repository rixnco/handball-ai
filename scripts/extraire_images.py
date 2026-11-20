#!/usr/bin/env python3
"""Extrait des images JPEG réparties aléatoirement sur la durée d'une vidéo."""

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def duree_video(video: Path) -> float:
    resultat = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "V:0",
            "-show_entries", "stream=duration:format=duration", "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    )
    metadonnees = json.loads(resultat.stdout)
    flux = metadonnees.get("streams", [])
    if not flux:
        raise ValueError("Aucun flux vidéo trouvé.")
    for source in (flux[0], metadonnees.get("format", {})):
        valeur = source.get("duration")
        if valeur not in (None, "N/A"):
            duree = float(valeur)
            if math.isfinite(duree) and duree > 0:
                return duree
    raise ValueError("La durée de la vidéo est absente des métadonnées.")


def extraire_images(
    video: Path, nombre: int, cible: Path, prefixe: str, seed: int | None,
    progression: bool = True,
) -> None:
    duree = duree_video(video)
    largeur = duree / nombre
    aleatoire = random.Random(seed)

    cible.mkdir(parents=True, exist_ok=True)
    destinations = [cible / f"{prefixe}_{numero:04d}.jpg" for numero in range(1, nombre + 1)]
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Le fichier existe déjà : {destination}")

    with TemporaryDirectory(dir=cible) as temporaire:
        dossier = Path(temporaire)
        sources = [dossier / f"{numero}.jpg" for numero in range(1, nombre + 1)]
        if progression:
            print(f"Extraction des images (vidéo de {duree:.1f} s) : 0/{nombre}",
                  file=sys.stderr, flush=True)
        for tranche, source in enumerate(sources):
            debut = tranche * largeur
            instant = debut + aleatoire.random() * largeur
            # Près de la fin, le tirage peut tomber après la dernière image décodable.
            for position in (instant, debut):
                subprocess.run(
                    [
                        "ffmpeg", "-v", "error", "-nostdin", "-ss", f"{position:.9f}",
                        "-i", str(video), "-map", "0:V:0", "-frames:v", "1", "-q:v", "2",
                        "-f", "image2", "-update", "1", str(source),
                    ],
                    check=True, capture_output=True, text=True,
                )
                if source.is_file():
                    break
            else:
                raise ValueError(f"Aucune image décodable près de {instant:.3f} s.")
            if progression:
                print(f"\rExtraction des images : {tranche + 1}/{nombre}",
                      end="\n" if tranche + 1 == nombre else "",
                      file=sys.stderr, flush=True)

        publiees = []
        try:
            for source, destination in zip(sources, destinations, strict=True):
                destination.hardlink_to(source)
                publiees.append(destination)
        except OSError:
            for destination in publiees:
                destination.unlink()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positionnels", nargs="*", metavar="ARG", help="vidéo, nombre, cible")
    parser.add_argument("-input", "--input", dest="video", type=Path, help="vidéo source")
    parser.add_argument("-count", "--count", dest="nombre", type=int, help="nombre d’images")
    parser.add_argument("-output", "--output", dest="cible", type=Path, help="répertoire cible")
    parser.add_argument("-prefix", "--prefix", "--prefixe", dest="prefixe", default="image",
                        help="préfixe des noms de fichiers")
    parser.add_argument("-seed", "--seed", type=int, help="graine pour reproduire la sélection")
    parser.add_argument("-no-progress", "--no-progress", action="store_true",
                        help="masquer l’indicateur de progression")
    args = parser.parse_args()
    parametres = (args.video, args.nombre, args.cible)
    if args.positionnels:
        if len(args.positionnels) != 3 or any(valeur is not None for valeur in parametres):
            parser.error("indiquez soit vidéo nombre cible, soit -input, -count et -output")
        video, nombre, cible = args.positionnels
        args.video, args.cible = Path(video), Path(cible)
        try:
            args.nombre = int(nombre)
        except ValueError:
            parser.error("le nombre d’images doit être un entier")
    elif any(valeur is None for valeur in parametres):
        parser.error("les options -input, -count et -output sont requises")
    if args.nombre < 1:
        parser.error("le nombre d’images doit être supérieur à zéro")
    if not args.prefixe or "/" in args.prefixe or "\\" in args.prefixe:
        parser.error("le préfixe doit être un nom de fichier sans séparateur de chemin")

    try:
        extraire_images(
            args.video.resolve(), args.nombre, args.cible.resolve(), args.prefixe, args.seed,
            progression=not args.no_progress,
        )
    except subprocess.CalledProcessError as erreur:
        parser.exit(1, f"Erreur de {erreur.cmd[0]} : {erreur.stderr.strip()}\n")
    except (OSError, ValueError) as erreur:
        parser.exit(1, f"Erreur : {erreur}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nExtraction interrompue.\n")

    print(f"{args.nombre} images extraites dans {args.cible}")


if __name__ == "__main__":
    main()
