#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    echo "Usage : bash copier_labels_yolo.sh [dossier_dataset]"
    exit 0
fi
if (( $# > 1 )); then
    echo "Usage : bash copier_labels_yolo.sh [dossier_dataset]" >&2
    exit 2
fi

dataset=${1:-.}
source_labels="$dataset/train/labels"

for dossier in "$source_labels" "$dataset/test/images" "$dataset/valid/images"; do
    if [[ ! -d "$dossier" ]]; then
        echo "Dossier introuvable : $dossier" >&2
        exit 1
    fi
done

shopt -s nullglob nocaseglob
manquants=0
for split in test valid; do
    destination="$dataset/$split/labels"
    mkdir -p -- "$destination"
    copies=0
    existants=0

    for image in "$dataset/$split/images/"*.{jpg,jpeg,png,webp,bmp,tif,tiff,dng,mpo,pfm,heic,avif}; do
        [[ -f "$image" ]] || continue
        nom=${image##*/}
        label="${nom%.*}.txt"
        if [[ -f "$destination/$label" ]]; then
            existants=$((existants + 1))
        elif [[ -f "$source_labels/$label" ]]; then
            cp -- "$source_labels/$label" "$destination/$label"
            copies=$((copies + 1))
        else
            echo "Annotation introuvable pour $split/images/$nom : $label" >&2
            manquants=$((manquants + 1))
        fi
    done

    echo "$split : $copies annotation(s) copiée(s), $existants déjà présente(s)"
done

if (( manquants > 0 )); then
    echo "$manquants annotation(s) introuvable(s)." >&2
    exit 1
fi
