#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    echo "Usage: bash copy_yolo_labels.sh [dataset_folder]"
    exit 0
fi
if (( $# > 1 )); then
    echo "Usage: bash copy_yolo_labels.sh [dataset_folder]" >&2
    exit 2
fi

dataset=${1:-.}
source_labels="$dataset/train/labels"

for folder in "$source_labels" "$dataset/test/images" "$dataset/valid/images"; do
    if [[ ! -d "$folder" ]]; then
        echo "Folder not found: $folder" >&2
        exit 1
    fi
done

shopt -s nullglob nocaseglob
missing=0
for split in test valid; do
    destination="$dataset/$split/labels"
    mkdir -p -- "$destination"
    copied=0
    existing=0

    for image in "$dataset/$split/images/"*.{jpg,jpeg,png,webp,bmp,tif,tiff,dng,mpo,pfm,heic,avif}; do
        [[ -f "$image" ]] || continue
        name=${image##*/}
        label="${name%.*}.txt"
        if [[ -f "$destination/$label" ]]; then
            existing=$((existing + 1))
        elif [[ -f "$source_labels/$label" ]]; then
            cp -- "$source_labels/$label" "$destination/$label"
            copied=$((copied + 1))
        else
            echo "Label not found for $split/images/$name: $label" >&2
            missing=$((missing + 1))
        fi
    done

    echo "$split: $copied label(s) copied, $existing already present"
done

if (( missing > 0 )); then
    echo "$missing label(s) not found." >&2
    exit 1
fi
