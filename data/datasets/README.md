# Provenance des datasets d’entraînement

L’export Roboflow Universe utilise le format d’annotation YOLOv8 (compatible YOLO26) et les classes
`ball`, `goalkeeper`, `players` et `referees`. Ses fichiers
`README.dataset.txt`, `README.roboflow.txt` et `data.yaml` d’origine sont
conservés dans son dossier.

| Dossier local | Source et attribution | Licence | Contenu de l’export |
| --- | --- | --- | --- |
| `Handball-Detection-1` | [Handball Detection v3, espace `erics-workspace-it3ft`](https://universe.roboflow.com/erics-workspace-it3ft/handball-detection-fj8rc-omtda/dataset/3) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 350 images, sans augmentation, export du 19 novembre 2026 |

Pour installer les données manuellement, télécharger la version indiquée au format
YOLOv8 et extraire son contenu dans le dossier indiqué ci-dessus. Le
notebook peut aussi télécharger l’export sélectionné dans `data/datasets`.
