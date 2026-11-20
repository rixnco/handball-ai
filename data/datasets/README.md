# Provenance des datasets d’entraînement

Les deux exports Roboflow Universe utilisent le format YOLOv8 et les classes
`ball`, `goalkeeper`, `players` et `referees`. Leurs fichiers
`README.dataset.txt`, `README.roboflow.txt` et `data.yaml` d’origine sont
conservés dans chaque dossier d’export.

| Dossier local | Source et attribution | Licence | Contenu de l’export |
| --- | --- | --- | --- |
| `Handball-Detection-1` | [Handball Detection v1, espace `erics-workspace-it3ft`](https://universe.roboflow.com/erics-workspace-it3ft/handball-detection-fj8rc-omtda/dataset/1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 310 images, sans augmentation, export du 18 novembre 2026 |
| `Handball-Detection-2` | [HandballPlayerDetection_v2, espace `handballdetectionvictorcollado`](https://universe.roboflow.com/handballdetectionvictorcollado/handball-detection-fj8rc/dataset/2) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 377 images, orientation automatique, redimensionnement en 640 × 640 et retournements horizontaux, export du 4 septembre 2024 |

Pour installer les données manuellement, télécharger la version indiquée au format
YOLOv8 et extraire son contenu dans le dossier correspondant ci-dessus. Le
notebook peut aussi télécharger l’export sélectionné dans `data/datasets`.
