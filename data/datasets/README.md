# Training dataset provenance

The Roboflow Universe export uses the YOLOv8 annotation format (compatible with YOLO26) and the classes
`ball`, `goalkeeper`, `players` and `referees`. Its original
`README.dataset.txt`, `README.roboflow.txt` and `data.yaml` files are
kept in its folder.

| Local folder | Source and attribution | Licence | Export content |
| --- | --- | --- | --- |
| `Handball-Detection-1` | [Handball Detection v3, `erics-workspace-it3ft` workspace](https://universe.roboflow.com/erics-workspace-it3ft/handball-detection-fj8rc-omtda/dataset/3) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 350 images, no augmentation, exported on 19 November 2026 |

To install the data manually, download the listed version in the YOLOv8 format
and extract its content into the folder above. The notebook can also
download the selected export into `data/datasets`.
