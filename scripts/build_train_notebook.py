"""Generate the training notebook of the handball YOLO model.

Fine-tunes a YOLO26 on a Roboflow Handball Detection export
(4 classes players/goalkeeper/referees/ball) on GPU, then exports the
weights for the inference pipeline.

Usage: python scripts/build_train_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "train_handball_yolo.ipynb"


def md(source: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]],
    }


def code(source: str) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]],
    }


CELLS = [
    md(
        "# Handball YOLO model training\n"
        "\n"
        "Fine-tunes a YOLO26 on a **Handball Detection** export (Roboflow Universe):\n"
        "4 classes `players` / `goalkeeper` / `referees` / `ball`.\n"
        "\n"
        "Goal: replace the generic YOLO26m COCO model (which detects 'person' and fragments\n"
        "the tracking) with a handball detector that tells players, goalkeepers,\n"
        "referees and ball apart.\n"
        "\n"
        "**Output**: `best.pt` saved in `models/`, then passed to\n"
        "`Detector(...)` in the inference notebook.\n"
        "\n"
        "**Requirement**: CUDA GPU."
    ),
    md(
        "## 1. Prepare the environment\n"
        "\n"
        "In the local repository or the devcontainer, use the project Python environment.\n"
        "\n"
        "To download a missing export, set `ROBOFLOW_API_KEY` in your environment. Exports already present can be used without a key."
    ),
    code(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "os.environ.setdefault(\"YOLO_CONFIG_DIR\", \"/tmp\")\n"
        "os.environ.setdefault(\"MPLCONFIGDIR\", \"/tmp/matplotlib\")\n"
        "\n"
        "import torch\n"
        "\n"
        "if not torch.cuda.is_available():\n"
        "    raise RuntimeError(\"Enable a CUDA GPU in your environment.\")\n"
        "print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n"
        "print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")"
    ),
    md(
        "## 2. Select the dataset\n"
        "\n"
        "`Handball-Detection-1` is the export used for the panorama fine-tuning, under the CC BY 4.0 licence.\n"
        "\n"
        "- [Handball-Detection-1, erics-workspace-it3ft workspace](https://universe.roboflow.com/erics-workspace-it3ft/handball-detection-fj8rc-omtda/dataset/3)\n"
        "\n"
        "Provenance and manual installation instructions are in `data/datasets/README.md`."
    ),
    code(
        "from getpass import getpass\n"
        "\n"
        "EXPORTS = {\n"
        "    \"Handball-Detection-1\": (\"erics-workspace-it3ft\", \"handball-detection-fj8rc-omtda\", 3),\n"
        "}\n"
        "EXPORT = \"Handball-Detection-1\"\n"
        "\n"
        "current = Path.cwd().resolve()\n"
        "root = next(\n"
        "    (p for p in (current, *current.parents)\n"
        "     if (p / \"scripts\" / \"build_train_notebook.py\").is_file()),\n"
        "    current,\n"
        ")\n"
        "dataset_dir = root / \"data\" / \"datasets\" / EXPORT\n"
        "if not (dataset_dir / \"data.yaml\").is_file():\n"
        "    try:\n"
        "        from roboflow import Roboflow\n"
        "    except ImportError as exc:\n"
        "        raise RuntimeError(\"Install roboflow to download a missing export.\") from exc\n"
        "\n"
        "    key = os.environ.get(\"ROBOFLOW_API_KEY\")\n"
        "    if not key:\n"
        "        key = getpass(\"Roboflow API key: \")\n"
        "    workspace, project, version = EXPORTS[EXPORT]\n"
        "    rf = Roboflow(api_key=key)\n"
        "    dataset_dir.parent.mkdir(parents=True, exist_ok=True)\n"
        "    rf.workspace(workspace).project(project).version(version).download(\n"
        "        \"yolov8\", location=str(dataset_dir)\n"
        "    )\n"
        "\n"
        "print(f\"Dataset: {dataset_dir}\")"
    ),
    md(
        "## 3. Inspect the dataset"
    ),
    code(
        "from tempfile import TemporaryDirectory\n"
        "\n"
        "import yaml\n"
        "\n"
        "cfg = yaml.safe_load((dataset_dir / \"data.yaml\").read_text(encoding=\"utf-8\"))\n"
        "cfg[\"path\"] = str(dataset_dir)\n"
        "for split_key, split in ((\"train\", \"train\"), (\"val\", \"valid\"), (\"test\", \"test\")):\n"
        "    images = dataset_dir / split / \"images\"\n"
        "    if not images.is_dir():\n"
        "        raise FileNotFoundError(f\"Missing image folder: {images}\")\n"
        "    cfg[split_key] = f\"{split}/images\"\n"
        "    print(f\"{split:6s}: {sum(p.is_file() for p in images.iterdir())} images\")\n"
        "\n"
        "# Ultralytics gets a YAML usable from any directory.\n"
        "temp_config = TemporaryDirectory()\n"
        "data_yaml = str(Path(temp_config.name) / \"data.yaml\")\n"
        "Path(data_yaml).write_text(yaml.safe_dump(cfg), encoding=\"utf-8\")\n"
        "print(\"Classes (dataset order):\", cfg[\"names\"])"
    ),
    md(
        "## 4. Train the model\n"
        "\n"
        "The notebook resumes from the `models/handball_<YOLO_BASE>.pt` weights when they exist and trains at 1280 px with an automatic batch size. It uses the COCO weights of `YOLO_BASE` (YOLO26m by default) when no local weights are available."
    ),
    code(
        "from ultralytics import YOLO\n"
        "\n"
        "YOLO_BASE = \"yolo26m\"\n"
        "local_weights = root / \"models\" / (\"handball_\"+YOLO_BASE+\".pt\")\n"
        "BASE_MODEL = str(local_weights) if local_weights.is_file() else YOLO_BASE+\".pt\"\n"
        "EPOCHS = 50\n"
        "IMGSZ = 1280\n"
        "BATCH = -1\n"
        "\n"
        "model = YOLO(BASE_MODEL)\n"
        "results = model.train(\n"
        "    data=data_yaml,\n"
        "    epochs=EPOCHS,\n"
        "    imgsz=IMGSZ,\n"
        "    batch=BATCH,\n"
        "    device=0,\n"
        "    workers=8,\n"
        "    patience=15,\n"
        "    name=\"handball_\"+YOLO_BASE,\n"
        "    project=str(root / \"data\" / \"training\"),\n"
        "    verbose=True,\n"
        ")\n"
        "print(f\"Best weights: {model.trainer.best}\")"
    ),
    md("## 5. Validation metrics"),
    code(
        "metrics = model.val(data=data_yaml)\n"
        "print(f\"mAP50-95: {metrics.box.map:.4f}\")\n"
        "print(f\"mAP50   : {metrics.box.map50:.4f}\")\n"
        "print(f\"mAP75   : {metrics.box.map75:.4f}\")\n"
        "print(\"\\nmAP50 per class:\")\n"
        "for i, name in enumerate(cfg.get(\"names\", [])):\n"
        "    try:\n"
        "        print(f\"  {name:12s}: {metrics.box.ap50[i]:.4f}\")\n"
        "    except (IndexError, TypeError):\n"
        "        pass"
    ),
    md(
        "## 6. Export the weights"
    ),
    code(
        "import shutil\n"
        "\n"
        "models_dir = root / \"models\"\n"
        "models_dir.mkdir(parents=True, exist_ok=True)\n"
        "target_path = models_dir / (\"handball_\"+YOLO_BASE+\".pt\")\n"
        "shutil.copy2(model.trainer.best, target_path)\n"
        "print(f\"Model saved: {target_path}\")"
    ),
    md("## 7. Visual check on a few validation images"),
    code(
        "import matplotlib.pyplot as plt\n"
        "import cv2\n"
        "\n"
        "val_images = [str(p) for p in sorted((dataset_dir / \"valid\" / \"images\").iterdir()) if p.is_file()]\n"
        "sample = val_images[:6] if len(val_images) >= 6 else val_images\n"
        "\n"
        "if sample:\n"
        "    preds = model.predict(sample, conf=0.35, verbose=False)\n"
        "    fig, axes = plt.subplots(2, 3, figsize=(20, 10))\n"
        "    for ax, pred in zip(axes.flat, preds):\n"
        "        annotated = pred.plot()  # annotated BGR\n"
        "        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))\n"
        "        ax.axis('off')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print(\"No validation images found.\")"
    ),
    md(
        "## Next: use the model in the pipeline\n"
        "\n"
        "In the inference notebook, use the path printed at export time to point\n"
        "to the handball model:\n"
        "\n"
        "```python\n"
        "from handball_ai.detection import Detector\n"
        "\n"
        "detector = Detector(\"PATH_TO_THE_EXPORTED_MODEL\")\n"
        "result = process_match(\n"
        "    video_path=CLIP,\n"
        "    court_mapping=court_mapping,\n"
        "    output_dir=OUTPUTS_DIR,\n"
        "    subsample=2,\n"
        "    detector=detector,\n"
        ")\n"
        "```\n"
        "\n"
        "The pipeline then detects players / goalkeepers / referees / ball separately,\n"
        "which strongly reduces tracking fragmentation."
    ),
]


def build_notebook() -> dict:
    """Assemble the notebook JSON structure."""
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": ".venv (3.11.18)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.18"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    """Write the notebook to disk."""
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Notebook written: {NOTEBOOK_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
