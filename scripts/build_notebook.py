"""Generate the local handball-ai notebook from cells defined in Python.

Keeps the content in readable Python code instead of hand-editing the .ipynb
JSON. Idempotent: can be run again.

Usage: python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "handball_ai.ipynb"


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
        "# handball-ai\n"
        "\n"
        "Open this notebook in the repository with the project Python kernel. Put your video in `data/raw_clips/`, then set its name in the configuration cell. The notebook looks for the court mapping next to the video. Results are written to `data/outputs/`."
    ),
    md(
        "## 1. Configure the match"
    ),
    code(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "import handball_ai\n"
        "from handball_ai.config import court_mapping_path\n"
        "\n"
        "current = Path.cwd().resolve()\n"
        "root = next(\n"
        "    (p for p in (current, *current.parents) if (p / \"scripts\" / \"build_notebook.py\").is_file()),\n"
        "    None,\n"
        ")\n"
        "if root is None:\n"
        "    raise RuntimeError(\"Open the notebook from the handball-ai repository.\")\n"
        "\n"
        "CLIPS_DIR = root / \"data\" / \"raw_clips\"\n"
        "CLIP = str(CLIPS_DIR / \"my_clip.mp4\")\n"
        "COURT_MAPPING_PATH = court_mapping_path(CLIP)\n"
        "OUTPUTS_DIR = root / \"data\" / \"outputs\"\n"
        "OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "print(f\"handball_ai version: {handball_ai.__version__}\")\n"
        "print(\"Available clips:\", [p.name for p in sorted(CLIPS_DIR.glob(\"*.mp4\"))])"
    ),
    md("## 2. Check the GPU"),
    code(
        "import torch\n"
        "print(f\"CUDA available: {torch.cuda.is_available()}\")\n"
        "if torch.cuda.is_available():\n"
        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n"
        "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")"
    ),
    md(
        "## 3. Pick a clip and record the homography points\n"
        "\n"
        "Shows a frame of the clip. **Hover with the mouse** to read the pixels of the court reference points.\n"
        "\n"
        "**Points to record (at least 4, ideally 6-8 to enable RANSAC):**\n"
        "\n"
        "| Name | Description | Court coords (m) |\n"
        "|------|-------------|------------------|\n"
        "| A | Top left court corner (near the bottom of the image) | (0, 0) |\n"
        "| B | Top right court corner | (40, 0) |\n"
        "| C | Bottom right court corner (near the top of the image) | (40, 20) |\n"
        "| D | Bottom left court corner | (0, 20) |\n"
        "| E | Halfway line, near side | (20, 0) |\n"
        "| F | Halfway line, far side | (20, 20) |\n"
        "| M_prime | Left goal post, near side | (0, 8.5) |\n"
        "| N_prime | Left goal post, far side | (0, 11.5) |\n"
        "| M | Right goal post, near side | (40, 8.5) |\n"
        "| N | Right goal post, far side | (40, 11.5) |\n"
        "| K | Tip of the left 6 m area | (6, 10) |\n"
        "| L | Tip of the right 6 m area | (34, 10) |\n"
        "| O_prime | Tip of the left 9 m line | (9, 10) |\n"
        "| O | Tip of the right 9 m line | (31, 10) |\n"
        "\n"
        "See `handball_ai/config.py` (COURT_POINTS_M) for the full list."
    ),
    code(
        "import cv2\n"
        "import plotly.express as px\n"
        "\n"
        "if not Path(CLIP).is_file():\n"
        "    raise FileNotFoundError(f\"Set the name of your clip in the configuration cell: {CLIP}\")\n"
        "\n"
        "cap = cv2.VideoCapture(CLIP)\n"
        "total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
        "fps = cap.get(cv2.CAP_PROP_FPS)\n"
        "REF_FRAME = total // 2\n"
        "cap.set(cv2.CAP_PROP_POS_FRAMES, REF_FRAME)\n"
        "ok, ref_frame = cap.read()\n"
        "cap.release()\n"
        "if not ok or fps <= 0:\n"
        "    raise RuntimeError(f\"Cannot read the clip: {CLIP}\")\n"
        "\n"
        "print(f\"Clip: {total} frames @ {fps:.1f} fps ({total/fps:.1f}s)\")\n"
        "\n"
        "fig = px.imshow(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB))\n"
        "fig.update_layout(width=1300, height=730, dragmode='pan',\n"
        "                  title=f\"Frame {REF_FRAME} — hover to read the pixels\")\n"
        "fig.show()"
    ),
    code(
        "import json\n"
        "\n"
        "if not COURT_MAPPING_PATH.is_file():\n"
        "    raise FileNotFoundError(\n"
        "        f\"Create {COURT_MAPPING_PATH} with the points recorded on this clip. \"\n"
        "        \"Use python -m scripts.calibrate_court with the clip path from the repository root.\"\n"
        "    )\n"
        "\n"
        "court_mapping = json.loads(COURT_MAPPING_PATH.read_text(encoding=\"utf-8\"))\n"
        "print(f\"Court mapping loaded: {len(court_mapping)} points\")"
    ),
    md(
        "## 4. Detection model\n"
        "\n"
        "The pipeline requires a YOLO model trained on handball with\n"
        "`notebooks/train_handball_yolo.ipynb`: it tells players, goalkeepers, referees and ball apart.\n"
        "When the panorama weights exist in `models/`, they are selected first; change\n"
        "`HANDBALL_MODEL_PATH` to pick another file."
    ),
    code(
        "from handball_ai.detection import Detector\n"
        "\n"
        "MODELS_DIR = root / \"models\"\n"
        "panorama_weights = MODELS_DIR / \"handball_yolo26m_panorama.pt\"\n"
        "standard_weights = MODELS_DIR / \"handball_yolo26m.pt\"\n"
        "HANDBALL_MODEL_PATH = panorama_weights if panorama_weights.is_file() else standard_weights\n"
        "\n"
        "if not HANDBALL_MODEL_PATH.is_file():\n"
        "    raise FileNotFoundError(\n"
        "        f\"Missing handball model: {HANDBALL_MODEL_PATH}. \"\n"
        "        \"Train it with notebooks/train_handball_yolo.ipynb.\"\n"
        "    )\n"
        "detector = Detector(HANDBALL_MODEL_PATH)\n"
        "print(f\"Handball model: {HANDBALL_MODEL_PATH}\")\n"
        "\n"
        "# Tracking backend:\n"
        "#  - \"bytetrack\": motion only, fast. BEST on a wide broadcast shot\n"
        "#    (measured KPIs: fewer ID switches than BoT-SORT on this kind of source).\n"
        "#  - \"botsort\"  : appearance ReID. Only useful when players are large enough\n"
        "#    in the image (close shot); on a wide shot it degrades (noisy fingerprints).\n"
        "TRACKER = \"bytetrack\"\n"
        "print(f\"Tracker: {TRACKER}\")"
    ),
    md(
        "## 5. Quick test on a short clip\n"
        "\n"
        "Before the full pipeline, run a quick pass to validate the chain:\n"
        "- aggressive subsample (1 frame out of 5)\n"
        "- no radar video nor clips: only check that the chain holds\n"
        "\n"
        "On a 60 s clip at 25 fps: ~300 inferences instead of 1500."
    ),
    code(
        "from handball_ai.pipeline import process_match\n"
        "import time\n"
        "\n"
        "QUICK_TEST_DIR = f\"{OUTPUTS_DIR}/_quick_test\"\n"
        "\n"
        "t0 = time.time()\n"
        "test_result = process_match(\n"
        "    video_path=CLIP,\n"
        "    court_mapping=court_mapping,\n"
        "    output_dir=QUICK_TEST_DIR,\n"
        "    detector=detector,\n"
        "    subsample=5,\n"
        "    with_radar_video=False,\n"
        "    with_clips=False,\n"
        "    min_track_frames=10,  # drops short tracking fragments\n"
        "    tracker=TRACKER,\n"
        ")\n"
        "duration = time.time() - t0\n"
        "\n"
        "meta = test_result.metadata\n"
        "print(f\"\\nQuick test OK in {duration:.1f}s (tracker={meta['tracker']})\")\n"
        "print(f\"Total tracks: {meta['nb_players_before_filter']}\")\n"
        "print(f\"--- Tracking quality KPIs ---\")\n"
        "print(f\"ID switches (suspected)          : {meta['nb_id_switches']}\")\n"
        "print(f\"Estimated players (after joining): {meta['nb_estimated_players']}\")\n"
        "print(f\"Mean track length                : {meta['mean_track_length']} frames\")\n"
        "print(f\"Usable players (after filter)    : {meta['nb_players_stats']}\")\n"
        "print(f\"\\nStats (first 10 rows):\")\n"
        "print(test_result.player_stats.head(10))"
    ),
    md(
        "## 6. Full pipeline\n"
        "\n"
        "The pipeline processes one frame out of two with `subsample=2`."
    ),
    code(
        "import torch\n"
        "if torch.cuda.is_available():\n"
        "    torch.cuda.empty_cache()\n"
        "\n"
        "t0 = time.time()\n"
        "result = process_match(\n"
        "    video_path=CLIP,\n"
        "    court_mapping=court_mapping,\n"
        "    output_dir=OUTPUTS_DIR,\n"
        "    detector=detector,\n"
        "    subsample=2,\n"
        "    with_radar_video=True,\n"
        "    with_clips=True,\n"
        "    min_track_frames=10,  # drops short tracking fragments\n"
        "    tracker=TRACKER,\n"
        ")\n"
        "duration_min = (time.time() - t0) / 60\n"
        "\n"
        "meta = result.metadata\n"
        "print(f\"\\nFull pipeline done in {duration_min:.1f} min (tracker={meta['tracker']})\")\n"
        "\n"
        "print(\"\\n=== TRACKING QUALITY KPIs ===\")\n"
        "print(f\"Total tracks                     : {meta['nb_players_before_filter']}\")\n"
        "print(f\"ID switches (suspected)          : {meta['nb_id_switches']}\")\n"
        "print(f\"Estimated players (after joining): {meta['nb_estimated_players']}\")\n"
        "print(f\"Mean track length                : {meta['mean_track_length']} frames\")\n"
        "print(f\"Usable players (after filter)    : {meta['nb_players_stats']}\")\n"
        "\n"
        "print(\"\\n=== PLAYER STATS ===\")\n"
        "print(result.player_stats)\n"
        "\n"
        "print(f\"\\n=== DETECTED ACTIONS: {len(result.detected_actions)} ===\")\n"
        "for i, action in enumerate(result.detected_actions):\n"
        "    print(f\"  Action {i+1}: frames {action.start_frame}-{action.end_frame} \"\n"
        "          f\"({action.duration_s:.1f}s, {action.mean_nb_players:.1f} players on average)\")\n"
        "\n"
        "print(f\"\\n=== ACTION CLIPS: {len(result.action_clips)} ===\")\n"
        "for clip in result.action_clips:\n"
        "    print(f\"  {clip}\")\n"
        "\n"
        "print(f\"\\n=== ARTEFACTS ===\")\n"
        "print(f\"  CSV  : {OUTPUTS_DIR}/player_stats.csv\")\n"
        "print(f\"  Parquet : {OUTPUTS_DIR}/player_stats.parquet\")\n"
        "print(f\"  Radar video : {result.radar_video}\")"
    ),
    md("## 7. Look at a few frames of the final rendering"),
    code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "radar_video = result.radar_video\n"
        "cap = cv2.VideoCapture(str(radar_video))\n"
        "total_radar = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
        "\n"
        "fig, axes = plt.subplots(5, 1, figsize=(20, 22))\n"
        "for ax, idx in zip(axes, [0, total_radar//4, total_radar//2, 3*total_radar//4, total_radar-1]):\n"
        "    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)\n"
        "    ok, frame = cap.read()\n"
        "    if ok:\n"
        "        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))\n"
        "        ax.set_title(f\"Frame {idx}/{total_radar}\")\n"
        "        ax.axis('off')\n"
        "cap.release()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 8. Player heatmap\n"
        "\n"
        "Shows the presence density of a player on the court. Defaults to the most present tracker."
    ),
    code(
        "import polars as pl\n"
        "import plotly.express as px\n"
        "from handball_ai.stats import generate_player_heatmap\n"
        "\n"
        "# Most present tracker in the clip\n"
        "sorted_stats = result.player_stats.sort(\"nb_detected_frames\", descending=True)\n"
        "chosen_tracker = int(sorted_stats[\"tracker_id\"][0])\n"
        "print(f\"Selected tracker: {chosen_tracker} \"\n"
        "      f\"({sorted_stats['nb_detected_frames'][0]} detected frames)\")\n"
        "print(sorted_stats.filter(pl.col(\"tracker_id\") == chosen_tracker))\n"
        "\n"
        "# Real heatmap from the court positions stored in PipelineResult\n"
        "player_positions = result.positions_by_tracker.get(chosen_tracker, [])\n"
        "heatmap = generate_player_heatmap(player_positions, resolution=0.5, smoothing_sigma=1.5)\n"
        "\n"
        "fig = px.imshow(\n"
        "    heatmap,\n"
        "    origin=\"lower\",\n"
        "    aspect=\"equal\",\n"
        "    color_continuous_scale=\"Hot\",\n"
        "    title=f\"Player presence heatmap, tracker_id={chosen_tracker}\",\n"
        "    labels={\"x\": \"court x (0.5 m cells)\", \"y\": \"court y (0.5 m cells)\"},\n"
        ")\n"
        "fig.update_layout(width=900, height=500)\n"
        "fig.show()"
    ),
    md(
        "## 9. Defensive block width over time\n"
        "\n"
        "Loads the defensive block width DataFrames (home and away) and plots them with plotly."
    ),
    code(
        "import polars as pl\n"
        "import plotly.express as px\n"
        "\n"
        "home_block_path = f\"{OUTPUTS_DIR}/defensive_block_width_home.csv\"\n"
        "away_block_path = f\"{OUTPUTS_DIR}/defensive_block_width_away.csv\"\n"
        "\n"
        "if os.path.exists(home_block_path):\n"
        "    df_home = pl.read_csv(home_block_path).with_columns(team=pl.lit(\"HOME\"))\n"
        "    df_away = pl.read_csv(away_block_path).with_columns(team=pl.lit(\"AWAY\"))\n"
        "    df = pl.concat([df_home, df_away])\n"
        "    fig = px.line(\n"
        "        df.to_pandas(),\n"
        "        x=\"time_s\", y=\"width_y_m\", color=\"team\",\n"
        "        title=\"Defensive block width over time\",\n"
        "        labels={\"time_s\": \"Time (s)\", \"width_y_m\": \"Width (m)\"},\n"
        "    )\n"
        "    fig.show()\n"
        "else:\n"
        "    print(\"No defensive_block_width_*.csv file: team classification may have failed (too few players?)\")"
    ),
]


def build_notebook() -> dict:
    """Assemble the notebook JSON structure."""
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
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
