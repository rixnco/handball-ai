# handball-ai

Computer vision pipeline for handball scouting: detection, tracking, per-player stats, automatic split into actions.

## Architecture

```
handball-ai/
├── handball_ai/             # Main package
│   ├── detection.py         # Local YOLO detection (GPU inference)
│   ├── tracking.py          # Ultralytics ByteTrack / BoT-SORT + class stabilization
│   ├── homography.py        # Pixel -> court coordinates projection
│   ├── teams.py             # Team classification with KMeans
│   ├── radar.py             # 2D radar rendering
│   ├── stats.py             # Player stats (distance, speed, heatmap)
│   ├── metrics.py           # Tracking quality KPIs (ID switches)
│   ├── clipping.py          # Automatic split into actions
│   ├── pipeline.py          # End-to-end orchestrator
│   ├── config.py            # Constants and court configuration
│   └── cli.py               # `handball-ai` CLI
├── notebooks/
│   ├── handball_ai.ipynb          # Local inference
│   └── train_handball_yolo.ipynb  # Local training
├── scripts/
│   ├── build_notebook.py          # Generates the inference notebook from Python
│   ├── build_train_notebook.py    # Generates the training notebook from Python
│   ├── calibrate_court.py         # Court mapping editor (Tkinter)
│   ├── extract_frames.py          # Random frame extraction for annotation
│   └── copy_yolo_labels.sh        # Copies the train labels to test/valid
├── tests/                   # Unit tests
├── docs/
│   └── court_mapping_example.json  # 10-point court mapping template
├── legacy/                  # Original monolithic code (reference)
├── data/                    # Videos, caches, outputs (ignored by git)
└── pyproject.toml
```

## Handball detection model

The pipeline requires a YOLO model **trained on handball**, which tells players, goalkeepers,
referees and ball apart. There is no generic default model. To train it:

1. Open `notebooks/train_handball_yolo.ipynb` with the project Python kernel under WSL and a CUDA GPU.
2. Pick an export listed in the [dataset provenance](data/datasets/README.md).
   The notebook looks for it in `data/datasets/`. To download a missing export, set
   a Roboflow key in the `ROBOFLOW_API_KEY` environment variable. Install the SDK if needed with
   `python -m pip install roboflow`.
3. The notebook trains YOLO26m (`YOLO_BASE` variable), shows the metrics and exports the best weights to
   `models/handball_yolo26m.pt`.
4. Use the path printed at export time in the "Detection model" cell of the inference
   notebook.

The detector maps the model classes to `HANDBALL_CLASSES` **by name** (not by order), so the
class order of the training dataset does not matter. A model without any handball class
is rejected at load time.

From Python, give the model explicitly:

```python
from handball_ai.detection import Detector
from handball_ai.pipeline import process_match

detector = Detector("models/handball_yolo26m.pt", confidence=0.35, exclude=["ball"])
process_match(..., detector=detector)
```

### Extract frames to annotate with FFmpeg

From the repository root, create the default destination folder.

```bash
mkdir -p data/annotation/images
```

**Sample a video every two seconds**

```bash
start_index=1
ffmpeg -n -i data/raw_clips/my_match.mp4 -vf "fps=1/2" -q:v 2 \
  -start_number "$start_index" \
  data/annotation/images/my_match_%05d.jpg
```

`fps=1/2` extracts one frame every two seconds. Use `fps=1/5` to space frames
by five seconds. `%05d` numbers the frames on five digits.
To resume after `my_match_00060.jpg`, set `start_index=61`. This value
is passed to the FFmpeg `-start_number` option and sets the number of the first JPEG.

**Extract frames over a 30-second passage**

```bash
start_index=1
ffmpeg -n -ss 00:12:30 -i data/raw_clips/my_match.mp4 -t 30 \
  -vf "fps=2" -q:v 2 -start_number "$start_index" \
  data/annotation/images/my_match_12m30_%05d.jpg
```

**Extract a single frame at a precise instant**

```bash
ffmpeg -n -ss 00:12:42.500 -i data/raw_clips/my_match.mp4 -frames:v 1 \
  -q:v 2 data/annotation/images/my_match_12m42s500.jpg
```

Frames keep the video resolution. `-q:v 2` produces high quality JPEGs and
`-n` protects existing files. Change the prefix for each video or passage to
keep distinct names.

`scripts/extract_frames.py` picks frames at random over the whole video instead:

```bash
python scripts/extract_frames.py -input data/raw_clips/my_match.mp4 -count 60 \
  -output data/annotation/images -prefix my_match
```

### Cut a match video into excerpts

**Extract 30 seconds from 12 min 30 s**

```bash
ffmpeg -n -ss 00:12:30 -i data/raw_clips/my_full_match.mp4 -t 30 \
  -map 0:v:0 -map '0:a:0?' -c:v libx264 -crf 18 -c:a aac \
  data/raw_clips/my_match_12m30_30s.mp4
```

Replace `00:12:30` with the start instant and `30` with the wanted duration in
seconds. The resolution is kept. Audio is included when the video has some.

**Cut the whole match into successive 30-second excerpts**

```bash
duration=30
folder="data/raw_clips/my_match_${duration}s"
mkdir "$folder" && ffmpeg -i data/raw_clips/my_full_match.mp4 \
  -map 0:v:0 -map '0:a:0?' -c:v libx264 -crf 18 -c:a aac \
  -force_key_frames "expr:gte(t,n_forced*${duration})" \
  -f segment -segment_time "$duration" -reset_timestamps 1 \
  "$folder/excerpt_%03d.mp4"
```

Change `duration` to choose the excerpt length. The last one may be shorter.
Cutting only starts when the destination folder is created, which avoids
overwriting existing excerpts.

## Local installation under WSL

In Ubuntu (WSL), from the repository root, use Python 3.11 and FFmpeg.
With Python 3.11 already installed, create the environment with `python3.11 -m venv .venv`.
With [uv](https://docs.astral.sh/uv/getting-started/installation/), you can install
Python 3.11 and create the environment in one command.

```bash
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cu132 -c .devcontainer/constraints.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The last command must print `True` to use the NVIDIA GPU.

### Run the notebooks

Run `jupyter notebook` from the repository with the `venv` activated. In VS Code,
open the folder from a WSL terminal with `code .` and select
`.venv/bin/python` as interpreter and Jupyter kernel.
To register a named kernel, run from the activated environment
`python -m ipykernel install --user --name handball-ai-wsl --display-name "Python (handball-ai WSL)"`.

In `notebooks/handball_ai.ipynb`, set the name of the clip placed in
`data/raw_clips/`. The notebook loads the court mapping JSON next to the video.
Results are written to `data/outputs/`.

### Run the pipeline from the CLI

```bash
handball-ai process \
  --video data/raw_clips/my_clip.mp4 \
  --output data/outputs/ \
  --model models/handball_yolo26m.pt \
  --subsample 2
```

Options:
- `--model models/handball_yolo26m.pt` (required): YOLO weights trained on handball.
- `--start HH:MM:SS:cc` sets where to start in the video, with `cc` in hundredths of a second (`00:00:00:00` by default). The FFmpeg notation `HH:MM:SS.cc` is accepted too.
- `--duration SECONDS` limits the processing to this duration from the chosen start. Without it, processing goes to the end of the video.
- `--court-mapping PATH`: replaces the court mapping path derived from the video.
- `--exclude CLASS` ignores a model class by name (repeatable), e.g. `--exclude ball`.
- `--conf THRESHOLD`: YOLO confidence threshold between 0 and 1 (0.35 by default).
- `--no-radar-video` disables the output video. By default it stacks the annotated source above the projection. Each panel is 1920 × 540 (32:9).
- `--no-fisheye` disables the radial correction of the radar. By default, when A to F are present, it uses A–E–B and C–F–D and maps the four corners onto a 40 × 20 m rectangle. The source video is left untouched.
- `--teams` enables team classification by jersey colour.
- `--stats` enables the computation and export of the statistics. Combine both options to also export the defensive block widths.
- `--no-clips`: does not cut the detected actions.
- `--tracker {bytetrack,botsort}`: tracking backend. `bytetrack` (default,
  fast, motion only) or `botsort` (appearance ReID: keeps the IDs on
  contact/occlusion, slower). Both run through Ultralytics `model.track` with
  the configurations of `handball_ai/trackers/`.
- `--min-track-frames N`: drops the trackers seen on fewer than N frames (fragments).

For example, add `--start 00:01:00:25 --duration 30` to analyse 30 seconds from 1 min and 25 hundredths. The pipeline seeks to the closest frame in the source video and processes the sequence directly. Result timestamps start from zero.

Exit codes:
- `0`: success
- `1`: runtime error (invalid video, court mapping or model, pipeline exception)
- `2`: missing required argument

Outputs in the `--output` folder:
- `player_stats.csv` / `player_stats.parquet` (with `--stats`)
- `defensive_block_width_home.csv` / `defensive_block_width_away.csv` (with `--stats --teams`)
- `match_radar.mp4` (unless `--no-radar-video`)
- `actions/action_XXX_Ns.mp4` (unless `--no-clips`, needs `ffmpeg`)

### Create the court mapping with the mouse

From the repository root, in an environment with a graphical display and Tkinter
(`python3-tk` on Debian or Ubuntu), run:

```bash
python -m scripts.calibrate_court data/raw_clips/my_clip.mp4
```

The script shows the middle frame. Use `--frame 250` to pick another
frame. To calibrate from a still image, pass the image as input and add
`--reference-video path/to/match.mp4`. Click a reference point on the diagram below the image,
then click its location in the image or in the grey margin. The wheel
zooms around the pointer. Drag with the right button to move the view,
or click **Fit image** to go back to the overview.
By default, the margin adds 10 %
of the width on the left and right, and 10 % of the height at the top and bottom.
Use `--margin 150` to set all four sides to 150 pixels. JSON coordinates
stay relative to the original image and may therefore exceed its frame.
Place at least four reference points spread over the court,
then click **Save**. For `data/raw_clips/my_clip.mp4`, the script writes
`data/raw_clips/my_clip.mp4.court_mapping.json`. The notebook and the CLI find
this file automatically.
If it already exists, give another destination with `--output` or use `--force`
to replace it.

### Court mapping format

JSON, at least 4 points (>= 5 enables RANSAC). See `docs/court_mapping_example.json` for a template.

```json
{
  "A": {"pixel": [120, 540], "court_m": [0.0, 0.0]},
  "B": {"pixel": [1800, 540], "court_m": [40.0, 0.0]},
  "C": {"pixel": [1700, 300], "court_m": [40.0, 20.0]},
  "D": {"pixel": [220, 300], "court_m": [0.0, 20.0]}
}
```

Keys starting with `_` are treated as comments (handy to document the JSON inline).

## Tests

```bash
pytest tests/ -v --cov=handball_ai
```

Tests that need `ffmpeg` (3 video clipping tests) are skipped when `ffmpeg` is not in the PATH.

## Code conventions

- Python 3.11+, type hints required on public signatures
- Code, docstrings, comments, logs and documentation in **English**, snake_case names
- Short Google-style docstrings
- Logging instead of print
- No bare except
- pytest tests for every critical function
- Ruff enabled with `select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]`

## Roadmap

- [x] Repository skeleton
- [x] Refactor of the legacy code into modules
- [x] Migration from Roboflow Cloud to local GPU inference (`Detector`)
- [x] Player stats module (distance, speed, heatmap, defensive block)
- [x] Action split with a simple heuristic
- [x] Full match pipeline (with interpolation and radar video)
- [x] `handball-ai process` CLI
- [x] Local inference notebook
- [x] Fine-tuned handball YOLO model (4 classes) + training notebook
- [x] Optional BoT-SORT + ReID tracker (keeps the IDs on contact)
- [x] Single tracking path through Ultralytics trackers (no `sv.ByteTrack`)
- [ ] Ball detection -> possession and passes
- [ ] Event detection (shots, goals) with heuristics
- [ ] Automatic detection of the attacking team (direction of play)
- [ ] Automatic court line detection (automatic homography)
- [ ] Individual player identification (jersey numbers)
- [ ] ML event classification

## Licence

Proprietary. All rights reserved.
