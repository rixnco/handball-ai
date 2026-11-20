# CLAUDE.md - handball-ai project configuration

## Mission

Computer vision pipeline for handball scouting: detection, tracking, player stats, automatic split into actions. The repository was pre-scaffolded: folder structure, `pyproject.toml`, modules with signatures and docstrings, skeleton tests. The original legacy code is in `legacy/pipeline_original.py`.

## User context

Tristan: freelance data scientist, writes Python with strict conventions:
- **English** for everything: code, snake_case variables/functions (`process_clip`, `compute_distance_covered`), docstrings, comments, logs, CLI, notebooks and documentation
- Type hints required on public signatures
- Short Google-style docstrings
- Logging instead of print, no bare except
- No TODO left in the final code, no invented library APIs

## MANDATORY git workflow

- `main` branch always stable
- One branch per feature: `feat/local-inference`, `feat/player-stats`, etc.
- Atomic commits in English
- PR opened at the end of each feature

## Process

1. Read `legacy/pipeline_original.py` entirely BEFORE coding
2. Propose a detailed plan to Tristan in a few lines
3. Wait for his GO
4. Code one step at a time, commit, move on
5. At each step, show a concrete test proving that it works
6. Check the library imports and signatures (Ultralytics 8.3+, supervision 0.25+)

## Current module status

- `config.py`: COMPLETE - constants, CourtConfig, court reference points
- `detection.py`: COMPLETE - Detector (handball model required) + detect_video
- `tracking.py`: COMPLETE - ByteTrack / BoT-SORT + class stabilization
- `homography.py`: COMPLETE - calibrate_homography with RANSAC fallback and fisheye correction
- `teams.py`: COMPLETE - extract_torso_color + classify_teams
- `radar.py`: COMPLETE - draw_radar
- `stats.py`: COMPLETE - compute_player_stats, generate_player_heatmap, compute_defensive_block_width
- `metrics.py`: COMPLETE - tracking quality KPIs (ID switches)
- `clipping.py`: COMPLETE - detect_actions + cut_video_clips
- `pipeline.py`: COMPLETE - process_match orchestrator
- `cli.py`: COMPLETE - `handball-ai process`

## Sprint steps

### Step 1: Validate the existing code (1h)
Run `pytest tests/` and check that the implemented modules pass.
Find possible bugs in the COMPLETE modules and fix them.

### Step 2: Implement stats.py (2h)
- `compute_player_stats`: Polars DataFrame with all the specified columns
- `generate_player_heatmap`: 2D heatmap with optional Gaussian smoothing
- `compute_defensive_block_width`: time series DataFrame
- Tests: `tests/test_stats.py`

### Step 3: Implement clipping.py (2h)
- `detect_actions`: state machine on the number of players present
- `cut_video_clips`: ffmpeg calls through subprocess
- Tests: `tests/test_clipping.py` with a synthetic sequence

### Step 4: Implement pipeline.py (3h)
- `process_match`: chain all the modules
- Linear interpolation of the positions on non-sampled frames (visual rendering only)
- Radar video generation (broadcast above the radar)
- Stats saved as CSV and Parquet

### Step 5: Finalize cli.py (30 min)
- Load the court mapping from a JSON file
- Call `process_match`
- Output: summary message with the paths of the produced artefacts

### Step 6: Local end-to-end notebook (1h)
- Test on a short clip (3-10 s) first
- Validate on a 58 s clip

## Technical constraints

- Python 3.11+
- Adapt the batch size to the GPU memory.
- No arbitrary torch.load (check the checkpoints, avoid unsigned pickle)

## Out of scope (Tristan said NO)

- Automatic court line detection for an automatic homography -> later
- Individual player identification (numbers, names) -> later
- Multi-shot tracking with automatic re-registration -> later
- ML event classification model -> later
- SAM3 or any other experimental model -> later
- Web interface or app -> later

## Definition of done

The `notebooks/handball_ai.ipynb` notebook runs end to end locally:
- Input: MP4 clip in `data/raw_clips/` + court mapping
- Output: player stats DataFrame + action clips + radar video
- No manual step beyond entering the court mapping

And `pytest tests/ -v` is green.
