# Pipeline refactoring plan, starting with tracking

Branch: `refactor/tracking`. Same approach as `refactor/detection`: one minimal
code path, then the pipeline stages are added back one at a time.

## Starting point

- Two tracking paths with different shapes:
  - `track_detections` runs `sv.ByteTrack` on precomputed detections.
  - `track_video_botsort` detects and tracks in a single pass with `model.track`.
  - `pipeline.py` branches between the two, with different parameters.
- `sv.ByteTrack` is deprecated since supervision 0.28 and will be removed in 0.30.
- BoT-SORT state leaks between videos. `Detector.track` calls `model.track(frame, persist=True)`
  frame by frame, so Ultralytics never sees a new video path and never resets
  the tracker (`ultralytics/trackers/track.py`, `on_predict_start`).
- `Detector.track` passes the detection confidence (0.35) to `model.track`.
  The ByteTrack second association stage uses detections between
  `track_low_thresh` (0.1) and `track_high_thresh` (0.25), so it currently
  receives nothing. `Model.track` defaults to `conf=0.1` for this reason.
- `process_match` is around 250 lines with many `with_*` flags. Tracking alone
  cannot be run cleanly.

## Facts checked in Ultralytics 8.4.216

- `Model.predict` creates a new predictor when `model.predictor` is `None`,
  so resetting it gives fresh trackers. Ultralytics does the same internally.
- `BYTETracker` has no `frame_rate` parameter: `track_buffer` counts the
  frames actually fed to the tracker, so it scales with `subsample`.
- Both `bytetrack` and `botsort` read their settings from a YAML file
  (`tracker_type`, thresholds, `track_buffer`, ReID options).

## Steps

### Step 1: one tracking function

- `track_video(video, detector, *, tracker="bytetrack" | "botsort", subsample,
  start_frame, max_frames)` in `tracking.py`, going through `Detector.track`
  with the Ultralytics trackers in both cases.
- Tracker configurations shipped in `handball_ai/trackers/`:
  `bytetrack.yaml` (Ultralytics defaults) and `botsort_reid.yaml`.
- `Detector.reset_tracking()` sets `model.predictor = None`, called at the
  start of every video.
- `Detector.track` uses a dedicated tracking confidence (0.1 by default).
  The YAML thresholds decide which detections start or extend tracks.
- Class stabilization by majority vote is kept.
- Removed: `track_detections`, `track_video_botsort`, `sv.ByteTrack`, and the
  now unused `batch_size` of `process_match`.
- Tests: tracking stub detector (reset, subsample, start/max frames, majority
  class, unknown tracker), configuration files, `Detector.track` options with a
  fake YOLO model.

### Step 2: tracking-only pipeline

- `track_clip(video, detector, output_dir, ...)`: video -> tracking ->
  annotated video (boxes + IDs) and a summary (number of tracks, mean length).
- No homography, teams, statistics or clips at this stage.

### Step 3: CLI

- `handball-ai track <video> --model ...` to run the step 2 pipeline alone.

### Step 4: add the stages back

One commit and one test per stage: court projection, teams, statistics, radar
video, action clips.

## Validation at each step

- Unit tests with a stub detector, `pytest tests/ -v` green.
- Real run on a short clip from `data/raw_clips/` on the local GPU, with the
  numbers reported (tracks, frames, duration).
