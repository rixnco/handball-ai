# Legacy code

`pipeline_original.py` holds the original monolithic pipeline, kept as the specification reference for the refactor. It is left as it was written, in French.

**Do not run this file directly.** All the logic must be migrated into the `handball_ai/` modules.

Key points to keep during the refactor:
- Detection structure with `supervision.Detections`
- ByteTrack tracking with class stabilization by majority voting
- Team classification with KMeans on the torso colour
- 4-7 point homography with RANSAC fallback
- Asymmetric filter `[0,40]x[2,18]`
- Radar rendering with 6 m / 9 m lines and 9 m arcs

Points to REMOVE or replace:
- Dependency on `detect.roboflow.com` (cloud API) -> local inference with `Detector`
