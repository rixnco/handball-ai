"""Create the homography court mapping by clicking on a frame of the match."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_image(source: Path, frame_index: int | None):
    """Read a still image or extract a frame from the video."""
    import cv2

    if frame_index is None:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is not None:
            return image, source.name

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open {source}.")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        index = (total // 2 if total > 0 else 0) if frame_index is None else frame_index
        if index < 0 or (total > 0 and index >= total):
            raise ValueError(f"Frame {index} outside the video ({total} frames).")
        if index and not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError(f"Cannot reach frame {index}.")
        ok, image = capture.read()
        if not ok:
            raise ValueError(f"Cannot read frame {index}.")
        return image, f"{source.name} · frame {index}"
    finally:
        capture.release()


def save(points: dict[str, tuple[int, int]], output: Path, force: bool) -> None:
    """Validate the points, then write the JSON expected by the pipeline."""
    import numpy as np

    from handball_ai.config import COURT_POINTS_M
    from handball_ai.homography import calibrate_homography

    if len(set(points.values())) != len(points):
        raise ValueError("Each reference point must match a distinct pixel.")
    court_mapping = {
        name: {"pixel": pixel, "court_m": COURT_POINTS_M[name]} for name, pixel in points.items()
    }
    matrix = calibrate_homography(court_mapping).matrix
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Degenerate points. Spread the reference points over the court.")

    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(court_mapping, indent=2, ensure_ascii=False) + "\n"
    with output.open("w" if force else "x", encoding="utf-8") as file:
        file.write(content)


def open_editor(image, title: str, output: Path, force: bool, margin: int | None = None) -> bool:
    """Match the reference points of the diagram with the pixels of the image."""
    try:
        import tkinter as tk
    except ImportError as error:
        raise ValueError("Tkinter is required for the editor. On Debian or Ubuntu, install python3-tk.") from error
    from tkinter import ttk

    from PIL import Image, ImageTk

    from handball_ai.config import COURT_POINTS_M

    height, width = image.shape[:2]
    margin_x = max(1, round(width * 0.1)) if margin is None else margin
    margin_y = max(1, round(height * 0.1)) if margin is None else margin
    rgb = Image.fromarray(image[:, :, ::-1])

    try:
        window = tk.Tk()
    except tk.TclError as error:
        raise ValueError(f"Cannot open the Tk editor. Check the graphical display ({error}).") from error
    window.title("Calibrate the court")
    window.geometry("1200x760")
    window.minsize(620, 460)
    window.columnconfigure(0, weight=1)
    window.rowconfigure(1, weight=1)

    ttk.Label(window, text=title, padding=(12, 6)).grid(row=0, sticky="w")
    photo = tk.Canvas(window, background="#252525", highlightthickness=0)
    photo.grid(row=1, sticky="nsew")
    ttk.Label(
        window, text="Wheel: zoom   ·   Right click + drag: move   ·   Left click: place",
        padding=(12, 4),
    ).grid(row=2, sticky="w")

    court = tk.Canvas(window, height=145, background="#f1f4f5", highlightthickness=0)
    court.grid(row=3, sticky="ew")
    controls = ttk.Frame(window, padding=(12, 8))
    controls.grid(row=4, sticky="ew")
    controls.columnconfigure(0, weight=1)
    status = tk.StringVar()
    ttk.Label(controls, textvariable=status, wraplength=440).grid(row=0, column=0, sticky="w")

    points: dict[str, tuple[int, int]] = {}
    history: list[tuple[str, tuple[int, int] | None]] = []
    selection = None
    saved = False
    scale = 1.0
    origin_x = origin_y = 0.0
    view_moved = False
    rendering = None
    anchor = None

    def show_photo() -> None:
        nonlocal rendering
        inverse = 1 / scale
        view = rgb.transform(
            (photo.winfo_width(), photo.winfo_height()),
            Image.Transform.AFFINE,
            (inverse, 0, -origin_x * inverse, 0, inverse, -origin_y * inverse),
            resample=Image.Resampling.BILINEAR if scale < 1 else Image.Resampling.NEAREST,
            fillcolor=(53, 53, 53),
        )
        rendering = ImageTk.PhotoImage(view, master=photo)
        photo.delete("all")
        photo.create_image(0, 0, image=rendering, anchor="nw")
        for name, (x, y) in points.items():
            px = origin_x + (x + 0.5) * scale
            py = origin_y + (y + 0.5) * scale
            photo.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#8bff7b", outline="black")
            photo.create_text(px + 8, py - 8, text=name, anchor="sw", fill="white")

    def fit(_event=None) -> None:
        nonlocal scale, origin_x, origin_y, view_moved
        w, h = photo.winfo_width(), photo.winfo_height()
        scale = min(w / (width + 2 * margin_x), h / (height + 2 * margin_y))
        origin_x = (w - width * scale) / 2
        origin_y = (h - height * scale) / 2
        view_moved = False
        show_photo()

    def resize(_event) -> None:
        if view_moved:
            show_photo()
        else:
            fit()

    def zoom(event) -> None:
        nonlocal scale, origin_x, origin_y, view_moved
        zoom_in = event.num == 4 or event.delta > 0
        factor = 1.25 if zoom_in else 0.8
        min_scale = min(
            photo.winfo_width() / (width + 2 * margin_x),
            photo.winfo_height() / (height + 2 * margin_y),
        ) / 2
        new_scale = max(min_scale, min(max(24.0, min_scale), scale * factor))
        ratio = new_scale / scale
        x, y = event.x + 0.5, event.y + 0.5
        origin_x = x - (x - origin_x) * ratio
        origin_y = y - (y - origin_y) * ratio
        scale = new_scale
        view_moved = True
        show_photo()

    def start_move(event) -> None:
        nonlocal anchor
        anchor = event.x, event.y, origin_x, origin_y

    def move(event) -> None:
        nonlocal origin_x, origin_y, view_moved
        if anchor is not None:
            x, y, ox, oy = anchor
            origin_x = ox + event.x - x
            origin_y = oy + event.y - y
            view_moved = True
            show_photo()

    def refresh(message: str = "") -> None:
        court.delete("all")
        center = court.winfo_width() / 2
        court_scale = min(5.5, (court.winfo_width() - 40) / 40)
        center_y = 72
        left = center - 20 * court_scale
        right = center + 20 * court_scale
        top = center_y - 10 * court_scale
        bottom = center_y + 10 * court_scale
        court.create_rectangle(left, top, right, bottom, fill="#dceced", outline="#466671")
        court.create_line(center, top, center, bottom, fill="#466671")
        for name, (x, y) in COURT_POINTS_M.items():
            px, py = left + x * court_scale, bottom - y * court_scale
            color = "#da7318" if name == selection else "#168044" if name in points else "#14628c"
            tag = f"point:{name}"
            court.create_oval(px - 5, py - 5, px + 5, py + 5, fill=color, outline="", tags=tag)
            if x == 0:
                court.create_text(px - 8, py, text=name, anchor="e", tags=tag)
            elif x == 40:
                court.create_text(px + 8, py, text=name, anchor="w", tags=tag)
            else:
                offset_y = 10 if y == 0 or name in {"K", "L"} else -10
                court.create_text(px, py + offset_y, text=name, tags=tag)
            court.tag_bind(tag, "<Button-1>", lambda _evt, name=name: choose(name))

        instruction = f"Place {selection} in the image or the margin." if selection else "Pick a reference point above."
        status.set(f"{len(points)} point(s), at least 4. {message or instruction}")

    def choose(name: str) -> None:
        nonlocal selection
        selection = name
        refresh()

    def place(event) -> None:
        nonlocal selection
        if selection is None:
            return
        x = math.floor((event.x + 0.5 - origin_x) / scale)
        y = math.floor((event.y + 0.5 - origin_y) / scale)
        if -margin_x <= x < width + margin_x and -margin_y <= y < height + margin_y:
            history.append((selection, points.get(selection)))
            points[selection] = (x, y)
            selection = None
            show_photo()
            refresh()

    def undo() -> None:
        nonlocal selection
        if history:
            selection, previous = history.pop()
            if previous is None:
                del points[selection]
            else:
                points[selection] = previous
            show_photo()
            refresh()

    def save_and_close() -> None:
        nonlocal saved
        try:
            save(points, output, force)
        except (ValueError, OSError) as error:
            refresh(str(error))
            return
        saved = True
        print(f"{len(points)} points saved to {output}")
        window.destroy()

    ttk.Button(controls, text="Fit image", command=fit).grid(row=0, column=1, padx=4)
    ttk.Button(controls, text="Undo point", command=undo).grid(row=0, column=2, padx=4)
    ttk.Button(controls, text="Save", command=save_and_close).grid(row=0, column=3, padx=4)
    photo.bind("<Configure>", resize)
    for signal in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        photo.bind(signal, zoom)
    for button in (2, 3):
        photo.bind(f"<ButtonPress-{button}>", start_move)
        photo.bind(f"<B{button}-Motion>", move)
    photo.bind("<Button-1>", place)
    court.bind("<Configure>", lambda _evt: refresh())
    window.mainloop()
    return saved


def main() -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="match video or calibration image")
    parser.add_argument("--frame", type=int, help="frame index, from 0 (default: middle)")
    parser.add_argument("--margin", type=int, help="clickable margin in pixels (default: 10 %% per axis)")
    parser.add_argument("--reference-video", type=Path, help="video to associate when the source is an image")
    parser.add_argument("--output", type=Path, help="JSON file to create (default: next to the video)")
    parser.add_argument("--force", action="store_true", help="allow replacing the JSON file")
    args = parser.parse_args()

    if args.margin is not None and args.margin < 0:
        parser.error("--margin must be positive or zero.")
    if not args.source.is_file():
        parser.error(f"File not found: {args.source}")
    if args.reference_video is not None and not args.reference_video.is_file():
        parser.error(f"Reference video not found: {args.reference_video}")

    from handball_ai.config import court_mapping_path

    video = args.reference_video or args.source
    output = args.output or court_mapping_path(video)
    if output.resolve() in {args.source.resolve(), video.resolve()}:
        parser.error("The JSON output must differ from the source file.")
    if output.exists() and not args.force:
        parser.error(f"{output} already exists. Use --force or choose another --output.")

    try:
        image, title = read_image(args.source, args.frame)
        if open_editor(image, title, output, args.force, margin=args.margin):
            return 0
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"{error}\n")

    print("No court mapping saved.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
