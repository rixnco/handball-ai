#!/usr/bin/env python3
"""Extract JPEG frames randomly spread over the duration of a video."""

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def video_duration(video: Path) -> float:
    """Return the duration of the video in seconds, read with ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "V:0",
            "-show_entries", "stream=duration:format=duration", "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    )
    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    if not streams:
        raise ValueError("No video stream found.")
    for source in (streams[0], metadata.get("format", {})):
        value = source.get("duration")
        if value not in (None, "N/A"):
            duration = float(value)
            if math.isfinite(duration) and duration > 0:
                return duration
    raise ValueError("The video duration is missing from the metadata.")


def extract_frames(
    video: Path, count: int, target: Path, prefix: str, seed: int | None,
    progress: bool = True,
) -> None:
    """Write `count` frames, one picked at random in each equal slice of the video."""
    duration = video_duration(video)
    slice_width = duration / count
    rng = random.Random(seed)

    target.mkdir(parents=True, exist_ok=True)
    destinations = [target / f"{prefix}_{number:04d}.jpg" for number in range(1, count + 1)]
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"File already exists: {destination}")

    with TemporaryDirectory(dir=target) as temp:
        folder = Path(temp)
        sources = [folder / f"{number}.jpg" for number in range(1, count + 1)]
        if progress:
            print(f"Extracting frames ({duration:.1f} s video): 0/{count}",
                  file=sys.stderr, flush=True)
        for index, source in enumerate(sources):
            start = index * slice_width
            instant = start + rng.random() * slice_width
            # Near the end, the draw may fall after the last decodable frame.
            for position in (instant, start):
                subprocess.run(
                    [
                        "ffmpeg", "-v", "error", "-nostdin", "-ss", f"{position:.9f}",
                        "-i", str(video), "-map", "0:V:0", "-frames:v", "1", "-q:v", "2",
                        "-f", "image2", "-update", "1", str(source),
                    ],
                    check=True, capture_output=True, text=True,
                )
                if source.is_file():
                    break
            else:
                raise ValueError(f"No decodable frame near {instant:.3f} s.")
            if progress:
                print(f"\rExtracting frames: {index + 1}/{count}",
                      end="\n" if index + 1 == count else "",
                      file=sys.stderr, flush=True)

        published = []
        try:
            for source, destination in zip(sources, destinations, strict=True):
                destination.hardlink_to(source)
                published.append(destination)
        except OSError:
            for destination in published:
                destination.unlink()
            raise


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positional", nargs="*", metavar="ARG", help="video, count, target")
    parser.add_argument("-input", "--input", dest="video", type=Path, help="source video")
    parser.add_argument("-count", "--count", dest="count", type=int, help="number of frames")
    parser.add_argument("-output", "--output", dest="target", type=Path, help="target folder")
    parser.add_argument("-prefix", "--prefix", dest="prefix", default="image",
                        help="file name prefix")
    parser.add_argument("-seed", "--seed", type=int, help="seed to reproduce the selection")
    parser.add_argument("-no-progress", "--no-progress", action="store_true",
                        help="hide the progress indicator")
    args = parser.parse_args()
    params = (args.video, args.count, args.target)
    if args.positional:
        if len(args.positional) != 3 or any(value is not None for value in params):
            parser.error("give either video count target, or -input, -count and -output")
        video, count, target = args.positional
        args.video, args.target = Path(video), Path(target)
        try:
            args.count = int(count)
        except ValueError:
            parser.error("the number of frames must be an integer")
    elif any(value is None for value in params):
        parser.error("the -input, -count and -output options are required")
    if args.count < 1:
        parser.error("the number of frames must be greater than zero")
    if not args.prefix or "/" in args.prefix or "\\" in args.prefix:
        parser.error("the prefix must be a file name without path separator")

    try:
        extract_frames(
            args.video.resolve(), args.count, args.target.resolve(), args.prefix, args.seed,
            progress=not args.no_progress,
        )
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"{error.cmd[0]} error: {error.stderr.strip()}\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nExtraction interrupted.\n")

    print(f"{args.count} frames extracted to {args.target}")


if __name__ == "__main__":
    main()
