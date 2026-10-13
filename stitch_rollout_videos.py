#!/usr/bin/env python3
r"""Join previously recorded CaMI rollout videos across trials.

Example, from ~/robomimic after saving this file there:
    python stitch_rollout_videos.py \
        results/cami_best_eval_seed0_50_videos/CaMI/checkpoint_env/videos

Recursively recognizes successful_rollout_*.mp4, failed_rollout_*.mp4 and
errors_rollout_*.mp4 from run_trained_agent_multi_eval.py and rollout_best.py.
Existing compilations and unfinished rollout_*.mp4 files are ignored. Error
clips join the failure compilation. Files are ordered by trial, then rollout,
using natural path order (trial_2 before trial_10).

Creates SUCCESSFUL_ALL.mp4, FAILED_ALL.mp4 and a CSV index for each outcome in
INPUT/stitched by default. No video is created for an outcome with no clips.
Source clips are never changed. This does not rerun the policy or simulator.
CSV indexes distinguish trial seeds from rollout seeds when present in paths.

Uses FFmpeg stream copying to retain the original frames and playback speed.
Clips of a given outcome must have matching video settings. Only video is
included, which matches the silent robot rollout recordings. Install FFmpeg
(Ubuntu: sudo apt install ffmpeg), or use the imageio-ffmpeg package already
used by rollout_best.py (python -m pip install imageio-ffmpeg).

Options:
    --output-dir PATH   Select the compilation folder.
    --overwrite         Replace existing compilations and their CSV indexes.
    --dry-run           Show clip counts without creating any files.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


CLIP_NAME = re.compile(
    r"^(successful|failed|errors?)_rollout_(\d+)(?:_seed_(\d+))?\.mp4$",
    re.IGNORECASE,
)
OUTCOMES = (("success", "SUCCESSFUL_ALL"), ("failure", "FAILED_ALL"))
FIELDS = (
    "clip", "trial", "trial_seed", "rollout", "rollout_seed", "status", "source_video",
    "start_seconds", "end_seconds", "duration_seconds", "frames", "fps",
)


def natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", str(path)))


def collect_clips(root):
    groups = {outcome: [] for outcome, _ in OUTCOMES}
    for path in sorted(root.rglob("*"), key=natural_key):
        match = CLIP_NAME.fullmatch(path.name)
        if not path.is_file() or match is None:
            continue
        status, rollout, seed = match.groups()
        status = status.lower()
        trial = next((part for part in reversed(path.parts[:-1])
                      if re.fullmatch(r"trial_\d+(?:_seed_(?:\d+|None))?", part)), "")
        trial_seed = trial.rsplit("_seed_", 1)[1] if "_seed_" in trial else ""
        if trial_seed == "None":
            trial_seed = ""
        groups["success" if status == "successful" else "failure"].append({
            "path": path.resolve(), "trial": trial, "trial_seed": trial_seed,
            "rollout": int(rollout), "rollout_seed": "" if seed is None else int(seed),
            "status": status,
        })
    return groups


def run_tool(command):
    result = subprocess.run(command, capture_output=True, text=True)
    # At loglevel error, stderr also catches decode failures with a zero exit code.
    if result.returncode or result.stderr.strip():
        raise RuntimeError(result.stderr.strip() or "Command exited with status {}".format(
            result.returncode))
    return result.stdout


def find_tools():
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    imageio_ffmpeg = None
    if not ffmpeg or not ffprobe:
        try:
            import imageio_ffmpeg
        except ImportError:
            pass
    if not ffmpeg and imageio_ffmpeg is not None:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg or (not ffprobe and imageio_ffmpeg is None):
        raise RuntimeError("Install FFmpeg: sudo apt install ffmpeg\n"
                           "Or: python -m pip install imageio-ffmpeg")
    return ffmpeg, ffprobe, imageio_ffmpeg


def probe_video(path, ffprobe, imageio_ffmpeg):
    if ffprobe:
        data = json.loads(run_tool([
            ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate,"
            "r_frame_rate,time_base,nb_read_frames", "-of", "json", str(path),
        ]))
        streams = data.get("streams", [])
        if not streams:
            raise ValueError("No video stream: {}".format(path))
        stream = streams[0]
        frames = int(stream["nb_read_frames"])
        fps = Fraction(stream["avg_frame_rate"])
        signature = (stream["codec_name"], stream["width"], stream["height"],
                     stream["pix_fmt"], fps, stream["time_base"])
    else:
        # Works with imageio-ffmpeg's bundled binary when ffprobe is unavailable.
        reader = imageio_ffmpeg.read_frames(str(path))
        try:
            metadata = next(reader)
        finally:
            reader.close()
        frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
        fps = Fraction(str(metadata["fps"]))
        signature = (metadata["codec"], *metadata["size"], metadata["pix_fmt"], fps)
    if frames < 1 or fps <= 0:
        raise ValueError("Video has no usable frames or frame rate: {}".format(path))
    return frames, fps, signature


def concat_name(path):
    # FFmpeg concat uses its own quoting syntax, not shell quoting.
    name = str(path)
    if "\n" in name or "\r" in name:
        raise ValueError("Video paths cannot contain newline characters")
    return "'" + name.replace("'", "'\\''") + "'"


def stitch(records, stem, output, ffmpeg, ffprobe, imageio_ffmpeg):
    expected_frames, signature, elapsed = 0, None, Fraction(0)
    rows, listing = [], []
    for number, record in enumerate(records, 1):
        frames, fps, settings = probe_video(record["path"], ffprobe, imageio_ffmpeg)
        if signature is None:
            signature = settings
        elif settings != signature:
            raise ValueError("Video settings differ: {}\n"
                             "Stitch recordings with matching resolution, codec and fps together.".format(
                                 record["path"]))
        duration = Fraction(frames, 1) / fps
        rows.append({
            "clip": number, "trial": record["trial"], "trial_seed": record["trial_seed"],
            "rollout": record["rollout"], "rollout_seed": record["rollout_seed"],
            "status": record["status"],
            "source_video": str(record["path"]), "start_seconds": round(float(elapsed), 6),
            "end_seconds": round(float(elapsed + duration), 6),
            "duration_seconds": round(float(duration), 6), "frames": frames, "fps": float(fps),
        })
        listing += ["file {}\n".format(concat_name(record["path"])),
                    "duration {:.12f}\n".format(float(duration))]
        expected_frames += frames
        elapsed += duration

    # Write to temporary files, verify the full frame count, then publish the result.
    # A failed join leaves existing compilations intact, even with --overwrite.
    with tempfile.TemporaryDirectory(prefix=".stitch-", dir=str(output)) as work:
        work = Path(work)
        (work / "clips.txt").write_text("".join(listing), encoding="utf-8")
        target = work / (stem + ".mp4")
        run_tool([
            ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(work / "clips.txt"), "-map", "0:v:0", "-an", "-c:v", "copy",
            "-movflags", "+faststart", str(target),
        ])
        actual_frames, _, _ = probe_video(target, ffprobe, imageio_ffmpeg)
        if actual_frames != expected_frames:
            raise RuntimeError("Join lost frames: got {}, expected {}".format(
                actual_frames, expected_frames))
        index = work / (stem + ".csv")
        with index.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        target.replace(output / target.name)
        index.replace(output / index.name)
    print("{}: {} clips, {} frames, {:.2f} seconds".format(
        stem + ".mp4", len(records), expected_frames, float(elapsed)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Video folder or results root to scan recursively")
    parser.add_argument("--output-dir", type=Path, help="Default: INPUT/stitched")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.input_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error("Input directory does not exist: {}".format(root))
    groups = collect_clips(root)
    if not any(groups.values()):
        parser.error("No labeled rollout videos found in {}".format(root))
    for outcome, _ in OUTCOMES:
        print("{} clips: {}".format(outcome.capitalize(), len(groups[outcome])), flush=True)
    if args.dry_run:
        return
    output = (args.output_dir.expanduser() if args.output_dir else root / "stitched").resolve()
    targets = [output / (stem + suffix) for outcome, stem in OUTCOMES if groups[outcome]
               for suffix in (".mp4", ".csv")]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        parser.error("Output already exists: {}. Use --overwrite or a new --output-dir.".format(
            existing[0]))
    ffmpeg, ffprobe, imageio_ffmpeg = find_tools()
    output.mkdir(parents=True, exist_ok=True)
    for outcome, stem in OUTCOMES:
        if groups[outcome]:
            stitch(groups[outcome], stem, output, ffmpeg, ffprobe, imageio_ffmpeg)
        else:
            print("No {} clips saved; no new {}.mp4 created.".format(outcome, stem))
    print("Saved compilations and CSV indexes in {}".format(output), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, ZeroDivisionError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        sys.exit(1)
