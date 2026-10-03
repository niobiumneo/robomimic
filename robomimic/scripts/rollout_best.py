"""Evaluate a saved policy and keep successful videos plus all rollout paths.

Example (run from the repository root):
    python -m robomimic.scripts.rollout_best \
        --run-dir trained_models/square_cami_continuous --n-rollouts 50

Selection uses the trainer's *_success_<rate>.pth filenames. Checkpoint
variable_state.best_success_rate is a historical maximum, not necessarily
the performance of the weights in that file, so it must not rank last.pth.

Every successful rollout gets its own video (--keep-failures adds the failed
ones), and the clips of each outcome are also joined, in rollout order, into
SUCCESSFUL_ALL.mp4 and FAILED_ALL.mp4 with a CSV saying which rollout plays when.
"""
import argparse
import csv
import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
# select_checkpoint lives in the standard-library trial_metrics module so the
# train_trials launcher can choose the checkpoint without importing torch.
from robomimic.utils.trial_metrics import describe, seed_environment, select_checkpoint

# Outcomes whose clips are joined into one video, in rollout order. A failure
# is a rollout that ended without success or with a simulator error.
COMPILATIONS = (("SUCCESSFUL_ALL", ("successful",)), ("FAILED_ALL", ("failed", "errors")))
INDEX_FIELDS = ("clip", "rollout", "seed", "status", "file", "start_seconds", "duration_seconds")


def collect_episode(policy, env, horizon, camera_names, video_skip, terminate_on_success=True):
    """Collect aligned pre-action states, post-action states, and video frames.

    Reset recurrent policy state for each independent rollout. Success uses the
    environment's task criterion and stops the rollout at its first success,
    matching the Square training template. No force injection is required.
    """
    policy.start_episode()
    obs = env.reset()
    initial_state = env.get_state()
    state = initial_state
    trajectory = {k: [] for k in ("states", "next_states", "actions", "rewards", "dones")}
    frames = []
    success = False
    success_metrics = {}
    error = None
    goal = env.get_goal() if getattr(policy.policy.global_config, "use_goals", False) else None
    try:
        for step in range(horizon):
            with torch.no_grad():
                action = policy(ob=obs, goal=goal)
            obs, reward, done, _ = env.step(action)
            next_state = env.get_state()
            for key, value in (("states", state["states"]),
                               ("next_states", next_state["states"]),
                               ("actions", action), ("rewards", reward), ("dones", done)):
                trajectory[key].append(np.array(value, copy=True))
            current = env.is_success()
            for key, value in current.items():
                success_metrics[key] = success_metrics.get(key, False) or bool(value)
            success = success_metrics["task"]
            # Always include the success / terminal frame even when skipped.
            if step % video_skip == 0 or success or done or step == horizon - 1:
                frames.append(np.concatenate([
                    env.render(mode="rgb_array", height=512, width=512, camera_name=camera)
                    for camera in camera_names
                ], axis=1))
            state = next_state
            if done or (success and terminate_on_success):
                break
    except env.rollout_exceptions as exc:
        # Record simulator errors separately; they remain in the denominator.
        error = "{}: {}".format(type(exc).__name__, exc)
        success = False
    stats = {"Return": float(np.sum(trajectory["rewards"])),
             "Horizon": len(trajectory["actions"]), "Success_Rate": float(success),
             "error": error}
    stats.update({key + "_Success_Rate": float(value)
                  for key, value in success_metrics.items() if key != "task"})
    return stats, {k: np.asarray(v) for k, v in trajectory.items()}, initial_state, frames


def save_trajectory(data, name, trajectory, initial_state, stats, seed):
    """Store both successful and unsuccessful paths in robomimic HDF5 format."""
    group = data.create_group(name)
    for key, values in trajectory.items():
        group.create_dataset(key, data=values)
    group.attrs["num_samples"] = stats["Horizon"]
    group.attrs["success"] = bool(stats["Success_Rate"])
    group.attrs["seed"] = seed
    group.attrs["return"] = stats["Return"]
    if stats["error"]:
        group.attrs["rollout_error"] = stats["error"]
    if "model" in initial_state:
        group.attrs["model_file"] = initial_state["model"]
    if "ep_meta" in initial_state:
        group.attrs["ep_meta"] = initial_state["ep_meta"]


def write_summary(output, summary):
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def write_video(path, frames, fps):
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def join_clips(clips, target, frames):
    """Copy already encoded clips, one after another, into one video without re-encoding.

    The clips come from one writer with one set of settings, which is what lets
    ffmpeg's concat demuxer copy them as they are: no second round of
    compression, and the result plays exactly the frames of the clips.

    ffmpeg exits normally after skipping a clip it cannot open, so a join only
    counts when the video holds all @frames frames; otherwise no video is left
    behind.
    """
    import imageio_ffmpeg
    listing = target.with_name(target.name + ".txt")
    # A quote inside a quoted name is written '\'' in ffmpeg's list format.
    listing.write_text("".join("file '{}'\n".format(str(clip).replace("'", "'\\''")) for clip in clips),
                       encoding="utf-8")
    messages = ""
    try:
        result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat",
                                 "-safe", "0", "-i", str(listing), "-c", "copy", str(target)],
                                capture_output=True, text=True)
        messages = result.stderr.strip()
        if result.returncode != 0:
            raise RuntimeError("exit status {}".format(result.returncode))
        joined, _ = imageio_ffmpeg.count_frames_and_secs(str(target))
        if joined != frames:
            raise RuntimeError("{} has {} frames, expected {}".format(target.name, joined, frames))
    except BaseException as exc:
        target.unlink(missing_ok=True)
        if not isinstance(exc, Exception):
            raise
        raise RuntimeError("ffmpeg could not join the clips: {}{}".format(
            exc, "\n" + messages if messages else "")) from exc
    finally:
        listing.unlink()


def stitch_videos(output, records, fps):
    """Join the clips of each outcome into one video, plus a CSV of what plays when.

    Returns {name: details} for the compilations that were written; an outcome
    with no clip has none.
    """
    made = {}
    for name, statuses in COMPILATIONS:
        clips = [record for record in records if record["video"] and record["status"] in statuses]
        if not clips:
            continue
        join_clips([output / record["video"] for record in clips], output / (name + ".mp4"),
                   sum(record["video_frames"] for record in clips))
        rows, frames = [], 0
        for number, record in enumerate(clips, 1):
            rows.append({"clip": number, "rollout": record["rollout"], "seed": record["seed"],
                         "status": record["status"], "file": record["video"],
                         "start_seconds": round(frames / fps, 3),
                         "duration_seconds": round(record["video_frames"] / fps, 3)})
            frames += record["video_frames"]
        with (output / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(INDEX_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        made[name] = {"video": name + ".mp4", "index": name + ".csv", "clips": len(clips),
                      "frames": frames, "seconds": round(frames / fps, 3)}
    return made


def playback_speed(args, ckpt):
    """How much faster than the simulation a video plays, or None if the control rate is unknown."""
    control_freq = (ckpt["env_metadata"].get("env_kwargs") or {}).get("control_freq")
    if not control_freq:
        return None, None
    return args.fps * args.video_skip / control_freq, control_freq


def run(args):
    """Restore the policy once and run a fixed, reproducible rollout budget.

    One call is one evaluation trial: --n-rollouts episodes of one checkpoint.
    The train_trials launcher repeats it with disjoint seed ranges.
    """
    selection = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    else:
        checkpoint, selection = select_checkpoint(args.run_dir, args.metric_key)
    device = TorchUtils.get_torch_device(try_to_use_cuda=not args.cpu)
    # Load on CPU first, avoiding optimizer tensors occupying the GPU.
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt)
    horizon = args.horizon or config.experiment.rollout.horizon
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Rollout horizon must be a positive integer")
    if isinstance(ckpt["env_metadata"], list):
        raise ValueError("This helper requires a single-environment checkpoint")
    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
              checkpoint.parent.parent / "successful_rollouts" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    # A fresh output directory prevents overwriting earlier evidence.
    output.mkdir(parents=True, exist_ok=False)
    print("Checkpoint: {}".format(checkpoint), flush=True)
    if selection:
        print("Training evaluation success: {:.1%} | epoch {}".format(
            selection["training_success_rate"], selection["epoch"]), flush=True)
    print("Device: {} | output: {}".format(device, output), flush=True)
    speed, control_freq = playback_speed(args, ckpt)
    print("Videos: every {} control step(s) at {:g} fps{}".format(
        args.video_skip, args.fps,
        "" if speed is None else " = {:.3g}x real time ({:g} Hz control{})".format(
            speed, control_freq, "" if abs(speed - 1) < 1e-9 else
            "; --fps {:g} plays in real time".format(control_freq / args.video_skip))), flush=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt, device=device, verbose=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt, render=False,
                                          render_offscreen=True, verbose=False)
    records = []
    masks = {"successful": [], "failed": [], "errors": []}
    summary = {"checkpoint": str(checkpoint), "selection": selection,
               "n_rollouts_requested": args.n_rollouts, "horizon": horizon,
               "seed": args.seed, "camera_names": args.camera_names,
               "video_skip": args.video_skip, "fps": args.fps, "compilations": {},
               "episodes": records}
    try:
        with h5py.File(output / "rollouts.hdf5", "x") as dataset:
            data = dataset.create_group("data")
            data.attrs["env_args"] = json.dumps(env.serialize())
            data.attrs["checkpoint"] = str(checkpoint)
            data.attrs["total"] = 0
            for index in range(args.n_rollouts):
                seed = args.seed + index
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                seed_environment(env, seed)
                started = time.time()
                stats, trajectory, initial, frames = collect_episode(
                    policy, env, horizon, args.camera_names, args.video_skip,
                    terminate_on_success=getattr(config.experiment.rollout, "terminate_on_success", True))
                stats["time"] = time.time() - started
                name = "demo_{}".format(index)
                save_trajectory(data, name, trajectory, initial, stats, seed)
                data.attrs.modify("total", int(data.attrs["total"]) + stats["Horizon"])
                status = "errors" if stats["error"] else (
                    "successful" if stats["Success_Rate"] else "failed")
                masks[status].append(name)
                video = None
                if frames and (stats["Success_Rate"] or args.keep_failures):
                    video = "{}_rollout_{:03d}_seed_{}.mp4".format(status, index, seed)
                    # Encode after classification: unsuccessful frames need not
                    # create temporary videos that would then require deleting.
                    write_video(output / video, frames, args.fps)
                records.append(dict(stats, rollout=index, seed=seed, status=status,
                                    trajectory=name, video=video,
                                    video_frames=len(frames) if video else 0))
                dataset.flush()
                summary.update(n_rollouts_completed=len(records),
                               num_success=len(masks["successful"]),
                               num_errors=len(masks["errors"]),
                               success_rate=len(masks["successful"]) / len(records))
                metric_keys = {key for record in records for key in record
                               if key in ("Return", "Horizon", "Success_Rate", "time")
                               or key.endswith("_Success_Rate")}
                summary["metrics"] = {key: describe([record.get(key) for record in records])["mean"]
                                      for key in sorted(metric_keys)}
                summary["metrics"]["Simulator_Error_Rate"] = len(masks["errors"]) / len(records)
                write_summary(output, summary)
                print("Rollout {}/{} | {} | steps={} | success so far={:.1%}".format(
                    index + 1, args.n_rollouts, status, stats["Horizon"],
                    summary["success_rate"]), flush=True)
                del frames
            for key, names in masks.items():
                dataset.create_dataset("mask/" + key, data=np.asarray(names, dtype="S"))
    finally:
        # EnvBase has no close method; robosuite's wrapped simulator does.
        base = getattr(env, "unwrapped", env)
        simulator = getattr(base, "env", base)
        close = getattr(simulator, "close", None)
        if callable(close):
            close()
    print("Success: {}/{} ({:.1%}); simulator errors: {}".format(
        summary["num_success"], len(records), summary["success_rate"],
        summary["num_errors"]), flush=True)
    if not args.no_stitch:
        try:
            summary["compilations"] = stitch_videos(output, records, args.fps)
        except Exception as exc:
            # The individual videos are already safe; the evaluation itself must not fail over this.
            print("Could not join the videos ({}: {}). The individual videos are unaffected.".format(
                type(exc).__name__, exc), flush=True)
        write_summary(output, summary)
        for name, details in summary["compilations"].items():
            print("{}: {} clips, {:.1f} s at {:g} fps -> {}".format(
                name, details["clips"], details["seconds"], args.fps, output / details["video"]), flush=True)
    if not args.keep_failures and (masks["failed"] or masks["errors"]):
        print("{} unsuccessful rollouts have no video; add --keep-failures to save them{}.".format(
            len(masks["failed"]) + len(masks["errors"]),
            "" if args.no_stitch else " (and join them into FAILED_ALL.mp4)"), flush=True)
    print("Videos, all trajectories, and summary saved in {}".format(output), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Timestamped run, models folder, or experiment folder (latest run)")
    source.add_argument("--checkpoint", help="Explicit .pth file instead of automatic best selection")
    parser.add_argument("--metric-key", help="Dataset metric suffix when a run has multiple metrics")
    parser.add_argument("--output-dir", help="New output directory; must not exist")
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--horizon", type=int, help="Default: checkpoint rollout horizon")
    parser.add_argument("--seed", type=int, default=10000, help="Rollout i uses seed+i")
    parser.add_argument("--camera-names", nargs="+", default=["agentview"])
    parser.add_argument("--video-skip", type=int, default=1)
    parser.add_argument("--fps", type=float, default=20,
                        help="Playback rate of the videos. control_freq/video_skip plays in real time; "
                             "a higher rate plays faster than the simulation")
    parser.add_argument("--keep-failures", action="store_true",
                        help="Also save videos of failures/errors, and join them into FAILED_ALL.mp4")
    parser.add_argument("--no-stitch", action="store_true",
                        help="Do not also join the clips into SUCCESSFUL_ALL.mp4 and FAILED_ALL.mp4")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.n_rollouts < 1 or args.video_skip < 1 or args.fps <= 0 or args.seed < 0:
        parser.error("n-rollouts, video-skip and fps must be positive; seed must be nonnegative")
    if args.horizon is not None and args.horizon < 1:
        parser.error("horizon must be positive")
    run(args)


if __name__ == "__main__":
    main()
