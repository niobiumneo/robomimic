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
rollouts.csv has one row per rollout and summary.json the totals, with a 95%
confidence interval on the success rate.

Weights & Biases is opt-in (--wandb, --wandb-project or --wandb-mode), as in
train.py. The run gets every rollout as it finishes, then the summary, a table of
the rollouts, the joined videos and an artifact with the result files. A W&B
failure prints one warning and never stops the evaluation. For the Square task,
--insertion-report also replays the saved rollouts and reports how the nut ended
up on the post (see inspect_square_insertions), and logs that too:
    python -m robomimic.scripts.rollout_best \
        --run-dir trained_models/square_cami_continuous --n-rollouts 50 \
        --keep-failures --insertion-report --output-dir square_rollouts_cami \
        --wandb-project cami-square-v3 --name square_rollouts_cami
"""
import argparse
import csv
import json
import math
import os
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
from robomimic.utils.trial_metrics import describe, resolve_wandb_mode, seed_environment, select_checkpoint

# Outcomes whose clips are joined into one video, in rollout order. A failure
# is a rollout that ended without success or with a simulator error.
COMPILATIONS = (("SUCCESSFUL_ALL", ("successful",)), ("FAILED_ALL", ("failed", "errors")))
INDEX_FIELDS = ("clip", "rollout", "seed", "status", "file", "start_seconds", "duration_seconds")
ROLLOUT_FIELDS = ("rollout", "seed", "status", "success", "return", "horizon", "seconds", "error", "video")
WANDB_VIDEOS = ("joined", "all", "none")


def wilson_interval(successes, n, z=1.959964):
    """95% confidence interval of a success rate from ``successes`` out of ``n`` (Wilson score), or (None, None)."""
    if n < 1:
        return None, None
    rate = successes / n
    scale = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / scale
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / scale
    # the rate is always inside its interval; rounding must not put it a hair outside at 0 or n successes
    return max(0., min(rate, centre - half)), min(1., max(rate, centre + half))


def rollout_row(record):
    return {"rollout": record["rollout"], "seed": record["seed"], "status": record["status"],
            "success": int(bool(record["Success_Rate"])), "return": record["Return"], "horizon": record["Horizon"],
            "seconds": round(record["time"], 3), "error": record["error"] or "", "video": record["video"] or ""}


def write_rollout_table(output, records):
    """rollouts.csv: one row per rollout, with the extra success metrics some tasks report."""
    extra = sorted({key for record in records for key in record if key.endswith("_Success_Rate") and key != "Success_Rate"})
    with (output / "rollouts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ROLLOUT_FIELDS) + extra)
        writer.writeheader()
        for record in records:
            writer.writerow(dict(rollout_row(record), **{key: record.get(key) for key in extra}))


def metric_keys(records):
    return sorted({key for record in records for key in record
                   if key in ("Return", "Horizon", "Success_Rate", "time") or key.endswith("_Success_Rate")})


def summary_values(records):
    """What the finished evaluation adds to a tracker's summary: counts, mean/SD/SE of each metric, a CI on success."""
    successes = sum(1 for record in records if record["status"] == "successful")
    errors = sum(1 for record in records if record["status"] == "errors")
    values = {"Summary/n_rollouts": len(records), "Summary/num_success": successes, "Summary/num_errors": errors,
              "Summary/Simulator_Error_Rate": errors / len(records) if records else None}
    for key in metric_keys(records):
        for stat, value in describe([record.get(key) for record in records]).items():
            if value is not None:
                values["Summary/{}/{}".format(key, stat)] = value
    low, high = wilson_interval(successes, len(records))
    if low is not None:
        values["Summary/Success_Rate/ci95_low"], values["Summary/Success_Rate/ci95_high"] = low, high
    return {key: value for key, value in values.items() if value is not None}


def print_results(summary, records, output, wandb_url=None):
    """The numbers of the evaluation, as they are also saved in summary.json and sent to W&B."""
    n, successes = len(records), summary["num_success"]
    low, high = wilson_interval(successes, n)
    print("\nResults over {} rollouts".format(n), flush=True)
    print("  success rate {:.1%} ({}/{}), 95% CI {:.1%} to {:.1%}; simulator errors {}".format(
        summary["success_rate"], successes, n, low, high, summary["num_errors"]), flush=True)
    for key, label in (("Return", "return"), ("Horizon", "steps"), ("time", "seconds")):
        stats = describe([record.get(key) for record in records])
        if stats["mean"] is not None:
            print("  {} per rollout: mean {:.3g}{}".format(label, stats["mean"], "" if stats["std"] is None else
                                                          ", SD {:.3g}".format(stats["std"])), flush=True)
    print("  saved in {}: summary.json, rollouts.csv, rollouts.hdf5{}".format(
        output, "".join(", " + name for name in ("insertions.csv", "insertions.json") if (output / name).exists())),
          flush=True)
    if wandb_url:
        print("  W&B: {}".format(wandb_url), flush=True)


class WandbTracker:
    """The W&B run that follows an evaluation.

    It opens before the first rollout, gets each rollout as it finishes (with the running success rate and its
    standard error), and at the end the summary, a table of the rollouts, the joined videos and the result files.
    Tracking is a convenience and must never cost an evaluation: if W&B fails, one warning is printed and the
    evaluation carries on without it.
    """

    def __init__(self, mode, project, entity, group, name, directory, config):
        self.run, self.wandb, self.mode, self.project = None, None, mode, project
        try:
            import wandb
            from robomimic import macros as Macros
            if Macros.WANDB_API_KEY is not None:
                os.environ.setdefault("WANDB_API_KEY", Macros.WANDB_API_KEY)
            entity = entity or os.environ.get("WANDB_ENTITY") or Macros.WANDB_ENTITY
            self.wandb = wandb
            self.run = wandb.init(project=project, entity=entity, group=group, job_type="rollouts", name=name,
                                  mode=mode, dir=str(directory), config=config)
            self.run.define_metric("rollout")
            self.run.define_metric("Rollout/*", step_metric="rollout")
            self.run.define_metric("Running/*", step_metric="rollout")
        except Exception as exc:
            self.give_up(exc)

    @property
    def url(self):
        return None if self.run is None else (getattr(self.run, "url", None) or getattr(self.run, "id", None))

    def info(self):
        if self.run is None:
            return None
        return {"project": self.project, "mode": self.mode, "run_id": getattr(self.run, "id", None),
                "url": getattr(self.run, "url", None)}

    def give_up(self, exc):
        print("W&B tracking stopped ({}: {}). The evaluation continues and everything is still saved in the "
              "output directory.".format(type(exc).__name__, exc), flush=True)
        self.close()

    def log_rollout(self, record, records):
        if self.run is None:
            return
        try:
            values = {"rollout": record["rollout"], "Rollout/Success": float(bool(record["Success_Rate"])),
                      "Rollout/Return": record["Return"], "Rollout/Horizon": record["Horizon"],
                      "Rollout/Seconds": record["time"], "Rollout/Simulator_Error": float(record["status"] == "errors")}
            for key in metric_keys([record]):
                if key.endswith("_Success_Rate") and key != "Success_Rate":
                    values["Rollout/" + key] = record[key]
            stats = describe([r["Success_Rate"] for r in records])
            values["Running/Success_Rate/mean"] = stats["mean"]
            if stats["se"] is not None:
                values["Running/Success_Rate/se"] = stats["se"]
            self.run.log(values)
        except Exception as exc:
            self.give_up(exc)

    def publish(self, summary, records, output, videos="joined", insertion=None):
        """Add the results of the finished evaluation: summary, table, videos, files and the insertion report."""
        if self.run is None:
            return
        try:
            wandb, run = self.wandb, self.run
            run.summary.update(summary_values(records))
            columns = [field for field in ROLLOUT_FIELDS if field != "video" or videos == "all"]
            rows = []
            for record in records:
                row = rollout_row(record)
                if videos == "all" and record["video"]:
                    row["video"] = wandb.Video(str(output / record["video"]), format="mp4")
                elif videos == "all":
                    row["video"] = None
                rows.append([row[field] for field in columns])
            run.log({"rollouts": wandb.Table(columns=columns, data=rows)})
            if videos != "none":
                for name, details in summary.get("compilations", {}).items():
                    run.log({"Video/" + name: wandb.Video(str(output / details["video"]), format="mp4")})
                if videos == "joined" and not summary.get("compilations") and any(record["video"] for record in records):
                    print("No joined videos to send to W&B (--no-stitch); add --wandb-videos all to send each clip.", flush=True)
            if insertion is not None:
                from robomimic.scripts import inspect_square_insertions as insertions
                report, report_summary = insertion
                run.summary.update(insertions.logged_values(report_summary))
                run.log({"insertions": wandb.Table(columns=list(insertions.TABLE_COLUMNS), data=insertions.table_rows(report))})
            artifact = wandb.Artifact("rollouts-" + str(run.id), type="rollouts")
            names = ["summary.json", "rollouts.csv", "insertions.csv", "insertions.json"]
            names += [details["index"] for details in summary.get("compilations", {}).values()]
            for name in names:
                if (output / name).exists():
                    artifact.add_file(str(output / name))
            run.log_artifact(artifact)
        except Exception as exc:
            self.give_up(exc)

    def close(self, failed=False):
        """Finish the run; a failed evaluation marks it failed."""
        run, self.run = self.run, None
        if run is not None:
            try:
                run.finish(exit_code=1 if failed else 0)
            except Exception as exc:
                print("Could not close the W&B run ({}: {})".format(type(exc).__name__, exc), flush=True)


class NoTracker:
    """Stands in for WandbTracker when W&B was not asked for."""
    url = None

    def info(self):
        return None

    def log_rollout(self, record, records):
        pass

    def publish(self, *args, **kwargs):
        pass

    def close(self, failed=False):
        pass


def insertion_report(output):
    """Replay the saved rollouts and report how the nut ended up on the post; None if that fails.

    Like the videos, this is a diagnostic that must not cost the evaluation, so a failure is printed.
    """
    try:
        from robomimic.scripts import inspect_square_insertions as insertions
        print("\nInsertion report (replaying the saved rollouts)", flush=True)
        report = insertions.build_report(str(output / "rollouts.hdf5"))
        summary = insertions.summarize(report)
        insertions.print_summary(summary)
        insertions.write_csv(output / "insertions.csv", report)
        with (output / "insertions.json").open("w", encoding="utf-8") as stream:
            json.dump(insertions.jsonable(dict(summary=summary, **report)), stream, indent=2, allow_nan=False)
        return report, summary
    except Exception as exc:
        print("Could not make the insertion report ({}: {}). The rollouts are saved and can be analyzed with "
              "python -m robomimic.scripts.inspect_square_insertions --dataset {}".format(
                  type(exc).__name__, exc, output / "rollouts.hdf5"), flush=True)
        return None


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
    wandb_mode = resolve_wandb_mode(args)
    use_wandb = wandb_mode is not None and wandb_mode != "disabled"
    project = None
    if use_wandb:
        project = getattr(args, "wandb_project", None) or getattr(
            getattr(config.experiment, "logging", None), "wandb_proj_name", None)
        if not project:
            raise ValueError("Set --wandb-project (or --wandb-mode disabled) to send the results to W&B")
    insertion_requested = bool(getattr(args, "insertion_report", False))
    env_name = ckpt["env_metadata"].get("env_name")
    if insertion_requested and env_name is not None and "NutAssemblySquare" not in env_name:
        raise ValueError("--insertion-report is for the Square task, but this checkpoint's environment is {}".format(env_name))
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
    tracker = NoTracker()
    if use_wandb:
        tracker = WandbTracker(
            wandb_mode, project, getattr(args, "wandb_entity", None), getattr(args, "group", None),
            getattr(args, "name", None) or output.name, output,
            {"checkpoint": str(checkpoint), "epoch": (selection or {}).get("epoch"),
             "training_success_rate": (selection or {}).get("training_success_rate"), "n_rollouts": args.n_rollouts,
             "horizon": horizon, "seed": args.seed, "camera_names": args.camera_names, "video_skip": args.video_skip,
             "fps": args.fps, "keep_failures": bool(args.keep_failures), "output_dir": str(output),
             "insertion_report": insertion_requested, "wandb_videos": getattr(args, "wandb_videos", "joined")})
    summary = {"checkpoint": str(checkpoint), "selection": selection, "wandb": tracker.info(),
               "n_rollouts_requested": args.n_rollouts, "horizon": horizon,
               "seed": args.seed, "camera_names": args.camera_names,
               "video_skip": args.video_skip, "fps": args.fps, "compilations": {},
               "episodes": records}
    completed = False
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
                tracker.log_rollout(records[-1], records)
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
        completed = True
    finally:
        if not completed:
            tracker.close(failed=True)
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
    write_rollout_table(output, records)
    insertion = insertion_report(output) if insertion_requested else None
    summary["success_rate_ci95"] = list(wilson_interval(summary["num_success"], len(records)))
    write_summary(output, summary)
    tracker.publish(summary, records, output, videos=getattr(args, "wandb_videos", "joined"), insertion=insertion)
    print_results(summary, records, output, tracker.url)
    tracker.close()
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
    parser.add_argument("--insertion-report", action="store_true",
                        help="Square task: replay the saved rollouts and report how the nut ended up on the post "
                             "(insertions.csv/.json; also sent to W&B). Needs the same simulator as the rollouts")
    parser.add_argument("--wandb", action="store_true", help="Send the results to Weights & Biases")
    parser.add_argument("--wandb-project", help="W&B project name; setting this also enables W&B (default: the checkpoint's)")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        help="W&B mode; default online (or $WANDB_MODE) when W&B is enabled")
    parser.add_argument("--wandb-videos", choices=WANDB_VIDEOS, default="joined",
                        help="Videos to send: the joined SUCCESSFUL_ALL/FAILED_ALL ones, every kept clip, or none")
    parser.add_argument("--name", help="W&B run name (default: the output directory's name)")
    parser.add_argument("--group", help="W&B group, to put runs of one experiment together")
    args = parser.parse_args()
    if args.n_rollouts < 1 or args.video_skip < 1 or args.fps <= 0 or args.seed < 0:
        parser.error("n-rollouts, video-skip and fps must be positive; seed must be nonnegative")
    if args.horizon is not None and args.horizon < 1:
        parser.error("horizon must be positive")
    run(args)


if __name__ == "__main__":
    main()
