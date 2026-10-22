"""Evaluate a saved policy and keep successful videos plus all rollout paths.

Example (run from the repository root):
    python -m robomimic.scripts.rollout_best \
        --run-dir trained_models/square_cami_continuous --n-rollouts 50

Selection uses the trainer's *_success_<rate>.pth filenames. Checkpoint
variable_state.best_success_rate is a historical maximum, not necessarily
the performance of the weights in that file, so it must not rank last.pth.

Every successful rollout gets its own video (--keep-failures adds the failed
ones), and the clips of each outcome are also joined, in rollout order, into
SUCCESSFUL_ALL.mp4 and FAILED_ALL.mp4 with a CSV saying which rollout plays when
(--no-stitch skips the joined copies; each rollout keeps its own video).
rollouts.csv has one row per rollout and summary.json the totals, with a 95%
confidence interval on the success rate.

--n-trials N repeats the evaluation N times with disjoint seed ranges (trial k
starts at --seed + (k-1) * --n-rollouts, as train_trials.py does) and reports each
trial's average and the average over the trials, with a trial_NN folder per trial.

Weights & Biases is opt-in (--wandb, --wandb-project or --wandb-mode), as in
train.py. The run gets every rollout as it finishes (its numbers and, unless
--wandb-videos says otherwise, its video), then the summary, a table of the
rollouts, the joined videos and an artifact with the result files. A W&B
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
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
# select_checkpoint lives in the standard-library trial_metrics module so the
# train_trials launcher can choose the checkpoint without importing torch.
from robomimic.utils.square_success import REASONS, RuleTracker, reason_counts
from robomimic.utils.trial_metrics import describe, resolve_wandb_mode, seed_environment, select_checkpoint

# Outcomes whose clips are joined into one video, in rollout order. A failure
# is a rollout that ended without success or with a simulator error.
COMPILATIONS = (("SUCCESSFUL_ALL", ("successful",)), ("FAILED_ALL", ("failed", "errors")))
INDEX_FIELDS = ("clip", "rollout", "seed", "status", "file", "start_seconds", "duration_seconds")
ROLLOUT_FIELDS = ("rollout", "seed", "status", "success", "return", "horizon", "seconds", "error", "video")
# What each rollout of the Square task ended in and what came closest to success (see robomimic.utils.square_success).
REASON_FIELDS = ("Failure_Reason", "Final_Nut_Height_cm", "Final_XY_Offset_cm", "Final_Hand_Distance_cm", "Final_Tilt_deg",
                 "Closest_Reason", "Closest_Step", "Rule_Mismatch_Steps")
WANDB_VIDEOS = ("all", "joined", "none")
DEFAULT_WANDB_VIDEOS = "all"


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
    extra += reason_fields(records)
    with (output / "rollouts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ROLLOUT_FIELDS) + extra)
        writer.writeheader()
        for record in records:
            writer.writerow(dict(rollout_row(record), **{key: record.get(key) for key in extra}))


def reason_fields(records):
    return [field for field in REASON_FIELDS if any(field in record for record in records)]


def metric_keys(records):
    return sorted({key for record in records for key in record
                   if key in ("Return", "Horizon", "Success_Rate", "time") or key.endswith("_Success_Rate")})


def metric_name(key):
    return "Seconds" if key == "time" else key


def trial_metrics(records, insertion=None):
    """The numbers of one trial: each metric averaged over its rollouts, plus the insertion numbers if there are any."""
    values = {metric_name(key): describe([record.get(key) for record in records])["mean"] for key in metric_keys(records)}
    values["Simulator_Error_Rate"] = sum(record["status"] == "errors" for record in records) / len(records)
    values["num_success"] = sum(record["status"] == "successful" for record in records)
    values["num_errors"] = sum(record["status"] == "errors" for record in records)
    followed = [record for record in records if record.get("Failure_Reason") is not None]
    if followed:  # a trial whose rollouts were not followed has no such numbers, rather than zeros
        counts = reason_counts(followed)
        for reason in REASONS:
            if reason != "success":
                values["Failure/" + reason] = counts.get(reason, 0) / len(followed)  # share of the rollouts that ended this way
    if insertion is not None:
        from robomimic.scripts import inspect_square_insertions as insertions
        values.update(insertions.logged_values(insertion[1], prefix="Insertion/"))
    return {key: value for key, value in values.items() if value is not None}


def trial_value(row, name):
    """A trial's number, or None if the trial has none (see trial_metrics)."""
    return row.get(name)


def across_trials(rows):
    """Mean, sample SD, SE and n of each metric over the trials (one row of ``trial_metrics`` each)."""
    return {name: describe([trial_value(row, name) for row in rows]) for name in sorted({name for row in rows for name in row})}


def trial_values(trial, rows):
    """What the finished trial (the last of ``rows``) adds to W&B: its numbers, and their mean and SE over the trials so far."""
    values = {"trial": trial}
    for name in sorted({name for row in rows for name in row}):
        value = trial_value(rows[-1], name)
        if value is None:
            continue
        stats = describe([trial_value(row, name) for row in rows])
        values["Trial/" + name] = value
        values["Running/{}/mean".format(name)] = stats["mean"]
        if stats["se"] is not None:
            values["Running/{}/se".format(name)] = stats["se"]
    return values


def summary_fields(rows, records):
    """What the finished evaluation adds to a tracker's summary.

    Summary/<metric>/... is over the trials (n = number of trials); Pooled/... is over every rollout of every trial,
    with a 95% interval on the success rate.
    """
    successes = sum(record["status"] == "successful" for record in records)
    errors = sum(record["status"] == "errors" for record in records)
    values = {"Summary/n_trials": len(rows), "Pooled/n_rollouts": len(records), "Pooled/num_success": successes,
              "Pooled/num_errors": errors}
    for name, stats in across_trials(rows).items():
        for stat, value in stats.items():
            if value is not None:
                values["Summary/{}/{}".format(name, stat)] = value
    for key in metric_keys(records):
        for stat, value in describe([record.get(key) for record in records]).items():
            if value is not None:
                values["Pooled/{}/{}".format(metric_name(key), stat)] = value
    followed = [record for record in records if record.get("Failure_Reason") is not None]
    if followed:
        counts = reason_counts(followed)
        for reason in REASONS:
            if reason != "success":
                values["Pooled/Failure/" + reason] = counts.get(reason, 0) / len(followed)
        values["Pooled/Rule_Mismatch_Steps"] = sum(record.get("Rule_Mismatch_Steps", 0) for record in followed)
    low, high = wilson_interval(successes, len(records))
    if low is not None:
        values["Pooled/Success_Rate/ci95_low"], values["Pooled/Success_Rate/ci95_high"] = low, high
    return values


def format_count(value):
    """A number for the printout: three significant digits, but never an exponent for a count in the thousands."""
    return "{:,.0f}".format(value) if abs(value) >= 1000 else "{:.3g}".format(value)


def print_results(summary, records, output, wandb_url=None, title="Results"):
    """The numbers of one trial, as they are also saved in its summary.json and sent to W&B."""
    n, successes = len(records), summary["num_success"]
    low, high = wilson_interval(successes, n)
    print("\n{} over {} rollouts".format(title, n), flush=True)
    print("  success rate {:.1%} ({}/{}), 95% CI {:.1%} to {:.1%}; simulator errors {}".format(
        summary["success_rate"], successes, n, low, high, summary["num_errors"]), flush=True)
    for key, label in (("Return", "return"), ("Horizon", "steps"), ("time", "seconds")):
        stats = describe([record.get(key) for record in records])
        if stats["mean"] is not None:
            print("  {} per rollout: mean {}{}".format(label, format_count(stats["mean"]), "" if stats["std"] is None else
                                                      ", SD {}".format(format_count(stats["std"]))), flush=True)
    print("  saved in {}: summary.json, rollouts.csv, rollouts.hdf5{}".format(
        output, "".join(", " + name for name in ("insertions.csv", "insertions.json") if (output / name).exists())),
          flush=True)
    if wandb_url:
        print("  W&B: {}".format(wandb_url), flush=True)


def print_failures(records, title="Why the rollouts that did not succeed did not"):
    """For the Square task: what each unsuccessful rollout ended in, which is what the end of its video shows."""
    mismatches = sum(record.get("Rule_Mismatch_Steps", 0) for record in records)
    if mismatches:
        print("\nThe reasons are unreliable here: the rule as written disagreed with the environment's own success on "
              "{} steps.".format(mismatches), flush=True)
    failed = [record for record in records if record.get("Failure_Reason") not in (None, "success")]
    unsuccessful = [record for record in records if record["status"] != "successful"]
    if not failed:
        return
    missing = sum(record.get("Failure_Reason") is None for record in unsuccessful)
    print("\n{} ({} of {} rollouts{})".format(
        title, len(unsuccessful), len(records), "; {} without a reason".format(missing) if missing else ""), flush=True)
    counts = reason_counts(failed)
    for reason in sorted(counts, key=lambda name: -counts[name]):
        group = [record for record in failed if record["Failure_Reason"] == reason]
        middle = lambda key: statistics.median(record[key] for record in group)
        print("  {:<22} {:>3}  {}".format(reason, len(group), REASONS.get(reason, "")), flush=True)
        print("  {:<22} {:>3}  at the end: nut {:.1f} cm above the table and {:.1f} cm off the post, hand {:.1f} cm "
              "from the nut, tilt {:.0f} deg (medians)".format(
                  "", "", middle("Final_Nut_Height_cm"), middle("Final_XY_Offset_cm"), middle("Final_Hand_Distance_cm"),
                  middle("Final_Tilt_deg")), flush=True)
        nearer = sum(record["Closest_Reason"] != reason for record in group)
        if nearer:
            print("  {:<22} {:>3}  ({} of them came closest to the rule in another way earlier: see Closest_Reason in "
                  "rollouts.csv)".format("", "", nearer), flush=True)


def print_trials(trials, rows, records, url=None):
    """One line per trial and the average over the trials."""
    print("\nTrials: {} x {} rollouts".format(len(trials), len(trials[0]["records"])), flush=True)
    print("  trial  first seed  success rate  successes   return   steps  errors", flush=True)
    for trial, row in zip(trials, rows):
        print("  {:>5}  {:>10}  {:>12.1%}  {:>9}  {:>7}  {:>6}  {:>6}".format(
            trial["number"], trial["seed"], row["Success_Rate"], "{}/{}".format(row["num_success"], len(trial["records"])),
            format_count(row["Return"]), format_count(row["Horizon"]), row["num_errors"]), flush=True)
    stats = across_trials(rows)
    rate = stats["Success_Rate"]
    print("  average of the trial success rates: {:.1%}{}  (n = {} trials)".format(
        rate["mean"], "" if rate["se"] is None else " +/- {:.1%} (standard error)".format(rate["se"]), rate["n"]), flush=True)
    successes = sum(record["status"] == "successful" for record in records)
    low, high = wilson_interval(successes, len(records))
    print("  all {} rollouts together: {}/{} = {:.1%}, 95% CI {:.1%} to {:.1%}".format(
        len(records), successes, len(records), successes / len(records), low, high), flush=True)
    if url:
        print("  W&B: {}".format(url), flush=True)


class WandbTracker:
    """The W&B run that follows an evaluation of one or several trials.

    It opens before the first rollout and gets each rollout as it finishes (its numbers and, with --wandb-videos all,
    its video), each trial as it finishes (its numbers and their mean and standard error over the trials so far),
    and at the end the summary, the tables, the result files and the insertion report. Tracking is a convenience
    and must never cost an evaluation: if W&B fails, one warning is printed and the evaluation carries on without
    sending more. The run stays open until the evaluation ends, so W&B does not show it as finished while rollouts
    are still running, and it is then marked failed. One video that cannot be sent is skipped and stops nothing.
    """

    def __init__(self, mode, project, entity, group, name, directory, config, videos=DEFAULT_WANDB_VIDEOS):
        self.run, self.wandb, self.mode, self.project = None, None, mode, project
        self.output, self.videos = Path(directory), videos
        # W&B's media panels index by _step, so every row goes at the rollout it belongs to (see log_rollout).
        self.step = 0
        self.stopped, self.videos_lost = False, 0
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
            self.run.define_metric("trial")
            self.run.define_metric("Trial/*", step_metric="trial")
            self.run.define_metric("Running/*", step_metric="trial")
            if mode == "offline":
                print("W&B is offline: nothing is sent until you run: wandb sync {}".format(
                    self.output / "wandb" / "latest-run"), flush=True)
        except Exception as exc:
            self.give_up(exc)
            self.run = None  # nothing was opened, so there is nothing to keep open

    @property
    def active(self):
        return self.run is not None and not self.stopped

    @property
    def url(self):
        if self.run is None or self.mode == "offline":
            return None
        return getattr(self.run, "url", None) or getattr(self.run, "id", None)

    def info(self):
        if self.run is None:
            return None
        return {"project": self.project, "mode": self.mode, "run_id": getattr(self.run, "id", None),
                "url": getattr(self.run, "url", None)}

    def give_up(self, exc):
        """Stop sending. The run is not closed here: it stays open until the evaluation ends, then ends failed."""
        if not self.stopped:
            print("W&B tracking stopped ({}: {}). The evaluation continues and everything is still saved in the "
                  "output directory; the W&B run stays open until the evaluation ends and is then marked failed.".format(
                      type(exc).__name__, exc), flush=True)
        self.stopped = True

    def log_rollout(self, record, records, number, trial, folder):
        """One finished rollout: ``number`` counts the rollouts of all trials, ``records`` are those so far.

        ``folder`` is where the trial keeps its clips.
        """
        if not self.active:
            return
        try:
            values = {"rollout": number, "Rollout/Trial": trial, "Rollout/Success": float(bool(record["Success_Rate"])),
                      "Rollout/Return": record["Return"], "Rollout/Horizon": record["Horizon"],
                      "Rollout/Seconds": record["time"], "Rollout/Simulator_Error": float(record["status"] == "errors")}
            for key in metric_keys([record]):
                if key.endswith("_Success_Rate") and key != "Success_Rate":
                    values["Rollout/" + key] = record[key]
            for key in REASON_FIELDS[1:5]:
                if key in record:
                    values["Rollout/" + key] = record[key]
            stats = describe([r["Success_Rate"] for r in records])
            values["Rollout/Running/Success_Rate/mean"] = stats["mean"]
            if stats["se"] is not None:
                values["Rollout/Running/Success_Rate/se"] = stats["se"]
            self.step = number
            self.run.log(values, step=self.step)
        except Exception as exc:
            self.give_up(exc)
            return
        if self.videos == "all" and record["video"]:
            # a video is sent apart from the numbers, so one that cannot be read costs neither them nor the rest
            try:
                self.run.log({"rollout": number, "Rollout/Video": self.wandb.Video(
                    str(Path(folder) / record["video"]), format="mp4",
                    caption="trial {} rollout {} | {}{} | seed {}".format(
                        trial, record["rollout"], record["status"],
                        " ({})".format(record["Failure_Reason"]) if record.get("Failure_Reason") not in (None, "success") else "",
                        record["seed"]))}, step=self.step)
            except Exception as exc:
                self.videos_lost += 1
                if self.videos_lost <= 3:
                    print("W&B could not take the video of trial {} rollout {} ({}: {}); it is skipped.".format(
                        trial, record["rollout"], type(exc).__name__, exc), flush=True)

    def log_trial(self, trial, rows, folder, summary, n_trials):
        """One finished trial: its numbers with the running mean/SE over the trials, and its joined videos."""
        if not self.active:
            return
        try:
            self.run.log(trial_values(trial, rows), step=self.step)
            if self.videos != "none":
                for name, details in summary.get("compilations", {}).items():
                    key = "Video/" + name if n_trials == 1 else "Video/trial_{:02d}/{}".format(trial, name)
                    self.run.log({key: self.wandb.Video(str(folder / details["video"]), format="mp4")}, step=self.step)
                if not summary.get("compilations") and summary.get("num_videos"):
                    print("No joined videos to send to W&B (--no-stitch or the join failed){}.".format(
                        "" if self.videos == "all" else "; use --wandb-videos all to send each clip"), flush=True)
        except Exception as exc:
            self.give_up(exc)

    def publish(self, trials, rows, records, output):
        """Add the results of the finished evaluation: summary fields, tables and the result files."""
        if not self.active:
            return
        try:
            wandb, run = self.wandb, self.run
            run.summary.update(summary_fields(rows, records))
            names = sorted({name for row in rows for name in row})
            run.log({"trials": wandb.Table(columns=["trial", "seed", "num_rollouts"] + names, data=[
                [trial["number"], trial["seed"], len(trial["records"])] + [row.get(name) for name in names]
                for trial, row in zip(trials, rows)])}, step=self.step)
            extra = reason_fields(records)
            run.log({"rollouts": wandb.Table(columns=["trial"] + list(ROLLOUT_FIELDS) + extra, data=[
                [trial["number"]] + [rollout_row(record)[field] for field in ROLLOUT_FIELDS] + [record.get(field) for field in extra]
                for trial in trials for record in trial["records"]])}, step=self.step)
            reports = [(trial["number"], trial["insertion"]) for trial in trials if trial["insertion"] is not None]
            if reports:
                from robomimic.scripts import inspect_square_insertions as insertions
                run.log({"insertions": wandb.Table(columns=["trial"] + list(insertions.TABLE_COLUMNS), data=[
                    [number] + row for number, (report, _) in reports for row in insertions.table_rows(report)])},
                        step=self.step)
            artifact = wandb.Artifact("rollouts-" + str(run.id), type="rollouts")
            for path in sorted(output.rglob("*")):
                relative = path.relative_to(output)
                if path.is_file() and path.suffix in (".csv", ".json") and "wandb" not in relative.parts:
                    artifact.add_file(str(path), name=str(relative))
            run.log_artifact(artifact)
        except Exception as exc:
            self.give_up(exc)

    def close(self, failed=False):
        """Finish the run, as failed if the evaluation failed or the sending stopped before the end."""
        run, self.run = self.run, None
        if run is not None:
            if self.videos_lost:
                print("{} videos could not be sent to W&B; they are in the output directory.".format(self.videos_lost), flush=True)
            try:
                run.finish(exit_code=1 if failed or self.stopped else 0)
            except Exception as exc:
                print("Could not close the W&B run ({}: {})".format(type(exc).__name__, exc), flush=True)


class NoTracker:
    """Stands in for WandbTracker when W&B was not asked for."""
    url = None

    def info(self):
        return None

    def log_rollout(self, *args, **kwargs):
        pass

    def log_trial(self, *args, **kwargs):
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


_WARNED = set()


def warn_once(message):
    if message not in _WARNED:
        _WARNED.add(message)
        print(message, flush=True)


def collect_episode(policy, env, horizon, camera_names, video_skip, terminate_on_success=True, probe_factory=None):
    """Collect aligned pre-action states, post-action states, and video frames.

    Reset recurrent policy state for each independent rollout. Success uses the
    environment's task criterion and stops the rollout at its first success,
    matching the Square training template. No force injection is required.

    With ``probe_factory`` (Square task) the pieces of the success rule are read after every step, and the stats
    say which came closest to a success and why it was not one (see robomimic.utils.square_success).
    """
    policy.start_episode()
    obs = env.reset()
    probe, closest = None, RuleTracker()
    if probe_factory is not None:
        try:
            probe = probe_factory(env)  # after the reset: robosuite may have built a new simulator
        except Exception as exc:
            warn_once("Failure reasons are not available ({}: {}); the rollouts are unaffected.".format(type(exc).__name__, exc))
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
            if probe is not None:
                try:
                    closest.update(probe.read(), current["task"])
                except Exception as exc:
                    probe, closest = None, RuleTracker()  # a rollout followed only in part has no reason, not a stale one
                    warn_once("Failure reasons stopped ({}: {}); the rollouts are unaffected.".format(type(exc).__name__, exc))
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
    stats.update(closest.result())
    if error and "Failure_Reason" in stats:
        stats["Failure_Reason"] = "simulator_error"  # a rollout that ended in an error is not a success whatever the nut did before
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


def evaluate_trial(ctx, folder, number, first_seed, offset):
    """One trial: ``--n-rollouts`` rollouts of the checkpoint with seeds first_seed, first_seed + 1, ...

    Writes the trial's rollouts.hdf5, videos, rollouts.csv and summary.json (and the insertion report if asked) into
    ``folder``. ``offset`` is the number of rollouts of the earlier trials, so W&B can order all rollouts.
    """
    args, tracker = ctx.args, ctx.tracker
    records = []
    masks = {"successful": [], "failed": [], "errors": []}
    summary = {"checkpoint": str(ctx.checkpoint), "selection": ctx.selection, "wandb": tracker.info(),
               "n_rollouts_requested": args.n_rollouts, "horizon": ctx.horizon,
               "seed": first_seed, "camera_names": args.camera_names,
               "video_skip": args.video_skip, "fps": args.fps, "compilations": {},
               "episodes": records}
    with h5py.File(folder / "rollouts.hdf5", "x") as dataset:
        data = dataset.create_group("data")
        data.attrs["env_args"] = json.dumps(ctx.env.serialize())
        data.attrs["checkpoint"] = str(ctx.checkpoint)
        data.attrs["total"] = 0
        for index in range(args.n_rollouts):
            seed = first_seed + index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            seed_environment(ctx.env, seed)
            started = time.time()
            stats, trajectory, initial, frames = collect_episode(
                ctx.policy, ctx.env, ctx.horizon, args.camera_names, args.video_skip,
                terminate_on_success=ctx.terminate_on_success, probe_factory=ctx.probe_factory)
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
                write_video(folder / video, frames, args.fps)
            records.append(dict(stats, rollout=index, seed=seed, status=status,
                                trajectory=name, video=video,
                                video_frames=len(frames) if video else 0))
            dataset.flush()
            ctx.records.append(records[-1])
            tracker.log_rollout(records[-1], ctx.records, offset + index, number, folder)
            summary.update(n_rollouts_completed=len(records),
                           num_success=len(masks["successful"]),
                           num_errors=len(masks["errors"]),
                           success_rate=len(masks["successful"]) / len(records))
            summary["metrics"] = {key: describe([record.get(key) for record in records])["mean"]
                                  for key in metric_keys(records)}
            summary["metrics"]["Simulator_Error_Rate"] = len(masks["errors"]) / len(records)
            write_summary(folder, summary)
            print("Rollout {}/{} | {} | steps={} | success so far={:.1%}".format(
                index + 1, args.n_rollouts, status, stats["Horizon"],
                summary["success_rate"]), flush=True)
            del frames
        for key, names in masks.items():
            dataset.create_dataset("mask/" + key, data=np.asarray(names, dtype="S"))
    print("Success: {}/{} ({:.1%}); simulator errors: {}".format(
        summary["num_success"], len(records), summary["success_rate"],
        summary["num_errors"]), flush=True)
    if not args.no_stitch:
        try:
            summary["compilations"] = stitch_videos(folder, records, args.fps)
        except Exception as exc:
            # The individual videos are already safe; the evaluation itself must not fail over this.
            print("Could not join the videos ({}: {}). The individual videos are unaffected.".format(
                type(exc).__name__, exc), flush=True)
        write_summary(folder, summary)
        for name, details in summary["compilations"].items():
            print("{}: {} clips, {:.1f} s at {:g} fps -> {}".format(
                name, details["clips"], details["seconds"], args.fps, folder / details["video"]), flush=True)
    if not args.keep_failures and (masks["failed"] or masks["errors"]):
        print("{} unsuccessful rollouts have no video; add --keep-failures to save them{}.".format(
            len(masks["failed"]) + len(masks["errors"]),
            "" if args.no_stitch else " (and join them into FAILED_ALL.mp4)"), flush=True)
    write_rollout_table(folder, records)
    insertion = insertion_report(folder) if ctx.insertion else None
    summary["success_rate_ci95"] = list(wilson_interval(summary["num_success"], len(records)))
    summary["num_videos"] = sum(1 for record in records if record["video"])
    write_summary(folder, summary)
    return {"number": number, "seed": first_seed, "folder": folder, "summary": summary, "records": records,
            "insertion": insertion}


def write_trial_files(output, trials, rows, ctx, aggregate):
    """The files that cover every trial: trial_results.csv and rollout_results.csv (the columns of
    run_trained_agent_multi_eval.py, as train_trials.py writes them), trials.csv with every number, summary.json."""
    from robomimic.utils.trial_metrics import ROLLOUT_FIELDS as ALL_ROLLOUT_FIELDS, TRIAL_FIELDS, rollout_rows, trial_rows
    model = getattr(ctx.args, "name", None) or output.name
    entries = [{"index": trial["number"], "seed": trial["seed"],
                "evaluation": {"checkpoint": str(ctx.checkpoint), "episodes": trial["records"]}} for trial in trials]
    for name, fields, table in (("trial_results.csv", TRIAL_FIELDS, trial_rows(model, ctx.env_name, entries)),
                                ("rollout_results.csv", ALL_ROLLOUT_FIELDS, rollout_rows(model, ctx.env_name, entries))):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(table)
    names = sorted({name for row in rows for name in row})
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial", "seed", "num_rollouts"] + names)
        for trial, row in zip(trials, rows):
            writer.writerow([trial["number"], trial["seed"], len(trial["records"])] + [row.get(name) for name in names])
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    """Restore the policy once and run a fixed, reproducible rollout budget.

    One trial is --n-rollouts episodes of one checkpoint; --n-trials repeats it with disjoint seed ranges, as the
    train_trials launcher does, and averages the trials. A single trial writes its files directly into the output
    directory; several write one trial_NN folder each plus files that cover them all.
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
    n_trials = getattr(args, "n_trials", 1)
    if n_trials < 1:
        raise ValueError("n-trials must be positive")
    wandb_mode = resolve_wandb_mode(args)
    use_wandb = wandb_mode is not None and wandb_mode != "disabled"
    project, project_source = None, None
    if use_wandb:
        project, project_source = getattr(args, "wandb_project", None), "--wandb-project"
        if not project:
            project = getattr(getattr(config.experiment, "logging", None), "wandb_proj_name", None)
            project_source = "the checkpoint's config"
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
    if n_trials > 1:
        print("{} trials of {} rollouts, seeds {} to {}".format(
            n_trials, args.n_rollouts, args.seed, args.seed + n_trials * args.n_rollouts - 1), flush=True)
    if use_wandb:
        print("W&B: project {} (from {}), mode {}".format(project, project_source, wandb_mode), flush=True)
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
    wandb_videos = getattr(args, "wandb_videos", DEFAULT_WANDB_VIDEOS)
    tracker = NoTracker()
    if use_wandb:
        tracker = WandbTracker(
            wandb_mode, project, getattr(args, "wandb_entity", None), getattr(args, "group", None),
            getattr(args, "name", None) or output.name, output,
            {"checkpoint": str(checkpoint), "epoch": (selection or {}).get("epoch"),
             "training_success_rate": (selection or {}).get("training_success_rate"), "n_trials": n_trials,
             "n_rollouts": args.n_rollouts, "horizon": horizon, "seed": args.seed, "camera_names": args.camera_names,
             "video_skip": args.video_skip, "fps": args.fps, "keep_failures": bool(args.keep_failures),
             "output_dir": str(output), "insertion_report": insertion_requested, "wandb_videos": wandb_videos},
            videos=wandb_videos)
    probe_factory = None
    if env_name is not None and "NutAssemblySquare" in env_name:
        from robomimic.scripts.inspect_square_insertions import probe_from_env as probe_factory
    ctx = SimpleNamespace(args=args, policy=policy, env=env, horizon=horizon, checkpoint=checkpoint, selection=selection,
                          tracker=tracker, insertion=insertion_requested, env_name=env_name or "unknown", records=[],
                          probe_factory=probe_factory,
                          terminate_on_success=getattr(config.experiment.rollout, "terminate_on_success", True))
    trials, rows = [], []
    finished = False
    try:
        try:
            for index in range(n_trials):
                folder = output if n_trials == 1 else output / "trial_{:02d}".format(index + 1)
                if n_trials > 1:
                    folder.mkdir()
                    print("\nTrial {}/{} | seeds {} to {}".format(
                        index + 1, n_trials, args.seed + index * args.n_rollouts,
                        args.seed + (index + 1) * args.n_rollouts - 1), flush=True)
                trial = evaluate_trial(ctx, folder, index + 1, args.seed + index * args.n_rollouts, index * args.n_rollouts)
                trials.append(trial)
                rows.append(trial_metrics(trial["records"], trial["insertion"]))
                tracker.log_trial(index + 1, rows, folder, trial["summary"], n_trials)
                print_results(trial["summary"], trial["records"], folder, tracker.url if n_trials == 1 else None,
                              "Results" if n_trials == 1 else "Trial {}".format(index + 1))
                print_failures(trial["records"])
        finally:
            # EnvBase has no close method; robosuite's wrapped simulator does.
            base = getattr(env, "unwrapped", env)
            simulator = getattr(base, "env", base)
            close = getattr(simulator, "close", None)
            if callable(close):
                close()
        summary = trials[0]["summary"]
        if n_trials > 1:
            successes = sum(record["status"] == "successful" for record in ctx.records)
            summary = {"checkpoint": str(checkpoint), "selection": selection, "wandb": tracker.info(), "n_trials": n_trials,
                       "rollouts_per_trial": args.n_rollouts, "seed": args.seed, "horizon": horizon,
                       "trials": [{"trial": trial["number"], "seed": trial["seed"], "folder": trial["folder"].name,
                                   "metrics": row} for trial, row in zip(trials, rows)],
                       "across_trials": across_trials(rows),
                       "pooled": {"n_rollouts": len(ctx.records), "num_success": successes,
                                  "success_rate": successes / len(ctx.records),
                                  "success_rate_ci95": list(wilson_interval(successes, len(ctx.records)))}}
            write_trial_files(output, trials, rows, ctx, summary)
            print_trials(trials, rows, ctx.records, tracker.url)
            print_failures(ctx.records, "Why the rollouts of all trials that did not succeed did not")
        tracker.publish(trials, rows, ctx.records, output)
        print("Videos, all trajectories, and summary saved in {}".format(output), flush=True)
        finished = True
    finally:
        tracker.close(failed=not finished)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Timestamped run, models folder, or experiment folder (latest run)")
    source.add_argument("--checkpoint", help="Explicit .pth file instead of automatic best selection")
    parser.add_argument("--metric-key", help="Dataset metric suffix when a run has multiple metrics")
    parser.add_argument("--output-dir", help="New output directory; must not exist")
    parser.add_argument("--n-rollouts", type=int, default=50, help="Rollouts in each trial")
    parser.add_argument("--n-trials", type=int, default=1,
                        help="Repeat the evaluation this many times with disjoint seed ranges (trial k uses --seed + "
                             "(k-1)*--n-rollouts onwards, as train_trials.py does), and report each trial's average and "
                             "their average. With more than one, --output-dir gets a trial_NN folder per trial")
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
                        help="Do not also join the clips into SUCCESSFUL_ALL.mp4 and FAILED_ALL.mp4 (each rollout keeps its own "
                             "video; the joined videos are copies, so skipping them saves disk space and W&B uploads)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--insertion-report", action="store_true",
                        help="Square task: replay the saved rollouts and report how the nut ended up on the post "
                             "(insertions.csv/.json; also sent to W&B). Needs the same simulator as the rollouts")
    parser.add_argument("--wandb", action="store_true", help="Send the results to Weights & Biases")
    parser.add_argument("--wandb-project", help="W&B project name; setting this also enables W&B (default: the checkpoint's)")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        help="W&B mode; default online (or $WANDB_MODE) when W&B is enabled")
    parser.add_argument("--wandb-videos", choices=WANDB_VIDEOS, default=DEFAULT_WANDB_VIDEOS,
                        help="Videos to send: every kept clip as its rollout finishes plus the joined SUCCESSFUL_ALL/FAILED_ALL "
                             "ones (all, the default), only the joined ones, or none")
    parser.add_argument("--name", help="W&B run name (default: the output directory's name)")
    parser.add_argument("--group", help="W&B group, to put runs of one experiment together")
    args = parser.parse_args()
    if args.n_rollouts < 1 or args.n_trials < 1 or args.video_skip < 1 or args.fps <= 0 or args.seed < 0:
        parser.error("n-rollouts, n-trials, video-skip and fps must be positive; seed must be nonnegative")
    if args.seed + args.n_trials * args.n_rollouts > 2 ** 32:
        parser.error("seed + n-trials * n-rollouts must not exceed 2**32")
    if args.horizon is not None and args.horizon < 1:
        parser.error("horizon must be positive")
    run(args)


if __name__ == "__main__":
    main()
