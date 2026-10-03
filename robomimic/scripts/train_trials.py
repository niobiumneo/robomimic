"""Train one model, then evaluate its best checkpoint over repeated trials.

Run from the repository root, using the same Python environment as train.py:
    python -m robomimic.scripts.train_trials --config CONFIG --dataset DATASET \
        --n-trials 10 --epochs 2000 --group square-cami-c --wandb-project cami

Or skip training and evaluate a model you already have, either one checkpoint
or the best-success checkpoint of an existing run:
    python -m robomimic.scripts.train_trials --checkpoint PATH/model_epoch_1000_..._success_0.9.pth \
        --n-trials 10 --group square-cami-c-best --wandb-project cami
    python -m robomimic.scripts.train_trials --run-dir PATH/to/run \
        --n-trials 10 --group square-cami-c-best --wandb-project cami

This follows run_trained_agent_multi_eval.py: one trained model (a single
training seed), whose best checkpoint is evaluated --n-trials times with
--rollouts-per-trial episodes each. Every trial's trajectories and videos (one
per rollout, plus SUCCESSFUL_ALL.mp4 and FAILED_ALL.mp4 joining them), the
per-epoch curves, a resumable manifest, and W&B logs are kept. The W&B
evaluation run is updated as each trial finishes.

For 60 fps videos of every step, failures included, add:
    --video-skip 1 --fps 60 --keep-failures
(every step at 60 fps plays 3x faster than the 20 Hz simulation.)
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from robomimic.utils.trial_metrics import (
    ROLLOUT_FIELDS, TRIAL_FIELDS, describe, describe_checkpoint, read_history, rollout_rows,
    select_checkpoint, summarize, trial_rows)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = 2


def write_json(path, value):
    """Atomic manifests avoid losing completed trials if the launcher is stopped."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def make_plan(args):
    if args.checkpoint or args.run_dir:
        return make_evaluation_plan(args)
    config = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
    if args.dataset:
        dataset = str(Path(args.dataset).expanduser().resolve(strict=True))
        config["train"]["data"] = [{"path": dataset}]
    data = config["train"]["data"]
    if not isinstance(data, str) and len(data) != 1:
        raise ValueError("Automatic best-checkpoint evaluation requires a single-dataset checkpoint")
    experiment = config["experiment"]
    if experiment.get("additional_envs"):
        raise ValueError("Use a single evaluation environment for automatic best-checkpoint evaluation")
    if experiment.get("ckpt_path"):
        raise ValueError("The model must be trained from scratch; remove experiment.ckpt_path")
    experiment["rollout"]["enabled"] = True
    experiment["save"]["enabled"] = True
    experiment["save"]["on_best_rollout_success_rate"] = True
    experiment["logging"]["log_wandb"] = args.wandb_mode != "disabled"
    if args.wandb_project:
        experiment["logging"]["wandb_proj_name"] = args.wandb_project
    if args.rollouts is not None:
        experiment["rollout"]["n"] = args.rollouts
    if args.rollout_rate is not None:
        experiment["rollout"]["rate"] = args.rollout_rate
    if args.debug:
        args.epochs = 2
        args.rollouts_per_trial = 2
        experiment["epoch_every_n_steps"] = 3
        experiment["validation_epoch_every_n_steps"] = 3
        experiment["rollout"].update(n=2, rate=1, horizon=10, warmstart=0)
    if experiment["rollout"].get("warmstart", 0) >= args.epochs:
        raise ValueError("rollout.warmstart must be below the number of epochs")
    for key in ("n", "rate", "horizon"):
        if not isinstance(experiment["rollout"][key], int) or experiment["rollout"][key] < 1:
            raise ValueError("rollout.{} must be a positive integer".format(key))
    seed = config["train"]["seed"] if args.seed is None else args.seed
    if not isinstance(seed, int) or seed < 0 or seed >= 2 ** 32:
        raise ValueError("The training seed must be an integer in [0, 2**32)")
    settings = {"config": config, "epochs": args.epochs, "seed": seed,
                "n_trials": args.n_trials, "rollouts_per_trial": args.rollouts_per_trial,
                "eval_seed": args.eval_seed, "camera_names": args.camera_names,
                "video_skip": args.video_skip, "fps": args.fps or 20 / args.video_skip,
                "keep_failures": args.keep_failures, "metric_key": args.metric_key,
                "entity": args.wandb_entity}
    # Only a horizon override is recorded, so plans made without one keep the
    # fingerprint they had before the option existed and can still be resumed.
    if args.horizon is not None:
        settings["horizon"] = args.horizon
    if args.no_stitch:
        settings["no_stitch"] = True
    # A changed execution mode is allowed on resume; the scientific settings stay fixed.
    fingerprint = copy.deepcopy(settings)
    fingerprint["config"]["experiment"]["logging"].pop("log_wandb", None)
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    return settings, signature


def make_evaluation_plan(args):
    """Plan for a model that already exists: no training config, seed or epoch count."""
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser()
        if path.is_dir():
            raise ValueError("--checkpoint must be a .pth file; use --run-dir to pick the "
                             "best checkpoint of a run")
        if not path.is_file():
            raise FileNotFoundError("Checkpoint not found: {}".format(path))
        path = path.resolve()
        selection = describe_checkpoint(path, args.metric_key)
    else:
        path, selection = select_checkpoint(args.run_dir, args.metric_key)
    if args.wandb_mode != "disabled" and not args.wandb_project:
        raise ValueError("Set --wandb-project (or --wandb-mode disabled) to evaluate an existing model")
    horizon = args.horizon
    if args.debug:
        args.rollouts_per_trial = 2
        horizon = 10 if horizon is None else horizon
    settings = {"mode": "evaluate", "checkpoint": str(path), "selection": selection,
                "n_trials": args.n_trials, "rollouts_per_trial": args.rollouts_per_trial,
                "eval_seed": args.eval_seed, "camera_names": args.camera_names,
                "video_skip": args.video_skip, "fps": args.fps or 20 / args.video_skip,
                "keep_failures": args.keep_failures, "metric_key": args.metric_key,
                "entity": args.wandb_entity}
    if horizon is not None:
        settings["horizon"] = horizon
    if args.no_stitch:
        settings["no_stitch"] = True
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return settings, signature


def trial_seed(settings, index):
    """First episode seed of the 1-based trial @index.

    Each trial owns rollouts_per_trial consecutive seeds, so every rollout in
    the experiment has its own seed and no two trials share a start state.
    """
    return settings["eval_seed"] + (index - 1) * settings["rollouts_per_trial"]


def training_directory(experiment_dir):
    runs = sorted(path for path in Path(experiment_dir).glob("*")
                  if path.is_dir() and re.fullmatch(r"\d{14}", path.name))
    if not runs:
        raise ValueError("No timestamped training run in {}".format(experiment_dir))
    return runs[-1]


def load_manifest(group_dir):
    path = Path(group_dir) / "manifest.json"
    if not path.is_file():
        raise ValueError("No manifest.json in {}. Check --group and --output-dir. A folder left by a run "
                         "that stopped before writing one has nothing to resume and can be deleted.".format(
                             group_dir))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("{} was written by the earlier ten-seed launcher; "
                         "start a new --group for the single-model protocol".format(path))
    return manifest


def external_training(settings):
    """The 'training' entry for a model trained elsewhere: nothing to run.

    A run made by this repository's trainer keeps its exact per-epoch scalars in
    logs/metrics.jsonl next to models/. When that journal is there and covers the
    checkpoint's epoch, the report includes the curves too.
    """
    checkpoint, selection = Path(settings["checkpoint"]), settings["selection"]
    training = {"name": "existing-checkpoint", "seed": None, "status": "external",
                "training_complete": True, "wandb_id": None, "history": None,
                "checkpoint": dict(selection, path=str(checkpoint))}
    # The trainer writes models/<file> one level below the run folder and last.pth in it.
    # Any other file name or place may be a copy of some other run, so it gets no journal.
    if checkpoint.parent.name == "models":
        run_dir = checkpoint.parent.parent
    elif checkpoint.name in ("last.pth", "last_bak.pth") and (checkpoint.parent / "models").is_dir():
        run_dir = checkpoint.parent
    else:
        run_dir = None
    journal = None if run_dir is None else run_dir / "logs" / "metrics.jsonl"
    if journal is not None and journal.is_file():
        try:
            history = read_history(journal)
        except ValueError:
            history = None
        if history is not None and (selection["epoch"] is None or selection["epoch"] in history):
            training["history"] = str(journal)
    return training


def new_manifest(args, group_dir, settings, signature):
    evaluating = settings.get("mode") == "evaluate"
    # Read the model's journal before creating anything, so a failure here leaves no empty group folder.
    training = external_training(settings) if evaluating else None
    try:
        group_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError("{} already exists. To continue that experiment, repeat the same command with "
                         "--resume. To start over, choose another --group or delete that folder.".format(
                             group_dir)) from None
    if not evaluating:
        config = copy.deepcopy(settings["config"])
        name = args.group + "-train"
        config["train"].update(seed=settings["seed"], num_epochs=settings["epochs"],
                               output_dir=str(group_dir / "training"))
        config["experiment"]["name"] = name
        config_path = group_dir / "config.json"
        write_json(config_path, config)
        training = {"name": name, "seed": settings["seed"], "config": str(config_path),
                    "status": "pending", "training_complete": False,
                    "wandb_id": uuid.uuid4().hex[:8]}
    return {
        "version": MANIFEST_VERSION, "group": args.group, "signature": signature,
        "settings": settings, "training": training,
        "trials": [{"index": index, "name": "trial_{:02d}".format(index),
                    "seed": trial_seed(settings, index), "status": "pending"}
                   for index in range(1, settings["n_trials"] + 1)],
    }


def train_model(manifest, group_dir, args):
    """Run the single training job; an interrupted run resumes from last.pth."""
    training = manifest["training"]
    manifest_path = group_dir / "manifest.json"
    env = os.environ.copy()
    env.update(WANDB_RUN_GROUP=args.group, WANDB_JOB_TYPE="training",
               WANDB_RUN_ID=training["wandb_id"], WANDB_MODE=args.wandb_mode,
               PYTHONHASHSEED=str(training["seed"]))
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    # The training config can follow an intentional offline/disabled resume.
    config_path = Path(training["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["experiment"]["logging"]["log_wandb"] = args.wandb_mode != "disabled"
    write_json(config_path, config)
    command = [sys.executable, "-m", "robomimic.scripts.train", "--config", training["config"], "--quiet"]
    experiment_dir = group_dir / "training" / training["name"]
    if experiment_dir.exists():
        run_dir = training_directory(experiment_dir)
        if not (run_dir / "last.pth").is_file():
            raise ValueError("Incomplete run has no last.pth; use a fresh --group")
        command.append("--resume")
        env["WANDB_RESUME"] = "allow"
    else:
        env["WANDB_RESUME"] = "never"
    training["status"] = "training"
    write_json(manifest_path, manifest)
    print("Training {}".format(training["name"]), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    run_dir = training_directory(experiment_dir)
    history = run_dir / "logs" / "metrics.jsonl"
    if max(read_history(history)) != manifest["settings"]["epochs"]:
        raise ValueError("Training exited without all requested epochs")
    training.update(training_complete=True, training_dir=str(run_dir), history=str(history),
                    status="trained")
    training.pop("error", None)
    write_json(manifest_path, manifest)


def choose_checkpoint(manifest, group_dir):
    """Pick the best-success checkpoint once, so every trial evaluates the same weights."""
    training = manifest["training"]
    path, selection = select_checkpoint(training["training_dir"], manifest["settings"]["metric_key"])
    training["checkpoint"] = dict(selection, path=str(path))
    training.update(status="trained")
    training.pop("error", None)
    write_json(group_dir / "manifest.json", manifest)
    print("Best checkpoint: epoch {} | training-evaluation success {:.1%} | {}".format(
        selection["epoch"], selection["training_success_rate"], path), flush=True)


def evaluate_trial(entry, manifest, group_dir):
    """One trial: a fresh batch of rollouts of the selected checkpoint with its own seeds."""
    settings, training = manifest["settings"], manifest["training"]
    manifest_path = group_dir / "manifest.json"
    # Each attempt writes a new folder, preserving the evidence of earlier attempts.
    export_dir = (group_dir / "evaluations" / entry["name"]
                  / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    command = [sys.executable, "-m", "robomimic.scripts.rollout_best",
               "--checkpoint", training["checkpoint"]["path"], "--output-dir", str(export_dir),
               "--n-rollouts", str(settings["rollouts_per_trial"]), "--seed", str(entry["seed"]),
               "--video-skip", str(settings["video_skip"]), "--fps", str(settings["fps"])]
    if settings.get("horizon") is not None:
        command += ["--horizon", str(settings["horizon"])]
    command += ["--camera-names", *settings["camera_names"]]
    if settings["keep_failures"]:
        command.append("--keep-failures")
    if settings.get("no_stitch"):
        command.append("--no-stitch")
    env = os.environ.copy()
    # A model trained elsewhere has no training seed; any fixed value gives the same repeatability.
    hash_seed = training["seed"] if training["seed"] is not None else settings["eval_seed"]
    env["PYTHONHASHSEED"] = str(hash_seed)
    entry["status"] = "evaluating"
    write_json(manifest_path, manifest)
    print("Evaluating {} of {} | seeds {}-{}".format(
        entry["name"], len(manifest["trials"]), entry["seed"],
        entry["seed"] + settings["rollouts_per_trial"] - 1), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    evaluation = json.loads((export_dir / "summary.json").read_text(encoding="utf-8"))
    if evaluation["n_rollouts_completed"] != settings["rollouts_per_trial"]:
        raise ValueError("Evaluation exited before its fixed episode budget")
    if evaluation["checkpoint"] != training["checkpoint"]["path"]:
        raise ValueError("Trial evaluated different weights than the selected checkpoint")
    entry.update(evaluation=evaluation, evaluation_dir=str(export_dir), status="complete")
    entry.pop("error", None)
    write_json(manifest_path, manifest)


def launch(args, group_dir):
    settings, signature = make_plan(args)
    manifest_path = group_dir / "manifest.json"
    if args.resume:
        manifest = load_manifest(group_dir)
        if manifest["signature"] != signature:
            raise ValueError("Resume settings differ from the saved experiment plan")
    else:
        manifest = new_manifest(args, group_dir, settings, signature)
        write_json(manifest_path, manifest)
    training, settings = manifest["training"], manifest["settings"]
    if settings.get("mode") == "evaluate":
        epoch = training["checkpoint"]["epoch"]
        print("Group: {} | existing checkpoint {} (epoch {}) | {} evaluation trials x {} rollouts | output: {}".format(
            args.group, training["checkpoint"]["path"], "unknown" if epoch is None else epoch,
            settings["n_trials"], settings["rollouts_per_trial"], group_dir), flush=True)
        if not training["history"]:
            print("No metrics journal next to this checkpoint, so the report has the evaluation "
                  "statistics but no per-epoch curves.", flush=True)
    else:
        print("Group: {} | training seed {} for {} epochs | {} evaluation trials x {} rollouts | output: {}".format(
            args.group, training["seed"], settings["epochs"], settings["n_trials"],
            settings["rollouts_per_trial"], group_dir), flush=True)
    active = tracker = None
    try:
        if not training["training_complete"]:
            active = training
            train_model(manifest, group_dir, args)
        if "checkpoint" not in training:
            active = training
            choose_checkpoint(manifest, group_dir)
        active = None
        if any(entry["status"] != "complete" for entry in manifest["trials"]):
            tracker = EvaluationTracker(args, manifest, group_dir)
        for entry in manifest["trials"]:
            if entry["status"] == "complete":
                print("Skipping completed {}".format(entry["name"]), flush=True)
                continue
            active = entry
            evaluate_trial(entry, manifest, group_dir)
            active = None
            tracker.log_trial(entry)
    except BaseException as exc:
        if active is not None:
            active.update(status="failed", error="{}: {}".format(type(exc).__name__, exc))
            write_json(manifest_path, manifest)
            print("Stopped at {}. Fix the error, then repeat this command with --resume.".format(
                active["name"]), file=sys.stderr, flush=True)
        if tracker is not None:
            tracker.close(failed=True)
        raise
    return manifest, tracker


def write_report(report, manifest, group_dir):
    result_dir = group_dir / "results"
    result_dir.mkdir(exist_ok=True)
    write_json(result_dir / "summary.json", {key: value for key, value in report.items() if key != "curves"})
    write_csv(result_dir / "curves.csv", ("epoch", "metric", "value"), report["curves"])
    write_csv(result_dir / "summary.csv", ("metric", "mean", "std", "se", "n"),
              [{"metric": key, **value} for key, value in report["summary"].items()])
    model, env = manifest["group"], report["selection"]["metric_key"] or "checkpoint_env"
    write_csv(result_dir / "trial_results.csv", TRIAL_FIELDS, trial_rows(model, env, manifest["trials"]))
    write_csv(result_dir / "rollout_results.csv", ROLLOUT_FIELDS, rollout_rows(model, env, manifest["trials"]))
    return result_dir


def print_summary(report, manifest):
    """Mirror the overall summary of run_trained_agent_multi_eval.py, with sample SD and SE."""
    overview = {"model": manifest["group"], "n_trials": report["n_trials"],
                "rollouts_per_trial": report["rollouts_per_trial"],
                "checkpoint_epoch": report["selection"]["epoch"]}
    for metric in ("Success_Rate", "Return", "Horizon"):
        stats = report["summary"].get("Evaluation/" + metric)
        if stats is not None:
            name = metric.lower()
            overview.update({name + "_mean_over_trials": stats["mean"],
                             name + "_std_over_trials": stats["std"],
                             name + "_se_over_trials": stats["se"]})
    print("\n=== Overall Summary (std is the sample SD over trials) ===")
    print(json.dumps(overview, indent=4), flush=True)


def evaluation_config(manifest):
    """What the W&B evaluation run records about the model and the plan."""
    settings, training = manifest["settings"], manifest["training"]
    epochs = settings.get("epochs")
    if epochs is None and training.get("history"):
        try:
            epochs = max(read_history(training["history"]))
        except (OSError, ValueError):
            pass
    return {"n_trials": settings["n_trials"], "rollouts_per_trial": settings["rollouts_per_trial"],
            "epochs": epochs, "std_ddof": 1, "training_seed": training["seed"],
            "training_run_id": training["wandb_id"], "checkpoint": training["checkpoint"]["path"],
            "checkpoint_epoch": training["checkpoint"]["epoch"], "eval_seed": settings["eval_seed"],
            "video_skip": settings["video_skip"], "video_fps": settings["fps"]}


def open_wandb_run(args, manifest, directory, run_id):
    """Start the W&B evaluation run of this experiment, or continue it if @run_id already exists."""
    # An existing model has no training config to take the project from.
    config = manifest["settings"].get("config") or {}
    project = args.wandb_project or config.get("experiment", {}).get("logging", {}).get("wandb_proj_name")
    if not project:
        raise ValueError("Set --wandb-project (or --wandb-mode disabled) to upload the results")
    import wandb
    from robomimic import macros as Macros
    if Macros.WANDB_API_KEY is not None:
        os.environ.setdefault("WANDB_API_KEY", Macros.WANDB_API_KEY)
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or Macros.WANDB_ENTITY
    # The evaluation run joins the training run's group but always has its own run ID.
    run = wandb.init(project=project, entity=entity, group=manifest["group"], job_type="evaluation",
                     name=manifest["group"] + "-eval", id=run_id, resume="allow",
                     mode=args.wandb_mode, dir=str(directory), config=evaluation_config(manifest))
    run.define_metric("trial")
    run.define_metric("Trial/*", step_metric="trial")
    run.define_metric("Running/*", step_metric="trial")
    return run


def trial_values(manifest, entry):
    """What one finished trial adds to W&B: its metrics and the running mean/SE over the trials up to it."""
    done = [trial for trial in manifest["trials"]
            if trial["status"] == "complete" and trial["index"] <= entry["index"]]
    values = {"trial": entry["index"]}
    for key, value in entry["evaluation"]["metrics"].items():
        if value is None:
            continue
        stats = describe([trial["evaluation"]["metrics"].get(key) for trial in done])
        values["Trial/" + key] = value
        values["Running/" + key + "/mean"] = stats["mean"]
        if stats["se"] is not None:
            values["Running/" + key + "/se"] = stats["se"]
    return values


class EvaluationTracker:
    """The W&B run that follows the evaluation trials while they happen.

    The run opens before the first trial that has still to run and gets each trial's
    metrics as soon as that trial completes, with the mean and standard error over the
    trials so far. Its ID is kept in the manifest, so --resume continues the same run and
    first adds the trials an earlier invocation completed without logging.

    Tracking is a convenience and must never cost an evaluation: if W&B fails, one warning
    is printed and the evaluation carries on. publish_report then uploads everything to a
    fresh run when the trials are done.
    """

    def __init__(self, args, manifest, group_dir):
        self.manifest, self.manifest_path, self.run = manifest, group_dir / "manifest.json", None
        if args.wandb_mode == "disabled":
            return
        try:
            (group_dir / "results").mkdir(exist_ok=True)
            if "wandb_eval_id" not in manifest:
                manifest["wandb_eval_id"] = uuid.uuid4().hex[:8]
                write_json(self.manifest_path, manifest)
            self.run = open_wandb_run(args, manifest, group_dir / "results", manifest["wandb_eval_id"])
            print("W&B evaluation run, updated after every trial: {}".format(self.run.url or self.run.id),
                  flush=True)
            for entry in manifest["trials"]:
                if entry["status"] == "complete" and not entry.get("wandb_logged"):
                    self.log_trial(entry)
        except Exception as exc:
            self.give_up(exc)

    def log_trial(self, entry):
        if self.run is None:
            return
        try:
            self.run.log(trial_values(self.manifest, entry), step=entry["index"])
            entry["wandb_logged"] = True
            write_json(self.manifest_path, self.manifest)
        except Exception as exc:
            self.give_up(exc)

    def give_up(self, exc):
        print("W&B tracking of the trials stopped ({}: {}). The evaluation continues and the results "
              "are uploaded in full when it ends.".format(type(exc).__name__, exc), file=sys.stderr, flush=True)
        self.close()

    def close(self, failed=False):
        """Finish the live run; a failed evaluation marks it failed, and --resume continues it."""
        run, self.run = self.run, None
        if run is not None:
            try:
                run.finish(exit_code=1 if failed else 0)
            except Exception as exc:
                print("Could not close the W&B run ({}: {})".format(type(exc).__name__, exc),
                      file=sys.stderr, flush=True)


def publish_report(report, manifest, result_dir, args, tracker=None):
    """Add the summary, tables and result files to W&B, in the live run if the trials were tracked in one.

    Without a live run (--aggregate-only, or tracking stopped) a new run gets every trial first.
    """
    if args.wandb_mode == "disabled":
        return
    import wandb
    live = tracker is not None and tracker.run is not None
    run = tracker.run if live else open_wandb_run(args, manifest, result_dir, uuid.uuid4().hex[:8])
    try:
        if not live:
            for entry in manifest["trials"]:
                run.log(trial_values(manifest, entry), step=entry["index"])
        run.summary.update({"Summary/" + key + "/" + stat: value
                            for key, stats in report["summary"].items()
                            for stat, value in stats.items() if value is not None})
        table = wandb.Table(columns=["metric", "mean", "std", "se", "n"],
                            data=[[key, stats["mean"], stats["std"], stats["se"], stats["n"]]
                                  for key, stats in report["summary"].items()])
        trials = wandb.Table(columns=["trial", "seed", "metric", "value", "checkpoint"],
                             data=[[trial["name"], trial["seed"], key, value, trial["checkpoint"]]
                                   for trial in report["trials"] for key, value in trial["metrics"].items()])
        run.log({"metric_summary": table, "trial_summary": trials}, step=report["n_trials"] + 1)
        artifact = wandb.Artifact("evaluation-results-" + run.id, type="evaluation")
        for path in sorted(result_dir.glob("*.csv")) + [result_dir / "summary.json"]:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        print("W&B evaluation run: {}".format(run.url or run.id), flush=True)
    finally:
        run.finish()
        if live:
            tracker.run = None


def run(args):
    args.group = args.group or datetime.now().strftime("cami_%Y%m%d_%H%M%S_%f")
    if Path(args.group).name != args.group or args.group in (".", ".."):
        raise ValueError("Group must be a folder name without path separators")
    group_dir = Path(args.output_dir).expanduser().resolve() / args.group
    manifest, tracker = (load_manifest(group_dir), None) if args.aggregate_only else launch(args, group_dir)
    try:
        training, settings = manifest["training"], manifest["settings"]
        if (training["status"] not in ("trained", "external") or not manifest["trials"]
                or any(trial["status"] != "complete" for trial in manifest["trials"])):
            raise ValueError("Complete training and every evaluation trial before summarizing the experiment")
        history = read_history(training["history"]) if training.get("history") else None
        # A model trained elsewhere reports through the last epoch its journal recorded.
        epochs = settings["epochs"] if settings.get("mode") != "evaluate" else (max(history) if history else None)
        report = summarize(history, epochs, training["checkpoint"], manifest["trials"])
        result_dir = write_report(report, manifest, group_dir)
        if settings.get("mode") == "evaluate":
            done = "{} evaluation trials complete. Trial results and statistics saved in {}"
        else:
            done = "Training and {} evaluation trials complete. Curves, trial results and statistics saved in {}"
        print(done.format(report["n_trials"], result_dir), flush=True)
        print_summary(report, manifest)
        publish_report(report, manifest, result_dir, args, tracker)
    except BaseException:
        if tracker is not None:
            tracker.close(failed=True)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Training JSON template; required to train a model")
    parser.add_argument("--dataset", help="Override train.data with this local dataset")
    parser.add_argument("--epochs", type=int, help="Training epochs (default 2000)")
    parser.add_argument("--seed", type=int, help="Training seed (default: train.seed from the template)")
    parser.add_argument("--checkpoint",
                        help="Skip training: evaluate this existing .pth checkpoint")
    parser.add_argument("--run-dir",
                        help="Skip training: evaluate the best-success checkpoint of this existing run "
                             "(experiment folder, timestamped run, or its models folder)")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Repeated evaluation trials of the best checkpoint")
    parser.add_argument("--rollouts-per-trial", type=int, default=50, help="Episodes in each evaluation trial")
    parser.add_argument("--eval-seed", type=int, default=10000,
                        help="Trial t, rollout j (both counted from 0) uses seed eval-seed + t * rollouts-per-trial + j")
    parser.add_argument("--horizon", type=int,
                        help="Override the rollout horizon of the checkpoint in the evaluation trials")
    parser.add_argument("--group", help="Unique experiment group; defaults to a timestamped name")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "trained_models"))
    parser.add_argument("--rollouts", type=int, help="Training evaluation episodes (default: template)")
    parser.add_argument("--rollout-rate", type=int, help="Evaluate every N epochs (default: template), plus final epoch")
    parser.add_argument("--metric-key", help="Dataset key for best success checkpoint selection")
    parser.add_argument("--camera-names", nargs="+", default=["agentview"])
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--fps", type=float,
                        help="Video playback rate (default: 20/video-skip, which is real time at 20 Hz control). "
                             "--video-skip 1 --fps 60 records every step and plays 3x faster than real time")
    parser.add_argument("--keep-failures", action="store_true",
                        help="Keep a video of each failed or errored rollout, too, and join them into FAILED_ALL.mp4; "
                             "all trajectories are always kept")
    parser.add_argument("--no-stitch", action="store_true",
                        help="Keep only the individual videos: do not also join each trial's clips into "
                             "SUCCESSFUL_ALL.mp4 and FAILED_ALL.mp4")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--resume", action="store_true", help="Skip completed work and resume an incomplete training run")
    parser.add_argument("--aggregate-only", action="store_true", help="Rebuild/re-upload results for a completed group")
    parser.add_argument("--debug", action="store_true", help="Two epochs, three gradient steps/epoch, two short rollouts")
    args = parser.parse_args()
    evaluating = bool(args.checkpoint or args.run_dir)
    if args.checkpoint and args.run_dir:
        parser.error("--checkpoint and --run-dir are alternatives; give one")
    if evaluating:
        training_only = [name for name in ("config", "dataset", "epochs", "seed", "rollouts", "rollout_rate")
                         if getattr(args, name) is not None]
        if training_only:
            parser.error("cannot combine {} with --checkpoint/--run-dir: they only apply when "
                         "training a model".format(
                             ", ".join("--" + name.replace("_", "-") for name in training_only)))
    elif args.epochs is None:
        args.epochs = 2000
    if not args.aggregate_only and not args.config and not evaluating:
        parser.error("--config is required to train a model (or give --checkpoint / --run-dir to evaluate one)")
    if (args.resume or args.aggregate_only) and not args.group:
        parser.error("--group is required for --resume or --aggregate-only")
    for name in ("n_trials", "epochs", "rollouts_per_trial", "video_skip", "horizon"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    if args.eval_seed < 0 or args.eval_seed + args.n_trials * args.rollouts_per_trial > 2 ** 32:
        parser.error("Evaluation seeds must be in [0, 2**32)")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("fps must be finite and positive")
    run(args)


if __name__ == "__main__":
    main()
