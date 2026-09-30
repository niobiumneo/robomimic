"""Train independent seeds, export each best checkpoint's paths, and average metrics.

Run from the repository root, using the same Python environment as train.py:
    python -m robomimic.scripts.train_trials --config CONFIG --dataset DATASET \
        --n-trials 10 --epochs 2000 --group square-cami-c-10 --wandb-project cami
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from robomimic.utils.trial_metrics import aggregate, read_history


REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    """Atomic manifests avoid losing completed trials if the launcher is stopped."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_plan(args):
    config = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
    if args.dataset:
        dataset = str(Path(args.dataset).expanduser().resolve(strict=True))
        config["train"]["data"] = [{"path": dataset}]
    data = config["train"]["data"]
    if not isinstance(data, str) and len(data) != 1:
        raise ValueError("Automatic best-path export requires a single-dataset checkpoint")
    experiment = config["experiment"]
    if experiment.get("additional_envs"):
        raise ValueError("Use a single evaluation environment for automatic best-path export")
    if experiment.get("ckpt_path"):
        raise ValueError("Independent trials must start fresh; remove experiment.ckpt_path")
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
        args.eval_rollouts = 2
        experiment["epoch_every_n_steps"] = 3
        experiment["validation_epoch_every_n_steps"] = 3
        experiment["rollout"].update(n=2, rate=1, horizon=10, warmstart=0)
    if experiment["rollout"].get("warmstart", 0) >= args.epochs:
        raise ValueError("rollout.warmstart must be below the number of epochs")
    for key in ("n", "rate", "horizon"):
        if not isinstance(experiment["rollout"][key], int) or experiment["rollout"][key] < 1:
            raise ValueError("rollout.{} must be a positive integer".format(key))
    seeds = args.seeds or list(range(args.seed_start, args.seed_start + args.n_trials))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be distinct")
    if min(seeds) < 0 or max(seeds) >= 2 ** 32:
        raise ValueError("Training seeds must be in [0, 2**32)")
    settings = {"config": config, "epochs": args.epochs, "seeds": seeds,
                "eval_rollouts": args.eval_rollouts, "eval_seed": args.eval_seed,
                "camera_names": args.camera_names, "video_skip": args.video_skip,
                "fps": args.fps or 20 / args.video_skip, "keep_failures": args.keep_failures,
                "metric_key": args.metric_key, "entity": args.wandb_entity}
    # A changed execution mode is allowed on resume; the scientific settings stay fixed.
    fingerprint = copy.deepcopy(settings)
    fingerprint["config"]["experiment"]["logging"].pop("log_wandb", None)
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    return settings, signature


def training_directory(experiment_dir):
    runs = sorted(path for path in Path(experiment_dir).glob("*")
                  if path.is_dir() and re.fullmatch(r"\d{14}", path.name))
    if not runs:
        raise ValueError("No timestamped training run in {}".format(experiment_dir))
    return runs[-1]


def launch_trials(args, group_dir):
    settings, signature = make_plan(args)
    manifest_path = group_dir / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
            raise ValueError("Resume settings differ from the saved experiment plan")
    else:
        group_dir.mkdir(parents=True, exist_ok=False)
        (group_dir / "configs").mkdir()
        manifest = {"group": args.group, "signature": signature, "settings": settings, "trials": []}
        for index, seed in enumerate(settings["seeds"], 1):
            name = "trial_{:02d}_seed_{}".format(index, seed)
            config_path = group_dir / "configs" / (name + ".json")
            config = copy.deepcopy(settings["config"])
            config["train"].update(seed=seed, num_epochs=args.epochs,
                                   output_dir=str(group_dir / "trials"))
            config["experiment"]["name"] = name
            write_json(config_path, config)
            manifest["trials"].append({"name": name, "seed": seed, "index": index,
                                       "config": str(config_path), "status": "pending",
                                       "training_complete": False, "wandb_id": uuid.uuid4().hex[:8]})
        write_json(manifest_path, manifest)
    print("Group: {} | {} seeds | {} epochs each | output: {}".format(
        args.group, len(manifest["trials"]), settings["epochs"], group_dir), flush=True)
    for entry in manifest["trials"]:
        if entry["status"] == "complete":
            print("Skipping completed {}".format(entry["name"]), flush=True)
            continue
        env = os.environ.copy()
        env.update(WANDB_RUN_GROUP=args.group, WANDB_JOB_TYPE="training",
                   WANDB_RUN_ID=entry["wandb_id"], WANDB_MODE=args.wandb_mode,
                   CAMI_TRIAL_INDEX=str(entry["index"]), PYTHONHASHSEED=str(entry["seed"]))
        if args.wandb_entity:
            env["WANDB_ENTITY"] = args.wandb_entity
        # Child training config can follow an intentional offline/disabled resume.
        config_path = Path(entry["config"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["experiment"]["logging"]["log_wandb"] = args.wandb_mode != "disabled"
        write_json(config_path, config)
        try:
            if not entry["training_complete"]:
                command = [sys.executable, "-m", "robomimic.scripts.train", "--config", entry["config"], "--quiet"]
                experiment_dir = group_dir / "trials" / entry["name"]
                if experiment_dir.exists():
                    run_dir = training_directory(experiment_dir)
                    if not (run_dir / "last.pth").is_file():
                        raise ValueError("Incomplete run has no last.pth; use a fresh --group")
                    command.append("--resume")
                    env["WANDB_RESUME"] = "allow"
                else:
                    env["WANDB_RESUME"] = "never"
                entry["status"] = "training"
                write_json(manifest_path, manifest)
                print("Training {}".format(entry["name"]), flush=True)
                subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
                run_dir = training_directory(experiment_dir)
                history = run_dir / "logs" / "metrics.jsonl"
                if max(read_history(history)) != settings["epochs"]:
                    raise ValueError("Training exited without all requested epochs")
                entry.update(training_complete=True, training_dir=str(run_dir), history=str(history), status="trained")
                write_json(manifest_path, manifest)
            # These are fresh paths from the selected checkpoint, not the earlier
            # training-evaluation paths. The same held-out seeds are used for every model.
            export_dir = group_dir / "evaluations" / entry["name"] / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            command = [sys.executable, "-m", "robomimic.scripts.rollout_best",
                       "--run-dir", entry["training_dir"], "--output-dir", str(export_dir),
                       "--n-rollouts", str(settings["eval_rollouts"]), "--seed", str(settings["eval_seed"]),
                       "--video-skip", str(settings["video_skip"]), "--fps", str(settings["fps"]),
                       "--camera-names", *settings["camera_names"]]
            if settings["keep_failures"]:
                command.append("--keep-failures")
            if settings["metric_key"]:
                command += ["--metric-key", settings["metric_key"]]
            entry["status"] = "evaluating"
            write_json(manifest_path, manifest)
            subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
            evaluation = json.loads((export_dir / "summary.json").read_text(encoding="utf-8"))
            if evaluation["n_rollouts_completed"] != settings["eval_rollouts"]:
                raise ValueError("Evaluation exited before its fixed episode budget")
            entry.update(evaluation=evaluation, evaluation_dir=str(export_dir), status="complete")
            entry.pop("error", None)
            write_json(manifest_path, manifest)
        except BaseException as exc:
            entry.update(status="failed", error="{}: {}".format(type(exc).__name__, exc))
            write_json(manifest_path, manifest)
            print("Stopped at {}. Fix the error, then repeat this command with --resume.".format(
                entry["name"]), file=sys.stderr, flush=True)
            raise
    return manifest


def write_report(report, group_dir):
    result_dir = group_dir / "aggregate"
    result_dir.mkdir(exist_ok=True)
    write_json(result_dir / "summary.json", {key: value for key, value in report.items() if key != "curves"})
    with (result_dir / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "metric", "mean", "std", "se", "n"])
        writer.writeheader()
        writer.writerows(report["curves"])
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "mean", "std", "se", "n"])
        writer.writeheader()
        writer.writerows({"metric": key, **value} for key, value in report["summary"].items())
    return result_dir


def publish_report(report, manifest, result_dir, args):
    if args.wandb_mode == "disabled":
        return
    import wandb
    from robomimic import macros as Macros
    if Macros.WANDB_API_KEY is not None:
        os.environ.setdefault("WANDB_API_KEY", Macros.WANDB_API_KEY)
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or Macros.WANDB_ENTITY
    project = args.wandb_project or manifest["settings"]["config"]["experiment"]["logging"]["wandb_proj_name"]
    # An aggregate uses its own run ID, never a child training run's environment ID.
    run = wandb.init(project=project, entity=entity, group=manifest["group"], job_type="aggregate",
                     name=manifest["group"] + "-mean", id=uuid.uuid4().hex[:8], resume="never",
                     mode=args.wandb_mode, dir=str(result_dir),
                     config={"n_trials": report["n_trials"], "epochs": report["epochs"],
                             "training_seeds": manifest["settings"]["seeds"], "std_ddof": 1,
                             "eval_seed": manifest["settings"]["eval_seed"],
                             "eval_rollouts": manifest["settings"]["eval_rollouts"]})
    try:
        run.define_metric("epoch")
        for prefix in ("Mean", "Std", "SE", "N"):
            run.define_metric(prefix + "/*", step_metric="epoch")
        rows = defaultdict(dict)
        for row in report["curves"]:
            for label, field in (("Mean", "mean"), ("Std", "std"), ("SE", "se"), ("N", "n")):
                if row[field] is not None:
                    rows[row["epoch"]][label + "/" + row["metric"]] = row[field]
        for epoch, values in sorted(rows.items()):
            run.log({"epoch": epoch, **values}, step=epoch)
        run.summary.update({"Summary/" + key + "/" + stat: value
                            for key, stats in report["summary"].items()
                            for stat, value in stats.items() if value is not None})
        table = wandb.Table(columns=["metric", "mean", "std", "se", "n"],
                            data=[[key, stats["mean"], stats["std"], stats["se"], stats["n"]]
                                  for key, stats in report["summary"].items()])
        trials = wandb.Table(columns=["trial", "seed", "metric", "value", "checkpoint"],
                             data=[[trial["name"], trial["seed"], key, value, trial["checkpoint"]]
                                   for trial in report["trials"] for key, value in trial["metrics"].items()])
        run.log({"metric_summary": table, "trial_summary": trials}, step=report["epochs"] + 1)
        artifact = wandb.Artifact("trial-results-" + run.id, type="evaluation")
        for path in sorted(result_dir.glob("*.csv")) + [result_dir / "summary.json"]:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        print("W&B mean run: {}".format(run.url or run.id), flush=True)
    finally:
        run.finish()


def run(args):
    args.group = args.group or datetime.now().strftime("cami_%Y%m%d_%H%M%S_%f")
    if Path(args.group).name != args.group or args.group in (".", ".."):
        raise ValueError("Group must be a folder name without path separators")
    group_dir = Path(args.output_dir).expanduser().resolve() / args.group
    if args.aggregate_only:
        manifest = json.loads((group_dir / "manifest.json").read_text(encoding="utf-8"))
    else:
        manifest = launch_trials(args, group_dir)
    if not manifest["trials"] or any(trial["status"] != "complete" for trial in manifest["trials"]):
        raise ValueError("Complete every planned trial before calculating the experiment mean")
    report = aggregate(manifest["trials"], manifest["settings"]["epochs"])
    result_dir = write_report(report, group_dir)
    print("{} complete training trials. Mean, sample SD, SE and n saved in {}".format(
        report["n_trials"], result_dir), flush=True)
    publish_report(report, manifest, result_dir, args)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Training JSON template; required unless --aggregate-only")
    parser.add_argument("--dataset", help="Override train.data with this local dataset")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit distinct seeds; overrides n-trials/seed-start")
    parser.add_argument("--group", help="Unique experiment group; defaults to a timestamped name")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "trained_models"))
    parser.add_argument("--rollouts", type=int, help="Training evaluation episodes (default: template)")
    parser.add_argument("--rollout-rate", type=int, help="Evaluate every N epochs (default: template), plus final epoch")
    parser.add_argument("--eval-rollouts", type=int, default=50, help="Fresh episodes per best checkpoint")
    parser.add_argument("--eval-seed", type=int, default=10000, help="Same episode seeds for all best checkpoints")
    parser.add_argument("--metric-key", help="Dataset key for best success checkpoint selection")
    parser.add_argument("--camera-names", nargs="+", default=["agentview"])
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--fps", type=float, help="Video frame rate (default: 20/video-skip)")
    parser.add_argument("--keep-failures", action="store_true", help="Keep failure/error videos, too; all paths are always kept")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--resume", action="store_true", help="Skip completed trials and resume incomplete training")
    parser.add_argument("--aggregate-only", action="store_true", help="Rebuild/re-upload metrics for a completed group")
    parser.add_argument("--debug", action="store_true", help="Two epochs, three gradient steps/epoch, two short rollouts")
    args = parser.parse_args()
    if not args.aggregate_only and not args.config:
        parser.error("--config is required for training")
    if (args.resume or args.aggregate_only) and not args.group:
        parser.error("--group is required for --resume or --aggregate-only")
    for name in ("n_trials", "epochs", "eval_rollouts", "video_skip"):
        if getattr(args, name) < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    if args.eval_seed < 0 or args.eval_seed + args.eval_rollouts > 2 ** 32:
        parser.error("Evaluation seeds must be in [0, 2**32)")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("fps must be finite and positive")
    run(args)


if __name__ == "__main__":
    main()
