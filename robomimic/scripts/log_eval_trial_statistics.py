r"""Upload per-trial statistics from existing rollout_results.csv; no simulations.

Example from the repository root:
    python -m robomimic.scripts.log_eval_trial_statistics \
        --csv results/cami_best_eval_seed0_50_videos/CaMI/checkpoint_env/rollout_results.csv \
        --rollouts-per-trial 50 --wandb-project cami-contact-state \
        --wandb-name square-best-eval50-seed0-trial-stats

Creates a separate W&B analysis run. Does not modify the original evaluation
run or original CSV. Set --wandb-mode disabled to compute only local results.
Choose one model/environment/checkpoint CSV. A trial must have exactly the
declared rollout budget, with unique zero-based rollout indices. Trial indices
are detected as zero-based (multi-eval) or one-based (train_trials); an explicit
--trial-index-base override supports files containing only later trials.
"""

import argparse
import csv
import json
import os
from pathlib import Path

from robomimic.utils.eval_trial_statistics import (
    completed_trial_statistics, define_trial_metrics, summarize_trial_rollouts, trial_log_values,
)


def load_trials(path, expected_rollouts, trial_index_base="auto"):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("The input CSV has no rollout records")
    required = {"model", "env", "checkpoint", "trial", "trial_seed", "rollout",
                "success", "return", "horizon"}
    missing = required - set(records[0])
    if missing:
        raise ValueError("Missing CSV columns: {}".format(", ".join(sorted(missing))))
    identities = {(row["model"], row["env"], row["checkpoint"]) for row in records}
    if len(identities) != 1:
        raise ValueError("Use one model/environment/checkpoint CSV, for example "
                         "CaMI/checkpoint_env/rollout_results.csv")
    grouped = {}
    for row in records:
        grouped.setdefault(int(row["trial"]), []).append(row)
    if trial_index_base == "auto":
        first_index = min(grouped)
        if first_index == 0:
            index_base = 0
        elif {"rollout_seed", "status"} <= set(records[0]) and first_index >= 1:
            # train_trials CSVs carry per-episode seeds/outcomes and one-based trial IDs.
            index_base = 1
        else:
            # A subset beginning at trial 1 could mean the second zero-based trial.
            raise ValueError("Cannot detect trial index base; specify --trial-index-base 0 or 1")
    else:
        index_base = int(trial_index_base)
    if min(grouped) < index_base:
        raise ValueError("Trial indices are below the selected --trial-index-base")
    trials = []
    for trial, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["rollout"]))
        indices = [int(row["rollout"]) for row in rows]
        if len(rows) != expected_rollouts or indices != list(range(expected_rollouts)):
            raise ValueError("Trial {} has an incomplete or duplicated rollout sequence. "
                             "Expected indices 0..{}; found {} rows. Check --rollouts-per-trial.".format(
                                 trial - index_base + 1, expected_rollouts - 1, len(rows)))
        seeds = {row["trial_seed"] for row in rows}
        if len(seeds) != 1:
            raise ValueError("Trial {} contains conflicting trial seeds".format(trial - index_base + 1))
        seed = rows[0]["trial_seed"]
        stats = [{"Success_Rate": float(row["success"]), "Return": float(row["return"]),
                  "Horizon": float(row["horizon"])} for row in rows]
        trials.append({"model": rows[0]["model"], "env": rows[0]["env"],
                       "checkpoint": rows[0]["checkpoint"], "trial": trial - index_base,
                       "trial_seed": None if seed in ("", "None") else int(seed),
                       **summarize_trial_rollouts(stats)})
    return trials


def run(args):
    source = args.csv.expanduser().resolve(strict=True)
    trials = load_trials(source, args.rollouts_per_trial, args.trial_index_base)
    summary = completed_trial_statistics(trials)
    summary.update({"Analysis/Source_Trials_Complete": True, "Evaluation/Posthoc": True})
    output = (args.output_dir.expanduser() if args.output_dir else source.parent / "trial_statistics").resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "trial_statistics.csv"
    json_path = output / "trial_statistics_summary.json"
    if source in (csv_path, json_path):
        raise ValueError("The source CSV cannot also be an output file")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved trial statistics and summary in {}".format(output), flush=True)
    print("Best completed trial: {} | success rate: {:.1%}".format(
        summary["Evaluation/Best_Trial"], summary["Evaluation/Success_Rate_Max_Over_Trials"]), flush=True)
    if args.wandb_mode == "disabled":
        return
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Local statistics are saved. Install W&B with "
                           "python -m pip install wandb, then repeat this command.") from exc
    first = trials[0]
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                    name=args.wandb_name or first["model"] + "-trial-stats",
                    job_type="evaluation-analysis", mode=args.wandb_mode, dir=str(output),
                    config={"source_csv": str(source), "model": first["model"],
                            "environment": first["env"], "checkpoint": first["checkpoint"],
                            "rollouts_per_trial": args.rollouts_per_trial, "posthoc": True,
                            "trial_index_base": args.trial_index_base}) as logged:
        define_trial_metrics(logged)
        for row in trials:
            logged.log(trial_log_values(row))
        logged.summary.update(summary)
        columns = list(first)
        logged.log({"trial_results": wandb.Table(columns=columns,
                    data=[[row[key] for key in columns] for row in trials])})
        artifact = wandb.Artifact("trial-statistics-" + logged.id, type="evaluation")
        artifact.add_file(str(csv_path))
        artifact.add_file(str(json_path))
        logged.log_artifact(artifact)
        if logged.url:
            print("W&B analysis run: {}".format(logged.url), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--rollouts-per-trial", type=int, default=50)
    parser.add_argument("--trial-index-base", choices=("auto", "0", "1"), default="auto",
                        help="Input CSV numbering: automatic detection, zero-based, or one-based")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wandb-project", default="cami-contact-state")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    if args.rollouts_per_trial < 1:
        parser.error("--rollouts-per-trial must be positive")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, "Error: {}\n".format(exc))


if __name__ == "__main__":
    main()
