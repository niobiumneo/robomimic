"""Exact scalar histories and statistics for one trained model and its repeated evaluations.

One model is trained. Its best checkpoint is then evaluated over several trials
of the same number of rollouts, following run_trained_agent_multi_eval.py. This
module only needs the standard library. Missing/non-finite measurements are
excluded with an explicit n; they are never replaced by zero or interpolated.
"""
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


# Column names match run_trained_agent_multi_eval.py, so these CSV files can be
# concatenated with its output and read by CAMI/cami_eval_plots_dist.py.
TRIAL_FIELDS = ("model", "env", "checkpoint", "trial", "trial_seed", "num_rollouts",
                "success_rate_mean", "num_success", "return_mean", "return_std",
                "horizon_mean", "horizon_std")
ROLLOUT_FIELDS = ("model", "env", "checkpoint", "trial", "trial_seed", "rollout",
                  "rollout_seed", "return", "horizon", "success", "status")


class ScalarJournal:
    """Append one durable JSONL row per completed epoch, also when W&B is offline."""

    def __init__(self, log_dir):
        self.path = Path(log_dir) / "metrics.jsonl"
        self._file = self.path.open("a", encoding="utf-8")
        self._epoch = None
        self._metrics = {}

    def record(self, key, value, epoch):
        if self._epoch is not None and self._epoch != epoch:
            self.flush(self._epoch)
        self._epoch = int(epoch)
        number = float(value)
        self._metrics[key] = number if math.isfinite(number) else None

    def flush(self, epoch):
        if self._metrics:
            if self._epoch != epoch:
                raise ValueError("Cannot flush metrics for a different epoch")
            self._file.write(json.dumps({"epoch": int(epoch), "metrics": self._metrics},
                                        allow_nan=False) + "\n")
            self._file.flush()
            self._metrics = {}
        self._epoch = None

    def close(self):
        if self._epoch is not None:
            self.flush(self._epoch)
        self._file.close()


def read_history(path):
    """Read full history; the latest row wins if an epoch was repeated on resume."""
    history = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Malformed metrics row {} in {}".format(number, path)) from exc
            history[int(row["epoch"])] = row["metrics"]
    if not history:
        raise ValueError("No completed epochs in {}".format(path))
    return history


def describe(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(values)
    std = statistics.stdev(values) if n > 1 else None
    return {"mean": statistics.fmean(values) if n else None,
            "std": std, "se": std / math.sqrt(n) if std is not None else None, "n": n}


def select_checkpoint(run_dir, metric_key=None):
    """Find the best success-tagged checkpoint in one timestamped run.

    An experiment directory selects its latest timestamped run, never a
    mixture of runs. Multiple evaluation datasets require --metric-key.
    Equal success rates prefer the earlier epoch, without a return tie-break.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if run_dir.name == "models":
        run_dir = run_dir.parent
    if not (run_dir / "models").is_dir():
        runs = sorted(p for p in run_dir.glob("*")
                      if p.is_dir() and re.fullmatch(r"\d{14}", p.name)
                      and (p / "models").is_dir())
        if not runs:
            raise ValueError("No timestamped training run found in {}".format(run_dir))
        run_dir = runs[-1]

    # Dataset keys can contain underscores. Parse return and success tags in
    # order so a return tag never absorbs a preceding success tag.
    pattern = re.compile(
        r"_(?P<key>.+?)_(?P<kind>success|return)_(?P<rate>-?\d+(?:\.\d+)?)"
        r"(?=_|$)"
    )
    candidates = []
    for path in sorted((run_dir / "models").glob("model_epoch_*.pth")):
        epoch_match = re.match(r"model_epoch_(\d+)", path.stem)
        if epoch_match is None:
            continue
        suffix = path.stem[epoch_match.end():]
        suffix = re.sub(r"^_best_validation_-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", "", suffix)
        for match in pattern.finditer(suffix):
            if match["kind"] == "success":
                candidates.append((match["key"], float(match["rate"]),
                                   int(epoch_match[1]), path))
    keys = sorted({key for key, _, _, _ in candidates})
    if metric_key is None and len(keys) > 1:
        raise ValueError("Multiple rollout metrics found; use --metric-key: {}".format(keys))
    candidates = [c for c in candidates if metric_key is None or c[0] == metric_key]
    if not candidates:
        raise ValueError("No matching success-tagged checkpoint in {}. "
                         "Wait for evaluation, or pass --checkpoint explicitly.".format(run_dir))
    key, rate, epoch, path = max(candidates, key=lambda c: (c[1], -c[2]))
    return path, {"metric_key": key, "training_success_rate": rate, "epoch": epoch}


def summarize(history, epochs, selection, trials):
    """Report one trained model evaluated over repeated trials of its best checkpoint.

    Curves are the exact per-epoch scalars of the single training run; nothing is
    averaged. Final is epoch N and BestCheckpoint is every metric at the selected
    checkpoint's epoch. Evaluation statistics are over trials: each trial
    contributes the mean of its own rollouts, so n is the number of trials and
    the spread measures evaluation variability of this one model, not training
    variability. Individual per-metric extrema are deliberately not mixed into
    BestCheckpoint.
    """
    if max(history) != epochs:
        raise ValueError("Training history ends at epoch {}, not {}".format(max(history), epochs))
    best_epoch = int(selection["epoch"])
    if best_epoch not in history:
        raise ValueError("Selected checkpoint epoch {} has no metrics".format(best_epoch))
    if not trials:
        raise ValueError("At least one evaluation trial is required")
    summaries = defaultdict(list)
    summaries["Selection/Epoch"].append(best_epoch)
    for scope, metrics in (("Final", history[epochs]), ("BestCheckpoint", history[best_epoch])):
        for key, value in metrics.items():
            summaries[scope + "/" + key].append(value)
    per_trial = []
    for trial in trials:
        evaluation = trial["evaluation"]
        values = {"Evaluation/" + key: value for key, value in evaluation["metrics"].items()}
        for key, value in values.items():
            summaries[key].append(value)
        per_trial.append({"index": trial["index"], "name": trial["name"], "seed": trial["seed"],
                          "checkpoint": evaluation["checkpoint"], "metrics": values})
    return {
        "n_trials": len(trials),
        "rollouts_per_trial": int(trials[0]["evaluation"]["n_rollouts_completed"]),
        "epochs": epochs, "std_ddof": 1, "selection": selection,
        "curves": [{"epoch": epoch, "metric": key, "value": value}
                   for epoch, metrics in sorted(history.items())
                   for key, value in sorted(metrics.items())],
        "summary": {key: describe(values) for key, values in sorted(summaries.items())},
        "trials": per_trial,
    }


def trial_rows(model, env, trials):
    """One row per trial, with the same fields as run_trained_agent_multi_eval.py.

    Within-trial spreads are population SDs, as in that script. Trial numbers
    start at 1 here, matching the trial_XX folders.
    """
    rows = []
    for trial in trials:
        evaluation = trial["evaluation"]
        episodes = evaluation["episodes"]
        successes = [episode["Success_Rate"] for episode in episodes]
        returns = [episode["Return"] for episode in episodes]
        horizons = [episode["Horizon"] for episode in episodes]
        rows.append({
            "model": model, "env": env, "checkpoint": evaluation["checkpoint"],
            "trial": trial["index"], "trial_seed": trial["seed"], "num_rollouts": len(episodes),
            "success_rate_mean": statistics.fmean(successes), "num_success": int(sum(successes)),
            "return_mean": statistics.fmean(returns), "return_std": statistics.pstdev(returns),
            "horizon_mean": statistics.fmean(horizons), "horizon_std": statistics.pstdev(horizons),
        })
    return rows


def rollout_rows(model, env, trials):
    """One row per rollout; rollout numbers start at 0, like the demo_N trajectory names."""
    rows = []
    for trial in trials:
        evaluation = trial["evaluation"]
        for episode in evaluation["episodes"]:
            rows.append({
                "model": model, "env": env, "checkpoint": evaluation["checkpoint"],
                "trial": trial["index"], "trial_seed": trial["seed"], "rollout": episode["rollout"],
                "rollout_seed": episode["seed"], "return": episode["Return"],
                "horizon": episode["Horizon"], "success": episode["Success_Rate"],
                "status": episode["status"],
            })
    return rows


def seed_environment(env, seed):
    """Seed Hisham's robosuite Generator, keeping sampler references intact.

    np.random.seed alone does not seed np.random.default_rng. Changing the
    generator's state preserves the reference held by placement samplers.
    Legacy environments without a Generator use the caller's global RNG seed.
    """
    import numpy as np
    base = getattr(env, "unwrapped", env)
    simulator = getattr(base, "env", base)
    rng = getattr(simulator, "rng", None)
    if isinstance(rng, np.random.Generator):
        seeded = type(rng.bit_generator)(int(seed))
        rng.bit_generator.state = seeded.state
