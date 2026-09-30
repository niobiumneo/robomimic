"""Exact scalar histories and statistics across independent training seeds.

This module only needs the standard library. Missing/non-finite measurements
are excluded with an explicit n; they are never replaced by zero or interpolated.
"""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


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


def aggregate(trials, epochs):
    """Each training seed contributes one value to a metric at a given epoch.

    Final is epoch N, BestCheckpoint is all metrics at the selected checkpoint's
    epoch, and Evaluation is a fresh evaluation of that checkpoint. Individual
    per-metric extrema are deliberately not mixed into BestCheckpoint.
    """
    curves = defaultdict(list)
    summaries = defaultdict(list)
    per_trial = []
    for trial in trials:
        history = read_history(trial["history"])
        if max(history) != epochs:
            raise ValueError("{} did not complete {} epochs".format(trial["name"], epochs))
        best_epoch = int(trial["evaluation"]["selection"]["epoch"])
        if best_epoch not in history:
            raise ValueError("Selected checkpoint epoch {} has no metrics".format(best_epoch))
        for epoch, metrics in history.items():
            for key, value in metrics.items():
                curves[epoch, key].append(value)
        values = {"Selection/Epoch": best_epoch}
        for scope, metrics in (("Final", history[epochs]),
                               ("BestCheckpoint", history[best_epoch]),
                               ("Evaluation", trial["evaluation"]["metrics"])):
            values.update({scope + "/" + key: value for key, value in metrics.items()})
        for key, value in values.items():
            summaries[key].append(value)
        per_trial.append({"name": trial["name"], "seed": trial["seed"],
                          "checkpoint": trial["evaluation"]["checkpoint"], "metrics": values})
    return {
        "n_trials": len(trials), "epochs": epochs, "std_ddof": 1,
        "curves": [{"epoch": epoch, "metric": key, **describe(values)}
                   for (epoch, key), values in sorted(curves.items())],
        "summary": {key: describe(values) for key, values in sorted(summaries.items())},
        "trials": per_trial,
    }


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
