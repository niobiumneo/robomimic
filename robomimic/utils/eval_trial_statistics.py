"""Per-trial evaluation statistics shared by live evaluation and CSV backfill.

Success is binary per rollout. A trial's peak running success rate is the
maximum success fraction over its prefixes, starting with the first rollout.
It is descriptive and does not replace the final fixed-budget success rate.
The counts at the peak make small-denominator peaks visible. Ties use the
earliest prefix. Return/horizon SD uses ddof=0, matching the existing evaluator.
This module uses only the Python standard library and does not change RNG state.
"""

import math
import statistics


TRIAL_METRICS = {
    "Success_Rate": "success_rate_mean",
    "Num_Success": "num_success",
    "Num_Rollouts": "num_rollouts",
    "Peak_Running_Success_Rate": "peak_running_success_rate",
    "Peak_Running_Num_Rollouts": "peak_running_num_rollouts",
    "Peak_Running_Num_Success": "peak_running_num_success",
    "Return": "return_mean", "Return_Std": "return_std",
    "Return_Min": "return_min", "Return_Max": "return_max",
    "Horizon": "horizon_mean", "Horizon_Std": "horizon_std",
    "Horizon_Min": "horizon_min", "Horizon_Max": "horizon_max",
}


def summarize_trial_rollouts(records):
    """Summarize rollout-order records with Success_Rate, Return and Horizon."""
    if not records:
        raise ValueError("Cannot summarize an empty evaluation trial")
    successes = [float(record["Success_Rate"]) for record in records]
    returns = [float(record["Return"]) for record in records]
    horizons = [float(record["Horizon"]) for record in records]
    if any(value not in (0.0, 1.0) for value in successes):
        raise ValueError("Rollout Success_Rate must be binary (0 or 1)")
    if not all(math.isfinite(value) for value in returns + horizons):
        raise ValueError("Return and Horizon must be finite for every rollout")
    if any(value < 0 for value in horizons):
        raise ValueError("Horizon cannot be negative")
    count = len(records)
    running_successes = 0
    peak_rate, peak_count, peak_successes = -1.0, 0, 0
    for number, success in enumerate(successes, 1):
        running_successes += int(success)
        rate = running_successes / number
        if rate > peak_rate:
            peak_rate, peak_count, peak_successes = rate, number, running_successes
    return {
        "num_rollouts": count, "success_rate_mean": sum(successes) / count,
        "num_success": int(sum(successes)),
        "return_mean": statistics.mean(returns), "return_std": statistics.pstdev(returns),
        "horizon_mean": statistics.mean(horizons), "horizon_std": statistics.pstdev(horizons),
        "return_min": min(returns), "return_max": max(returns),
        "horizon_min": min(horizons), "horizon_max": max(horizons),
        "peak_running_success_rate": peak_rate,
        "peak_running_num_rollouts": peak_count, "peak_running_num_success": peak_successes,
    }


def define_trial_metrics(run):
    """Use trial number as the chart axis; retain min/max/mean in run summaries."""
    run.define_metric("trial")
    run.define_metric("Trial/*", step_metric="trial")
    for name in ("Success_Rate", "Peak_Running_Success_Rate", "Return", "Return_Max", "Horizon"):
        for aggregation in ("min", "max", "mean"):
            run.define_metric("Trial/" + name, step_metric="trial", summary=aggregation)


def trial_log_values(row):
    """CSV uses zero-based trial indices; W&B displays them starting at 1."""
    values = {"trial": int(row["trial"]) + 1, "trial_seed": row.get("trial_seed")}
    values.update({"Trial/" + name: row[field] for name, field in TRIAL_METRICS.items()
                   if field in row})
    return values


def completed_trial_statistics(rows):
    """Summaries of completed trial rates, not peaks within incomplete trials.

    SE is the sample SD of final trial rates divided by sqrt(n). The existing
    population SD field stays compatible; sample SD has a separate field.
    """
    if not rows:
        return {}
    rates = [float(row["success_rate_mean"]) for row in rows]
    best = max(rows, key=lambda row: row["success_rate_mean"])
    total = sum(int(row["num_rollouts"]) for row in rows)
    success_count = sum(int(row["num_success"]) for row in rows)
    values = {
        "Evaluation/Num_Trials": len(rows),
        "Evaluation/Num_Rollouts": total, "Evaluation/Num_Success": success_count,
        "Evaluation/Success_Rate": success_count / total,
        "Evaluation/Success_Rate_Mean_Over_Trials": statistics.mean(rates),
        "Evaluation/Success_Rate_Min_Over_Trials": min(rates),
        "Evaluation/Success_Rate_Max_Over_Trials": max(rates),
        "Evaluation/Success_Rate_Std_Over_Trials": statistics.pstdev(rates),
        "Evaluation/Best_Trial": int(best["trial"]) + 1,
        "Evaluation/Best_Trial_Seed": best.get("trial_seed"),
    }
    if len(rows) > 1:
        sd = statistics.stdev(rates)
        values["Evaluation/Success_Rate_Sample_Std_Over_Trials"] = sd
        values["Evaluation/Success_Rate_SE_Over_Trials"] = sd / math.sqrt(len(rows))
    for field, name in (("return_mean", "Return"), ("horizon_mean", "Horizon")):
        values["Evaluation/" + name] = sum(
            row[field] * row["num_rollouts"] for row in rows) / total
        means = [row[field] for row in rows]
        values["Evaluation/" + name + "_Min_Over_Trials"] = min(means)
        values["Evaluation/" + name + "_Max_Over_Trials"] = max(means)
    return values
