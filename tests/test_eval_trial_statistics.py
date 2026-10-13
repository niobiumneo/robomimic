"""Check evaluation statistics and CSV backfill without a simulator or account."""
import argparse
import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from robomimic.scripts import log_eval_trial_statistics as backfill
from robomimic.utils.eval_trial_statistics import (
    completed_trial_statistics, summarize_trial_rollouts, trial_log_values,
)


def episodes(successes):
    return [{"Success_Rate": success, "Return": i - 4, "Horizon": 10 + i}
            for i, success in enumerate(successes)]


class TrialStatisticsTests(unittest.TestCase):
    def test_running_peak_records_its_actual_denominator(self):
        result = summarize_trial_rollouts(episodes([0] + [1] * 16 + [0] * 33))
        self.assertEqual(result["num_rollouts"], 50)
        self.assertEqual(result["success_rate_mean"], 16 / 50)
        self.assertEqual(result["peak_running_success_rate"], 16 / 17)
        self.assertEqual(result["peak_running_num_rollouts"], 17)
        self.assertEqual(result["peak_running_num_success"], 16)
        self.assertEqual((result["return_min"], result["return_max"]), (-4, 45))

    def test_peaks_reset_and_a_first_success_has_a_one_rollout_peak(self):
        failed = summarize_trial_rollouts(episodes([0] * 50))
        mixed = summarize_trial_rollouts(episodes([1, 0] * 25))
        self.assertEqual(failed["peak_running_success_rate"], 0)
        self.assertEqual(mixed["peak_running_success_rate"], 1)
        self.assertEqual(mixed["peak_running_num_rollouts"], 1)
        self.assertEqual(mixed["success_rate_mean"], .5)

    def test_best_completed_trial_uses_final_rates_and_sample_se(self):
        rows = [dict(trial=i, trial_seed=i, **summarize_trial_rollouts(episodes(seq)))
                for i, seq in enumerate(([1, 0, 0, 0], [0, 1, 1, 1]))]
        result = completed_trial_statistics(rows)
        self.assertEqual(result["Evaluation/Best_Trial"], 2)
        self.assertEqual(result["Evaluation/Success_Rate_Max_Over_Trials"], .75)
        self.assertEqual(result["Evaluation/Success_Rate_Min_Over_Trials"], .25)
        self.assertEqual(result["Evaluation/Success_Rate_Std_Over_Trials"], .25)
        self.assertAlmostEqual(result["Evaluation/Success_Rate_Sample_Std_Over_Trials"], math.sqrt(.125))
        self.assertAlmostEqual(result["Evaluation/Success_Rate_SE_Over_Trials"], .25)
        self.assertNotIn("Evaluation/Success_Rate_SE_Over_Trials", completed_trial_statistics(rows[:1]))

    def test_invalid_success_and_nonfinite_statistics_are_rejected(self):
        for values in ([], [{"Success_Rate": .94, "Return": 1, "Horizon": 1}],
                       [{"Success_Rate": 1, "Return": float("nan"), "Horizon": 1}]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                summarize_trial_rollouts(values)


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, one_based=False):
        rows = []
        for trial, successes in enumerate(([1, 0, 0], [0, 1, 1])):
            for index, record in enumerate(episodes(successes)):
                row = dict(model="CaMI", env="checkpoint_env", checkpoint="c.pth",
                           trial=trial + int(one_based), trial_seed=trial + 42, rollout=index,
                           success=record["Success_Rate"], horizon=record["Horizon"],
                           **{"return": record["Return"]})
                if one_based:
                    row.update(rollout_seed=10000 + trial * 3 + index,
                               status="successful" if record["Success_Rate"] else "failed")
                rows.append(row)
        path = self.root / ("one.csv" if one_based else "zero.csv")
        self.write(path, list(reversed(rows)))
        return path, rows

    def write(self, path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_both_csv_protocols_preserve_trial_labels_and_rollout_order(self):
        for one_based in (False, True):
            with self.subTest(one_based=one_based):
                path, _ = self.fixture(one_based)
                rows = backfill.load_trials(path, 3)
                self.assertEqual([trial_log_values(row)["trial"] for row in rows], [1, 2])
                self.assertEqual(rows[0]["peak_running_num_rollouts"], 1)
                self.assertEqual(rows[1]["peak_running_success_rate"], 2 / 3)
                self.assertEqual(rows[1]["peak_running_num_rollouts"], 3)
                self.assertEqual(completed_trial_statistics(rows)["Evaluation/Best_Trial"], 2)

    def test_ambiguous_subset_requires_override_and_preserves_trial_number(self):
        path, rows = self.fixture()
        self.write(path, [row for row in rows if row["trial"] == 1])
        with self.assertRaisesRegex(ValueError, "trial index base"):
            backfill.load_trials(path, 3)
        row = backfill.load_trials(path, 3, "0")[0]
        self.assertEqual(trial_log_values(row)["trial"], 2)

    def test_duplicate_rollouts_and_mixed_checkpoints_are_rejected(self):
        path, rows = self.fixture()
        rows[1]["rollout"] = rows[0]["rollout"]
        self.write(path, rows)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            backfill.load_trials(path, 3)
        path, rows = self.fixture()
        rows[0]["checkpoint"] = "different.pth"
        self.write(path, rows)
        with self.assertRaisesRegex(ValueError, "one model/environment/checkpoint"):
            backfill.load_trials(path, 3)

    def test_disabled_mode_writes_results_without_wandb_or_source_changes(self):
        path, _ = self.fixture()
        original = path.read_bytes()
        args = argparse.Namespace(csv=path, rollouts_per_trial=3, trial_index_base="auto",
                                  output_dir=None, wandb_mode="disabled")
        with mock.patch.dict("sys.modules", {"wandb": None}), contextlib.redirect_stdout(io.StringIO()):
            backfill.run(args)
        result = json.loads((self.root / "trial_statistics/trial_statistics_summary.json").read_text())
        self.assertEqual(result["Evaluation/Best_Trial"], 2)
        self.assertEqual(result["Evaluation/Success_Rate_Max_Over_Trials"], 2 / 3)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
