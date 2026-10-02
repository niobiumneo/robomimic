"""Framework-free checks of the one-model, repeated-trial protocol.

Run with: python -m unittest discover -s tests -p test_train_trials.py -v
Synthetic child jobs exercise the launcher without a GPU, simulator or W&B account.
"""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robomimic.scripts import train_trials
from robomimic.utils.trial_metrics import (
    ROLLOUT_FIELDS, TRIAL_FIELDS, ScalarJournal, describe, describe_checkpoint,
    parse_checkpoint_name, read_history, seed_environment, select_checkpoint, summarize,
    trial_rows)

# Trainer-style checkpoint names. Epochs 100 and 150 tie at 0.8: the earlier one wins.
CHECKPOINTS = ("model_epoch_50_square_success_0.4.pth", "model_epoch_100_square_success_0.8.pth",
               "model_epoch_150_square_success_0.8.pth", "model_epoch_200.pth")
TRAINING_SUCCESS = {50: .4, 100: .8, 150: .8}


def write_finished_run(run_dir, epochs, seed):
    """Checkpoints and a metrics journal laid out like the trainer's output."""
    for name in CHECKPOINTS:
        (run_dir / "models" / name).touch()
    with (run_dir / "logs" / "metrics.jsonl").open("w") as stream:
        for epoch in range(1, epochs + 1):
            metrics = {"Train/Loss": epochs - epoch + seed, "System/RAM Usage (MB)": 1234}
            if epoch % 50 == 0 or epoch == epochs:
                metrics["Rollout/Success_Rate/square"] = TRAINING_SUCCESS.get(epoch, .5)
            stream.write(json.dumps({"epoch": epoch, "metrics": metrics}) + "\n")


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.config = self.root / "config.json"
        template = train_trials.REPO_ROOT / "robomimic/exps/templates/bc_cami_square.json"
        self.config.write_text(template.read_text())
        self.dataset = self.root / "data.hdf5"
        self.dataset.touch()
        self.group_dir = self.root / "outputs/test-group"
        self.calls = []
        self.fail_training_once = False
        self.fail_trial_once = None
        self.summary_overrides = {}

    def args(self, **changes):
        values = dict(config=str(self.config), dataset=str(self.dataset), epochs=2000, seed=None,
                      n_trials=10, rollouts_per_trial=50, eval_seed=10000, group="test-group",
                      output_dir=str(self.root / "outputs"), rollouts=None, rollout_rate=None,
                      metric_key=None, camera_names=["agentview"], video_skip=5, fps=None,
                      keep_failures=False, wandb_project="cami-test", wandb_entity=None,
                      wandb_mode="disabled", resume=False, aggregate_only=False, debug=False,
                      checkpoint=None, run_dir=None, horizon=None)
        values.update(changes)
        return argparse.Namespace(**values)

    def evaluation_args(self, **changes):
        """Arguments for evaluating a model that already exists: no training options."""
        values = dict(config=None, dataset=None, epochs=None)
        values.update(changes)
        return self.args(**values)

    def make_run(self, epochs=120, seed=2):
        """A finished run like the ten-seed launcher left behind: trials/trial_02_seed_2."""
        run_dir = self.root / "old/trials/trial_02_seed_2/20260930000000"
        (run_dir / "models").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        write_finished_run(run_dir, epochs, seed)
        return run_dir

    def child(self, command, cwd, env, check):
        self.calls.append((command, env.copy()))
        self.assertEqual(Path(cwd), train_trials.REPO_ROOT)
        self.assertTrue(check)
        if command[2] == "robomimic.scripts.train":
            self.fake_training(command, env)
        else:
            self.assertEqual(command[2], "robomimic.scripts.rollout_best")
            self.fake_evaluation(command)

    def fake_training(self, command, env):
        cfg = json.loads(Path(command[command.index("--config") + 1]).read_text())
        seed, epochs = cfg["train"]["seed"], cfg["train"]["num_epochs"]
        self.assertEqual(env["PYTHONHASHSEED"], str(seed))
        self.assertEqual(env["WANDB_RUN_GROUP"], "test-group")
        self.assertEqual(env["WANDB_JOB_TYPE"], "training")
        run_dir = Path(cfg["train"]["output_dir"]) / cfg["experiment"]["name"] / "20260930000000"
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "models").mkdir(exist_ok=True)
        (run_dir / "last.pth").touch()
        if self.fail_training_once:
            self.fail_training_once = False
            raise subprocess.CalledProcessError(1, command)
        write_finished_run(run_dir, epochs, seed)

    def fake_evaluation(self, command):
        self.assertNotIn("--run-dir", command)
        checkpoint = command[command.index("--checkpoint") + 1]
        seed = int(command[command.index("--seed") + 1])
        rollouts = int(command[command.index("--n-rollouts") + 1])
        export = Path(command[command.index("--output-dir") + 1])
        export.mkdir(parents=True, exist_ok=False)
        index = (seed - 10000) // rollouts + 1
        if self.fail_trial_once == index:
            self.fail_trial_once = None
            raise subprocess.CalledProcessError(1, command)
        # Trial i succeeds in 25 + i of its rollouts, so every statistic is easy to verify.
        episodes = []
        for rollout in range(rollouts):
            success = float(rollout < 25 + index)
            episodes.append({"Return": success, "Horizon": 100 + rollout, "Success_Rate": success,
                             "error": None, "rollout": rollout, "seed": seed + rollout,
                             "status": "successful" if success else "failed"})
        summary = {"checkpoint": checkpoint, "selection": None, "n_rollouts_completed": rollouts,
                   "metrics": {"Success_Rate": (25 + index) / rollouts,
                               "Return": (25 + index) / rollouts,
                               "Horizon": statistics.fmean(e["Horizon"] for e in episodes),
                               "Simulator_Error_Rate": 0.0},
                   "episodes": episodes}
        summary.update(self.summary_overrides)
        (export / "summary.json").write_text(json.dumps(summary))

    def commands(self, module):
        return [command for command, _ in self.calls if command[2] == "robomimic.scripts." + module]

    def test_one_training_run_then_ten_trials_of_the_best_checkpoint(self):
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            report = train_trials.run(self.args())
        training, evaluations = self.commands("train"), self.commands("rollout_best")
        self.assertEqual(len(training), 1)
        self.assertEqual(len(evaluations), 10)
        cfg = json.loads(Path(training[0][training[0].index("--config") + 1]).read_text())
        self.assertEqual((cfg["train"]["seed"], cfg["train"]["num_epochs"]), (1, 2000))
        self.assertEqual(cfg["experiment"]["name"], "test-group-train")
        self.assertEqual(Path(cfg["train"]["output_dir"]), self.group_dir / "training")
        self.assertNotIn("--resume", training[0])

        # Every trial restores the same weights: the earliest of the tied best checkpoints.
        selected = {command[command.index("--checkpoint") + 1] for command in evaluations}
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected.pop().endswith("model_epoch_100_square_success_0.8.pth"))
        self.assertEqual(report["selection"]["epoch"], 100)

        # Trial t owns seeds 10000 + 50t ... 10049 + 50t: 500 distinct rollouts overall.
        starts = [int(command[command.index("--seed") + 1]) for command in evaluations]
        self.assertEqual(starts, [10000 + 50 * t for t in range(10)])
        self.assertEqual({c[c.index("--n-rollouts") + 1] for c in evaluations}, {"50"})
        with (self.group_dir / "results/rollout_results.csv").open() as stream:
            rollouts = list(csv.DictReader(stream))
        self.assertEqual(len(rollouts), 500)
        self.assertEqual(len({row["rollout_seed"] for row in rollouts}), 500)

        # The statistics describe the repeated evaluations of this one model.
        rates = [(25 + t) / 50 for t in range(1, 11)]
        self.assertEqual((report["n_trials"], report["rollouts_per_trial"]), (10, 50))
        success = report["summary"]["Evaluation/Success_Rate"]
        self.assertAlmostEqual(success["mean"], statistics.fmean(rates))
        self.assertAlmostEqual(success["std"], statistics.stdev(rates))
        self.assertAlmostEqual(success["se"], statistics.stdev(rates) / math.sqrt(10))
        self.assertEqual(success["n"], 10)
        # One training run: Final and BestCheckpoint are single measurements, not means.
        final = report["summary"]["Final/Train/Loss"]
        self.assertEqual((final["mean"], final["n"], final["std"]), (1, 1, None))
        self.assertEqual(report["summary"]["BestCheckpoint/Train/Loss"]["mean"], 1901)
        self.assertEqual(report["summary"]["Selection/Epoch"]["mean"], 100)
        self.assertEqual(report["summary"]["BestCheckpoint/Rollout/Success_Rate/square"]["mean"], .8)
        # Curves are the exact history; rollout metrics keep their actual evaluation epochs.
        self.assertEqual(len([r for r in report["curves"] if r["metric"] == "Train/Loss"]), 2000)
        self.assertFalse(any(r["epoch"] == 51 and r["metric"].startswith("Rollout/") for r in report["curves"]))

        manifest = json.loads((self.group_dir / "manifest.json").read_text())
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["training"]["status"], "trained")
        self.assertEqual([t["status"] for t in manifest["trials"]], ["complete"] * 10)
        self.assertEqual(len({t["evaluation_dir"] for t in manifest["trials"]}), 10)
        for name in ("curves.csv", "summary.csv", "summary.json", "trial_results.csv", "rollout_results.csv"):
            self.assertTrue((self.group_dir / "results" / name).is_file(), name)

        with mock.patch.object(train_trials.subprocess, "run") as child:
            resumed = train_trials.run(self.args(resume=True))
            child.assert_not_called()
        self.assertEqual(report["summary"], resumed["summary"])
        with mock.patch.object(train_trials.subprocess, "run") as child:
            rebuilt = train_trials.run(self.args(aggregate_only=True, config=None))
            child.assert_not_called()
        self.assertEqual(report["summary"], rebuilt["summary"])
        with self.assertRaisesRegex(ValueError, "Resume settings differ"):
            train_trials.run(self.args(resume=True, epochs=1999))
        with self.assertRaisesRegex(ValueError, "Resume settings differ"):
            train_trials.run(self.args(resume=True, rollouts_per_trial=49))

    def test_trial_results_use_the_multi_eval_columns_and_formulas(self):
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            train_trials.run(self.args(n_trials=3))
        with (self.group_dir / "results/trial_results.csv").open() as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, list(TRIAL_FIELDS))
        # The columns exist in the script whose output these files must be interchangeable with.
        script = (train_trials.REPO_ROOT / "robomimic/scripts/run_trained_agent_multi_eval.py").read_text()
        for field in TRIAL_FIELDS:
            self.assertIn('"{}"'.format(field), script)
        manifest = json.loads((self.group_dir / "manifest.json").read_text())
        for row, trial in zip(rows, manifest["trials"]):
            episodes = trial["evaluation"]["episodes"]
            self.assertEqual(row["model"], "test-group")
            self.assertEqual(row["env"], "square")
            self.assertEqual((int(row["trial"]), int(row["trial_seed"]), int(row["num_rollouts"])),
                             (trial["index"], trial["seed"], 50))
            self.assertAlmostEqual(float(row["success_rate_mean"]), np.mean([e["Success_Rate"] for e in episodes]))
            self.assertEqual(int(row["num_success"]), int(np.sum([e["Success_Rate"] for e in episodes])))
            self.assertAlmostEqual(float(row["return_std"]), np.std([e["Return"] for e in episodes]))
            self.assertAlmostEqual(float(row["horizon_mean"]), np.mean([e["Horizon"] for e in episodes]))
            self.assertAlmostEqual(float(row["horizon_std"]), np.std([e["Horizon"] for e in episodes]))
        self.assertEqual(float(rows[0]["success_rate_mean"]), .52)
        with (self.group_dir / "results/rollout_results.csv").open() as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, list(ROLLOUT_FIELDS))
        self.assertTrue({"model", "env", "checkpoint", "trial", "trial_seed", "rollout", "return",
                         "horizon", "success"} <= set(ROLLOUT_FIELDS))

    def test_failed_trial_is_recorded_and_resume_repeats_only_unfinished_work(self):
        self.fail_trial_once = 2
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            with self.assertRaises(subprocess.CalledProcessError):
                train_trials.run(self.args(n_trials=3))
        manifest = json.loads((self.group_dir / "manifest.json").read_text())
        self.assertEqual(manifest["training"]["status"], "trained")
        self.assertIn("checkpoint", manifest["training"])
        self.assertEqual([t["status"] for t in manifest["trials"]], ["complete", "failed", "pending"])
        self.assertFalse((self.group_dir / "results/summary.json").exists())
        with self.assertRaisesRegex(ValueError, "Complete training and every"):
            train_trials.run(self.args(n_trials=3, aggregate_only=True))
        self.calls.clear()
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            result = train_trials.run(self.args(n_trials=3, resume=True))
        # The trained model and finished trial are reused; only trials 2 and 3 run.
        self.assertEqual(self.commands("train"), [])
        self.assertEqual([int(c[c.index("--seed") + 1]) for c in self.commands("rollout_best")], [10050, 10100])
        self.assertEqual(result["n_trials"], 3)
        # The failed attempt's folder is preserved next to the new attempt.
        self.assertEqual(len(list((self.group_dir / "evaluations/trial_02").iterdir())), 2)

    def check_rejected_trial(self, overrides, message):
        self.summary_overrides = overrides
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            with self.assertRaisesRegex(ValueError, message):
                train_trials.run(self.args(n_trials=2))
        manifest = json.loads((self.group_dir / "manifest.json").read_text())
        self.assertEqual([t["status"] for t in manifest["trials"]], ["failed", "pending"])
        self.assertIn(message, manifest["trials"][0]["error"])
        self.assertNotIn("evaluation", manifest["trials"][0])
        self.assertFalse((self.group_dir / "results").exists())

    def test_a_trial_that_used_different_weights_is_rejected(self):
        self.check_rejected_trial({"checkpoint": "/elsewhere/model.pth"}, "different weights")

    def test_a_trial_that_stopped_before_its_episode_budget_is_rejected(self):
        self.check_rejected_trial({"n_rollouts_completed": 49}, "fixed episode budget")

    def test_disabled_wandb_mode_never_touches_the_sdk(self):
        sdk = types.SimpleNamespace(init=mock.Mock())
        with mock.patch.dict(sys.modules, {"wandb": sdk}), \
             mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            train_trials.run(self.args(n_trials=2, wandb_mode="disabled"))
        sdk.init.assert_not_called()
        training_config = json.loads((self.group_dir / "config.json").read_text())
        self.assertFalse(training_config["experiment"]["logging"]["log_wandb"])
        self.assertEqual(self.calls[0][1]["WANDB_MODE"], "disabled")

    def test_interrupted_training_resumes_the_same_wandb_run(self):
        self.fail_training_once = True
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            with self.assertRaises(subprocess.CalledProcessError):
                train_trials.run(self.args(n_trials=2))
        manifest = json.loads((self.group_dir / "manifest.json").read_text())
        self.assertEqual(manifest["training"]["status"], "failed")
        self.assertFalse(manifest["training"]["training_complete"])
        self.assertEqual([t["status"] for t in manifest["trials"]], ["pending", "pending"])
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            train_trials.run(self.args(n_trials=2, resume=True))
        first, second = [call for call in self.calls if call[0][2] == "robomimic.scripts.train"]
        self.assertNotIn("--resume", first[0])
        self.assertEqual(first[1]["WANDB_RESUME"], "never")
        self.assertIn("--resume", second[0])
        self.assertEqual(second[1]["WANDB_RESUME"], "allow")
        self.assertEqual(first[1]["WANDB_RUN_ID"], second[1]["WANDB_RUN_ID"])

    def test_summary_requires_complete_training_history_and_known_checkpoint_epoch(self):
        trials = [{"index": 1, "name": "trial_01", "seed": 10000,
                   "evaluation": {"checkpoint": "c.pth", "n_rollouts_completed": 1, "metrics": {"Success_Rate": 1.0}}}]
        history = {1: {"Train/Loss": 3.0}, 2: {"Train/Loss": 2.0}}
        with self.assertRaisesRegex(ValueError, "ends at epoch 2"):
            summarize(history, 3, {"epoch": 1}, trials)
        with self.assertRaisesRegex(ValueError, "no metrics"):
            summarize(history, 2, {"epoch": 7}, trials)
        with self.assertRaisesRegex(ValueError, "At least one"):
            summarize(history, 2, {"epoch": 1}, [])
        report = summarize(history, 2, {"epoch": 1}, trials)
        self.assertEqual(report["summary"]["BestCheckpoint/Train/Loss"]["mean"], 3.0)
        self.assertIsNone(report["summary"]["Evaluation/Success_Rate"]["std"])

    def test_manifest_from_the_ten_seed_launcher_is_rejected(self):
        self.group_dir.mkdir(parents=True)
        (self.group_dir / "manifest.json").write_text(json.dumps({"group": "test-group", "trials": []}))
        with self.assertRaisesRegex(ValueError, "ten-seed"):
            train_trials.run(self.args(aggregate_only=True, config=None))

    def test_debug_plan_is_short_and_seed_defaults_to_the_template(self):
        settings, _ = train_trials.make_plan(self.args(debug=True))
        rollout = settings["config"]["experiment"]["rollout"]
        self.assertEqual((settings["epochs"], settings["rollouts_per_trial"]), (2, 2))
        self.assertEqual((rollout["n"], rollout["horizon"], rollout["rate"]), (2, 10, 1))
        self.assertEqual(settings["seed"], 1)
        self.assertEqual(train_trials.make_plan(self.args(seed=7))[0]["seed"], 7)
        with self.assertRaisesRegex(ValueError, "training seed"):
            train_trials.make_plan(self.args(seed=-1))

    def test_command_line_defaults_follow_the_multi_eval_script(self):
        with mock.patch.object(train_trials, "run") as launcher, \
             mock.patch.object(sys, "argv", ["train_trials", "--config", str(self.config)]):
            train_trials.main()
        args = launcher.call_args.args[0]
        self.assertEqual((args.n_trials, args.rollouts_per_trial, args.epochs, args.seed, args.eval_seed),
                         (10, 50, 2000, None, 10000))
        # The earlier ten-seed flags fail loudly rather than silently changing meaning.
        for flag in ("--seed-start", "--seeds", "--eval-rollouts"):
            with mock.patch.object(sys, "argv", ["train_trials", "--config", str(self.config), flag, "1"]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train_trials.main()

    def test_online_run_publishes_one_evaluation_run_with_per_trial_values(self):
        fake = FakeRun()

        class Artifact:
            def __init__(self, *args, **kwargs):
                self.paths = []

            def add_file(self, path):
                self.paths.append(path)

        wandb = types.SimpleNamespace(init=mock.Mock(return_value=fake), Artifact=Artifact,
                                      Table=lambda **kwargs: types.SimpleNamespace(**kwargs))
        with mock.patch.dict(sys.modules, {"wandb": wandb}), \
             mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            train_trials.run(self.args(wandb_mode="online", wandb_project=None))
        manifest = json.loads((self.group_dir / "manifest.json").read_text())
        kwargs = wandb.init.call_args.kwargs
        self.assertEqual(wandb.init.call_count, 1)
        self.assertEqual((kwargs["group"], kwargs["job_type"], kwargs["name"]),
                         ("test-group", "evaluation", "test-group-eval"))
        self.assertEqual(kwargs["resume"], "never")
        self.assertEqual(kwargs["project"], "cami")
        self.assertEqual(kwargs["config"]["training_seed"], 1)
        self.assertEqual(kwargs["config"]["checkpoint_epoch"], 100)
        self.assertEqual(kwargs["config"]["training_run_id"], manifest["training"]["wandb_id"])
        for index, (values, options) in enumerate(fake.rows[:10], 1):
            self.assertEqual(values["trial"], index)
            self.assertAlmostEqual(values["Trial/Success_Rate"], (25 + index) / 50)
            self.assertEqual(options["step"], index)
        self.assertEqual(set(fake.rows[10][0]), {"metric_summary", "trial_summary"})
        self.assertEqual(fake.rows[10][1]["step"], 11)
        self.assertEqual(fake.summary["Summary/Evaluation/Success_Rate/n"], 10)
        self.assertAlmostEqual(fake.summary["Summary/Evaluation/Success_Rate/mean"], .61)
        self.assertEqual(len(fake.artifact.paths), 5)
        self.assertTrue(fake.finished)
        # Training logs were requested from the child, which owns the group's training run.
        training_config = json.loads((self.group_dir / "config.json").read_text())
        self.assertTrue(training_config["experiment"]["logging"]["log_wandb"])
        self.assertEqual(training_config["experiment"]["logging"]["wandb_proj_name"], "cami")


    # --- evaluating a model that already exists (--checkpoint / --run-dir) -----------------

    def test_best_checkpoint_of_an_existing_run_is_evaluated_without_training(self):
        run_dir = self.make_run()
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            report = train_trials.run(self.evaluation_args(run_dir=str(run_dir.parent), n_trials=3))
        self.assertEqual(self.commands("train"), [])
        evaluations = self.commands("rollout_best")
        self.assertEqual(len(evaluations), 3)
        best = (run_dir / "models/model_epoch_100_square_success_0.8.pth").resolve()
        self.assertEqual({c[c.index("--checkpoint") + 1] for c in evaluations}, {str(best)})
        self.assertEqual([int(c[c.index("--seed") + 1]) for c in evaluations], [10000, 10050, 10100])
        self.assertEqual(report["selection"], {"metric_key": "square", "training_success_rate": 0.8,
                                               "epoch": 100, "path": str(best)})
        # The run's own journal supplies the curves, the last epoch, and the selected epoch.
        self.assertEqual(report["epochs"], 120)
        self.assertEqual(report["summary"]["Final/Train/Loss"]["mean"], 2)
        self.assertEqual(report["summary"]["BestCheckpoint/Train/Loss"]["mean"], 22)
        self.assertEqual(len([r for r in report["curves"] if r["metric"] == "Train/Loss"]), 120)
        manifest = json.loads((self.group_dir / "manifest.json").read_text())
        self.assertEqual((manifest["training"]["status"], manifest["training"]["seed"]), ("external", None))
        self.assertEqual([t["status"] for t in manifest["trials"]], ["complete"] * 3)
        self.assertFalse((self.group_dir / "training").exists())
        self.assertFalse((self.group_dir / "config.json").exists())
        # PYTHONHASHSEED must be an integer even though there is no training seed.
        for _, env in self.calls:
            self.assertTrue(env["PYTHONHASHSEED"].isdigit())

    def test_a_hand_picked_checkpoint_gets_its_epoch_and_success_from_its_name(self):
        # The case of a run's best model at epoch 1000 with 90% in-training success.
        run_dir = self.make_run(epochs=1200)
        picked = run_dir / "models/model_epoch_1000_square_success_0.9.pth"
        picked.touch()
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            report = train_trials.run(self.evaluation_args(checkpoint=str(picked), n_trials=2))
        self.assertEqual(report["selection"], {"metric_key": "square", "training_success_rate": 0.9,
                                               "epoch": 1000, "path": str(picked.resolve())})
        # The explicit file wins over the 0.8 checkpoint that automatic selection would pick.
        self.assertEqual({c[c.index("--checkpoint") + 1] for c in self.commands("rollout_best")},
                         {str(picked.resolve())})
        self.assertEqual(report["epochs"], 1200)
        self.assertEqual(report["summary"]["BestCheckpoint/Train/Loss"]["mean"], 202)
        self.assertEqual(report["summary"]["Selection/Epoch"]["mean"], 1000)

    def test_a_renamed_checkpoint_without_a_journal_reports_only_the_evaluation(self):
        checkpoint = self.root / "best.pth"
        checkpoint.touch()
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            report = train_trials.run(self.evaluation_args(checkpoint=str(checkpoint), n_trials=2))
        self.assertEqual(report["curves"], [])
        self.assertIsNone(report["epochs"])
        self.assertTrue(report["summary"])
        self.assertTrue(all(key.startswith("Evaluation/") for key in report["summary"]))
        self.assertEqual(report["selection"], {"metric_key": None, "training_success_rate": None,
                                               "epoch": None, "path": str(checkpoint.resolve())})
        with (self.group_dir / "results/trial_results.csv").open() as stream:
            self.assertEqual({row["env"] for row in csv.DictReader(stream)}, {"checkpoint_env"})
        self.assertEqual((self.group_dir / "results/curves.csv").read_text().strip(), "epoch,metric,value")

    def test_a_journal_that_does_not_cover_the_checkpoint_epoch_is_ignored(self):
        run_dir = self.make_run(epochs=80)  # the checkpoint below is epoch 100
        picked = run_dir / "models/model_epoch_100_square_success_0.8.pth"
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            report = train_trials.run(self.evaluation_args(checkpoint=str(picked), n_trials=1))
        self.assertEqual(report["curves"], [])
        # The epoch is still known from the file name; nothing that needs the journal is reported.
        self.assertEqual(report["selection"]["epoch"], 100)
        self.assertEqual(report["summary"]["Selection/Epoch"]["mean"], 100)
        self.assertFalse(any(key.startswith(("Final/", "BestCheckpoint/")) for key in report["summary"]))
        self.assertIn("Evaluation/Success_Rate", report["summary"])

    def test_last_checkpoint_has_curves_but_no_selected_epoch_and_copies_get_no_journal(self):
        run_dir = self.make_run()
        (run_dir / "last.pth").touch()
        (run_dir / "copy.pth").touch()
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            report = train_trials.run(self.evaluation_args(checkpoint=str(run_dir / "last.pth"), n_trials=1))
        self.assertIsNone(report["selection"]["epoch"])
        self.assertEqual(report["epochs"], 120)
        self.assertEqual(report["summary"]["Final/Train/Loss"]["mean"], 2)
        self.assertFalse(any(key.startswith(("BestCheckpoint/", "Selection/")) for key in report["summary"]))
        self.assertTrue(report["curves"])
        # The same weights under another name, in the same folder, could belong to any run.
        shutil.rmtree(self.group_dir)
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            report = train_trials.run(self.evaluation_args(checkpoint=str(run_dir / "copy.pth"), n_trials=1))
        self.assertEqual(report["curves"], [])
        self.assertIsNone(report["epochs"])

    def test_checkpoint_evaluation_resumes_and_rebuilds_its_report(self):
        run_dir = self.make_run()
        self.fail_trial_once = 2
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            with self.assertRaises(subprocess.CalledProcessError):
                train_trials.run(self.evaluation_args(run_dir=str(run_dir), n_trials=3))
        manifest = json.loads((self.group_dir / "manifest.json").read_text())
        # A failed trial must not relabel the model itself as failed.
        self.assertEqual(manifest["training"]["status"], "external")
        self.assertEqual([t["status"] for t in manifest["trials"]], ["complete", "failed", "pending"])
        self.calls.clear()
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            result = train_trials.run(self.evaluation_args(run_dir=str(run_dir), n_trials=3, resume=True))
        self.assertEqual([int(c[c.index("--seed") + 1]) for c in self.commands("rollout_best")], [10050, 10100])
        with mock.patch.object(train_trials.subprocess, "run") as child:
            rebuilt = train_trials.run(self.evaluation_args(aggregate_only=True))
            child.assert_not_called()
        self.assertEqual(result["summary"], rebuilt["summary"])
        with self.assertRaisesRegex(ValueError, "Resume settings differ"):
            train_trials.run(self.evaluation_args(run_dir=str(run_dir), n_trials=3, resume=True, horizon=5))

    def test_missing_or_misplaced_checkpoint_paths_get_clear_errors(self):
        with self.assertRaisesRegex(FileNotFoundError, "Checkpoint not found: .*nowhere/model.pth"):
            train_trials.make_plan(self.evaluation_args(checkpoint=str(self.root / "nowhere/model.pth")))
        with self.assertRaisesRegex(ValueError, "use --run-dir"):
            train_trials.make_plan(self.evaluation_args(checkpoint=str(self.make_run())))
        with self.assertRaisesRegex(ValueError, "No timestamped training run"):
            train_trials.make_plan(self.evaluation_args(run_dir=str(self.root)))

    def test_existing_model_needs_a_wandb_project_unless_logging_is_disabled(self):
        run_dir = self.make_run()
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            with self.assertRaisesRegex(ValueError, "wandb-project"):
                train_trials.run(self.evaluation_args(run_dir=str(run_dir), wandb_mode="online",
                                                      wandb_project=None))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.group_dir.exists())

    def test_online_evaluation_of_an_existing_model_creates_no_training_run(self):
        run_dir = self.make_run()
        fake = FakeRun()

        class Artifact:
            def __init__(self, *args, **kwargs):
                self.paths = []

            def add_file(self, path):
                self.paths.append(path)

        wandb = types.SimpleNamespace(init=mock.Mock(return_value=fake), Artifact=Artifact,
                                      Table=lambda **kwargs: types.SimpleNamespace(**kwargs))
        with mock.patch.dict(sys.modules, {"wandb": wandb}), \
             mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            train_trials.run(self.evaluation_args(run_dir=str(run_dir), n_trials=2, wandb_mode="online",
                                                  wandb_project="cami-best"))
        kwargs = wandb.init.call_args.kwargs
        self.assertEqual(wandb.init.call_count, 1)
        self.assertEqual((kwargs["project"], kwargs["job_type"], kwargs["group"], kwargs["name"]),
                         ("cami-best", "evaluation", "test-group", "test-group-eval"))
        self.assertIsNone(kwargs["config"]["training_seed"])
        self.assertIsNone(kwargs["config"]["training_run_id"])
        self.assertEqual(kwargs["config"]["checkpoint_epoch"], 100)
        self.assertEqual(self.commands("train"), [])
        self.assertTrue(fake.finished)

    def test_horizon_override_reaches_the_rollouts_only_when_given(self):
        run_dir = self.make_run()
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            train_trials.run(self.evaluation_args(run_dir=str(run_dir), n_trials=1, horizon=123))
        command = self.commands("rollout_best")[0]
        self.assertEqual(command[command.index("--horizon") + 1], "123")

    def test_training_plans_are_unchanged_by_the_evaluate_only_options(self):
        settings, _ = train_trials.make_plan(self.args())
        self.assertNotIn("mode", settings)
        self.assertNotIn("horizon", settings)
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            train_trials.run(self.args(n_trials=1))
        self.assertNotIn("--horizon", self.commands("rollout_best")[0])
        self.assertEqual(len(self.commands("train")), 1)

    def test_debug_evaluation_is_short(self):
        run_dir = self.make_run()
        settings, _ = train_trials.make_plan(self.evaluation_args(run_dir=str(run_dir), debug=True))
        self.assertEqual((settings["rollouts_per_trial"], settings["horizon"]), (2, 10))
        settings, _ = train_trials.make_plan(self.evaluation_args(run_dir=str(run_dir), debug=True, horizon=30))
        self.assertEqual(settings["horizon"], 30)

    def test_command_line_rules_for_evaluating_an_existing_model(self):
        run_dir = self.make_run()

        def parse(*argv):
            with mock.patch.object(train_trials, "run") as launcher, \
                 mock.patch.object(sys, "argv", ["train_trials", *argv]):
                train_trials.main()
            return launcher.call_args.args[0]

        args = parse("--run-dir", str(run_dir), "--wandb-mode", "disabled")
        self.assertEqual((args.epochs, args.config, args.seed, args.n_trials), (None, None, None, 10))
        self.assertEqual(parse("--checkpoint", "m.pth", "--horizon", "9").horizon, 9)
        # Training options, both sources at once, nothing to do, or a bad horizon: all rejected.
        for bad in (["--checkpoint", "a.pth", "--run-dir", "b"],
                    ["--run-dir", "b", "--config", str(self.config)],
                    ["--run-dir", "b", "--epochs", "5"],
                    ["--run-dir", "b", "--seed", "3"],
                    ["--run-dir", "b", "--horizon", "0"],
                    []):
            with mock.patch.object(sys, "argv", ["train_trials", *bad]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train_trials.main()
        # Training still defaults to 2000 epochs.
        self.assertEqual(parse("--config", str(self.config)).epochs, 2000)


class MetricTests(unittest.TestCase):
    def test_checkpoint_names_give_epoch_and_success(self):
        self.assertEqual(parse_checkpoint_name("m/model_epoch_1000_square_success_0.9.pth"),
                         (1000, [("square", 0.9)]))
        self.assertEqual(parse_checkpoint_name("m/model_epoch_100_square_image_84_with_force_success_0.8.pth"),
                         (100, [("square_image_84_with_force", 0.8)]))
        self.assertEqual(parse_checkpoint_name("best.pth"), (None, []))
        self.assertEqual(parse_checkpoint_name("model_epoch_50_square_return_12.0_square_success_0.8"
                                               "_tool_hang_return_3.0_tool_hang_success_0.4.pth")[1],
                         [("square", 0.8), ("tool_hang", 0.4)])
        self.assertEqual(describe_checkpoint("model_epoch_7.pth"),
                         {"metric_key": None, "training_success_rate": None, "epoch": 7})
        ambiguous = "model_epoch_50_square_success_0.8_tool_hang_success_0.4.pth"
        self.assertIsNone(describe_checkpoint(ambiguous)["training_success_rate"])
        self.assertEqual(describe_checkpoint(ambiguous, "tool_hang"),
                         {"metric_key": "tool_hang", "training_success_rate": 0.4, "epoch": 50})


    def test_nonfinite_values_have_visible_counts_and_no_fabricated_sd(self):
        stats = describe([1, None, float("nan"), 3])
        self.assertEqual(stats["mean"], 2)
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["std"], math.sqrt(2))
        self.assertEqual(describe([1])["std"], None)
        self.assertEqual(describe([])["mean"], None)

    def test_journal_repeated_epochs_and_rng_shared_with_sampler(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = ScalarJournal(temporary)
            journal.record("Train/Loss", 9.0, 1)
            journal.flush(1)
            journal.record("Train/Loss", 2.0, 1)
            journal.record("bad", float("nan"), 1)
            journal.close()
            history = read_history(journal.path)
        self.assertEqual(history[1]["Train/Loss"], 2)
        self.assertIsNone(history[1]["bad"])
        rng = np.random.default_rng(0)
        sampler = types.SimpleNamespace(rng=rng)
        env = types.SimpleNamespace(env=types.SimpleNamespace(rng=rng))
        seed_environment(env, 10000)
        first = sampler.rng.random(5)
        seed_environment(env, 10000)
        np.testing.assert_array_equal(first, sampler.rng.random(5))
        self.assertIs(sampler.rng, env.env.rng)

    def test_trial_rows_match_the_numpy_formulas_of_the_multi_eval_script(self):
        episodes = [{"Return": r, "Horizon": h, "Success_Rate": s, "rollout": i, "seed": 5 + i, "status": "x"}
                    for i, (r, h, s) in enumerate([(0.0, 10, 0.0), (1.0, 4, 1.0), (1.0, 7, 1.0)])]
        trial = {"index": 1, "seed": 5, "evaluation": {"checkpoint": "c.pth", "episodes": episodes}}
        row = trial_rows("model", "env", [trial])[0]
        self.assertAlmostEqual(row["success_rate_mean"], 2 / 3)
        self.assertEqual(row["num_success"], 2)
        self.assertAlmostEqual(row["return_std"], np.std([0, 1, 1]))
        self.assertAlmostEqual(row["horizon_std"], np.std([10, 4, 7]))


class CheckpointSelectionTests(unittest.TestCase):
    def make(self, root, run, *names):
        models = Path(root) / run / "models"
        models.mkdir(parents=True, exist_ok=True)
        for name in names:
            (models / name).touch()
        return models

    def test_best_success_wins_ties_prefer_the_earlier_epoch_and_latest_run_is_used(self):
        with tempfile.TemporaryDirectory() as root:
            self.make(root, "20260101000000", "model_epoch_50_square_success_1.0.pth")
            models = self.make(root, "20260930000000", "model_epoch_100_square_success_0.8.pth",
                               "model_epoch_50_square_success_0.6.pth", "model_epoch_150_square_success_0.8.pth",
                               "model_epoch_150.pth")
            path, selection = select_checkpoint(root)
            self.assertEqual(path, (models / "model_epoch_100_square_success_0.8.pth").resolve())
            self.assertEqual(selection, {"metric_key": "square", "training_success_rate": 0.8, "epoch": 100})
            self.assertEqual(select_checkpoint(models.parent)[0], path)

    def test_selection_needs_an_unambiguous_metric(self):
        with tempfile.TemporaryDirectory() as root:
            self.make(root, "20260930000000", "model_epoch_50_square_return_12.0_square_success_0.8"
                      "_tool_hang_return_3.0_tool_hang_success_0.4.pth")
            with self.assertRaisesRegex(ValueError, "Multiple rollout metrics"):
                select_checkpoint(root)
            self.assertEqual(select_checkpoint(root, "tool_hang")[1]["training_success_rate"], 0.4)
            with self.assertRaisesRegex(ValueError, "No matching"):
                select_checkpoint(root, "missing")


class FakeRun:
    def __init__(self):
        self.id, self.url, self.offline = "fake-run", "https://example.invalid/test", False
        self.rows, self.definitions, self.summary = [], [], {}
        self.finished = False

    def log(self, values, **kwargs):
        self.rows.append((values, kwargs))

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def finish(self):
        self.finished = True

    def log_artifact(self, artifact):
        self.artifact = artifact


class LoggingTests(unittest.TestCase):
    def test_logger_records_every_scalar_and_groups_wandb_by_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake = FakeRun()
            wandb = types.SimpleNamespace(init=mock.Mock(return_value=fake),
                                          Settings=lambda **kwargs: kwargs)
            config = types.SimpleNamespace(train=types.SimpleNamespace(seed=7), meta={},
                                           experiment=types.SimpleNamespace(name="single-model-train",
                                               logging=types.SimpleNamespace(wandb_proj_name="cami")),
                                           to_dict=lambda: {"algo_name": "bc_cami"})
            # Warning colors/progress bars do not affect scalar logging.
            termcolor = types.SimpleNamespace(colored=lambda text, *args, **kwargs: text)
            progress = types.SimpleNamespace(tqdm=type("tqdm", (object,), {}))
            spec = importlib.util.spec_from_file_location(
                "trial_test_logger", train_trials.REPO_ROOT / "robomimic/utils/log_utils.py")
            logger_module = importlib.util.module_from_spec(spec)
            with mock.patch.dict(sys.modules, {"wandb": wandb, "termcolor": termcolor, "tqdm": progress}), \
                 mock.patch.dict(os.environ, {"WANDB_RUN_GROUP": "single-model", "WANDB_JOB_TYPE": "training"}):
                spec.loader.exec_module(logger_module)
                logger = logger_module.DataLogger(temporary, config, log_tb=False, log_wandb=True)
                logger.record("Train/Loss", 2, 50)
                logger.record("Rollout/Success_Rate/square", .8, 50, log_stats=True)
                logger.flush(50)
                logger.close()
            self.assertEqual(wandb.init.call_args.kwargs["group"], "single-model")
            self.assertEqual(wandb.init.call_args.kwargs["job_type"], "training")
            self.assertEqual(wandb.init.call_args.kwargs["config"]["train_seed"], 7)
            self.assertNotIn("trial_seed", wandb.init.call_args.kwargs["config"])
            history = read_history(Path(temporary) / "metrics.jsonl")
            self.assertEqual(history[50]["Train/Loss"], 2)
            self.assertEqual(history[50]["Rollout/Success_Rate/square"], .8)
            self.assertEqual(history[50]["Rollout/Success_Rate/square/mean"], .8)
            self.assertEqual(fake.rows[-1], ({"epoch": 50}, {"step": 50, "commit": True}))
            self.assertTrue(fake.finished)


if __name__ == "__main__":
    unittest.main()
