"""Framework-free checks of seed orchestration, exact means, resume, and W&B logging.

Run with: python -m unittest discover -s tests -p test_train_trials.py -v
Synthetic child jobs exercise the launcher without a GPU, simulator or W&B account.
"""
import argparse
import importlib.util
import json
import math
import os
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
from robomimic.utils.trial_metrics import ScalarJournal, aggregate, describe, read_history, seed_environment


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.config = self.root / "config.json"
        template = train_trials.REPO_ROOT / "robomimic/exps/templates/bc_cami_square.json"
        self.config.write_text(template.read_text())
        self.dataset = self.root / "data.hdf5"
        self.dataset.touch()
        self.calls = []
        self.fail_seed_once = None

    def args(self, **changes):
        values = dict(config=str(self.config), dataset=str(self.dataset), n_trials=10, epochs=2000,
                      seed_start=1, seeds=None, group="test-10", output_dir=str(self.root / "outputs"),
                      rollouts=None, rollout_rate=None, debug=False, eval_rollouts=50, eval_seed=10000,
                      camera_names=["agentview"], video_skip=5, fps=None, keep_failures=False,
                      metric_key=None, wandb_project="cami-test", wandb_entity=None,
                      wandb_mode="disabled", resume=False, aggregate_only=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def child(self, command, cwd, env, check):
        self.calls.append((command, env.copy()))
        self.assertEqual(Path(cwd), train_trials.REPO_ROOT)
        self.assertTrue(check)
        if command[2] == "robomimic.scripts.train":
            cfg = json.loads(Path(command[command.index("--config") + 1]).read_text())
            seed, epochs = cfg["train"]["seed"], cfg["train"]["num_epochs"]
            self.assertEqual(env["PYTHONHASHSEED"], str(seed))
            self.assertEqual(env["WANDB_RUN_GROUP"], "test-10")
            self.assertEqual(env["WANDB_JOB_TYPE"], "training")
            run_dir = Path(cfg["train"]["output_dir"]) / cfg["experiment"]["name"] / "20260930000000"
            logs = run_dir / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            (run_dir / "models").mkdir(exist_ok=True)
            (run_dir / "last.pth").touch()
            if self.fail_seed_once == seed:
                self.fail_seed_once = None
                raise subprocess.CalledProcessError(1, command)
            with (logs / "metrics.jsonl").open("w") as stream:
                for epoch in range(1, epochs + 1):
                    metrics = {"Train/Loss": epochs - epoch + seed, "System/RAM Usage (MB)": 1234}
                    if epoch % 50 == 0 or epoch == epochs:
                        metrics["Rollout/Success_Rate/square"] = seed / 10
                    stream.write(json.dumps({"epoch": epoch, "metrics": metrics}) + "\n")
        else:
            self.assertEqual(command[2], "robomimic.scripts.rollout_best")
            run_dir = Path(command[command.index("--run-dir") + 1])
            seed = int(run_dir.parent.name.split("seed_")[1])
            export = Path(command[command.index("--output-dir") + 1])
            export.mkdir(parents=True, exist_ok=False)
            self.assertEqual(command[command.index("--seed") + 1], "10000")
            summary = {"checkpoint": str(run_dir / "models" / "selected.pth"),
                       "selection": {"epoch": 100, "training_success_rate": seed / 10},
                       "n_rollouts_completed": 50,
                       "metrics": {"Success_Rate": seed / 10, "Return": seed,
                                   "Horizon": 400 - seed, "Simulator_Error_Rate": 0.0}}
            (export / "summary.json").write_text(json.dumps(summary))

    def test_ten_fresh_2000_epoch_jobs_and_exact_statistics(self):
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            report = train_trials.run(self.args())
        training = [call for call in self.calls if call[0][2].endswith(".train")]
        exports = [call for call in self.calls if call[0][2].endswith(".rollout_best")]
        self.assertEqual(len(training), 10)
        self.assertEqual(len(exports), 10)
        self.assertEqual(len({env["WANDB_RUN_ID"] for _, env in training}), 10)
        for command, _ in training:
            cfg = json.loads(Path(command[command.index("--config") + 1]).read_text())
            self.assertEqual(cfg["train"]["num_epochs"], 2000)
            self.assertNotIn("--resume", command)
        final = report["summary"]["Final/Train/Loss"]
        self.assertEqual(final["mean"], 5.5)
        self.assertEqual(final["n"], 10)
        self.assertAlmostEqual(final["std"], statistics.stdev(range(1, 11)))
        self.assertAlmostEqual(final["se"], final["std"] / math.sqrt(10))
        self.assertEqual(report["summary"]["BestCheckpoint/Train/Loss"]["mean"], 1905.5)
        self.assertAlmostEqual(report["summary"]["Evaluation/Success_Rate"]["mean"], .55)
        self.assertEqual(report["summary"]["Evaluation/Success_Rate"]["n"], 10)
        # Rollout metrics retain their actual evaluation epochs; no invented values at epoch 51.
        self.assertFalse(any(row["epoch"] == 51 and row["metric"].startswith("Rollout/") for row in report["curves"]))
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

    def test_failed_trial_does_not_become_a_partial_mean_and_can_resume(self):
        self.fail_seed_once = 2
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            with self.assertRaises(subprocess.CalledProcessError):
                train_trials.run(self.args(n_trials=3))
        group_dir = self.root / "outputs/test-10"
        manifest = json.loads((group_dir / "manifest.json").read_text())
        self.assertEqual([entry["status"] for entry in manifest["trials"]], ["complete", "failed", "pending"])
        self.assertFalse((group_dir / "aggregate/summary.json").exists())
        with self.assertRaisesRegex(ValueError, "Complete every"):
            train_trials.run(self.args(aggregate_only=True))
        self.calls.clear()
        with mock.patch.object(train_trials.subprocess, "run", side_effect=self.child):
            result = train_trials.run(self.args(n_trials=3, resume=True))
        resumed_commands = [command for command, _ in self.calls if command[2].endswith(".train")]
        self.assertEqual(len(resumed_commands), 2)
        self.assertIn("--resume", resumed_commands[0])
        self.assertNotIn("--resume", resumed_commands[1])
        self.assertEqual(result["n_trials"], 3)

    def test_nonfinite_values_have_visible_counts_and_no_fabricated_sd(self):
        stats = describe([1, None, float("nan"), 3])
        self.assertEqual(stats["mean"], 2)
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["std"], math.sqrt(2))
        self.assertEqual(describe([1])["std"], None)
        self.assertEqual(describe([])["mean"], None)

    def test_journal_repeated_epochs_and_rng_shared_with_sampler(self):
        journal = ScalarJournal(self.root)
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
                                           experiment=types.SimpleNamespace(name="trial_07_seed_7",
                                               logging=types.SimpleNamespace(wandb_proj_name="cami")),
                                           to_dict=lambda: {"algo_name": "bc_cami"})
            # Warning colors/progress bars do not affect scalar logging.
            termcolor = types.SimpleNamespace(colored=lambda text, *args, **kwargs: text)
            progress = types.SimpleNamespace(tqdm=type("tqdm", (object,), {}))
            spec = importlib.util.spec_from_file_location("trial_test_logger", train_trials.REPO_ROOT / "robomimic/utils/log_utils.py")
            logger_module = importlib.util.module_from_spec(spec)
            with mock.patch.dict(sys.modules, {"wandb": wandb, "termcolor": termcolor, "tqdm": progress}), \
                 mock.patch.dict(os.environ, {"WANDB_RUN_GROUP": "ten-seeds", "WANDB_JOB_TYPE": "training", "CAMI_TRIAL_INDEX": "7"}):
                spec.loader.exec_module(logger_module)
                logger = logger_module.DataLogger(temporary, config, log_tb=False, log_wandb=True)
                logger.record("Train/Loss", 2, 50)
                logger.record("Rollout/Success_Rate/square", .8, 50, log_stats=True)
                logger.flush(50)
                logger.close()
            self.assertEqual(wandb.init.call_args.kwargs["group"], "ten-seeds")
            self.assertEqual(wandb.init.call_args.kwargs["job_type"], "training")
            self.assertEqual(wandb.init.call_args.kwargs["config"]["trial_seed"], 7)
            history = read_history(Path(temporary) / "metrics.jsonl")
            self.assertEqual(history[50]["Train/Loss"], 2)
            self.assertEqual(history[50]["Rollout/Success_Rate/square"], .8)
            self.assertEqual(history[50]["Rollout/Success_Rate/square/mean"], .8)
            self.assertEqual(fake.rows[-1], ({"epoch": 50}, {"step": 50, "commit": True}))
            self.assertTrue(fake.finished)

    def test_aggregate_wandb_run_logs_means_sd_se_and_counts_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            (result_dir / "curves.csv").touch()
            (result_dir / "summary.json").write_text("{}")
            fake = FakeRun()

            class Artifact:
                def __init__(self, *args, **kwargs):
                    self.paths = []

                def add_file(self, path):
                    self.paths.append(path)

            wandb = types.SimpleNamespace(init=mock.Mock(return_value=fake), Artifact=Artifact,
                                          Table=lambda **kwargs: types.SimpleNamespace(**kwargs))
            stats = {"mean": .55, "std": .3, "se": .3 / math.sqrt(10), "n": 10}
            report = {"n_trials": 10, "epochs": 2000,
                      "curves": [{"epoch": 50, "metric": "Rollout/Success_Rate/square", **stats}],
                      "summary": {"Evaluation/Success_Rate": stats}, "trials": []}
            manifest = {"group": "ten-seeds", "settings": {"seeds": list(range(1, 11)),
                        "eval_seed": 10000, "eval_rollouts": 50,
                        "config": {"experiment": {"logging": {"wandb_proj_name": "cami"}}}}}
            args = types.SimpleNamespace(wandb_mode="online", wandb_entity=None, wandb_project=None)
            cosmetics = {"wandb": wandb,
                         "termcolor": types.SimpleNamespace(colored=lambda text, *args, **kwargs: text),
                         "tqdm": types.SimpleNamespace(tqdm=type("tqdm", (object,), {}))}
            with mock.patch.dict(sys.modules, cosmetics):
                train_trials.publish_report(report, manifest, result_dir, args)
            self.assertEqual(wandb.init.call_args.kwargs["group"], "ten-seeds")
            self.assertEqual(wandb.init.call_args.kwargs["job_type"], "aggregate")
            values, options = fake.rows[0]
            self.assertEqual(values["epoch"], 50)
            self.assertEqual(values["Mean/Rollout/Success_Rate/square"], .55)
            self.assertEqual(values["N/Rollout/Success_Rate/square"], 10)
            self.assertEqual(values["Std/Rollout/Success_Rate/square"], .3)
            self.assertAlmostEqual(values["SE/Rollout/Success_Rate/square"], .3 / math.sqrt(10))
            self.assertEqual(options["step"], 50)
            self.assertEqual(fake.summary["Summary/Evaluation/Success_Rate/n"], 10)
            self.assertEqual(len(fake.artifact.paths), 2)
            self.assertTrue(fake.finished)


if __name__ == "__main__":
    unittest.main()
