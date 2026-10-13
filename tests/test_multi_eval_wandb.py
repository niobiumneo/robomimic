"""Check W&B evaluation logging without a GPU, simulator, or W&B account.

Run: python -m unittest discover -s tests -p test_multi_eval_wandb.py -v
The evaluator's real episode loop runs against a small deterministic environment.
"""
import contextlib
import csv
import importlib.util
import io
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "robomimic/scripts/run_trained_agent_multi_eval.py"


def package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def load_evaluator():
    """Stub only simulator/ML dependencies; exercise the actual logging and rollout code."""
    torch = types.ModuleType("torch")
    torch.cpu_state = 0
    torch.cuda_state = 0

    def seed(value):
        torch.cpu_state = value
        torch.cuda_state = value

    torch.manual_seed = seed
    torch.get_rng_state = lambda: torch.cpu_state
    torch.set_rng_state = lambda state: setattr(torch, "cpu_state", state)
    torch.cuda = types.SimpleNamespace(
        is_initialized=lambda: True,
        get_rng_state_all=lambda: [torch.cuda_state],
        set_rng_state_all=lambda states: setattr(torch, "cuda_state", states[0]),
    )
    torch_utils = types.ModuleType("robomimic.utils.torch_utils")
    torch_utils.get_torch_device = lambda **kwargs: "cpu"
    tensor_utils = types.ModuleType("robomimic.utils.tensor_utils")
    tensor_utils.list_of_flat_dict_to_dict_of_list = lambda rows: {
        key: [row[key] for row in rows] for key in rows[0]
    }
    env_base = types.ModuleType("robomimic.envs.env_base")
    env_base.EnvBase = type("EnvBase", (), {})
    wrappers = types.ModuleType("robomimic.envs.wrappers")
    wrappers.EnvWrapper = type("EnvWrapper", (), {})
    algo = types.ModuleType("robomimic.algo")
    algo.RolloutPolicy = type("RolloutPolicy", (), {})
    imageio = package("imageio")
    imageio.v2 = types.ModuleType("imageio.v2")
    modules = {
        "torch": torch, "h5py": types.ModuleType("h5py"), "imageio": imageio, "imageio.v2": imageio.v2,
        "robomimic": package("robomimic"), "robomimic.utils": package("robomimic.utils"),
        "robomimic.utils.file_utils": types.ModuleType("robomimic.utils.file_utils"),
        "robomimic.utils.torch_utils": torch_utils, "robomimic.utils.tensor_utils": tensor_utils,
        "robomimic.envs": package("robomimic.envs"), "robomimic.envs.env_base": env_base,
        "robomimic.envs.wrappers": wrappers, "robomimic.algo": algo,
    }
    # Pure statistics helpers are real code, not simulator/ML dependencies.
    modules["robomimic.utils"].__path__ = [str(SCRIPT.parents[1] / "utils")]
    spec = importlib.util.spec_from_file_location("multi_eval_wandb_test_target", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class FakeWandb:
    def __init__(self, evaluator, fail_init=False, fail_log=False):
        self.evaluator = evaluator
        self.fail_init = fail_init
        self.fail_log = fail_log
        self.runs = []
        self.init_calls = []

    def consume_random(self):
        # Deliberately consume all RNG streams to detect any logging-induced changes.
        random.random()
        np.random.random()
        self.evaluator.torch.cpu_state += 100
        self.evaluator.torch.cuda_state += 100

    def init(self, **kwargs):
        self.consume_random()
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise RuntimeError("login unavailable")
        run = FakeRun(self, "run-{}".format(len(self.runs) + 1))
        self.runs.append(run)
        return run

    def Table(self, **kwargs):
        self.consume_random()
        return types.SimpleNamespace(**kwargs)

    def Video(self, path, **kwargs):
        self.consume_random()
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return types.SimpleNamespace(path=path, **kwargs)

    def Artifact(self, name, type):
        self.consume_random()
        return FakeArtifact(name, type)


class FakeArtifact:
    def __init__(self, name, kind):
        self.name, self.type = name, kind
        self.files = []

    def add_file(self, path):
        self.files.append(path)


class FakeRun:
    def __init__(self, sdk, run_id):
        self.sdk, self.id = sdk, run_id
        self.url = "https://example.invalid/" + run_id
        self.summary, self.history, self.definitions, self.artifacts = {}, [], [], []
        self.exit_code = None

    def define_metric(self, name, **kwargs):
        self.sdk.consume_random()
        self.definitions.append((name, kwargs))

    def log(self, values):
        self.sdk.consume_random()
        if self.sdk.fail_log:
            self.sdk.fail_log = False
            raise RuntimeError("connection lost")
        self.history.append(values.copy())

    def log_artifact(self, artifact):
        self.sdk.consume_random()
        self.artifacts.append(artifact)

    def finish(self, exit_code):
        self.sdk.consume_random()
        self.exit_code = exit_code


class EvaluationWandbTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = load_evaluator()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.envs = []
        evaluator = self.evaluator

        class Env(evaluator.EnvBase):
            rollout_exceptions = (RuntimeError,)

            def __init__(self):
                self.seeds, self.placements = [], []
                self.fail_at = None

            def seed(self, value):
                self.seeds.append(value)

            def reset(self):
                if self.fail_at == len(self.placements):
                    raise KeyError("simulator reset failed")
                self.placement = float(np.random.random())
                self.placements.append(self.placement)
                return {"object": np.array([self.placement])}

            def get_state(self):
                return {"states": np.array([self.placement])}

            def reset_to(self, state):
                return {"object": state["states"]}

            def step(self, action):
                success = float(self.placement > 0.2)
                return {"object": np.array([self.placement])}, success, True, {}

            def is_success(self):
                return {"task": self.placement > 0.2}

            def render(self, mode, height, width, camera_name):
                # Rendering must not change the randomness used by later episodes.
                random.random()
                np.random.random()
                evaluator.torch.cpu_state += 100
                evaluator.torch.cuda_state += 100
                color = 10 if camera_name == "agentview" else 20
                return np.full((height, width, 3), color, dtype=np.uint8)

        class Policy(evaluator.RolloutPolicy):
            def start_episode(self):
                pass

            def __call__(self, ob):
                evaluator.torch.cpu_state += 1
                evaluator.torch.cuda_state += 1
                return np.array([0.0])

        self.Env, self.Policy = Env, Policy

    def video_writer(self, path, **kwargs):
        """Capture frames/writer lifetime; no external encoder is needed for these checks."""
        evaluator = self.evaluator
        records = self.video_writers

        def consume():
            random.random()
            np.random.random()
            evaluator.torch.cpu_state += 100
            evaluator.torch.cuda_state += 100

        class Writer:
            def __init__(self):
                self.path, self.kwargs, self.frames, self.closed = path, kwargs, [], False
                consume()

            def append_data(self, frame):
                consume()
                self.frames.append((frame.shape, int(frame[0, 0, 0]), int(frame[0, -1, 0])))

            def close(self):
                consume()
                self.closed = True
                if self.frames:
                    Path(path).write_bytes(b"synthetic clip for evaluator tests")

        writer = Writer()
        records.append(writer)
        return writer

    def evaluate_with_videos(self, args, sdk=None, create_env=None):
        self.video_writers = []
        with mock.patch.object(self.evaluator.imageio, "get_writer", side_effect=self.video_writer, create=True):
            self.evaluate(args, sdk, create_env)

    def args(self, output="results", *extra):
        return self.evaluator.build_parser().parse_args([
            "--agents", "/models/best_seed42.pth", "--agent_names", "CaMI",
            "--n_trials", "1", "--rollouts_per_trial", "50", "--horizon", "400",
            "--seed", "42", "--results_dir", str(self.root / output), *extra,
        ])

    def create_env(self, checkpoint, device, args, env_name=None):
        env = self.Env()
        self.envs.append(env)
        return self.Policy(), env, args.horizon or 400

    def evaluate(self, args, sdk=None, create_env=None):
        np.random.seed(999)
        random.seed(123)
        self.evaluator.torch.manual_seed(711)
        with mock.patch.object(self.evaluator, "create_env_and_policy", side_effect=create_env or self.create_env), \
             mock.patch.dict(sys.modules, {"wandb": sdk}), contextlib.redirect_stdout(io.StringIO()):
            self.evaluator.run_multi_eval(args)

    def csv_rows(self, args, name):
        with (Path(args.results_dir) / name).open() as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def test_one_seed42_batch_logs_all_50_episodes_and_final_success(self):
        sdk = FakeWandb(self.evaluator)
        args = self.args("online", "--wandb-project", "cami-contact-state",
                         "--wandb-name", "seed42-eval", "--wandb-group", "comparison")
        self.evaluate(args, sdk)
        self.assertEqual(len(self.envs), 1)
        self.assertEqual(self.envs[0].seeds, [42])
        self.assertEqual(len(self.envs[0].placements), 50)
        self.assertGreater(len(set(self.envs[0].placements)), 1)
        init = sdk.init_calls[0]
        self.assertEqual((init["project"], init["name"], init["group"], init["job_type"]),
                         ("cami-contact-state", "seed42-eval", "comparison", "evaluation"))
        self.assertEqual(init["config"]["eval_seed"], 42)
        self.assertEqual(init["config"]["rollout_horizon"], 400)
        run = sdk.runs[0]
        episodes = [row for row in run.history if "Rollout/Success_Rate" in row]
        self.assertEqual([row["rollout"] for row in episodes], list(range(1, 51)))
        self.assertEqual({row["trial_seed"] for row in episodes}, {42})
        success = sum(row["Rollout/Success_Rate"] for row in episodes)
        self.assertEqual(run.summary["Evaluation/Num_Rollouts"], 50)
        self.assertEqual(run.summary["Evaluation/Num_Success"], success)
        self.assertEqual(run.summary["Evaluation/Success_Rate"], success / 50)
        self.assertEqual(episodes[-1]["Running/Success_Rate"], success / 50)
        tables = run.history[-1]
        self.assertEqual(len(tables["rollout_results"].data), 50)
        self.assertEqual(len(tables["trial_results"].data), 1)
        self.assertEqual({Path(path).name for path in run.artifacts[0].files},
                         {"trial_results.csv", "rollout_results.csv"})
        self.assertTrue(all(Path(path).is_file() for path in run.artifacts[0].files))
        self.assertEqual(run.exit_code, 0)

    def test_logging_preserves_episode_sequence_and_python_numpy_torch_cuda_states(self):
        baseline = self.args("baseline")
        self.evaluate(baseline)
        placements = self.envs[0].placements.copy()
        expected_python = random.getstate()
        expected_numpy = np.random.get_state()
        expected_torch = self.evaluator.torch.cpu_state
        expected_cuda = self.evaluator.torch.cuda_state
        logged = self.args("logged", "--wandb-project", "cami")
        self.evaluate(logged, FakeWandb(self.evaluator))
        self.assertEqual(self.envs[1].placements, placements)
        self.assertEqual(self.csv_rows(baseline, "rollout_results.csv"),
                         self.csv_rows(logged, "rollout_results.csv"))
        self.assertEqual(random.getstate(), expected_python)
        numpy_state = np.random.get_state()
        self.assertEqual(numpy_state[0], expected_numpy[0])
        np.testing.assert_array_equal(numpy_state[1], expected_numpy[1])
        self.assertEqual(numpy_state[2:], expected_numpy[2:])
        self.assertEqual(self.evaluator.torch.cpu_state, expected_torch)
        self.assertEqual(self.evaluator.torch.cuda_state, expected_cuda)

    def test_disabled_mode_has_no_wandb_dependency(self):
        args = self.args("disabled", "--wandb-project", "cami", "--wandb-mode", "disabled")
        self.evaluate(args)
        self.assertEqual(len(self.csv_rows(args, "rollout_results.csv")[1]), 50)

    def test_offline_mode_is_explicit_and_records_the_budget(self):
        sdk = FakeWandb(self.evaluator)
        self.evaluate(self.args("offline", "--wandb", "--wandb-mode", "offline"), sdk)
        self.assertEqual(sdk.init_calls[0]["mode"], "offline")
        self.assertEqual(sdk.init_calls[0]["project"], "cami-contact-state")
        self.assertEqual(sdk.runs[0].summary["Evaluation/Num_Rollouts"], 50)

    def test_initialization_failure_stops_before_rollouts(self):
        sdk = FakeWandb(self.evaluator, fail_init=True)
        with self.assertRaisesRegex(RuntimeError, "W&B evaluation initialization failed"):
            self.evaluate(self.args("init-failure", "--wandb-project", "cami"), sdk)
        self.assertEqual(self.envs[0].placements, [])

    def test_missing_wandb_has_install_guidance_before_rollouts(self):
        with self.assertRaisesRegex(RuntimeError, "pip install wandb"):
            self.evaluate(self.args("missing-sdk", "--wandb-project", "cami"))
        self.assertEqual(self.envs[0].placements, [])

    def test_streaming_failure_does_not_reduce_the_episode_budget_or_drop_csvs(self):
        sdk = FakeWandb(self.evaluator, fail_log=True)
        args = self.args("log-failure", "--wandb-project", "cami")
        self.evaluate(args, sdk)
        self.assertEqual(len(self.envs[0].placements), 50)
        self.assertEqual(len(self.csv_rows(args, "rollout_results.csv")[1]), 50)
        self.assertEqual(sdk.runs[0].summary["Evaluation/Num_Rollouts"], 50)
        self.assertEqual(sdk.runs[0].exit_code, 1)

    def test_simulation_failure_closes_run_and_marks_partial_evaluation(self):
        sdk = FakeWandb(self.evaluator)

        def failing_environment(*args, **kwargs):
            policy, env, horizon = self.create_env(*args, **kwargs)
            env.fail_at = 3
            return policy, env, horizon

        with self.assertRaisesRegex(KeyError, "simulator reset failed"):
            self.evaluate(self.args("sim-failure", "--wandb-project", "cami"), sdk, failing_environment)
        self.assertEqual(sdk.runs[0].summary["Evaluation/Num_Rollouts"], 3)
        self.assertFalse(sdk.runs[0].summary["Evaluation/Complete"])
        self.assertEqual(sdk.runs[0].exit_code, 1)

    def test_multiple_trials_seed_once_each_and_keep_csv_indices(self):
        sdk = FakeWandb(self.evaluator)
        args = self.args("trials", "--wandb-project", "cami", "--n_trials", "2")
        self.evaluate(args, sdk)
        self.assertEqual(self.envs[0].seeds, [42, 43])
        self.assertEqual(len(self.envs[0].placements), 100)
        _, trials = self.csv_rows(args, "trial_results.csv")
        self.assertEqual([(row["trial"], row["trial_seed"]) for row in trials], [("0", "42"), ("1", "43")])
        self.assertEqual(sdk.runs[0].summary["Evaluation/Num_Trials"], 2)

    def test_multiple_checkpoints_get_separate_runs_and_running_means(self):
        sdk = FakeWandb(self.evaluator)
        args = self.args("models", "--wandb-project", "cami", "--wandb-name", "eval")
        args.agents, args.agent_names = ["/models/bc.pth", "/models/cami.pth"], ["BC", "CaMI"]
        self.evaluate(args, sdk)
        self.assertEqual(len(sdk.runs), 2)
        self.assertEqual([call["name"] for call in sdk.init_calls],
                         ["eval-BC-checkpoint_env", "eval-CaMI-checkpoint_env"])
        self.assertEqual([run.summary["Evaluation/Num_Rollouts"] for run in sdk.runs], [50, 50])
        for run in sdk.runs:
            episodes = [row for row in run.history if "rollout" in row]
            self.assertEqual([row["rollout"] for row in episodes], list(range(1, 51)))
        self.assertEqual(len(self.csv_rows(args, "rollout_results.csv")[1]), 100)

    def test_completed_trial_peaks_and_summary_match_the_logged_rollouts(self):
        sdk = FakeWandb(self.evaluator)
        self.evaluate(self.args("trial-stats", "--wandb-project", "cami", "--n_trials", "2"), sdk)
        run = sdk.runs[0]
        trials = [row for row in run.history if "Trial/Success_Rate" in row]
        for trial in trials:
            episodes = [row for row in run.history if "Rollout/Success_Rate" in row
                        and row["trial"] == trial["trial"]]
            values = [row["Rollout/Success_Rate"] for row in episodes]
            rates = [sum(values[:i]) / i for i in range(1, 51)]
            peak = max(rates)
            self.assertEqual(trial["Trial/Peak_Running_Success_Rate"], peak)
            self.assertEqual(trial["Trial/Peak_Running_Num_Rollouts"], rates.index(peak) + 1)
            self.assertEqual(trial["Trial/Return_Max"], max(row["Rollout/Return"] for row in episodes))
            self.assertEqual(trial["Trial/Success_Rate"], sum(values) / 50)
        self.assertEqual(run.summary["Evaluation/Success_Rate_Max_Over_Trials"],
                         max(row["Trial/Success_Rate"] for row in trials))
        self.assertIn(("Trial/Success_Rate", {"step_metric": "trial", "summary": "max"}), run.definitions)

    def test_completed_trial_csv_survives_failure_in_the_next_trial(self):
        sdk = FakeWandb(self.evaluator)
        args = self.args("trial-interrupt", "--wandb-project", "cami", "--n_trials", "2")
        def failing_environment(*values, **kwargs):
            policy, env, horizon = self.create_env(*values, **kwargs)
            env.fail_at = 53
            return policy, env, horizon
        with self.assertRaisesRegex(KeyError, "simulator reset failed"):
            self.evaluate(args, sdk, failing_environment)
        folder = Path(args.results_dir) / "CaMI/checkpoint_env"
        with (folder / "trial_results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["num_rollouts"], "50")
        with (folder / "rollout_results.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 50)
        self.assertEqual(sdk.runs[0].summary["Evaluation/Num_Rollouts"], 53)
        self.assertFalse(sdk.runs[0].summary["Evaluation/Complete"])

    def test_trial_statistics_stream_after_every_rollout_and_reset_per_trial(self):
        sdk = FakeWandb(self.evaluator)
        self.evaluate(self.args("live-trial-stats", "--wandb-project", "cami",
                                "--n_trials", "2", "--rollouts_per_trial", "4"), sdk)
        run = sdk.runs[0]
        episodes = [row for row in run.history if "Rollout/Success_Rate" in row]
        self.assertEqual(len(episodes), 8)
        for trial in (1, 2):
            rows = [row for row in episodes if row["trial"] == trial]
            for count, row in enumerate(rows, 1):
                prefix = rows[:count]
                successes = [item["Rollout/Success_Rate"] for item in prefix]
                rates = [sum(successes[:i]) / i for i in range(1, count + 1)]
                self.assertEqual(row["TrialRunning/Num_Rollouts"], count)
                self.assertEqual(row["TrialRunning/Success_Rate"], sum(successes) / count)
                self.assertEqual(row["TrialRunning/Peak_Running_Success_Rate"], max(rates))
                self.assertEqual(row["TrialRunning/Peak_Running_Num_Rollouts"], rates.index(max(rates)) + 1)
                self.assertEqual(row["TrialRunning/Return_Max"], max(item["Rollout/Return"] for item in prefix))
                self.assertAlmostEqual(row["TrialRunning/Return_Std"],
                                       np.std([item["Rollout/Return"] for item in prefix]))
                self.assertNotIn("Trial/Success_Rate", row)
        self.assertIn(("TrialRunning/*", {"step_metric": "rollout"}), run.definitions)

    def test_best_trial_summary_is_visible_before_the_next_trial_starts(self):
        sdk = FakeWandb(self.evaluator)
        snapshots = []
        def observing_environment(*values, **kwargs):
            policy, env, horizon = self.create_env(*values, **kwargs)
            original_reset = env.reset
            def reset():
                if len(env.placements) == 4:
                    snapshots.append(sdk.runs[0].summary.copy())
                return original_reset()
            env.reset = reset
            return policy, env, horizon
        self.evaluate(self.args("live-summary", "--wandb-project", "cami",
                                "--n_trials", "2", "--rollouts_per_trial", "4"),
                      sdk, observing_environment)
        self.assertEqual(len(snapshots), 1)
        first_trial = next(row for row in sdk.runs[0].history if "Trial/Success_Rate" in row)
        self.assertEqual(snapshots[0]["Evaluation/Num_Trials"], 1)
        self.assertEqual(snapshots[0]["Evaluation/Num_Rollouts"], 4)
        self.assertEqual(snapshots[0]["Evaluation/Best_Trial"], 1)
        self.assertEqual(snapshots[0]["Evaluation/Success_Rate_Max_Over_Trials"],
                         first_trial["Trial/Success_Rate"])

    def test_trial_summary_connection_failure_preserves_rollouts_and_csvs(self):
        baseline = self.args("summary-baseline", "--n_trials", "2", "--rollouts_per_trial", "4")
        self.evaluate(baseline)
        expected_placements = self.envs[-1].placements.copy()
        sdk = FakeWandb(self.evaluator)
        original_init = sdk.init
        class Summary(dict):
            failed_once = False
            def update(self, *args, **kwargs):
                sdk.consume_random()
                if not self.failed_once:
                    self.failed_once = True
                    raise RuntimeError("trial summary connection lost")
                return super().update(*args, **kwargs)
        def init(**kwargs):
            run = original_init(**kwargs)
            run.summary = Summary()
            return run
        sdk.init = init
        args = self.args("summary-failure", "--wandb-project", "cami",
                         "--n_trials", "2", "--rollouts_per_trial", "4")
        self.evaluate(args, sdk)
        self.assertEqual(self.envs[-1].placements, expected_placements)
        self.assertEqual(len(self.csv_rows(args, "rollout_results.csv")[1]), 8)
        self.assertEqual(len(self.csv_rows(args, "trial_results.csv")[1]), 2)
        self.assertEqual(sdk.runs[0].summary["Evaluation/Num_Rollouts"], 8)
        self.assertEqual(sdk.runs[0].exit_code, 1)

    def test_defaults_and_underscore_aliases_remain_supported(self):
        parser = self.evaluator.build_parser()
        defaults = parser.parse_args(["--agents", "model.pth"])
        self.assertEqual((defaults.n_trials, defaults.rollouts_per_trial, defaults.seed), (10, 50, 0))
        args = self.args("aliases", "--wandb_project", "cami", "--wandb_entity", "team")
        self.assertEqual((args.wandb_project, args.wandb_entity), ("cami", "team"))

    def test_invalid_budget_fails_before_model_loading(self):
        for field in ("n_trials", "rollouts_per_trial", "horizon"):
            with self.subTest(field=field):
                args = self.args()
                setattr(args, field, 0)
                with mock.patch.object(self.evaluator, "create_env_and_policy") as create:
                    with self.assertRaisesRegex(ValueError, "positive"):
                        self.evaluator.run_multi_eval(args)
                create.assert_not_called()

    def test_50_seed0_rollouts_save_both_outcomes_and_csv_video_paths(self):
        args = self.args("videos", "--save-videos", "--video_skip", "1", "--seed", "0",
                         "--camera_names", "agentview", "robot0_eye_in_hand")
        self.evaluate_with_videos(args)
        folder = Path(args.results_dir) / "CaMI/checkpoint_env/videos/trial_01_seed_0"
        self.assertEqual(len(list(folder.glob("*.mp4"))), 50)
        self.assertTrue(list(folder.glob("successful_rollout_*.mp4")))
        self.assertTrue(list(folder.glob("failed_rollout_*.mp4")))
        self.assertEqual(self.envs[0].seeds, [0])
        self.assertEqual(len(self.video_writers), 50)
        self.assertTrue(all(writer.closed for writer in self.video_writers))
        self.assertEqual({writer.kwargs["fps"] for writer in self.video_writers}, {20})
        self.assertTrue(all(writer.frames == [((512, 1024, 3), 10, 20)] for writer in self.video_writers))
        _, rows = self.csv_rows(args, "rollout_results.csv")
        self.assertTrue(all(Path(row["video"]).is_file() for row in rows))
        for row in rows:
            expected = "successful" if float(row["success"]) else "failed"
            self.assertEqual(Path(row["video"]).name, "{}_rollout_{:03d}.mp4".format(expected, int(row["rollout"])))

    def test_video_and_wandb_record_the_same_episode_sequence_as_metrics_only(self):
        baseline = self.args("no-video", "--seed", "0")
        self.evaluate(baseline)
        placements = self.envs[0].placements.copy()
        torch_state, cuda_state = self.evaluator.torch.cpu_state, self.evaluator.torch.cuda_state
        args = self.args("recorded", "--seed", "0", "--wandb-project", "cami", "--wandb-videos")
        sdk = FakeWandb(self.evaluator)
        self.evaluate_with_videos(args, sdk)
        self.assertEqual(self.envs[1].placements, placements)
        _, baseline_rows = self.csv_rows(baseline, "rollout_results.csv")
        _, recorded_rows = self.csv_rows(args, "rollout_results.csv")
        self.assertEqual([{key: row[key] for key in baseline_rows[0]} for row in recorded_rows], baseline_rows)
        self.assertEqual(self.evaluator.torch.cpu_state, torch_state)
        self.assertEqual(self.evaluator.torch.cuda_state, cuda_state)
        episodes = [row for row in sdk.runs[0].history if "Rollout/Success_Rate" in row]
        self.assertEqual(len(episodes), 50)
        self.assertTrue(all(Path(row["Rollout/Video"].path).is_file() for row in episodes))
        self.assertEqual(sdk.runs[0].exit_code, 0)
        self.assertEqual(sdk.runs[0].summary["Evaluation/Num_Rollouts"], 50)

    def test_video_writers_close_when_simulation_raises(self):
        def failing_environment(*args, **kwargs):
            policy, env, horizon = self.create_env(*args, **kwargs)
            env.fail_at = 0
            return policy, env, horizon

        args = self.args("failed-video", "--save-videos")
        with self.assertRaisesRegex(KeyError, "simulator reset failed"):
            self.evaluate_with_videos(args, create_env=failing_environment)
        self.assertEqual(len(self.video_writers), 1)
        self.assertTrue(self.video_writers[0].closed)

    def test_custom_video_root_and_frame_rate(self):
        args = self.args("custom", "--video_path", str(self.root / "clips"), "--video-fps", "60",
                         "--rollouts_per_trial", "2", "--seed", "0")
        self.evaluate_with_videos(args)
        folder = self.root / "clips/CaMI/checkpoint_env/trial_01_seed_0"
        self.assertEqual(len(list(folder.glob("*.mp4"))), 2)
        self.assertEqual({writer.kwargs["fps"] for writer in self.video_writers}, {60})

    def test_video_recording_rejects_overwriting_an_existing_batch(self):
        args = self.args("protected", "--save-videos", "--rollouts_per_trial", "2")
        self.evaluate_with_videos(args)
        folder = Path(args.results_dir) / "CaMI/checkpoint_env/videos/trial_01_seed_42"
        original = {path.name: path.read_bytes() for path in folder.glob("*.mp4")}
        with self.assertRaises(FileExistsError):
            self.evaluate_with_videos(args)
        self.assertEqual({path.name: path.read_bytes() for path in folder.glob("*.mp4")}, original)

    def test_invalid_video_options_fail_before_loading_the_model(self):
        for extra in (("--video_skip", "0"), ("--video-fps", "0"), ("--render", "--save-videos")):
            with self.subTest(extra=extra):
                args = self.args("bad-video", *extra)
                with mock.patch.object(self.evaluator, "create_env_and_policy") as create:
                    with self.assertRaises((ValueError, AssertionError)):
                        self.evaluator.run_multi_eval(args)
                create.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe are not installed")
    def test_real_mp4_encoding_produces_playable_episode_clips(self):
        """Exercise the writer boundary with real encoding, while keeping policy/simulator synthetic."""
        class Writer:
            def __init__(self, path, fps, **kwargs):
                self.process = subprocess.Popen([
                    shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", "512x512", "-r", str(fps), "-i", "pipe:0",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
                ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            def append_data(self, frame):
                self.process.stdin.write(frame.tobytes())

            def close(self):
                self.process.stdin.close()
                self.process.wait(timeout=15)
                error = self.process.stderr.read().decode()
                self.process.stderr.close()
                if self.process.returncode:
                    raise RuntimeError(error)

        args = self.args("real-mp4", "--save-videos", "--video_skip", "1", "--seed", "0",
                         "--rollouts_per_trial", "3")
        with mock.patch.object(self.evaluator.imageio, "get_writer", side_effect=Writer, create=True):
            self.evaluate(args)
        _, rows = self.csv_rows(args, "rollout_results.csv")
        for row in rows:
            metadata = subprocess.check_output([
                shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
                "-of", "json", row["video"],
            ], text=True)
            import json
            stream = json.loads(metadata)["streams"][0]
            self.assertEqual((stream["width"], stream["height"], stream["nb_frames"], stream["r_frame_rate"]),
                             (512, 512, "1", "20/1"))


if __name__ == "__main__":
    unittest.main()
