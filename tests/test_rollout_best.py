"""Check checkpoint ranking and preservation of successful / failed paths."""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from robomimic.scripts import rollout_best as helper


def test_select_best_uses_evaluated_weights_and_latest_run(tmp_path):
    old = tmp_path / "20260101000000" / "models"
    current = tmp_path / "20260930000000" / "models"
    old.mkdir(parents=True)
    current.mkdir(parents=True)
    (old / "model_epoch_50_square_image_84_with_force_success_1.0.pth").touch()
    best = current / "model_epoch_100_square_image_84_with_force_success_0.8.pth"
    best.touch()
    (current / "model_epoch_50_square_image_84_with_force_success_0.6.pth").touch()
    (current / "model_epoch_150.pth").touch()
    # Later weights inherit the historical maximum but are not the best model.
    torch.save({"variable_state": {"epoch": 151, "best_success_rate": {"square": 0.8}}},
               current.parent / "last.pth")
    path, selection = helper.select_checkpoint(tmp_path)
    assert path == best
    assert selection == {"metric_key": "square_image_84_with_force",
                         "training_success_rate": 0.8, "epoch": 100}
    assert helper.select_checkpoint(current)[0] == best


def test_selection_requires_unambiguous_metric_and_handles_return_suffix(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    checkpoint = models / "model_epoch_50_square_return_12.0_square_success_0.8_tool_hang_return_3.0_tool_hang_success_0.4.pth"
    checkpoint.touch()
    with pytest.raises(ValueError, match="Multiple rollout metrics"):
        helper.select_checkpoint(tmp_path)
    assert helper.select_checkpoint(tmp_path, "tool_hang")[1]["training_success_rate"] == 0.4
    assert helper.select_checkpoint(tmp_path, "square")[1]["training_success_rate"] == 0.8
    with pytest.raises(ValueError, match="No matching"):
        helper.select_checkpoint(tmp_path, "missing")


class FakePolicy:
    def __init__(self):
        self.policy = SimpleNamespace(global_config=SimpleNamespace(use_goals=False))
        self.episodes = 0

    def start_episode(self):
        self.episodes += 1

    def __call__(self, ob, goal=None):
        return np.array([0.25])


class FakeEnv:
    """Three episodes: success at step 2, timeout, simulator error at step 2."""
    rollout_exceptions = (RuntimeError,)

    def __init__(self):
        self.episode = -1
        self.closed = False

    def reset(self):
        self.episode += 1
        self.step_index = 0
        return {}

    def get_state(self):
        return {"states": np.array([self.step_index]), "model": "<mujoco/>",
                "ep_meta": "{}"}

    def step(self, action):
        if self.episode == 2 and self.step_index == 1:
            raise RuntimeError("simulator failed")
        self.step_index += 1
        return {}, 1.0, False, {}

    def is_success(self):
        return {"task": self.episode == 0 and self.step_index == 2}

    def render(self, **kwargs):
        return np.full((32, 32, 3), self.step_index * 50, dtype=np.uint8)

    def serialize(self):
        return {"env_name": "test", "type": 1, "env_kwargs": {}}

    def close(self):
        self.closed = True


@pytest.mark.parametrize("keep_failures,expected_videos", [(False, 1), (True, 3)])
def test_collect_export_and_video_encoding(tmp_path, monkeypatch, keep_failures, expected_videos):
    checkpoint = tmp_path / "model.pth"
    torch.save({"env_metadata": {}}, checkpoint)
    config = SimpleNamespace(experiment=SimpleNamespace(rollout=SimpleNamespace(horizon=3)))
    policy, env = FakePolicy(), FakeEnv()
    monkeypatch.setattr(helper.FileUtils, "config_from_checkpoint", lambda **kw: (config, kw["ckpt_dict"]))
    monkeypatch.setattr(helper.FileUtils, "policy_from_checkpoint", lambda **kw: (policy, kw["ckpt_dict"]))
    monkeypatch.setattr(helper.FileUtils, "env_from_checkpoint", lambda **kw: (env, kw["ckpt_dict"]))
    output = tmp_path / "export"
    args = argparse.Namespace(checkpoint=str(checkpoint), run_dir=None, metric_key=None,
                              cpu=True, horizon=None, output_dir=str(output), seed=10000,
                              n_rollouts=3, camera_names=["agentview", "wrist"],
                              video_skip=3, fps=20, keep_failures=keep_failures, no_stitch=False)
    helper.run(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["num_errors"] == 1
    assert [r["Horizon"] for r in summary["episodes"]] == [2, 3, 1]
    assert [r["seed"] for r in summary["episodes"]] == [10000, 10001, 10002]
    assert policy.episodes == 3 and env.closed
    clips = {path.name for path in output.glob("*_rollout_*.mp4")}
    assert len(clips) == expected_videos
    # An episode is a rollout; "trial" now names a whole evaluation of many rollouts.
    assert [r["rollout"] for r in summary["episodes"]] == [0, 1, 2]
    assert all("trial" not in r for r in summary["episodes"])
    assert "successful_rollout_000_seed_10000.mp4" in clips
    assert clips == ({"successful_rollout_000_seed_10000.mp4", "failed_rollout_001_seed_10001.mp4",
                      "errors_rollout_002_seed_10002.mp4"} if keep_failures
                     else {"successful_rollout_000_seed_10000.mp4"})
    # The clips of each outcome are also joined; failures only exist as clips when they are kept.
    joined = {path.name for path in output.glob("*_ALL.mp4")}
    assert joined == ({"SUCCESSFUL_ALL.mp4", "FAILED_ALL.mp4"} if keep_failures else {"SUCCESSFUL_ALL.mp4"})
    assert set(summary["compilations"]) == {name[:-4] for name in joined}
    # Read real encoded videos, checking two camera views and final success frame.
    with imageio.get_reader(str(output / summary["episodes"][0]["video"])) as video:
        assert video.get_data(0).shape == (32, 64, 3)
        assert video.get_data(1).mean() == pytest.approx(100, abs=3)
    with h5py.File(output / "rollouts.hdf5", "r") as dataset:
        assert dataset["mask/successful"][:].tolist() == [b"demo_0"]
        assert dataset["mask/failed"][:].tolist() == [b"demo_1"]
        assert dataset["mask/errors"][:].tolist() == [b"demo_2"]
        assert dataset["data"].attrs["total"] == 6
        assert dataset["data/demo_0/states"][:].tolist() == [[0], [1]]
        assert dataset["data/demo_0/next_states"][:].tolist() == [[1], [2]]
        assert dataset["data/demo_0"].attrs["ep_meta"] == "{}"
        assert "simulator failed" in dataset["data/demo_2"].attrs["rollout_error"]
    with pytest.raises(FileExistsError):
        helper.run(args)


class ScriptedEnv(FakeEnv):
    """Rollout r follows script[r]: ("success", n) succeeds after n steps, ("fail", None) times out
    and ("error", n) breaks on its (n+1)th step. A frame's brightness says which rollout and step it
    shows, so decoded videos can be checked in order."""

    def __init__(self, script):
        super().__init__()
        self.script = script

    def step(self, action):
        kind, steps = self.script[self.episode]
        if kind == "error" and self.step_index == steps:
            raise RuntimeError("simulator failed")
        self.step_index += 1
        return {}, 1.0, False, {}

    def is_success(self):
        kind, steps = self.script[self.episode]
        return {"task": kind == "success" and self.step_index == steps}

    def render(self, **kwargs):
        return np.full((32, 32, 3), brightness(self.episode, self.step_index), dtype=np.uint8)


def brightness(rollout, step):
    return 10 + 35 * rollout + 2 * step


SCRIPT = [("success", 4), ("fail", None), ("success", 3), ("error", 2), ("fail", None), ("success", 5)]
FRAMES = [4, 6, 3, 2, 6, 5]  # horizon 6 for the timeouts; an error keeps the frames before it


def run_scripted(tmp_path, monkeypatch, project=None, policy=None, env_name=None, script=None, env_class=None, **changes):
    """Run the scripted rollouts; ``project`` is the W&B project in the checkpoint's config, if any."""
    checkpoint = tmp_path / "model.pth"
    metadata = {"env_kwargs": {"control_freq": 20}}
    if env_name:
        metadata["env_name"] = env_name
    torch.save({"env_metadata": metadata}, checkpoint)
    experiment = SimpleNamespace(rollout=SimpleNamespace(horizon=6))
    if project:
        experiment.logging = SimpleNamespace(wandb_proj_name=project)
    config = SimpleNamespace(experiment=experiment)
    env = (env_class or ScriptedEnv)(script or SCRIPT)
    monkeypatch.setattr(helper.FileUtils, "config_from_checkpoint", lambda **kw: (config, kw["ckpt_dict"]))
    monkeypatch.setattr(helper.FileUtils, "policy_from_checkpoint", lambda **kw: (policy or FakePolicy(), kw["ckpt_dict"]))
    monkeypatch.setattr(helper.FileUtils, "env_from_checkpoint", lambda **kw: (env, kw["ckpt_dict"]))
    values = dict(checkpoint=str(checkpoint), run_dir=None, metric_key=None, cpu=True, horizon=None,
                  output_dir=str(tmp_path / "export"), seed=10000, n_rollouts=len(SCRIPT),
                  camera_names=["agentview"], video_skip=1, fps=60, keep_failures=True, no_stitch=False)
    values.update(changes)
    summary = helper.run(argparse.Namespace(**values))
    return Path(values["output_dir"]), summary


def decode(path):
    with imageio.get_reader(str(path)) as video:
        return video.get_meta_data(), [float(frame.mean()) for frame in video]


def expected_brightness(rollouts):
    return [brightness(r, step) for r in rollouts for step in range(1, FRAMES[r] + 1)]


def test_every_outcome_is_joined_into_one_60_fps_video_in_rollout_order(tmp_path, monkeypatch):
    output, summary = run_scripted(tmp_path, monkeypatch)
    assert [r["status"] for r in summary["episodes"]] == [
        "successful", "failed", "successful", "errors", "failed", "successful"]
    assert [r["video_frames"] for r in summary["episodes"]] == FRAMES
    # Each rollout keeps its own file, successes and failures alike.
    clips = sorted(path.name for path in output.glob("*_rollout_*.mp4"))
    assert clips == ["errors_rollout_003_seed_10003.mp4", "failed_rollout_001_seed_10001.mp4",
                     "failed_rollout_004_seed_10004.mp4", "successful_rollout_000_seed_10000.mp4",
                     "successful_rollout_002_seed_10002.mp4", "successful_rollout_005_seed_10005.mp4"]
    for record in summary["episodes"]:
        meta, frames = decode(output / record["video"])
        assert len(frames) == record["video_frames"]
        assert meta["fps"] == pytest.approx(60)
    # The joined videos hold exactly those frames, in rollout order, at the same 60 fps.
    for name, rollouts in (("SUCCESSFUL_ALL", [0, 2, 5]), ("FAILED_ALL", [1, 3, 4])):
        meta, frames = decode(output / (name + ".mp4"))
        expected = expected_brightness(rollouts)
        assert len(frames) == len(expected) == sum(FRAMES[r] for r in rollouts)
        assert frames == pytest.approx(expected, abs=4)
        assert meta["duration"] == pytest.approx(len(expected) / 60, abs=0.02)
        assert meta["fps"] == pytest.approx(60, abs=0.5)
    # A failure means a timeout or a simulator error; both go into FAILED_ALL.
    with (output / "FAILED_ALL.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == list(helper.INDEX_FIELDS)
    assert [(r["clip"], r["rollout"], r["seed"], r["status"], r["file"]) for r in rows] == [
        ("1", "1", "10001", "failed", "failed_rollout_001_seed_10001.mp4"),
        ("2", "3", "10003", "errors", "errors_rollout_003_seed_10003.mp4"),
        ("3", "4", "10004", "failed", "failed_rollout_004_seed_10004.mp4")]
    assert [float(r["start_seconds"]) for r in rows] == [0, 0.1, 0.133]  # 6 and 2 frames at 60 fps
    assert [float(r["duration_seconds"]) for r in rows] == [0.1, 0.033, 0.1]
    assert summary["compilations"]["SUCCESSFUL_ALL"] == {
        "video": "SUCCESSFUL_ALL.mp4", "index": "SUCCESSFUL_ALL.csv", "clips": 3, "frames": 12, "seconds": 0.2}
    assert json.loads((output / "summary.json").read_text())["compilations"] == summary["compilations"]
    # Nothing is left over from joining.
    assert not list(output.glob("*.txt"))


def test_without_failure_videos_only_the_successes_are_joined(tmp_path, monkeypatch, capsys):
    output, summary = run_scripted(tmp_path, monkeypatch, keep_failures=False)
    assert sorted(summary["compilations"]) == ["SUCCESSFUL_ALL"]
    assert not (output / "FAILED_ALL.mp4").exists() and not (output / "FAILED_ALL.csv").exists()
    assert len(list(output.glob("failed_*.mp4"))) == 0
    assert len(decode(output / "SUCCESSFUL_ALL.mp4")[1]) == 12
    # The unsuccessful rollouts still exist as trajectories; the hint says how to get their videos.
    assert "3 unsuccessful rollouts have no video; add --keep-failures" in capsys.readouterr().out


def test_stitching_can_be_turned_off(tmp_path, monkeypatch):
    output, summary = run_scripted(tmp_path, monkeypatch, no_stitch=True)
    assert summary["compilations"] == {}
    assert not list(output.glob("*_ALL.*"))
    assert len(list(output.glob("*_rollout_*.mp4"))) == 6


def test_a_failed_join_does_not_fail_the_evaluation(tmp_path, monkeypatch, capsys):
    def broken(clips, target, frames):
        raise RuntimeError("ffmpeg could not join the clips: boom")

    monkeypatch.setattr(helper, "join_clips", broken)
    output, summary = run_scripted(tmp_path, monkeypatch)
    assert summary["compilations"] == {} and summary["n_rollouts_completed"] == 6
    assert len(list(output.glob("*_rollout_*.mp4"))) == 6
    assert "Could not join the videos (RuntimeError: ffmpeg could not join the clips: boom)" in capsys.readouterr().out
    assert json.loads((output / "summary.json").read_text())["compilations"] == {}


def test_joining_survives_quotes_in_folder_names_and_cleans_up(tmp_path):
    folder = tmp_path / "it's a trial"
    folder.mkdir()
    clips = []
    for number, level in enumerate((40, 120)):
        clip = folder / "clip_{}.mp4".format(number)
        helper.write_video(clip, [np.full((32, 32, 3), level, dtype=np.uint8)] * (number + 2), 60)
        clips.append(clip)
    helper.join_clips(clips, folder / "joined.mp4", 5)
    assert [round(value / 10) for value in decode(folder / "joined.mp4")[1]] == [4, 4, 12, 12, 12]
    assert sorted(path.name for path in folder.iterdir()) == ["clip_0.mp4", "clip_1.mp4", "joined.mp4"]
    # ffmpeg exits normally after skipping a clip it cannot open: that must not pass for a join,
    # and neither must a video with other than the expected number of frames.
    (folder / "broken.mp4").write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match=r"(?s)ffmpeg could not join the clips: .*broken\.mp4"):
        helper.join_clips([clips[0], folder / "broken.mp4"], folder / "bad.mp4", 4)
    with pytest.raises(RuntimeError, match="bad.mp4 has 5 frames, expected 6"):
        helper.join_clips(clips, folder / "bad.mp4", 6)
    assert not (folder / "bad.mp4").exists() and not list(folder.glob("*.txt"))


def test_the_speed_of_a_video_is_reported_against_the_control_rate(tmp_path, monkeypatch, capsys):
    ckpt = {"env_metadata": {"env_kwargs": {"control_freq": 20}}}
    speed = lambda fps, skip, meta=ckpt: helper.playback_speed(argparse.Namespace(fps=fps, video_skip=skip), meta)
    assert speed(60, 1) == (3.0, 20)
    assert speed(20, 1) == (1.0, 20)
    assert speed(4, 5)[0] == 1.0
    assert speed(60, 1, {"env_metadata": {}}) == (None, None)
    run_scripted(tmp_path, monkeypatch)
    assert ("Videos: every 1 control step(s) at 60 fps = 3x real time (20 Hz control; "
            "--fps 20 plays in real time)") in capsys.readouterr().out
    (tmp_path / "again").mkdir()
    run_scripted(tmp_path / "again", monkeypatch, fps=20)
    assert "at 20 fps = 1x real time (20 Hz control)" in capsys.readouterr().out


@pytest.mark.parametrize("algorithm", ["bc", "bc_cami"])
def test_actual_recurrent_checkpoint_restore_and_inference(tmp_path, algorithm):
    from robomimic.algo import algo_factory
    from robomimic.config import config_factory
    from robomimic.utils import obs_utils, train_utils

    template = Path(__file__).resolve().parents[1] / "robomimic/exps/templates/bc_cami_square.json"
    raw = json.loads(template.read_text())
    raw["algo_name"] = algorithm
    raw["train"]["cuda"] = False
    raw["observation"]["modalities"]["obs"]["rgb"] = []
    raw["observation"]["modalities"]["obs"]["low_dim"] = ["robot0_eef_pos"]
    raw["algo"]["actor_layer_dims"] = [8]
    raw["algo"]["rnn"]["hidden_dim"] = 8
    raw["algo"]["rnn"]["num_layers"] = 1
    if algorithm == "bc":
        raw["algo"].pop("cami")
        raw["algo"]["optim_params"] = {"policy": raw["algo"]["optim_params"]["policy"]}
        raw["train"]["dataset_keys"] = ["actions"]
    else:
        raw["algo"]["cami"]["continuous_contact"]["force_scale"] = 1.0
        raw["algo"]["cami"].update(policy_latent_dim=8, snippet_hidden_dim=8, contrastive_dim=8)
    config = config_factory(algorithm, dic=raw)
    obs_utils.initialize_obs_utils_with_config(config, verbose=False)
    shapes = {"all_shapes": {"robot0_eef_pos": (3,)}, "ac_dim": 1, "use_images": False}
    model = algo_factory(algorithm, config, shapes["all_shapes"], 1, torch.device("cpu"))
    checkpoint = tmp_path / "policy.pth"
    train_utils.save_model(model, config, FakeEnv().serialize(), shapes, str(checkpoint), verbose=False)
    policy, _ = helper.FileUtils.policy_from_checkpoint(ckpt_path=str(checkpoint), device=torch.device("cpu"))

    class PolicyEnv(FakeEnv):
        def reset(self):
            super().reset()
            return {"robot0_eef_pos": np.zeros(3, dtype=np.float32)}

        def step(self, action):
            assert action.shape == (1,) and np.isfinite(action).all()
            _, reward, done, info = super().step(action)
            return {"robot0_eef_pos": np.zeros(3, dtype=np.float32)}, reward, done, info

    stats, trajectory, _, frames = helper.collect_episode(policy, PolicyEnv(), 3, ["agentview"], 1)
    assert stats["Success_Rate"] == 1.0 and stats["Horizon"] == 2
    assert trajectory["actions"].shape == (2, 1) and len(frames) == 2


# ---- saved results, trials, the success interval, Weights & Biases and the insertion report ---------------------------

ALL_SUCCEED = [("success", 3)] * 6          # a second trial in which every rollout succeeds
TWO_TRIALS = dict(script=SCRIPT + ALL_SUCCEED, n_trials=2)


@pytest.fixture(autouse=True)
def clean_wandb_environment(monkeypatch):
    """Whether W&B is on must come from the arguments, not from the shell the tests run in."""
    for name in ("WANDB_MODE", "WANDB_ENTITY", "WANDB_API_KEY", "WANDB_PROJECT"):
        monkeypatch.delenv(name, raising=False)


class FakeSummary(dict):
    fail = False

    def update(self, *args, **kwargs):
        if self.fail:
            raise ConnectionError("summary rejected")
        super().update(*args, **kwargs)


class FakeRun:
    id, url = "fake-run", "https://example.invalid/run"

    def __init__(self, fail_log_at=None, fail_finish=False, fail_summary=False, fail_artifact=False):
        self.rows, self.definitions, self.summary = [], [], FakeSummary()
        self.summary.fail = fail_summary
        self.finished, self.artifact, self.fail_log_at, self.fail_finish = None, None, fail_log_at, fail_finish
        self.fail_artifact = fail_artifact

    def log(self, values, **kwargs):
        if self.fail_log_at is not None and len(self.rows) + 1 == self.fail_log_at:
            raise ConnectionError("lost the connection")
        self.rows.append(values)

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def log_artifact(self, artifact):
        if self.fail_artifact:
            raise ConnectionError("artifact rejected")
        self.artifact = artifact

    def finish(self, exit_code=None):
        if self.fail_finish:
            raise ConnectionError("cannot finish")
        self.finished = exit_code


class FakeArtifact:
    def __init__(self, name, type):
        self.name, self.type, self.files = name, type, []

    def add_file(self, local_path, name=None):
        self.files.append((str(local_path), name))


class FakeWandb:
    """The part of the wandb module that rollout_best uses, with the signatures of the real one (checked against
    wandb 0.30: Video(data_or_path, caption, fps, format), Table(columns, data), Artifact.add_file(local_path, name))."""

    def __init__(self, run=None, init_error=None, table_error=None):
        self.run, self.init_error, self.init_calls = run or FakeRun(), init_error, []
        self.table_error = table_error
        self.Artifact = FakeArtifact

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.init_error:
            raise self.init_error
        return self.run

    def Table(self, columns=None, data=None):
        if self.table_error:
            raise self.table_error
        return SimpleNamespace(columns=list(columns), data=[list(row) for row in data])

    def Video(self, data_or_path, caption=None, fps=None, format=None):
        if not Path(data_or_path).is_file():  # the real one reads the file when it is made
            raise FileNotFoundError(2, "No such file or directory", str(data_or_path))
        return SimpleNamespace(video=str(data_or_path), format=format, caption=caption)


def install_wandb(monkeypatch, **kwargs):
    fake = FakeWandb(**kwargs)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def logged(run, key):
    return [row[key] for row in run.rows if key in row]


def rollout_numbers(run):
    """The rollout number of each logged rollout (a rollout's video is sent as a second row with the same number)."""
    return [row["rollout"] for row in run.rows if "Rollout/Success" in row]


def test_the_success_rate_comes_with_a_wilson_interval():
    low, high = helper.wilson_interval(38, 50)
    assert (low, high) == pytest.approx((0.6255, 0.8570), abs=1e-3)
    assert helper.wilson_interval(0, 10)[0] == 0 and helper.wilson_interval(10, 10)[1] == 1
    assert 0 < helper.wilson_interval(0, 10)[1] < .35 and .65 < helper.wilson_interval(10, 10)[0] < 1
    assert helper.wilson_interval(0, 0) == (None, None)
    # a rate is always inside its own interval, and more rollouts narrow it
    for n in (1, 7, 50, 400):
        for k in range(n + 1):
            assert helper.wilson_interval(k, n)[0] <= k / n <= helper.wilson_interval(k, n)[1]
    assert helper.wilson_interval(100, 200)[1] - helper.wilson_interval(100, 200)[0] < \
        helper.wilson_interval(10, 20)[1] - helper.wilson_interval(10, 20)[0]


def test_counts_in_the_thousands_are_not_printed_with_an_exponent():
    assert helper.format_count(3.14159) == "3.14" and helper.format_count(0.000813) == "0.000813"
    assert helper.format_count(1234.4) == "1,234" and helper.format_count(12345) == "12,345" and helper.format_count(-2500) == "-2,500"


def test_results_are_saved_printed_and_do_not_need_wandb(tmp_path, monkeypatch, capsys):
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, no_stitch=True)
    assert fake.init_calls == [] and summary["wandb"] is None  # off unless asked for
    with (output / "rollouts.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == list(helper.ROLLOUT_FIELDS)
    assert [(r["rollout"], r["seed"], r["status"], r["success"]) for r in rows] == [
        ("0", "10000", "successful", "1"), ("1", "10001", "failed", "0"), ("2", "10002", "successful", "1"),
        ("3", "10003", "errors", "0"), ("4", "10004", "failed", "0"), ("5", "10005", "successful", "1")]
    assert "simulator failed" in rows[3]["error"] and rows[0]["error"] == "" and rows[0]["video"].startswith("successful_rollout_000")
    assert [int(r["horizon"]) for r in rows] == [4, 6, 3, 2, 6, 5]
    saved = json.loads((output / "summary.json").read_text())
    assert saved["success_rate"] == pytest.approx(.5) and saved["num_success"] == 3
    assert saved["success_rate_ci95"] == pytest.approx(list(helper.wilson_interval(3, 6)))
    # one trial writes straight into the output folder, as it always did
    assert not list(output.glob("trial_*")) and not (output / "trial_results.csv").exists()
    text = capsys.readouterr().out
    assert "Results over 6 rollouts" in text and "success rate 50.0% (3/6), 95% CI 18.8% to 81.2%; simulator errors 1" in text
    assert "saved in {}: summary.json, rollouts.csv, rollouts.hdf5".format(output) in text and "W&B" not in text.split("Results over")[1]


def test_trials_run_with_disjoint_seeds_and_each_gets_its_average(tmp_path, monkeypatch, capsys):
    output, summary = run_scripted(tmp_path, monkeypatch, no_stitch=True, n_rollouts=6, **TWO_TRIALS)
    assert sorted(path.name for path in output.glob("trial_??")) == ["trial_01", "trial_02"]
    for folder, first in (("trial_01", 10000), ("trial_02", 10006)):
        saved = json.loads((output / folder / "summary.json").read_text())
        assert saved["seed"] == first and [e["seed"] for e in saved["episodes"]] == list(range(first, first + 6))
        with h5py.File(output / folder / "rollouts.hdf5") as f:
            assert [int(f["data/demo_{}".format(i)].attrs["seed"]) for i in range(6)] == list(range(first, first + 6))
        assert (output / folder / "rollouts.csv").exists()
    first, second = summary["trials"]
    assert (first["trial"], first["seed"], first["folder"]) == (1, 10000, "trial_01")
    assert (second["trial"], second["seed"], second["folder"]) == (2, 10006, "trial_02")
    assert first["metrics"]["Success_Rate"] == pytest.approx(.5) and second["metrics"]["Success_Rate"] == 1.
    assert (first["metrics"]["num_success"], second["metrics"]["num_success"]) == (3, 6)
    stats = summary["across_trials"]["Success_Rate"]
    assert (stats["mean"], stats["n"]) == (pytest.approx(.75), 2) and stats["se"] == pytest.approx(.25)
    assert stats["std"] == pytest.approx(0.5 / 2 ** .5)
    assert summary["pooled"]["num_success"] == 9 and summary["pooled"]["n_rollouts"] == 12
    assert json.loads((output / "summary.json").read_text())["across_trials"]["Success_Rate"]["n"] == 2
    # the files that cover every trial, with the columns train_trials.py writes
    with (output / "trial_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == list(helper_trial_fields()) and [r["trial"] for r in rows] == ["1", "2"]
    assert [r["trial_seed"] for r in rows] == ["10000", "10006"] and [float(r["success_rate_mean"]) for r in rows] == [.5, 1.]
    assert [r["num_success"] for r in rows] == ["3", "6"] and rows[0]["model"] == "export"
    with (output / "rollout_results.csv").open() as stream:
        rollouts = list(csv.DictReader(stream))
    assert len(rollouts) == 12 and [r["trial"] for r in rollouts] == ["1"] * 6 + ["2"] * 6
    assert [r["rollout_seed"] for r in rollouts][5:7] == ["10005", "10006"]
    with (output / "trials.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert [r["Success_Rate"] for r in table] == ["0.5", "1.0"] and [r["num_errors"] for r in table] == ["1", "0"]
    text = capsys.readouterr().out
    assert "2 trials of 6 rollouts, seeds 10000 to 10011" in text and "Trial 2/2 | seeds 10006 to 10011" in text
    assert "Trial 1 over 6 rollouts" in text and "Trial 2 over 6 rollouts" in text
    assert "Trials: 2 x 6 rollouts" in text and "      1       10000         50.0%        3/6     4.33    4.33       1" in text
    assert "average of the trial success rates: 75.0% +/- 25.0% (standard error)  (n = 2 trials)" in text
    assert "all 12 rollouts together: 9/12 = 75.0%" in text


def helper_trial_fields():
    from robomimic.utils.trial_metrics import TRIAL_FIELDS
    return TRIAL_FIELDS


def test_one_trial_and_trials_of_one_rollout_are_handled(tmp_path, monkeypatch):
    output, summary = run_scripted(tmp_path, monkeypatch, no_stitch=True, n_rollouts=1, script=[("success", 2), ("fail", None)],
                                   n_trials=2)
    assert [t["metrics"]["Success_Rate"] for t in summary["trials"]] == [1., 0.]
    assert summary["across_trials"]["Success_Rate"]["se"] == pytest.approx(.5)
    assert [json.loads((output / f / "summary.json").read_text())["num_success"] for f in ("trial_01", "trial_02")] == [1, 0]


def test_a_trial_folder_that_cannot_be_made_stops_before_the_next_trial(tmp_path, monkeypatch):
    # the output folder must be new, and so must every trial folder
    with pytest.raises(FileExistsError):
        run_scripted(tmp_path, monkeypatch, no_stitch=True, n_rollouts=6, script=SCRIPT * 2, n_trials=0 + 2,
                     output_dir=str(tmp_path))


def test_the_wandb_run_follows_every_rollout_and_gets_the_results(tmp_path, monkeypatch, capsys):
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="cami-square", name="my-eval",
                                   group="exp1", wandb_entity="lab")
    run = fake.run
    (call,) = fake.init_calls
    assert (call["project"], call["entity"], call["group"], call["job_type"], call["name"], call["mode"]) == (
        "cami-square", "lab", "exp1", "rollouts", "my-eval", "online")
    assert call["dir"] == str(output) and call["config"]["n_rollouts"] == 6 and call["config"]["seed"] == 10000
    assert call["config"] == {
        "checkpoint": str(tmp_path / "model.pth"), "epoch": None, "training_success_rate": None, "n_trials": 1, "n_rollouts": 6,
        "horizon": 6, "seed": 10000, "camera_names": ["agentview"], "video_skip": 1, "fps": 60, "keep_failures": True,
        "output_dir": str(output), "insertion_report": False, "wandb_videos": "all"}
    assert ("rollout", {}) in run.definitions and ("Rollout/*", {"step_metric": "rollout"}) in run.definitions
    assert ("trial", {}) in run.definitions and ("Trial/*", {"step_metric": "trial"}) in run.definitions
    assert ("Running/*", {"step_metric": "trial"}) in run.definitions
    # one point per rollout, as it finishes, with the success so far
    assert rollout_numbers(run) == [0, 1, 2, 3, 4, 5] and logged(run, "Rollout/Trial") == [1] * 6
    assert logged(run, "Rollout/Success") == [1., 0., 1., 0., 0., 1.]
    assert logged(run, "Rollout/Simulator_Error") == [0., 0., 0., 1., 0., 0.]
    assert logged(run, "Rollout/Horizon") == [4, 6, 3, 2, 6, 5]
    assert logged(run, "Rollout/Return") == [4, 6, 3, 2, 6, 5]  # the scripted environment pays 1 per step
    assert len(logged(run, "Rollout/Seconds")) == 6 and all(0 <= seconds < 5 for seconds in logged(run, "Rollout/Seconds"))
    assert logged(run, "Rollout/Running/Success_Rate/mean") == pytest.approx([1., .5, 2 / 3, .5, .4, .5])
    assert "Rollout/Running/Success_Rate/se" not in run.rows[0]
    assert logged(run, "Rollout/Running/Success_Rate/se")[-1] == pytest.approx((.5 * .5 * 6 / 5 / 6) ** .5)
    # every rollout's video goes up with its rollout, captioned with what it shows
    clips = logged(run, "Rollout/Video")
    assert [c.video for c in clips] == [str(output / r["video"]) for r in summary["episodes"]] and len(clips) == 6
    assert clips[1].caption == "trial 1 rollout 1 | failed | seed 10001" and all(c.format == "mp4" for c in clips)
    # the trial: its average, and the running mean over the trials so far
    assert logged(run, "trial") == [1] and logged(run, "Trial/Success_Rate") == [.5] and logged(run, "Trial/num_success") == [3]
    assert logged(run, "Running/Success_Rate/mean") == [.5] and "Running/Success_Rate/se" not in logged_row(run, "trial")
    assert logged(run, "Trial/Horizon") == [pytest.approx(26 / 6)]
    # the finished evaluation: across trials (n = trials) and pooled over every rollout
    assert run.summary["Summary/n_trials"] == 1 and run.summary["Summary/Success_Rate/mean"] == .5
    assert run.summary["Summary/Success_Rate/n"] == 1 and "Summary/Success_Rate/se" not in run.summary
    assert run.summary["Pooled/n_rollouts"] == 6 and run.summary["Pooled/num_success"] == 3 and run.summary["Pooled/num_errors"] == 1
    assert run.summary["Pooled/Success_Rate/mean"] == pytest.approx(.5) and run.summary["Pooled/Success_Rate/n"] == 6
    assert run.summary["Pooled/Success_Rate/se"] == pytest.approx((.5 * .5 * 6 / 5 / 6) ** .5)
    assert (run.summary["Pooled/Success_Rate/ci95_low"], run.summary["Pooled/Success_Rate/ci95_high"]) == pytest.approx(
        helper.wilson_interval(3, 6))
    assert run.summary["Summary/Simulator_Error_Rate/mean"] == pytest.approx(1 / 6)
    assert run.summary["Summary/Return/mean"] == pytest.approx(26 / 6) and run.summary["Summary/Return/n"] == 1
    assert run.summary["Pooled/Return/mean"] == pytest.approx(26 / 6) and run.summary["Pooled/Return/n"] == 6
    assert run.summary["Pooled/Return/std"] == pytest.approx(statistics.stdev([4, 6, 3, 2, 6, 5]))
    assert run.summary["Pooled/Seconds/n"] == 6 and run.summary["Pooled/Horizon/mean"] == pytest.approx(26 / 6)
    (trials_table,) = logged(run, "trials")
    assert trials_table.columns[:3] == ["trial", "seed", "num_rollouts"] and trials_table.data[0][:3] == [1, 10000, 6]
    assert trials_table.data[0][trials_table.columns.index("Success_Rate")] == .5
    (table,) = logged(run, "rollouts")
    assert table.columns == ["trial"] + list(helper.ROLLOUT_FIELDS) and len(table.data) == 6 and {row[0] for row in table.data} == {1}
    assert [row[3] for row in table.data] == ["successful", "failed", "successful", "errors", "failed", "successful"]
    assert table.data[0][-1].startswith("successful_rollout_000") and isinstance(table.data[0][-1], str)  # the table names the clip
    # one video per outcome: the joined ones, whatever the individual clips did
    assert logged(run, "Video/SUCCESSFUL_ALL")[0].video == str(output / "SUCCESSFUL_ALL.mp4")
    assert logged(run, "Video/FAILED_ALL")[0].video == str(output / "FAILED_ALL.mp4")
    # the result files travel with it (by name relative to the output folder), the big files do not, and the run ends normally
    assert run.artifact.type == "rollouts" and run.artifact.name == "rollouts-fake-run"
    assert sorted(name for _, name in run.artifact.files) == ["FAILED_ALL.csv", "SUCCESSFUL_ALL.csv", "rollouts.csv", "summary.json"]
    assert all(Path(path) == output / name for path, name in run.artifact.files)
    assert run.finished == 0
    assert summary["wandb"] == {"project": "cami-square", "mode": "online", "run_id": "fake-run", "url": run.url}
    assert json.loads((output / "summary.json").read_text())["wandb"]["url"] == run.url
    text = capsys.readouterr().out
    assert "W&B: project cami-square (from --wandb-project), mode online" in text and "W&B: https://example.invalid/run" in text
    assert "return per rollout: mean 4.33, SD 1.63" in text and "steps per rollout: mean 4.33, SD 1.63" in text


def logged_row(run, key):
    return next(row for row in run.rows if key in row)


def test_trials_are_logged_with_their_running_mean_and_standard_error(tmp_path, monkeypatch, capsys):
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", n_rollouts=6, **TWO_TRIALS)
    run = fake.run
    assert rollout_numbers(run) == list(range(12)) and logged(run, "Rollout/Trial") == [1] * 6 + [2] * 6
    assert logged(run, "trial") == [1, 2] and logged(run, "Trial/Success_Rate") == [.5, 1.]
    assert logged(run, "Running/Success_Rate/mean") == pytest.approx([.5, .75])
    assert logged(run, "Running/Success_Rate/se") == pytest.approx([.25])  # one trial has no spread yet; two do
    assert logged(run, "Trial/num_errors") == [1, 0] and logged(run, "Trial/Simulator_Error_Rate") == pytest.approx([1 / 6, 0])
    # the pooled running rate keeps counting across the trials
    assert logged(run, "Rollout/Running/Success_Rate/mean")[-1] == pytest.approx(9 / 12)
    assert run.summary["Summary/n_trials"] == 2 and run.summary["Summary/Success_Rate/mean"] == pytest.approx(.75)
    assert run.summary["Summary/Success_Rate/n"] == 2 and run.summary["Summary/Success_Rate/se"] == pytest.approx(.25)
    assert run.summary["Summary/Success_Rate/std"] == pytest.approx(.5 / 2 ** .5)
    assert run.summary["Pooled/n_rollouts"] == 12 and run.summary["Pooled/num_success"] == 9
    (trials_table,) = logged(run, "trials")
    assert [row[:3] for row in trials_table.data] == [[1, 10000, 6], [2, 10006, 6]]
    (table,) = logged(run, "rollouts")
    assert len(table.data) == 12 and [row[0] for row in table.data] == [1] * 6 + [2] * 6
    assert [row[1] for row in table.data] == [0, 1, 2, 3, 4, 5] * 2  # the rollout number is the demo_N of its trial
    # each trial's joined videos are named by trial
    keys = {key for row in run.rows for key in row if key.startswith("Video/")}
    assert keys == {"Video/trial_01/SUCCESSFUL_ALL", "Video/trial_01/FAILED_ALL", "Video/trial_02/SUCCESSFUL_ALL"}
    assert sorted(name for _, name in run.artifact.files) == sorted([
        "FAILED_ALL.csv".replace("FAILED_ALL", "trial_01/FAILED_ALL"), "trial_01/SUCCESSFUL_ALL.csv", "trial_01/rollouts.csv",
        "trial_01/summary.json", "trial_02/SUCCESSFUL_ALL.csv", "trial_02/rollouts.csv", "trial_02/summary.json",
        "rollout_results.csv", "summary.json", "trial_results.csv", "trials.csv"])
    assert run.finished == 0 and summary["wandb"]["run_id"] == "fake-run"
    assert "W&B: https://example.invalid/run" in capsys.readouterr().out.split("Trials: 2 x 6 rollouts")[1]


def test_the_run_name_defaults_to_the_output_folder_and_the_project_to_the_checkpoints(tmp_path, monkeypatch, capsys):
    fake = install_wandb(monkeypatch)
    run_scripted(tmp_path, monkeypatch, project="from-the-checkpoint", wandb=True, no_stitch=True)
    (call,) = fake.init_calls
    assert call["project"] == "from-the-checkpoint" and call["name"] == "export" and call["group"] is None
    assert call["mode"] == "online"
    assert "W&B: project from-the-checkpoint (from the checkpoint's config), mode online" in capsys.readouterr().out


def test_wandb_needs_a_project_before_anything_is_run(tmp_path, monkeypatch):
    fake = install_wandb(monkeypatch)
    with pytest.raises(ValueError, match="Set --wandb-project"):
        run_scripted(tmp_path, monkeypatch, wandb=True)
    assert fake.init_calls == [] and not (tmp_path / "export").exists()


def test_disabled_mode_sends_nothing_even_with_a_project(tmp_path, monkeypatch):
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="disabled", wandb_project="p", no_stitch=True)
    assert fake.init_calls == [] and summary["wandb"] is None and (output / "rollouts.csv").exists()


def test_an_offline_run_says_how_to_sync_it_and_prints_no_link(tmp_path, monkeypatch, capsys):
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="offline", wandb_project="p", no_stitch=True)
    text = capsys.readouterr().out
    assert "W&B is offline: nothing is sent until you run: wandb sync {}".format(output / "wandb" / "latest-run") in text
    assert "  W&B: " not in text and fake.init_calls[0]["mode"] == "offline" and summary["wandb"]["mode"] == "offline"


def test_every_clip_goes_up_with_its_rollout_by_default(tmp_path, monkeypatch, capsys):
    fake = install_wandb(monkeypatch)
    run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", no_stitch=True)
    assert len(logged(fake.run, "Rollout/Video")) == 6
    assert not any(key.startswith("Video/") for row in fake.run.rows for key in row)  # nothing was joined
    assert "No joined videos to send to W&B (--no-stitch or the join failed)." in capsys.readouterr().out


@pytest.mark.parametrize("videos, clips, joined", [("all", 6, 2), ("joined", 0, 2), ("none", 0, 0)])
def test_the_videos_to_send_can_be_chosen(tmp_path, monkeypatch, videos, clips, joined):
    fake = install_wandb(monkeypatch)
    run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", wandb_videos=videos)
    assert len(logged(fake.run, "Rollout/Video")) == clips
    assert len({key for row in fake.run.rows for key in row if key.startswith("Video/")}) == joined
    assert len(logged(fake.run, "rollouts")) == 1  # the table is there either way


def test_clips_that_were_not_kept_are_not_sent(tmp_path, monkeypatch):
    fake = install_wandb(monkeypatch)
    run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", keep_failures=False, no_stitch=True)
    assert [c.caption for c in logged(fake.run, "Rollout/Video")] == [
        "trial 1 rollout {} | successful | seed {}".format(i, 10000 + i) for i in (0, 2, 5)]
    (table,) = logged(fake.run, "rollouts")
    assert [bool(row[-1]) for row in table.data] == [True, False, True, False, False, True]


def test_a_wandb_failure_never_costs_the_evaluation(tmp_path, monkeypatch, capsys):
    install_wandb(monkeypatch, init_error=ConnectionError("no network"))
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", no_stitch=True)
    assert summary["n_rollouts_completed"] == 6 and summary["wandb"] is None and (output / "rollouts.csv").exists()
    text = capsys.readouterr().out
    assert "W&B tracking stopped (ConnectionError: no network)" in text and text.count("W&B tracking stopped") == 1


def test_losing_the_connection_midway_stops_only_the_sending_and_the_run_stays_open_to_the_end(tmp_path, monkeypatch, capsys):
    fake = install_wandb(monkeypatch, run=FakeRun(fail_log_at=5))  # the 5th row: each rollout is a row of numbers and a row with its video
    open_at_start = []

    class Watches(FakePolicy):
        def start_episode(self):
            open_at_start.append(fake.run.finished is None)  # W&B must not show the run as finished while rollouts go on
            super().start_episode()

    output, summary = run_scripted(tmp_path, monkeypatch, policy=Watches(), wandb_mode="online", wandb_project="p", no_stitch=True)
    assert summary["n_rollouts_completed"] == 6 and len(rollout_numbers(fake.run)) == 2
    assert open_at_start == [True] * 6
    assert fake.run.finished == 1 and "rollouts" not in fake.run.rows[-1]  # failed: the results are incomplete
    text = capsys.readouterr().out
    assert text.count("W&B tracking stopped (ConnectionError: lost the connection)") == 1
    assert "stays open until the evaluation ends and is then marked failed" in text
    assert len(list((output / "rollouts.csv").read_text().splitlines())) == 7


def test_a_run_that_cannot_be_finished_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    install_wandb(monkeypatch, run=FakeRun(fail_finish=True))
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", no_stitch=True)
    assert summary["n_rollouts_completed"] == 6
    assert "Could not close the W&B run (ConnectionError: cannot finish)" in capsys.readouterr().out


def test_an_evaluation_that_breaks_marks_the_wandb_run_failed(tmp_path, monkeypatch):
    class Breaks(FakePolicy):
        def __call__(self, ob, goal=None):
            if self.episodes == 2:
                raise ValueError("policy broke")
            return super().__call__(ob, goal)

    fake = install_wandb(monkeypatch)
    with pytest.raises(ValueError, match="policy broke"):
        run_scripted(tmp_path, monkeypatch, policy=Breaks(), wandb_mode="online", wandb_project="p", no_stitch=True)
    assert fake.run.finished == 1 and rollout_numbers(fake.run) == [0]


@pytest.mark.parametrize("step", ["write_rollout_table", "stitch_videos", "print_results", "write_summary"])
def test_a_failure_after_the_rollouts_still_marks_the_wandb_run_failed(tmp_path, monkeypatch, step):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    fake = install_wandb(monkeypatch)
    if step == "write_summary":
        real = helper.write_summary
        calls = []
        monkeypatch.setattr(helper, "write_summary", lambda *a, **k: (calls.append(1), real(*a, **k) if len(calls) < 8 else broken())[1])
    elif step == "stitch_videos":
        monkeypatch.setattr(helper, "write_rollout_table", broken)  # stitching failures are tolerated; the next step is not
    else:
        monkeypatch.setattr(helper, step, broken)
    with pytest.raises(OSError, match="disk full"):
        run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p")
    assert fake.run.finished == 1


def test_a_failure_in_a_later_trial_marks_the_wandb_run_failed_after_the_first_trial_was_logged(tmp_path, monkeypatch):
    class BreaksLater(FakePolicy):
        def __call__(self, ob, goal=None):
            if self.episodes == 8:
                raise ValueError("second trial broke")
            return super().__call__(ob, goal)

    fake = install_wandb(monkeypatch)
    with pytest.raises(ValueError, match="second trial broke"):
        run_scripted(tmp_path, monkeypatch, policy=BreaksLater(), wandb_mode="online", wandb_project="p", no_stitch=True,
                     n_rollouts=6, **TWO_TRIALS)
    assert fake.run.finished == 1 and logged(fake.run, "trial") == [1]
    assert (tmp_path / "export" / "trial_01" / "summary.json").exists()


def synthetic_report():
    """A one-episode insertion report made by the real analysis from made-up features."""
    from robomimic.scripts import inspect_square_insertions as insertions
    T = 6
    height = np.array([1.1, 1.0, .9, .85, .8095, .8095])  # lowered over the post from above, then at rest
    features = dict(nut_pos=np.c_[np.full(T, .2), np.zeros(T), height], nut_rot=np.tile(np.eye(3), (T, 1, 1)),
                    eef=np.full((T, 3), np.nan), peg_pos=np.array([.2, 0., .9]), nut_table=np.ones(T), nut_peg=np.zeros(T),
                    nut_finger=np.array([0, 0, 2, 2, 2, 2.]), finger_table=np.zeros(T), hand_table=np.zeros(T),
                    arm_table=np.zeros(T), nut_peg_force=np.zeros(T), finger_table_force=np.zeros(T))
    geometry = dict(table_z=.8, peg_top=1., peg_half=.016, hole_half=.02275, outer_half=.04375, half_thickness=.01)
    metrics = insertions.analyze(features, geometry)
    metrics["flags"] = insertions.flags(metrics)
    return dict(dataset="x", episodes={"demo_0": metrics}, drift={}, skipped=[],
                thresholds=dict(min_drag_steps=5, min_post_contact_steps=40))


def test_the_insertion_report_is_saved_printed_and_sent(tmp_path, monkeypatch, capsys):
    from robomimic.scripts import inspect_square_insertions as insertions
    seen = []
    monkeypatch.setattr(insertions, "build_report", lambda path, **kw: seen.append(path) or synthetic_report())
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", no_stitch=True,
                                   insertion_report=True, env_name="NutAssemblySquare")
    assert seen == [str(output / "rollouts.hdf5")]
    assert (output / "insertions.csv").read_text().startswith("episode,steps,success")
    saved = json.loads((output / "insertions.json").read_text())
    assert saved["summary"]["successes"] == 1 and set(saved["episodes"]) == {"demo_0"}
    text = capsys.readouterr().out
    assert "Insertion report (replaying the saved rollouts)" in text and "raising a jam flag" in text
    assert "insertions.csv, insertions.json" in text
    run = fake.run
    # the numbers are part of the trial, so they get a point per trial, a running mean and a summary over the trials
    assert logged(run, "Trial/Insertion/successes") == [1.] and logged(run, "Trial/Insertion/jam_flag_share") == [0.]
    assert logged(run, "Trial/Insertion/seated_at_end") == [1.] and logged(run, "Running/Insertion/jam_flag_share/mean") == [0.]
    assert run.summary["Summary/Insertion/jam_flag_share/mean"] == 0 and run.summary["Summary/Insertion/episodes/n"] == 1
    (table,) = logged(run, "insertions")
    from robomimic.scripts import inspect_square_insertions as ins
    assert table.columns == ["trial"] + list(ins.TABLE_COLUMNS) and table.data[0][:2] == [1, "demo_0"]
    assert {name for _, name in run.artifact.files} >= {"insertions.csv", "insertions.json"}


def test_the_insertion_report_of_every_trial_is_averaged(tmp_path, monkeypatch):
    from robomimic.scripts import inspect_square_insertions as insertions
    reports = iter([synthetic_report(), synthetic_report()])
    jammed = synthetic_report()
    jammed["episodes"]["demo_0"]["flags"] = ["arrived_beside_post"]
    reports = iter([synthetic_report(), jammed])
    monkeypatch.setattr(insertions, "build_report", lambda path, **kw: next(reports))
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", no_stitch=True,
                                   insertion_report=True, n_rollouts=6, **TWO_TRIALS)
    assert [t["metrics"]["Insertion/jam_flag_share"] for t in summary["trials"]] == [0., 1.]
    assert summary["across_trials"]["Insertion/jam_flag_share"]["mean"] == .5
    assert logged(fake.run, "Trial/Insertion/jam_flag_share") == [0., 1.]
    assert logged(fake.run, "Running/Insertion/jam_flag_share/mean") == [0., .5]
    (table,) = logged(fake.run, "insertions")
    assert [row[0] for row in table.data] == [1, 2]
    for folder in ("trial_01", "trial_02"):
        assert (output / folder / "insertions.csv").exists()


def test_a_failing_insertion_report_never_costs_the_evaluation(tmp_path, monkeypatch, capsys):
    from robomimic.scripts import inspect_square_insertions as insertions

    def broken(path, **kwargs):
        raise RuntimeError("no EGL")

    monkeypatch.setattr(insertions, "build_report", broken)
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", no_stitch=True,
                                   insertion_report=True)
    assert summary["n_rollouts_completed"] == 6 and not (output / "insertions.csv").exists()
    text = capsys.readouterr().out
    assert "Could not make the insertion report (RuntimeError: no EGL)" in text
    assert "saved in {}: summary.json, rollouts.csv, rollouts.hdf5\n".format(output) in text  # nothing is claimed that was not written
    assert not any(k.startswith("Summary/Insertion/") for k in fake.run.summary) and not logged(fake.run, "insertions")
    assert fake.run.finished == 0


def test_the_insertion_report_is_for_the_square_task_and_is_refused_before_anything_is_run(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="--insertion-report is for the Square task, but this checkpoint's environment is Lift"):
        run_scripted(tmp_path, monkeypatch, insertion_report=True, env_name="Lift")
    assert not (tmp_path / "export").exists()


def test_the_command_line_has_the_wandb_trial_and_report_options(monkeypatch, capsys):
    captured = []
    monkeypatch.setattr(helper, "run", lambda args: captured.append(args))
    argv = ["rollout_best", "--checkpoint", "m.pth", "--keep-failures", "--output-dir", "out", "--insertion-report",
            "--wandb-project", "p", "--wandb-entity", "e", "--wandb-mode", "offline", "--wandb-videos", "joined",
            "--name", "n", "--group", "g", "--n-trials", "4", "--n-rollouts", "25"]
    monkeypatch.setattr(sys, "argv", argv)
    helper.main()
    (args,) = captured
    assert (args.keep_failures, args.output_dir, args.insertion_report) == (True, "out", True)
    assert (args.wandb_project, args.wandb_entity, args.wandb_mode, args.wandb_videos, args.name, args.group) == (
        "p", "e", "offline", "joined", "n", "g")
    assert (args.n_trials, args.n_rollouts) == (4, 25)
    monkeypatch.setattr(sys, "argv", ["rollout_best", "--checkpoint", "m.pth"])
    helper.main()
    defaults = captured[-1]
    assert (defaults.wandb, defaults.wandb_project, defaults.wandb_mode, defaults.wandb_videos, defaults.insertion_report) == (
        False, None, None, "all", False)
    assert defaults.n_trials == 1 and helper.resolve_wandb_mode(defaults) is None
    for bad in (["--wandb-videos", "bad"], ["--n-trials", "0"], ["--n-trials", "2", "--seed", str(2 ** 32 - 10)]):
        monkeypatch.setattr(sys, "argv", ["rollout_best", "--checkpoint", "m.pth"] + bad)
        with pytest.raises(SystemExit):
            helper.main()
    assert "must not exceed 2**32" in capsys.readouterr().err


@pytest.mark.parametrize("failure", [dict(run=FakeRun(fail_summary=True)), dict(run=FakeRun(fail_artifact=True)),
                                     dict(table_error=ValueError("table rejected"))])
def test_a_wandb_failure_while_sending_the_results_never_costs_the_evaluation(tmp_path, monkeypatch, capsys, failure):
    fake = install_wandb(monkeypatch, **failure)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p")
    assert summary["n_rollouts_completed"] == 6 and fake.run.finished == 1
    text = capsys.readouterr().out
    assert text.count("W&B tracking stopped") == 1
    assert "Results over 6 rollouts" in text and "Videos, all trajectories, and summary saved in" in text  # printing went on
    for name in ("summary.json", "rollouts.csv", "rollouts.hdf5", "SUCCESSFUL_ALL.mp4"):
        assert (output / name).exists()
    assert len(rollout_numbers(fake.run)) == 6 and logged(fake.run, "trial") == [1]  # what had gone up stays


class GraspEnv(ScriptedEnv):
    """Also reports a second success measure, true when the rollout succeeds."""

    def is_success(self):
        result = super().is_success()
        return dict(result, grasp=result["task"] or self.episode == 1)


def test_another_success_measure_is_saved_logged_and_averaged_per_trial(tmp_path, monkeypatch):
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, env_class=GraspEnv, wandb_mode="online", wandb_project="p", no_stitch=True)
    run = fake.run
    assert logged(run, "Rollout/grasp_Success_Rate") == [1., 1., 1., 0., 0., 1.]
    assert logged(run, "Trial/grasp_Success_Rate") == [pytest.approx(4 / 6)]
    assert run.summary["Summary/grasp_Success_Rate/mean"] == pytest.approx(4 / 6)
    assert run.summary["Pooled/grasp_Success_Rate/n"] == 6
    with (output / "rollouts.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["grasp_Success_Rate"] for r in rows] == ["1.0", "1.0", "1.0", "0.0", "0.0", "1.0"]
    assert summary["metrics"]["grasp_Success_Rate"] == pytest.approx(4 / 6)


def test_every_clip_is_sent_from_its_own_trial_folder(tmp_path, monkeypatch, capsys):
    """With several trials the clips are in trial_NN folders: looking in the output folder made the first upload fail,
    which closed the run while the evaluation went on."""
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", no_stitch=True,
                                   n_rollouts=6, **TWO_TRIALS)
    clips = logged(fake.run, "Rollout/Video")
    expected = [str(output / folder / record["video"]) for folder, trial in (("trial_01", 0), ("trial_02", 1))
                for record in json.loads((output / folder / "summary.json").read_text())["episodes"]]
    assert [c.video for c in clips] == expected and len(clips) == 12 and all(Path(c.video).is_file() for c in clips)
    assert clips[7].caption == "trial 2 rollout 1 | successful | seed 10007"
    assert fake.run.finished == 0 and "W&B tracking stopped" not in capsys.readouterr().out
    assert logged(fake.run, "trial") == [1, 2]


def test_a_video_that_cannot_be_sent_is_skipped_and_nothing_else_is_lost(tmp_path, monkeypatch, capsys):
    fake = install_wandb(monkeypatch)
    real_video = fake.Video

    def refuses_one(data_or_path, caption=None, fps=None, format=None):
        if "rollout_002" in str(data_or_path):
            raise OSError("cannot read this clip")
        return real_video(data_or_path, caption=caption, fps=fps, format=format)

    fake.Video = refuses_one
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", no_stitch=True)
    run = fake.run
    assert rollout_numbers(run)[:6] == [0, 1, 2, 3, 4, 5] and len(logged(run, "Rollout/Success")) == 6  # every rollout's numbers
    assert len(logged(run, "Rollout/Video")) == 5  # every clip but the one that failed
    assert logged(run, "trial") == [1] and run.finished == 0  # the rest was sent, and the run did not end failed
    text = capsys.readouterr().out
    assert "W&B could not take the video of trial 1 rollout 2 (OSError: cannot read this clip); it is skipped." in text
    assert "1 videos could not be sent to W&B; they are in the output directory." in text and "tracking stopped" not in text
    assert (output / summary["episodes"][2]["video"]).is_file()
