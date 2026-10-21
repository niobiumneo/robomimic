"""Check checkpoint ranking and preservation of successful / failed paths."""
import argparse
import csv
import json
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


def run_scripted(tmp_path, monkeypatch, project=None, policy=None, env_name=None, **changes):
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
    env = ScriptedEnv(SCRIPT)
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


# ---- saved results, the success interval, Weights & Biases and the insertion report -----------------------------------

class FakeSummary(dict):
    pass


class FakeRun:
    id, url = "fake-run", "https://example.invalid/run"

    def __init__(self, fail_log_at=None, fail_finish=False):
        self.rows, self.definitions, self.summary = [], [], FakeSummary()
        self.finished, self.artifact, self.fail_log_at, self.fail_finish = None, None, fail_log_at, fail_finish

    def log(self, values, **kwargs):
        if self.fail_log_at is not None and len(self.rows) + 1 == self.fail_log_at:
            raise ConnectionError("lost the connection")
        self.rows.append(values)

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def log_artifact(self, artifact):
        self.artifact = artifact

    def finish(self, exit_code=None):
        if self.fail_finish:
            raise ConnectionError("cannot finish")
        self.finished = exit_code


class FakeWandb:
    """The part of the wandb module that rollout_best uses; every object it makes just records its arguments."""

    def __init__(self, run=None, init_error=None):
        self.run, self.init_error, self.init_calls = run or FakeRun(), init_error, []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.init_error:
            raise self.init_error
        return self.run

    def Table(self, columns, data):
        return SimpleNamespace(columns=list(columns), data=[list(row) for row in data])

    def Video(self, path, format=None):
        return SimpleNamespace(video=str(path), format=format)

    def Artifact(self, name, type):
        return SimpleNamespace(name=name, type=type, files=[], add_file=lambda path: None)


def install_wandb(monkeypatch, **kwargs):
    fake = FakeWandb(**kwargs)
    artifact = fake.Artifact

    def tracked(name, type):
        made = artifact(name, type)
        made.add_file = made.files.append
        return made
    fake.Artifact = tracked
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def logged(run, key):
    return [row[key] for row in run.rows if key in row]


def test_the_success_rate_comes_with_a_wilson_interval():
    low, high = helper.wilson_interval(38, 50)
    assert (low, high) == pytest.approx((0.6255, 0.8570), abs=1e-3)
    assert helper.wilson_interval(0, 10)[0] == pytest.approx(0) and helper.wilson_interval(10, 10)[1] == pytest.approx(1)
    assert 0 < helper.wilson_interval(0, 10)[1] < .35 and .65 < helper.wilson_interval(10, 10)[0] < 1
    assert helper.wilson_interval(0, 0) == (None, None)
    # a rate is always inside its own interval, and more rollouts narrow it
    for n in (1, 7, 50, 400):
        for k in range(n + 1):
            assert helper.wilson_interval(k, n)[0] <= k / n <= helper.wilson_interval(k, n)[1]
    assert helper.wilson_interval(100, 200)[1] - helper.wilson_interval(100, 200)[0] < \
        helper.wilson_interval(10, 20)[1] - helper.wilson_interval(10, 20)[0]


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
    text = capsys.readouterr().out
    assert "Results over 6 rollouts" in text and "success rate 50.0% (3/6), 95% CI 18.8% to 81.2%; simulator errors 1" in text
    assert "saved in {}: summary.json, rollouts.csv, rollouts.hdf5".format(output) in text and "W&B" not in text.split("Results over")[1]


def test_the_wandb_run_follows_every_rollout_and_gets_the_results(tmp_path, monkeypatch, capsys):
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="cami-square", name="my-eval",
                                   group="exp1", wandb_entity="lab")
    run = fake.run
    (call,) = fake.init_calls
    assert (call["project"], call["entity"], call["group"], call["job_type"], call["name"], call["mode"]) == (
        "cami-square", "lab", "exp1", "rollouts", "my-eval", "online")
    assert call["dir"] == str(output) and call["config"]["n_rollouts"] == 6 and call["config"]["seed"] == 10000
    assert ("rollout", {}) in run.definitions and ("Rollout/*", {"step_metric": "rollout"}) in run.definitions
    # one point per rollout, as it finishes, with the success so far
    assert logged(run, "rollout") == [0, 1, 2, 3, 4, 5]
    assert logged(run, "Rollout/Success") == [1., 0., 1., 0., 0., 1.]
    assert logged(run, "Rollout/Simulator_Error") == [0., 0., 0., 1., 0., 0.]
    assert logged(run, "Rollout/Horizon") == [4, 6, 3, 2, 6, 5]
    assert logged(run, "Running/Success_Rate/mean") == pytest.approx([1., .5, 2 / 3, .5, .4, .5])
    assert "Running/Success_Rate/se" not in run.rows[0] and logged(run, "Running/Success_Rate/se")[-1] == pytest.approx(
        (.5 * .5 * 6 / 5 / 6) ** .5)
    # the finished evaluation
    assert run.summary["Summary/n_rollouts"] == 6 and run.summary["Summary/num_success"] == 3
    assert run.summary["Summary/num_errors"] == 1 and run.summary["Summary/Simulator_Error_Rate"] == pytest.approx(1 / 6)
    assert run.summary["Summary/Success_Rate/mean"] == pytest.approx(.5) and run.summary["Summary/Success_Rate/n"] == 6
    assert (run.summary["Summary/Success_Rate/ci95_low"], run.summary["Summary/Success_Rate/ci95_high"]) == pytest.approx(
        helper.wilson_interval(3, 6))
    (table,) = logged(run, "rollouts")
    assert table.columns == [f for f in helper.ROLLOUT_FIELDS if f != "video"] and len(table.data) == 6
    assert [row[2] for row in table.data] == ["successful", "failed", "successful", "errors", "failed", "successful"]
    # the joined videos are shown, the individual clips are not sent
    assert logged(run, "Video/SUCCESSFUL_ALL")[0].video == str(output / "SUCCESSFUL_ALL.mp4")
    assert logged(run, "Video/FAILED_ALL")[0].video == str(output / "FAILED_ALL.mp4")
    assert logged(run, "Video/SUCCESSFUL_ALL")[0].format == "mp4"
    # the result files travel with it, and the run is finished normally
    assert run.artifact.type == "rollouts" and run.artifact.name == "rollouts-fake-run"
    assert sorted(Path(f).name for f in run.artifact.files) == [
        "FAILED_ALL.csv", "SUCCESSFUL_ALL.csv", "rollouts.csv", "summary.json"]
    assert run.finished == 0
    assert summary["wandb"] == {"project": "cami-square", "mode": "online", "run_id": "fake-run", "url": run.url}
    assert json.loads((output / "summary.json").read_text())["wandb"]["url"] == run.url
    assert "W&B: https://example.invalid/run" in capsys.readouterr().out


def test_the_run_name_defaults_to_the_output_folder_and_the_project_to_the_checkpoints(tmp_path, monkeypatch):
    fake = install_wandb(monkeypatch)
    run_scripted(tmp_path, monkeypatch, project="from-the-checkpoint", wandb=True, no_stitch=True)
    (call,) = fake.init_calls
    assert call["project"] == "from-the-checkpoint" and call["name"] == "export" and call["group"] is None
    assert call["mode"] == "online"
    assert not any(isinstance(r, str) for r in fake.run.rows)


def test_wandb_needs_a_project_before_anything_is_run(tmp_path, monkeypatch):
    fake = install_wandb(monkeypatch)
    with pytest.raises(ValueError, match="Set --wandb-project"):
        run_scripted(tmp_path, monkeypatch, wandb=True)
    assert fake.init_calls == [] and not (tmp_path / "export").exists()


def test_disabled_mode_sends_nothing_even_with_a_project(tmp_path, monkeypatch):
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="disabled", wandb_project="p", no_stitch=True)
    assert fake.init_calls == [] and summary["wandb"] is None and (output / "rollouts.csv").exists()


@pytest.mark.parametrize("videos, expected", [("all", 6), ("none", 0), ("joined", 0)])
def test_the_videos_to_send_can_be_chosen(tmp_path, monkeypatch, capsys, videos, expected):
    fake = install_wandb(monkeypatch)
    run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", wandb_videos=videos, no_stitch=True)
    (table,) = logged(fake.run, "rollouts")
    clips = [row[table.columns.index("video")] for row in table.data] if "video" in table.columns else []
    assert len([c for c in clips if c is not None]) == expected and not any(k.startswith("Video/") for r in fake.run.rows for k in r)
    if videos == "all":
        assert all(c.video.endswith(".mp4") and c.format == "mp4" for c in clips)
    if videos == "joined":
        assert "No joined videos to send to W&B (--no-stitch); add --wandb-videos all" in capsys.readouterr().out


def test_clips_that_were_not_kept_are_empty_cells_when_every_clip_is_sent(tmp_path, monkeypatch):
    fake = install_wandb(monkeypatch)
    run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", wandb_videos="all", keep_failures=False,
                 no_stitch=True)
    (table,) = logged(fake.run, "rollouts")
    clips = [row[table.columns.index("video")] for row in table.data]
    assert [c is not None for c in clips] == [True, False, True, False, False, True]


def test_a_wandb_failure_never_costs_the_evaluation(tmp_path, monkeypatch, capsys):
    install_wandb(monkeypatch, init_error=ConnectionError("no network"))
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", no_stitch=True)
    assert summary["n_rollouts_completed"] == 6 and summary["wandb"] is None and (output / "rollouts.csv").exists()
    text = capsys.readouterr().out
    assert "W&B tracking stopped (ConnectionError: no network)" in text and text.count("W&B tracking stopped") == 1


def test_losing_the_connection_midway_stops_only_the_tracking(tmp_path, monkeypatch, capsys):
    fake = install_wandb(monkeypatch, run=FakeRun(fail_log_at=3))
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", no_stitch=True)
    assert summary["n_rollouts_completed"] == 6 and len(logged(fake.run, "rollout")) == 2
    assert fake.run.finished == 0 and "rollouts" not in fake.run.rows[-1]
    assert capsys.readouterr().out.count("W&B tracking stopped (ConnectionError: lost the connection)") == 1
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
    assert fake.run.finished == 1 and logged(fake.run, "rollout") == [0]


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
    assert run.summary["Insertion/successes"] == 1 and run.summary["Insertion/jam_flag_share"] == 0
    assert run.summary["Insertion/episodes"] == 1 and "Insertion/median_tilt_at_success" in run.summary
    assert run.summary["Insertion/seated_at_end"] == 1 and run.summary["Insertion/seated_checked"] == 1
    (table,) = logged(run, "insertions")
    assert table.columns == list(insertions.TABLE_COLUMNS) and table.data[0][0] == "demo_0"
    assert {Path(f).name for f in run.artifact.files} >= {"insertions.csv", "insertions.json"}


def test_a_failing_insertion_report_never_costs_the_evaluation(tmp_path, monkeypatch, capsys):
    from robomimic.scripts import inspect_square_insertions as insertions

    def broken(path, **kwargs):
        raise RuntimeError("no EGL")

    monkeypatch.setattr(insertions, "build_report", broken)
    fake = install_wandb(monkeypatch)
    output, summary = run_scripted(tmp_path, monkeypatch, wandb_mode="online", wandb_project="p", no_stitch=True,
                                   insertion_report=True)
    assert summary["n_rollouts_completed"] == 6 and not (output / "insertions.csv").exists()
    assert "Could not make the insertion report (RuntimeError: no EGL)" in capsys.readouterr().out
    assert not any(k.startswith("Insertion/") for k in fake.run.summary) and not logged(fake.run, "insertions")
    assert fake.run.finished == 0


def test_the_insertion_report_is_for_the_square_task_and_is_refused_before_anything_is_run(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="--insertion-report is for the Square task, but this checkpoint's environment is Lift"):
        run_scripted(tmp_path, monkeypatch, insertion_report=True, env_name="Lift")
    assert not (tmp_path / "export").exists()


def test_the_command_line_has_the_wandb_and_report_options(monkeypatch):
    captured = []
    monkeypatch.setattr(helper, "run", lambda args: captured.append(args))
    argv = ["rollout_best", "--checkpoint", "m.pth", "--keep-failures", "--output-dir", "out", "--insertion-report",
            "--wandb-project", "p", "--wandb-entity", "e", "--wandb-mode", "offline", "--wandb-videos", "all",
            "--name", "n", "--group", "g"]
    monkeypatch.setattr(sys, "argv", argv)
    helper.main()
    (args,) = captured
    assert (args.keep_failures, args.output_dir, args.insertion_report) == (True, "out", True)
    assert (args.wandb_project, args.wandb_entity, args.wandb_mode, args.wandb_videos, args.name, args.group) == (
        "p", "e", "offline", "all", "n", "g")
    monkeypatch.setattr(sys, "argv", ["rollout_best", "--checkpoint", "m.pth"])
    helper.main()
    defaults = captured[-1]
    assert (defaults.wandb, defaults.wandb_project, defaults.wandb_mode, defaults.wandb_videos, defaults.insertion_report) == (
        False, None, None, "joined", False)
    assert helper.resolve_wandb_mode(defaults) is None
    monkeypatch.setattr(sys, "argv", ["rollout_best", "--checkpoint", "m.pth", "--wandb-videos", "bad"])
    with pytest.raises(SystemExit):
        helper.main()
