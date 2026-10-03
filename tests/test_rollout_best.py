"""Check checkpoint ranking and preservation of successful / failed paths."""
import argparse
import csv
import json
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


def run_scripted(tmp_path, monkeypatch, **changes):
    checkpoint = tmp_path / "model.pth"
    torch.save({"env_metadata": {"env_kwargs": {"control_freq": 20}}}, checkpoint)
    config = SimpleNamespace(experiment=SimpleNamespace(rollout=SimpleNamespace(horizon=6)))
    env = ScriptedEnv(SCRIPT)
    monkeypatch.setattr(helper.FileUtils, "config_from_checkpoint", lambda **kw: (config, kw["ckpt_dict"]))
    monkeypatch.setattr(helper.FileUtils, "policy_from_checkpoint", lambda **kw: (FakePolicy(), kw["ckpt_dict"]))
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
