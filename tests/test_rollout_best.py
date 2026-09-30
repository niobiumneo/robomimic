"""Check checkpoint ranking and preservation of successful / failed paths."""
import argparse
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
                              video_skip=3, fps=20, keep_failures=keep_failures)
    helper.run(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["num_errors"] == 1
    assert [r["Horizon"] for r in summary["episodes"]] == [2, 3, 1]
    assert [r["seed"] for r in summary["episodes"]] == [10000, 10001, 10002]
    assert policy.episodes == 3 and env.closed
    assert len(list(output.glob("*.mp4"))) == expected_videos
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
