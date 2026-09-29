"""Exercise stored HDF5 supervision through loaders and actual optimizer steps.

Fixtures contain synthetic measurements for interface tests only. They are not
robot demonstration datasets and are never used to claim policy performance.
"""
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from robomimic.algo import algo_factory
from robomimic.config import config_factory
from robomimic.utils import obs_utils, train_utils, file_utils
from robomimic.utils import cami_dataset_utils as data_utils


TEMPLATES = Path(__file__).resolve().parents[1] / "robomimic/exps/templates"


@pytest.fixture
def dataset_path(tmp_path):
    path = tmp_path / "force.hdf5"
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as hdf5:
        data = hdf5.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "ToolHang", "type": 1, "env_kwargs": {}})
        data.attrs["total"] = 42
        for index in range(3):
            demo = data.create_group(f"demo_{index}")
            demo.attrs["num_samples"] = 14
            demo["actions"] = rng.uniform(-0.5, 0.5, (14, 7)).astype(np.float32)
            for key, width in (("robot0_eef_pos", 3), ("robot0_eef_quat", 4),
                               ("robot0_gripper_qpos", 2), ("object", 10)):
                demo[f"obs/{key}"] = rng.normal(size=(14, width)).astype(np.float32)
            for key in ("agentview_image", "sideview_image", "robot0_eye_in_hand_image"):
                demo[f"obs/{key}"] = rng.integers(0, 256, (14, 84, 84, 3), dtype=np.uint8)
            force = np.zeros((14, 6), np.float32)
            force[:, 0] = (np.arange(14) + 1) * (index + 1)
            # A large validation-only signal makes accidental scale leakage visible.
            if index == 2:
                force[:, 0] *= 1000
            force[:, 3:] = 5000
            demo["obs/force"] = force
            demo["contact_label"] = np.full((14, 1), index % 2, np.float32)
        hdf5["mask/train"] = np.array([b"demo_0", b"demo_1"])
        hdf5["mask/valid"] = np.array([b"demo_2"])
    return path


def make_config(dataset_path, template="bc_cami.json", images=False):
    raw = json.loads((TEMPLATES / template).read_text())
    config = config_factory(raw["algo_name"])
    with config.values_unlocked():
        config.update(raw)
        config.train.data = [{"path": str(dataset_path)}]
        config.train.batch_size = 2
        config.train.cuda = False
        config.algo.actor_layer_dims = [16]
        config.algo.rnn.hidden_dim = 12
        config.algo.rnn.num_layers = 1
        config.experiment.rollout.enabled = False
        config.experiment.logging.log_tb = False
        config.experiment.logging.log_wandb = False
        if not images:
            config.observation.modalities.obs.rgb = []
        else:
            config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 76
            config.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 76
        if config.algo_name == "bc_cami":
            config.algo.cami.policy_latent_dim = 16
            config.algo.cami.snippet_hidden_dim = 8
            config.algo.cami.contrastive_dim = 8
        else:
            config.algo.cami.lcp.gap_hidden_dims = [12]
            config.algo.cami.lcp.impulse_hidden_dims = [8]
    return config


def test_training_only_scale_and_repo_path(dataset_path, tmp_path, monkeypatch):
    config = make_config(dataset_path)
    monkeypatch.setattr(data_utils, "REPO_ROOT", tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with config.values_unlocked():
        config.train.data = [{"path": "force.hdf5", "demo_limit": 1}]
    config.lock()
    data_utils.prepare_cami_datasets(config)
    assert config.train.data[0]["path"] == str(dataset_path)
    assert config.algo.cami.continuous_contact.force_scale == pytest.approx(np.arange(1, 15).std() + 1e-6)
    assert "force" not in config.all_obs_keys


@pytest.mark.parametrize("corruption, message", [
    ("missing_force", "missing /data/demo_0/obs/force"),
    ("nan_force", "NaN or infinity"),
    ("short_force", "14 timesteps"),
    ("incomplete", "marked incomplete"),
    ("raw_only", "missing /data/demo_0/obs/robot0_eef_pos"),
])
def test_bad_datasets_fail_before_loading(dataset_path, corruption, message):
    with h5py.File(dataset_path, "a") as hdf5:
        if corruption == "nan_force":
            hdf5["data/demo_0/obs/force"][1, 0] = np.nan
        elif corruption == "incomplete":
            hdf5.attrs["complete"] = False
        elif corruption == "raw_only":
            del hdf5["data/demo_0/obs"]
        else:
            del hdf5["data/demo_0/obs/force"]
            if corruption == "short_force":
                hdf5["data/demo_0/obs/force"] = np.zeros((13, 6))
    with pytest.raises(ValueError, match=message):
        data_utils.prepare_cami_datasets(make_config(dataset_path))


def test_camera_crop_and_split_validation(dataset_path):
    config = make_config(dataset_path, images=True)
    with config.values_unlocked():
        config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 216
    with pytest.raises(ValueError, match="crop"):
        data_utils.prepare_cami_datasets(config)
    with config.values_unlocked():
        config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 76
        config.experiment.validate = True
        config.train.hdf5_validation_filter_key = "valid"
        config.train.data[0]["filter_key"] = "train"
    with pytest.raises(ValueError, match="overlap"):
        data_utils.prepare_cami_datasets(config)


@pytest.mark.parametrize("cache_mode", [None, "low_dim", "all"])
def test_force_cache_alignment_and_padding(dataset_path, cache_mode):
    config = make_config(dataset_path)
    with config.values_unlocked():
        config.train.hdf5_cache_mode = cache_mode
    obs_utils.initialize_obs_utils_with_config(config)
    dataset, _ = train_utils.load_data_for_training(config, config.all_obs_keys)
    first = dataset[0]
    np.testing.assert_array_equal(first["obs/force"][:, 0], np.arange(1, 12))
    final = dataset[13]
    assert final["pad_mask"].sum() == 1
    assert "force" not in first["obs"]
    dataset.close_and_delete_hdf5_handle()


@pytest.mark.parametrize("template, images", [
    ("bc_cami.json", False),
    ("bc_cami_square.json", True),
    ("bc_cami_lcp_v1.json", True),
    ("bc_cami_lcp_v2.json", False),
    ("bc_cami_lcp_v2_marginal.json", False),
])
def test_hdf5_to_optimizer_step(dataset_path, template, images):
    torch.set_num_threads(2)
    config = make_config(dataset_path, template, images=images)
    obs_utils.initialize_obs_utils_with_config(config)
    dataset, _ = train_utils.load_data_for_training(config, config.all_obs_keys)
    shapes = file_utils.get_shape_metadata_from_dataset(
        config.train.data[0], config.train.action_keys, config.all_obs_keys)
    model = algo_factory(config.algo_name, config, shapes["all_shapes"], shapes["ac_dim"], torch.device("cpu"))
    previous = next(model.nets["policy"].parameters()).detach().clone()
    logs = train_utils.run_epoch(model, DataLoader(dataset, batch_size=2), epoch=1, num_steps=1)
    assert np.isfinite(logs["Loss"])
    assert not torch.equal(previous, next(model.nets["policy"].parameters()))
    # The feature-returning training path and rollout must predict identical actions.
    model.set_eval()
    batch = model.postprocess_batch_for_training(
        model.process_batch_for_training(next(iter(DataLoader(dataset, batch_size=2)))),
        obs_normalization_stats=None,
    )
    with torch.no_grad():
        actions = model.nets["policy"](obs_dict=batch["obs"])
        feature_actions = model.nets["policy"].forward_with_features(obs=batch["obs"])[0]
    torch.testing.assert_close(actions, feature_actions)
    dataset.close_and_delete_hdf5_handle()


@pytest.mark.parametrize("algorithm", ["bc", "bc_cami"])
def test_rollouts_without_force_keep_success_metrics(dataset_path, capsys, algorithm):
    """Exercise real policies with a deterministic environment lacking force.

    One episode succeeds at step two and the other hits the three-step limit.
    This checks rollout accounting, not simulator performance.
    """
    from unittest.mock import Mock
    from robomimic.algo import RolloutPolicy
    from robomimic.envs.env_base import EnvBase

    config = make_config(dataset_path)
    if algorithm == "bc":
        raw = config.to_dict()
        raw["algo_name"] = "bc"
        raw["algo"].pop("cami")
        raw["algo"]["optim_params"] = {"policy": raw["algo"]["optim_params"]["policy"]}
        raw["train"]["dataset_keys"] = ["actions"]
        config = config_factory("bc")
        with config.values_unlocked():
            config.update(raw)
    obs_utils.initialize_obs_utils_with_config(config, verbose=False)
    data_utils.prepare_cami_datasets(config)
    shapes = file_utils.get_shape_metadata_from_dataset(
        config.train.data[0], config.train.action_keys, config.all_obs_keys)
    model = algo_factory(algorithm, config, shapes["all_shapes"], shapes["ac_dim"], torch.device("cpu"))

    with h5py.File(dataset_path, "r") as dataset:
        observation = {key: dataset[f"data/demo_0/obs/{key}"][0] for key in config.all_obs_keys}
    assert "force" not in observation
    state = {"episode": -1, "step": 0}
    env = Mock(spec=EnvBase)
    env.name = "NoForceFixture"
    env.rollout_exceptions = ()

    def reset():
        state["episode"] += 1
        state["step"] = 0
        return {key: value.copy() for key, value in observation.items()}

    def step(action):
        assert action.shape == (7,)
        assert np.isfinite(action).all()
        state["step"] += 1
        return {key: value.copy() for key, value in observation.items()}, 1.0, False, {}

    env.reset.side_effect = reset
    env.step.side_effect = step
    env.is_success.side_effect = lambda: {"task": state["episode"] == 0 and state["step"] == 2}
    logs, _ = train_utils.rollout_with_stats(
        RolloutPolicy(model), {"square": env}, horizon=3,
        num_episodes=2, terminate_on_success=True,
    )
    assert logs["square"]["Success_Rate"] == 0.5
    assert logs["square"]["Horizon"] == 2.5
    assert logs["square"]["Return"] == 2.5
    assert env.step.call_count == 5
    assert "failed to inject" not in capsys.readouterr().out


def test_binary_labels_are_required(dataset_path):
    config = make_config(dataset_path)
    with config.values_unlocked():
        config.algo.cami.continuous_contact.enabled = False
        config.train.dataset_keys = ["actions"]
    data_utils.prepare_cami_datasets(config)
    assert "contact_label" in config.train.dataset_keys
    with h5py.File(dataset_path, "a") as hdf5:
        del hdf5["data/demo_0/contact_label"]
    with pytest.raises(ValueError, match="contact_label"):
        data_utils.prepare_cami_datasets(config)


@pytest.mark.parametrize("quiet", [False, True])
def test_training_entrypoint_saves_resolved_config_and_checkpoint(dataset_path, tmp_path, capsys, quiet):
    from robomimic.scripts.train import train

    config = make_config(dataset_path)
    with config.values_unlocked():
        config.train.output_dir = str(tmp_path / "results")
        config.train.num_epochs = 1
        config.experiment.epoch_every_n_steps = 1
        config.experiment.validation_epoch_every_n_steps = 1
        config.experiment.validate = True
        config.train.hdf5_validation_filter_key = "valid"
        config.experiment.logging.terminal_output_to_txt = False
    config.lock()
    if quiet:
        from robomimic.utils.log_utils import log_warning
        log_warning("deferred setup warning remains visible", print_now=False)
    train(config, device=torch.device("cpu"), quiet=quiet)
    saved = list((tmp_path / "results").rglob("config.json"))
    assert len(saved) == 1
    stored = json.loads(saved[0].read_text())
    expected = np.r_[np.arange(1, 15), np.arange(1, 15) * 2].std() + 1e-6
    assert stored["algo"]["cami"]["continuous_contact"]["force_scale"] == pytest.approx(expected)
    assert stored["train"]["data"][0]["path"] == str(dataset_path)
    assert (saved[0].parent / "last.pth").is_file()
    output = capsys.readouterr().out
    if quiet:
        assert "New Training Run with Config" not in output
        assert "Model Summary" not in output
        assert '"BC_Action_Loss":' not in output
        assert "Train 1/1 | loss=" in output
        assert "Valid 1/1 | loss=" in output
        assert "Epoch 1/1 complete | checkpoint saved" in output
        assert "deferred setup warning remains visible" in output
    else:
        assert "New Training Run with Config" in output
        assert "Model Summary" in output


@pytest.mark.parametrize("quiet", [False, True])
def test_cli_reports_invalid_dataset_with_nonzero_exit(dataset_path, tmp_path, quiet):
    config = make_config(dataset_path)
    with config.values_unlocked():
        config.train.data = [{"path": str(tmp_path / "missing.hdf5")}]
        config.train.output_dir = str(tmp_path / "never_created")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(
        [sys.executable, "-m", "robomimic.scripts.train", "--config", str(path)]
        + (["--quiet"] if quiet else []),
        cwd=TEMPLATES.parents[2], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Dataset not found" in result.stdout + result.stderr
    assert not (tmp_path / "never_created").exists()
