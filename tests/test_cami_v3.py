"""v3 alignment, shared gradients, host adapters, baseline and checkpoint tests.

Synthetic demonstrations verify training mechanics, not robot task success.
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

from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.CAMI.cami_v3 import CaMIV3, warmup_factor
from robomimic.config import config_factory
from robomimic.scripts.generate_cami_v3_configs import make_configs, write_configs
from robomimic.scripts.prepare_cami_c_dataset import prepare_dataset
from robomimic.utils import cami_dataset_utils, file_utils, obs_utils, train_utils


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


@pytest.fixture
def data_path(tmp_path):
    raw, prepared = tmp_path / "raw.hdf5", tmp_path / "prepared.hdf5"
    rng = np.random.default_rng(8)
    with h5py.File(raw, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "NutAssemblySquare", "type": 1, "env_kwargs": {}})
        data.attrs["total"] = 72
        for i in range(3):
            d = data.create_group("demo_" + str(i))
            d.attrs["num_samples"] = 24
            d["actions"] = rng.uniform(-.5, .5, (24, 7)).astype(np.float32)
            d["obs/robot0_eef_pos"] = rng.normal(size=(24, 3)).astype(np.float32)
            d["obs/agentview_image"] = rng.integers(0, 256, (24, 84, 84, 3), dtype=np.uint8)
            t = np.arange(24, dtype=np.float32)
            physics = np.stack([.02 - .001 * t, .001 * t * (i + 1), .0003 * t ** 2], -1)
            if i == 2:
                physics *= 1000
            d["contact/physics"] = physics
            d["contact/physics"].attrs["feature_names"] = json.dumps(
                ["gap", "normal_impulse", "tangential_impulse_magnitude"])
            d["contact/dt"] = np.full(24, .05)
        f["mask/train"] = np.array([b"demo_0", b"demo_1"])
        f["mask/valid"] = np.array([b"demo_2"])
    prepare_dataset(raw, prepared, mode="physics", contact_key="contact/physics", quiet=True)
    return prepared


def config_for(path, *, prediction=False, enabled=True, recurrent=True, images=False):
    cfg = config_factory("bc_cami_v3")
    with cfg.values_unlocked():
        cfg.update(make_configs()["bc_cami_square_contactNetsv3.json"])
        cfg.train.data = [{"path": str(path)}]
        cfg.train.cuda = False
        cfg.train.batch_size = 3
        cfg.algo.actor_layer_dims = [16]
        cfg.algo.rnn.hidden_dim = 12
        cfg.algo.rnn.num_layers = 1
        cfg.algo.rnn.enabled = recurrent
        cfg.algo.cami.snippet_hidden_dim = 8
        cfg.algo.cami.contrastive_dim = 8
        cfg.algo.cami.query_proj_layers = [8]
        cfg.algo.cami.key_proj_layers = [8]
        cfg.algo.cami.v3.prediction.hidden_dims = [8]
        cfg.algo.cami.v3.prediction.enabled = prediction
        cfg.algo.cami.enabled = enabled
        cfg.algo.cami.v3.gradient_diagnostics_every = 1
        cfg.observation.modalities.obs.low_dim = ["robot0_eef_pos"]
        cfg.observation.modalities.obs.rgb = ["agentview_image"] if images else []
        cfg.experiment.rollout.enabled = False
        cfg.experiment.render_video = False
        cfg.experiment.logging.log_tb = False
        cfg.experiment.logging.log_wandb = False
        cfg.experiment.logging.terminal_output_to_txt = False
    return cfg


def setup_model(cfg):
    obs_utils.initialize_obs_utils_with_config(cfg, verbose=False)
    training, validation = train_utils.load_data_for_training(cfg, cfg.all_obs_keys)
    shapes = file_utils.get_shape_metadata_from_dataset(cfg.train.data[0], cfg.train.action_keys, cfg.all_obs_keys)
    model = algo_factory(cfg.algo_name, cfg, shapes["all_shapes"], shapes["ac_dim"], torch.device("cpu"))
    raw = next(iter(DataLoader(training, batch_size=3)))
    batch = model.postprocess_batch_for_training(model.process_batch_for_training(raw), obs_normalization_stats=None)
    return model, batch, training, validation, shapes


def close_datasets(*datasets):
    for dataset in datasets:
        if dataset is not None:
            dataset.close_and_delete_hdf5_handle()


def test_separate_configs_and_no_replacement(tmp_path):
    for name, value in make_configs().items():
        config = config_factory("bc_cami_v3")
        with config.values_unlocked():
            config.update(value)
        assert config.train.seed == 0
        assert config.train.num_epochs == 2000 and config.experiment.rollout.n == 50
        assert config.experiment.rollout.rate == 50
        if "baseline" in name:
            assert not config.algo.cami.enabled
    files = write_configs(tmp_path)
    assert len(files) == 9
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        write_configs(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize("epoch, expected", [(1, 0.), (51, .5), (101, 1.), (2000, 1.)])
def test_warmup_schedule(epoch, expected):
    assert warmup_factor(epoch, 100) == expected
    assert warmup_factor(epoch, 0) == 1.


@pytest.mark.parametrize("host_type", ["mlp", "gru"])
def test_core_has_generic_feature_interface_and_detached_labels(host_type):
    settings = dict(snippet_horizon=3, temperature=.1, normalize_embeddings=True,
        contrastive_dim=4, query_proj_layers=[8], key_proj_layers=[8], snippet_hidden_dim=6,
        snippet_num_layers=1, momentum=.9, use_momentum_target=True, feature_scales=[1., 1., 1.],
        feature_weights=[1., 1., 1.], contact_temperature=.05, gamma=1., huber_delta=.1,
        chunk_size=64, distance_penalty="huber", prediction_enabled=True,
        prediction_indices=[0], prediction_hidden_dims=[8], prediction_huber_delta=.1)
    host = torch.nn.Linear(3, 5) if host_type == "mlp" else torch.nn.GRU(3, 5, batch_first=True)
    features = host(torch.randn(4, 3)) if host_type == "mlp" else host(torch.randn(4, 6, 3))[0][:, -1]
    module = CaMIV3(5, 2, settings)
    labels = torch.rand(4, 3, 3, requires_grad=True)
    losses, _ = module(features, torch.randn(4, 3, 2), torch.tensor([3, 3, 2, 0]),
                       labels, torch.ones(4, 3, 3))
    sum(losses.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in host.parameters())
    assert labels.grad is None
    assert all(p.grad is None for p in module.snippet_target.parameters())
    assert all(p.grad is None for p in module.key_target.parameters())


@pytest.mark.parametrize("recurrent,prediction", [(True, False), (True, True), (False, True)])
def test_training_validation_reload_and_force_free_rollout(data_path, recurrent, prediction):
    cfg = config_for(data_path, recurrent=recurrent, prediction=prediction)
    model, batch, training, validation, shapes = setup_model(cfg)
    previous = next(model.nets["policy"].parameters()).detach().clone()
    info = model.train_on_batch(batch, epoch=101)
    logs = model.log_info(info)
    assert all(np.isfinite(x) for x in logs.values())
    assert not torch.equal(previous, next(model.nets["policy"].parameters()))
    assert logs["CaMI_V3/gradient/State_Policy_Norm"] > 0
    assert logs["CaMI_V3/state_weight"] == pytest.approx(.01)
    if prediction:
        assert logs["Prediction_CaMI_Loss"] > 0
        assert logs["CaMI_V3/gradient/Prediction_Policy_Norm"] > 0
    assert all(p.grad is None for p in model.nets["cami"].snippet_target.parameters())
    state = {k: v.clone() for k, v in model.nets.state_dict().items()}
    model.set_eval()
    model.train_on_batch(batch, epoch=101, validate=True)
    assert all(torch.equal(state[k], value) for k, value in model.nets.state_dict().items())
    restored = algo_factory(cfg.algo_name, cfg, shapes["all_shapes"], shapes["ac_dim"], torch.device("cpu"))
    restored.deserialize(model.serialize(), load_optimizers=True)
    assert restored._gradient_steps == 1
    with h5py.File(data_path) as f:
        observation = {key: f["data/demo_0/obs/" + key][0] for key in cfg.all_obs_keys}
    policy, replay = RolloutPolicy(model), RolloutPolicy(restored)
    policy.start_episode(); replay.start_episode()
    np.testing.assert_allclose(policy(observation), replay(observation), rtol=1e-6, atol=1e-6)
    close_datasets(training, validation)


def test_anchor_uses_context_and_matching_future(data_path, monkeypatch):
    cfg = config_for(data_path)
    model, batch, training, validation, _ = setup_model(cfg)
    captured = {}
    real_forward = model.nets["cami"].forward
    def capture(features, actions, lengths, contact, mask):
        captured.update(features=features, actions=actions, contact=contact, mask=mask)
        return real_forward(features, actions, lengths, contact, mask)
    monkeypatch.setattr(model.nets["cami"], "forward", capture)
    model.set_eval()
    expected_actions, expected_features = model._policy_training_outputs(batch["obs"], batch["goal_obs"])
    output = model._forward_training(batch)
    torch.testing.assert_close(output["actions"], expected_actions)
    torch.testing.assert_close(captured["features"], expected_features[:, 5])
    torch.testing.assert_close(captured["actions"], batch["actions"][:, 6:16])
    torch.testing.assert_close(captured["contact"], batch["contact_state"][:, 5])
    # Changes after the decision time cannot change its causal features.
    changed = {key: value.clone() for key, value in batch["obs"].items()}
    for value in changed.values():
        value[:, 6:] += 100
    later_features = model._policy_training_outputs(changed, batch["goal_obs"])[1]
    torch.testing.assert_close(later_features[:, 5], expected_features[:, 5])
    close_datasets(training, validation)


def test_no_cami_baseline_requires_no_descriptors_and_matches_native_policy(data_path):
    with h5py.File(data_path, "a") as f:
        for demo in f["data"].values():
            del demo["contact_state"], demo["contact_mask"]
    cfg = config_for(data_path, enabled=False)
    torch.manual_seed(3)
    model, batch, training, validation, shapes = setup_model(cfg)
    assert list(model.nets) == ["policy"] and list(model.optimizers) == ["policy"]
    assert "contact_state" not in batch
    native_dict = cfg.to_dict()
    native_dict["algo_name"] = "bc"
    native_cfg = config_factory("bc", dic=native_dict)
    torch.manual_seed(3)
    native = algo_factory("bc", native_cfg, shapes["all_shapes"], shapes["ac_dim"], torch.device("cpu"))
    for key, value in native.nets["policy"].state_dict().items():
        torch.testing.assert_close(model.nets["policy"].state_dict()[key], value)
    model.set_eval(); native.set_eval()
    actual = model._forward_training(batch)["actions"]
    expected = native.nets["policy"](batch["obs"])
    torch.testing.assert_close(actual, expected)
    losses = model.train_on_batch(batch, 101, validate=True)["losses"]
    torch.testing.assert_close(losses["action_loss"], losses["bc_action_loss"])
    assert losses["state_cami_loss"] == 0
    close_datasets(training, validation)


def test_all_padded_future_has_zero_auxiliary_loss(data_path):
    cfg = config_for(data_path, prediction=True)
    model, batch, training, validation, _ = setup_model(cfg)
    batch["pad_mask"][:, 5:] = 0
    batch["contact_mask"][:] = 0
    batch["contact_state"][:] = float("nan")
    info = model.train_on_batch(batch, epoch=101)
    for key in ("state_cami_loss", "traj_cami_loss", "prediction_cami_loss"):
        assert info["losses"][key].item() == 0
    close_datasets(training, validation)


def test_scales_use_only_training_demos(data_path):
    cfg = config_for(data_path)
    cami_dataset_utils.prepare_cami_datasets(cfg)
    with h5py.File(data_path) as f:
        values = np.concatenate([f["data/" + name + "/contact/physics"][1:]
                                 for name in ("demo_0", "demo_1")])
    c = cfg.algo.cami.continuous
    np.testing.assert_allclose([c.gap_scale, c.normal_impulse_scale, c.tangential_impulse_scale],
                               values.astype(np.float64).std(0) + 1e-6, rtol=1e-6)


def test_auxiliary_gradient_paths_are_distinct(data_path):
    cfg = config_for(data_path, prediction=True)
    model, batch, training, validation, _ = setup_model(cfg)
    output = model._forward_training(batch)
    losses = output["cami_losses"]
    params = list(model.nets["policy"].parameters())
    gradients = torch.autograd.grad(losses["traj_loss"], params, allow_unused=True, retain_graph=True)
    assert all(g is None for g in gradients)
    snippet_params = list(model.nets["cami"].snippet_encoder.parameters())
    gradients = torch.autograd.grad(losses["prediction_loss"], snippet_params, allow_unused=True, retain_graph=True)
    assert all(g is None for g in gradients)
    gradients = torch.autograd.grad(losses["prediction_loss"], params, allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)
    close_datasets(training, validation)


def test_camera_training_uses_existing_policy_and_single_pass_features(data_path):
    cfg = config_for(data_path, prediction=True, images=True)
    model, batch, training, validation, _ = setup_model(cfg)
    info = model.train_on_batch(batch, epoch=101)
    assert np.isfinite(model.log_info(info)["Loss"])
    model.set_eval()
    with torch.no_grad():
        actions = model._forward_training(batch)["actions"]
        torch.testing.assert_close(actions, model.nets["policy"](batch["obs"]))
    close_datasets(training, validation)


def test_real_v3_entry_point_and_checkpoint(data_path, tmp_path):
    cfg = config_for(data_path, prediction=True)
    with cfg.values_unlocked():
        cfg.experiment.name = "v3_smoke"
        cfg.train.output_dir = str(tmp_path / "models")
        cfg.train.num_epochs = 2
        cfg.experiment.epoch_every_n_steps = 2
        cfg.experiment.validation_epoch_every_n_steps = 1
        cfg.algo.cami.v3.warmup_epochs = 1
        for name in ("policy", "cami"):
            cfg.algo.optim_params[name].learning_rate.epoch_schedule = [1]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    result = subprocess.run([sys.executable, "-m", "robomimic.scripts.train_cami_v3",
                             "--config", str(path), "--quiet"], cwd=ROOT,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Valid 2/2" in result.stdout
    run = next((tmp_path / "models/v3_smoke").iterdir())
    checkpoint = torch.load(run / "last.pth", map_location="cpu", weights_only=False)
    assert checkpoint["model"]["cami_v3_gradient_steps"] == 4
    for name in ("policy", "cami"):
        assert checkpoint["model"]["optimizers"][name]["param_groups"][0]["lr"] == pytest.approx(5e-5)
    policy, _ = file_utils.policy_from_checkpoint(ckpt_path=str(run / "last.pth"), device=torch.device("cpu"))
    with h5py.File(data_path) as f:
        observation = {key: f["data/demo_0/obs/" + key][0] for key in cfg.all_obs_keys}
    policy.start_episode()
    action = policy(observation)
    assert action.shape == (7,) and np.isfinite(action).all() and (np.abs(action) <= 1).all()
