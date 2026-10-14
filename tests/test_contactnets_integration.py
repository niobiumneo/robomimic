"""Full-descriptor preparation, loader, optimizer and rollout interface tests.

Synthetic fixtures exercise the interface and do not measure task success.
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

from robomimic.algo import algo_factory, algo_name_to_factory_func, RolloutPolicy
from robomimic.config import config_factory
from robomimic.scripts.prepare_cami_c_dataset import prepare_dataset
from robomimic.utils import cami_dataset_utils, file_utils, obs_utils, train_utils


TEMPLATE = Path(__file__).resolve().parents[1] / "robomimic/exps/templates/bc_cami_square_contactNetsv2.json"


@pytest.fixture
def physics_dataset(tmp_path):
    source, output = tmp_path / "recorded.hdf5", tmp_path / "prepared.hdf5"
    rng = np.random.default_rng(4)
    with h5py.File(source, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "NutAssemblySquare", "type": 1, "env_kwargs": {}})
        data.attrs["total"] = 42
        for index in range(3):
            d = data.create_group("demo_" + str(index))
            d.attrs["num_samples"] = 14
            d["actions"] = rng.uniform(-.5, .5, (14, 7)).astype(np.float32)
            d["obs/robot0_eef_pos"] = rng.normal(size=(14, 3)).astype(np.float32)
            d["obs/agentview_image"] = rng.integers(0, 256, (14, 84, 84, 3), dtype=np.uint8)
            t = np.arange(14, dtype=np.float32)
            values = np.stack([.002 - .0002 * t, .001 * (index + 1) * t, .0005 * t ** 2], axis=1)
            if index == 2:
                values *= 1000  # Expose accidental use of validation data in fitting.
            d["contact/physics"] = values
            d["contact/physics"].attrs["feature_names"] = json.dumps(
                ["gap", "normal_impulse", "tangential_impulse_magnitude"])
            d["contact/dt"] = np.full(14, .05)
            d["contact/valid"] = np.ones((14, 3), np.uint8)
        f["mask/train"] = np.array([b"demo_0", b"demo_1"])
        f["mask/valid"] = np.array([b"demo_2"])
    prepare_dataset(source, output, mode="physics", contact_key="contact/physics", quiet=True)
    return output


def config_for(path, images=False):
    config = config_factory("bc_cami_contactNets")
    with config.values_unlocked():
        config.update(json.loads(TEMPLATE.read_text()))
        config.train.data = [{"path": str(path)}]
        config.train.cuda = False
        config.train.batch_size = 2
        config.algo.actor_layer_dims = [16]
        config.algo.rnn.hidden_dim = 12
        config.algo.rnn.num_layers = 1
        config.algo.cami.snippet_hidden_dim = 8
        config.algo.cami.contrastive_dim = 8
        config.algo.cami.query_proj_layers = [8]
        config.algo.cami.key_proj_layers = [8]
        config.observation.modalities.obs.low_dim = ["robot0_eef_pos"]
        config.observation.modalities.obs.rgb = ["agentview_image"] if images else []
        config.experiment.rollout.enabled = False
        config.experiment.render_video = False
        config.experiment.logging.log_tb = False
        config.experiment.logging.log_wandb = False
        config.experiment.logging.terminal_output_to_txt = False
    return config


def test_config_and_algorithm_registration():
    config = config_factory("bc_cami_contactNets")
    assert algo_name_to_factory_func(config.algo_name)(config.algo)[0].__name__ == "BC_CaMI_ContactNets"
    assert config_factory("bc_cami").algo_name == "bc_cami"
    with config.values_unlocked():
        config.update(json.loads(TEMPLATE.read_text()))
    assert config.train.num_epochs == 2000 and config.experiment.rollout.rate == 50
    assert config.experiment.validate and config.train.hdf5_validation_filter_key == "valid"
    for name in ("policy", "state_encoder", "snippet_encoder", "key_proj"):
        lr = config.algo.optim_params[name].learning_rate
        assert lr.initial == 1e-4 and lr.decay_factor == .5 and lr.epoch_schedule == [1000]
        assert not lr.step_every_batch


def test_legacy_force_only_checkpoint_selection():
    saved = config_factory("bc_cami").to_dict()
    saved['algo_name'] = 'bc_cami_contactNets'
    config = config_factory('bc_cami_contactNets', dic=saved)
    assert 'continuous' not in config.algo.cami
    assert algo_name_to_factory_func(config.algo_name)(config.algo)[0].__name__ == 'BC_CaMI'


def test_training_only_scales_and_alignment(physics_dataset):
    config = config_for(physics_dataset)
    config.lock()
    cami_dataset_utils.prepare_cami_datasets(config)
    with h5py.File(physics_dataset) as f:
        values = np.concatenate([f['data/' + n + '/contact/physics'][1:] for n in ('demo_0', 'demo_1')])
        np.testing.assert_array_equal(f['data/demo_0/contact_state'][0], f['data/demo_0/contact/physics'][1:11])
        assert not f['data/demo_0/contact_mask'][-1].any()
    c = config.algo.cami.continuous
    np.testing.assert_allclose([c.gap_scale, c.normal_impulse_scale, c.tangential_impulse_scale],
                               values.astype(np.float64).std(0) + 1e-6, rtol=1e-6)
    assert all('demo_2' not in name for name in c.normalization_fit_demos)


def test_training_config_preserves_reconstruction_provenance(physics_dataset):
    provenance = {'replay_mode': 'state',
                  'impulse_semantics': 'state_conditioned_simulator_estimate',
                  'max_qpos_replay_error': .0034,
                  'transitions_over_tolerance': 1}
    with h5py.File(physics_dataset, 'a') as f:
        meta = json.loads(f['data'].attrs['cami_c'])
        meta['reconstruction'] = provenance
        f['data'].attrs['cami_c'] = json.dumps(meta)
    config = config_for(physics_dataset)
    config.lock()
    cami_dataset_utils.prepare_cami_datasets(config)
    saved = config.to_dict()['algo']['cami']['continuous']['dataset_reconstruction']
    assert saved == [dict(provenance, dataset=str(physics_dataset))]


@pytest.mark.parametrize('corruption, message', [
    ('missing', 'contact_state'), ('features', 'incompatible physics'),
    ('padding', 'padded future'), ('negative', 'nonnegative'), ('nan', 'nonfinite'),
    ('units', 'incompatible physics'),
])
def test_corrupt_data_fails_before_loader(physics_dataset, corruption, message):
    with h5py.File(physics_dataset, 'a') as f:
        if corruption in ('features', 'units'):
            meta = json.loads(f['data'].attrs['cami_c'])
            meta['feature_names' if corruption == 'features' else 'units'] = ['force']
            f['data'].attrs['cami_c'] = json.dumps(meta)
        elif corruption == 'missing':
            del f['data/demo_0/contact_state']
        elif corruption == 'padding':
            f['data/demo_0/contact_mask'][-1, 0, 0] = 1
        else:
            f['data/demo_0/contact_state'][0, 0, 1] = -1 if corruption == 'negative' else np.nan
    with pytest.raises(ValueError, match=message):
        cami_dataset_utils.prepare_cami_datasets(config_for(physics_dataset))


@pytest.mark.parametrize('cache_mode', [None, 'low_dim', 'all'])
def test_padding_in_all_cache_modes(physics_dataset, cache_mode):
    config = config_for(physics_dataset)
    with config.values_unlocked():
        config.train.hdf5_cache_mode = cache_mode
    obs_utils.initialize_obs_utils_with_config(config, verbose=False)
    training, validation = train_utils.load_data_for_training(config, config.all_obs_keys)
    try:
        assert training[0]['contact_state'].shape == (11, 10, 3)
        assert training[13]['pad_mask'].sum() == 1
        assert not training[13]['contact_mask'][0].any()
        assert validation is not None
        assert 'contact_state' not in training[0]['obs']
    finally:
        training.close_and_delete_hdf5_handle()
        validation.close_and_delete_hdf5_handle()


@pytest.mark.parametrize('images', [False, True])
def test_optimizer_validation_and_force_free_rollout(physics_dataset, images):
    torch.set_num_threads(2)
    config = config_for(physics_dataset, images)
    obs_utils.initialize_obs_utils_with_config(config, verbose=False)
    training, validation = train_utils.load_data_for_training(config, config.all_obs_keys)
    shapes = file_utils.get_shape_metadata_from_dataset(config.train.data[0], config.train.action_keys, config.all_obs_keys)
    model = algo_factory(config.algo_name, config, shapes['all_shapes'], shapes['ac_dim'], torch.device('cpu'))
    previous = next(model.nets['policy'].parameters()).detach().clone()
    logs = train_utils.run_epoch(model, DataLoader(training, batch_size=2), epoch=1, num_steps=2)
    assert all(np.isfinite(logs[key]) for key in ['Loss', 'BC_Action_Loss', 'State_CaMI_Loss', 'Traj_CaMI_Loss'])
    assert not torch.equal(previous, next(model.nets['policy'].parameters()))
    assert all(param.grad is None for name in ('snippet_encoder_target', 'key_proj_target')
               for param in model.nets[name].parameters())
    previous = {name: tensor.detach().clone() for name, tensor in model.nets.state_dict().items()}
    valid_logs = train_utils.run_epoch(model, DataLoader(validation, batch_size=2), epoch=1, validate=True, num_steps=1)
    assert np.isfinite(valid_logs['Loss'])
    assert all(torch.equal(previous[name], value) for name, value in model.nets.state_dict().items())
    model.set_eval()
    raw = next(iter(DataLoader(training, batch_size=2)))
    batch = model.postprocess_batch_for_training(model.process_batch_for_training(raw), obs_normalization_stats=None)
    with torch.no_grad():
        feature_actions = model._forward_policy_with_latent(batch['obs'])[0]
        torch.testing.assert_close(feature_actions, model.nets['policy'](obs_dict=batch['obs']))
    reloaded = algo_factory(config.algo_name, config, shapes['all_shapes'], shapes['ac_dim'], torch.device('cpu'))
    reloaded.deserialize(model.serialize())
    with h5py.File(physics_dataset) as f:
        observation = {key: f['data/demo_0/obs/' + key][0] for key in config.all_obs_keys}
    policy, restored = RolloutPolicy(model), RolloutPolicy(reloaded)
    policy.start_episode(); restored.start_episode()
    np.testing.assert_allclose(policy(observation), restored(observation), rtol=1e-6, atol=1e-6)
    assert set(observation).isdisjoint({'force', 'contact_state', 'contact_mask'})
    training.close_and_delete_hdf5_handle()
    validation.close_and_delete_hdf5_handle()


def test_real_training_entry_point(physics_dataset, tmp_path):
    config = config_for(physics_dataset)
    with config.values_unlocked():
        config.experiment.name = 'contactnets_smoke'
        config.train.output_dir = str(tmp_path / 'models')
        config.train.num_epochs = 2
        config.experiment.epoch_every_n_steps = 2
        config.experiment.validation_epoch_every_n_steps = 1
        for name in ('policy', 'state_encoder', 'snippet_encoder', 'key_proj'):
            config.algo.optim_params[name].learning_rate.epoch_schedule = [1]
    path = tmp_path / 'train.json'
    path.write_text(json.dumps(config.to_dict()))
    result = subprocess.run([sys.executable, '-m', 'robomimic.scripts.train', '--config', str(path), '--quiet'],
                            cwd=TEMPLATE.parents[3], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Valid 2/2' in result.stdout
    run_dir = next((tmp_path / 'models/contactnets_smoke').iterdir())
    checkpoint = torch.load(run_dir / 'last.pth', map_location='cpu', weights_only=False)
    for name in ('policy', 'state_encoder', 'snippet_encoder', 'key_proj'):
        assert checkpoint['model']['optimizers'][name]['param_groups'][0]['lr'] == pytest.approx(5e-5)
