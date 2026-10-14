"""Native MuJoCo contact integration and action replay, using a small adapter."""

import json
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np
import pytest
import torch

from robomimic.CAMI import continuous_mujoco as physics
from robomimic.CAMI.continuous_contact import (
    contact_distance, embedding_logits, negative_weights, normalize_contact, weighted_infonce,
)
from robomimic.scripts.prepare_cami_c_dataset import prepare_dataset


XML = '''<mujoco><option timestep="0.002" integrator="Euler"/>
<worldbody><geom name="floor" type="plane" size="1 1 .1" friction="1 .005 .0001"/>
<body name="ball_body" pos="0 0 .049"><joint name="x" type="slide" axis="1 0 0"/>
<joint name="z" type="slide" axis="0 0 1"/>
<geom name="ball" type="sphere" size=".05" mass=".1" friction="1 .005 .0001"/>
</body></worldbody><actuator><motor joint="x"/><motor joint="z"/></actuator></mujoco>'''
PAIRS = {'groups': [{'a': ['ball'], 'b': ['floor']}]}


class Sim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def step2(self):
        mujoco.mj_step2(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)


class Env:
    def __init__(self):
        self.env = SimpleNamespace(sim=Sim(), robots=[SimpleNamespace(part_controllers={})], close=lambda: None)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.env.sim.model, self.env.sim.data)
        self.env.sim.forward()

    def reset_to(self, state):
        values, data = state['states'], self.env.sim.data
        data.time, data.qpos[:], data.qvel[:] = values[0], values[1:3], values[3:5]
        self.env.sim.forward()

    def get_state(self):
        data = self.env.sim.data
        return np.r_[data.time, data.qpos, data.qvel]

    def step(self, action):
        self.env.sim.data.ctrl[:] = action
        for _ in range(5):
            self.env.sim.step()
        return {}, 0., False, {}


def test_integrated_normal_and_tangential_impulses():
    env, reference = Env(), Env()
    sim = env.env.sim
    pairs = physics.resolve_pairs(sim.model, PAIRS)
    probe = physics.ContinuousContactProbe(sim.model, sim.data, pairs)
    manual = physics.ContinuousContactProbe(reference.env.sim.model, reference.env.sim.data, pairs)
    action = np.array([.3, -.8])
    expected = np.zeros(2)
    reference.env.sim.data.ctrl[:] = action
    for _ in range(5):
        reference.env.sim.step()
        for row in manual.contacts():
            expected += .002 * np.array([max(0., row['wrench'][0]), np.linalg.norm(row['wrench'][1:3])])
    _, measured = physics.step_with_contact(env, action, probe)
    np.testing.assert_allclose(measured['physics'][1:], expected, rtol=1e-6)
    assert np.all(expected > 0)
    assert measured['dt'] == pytest.approx(.01) and measured['substeps'] == 5
    assert measured['physics'][0] == pytest.approx(-.001, abs=1e-7)
    assert measured['valid'].all()
    assert 'step' not in vars(sim) and 'step2' not in vars(sim)


def recorded_dataset(path):
    env = Env()
    actions = np.tile([.3, -.8], (12, 1))
    states = []
    for action in actions:
        states.append(env.get_state())
        env.step(action)
    states.append(env.get_state())
    with h5py.File(path, 'w') as f:
        f['data/demo_0/actions'] = actions
        f['data/demo_0/states'] = states
        f['data/demo_0'].attrs.update(num_samples=len(actions), model_file=XML)
        f['data'].attrs['env_args'] = json.dumps({'env_name': 'Fixture', 'type': 1, 'env_kwargs': {}})
    return env


def test_replay_preparation_and_atomic_failure(tmp_path, monkeypatch):
    source, output = tmp_path / 'source.hdf5', tmp_path / 'physics.hdf5'
    recorded_dataset(source)
    monkeypatch.setattr(physics, 'make_replay_env', lambda group: Env())
    report = prepare_dataset(source, output, mode='physics', pairs=PAIRS, horizon=10, quiet=True)
    assert report['demos']['demo_0']['max_qpos_replay_error'] < 1e-12
    assert report['interval_dt'] == pytest.approx(.01)
    with h5py.File(output) as f:
        assert f['data/demo_0/contact_state'].shape == (12, 10, 3)
        assert f['data/demo_0/contact_state'][0, :, 1].max() > 0
        assert f['data/demo_0/contact_state'][0, :, 2].max() > 0
        assert not f['data/demo_0/contact_mask'][-1].any()
    with h5py.File(source, 'a') as f:
        assert 'contact_state' not in f['data/demo_0']
        f['data/demo_0/states'][3, 1] += .02
    with pytest.raises(ValueError, match='Replay diverged'):
        prepare_dataset(source, tmp_path / 'bad.hdf5', mode='physics', pairs=PAIRS, quiet=True)
    assert not (tmp_path / 'bad.hdf5').exists()
    assert not list(tmp_path.glob('bad.hdf5.*.tmp'))


def test_state_aligned_reconstruction_restores_anchors_and_reports_drift(tmp_path, monkeypatch):
    source, output = tmp_path / 'source.hdf5', tmp_path / 'state_aligned.hdf5'
    recorded_dataset(source)
    monkeypatch.setattr(physics, 'make_replay_env', lambda group: Env())
    # A saved state inconsistent with open-loop replay must not move later
    # estimated intervals away from their own recorded anchors.
    with h5py.File(source, 'a') as f:
        f['data/demo_0/states'][3, 2] += .01
    report = prepare_dataset(source, output, mode='physics', pairs=PAIRS,
                             horizon=10, replay_mode='state', quiet=True)
    reconstruction = report['reconstruction']
    assert reconstruction['replay_mode'] == 'state'
    assert reconstruction['impulse_semantics'] == 'state_conditioned_simulator_estimate'
    assert reconstruction['transitions_over_tolerance'] > 0
    assert reconstruction['checked_transitions'] == 12
    assert reconstruction['max_qpos_replay_error'] > .001
    assert reconstruction['mujoco_version'] == mujoco.__version__
    with h5py.File(source) as f:
        demo = f['data/demo_0']
        original_states = demo['states'][()]
        values, details = physics.replay_descriptors(Env(), demo, PAIRS, replay_mode='state')
        assert details['gap_semantics'] == 'recorded_state_geometry'
        reference = Env()
        sim = reference.env.sim
        probe = physics.ContinuousContactProbe(sim.model, sim.data,
                                              physics.resolve_pairs(sim.model, PAIRS))
        for t, action in enumerate(demo['actions']):
            sim.data.qacc_warmstart[:] = 0.
            reference.reset_to({'states': original_states[t]})
            expected_gap = probe.gap()
            reference.env.sim.data.ctrl[:] = action
            expected_impulses = np.zeros(2)
            for _ in range(5):
                sim.step()
                for contact in probe.contacts():
                    expected_impulses += .002 * np.array([
                        max(0., contact['wrench'][0]), np.linalg.norm(contact['wrench'][1:3])])
            np.testing.assert_allclose(values[t], np.r_[expected_gap, expected_impulses],
                                       rtol=1e-6, atol=1e-8)
        assert 'contact_state' not in demo
        np.testing.assert_array_equal(demo['states'][()], original_states)
    with h5py.File(output) as f:
        stored = json.loads(f['data'].attrs['cami_c'])
        assert stored['reconstruction'] == reconstruction
        np.testing.assert_allclose(f['data/demo_0/contact_state'][0], values[1:11])
        assert not f['data/demo_0/contact_mask'][-1].any()


@pytest.mark.parametrize('replay_mode', ['action', 'state'])
def test_replay_rejects_mismatched_action_timing(tmp_path, replay_mode):
    source = tmp_path / 'source.hdf5'
    recorded_dataset(source)
    with h5py.File(source, 'a') as f:
        f['data/demo_0/states'][:, 0] *= 2.
        with pytest.raises(ValueError, match='does not match recorded interval'):
            physics.replay_descriptors(Env(), f['data/demo_0'], PAIRS, replay_mode=replay_mode)


def test_state_reconstruction_requires_physics_pairs(tmp_path):
    source = tmp_path / 'source.hdf5'
    recorded_dataset(source)
    with pytest.raises(ValueError, match='State-aligned replay requires'):
        prepare_dataset(source, tmp_path / 'output.hdf5', mode='force', replay_mode='state')
    assert not (tmp_path / 'output.hdf5').exists()


def test_huber_weights_and_gradient_paths():
    labels = torch.tensor([[[0., 0., 0.]], [[.2, .2, .2]]], requires_grad=True)
    normalized = normalize_contact(labels, ['gap', 'normal_impulse', 'tangential_impulse_magnitude'],
                                   gap_scale=1., normal_impulse_scale=1., tangential_impulse_scale=1.)
    distance, valid = contact_distance(normalized, torch.ones_like(labels), [1., 1., 1.],
                                       delta=.1, penalty='huber')
    assert distance[0, 1] == pytest.approx(.045)
    weights = negative_weights(distance, valid, contact_temperature=.05)
    assert weights[0, 1] == pytest.approx(1 - np.exp(-.9))
    assert not weights.diagonal().any()
    query, key = torch.randn(2, 4, requires_grad=True), torch.randn(2, 4, requires_grad=True)
    logits = embedding_logits(query, key)
    loss, _ = weighted_infonce(logits, weights)
    expected = torch.log1p(weights[0, 1] * torch.exp(logits[0, 1] - logits[0, 0]))
    expected += torch.log1p(weights[1, 0] * torch.exp(logits[1, 0] - logits[1, 1]))
    torch.testing.assert_close(loss, expected / 2)
    loss.backward()
    assert query.grad is not None and key.grad is None and labels.grad is None
    zero, _ = weighted_infonce(logits.detach(), torch.zeros_like(weights))
    assert zero == 0
