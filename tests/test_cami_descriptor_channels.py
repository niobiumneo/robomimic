"""Per-channel masking, descriptor transforms and the per-pair gap rule.

Synthetic inputs check the arithmetic and the bookkeeping, not task success.
"""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from robomimic.CAMI import continuous_mujoco as physics
from robomimic.CAMI.cami_v3 import CaMIV3
from robomimic.CAMI.continuous_contact import contact_distance, transform_descriptors
from robomimic.scripts import recompute_cami_gaps
from robomimic.scripts.prepare_cami_c_dataset import prepare_dataset
from test_contactnets_physics import Env, PAIRS, recorded_dataset


@pytest.mark.parametrize("penalty", ["l1", "huber"])
def test_feature_masking_equals_timestep_masking_without_missing_data(penalty):
    torch.manual_seed(0)
    values, mask = torch.randn(6, 4, 3), torch.ones(6, 4, 3)
    weights = [1., .5, .25]
    a = contact_distance(values, mask, weights, penalty=penalty, missing="timestep")
    b = contact_distance(values, mask, weights, penalty=penalty, missing="feature")
    torch.testing.assert_close(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_an_unknown_gap_keeps_the_impulses_of_its_timestep():
    values = torch.tensor([[[.3, 1., 2.]], [[9., 4., 6.]]])
    mask = torch.ones(2, 1, 3)
    mask[1, 0, 0] = 0  # gap unknown in the second sample
    distance, valid = contact_distance(values, mask, [1., 1., 1.], missing="timestep")
    assert not valid[0, 1] and not valid[1, 0]
    distance, valid = contact_distance(values, mask, [1., 1., 1.], missing="feature")
    assert valid[0, 1] and valid[1, 0]
    # |1-4| + |2-6| over the two observed channels, rescaled by sum(w) / sum(observed w) = 3/2.
    assert distance[0, 1].item() == pytest.approx((3. + 4.) * 3 / 2)
    assert distance[0, 0].item() == 0 and valid[0, 0]


def test_feature_masking_averages_only_timesteps_with_an_observed_channel():
    values = torch.tensor([[[0., 0., 0.], [0., 0., 0.]], [[1., 1., 1.], [5., 5., 5.]]])
    mask = torch.ones(2, 2, 3)
    mask[1, 1] = 0  # nothing observed at the second step of the second sample
    distance, valid = contact_distance(values, mask, [1., 1., 1.], missing="feature")
    assert valid[0, 1] and distance[0, 1].item() == pytest.approx(3.)


def test_disabled_channels_never_affect_validity_or_distance():
    values = torch.tensor([[[.1, 1., 7.]], [[.2, 3., 0.]]])
    mask = torch.ones(2, 1, 3)
    mask[0, 0, 2] = 0
    for missing in ("timestep", "feature"):
        distance, valid = contact_distance(values, mask, [1., 1., 0.], missing=missing)
        assert valid[0, 1] and distance[0, 1].item() == pytest.approx(.1 + 2.)


def test_contact_distance_rejects_an_unknown_missing_mode():
    with pytest.raises(ValueError, match="missing"):
        contact_distance(torch.zeros(1, 1, 3), torch.ones(1, 1, 3), [1., 1., 1.], missing="impute")


def test_descriptor_transform_clips_gaps_and_compresses_impulses_on_numpy_and_torch():
    raw = np.array([[.02, 0., 0.], [-.001, .05, .02], [-.012, 2.4, 1.1]])
    expected = raw.copy()
    expected[:, 0] = np.maximum(raw[:, 0], -.005)
    expected[:, 1:] = .1 * np.log1p(raw[:, 1:] / .1)
    np.testing.assert_allclose(transform_descriptors(raw, -.005, .1), expected)
    np.testing.assert_allclose(transform_descriptors(torch.as_tensor(raw), -.005, .1).numpy(), expected)
    assert transform_descriptors(raw) is raw
    assert transform_descriptors(raw, gap_min=-.005)[2, 1] == raw[2, 1]
    with pytest.raises(ValueError, match="impulse_log_scale"):
        transform_descriptors(raw, impulse_log_scale=0.)


def module_settings(**extra):
    settings = dict(snippet_horizon=2, temperature=.1, normalize_embeddings=True,
        contrastive_dim=4, query_proj_layers=[8], key_proj_layers=[8], snippet_hidden_dim=6,
        snippet_num_layers=1, momentum=.9, use_momentum_target=True, feature_scales=[.01, .4, .3],
        feature_weights=[1., 1., 1.], contact_temperature=.05, gamma=1., huber_delta=.1,
        chunk_size=64, distance_penalty="huber", prediction_enabled=True,
        prediction_indices=[0, 1, 2], prediction_hidden_dims=[8], prediction_huber_delta=.1)
    settings.update(extra)
    return settings


def test_module_settings_saved_before_the_options_keep_their_behavior():
    module = CaMIV3(5, 2, module_settings())
    assert module.settings["missing_features"] == "timestep"
    assert module.settings["gap_min"] is None and module.settings["impulse_log_scale"] is None
    with pytest.raises(ValueError, match="missing_features"):
        CaMIV3(5, 2, module_settings(missing_features="impute"))


def test_module_applies_the_transform_before_scaling_and_the_prediction_targets_follow(monkeypatch):
    import robomimic.CAMI.cami_v3 as cami_v3
    seen = {}

    def spy(normalized, mask, *args, **kwargs):
        seen["normalized"], seen["missing"] = normalized.clone(), kwargs["missing"]
        return contact_distance(normalized, mask, *args, **kwargs)

    targets = []
    huber = cami_v3.F.huber_loss

    def huber_spy(predicted, target, **kwargs):
        targets.append(target.clone())
        return huber(predicted, target, **kwargs)

    monkeypatch.setattr(cami_v3, "contact_distance", spy)
    monkeypatch.setattr(cami_v3.F, "huber_loss", huber_spy)
    module = CaMIV3(5, 2, module_settings(missing_features="feature", gap_min=-.005, impulse_log_scale=.1))
    contact = torch.tensor([[[-.02, 3., 2.], [.01, 0., 0.]], [[.02, 0., 0.], [-.001, .5, .2]]])
    mask = torch.ones(2, 2, 3)
    mask[0, 1, 0] = 0
    losses, _ = module(torch.randn(2, 5), torch.randn(2, 2, 2), torch.tensor([2, 2]), contact, mask)
    expected = transform_descriptors(torch.where(mask.bool(), contact, 0.), -.005, .1) / module.feature_scales
    torch.testing.assert_close(seen["normalized"], torch.where(mask.bool(), expected, 0.))
    assert seen["missing"] == "feature"
    torch.testing.assert_close(targets[-1], seen["normalized"][..., list(module.prediction_indices)])
    assert torch.isfinite(losses["prediction_loss"])


class FakeDistance:
    """Stand-in for mujoco.mj_geomDistance that returns scripted (distance, witness length) per pair."""

    def __init__(self, results):
        self.results = results

    def mj_geomDistance(self, model, data, a, b, distmax, witness):
        distance, length = self.results[(a, b)]
        witness[:] = 0.
        witness[3] = length
        return distance


def probe_with(results):
    env = Env()
    sim = env.env.sim
    probe = physics.ContinuousContactProbe(sim.model, sim.data, physics.resolve_pairs(sim.model, PAIRS))
    pair = probe.pairs[0]
    probe.pairs = [pair, pair[::-1]]  # two pairs, each answered by the stand-in
    probe.mj = FakeDistance(dict(zip(probe.pairs, results)))
    return probe


@pytest.mark.parametrize("results, valid, gap", [
    ([(.004, .004), (.002, .002)], True, .002),        # both pairs measured
    ([(0., .03), (-.003, .003)], True, -.003),         # a CCD failure next to a penetrating pair
    ([(0., .03), (.006, .006)], False, 0.),            # the failed pair could be the closest
    ([(0., .03), (0., .02)], False, 0.),               # nothing measured
    ([(.02, .5), (.003, .003)], True, .003),           # at the cap the witness is not checked
])
def test_gap_is_judged_per_pair(results, valid, gap):
    probe = probe_with(results)
    assert probe.gap() == pytest.approx(gap)
    assert probe.gap_valid is valid


def test_recompute_restores_gaps_and_leaves_impulses_untouched(tmp_path, monkeypatch):
    source, prepared, output = tmp_path / "source.hdf5", tmp_path / "prepared.hdf5", tmp_path / "gapfix.hdf5"
    recorded_dataset(source)
    monkeypatch.setattr(physics, "make_replay_env", lambda group: Env())
    prepare_dataset(source, prepared, mode="physics", pairs=PAIRS, horizon=4, replay_mode="state", quiet=True)
    with h5py.File(prepared, "a") as f:
        demo = f["data/demo_0"]
        assert (demo["contact_state"][:, :, 0][demo["contact_mask"][:, :, 0] > 0] <= 0).any()
        mask = demo["contact_mask"][()]
        mask[2:5, :, 0] = 0  # as the earlier per-row rule could leave them
        demo["contact_mask"][...] = mask
        before = {k: demo[k][()] for k in ("contact_state", "contact_mask", "contact_label", "actions", "states")}
    monkeypatch.setattr(recompute_cami_gaps, "make_replay_env", lambda group: Env())
    source_bytes = prepared.read_bytes()
    result = recompute_cami_gaps.recompute_dataset(prepared, output, quiet=True)
    assert prepared.read_bytes() == source_bytes
    assert result["gap_rule"] == physics.ContinuousContactProbe.GAP_RULE
    assert result["after"] < result["before"]
    with h5py.File(output) as f:
        demo = f["data/demo_0"]
        state, mask = demo["contact_state"][()], demo["contact_mask"][()]
        np.testing.assert_array_equal(state[..., 1:], before["contact_state"][..., 1:])
        np.testing.assert_array_equal(mask[..., 1:], before["contact_mask"][..., 1:])
        for key in ("contact_label", "actions", "states"):
            np.testing.assert_array_equal(demo[key][()], before[key])
        assert mask[2:5, :, 0].any()
        kept = before["contact_mask"][..., 0] > 0
        np.testing.assert_allclose(state[..., 0][kept], before["contact_state"][..., 0][kept])
        report = json.loads(f["data"].attrs["cami_c"])
        assert report["gap_rule"] == physics.ContinuousContactProbe.GAP_RULE
        assert report["demos"]["demo_0"]["gap_rule"] == physics.ContinuousContactProbe.GAP_RULE
    with pytest.raises(ValueError, match="new, different path"):
        recompute_cami_gaps.recompute_dataset(prepared, output, quiet=True)

    def fail(*args, **kwargs):
        raise RuntimeError("simulator failed")

    monkeypatch.setattr(recompute_cami_gaps, "recompute_gaps", fail)
    with pytest.raises(RuntimeError, match="simulator failed"):
        recompute_cami_gaps.recompute_dataset(prepared, tmp_path / "failed.hdf5", quiet=True)
    assert not (tmp_path / "failed.hdf5").exists()
    assert not list(tmp_path.glob(".recompute_gaps_*"))
    assert prepared.read_bytes() == source_bytes


def test_recompute_refuses_gaps_measured_in_replayed_states(tmp_path, monkeypatch):
    source, prepared = tmp_path / "source.hdf5", tmp_path / "prepared.hdf5"
    recorded_dataset(source)
    monkeypatch.setattr(physics, "make_replay_env", lambda group: Env())
    prepare_dataset(source, prepared, mode="physics", pairs=PAIRS, horizon=4, quiet=True)  # action replay
    with pytest.raises(ValueError, match="recorded states"):
        recompute_cami_gaps.recompute_dataset(prepared, tmp_path / "gapfix.hdf5", quiet=True)
    assert not (tmp_path / "gapfix.hdf5").exists()
