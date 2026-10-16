"""Label-statistics report for prepared physics datasets, checked against the real loader."""

import json
import math

import h5py
import numpy as np
import pytest
import torch

from robomimic.config import config_factory
from robomimic.scripts import inspect_cami_v3_dataset as inspect_labels
from robomimic.scripts.generate_cami_v3_configs import make_configs
from robomimic.scripts.prepare_cami_c_dataset import prepare_dataset
from robomimic.utils.dataset import SequenceDataset

NAMES = ["gap", "normal_impulse", "tangential_impulse_magnitude"]


@pytest.fixture
def prepared(tmp_path):
    raw, output = tmp_path / "raw.hdf5", tmp_path / "prepared.hdf5"
    rng = np.random.default_rng(3)
    lengths = [40, 33, 27, 36]
    with h5py.File(raw, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "NutAssemblySquare", "type": 1, "env_kwargs": {}})
        data.attrs["total"] = sum(lengths)
        for i, length in enumerate(lengths):
            d = data.create_group("demo_{}".format(i))
            d.attrs["num_samples"] = length
            d["actions"] = rng.uniform(-.5, .5, (length, 7)).astype(np.float32)
            d["obs/robot0_eef_pos"] = rng.normal(size=(length, 3)).astype(np.float32)
            t = np.arange(length)
            touching = (t >= 18 + i) & (t < 24 + i)
            gap = np.where(touching, -.001, np.clip(.02 - .002 * np.maximum(0, t - 8 - i), 0, .02))
            normal = np.where(touching, .4 + .2 * rng.random(length), 0.)
            tangential = np.where(touching, .05 * rng.random(length), 0.)
            d["contact/physics"] = np.stack([gap, normal, tangential], -1).astype(np.float32)
            d["contact/physics"].attrs["feature_names"] = json.dumps(NAMES)
            d["contact/dt"] = np.full(length, .05)
        f["mask/train"] = np.array([b"demo_0", b"demo_1", b"demo_2"])
        f["mask/valid"] = np.array([b"demo_3"])
    prepare_dataset(raw, output, mode="physics", contact_key="contact/physics", quiet=True)
    return output


@pytest.mark.parametrize("variant, pad", [("prediction", True), ("reference", True), ("baseline_reference", True),
                                         ("prediction", False), ("faithful", True)])
def test_anchor_pool_is_the_loaders_anchor_pool(prepared, variant, pad):
    config = inspect_labels.load_config(variant)
    with config.values_unlocked():
        config.train.pad_seq_length = pad
    pool = inspect_labels.load_pool(str(prepared), config)
    horizon, context = config.algo.cami.snippet_horizon, config.algo.cami.v3.context_steps
    dataset = SequenceDataset(
        hdf5_path=str(prepared), obs_keys=("robot0_eef_pos",), dataset_keys=("actions", "contact_state", "contact_mask"),
        action_keys=("actions",), action_config=dict(config.train.action_config), frame_stack=1,
        seq_length=config.train.seq_length, pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length, get_pad_mask=True, goal_mode=None, hdf5_cache_mode=None,
        hdf5_use_swmr=False, hdf5_normalize_obs=False, filter_by_attribute=config.train.hdf5_filter_key, load_next_obs=False)
    try:
        def key(length, state, mask):
            return int(length), np.where(mask, state, 0).astype(np.float32).round(5).tobytes(), np.asarray(mask).tobytes()
        expected = []
        for i in range(len(dataset)):
            item = dataset[i]
            length = int(item["pad_mask"][context + 1:context + horizon + 1, 0].sum())
            mask = item["contact_mask"][context].astype(bool) & (np.arange(horizon)[:, None] < length)
            expected.append(key(length, item["contact_state"][context], mask))
        actual = [key(l, w, m) for l, w, m in zip(pool["lengths"], pool["windows"], pool["mask"])]
        assert sorted(actual) == sorted(expected)
        # the faithful variant has no filter key and trains on every demonstration
        assert pool["names"] == (["demo_0", "demo_1", "demo_2", "demo_3"] if variant == "faithful"
                                 else ["demo_0", "demo_1", "demo_2"])
        assert len(pool["lengths"]) == len(dataset)
    finally:
        dataset.close_and_delete_hdf5_handle()


def trainer_scales(path, variant):
    """The scales prepare_contactnets_datasets fits for this dataset and variant."""
    from robomimic.utils.contactnets_dataset_utils import prepare_contactnets_datasets
    config = inspect_labels.load_config(variant)
    with config.values_unlocked():
        config.train.data = [{"path": str(path)}]
        config.observation.modalities.obs.low_dim = ["robot0_eef_pos"]
        config.observation.modalities.obs.rgb = []
    prepare_contactnets_datasets(config, require_rnn=False)
    c = config.algo.cami.continuous
    return [c.gap_scale, c.normal_impulse_scale, c.tangential_impulse_scale]


@pytest.mark.parametrize("variant", ["prediction", "faithful"])
def test_scales_are_the_trainers_training_fit(prepared, variant):
    config = inspect_labels.load_config(variant)
    pool = inspect_labels.load_pool(str(prepared), config)
    assert inspect_labels.fit_scales(pool, config) == pytest.approx(trainer_scales(prepared, variant), rel=1e-9)


def make_module(scales, config, horizon, batch):
    from robomimic.CAMI.cami_v3 import CaMIV3
    c, a = config.algo.cami.continuous, config.algo.cami
    settings = dict(
        snippet_horizon=horizon, temperature=a.temperature, normalize_embeddings=True, contrastive_dim=8,
        query_proj_layers=[8], key_proj_layers=[8], snippet_hidden_dim=8, snippet_num_layers=1, momentum=.9,
        use_momentum_target=True, feature_scales=list(scales), feature_weights=list(c.feature_weights),
        contact_temperature=c.contact_temperature, gamma=c.gamma, huber_delta=c.huber_delta, chunk_size=c.chunk_size,
        distance_penalty=c.distance_penalty, prediction_enabled=False, prediction_indices=[0],
        prediction_hidden_dims=[8], prediction_huber_delta=.1)
    return CaMIV3(8, 7, settings)


@pytest.mark.parametrize("variant", ["prediction", "faithful"])
def test_weight_regime_equals_the_training_module_on_the_same_batches(prepared, variant):
    config = inspect_labels.load_config(variant)
    pool = inspect_labels.load_pool(str(prepared), config)
    scales = inspect_labels.fit_scales(pool, config)
    horizon = config.algo.cami.snippet_horizon
    labels = inspect_labels.classify(pool, .02, 0.)
    module = make_module(scales, config, horizon, 16)
    rng = np.random.default_rng(5)
    for _ in range(4):
        idx = rng.choice(len(pool["lengths"]), 16, replace=False)
        _, stats = module(torch.zeros(16, 8), torch.zeros(16, horizon, 7), torch.as_tensor(pool["lengths"][idx]),
                          torch.as_tensor(pool["windows"][idx], dtype=torch.float32), torch.as_tensor(pool["mask"][idx]).float())
        ours = inspect_labels.weight_regime(pool, scales, config, labels, 16, 1, 0, index_batches=[idx])
        assert ours["near_zero_fraction"] == pytest.approx(stats["weight/near_zero_fraction"].item(), abs=1e-6)
        assert ours["near_one_fraction"] == pytest.approx(stats["weight/near_one_fraction"].item(), abs=1e-6)
        assert ours["effective_negatives"] == pytest.approx(stats["state/effective_negatives"].item(), rel=1e-4)
        assert ours["anchors_with_any_negative"] == pytest.approx(stats["state/valid_anchor_fraction"].item(), abs=1e-6)


def test_two_anchor_weight_matches_the_formula():
    config = inspect_labels.load_config("prediction")
    h = config.algo.cami.snippet_horizon
    windows = np.zeros((2, h, 3))
    windows[:, :, 0] = .02
    windows[1, :, 1] = 2.  # normal impulse two scale units above the free anchor
    pool = dict(windows=windows, mask=np.ones((2, h, 3), bool), lengths=np.full(2, h))
    labels = np.array([0, 2])
    out = inspect_labels.weight_regime(pool, [.02, 1., 1.], config, labels, 2, 1, 0)
    delta, tau = config.algo.cami.continuous.huber_delta, config.algo.cami.continuous.contact_temperature
    rho = delta * (2. - .5 * delta)
    expected = -math.expm1(-rho / tau)
    assert out["effective_negatives"] == pytest.approx(expected)
    assert out["uninformed_state_loss"] == pytest.approx(math.log1p(expected))
    assert out["anchors_with_any_negative"] == 1.
    assert out["pair_mass"][0, 2] == pytest.approx(expected) and out["pair_mass"][2, 0] == pytest.approx(expected)
    assert out["pair_mass"][0, 0] == 0.


def test_anchors_without_a_future_action_have_no_negatives():
    config = inspect_labels.load_config("prediction")
    h = config.algo.cami.snippet_horizon
    windows = np.zeros((3, h, 3))
    windows[1, :, 1] = 3.
    mask = np.ones((3, h, 3), bool)
    pool = dict(windows=windows, mask=mask, lengths=np.array([h, h, 0]))
    out = inspect_labels.weight_regime(pool, [.02, 1., 1.], config, np.array([0, 2, -1]), 3, 1, 0)
    assert out["anchors_with_any_negative"] == pytest.approx(2 / 3)


def test_onset_lag_reports_rows_between_gap_and_impulse():
    state = np.zeros((12, 10, 3))
    state[:, 0, 0] = .02
    state[5:8, 0, 0] = -.001   # gap closes at row 5
    state[5:8, 0, 1] = .3      # impulse starts in the same row
    state[10:, 0, 0] = -.001   # a second contact whose impulse began a row earlier
    state[9:, 0, 1] = .3
    pool = dict(series={"demo_0": (state, np.ones_like(state, bool))})
    lags = inspect_labels.onset_lags(pool)
    assert lags["0"] == 1 and lags["-1"] == 1 and sum(lags.values()) == 2


def test_onset_lag_does_not_split_a_contact_at_an_unobserved_gap_row():
    state, mask = np.zeros((12, 10, 3)), np.ones((12, 10, 3), bool)
    state[:, 0, 0] = .02
    state[5:10, 0, 0] = -.001
    state[5:10, 0, 1] = .3
    mask[7, 0, 0] = False   # one unobserved gap row inside the contact: still one contact
    lags = inspect_labels.onset_lags(dict(series={"demo_0": (state, mask)}))
    assert lags["0"] == 1 and sum(lags.values()) == 1
    mask[:] = True
    mask[5:7, 0, 0] = False  # gap unobserved at the closing: measured from the first observed row, two rows after the impulse
    lags = inspect_labels.onset_lags(dict(series={"demo_0": (state, mask)}))
    assert lags["<=-2"] == 1 and sum(lags.values()) == 1


def test_rows_with_an_impulse_but_no_gap_are_counted():
    state, mask = np.zeros((6, 10, 3)), np.ones((6, 10, 3), bool)
    state[1:4, 0, 1] = .2
    mask[2:4, 0, 0] = False
    assert inspect_labels.unobserved_gap_rows(dict(series={"d": (state, mask)})) == dict(rows=2, impulse_rows=3, fraction=2 / 3)


def test_ablation_rows_keep_the_configs_weights(prepared):
    report = inspect_labels.main(["--dataset", str(prepared), "--variant", "faithful", "--batch-size", "8", "--batches", "3"])
    config = inspect_labels.load_config("faithful")
    pool = inspect_labels.load_pool(str(prepared), config)
    scales, labels = inspect_labels.fit_scales(pool, config), inspect_labels.classify(pool, .02, report["binary_threshold"])
    third = 1. / 3
    for row, weights in (("gap channel only", [third, 0., 0.]), ("impulse channels only", [0., third, third])):
        expected = inspect_labels.weight_regime(pool, scales, config, labels, 8, 3, 0, overrides=dict(feature_weights=weights))
        assert report["regimes"][row]["effective_negatives"] == pytest.approx(expected["effective_negatives"])


def test_inputs_the_trainer_would_refuse_are_refused(prepared):
    config = inspect_labels.load_config("prediction")
    pool = inspect_labels.load_pool(str(prepared), config)
    with pytest.raises(ValueError, match="exceeds the"):
        inspect_labels.weight_regime(pool, [1., 1., 1.], config, np.zeros(len(pool["lengths"])), len(pool["lengths"]) + 1, 1, 0)
    short = inspect_labels.load_config("prediction")
    with short.values_unlocked():
        short.train.pad_seq_length, short.train.seq_length = False, 31
    with pytest.raises(ValueError, match="invalid num_samples or sequence length"):
        inspect_labels.load_pool(str(prepared), short)
    for entries in ([{"path": "a"}, {"path": "b"}], [{"path": "a", "filter_key": "valid"}], [{"path": "a", "demo_limit": 3}], []):
        many = inspect_labels.load_config("prediction")
        with many.values_unlocked():
            many.train.data = entries
        with pytest.raises(ValueError, match="single dataset entry"):
            inspect_labels.dataset_path(many)


def test_dataset_path_accepts_a_string_and_resolves_relative_paths(prepared, monkeypatch):
    config = inspect_labels.load_config("prediction")
    with config.values_unlocked():
        config.train.data = str(prepared)
    assert inspect_labels.dataset_path(config) == str(prepared)
    assert inspect_labels.dataset_path(config, str(prepared)) == str(prepared)


def test_report_runs_and_json_is_written_once(prepared, tmp_path, capsys):
    out = tmp_path / "report.json"
    report = inspect_labels.main(["--dataset", str(prepared), "--variant", "prediction", "--batch-size", "8",
                                  "--batches", "5", "--json", str(out)])
    text = capsys.readouterr().out
    assert "Weights W_ij" in text and "init loss" in text
    saved = json.loads(out.read_text())
    assert saved["batch_size"] == 8 and saved["train_demos"] == 3
    assert sum(saved["classes"].values()) == pytest.approx(1.)
    for regime in report["regimes"].values():
        assert 0 <= regime["near_zero_fraction"] <= 1 and 0 <= regime["near_one_fraction"] <= 1
    with pytest.raises(SystemExit):
        inspect_labels.main(["--dataset", str(prepared), "--json", str(out)])


def test_unprepared_dataset_is_refused(tmp_path):
    path = tmp_path / "raw.hdf5"
    with h5py.File(path, "w") as f:
        f["data/demo_0/actions"] = np.zeros((4, 7))
        f["data/demo_0"].attrs["num_samples"] = 4
    with pytest.raises(ValueError, match="not a prepared physics dataset"):
        inspect_labels.load_pool(str(path), inspect_labels.load_config("prediction"))


def test_variant_and_config_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        inspect_labels.load_config()
    with pytest.raises(ValueError, match="Unknown variant"):
        inspect_labels.load_config("nope")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(make_configs()["bc_cami_square_contactNetsv3.json"]))
    assert inspect_labels.load_config(path=str(path)).algo.cami.v3.context_steps == 5
