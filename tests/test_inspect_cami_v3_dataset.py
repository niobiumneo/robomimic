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


@pytest.mark.parametrize("variant", ["prediction", "reference", "baseline_reference"])
def test_anchor_pool_is_the_loaders_anchor_pool(prepared, variant):
    config = inspect_labels.load_config(variant)
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
        assert pool["names"] == ["demo_0", "demo_1", "demo_2"]
    finally:
        dataset.close_and_delete_hdf5_handle()


def test_scales_are_the_trainers_training_fit(prepared):
    config = inspect_labels.load_config("prediction")
    pool = inspect_labels.load_pool(str(prepared), config)
    scales = inspect_labels.fit_scales(pool, config)
    with h5py.File(prepared) as f:
        rows = [(f["data/" + n + "/contact_state"][:, 0], f["data/" + n + "/contact_mask"][:, 0].astype(bool))
                for n in ("demo_0", "demo_1", "demo_2")]
    for i in range(3):
        values = np.concatenate([s[:, i][m[:, i]] for s, m in rows])
        assert scales[i] == pytest.approx(values.std() + 1e-6)


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
