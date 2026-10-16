"""Checkpoint config comparison: leaf diffs, CaMI-only grouping and unpickling safety."""

import json

import numpy as np
import pytest
import torch

from robomimic.scripts import compare_checkpoint_configs as compare_configs


def config(batch=16, seq=10, cami=False, lr_schedule=(), seed=1, name="a"):
    cfg = {"algo_name": "bc_cami_v3" if cami else "bc",
           "experiment": {"name": name, "epoch_every_n_steps": 500, "logging": {"log_wandb": cami}},
           "train": {"batch_size": batch, "seq_length": seq, "seed": seed, "output_dir": "/x/" + name,
                     "data": [{"path": "dataset/a.hdf5"}]},
           "algo": {"optim_params": {"policy": {"learning_rate": {"initial": 1e-4, "epoch_schedule": list(lr_schedule)}}}}}
    if cami:
        cfg["algo"]["cami"] = {"enabled": True, "state_loss_weight": .01, "v3": {"context_steps": 5}}
        cfg["algo"]["optim_params"]["cami"] = {"learning_rate": {"initial": 1e-4}}
    return cfg


def save(path, cfg, epoch=100, **extra):
    torch.save(dict(model={"w": torch.zeros(2)}, config=json.dumps(cfg), algo_name=cfg["algo_name"],
                    variable_state=dict(epoch=epoch, best_success_rate={"task": .5}), **extra), path)
    return str(path)


def test_lists_only_real_differences_and_hides_cami_only_settings(tmp_path, capsys):
    a = save(tmp_path / "a.pth", config(name="a"), epoch=541)
    b = save(tmp_path / "b.pth", config(batch=32, seq=16, cami=True, lr_schedule=[1000], seed=0, name="b"))
    rows = compare_configs.main([a, b])
    text = capsys.readouterr().out
    keys = {key for key, _ in rows}
    assert keys == {"train.batch_size", "train.seq_length", "train.seed", "algo_name", "algo.cami.enabled",
                    "algo.optim_params.policy.learning_rate.epoch_schedule"}
    assert "CaMI-only hidden" in text and "algo.cami.v3.context_steps" not in text
    assert "gradient_steps=270500" in text
    # bookkeeping settings are not rows; the output says they differ
    assert not keys & {"experiment.name", "train.output_dir", "experiment.logging.log_wandb"}
    assert "Not compared by default, but different:" in text and "experiment.name" in text.split("Not compared")[1]
    values = dict(rows)
    assert values["train.batch_size"] == [16, 32]
    assert values["algo.optim_params.policy.learning_rate.epoch_schedule"] == [[], [1000]]


def test_show_cami_and_extra_ignore(tmp_path):
    a = save(tmp_path / "a.pth", config())
    b = save(tmp_path / "b.pth", config(cami=True, batch=32))
    keys = {key for key, _ in compare_configs.main([a, b, "--show-cami", "--ignore", r"^train\.batch_size$"])}
    assert "algo.cami.state_loss_weight" in keys and "algo.optim_params.cami.learning_rate.initial" in keys
    assert "train.batch_size" not in keys


def test_identical_configs_report_no_differences(tmp_path, capsys):
    a = save(tmp_path / "a.pth", config(name="a"))
    b = save(tmp_path / "b.pth", config(name="b"))
    assert compare_configs.main([a, b]) == []
    assert "Differing settings: 0" in capsys.readouterr().out


def test_absent_setting_is_marked(tmp_path):
    cfg = config()
    other = config()
    other["train"]["frame_stack"] = 2
    a, b = save(tmp_path / "a.pth", cfg), save(tmp_path / "b.pth", other)
    assert dict(compare_configs.main([a, b]))["train.frame_stack"] == [compare_configs.ABSENT, 2]


class Payload:
    """An arbitrary object: loading it needs full unpickling."""


def test_trainer_style_numpy_arrays_load_without_trust(tmp_path):
    stats = {"x": {"mean": np.zeros(2), "std": np.float64(1.5), "n": np.int64(3), "f": np.ones((2, 2), np.float32)}}
    a = save(tmp_path / "a.pth", config())
    b = save(tmp_path / "b.pth", config(batch=32), obs_normalization_stats=stats)
    assert dict(compare_configs.main([a, b]))["train.batch_size"] == [16, 32]


def test_full_unpickling_needs_explicit_trust(tmp_path):
    a = save(tmp_path / "a.pth", config())
    b = save(tmp_path / "b.pth", config(batch=32), extra=Payload())
    with pytest.raises(SystemExit, match="--trust-checkpoints"):
        compare_configs.main([a, b])
    assert dict(compare_configs.main([a, b, "--trust-checkpoints"]))["train.batch_size"] == [16, 32]


def test_unreadable_inputs_give_one_clear_line(tmp_path):
    a = save(tmp_path / "a.pth", config())
    (tmp_path / "garbage.pth").write_bytes(b"not a checkpoint at all")
    with pytest.raises(SystemExit, match="not a readable checkpoint"):
        compare_configs.main([a, str(tmp_path / "garbage.pth"), "--trust-checkpoints"])
    with pytest.raises(SystemExit, match="not found"):
        compare_configs.main([a, str(tmp_path / "missing.pth")])
    torch.save(dict(config="{not json", variable_state={}), tmp_path / "badjson.pth")
    with pytest.raises(SystemExit, match="not JSON"):
        compare_configs.main([a, str(tmp_path / "badjson.pth")])
    torch.save(dict(config=None), tmp_path / "noconfig.pth")
    with pytest.raises(SystemExit, match="no stored config"):
        compare_configs.main([a, str(tmp_path / "noconfig.pth")])


def test_cami_settings_are_listed_between_runs_that_both_use_cami(tmp_path, capsys):
    one, two = config(cami=True, name="a"), config(cami=True, name="b")
    two["algo"]["cami"]["state_loss_weight"] = .02
    two["algo"]["cami"]["v3"]["context_steps"] = 0
    a, b = save(tmp_path / "a.pth", one), save(tmp_path / "b.pth", two)
    rows = dict(compare_configs.main([a, b]))
    assert rows["algo.cami.state_loss_weight"] == [.01, .02] and rows["algo.cami.v3.context_steps"] == [5, 0]
    assert "0 CaMI-only hidden" in capsys.readouterr().out
    # against a run with CaMI off the same settings are CaMI-only and are counted, not listed
    c = save(tmp_path / "c.pth", config(name="c"))
    assert not any(key.startswith("algo.cami.") and key != "algo.cami.enabled" for key, _ in compare_configs.main([a, c]))


def test_bookkeeping_differences_are_counted_not_silent(tmp_path, capsys):
    one, two = config(name="a"), config(name="b")
    two["train"]["cuda"] = True
    two["train"]["num_data_workers"] = 4
    a, b = save(tmp_path / "a.pth", one), save(tmp_path / "b.pth", two)
    assert compare_configs.main([a, b]) == []
    text = capsys.readouterr().out
    assert "Differing settings: 0" in text and "train.cuda" in text and "train.num_data_workers" in text
    assert "experiment.name" in text  # the names differ as well
    keys = {key for key, _ in compare_configs.main([a, b, "--compare-all"])}
    assert {"train.cuda", "train.num_data_workers", "experiment.name"} <= keys


def test_long_values_are_shortened_with_their_length(tmp_path, capsys):
    one, two = config(), config()
    two["train"]["demos"] = ["demo_{}".format(i) for i in range(500)]
    a, b = save(tmp_path / "a.pth", one), save(tmp_path / "b.pth", two)
    rows = dict(compare_configs.main([a, b]))
    assert len(rows["train.demos"][1]) == 500  # the returned rows keep the full value
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("      [1]")]
    assert lines and all(len(l) < 160 for l in lines) and "(500 items)" in lines[0]


def test_needs_two_checkpoints_with_configs(tmp_path):
    a = save(tmp_path / "a.pth", config())
    with pytest.raises(SystemExit):
        compare_configs.main([a])
    torch.save({"model": {}}, tmp_path / "bare.pth")
    with pytest.raises(SystemExit, match="no stored config"):
        compare_configs.main([a, str(tmp_path / "bare.pth")])
