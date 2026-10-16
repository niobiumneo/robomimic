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
    assert "experiment.name" not in text and "train.output_dir" not in text
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


def test_full_unpickling_needs_explicit_trust(tmp_path):
    a = save(tmp_path / "a.pth", config())
    b = save(tmp_path / "b.pth", config(batch=32), obs_normalization_stats={"x": {"mean": np.zeros(2)}})
    with pytest.raises(SystemExit, match="--trust-checkpoints"):
        compare_configs.main([a, b])
    assert dict(compare_configs.main([a, b, "--trust-checkpoints"]))["train.batch_size"] == [16, 32]


def test_needs_two_checkpoints_with_configs(tmp_path):
    a = save(tmp_path / "a.pth", config())
    with pytest.raises(SystemExit):
        compare_configs.main([a])
    torch.save({"model": {}}, tmp_path / "bare.pth")
    with pytest.raises(SystemExit, match="no stored config"):
        compare_configs.main([a, str(tmp_path / "bare.pth")])
