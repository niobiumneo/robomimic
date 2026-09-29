"""Check W&B setup without authentication or publishing test runs."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import robomimic.macros as macros
from robomimic.config import config_factory
from robomimic.utils.log_utils import DataLogger


TEMPLATE = Path(__file__).resolve().parents[1] / "robomimic/exps/templates/bc_cami_square.json"


def square_config():
    raw = json.loads(TEMPLATE.read_text())
    config = config_factory(raw["algo_name"])
    with config.values_unlocked():
        config.update(raw)
        config.algo.cami.continuous_contact.force_scale = 26.8
        # Older saved configs, including the reported run, use null here.
        config.meta.hp_keys = None
        config.meta.hp_values = None
    config.lock()
    return config


@pytest.fixture
def fake_wandb(monkeypatch):
    calls = []
    history = []
    run = SimpleNamespace(
        offline=False, url="https://wandb.ai/test/cami/runs/example", finished=False,
    )
    run.log = lambda values, step, **kwargs: history.append((values, step, kwargs))
    run.finish = lambda: setattr(run, "finished", True)

    def init(**kwargs):
        calls.append(kwargs)
        return run

    module = SimpleNamespace(init=init)
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setattr(macros, "WANDB_ENTITY", None)
    monkeypatch.setattr(macros, "WANDB_API_KEY", None)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    return SimpleNamespace(module=module, run=run, calls=calls, history=history)


def test_null_sweep_metadata_and_effective_config(tmp_path, fake_wandb):
    config = square_config()
    assert config.meta.hp_keys is None
    logger = DataLogger(str(tmp_path), config, log_tb=False, log_wandb=True)
    init = fake_wandb.calls[0]
    assert init["entity"] is None  # W&B may use its configured default account.
    assert "mode" not in init  # Honor WANDB_MODE instead of forcing online.
    assert init["config"]["sweep_parameters"] == {}
    assert init["config"]["train"]["hdf5_filter_key"] == "train"
    assert init["config"]["algo"]["cami"]["continuous_contact"]["force_scale"] == 26.8
    logger.record("Train/BC_Action_Loss", 0.2, epoch=1)
    logger.record("Train/State_CaMI_Loss", 0.7, epoch=1)
    assert fake_wandb.history == [
        ({"Train/BC_Action_Loss": 0.2}, 1, {}),
        ({"Train/State_CaMI_Loss": 0.7}, 1, {}),
    ]
    logger.flush(epoch=1)
    assert fake_wandb.history[-1] == ({}, 1, {"commit": True})
    logger.close()
    assert fake_wandb.run.finished


@pytest.mark.parametrize("entity, expected", [(None, "legacy-team"), ("lab-team", "lab-team")])
def test_entity_environment_overrides_macro(tmp_path, monkeypatch, fake_wandb, entity, expected):
    monkeypatch.setattr(macros, "WANDB_ENTITY", "legacy-team")
    if entity is not None:
        monkeypatch.setenv("WANDB_ENTITY", entity)
    logger = DataLogger(str(tmp_path), square_config(), log_tb=False, log_wandb=True)
    assert fake_wandb.calls[0]["entity"] == expected
    logger.close()


def test_auth_error_is_reported_without_offline_fallback(tmp_path, fake_wandb):
    attempts = []

    def denied(**kwargs):
        attempts.append(kwargs)
        raise PermissionError("team access denied")

    fake_wandb.module.init = denied
    with pytest.raises(RuntimeError, match="wandb login") as error:
        DataLogger(str(tmp_path), square_config(), log_tb=False, log_wandb=True)
    assert isinstance(error.value.__cause__, PermissionError)
    assert len(attempts) == 1


def test_offline_run_is_identified(tmp_path, fake_wandb, capsys):
    fake_wandb.run.offline = True
    fake_wandb.run.url = None
    logger = DataLogger(str(tmp_path), square_config(), log_tb=False, log_wandb=True)
    assert "W&B is offline" in capsys.readouterr().out
    logger.close()


@pytest.mark.parametrize("enabled, project, expected", [
    (False, None, False), (True, None, True), (False, "cami-tests", True),
])
def test_training_cli_wandb_overrides(monkeypatch, enabled, project, expected):
    from robomimic.scripts import train as training

    captured = []
    monkeypatch.setattr(training.TorchUtils, "get_torch_device", lambda **kwargs: "cpu")
    monkeypatch.setattr(training, "train", lambda config, **kwargs: captured.append(config))
    args = SimpleNamespace(
        config=str(TEMPLATE), dataset=None, name="wandb-cli-test", debug=False,
        resume=False, wandb=enabled, wandb_project=project,
    )
    training.main(args)
    assert captured[0].experiment.logging.log_wandb is expected
    assert captured[0].experiment.logging.wandb_proj_name == (project or "cami")


def test_real_sdk_offline_metrics(tmp_path, monkeypatch):
    """Exercise the real SDK while ensuring it cannot create an online run."""
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_ENTITY", "robomimic-test")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setattr(macros, "WANDB_API_KEY", None)
    logger = DataLogger(str(tmp_path), square_config(), log_tb=False, log_wandb=True)
    try:
        assert logger._wandb_logger.offline
        logger.record("Train/BC_Action_Loss", 0.2, epoch=1)
        logger.record("Train/State_CaMI_Loss", 0.7, epoch=1)
        logger.flush(epoch=1)
        assert logger._wandb_logger.summary["Train/BC_Action_Loss"] == pytest.approx(0.2)
        assert logger._wandb_logger.summary["Train/State_CaMI_Loss"] == pytest.approx(0.7)
        logger.record("Train/BC_Action_Loss", 0.1, epoch=2)
        logger.flush(epoch=2)
        assert logger._wandb_logger.summary["Train/BC_Action_Loss"] == pytest.approx(0.1)
        assert logger._wandb_logger.config["algo"]["cami"]["continuous_contact"]["force_scale"] == 26.8
    finally:
        logger.close()
    assert list(tmp_path.glob("wandb/offline-run-*/run-*.wandb"))
