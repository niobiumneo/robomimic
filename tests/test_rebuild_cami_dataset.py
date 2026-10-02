"""Test CaMI dataset replay using native MuJoCo and a small environment adapter.

Run from the repository root:
    python -m pytest -q tests/test_rebuild_cami_dataset.py

These tests need numpy, h5py, mujoco, and pytest. They exercise real dynamics
and sensors, but do not claim Square/ToolHang replay or rendering coverage.
"""
import json
import sys
from types import SimpleNamespace, ModuleType

import h5py
import mujoco
import numpy as np
import pytest

import rebuild_cami_dataset as code

XML = '''<mujoco><option timestep="0.002" integrator="Euler"/>
<worldbody><geom name="floor" type="plane" size="1 1 .1"/>
<body name="arm" pos="0 0 .049"><joint name="z" type="slide" axis="0 0 1"/>
<geom size=".01" mass="1" contype="0" conaffinity="0"/>
<body name="hand"><geom name="ball" type="sphere" size=".05" mass=".1"/>
<site name="wrist"/></body></body></worldbody>
<sensor><force name="force" site="wrist"/><torque name="torque" site="wrist"/></sensor>
<actuator><motor joint="z"/></actuator></mujoco>'''


class Sim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def step2(self):
        mujoco.mj_step2(self.model, self.data)


class Robot:
    part_controllers = {}

    def __init__(self, sim):
        self.sim = sim

    @property
    def ee_force(self):
        return {"right": self.sim.data.sensordata[:3]}

    @property
    def ee_torque(self):
        return {"right": self.sim.data.sensordata[3:6]}


class Env:
    action_dimension = 1

    def __init__(self):
        self.sim = Sim()
        self.env = SimpleNamespace(sim=self.sim, robots=[Robot(self.sim)], close=lambda: None)

    def reset(self):
        mujoco.mj_resetData(self.sim.model, self.sim.data)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        return self.get_observation()

    def reset_to(self, state):
        s = state['states']
        self.sim.data.time = s[0]
        self.sim.data.qpos[:] = s[1:2]
        self.sim.data.qvel[:] = s[2:3]
        mujoco.mj_forward(self.sim.model, self.sim.data)
        return self.get_observation()

    def get_state(self):
        return {"states": np.r_[self.sim.data.time, self.sim.data.qpos, self.sim.data.qvel],
                "model": XML, "ep_meta": "{}"}

    def get_observation(self):
        return {"robot0_eef_pos": self.sim.data.qpos.copy(),
                "agentview_image": np.full((8, 8, 3), 90, dtype=np.uint8)}

    def step(self, action):
        self.sim.data.ctrl[:] = action
        for _ in range(5):
            self.sim.step()
        return self.get_observation(), 0., False, {}

    def is_success(self):
        return {"task": False}

    def serialize(self):
        return {"env_name": "NutAssemblySquare", "env_kwargs": {}, "type": 1}


def fixture(path, divergent=False):
    env = Env()
    env.reset()
    actions = np.linspace(-.2, -.8, 24)[:, None]
    states, wrench = [], []
    for a in actions:
        states.append(env.get_state()["states"])
        wrench.append(code.wrist_wrench(env, "right"))
        env.step(a)
    states.append(env.get_state()["states"])
    final_wrench = code.wrist_wrench(env, "right").copy()
    if divergent:
        states[5][1] += .2
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(env.serialize())
        d = data.create_group("demo_0")
        d.attrs.update(num_samples=len(actions), model_file=XML)
        d.create_dataset("states", data=states)
        d.create_dataset("actions", data=actions)
        f.create_dataset("mask/train", data=np.asarray(["demo_0", "demo_9"], dtype="S"))
    return np.array(wrench), final_wrench


def args(src, dst, *extra):
    return code.parser().parse_args(["--dataset", str(src), "--output", str(dst),
                                    "--task", "square", "--include-next-obs", *extra])


def inject_env(monkeypatch):
    suite = ModuleType("robosuite")
    suite.__version__ = "synthetic-adapter"
    monkeypatch.setitem(sys.modules, "robosuite", suite)
    mod = ModuleType("robomimic.utils")
    mod.env_utils = SimpleNamespace(create_env_for_data_processing=lambda **kw: Env())
    monkeypatch.setitem(sys.modules, "robomimic", ModuleType("robomimic"))
    monkeypatch.setitem(sys.modules, "robomimic.utils", mod)


def test_future_alignment_and_end():
    w = np.zeros((5, 6)); w[2, 0] = 11.; w[1, 3] = 1e6
    labels, valid, counts = code.future_contact_labels(w, 2, 10.)
    assert labels[:, 0].tolist() == [1., 1., 0., 0., 0.]
    assert valid[:, 0].tolist() == [1, 1, 1, 1, 0]
    assert counts[:, 0].tolist() == [2, 2, 2, 1, 0]
    assert code.future_contact_labels(w[:1], 10, 10)[1].sum() == 0


def test_complete_native_replay(tmp_path, monkeypatch):
    inject_env(monkeypatch)
    src, dst = tmp_path / "raw.hdf5", tmp_path / "out.hdf5"
    raw, terminal = fixture(src)
    code.rebuild(args(src, dst))
    with h5py.File(dst) as f:
        d = f["data/demo_0"]
        np.testing.assert_allclose(d["contact/wrist_wrench_raw"], raw, atol=1e-10)
        np.testing.assert_allclose(d["obs/force"], raw - raw[0], atol=1e-6)
        np.testing.assert_allclose(d["next_obs/force"][:-1], d["obs/force"][1:], atol=1e-6)
        np.testing.assert_allclose(d["next_obs/force"][-1], terminal - raw[0], atol=1e-6)
        assert f.attrs["complete"]
        assert d["states"].shape == (24, 3)
        assert d["obs/agentview_image"].dtype == np.uint8
        assert d["contact_label_valid"][-1] == 0
        assert f["mask/train"].asstr()[()].tolist() == ["demo_0"]
        assert np.max(d["contact/replay_error"]) < 1e-10


def test_no_bias_mode(tmp_path):
    src = tmp_path / "raw.hdf5"
    raw, _ = fixture(src)
    with h5py.File(src) as f, h5py.File(tmp_path / "out.hdf5", "w") as out:
        g = out.create_group("d")
        code.process_demo(Env(), f["data/demo_0"], g, args(src, "unused", "--bias", "none"))
        np.testing.assert_allclose(g["obs/force"], raw, rtol=1e-6)


def test_drift_cannot_publish_partial_output(tmp_path, monkeypatch):
    inject_env(monkeypatch)
    src, dst = tmp_path / "raw.hdf5", tmp_path / "out.hdf5"
    fixture(src, divergent=True)
    before = src.read_bytes()
    with pytest.raises(RuntimeError, match="replay diverged"):
        code.rebuild(args(src, dst))
    assert not dst.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert src.read_bytes() == before


def test_refuse_overwrite(tmp_path, monkeypatch):
    inject_env(monkeypatch)
    src, dst = tmp_path / "raw.hdf5", tmp_path / "out.hdf5"
    fixture(src); dst.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        code.rebuild(args(src, dst))
    assert dst.read_bytes() == b"keep me"


def test_download_does_not_import_training_modules(tmp_path, monkeypatch):
    cached = tmp_path / "cached.hdf5"
    fixture(cached)
    registry = ModuleType("robomimic")
    registry.HF_REPO_ID = "test/repo"
    registry.DATASET_REGISTRY = {
        task: {"ph": {"raw": {"url": f"v1.5/{task}/ph/demo_v15.hdf5"}}}
        for task in ("square", "tool_hang")}
    hf = ModuleType("huggingface_hub")
    requests = []
    def fetch(**kwargs):
        requests.append(kwargs)
        return str(cached)
    hf.hf_hub_download = fetch
    monkeypatch.setitem(sys.modules, "robomimic", registry)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    code.download_demos(tmp_path / "download", ["square", "tool_hang"])
    code.download_demos(tmp_path / "download", ["square", "tool_hang"])
    assert len(requests) == 2
    for task in ("square", "tool_hang"):
        p = tmp_path / "download" / task / "ph/demo_v15.hdf5"
        assert p.read_bytes() == cached.read_bytes()
    assert code.TASKS["tool_hang"][2] == 240
