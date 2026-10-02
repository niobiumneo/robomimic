#!/usr/bin/env python3
"""Rebuild a CaMI HDF5 from complete robomimic simulator demonstrations.

Target: https://github.com/niobiumneo/robomimic/tree/contact-state
Dataset interface checked at commit 5aeabc58e8912f40e8697ca3698e0572e7fb23ea.

Run in your existing CaMI / robomimic Python environment. Input must contain
data.attrs['env_args'], and actions, states, model_file for every demonstration.
This replays actions continuously and checks position AND velocity drift.
It never teleports to a recorded state between actions to hide a mismatch.

Examples, from the robomimic checkout with this file at the repository root:

  python rebuild_cami_dataset.py download --download-dir ./datasets

  python rebuild_cami_dataset.py --task square \
      --dataset datasets/square/ph/demo_v15.hdf5 \
      --output datasets/square/ph/cami_check.hdf5 --low-dim --n 1

  MUJOCO_GL=egl python rebuild_cami_dataset.py --task square \
      --dataset datasets/square/ph/demo_v15.hdf5 \
      --output datasets/square/ph/cami_with_force.hdf5

  MUJOCO_GL=egl python rebuild_cami_dataset.py --task tool_hang \
      --dataset datasets/tool_hang/ph/demo_v15.hdf5 \
      --output datasets/tool_hang/ph/cami_with_force.hdf5

Default cameras: square = agentview + robot0_eye_in_hand;
tool_hang = sideview + robot0_eye_in_hand. Square defaults to 84 x 84;
Tool Hang defaults to 240 x 240, compatible with the branch's 216-pixel crop.
Use a Square-specific training config with an image crop smaller than 84.

obs/force is a 6D wrist wrench [Fx,Fy,Fz,Tx,Ty,Tz]. Default --bias initial
subtracts the first frame's wrench once per episode. Use --bias none for raw
readings. Raw readings are always retained in contact/wrist_wrench_raw. This
is not gravity compensation, nor a direct measure of nut/peg contact alone.
The first frame is a forward-dynamics estimate after reset. Subsequent rows
come from the previous action's last solved physics substep, before action t.
Future contact labels use only translational force and indices t+1 ... t+H.
The final label is zero with contact_label_valid=0 because no future row exists.

This branch uses wrist-force supervision. No modules from earlier CaMI-C ZIP
overlays are needed. The program does not import robomimic.algo or bc_cami.py.
The download subcommand uses the branch's dataset registry and Hugging Face
directly, keeping dataset preparation independent of training-module imports.

Recorded states alone do not retain controller memory, solver warm-starts,
or original sensor readings. Replay checks improve consistency but do not
prove exact recovery of the original forces. Match the collection simulator,
controller, XML and action convention. Do not relax checks to conceal drift.
No force-bias EMA is inferred from the scene-wide contact count.

This program writes a NEW output file, streams images to disk, preserves and
filters split masks, and removes its partial output if any demonstration fails.
It does not change training code or apply the Huber loss during collection.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import h5py
import numpy as np


TASKS = {
    "square": ("NutAssemblySquare", ["agentview", "robot0_eye_in_hand"], 84),
    "tool_hang": ("ToolHang", ["sideview", "robot0_eye_in_hand"], 240),
}


def as_text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_array(group, key, value):
    return group.create_dataset(key, data=np.asarray(value), compression="gzip",
                                compression_opts=4)


def future_contact_labels(wrench, horizon, threshold):
    """A label is positive iff an available FUTURE force norm exceeds threshold.

    Partial windows near the end use only available rows. valid indicates that
    at least one future row exists; future_count records the available length.
    The anchor and padding never supply missing future measurements.
    """
    values = np.asarray(wrench)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError("Expected a finite [T,6] wrench array")
    if horizon < 1 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid horizon or force threshold")
    length = len(values)
    events = np.linalg.norm(values[:, :3], axis=1) > threshold
    prefix = np.r_[0, np.cumsum(events, dtype=np.int64)]
    start = np.arange(length) + 1
    end = np.minimum(start + horizon, length)
    labels = ((prefix[end] - prefix[start]) > 0).astype(np.float32)[:, None]
    counts = (end - start).astype(np.int32)[:, None]
    return labels, (counts > 0).astype(np.uint8), counts


def wrist_wrench(env, arm):
    """Read native sensor properties, not possibly stale observable caches."""
    if len(env.env.robots) != 1:
        raise ValueError("This script expects the single-robot Square/ToolHang tasks")
    robot = env.env.robots[0]
    parts = []
    for key in ("ee_force", "ee_torque"):
        value = getattr(robot, key)
        if isinstance(value, Mapping):
            if arm not in value:
                raise ValueError(f"Arm {arm!r} missing in {key}; available: {list(value)}")
            value = value[arm]
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape != (3,) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid {key}: expected three finite sensor values")
        parts.append(arr)
    return np.concatenate(parts)


def write_observation(group, obs, wrench, index, length):
    """Write one row; fixed-size, compressed datasets keep RAM use bounded."""
    values = {key: np.asarray(value) for key, value in obs.items()}
    values["force"] = np.asarray(wrench, dtype=np.float32)
    if index and set(values) != set(group):
        raise ValueError("Observation keys changed within an episode")
    for key, value in values.items():
        if value.dtype.kind not in "buif" or not np.isfinite(value).all():
            raise ValueError(f"Invalid observation {key}")
        if value.ndim == 3 and key.endswith("_image") and value.dtype != np.uint8:
            raise ValueError(f"{key} must be channel-last uint8")
        if key not in group:
            chunk_rows = 1 if value.ndim >= 3 else min(length, 128)
            group.create_dataset(key, shape=(length,) + value.shape, dtype=value.dtype,
                chunks=(chunk_rows,) + value.shape, compression="gzip", compression_opts=4)
        if group[key].shape[1:] != value.shape:
            raise ValueError(f"Observation shape changed: {key}")
        group[key][index] = value


def native_state(env):
    sim = env.env.sim
    return getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)


def replay_error(model, data, expected):
    """Compare qpos in tangent space so equivalent quaternion signs agree."""
    import mujoco
    expected = np.asarray(expected, dtype=np.float64)
    # Robosuite's flattened MjSimState stores time, qpos, qvel, then any act.
    if expected.ndim != 1 or expected.size < 1 + model.nq + model.nv:
        raise ValueError("Stored state width does not match the simulator model")
    pos = expected[1:1 + model.nq]
    vel = expected[1 + model.nq:1 + model.nq + model.nv]
    delta = np.empty(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, delta, 1.0, pos, data.qpos)
    return np.array([np.max(np.abs(delta)), np.max(np.abs(vel - data.qvel)),
                     abs(float(expected[0]) - float(data.time))])


def reset_to_demo(env, demo):
    state = {"states": demo["states"][0], "model": as_text(demo.attrs["model_file"])}
    if "ep_meta" in demo.attrs:
        state["ep_meta"] = as_text(demo.attrs["ep_meta"])
    env.reset()
    env.reset_to(state)
    # XML restoration initializes controllers before stored qpos is restored.
    # Align references once at episode start, never at every action.
    for robot in env.env.robots:
        controllers = getattr(robot, "part_controllers", None)
        if controllers is None:
            controllers = {"arm": robot.controller}
        for controller in controllers.values():
            controller.update(force=True)
            if hasattr(controller, "update_initial_joints"):
                controller.update_initial_joints(controller.joint_pos)
            controller.reset_goal()
    return env.get_observation()


def copy_masks(source, output, demos):
    if "mask" not in source:
        return
    masks = output.create_group("mask")
    for key, dataset in source["mask"].items():
        retained = [as_text(x) for x in dataset[()] if as_text(x) in demos]
        result = masks.create_dataset(key, data=np.asarray(retained, dtype="S"))
        result.attrs.update(dict(dataset.attrs))


def process_demo(env, source, target, args):
    for key in ("states", "actions"):
        if key not in source:
            raise ValueError(f"{source.name}: missing {key}")
    if "model_file" not in source.attrs:
        raise ValueError(f"{source.name}: missing model_file XML")
    actions = source["actions"][()]
    states = source["states"][()]
    length = len(actions)
    if (length < 1 or actions.ndim != 2 or not np.isfinite(actions).all()
            or states.ndim != 2 or len(states) not in (length, length + 1)
            or not np.isfinite(states).all()):
        raise ValueError(f"{source.name}: invalid actions or states")
    obs = reset_to_demo(env, source)
    model, data = native_state(env)
    if actions.shape[1] != env.action_dimension:
        raise ValueError("Stored action size and controller action dimension differ")
    runtime_state = env.get_state()
    if states.shape[1] != np.asarray(runtime_state["states"]).size:
        raise ValueError("Stored state size and runtime state size differ")

    # Keep unrelated per-episode metadata, but never copy stale CaMI labels.
    target.attrs.update({k: v for k, v in source.attrs.items() if not k.startswith("cami")})
    target.attrs["num_samples"] = length
    target.attrs["model_file"] = runtime_state["model"]
    if "ep_meta" in runtime_state:
        target.attrs["ep_meta"] = runtime_state["ep_meta"]
    write_array(target, "actions", actions)
    write_array(target, "reference_states", states)
    recorded_states = target.create_dataset("states", shape=(length, states.shape[1]),
                                             dtype=np.float64, compression="gzip")
    obs_group = target.create_group("obs")
    next_group = target.create_group("next_obs") if args.include_next_obs else None
    raw_wrenches, corrected_wrenches = [], []
    rewards, dones, errors, post_errors = [], [], [], []
    raw = wrist_wrench(env, args.arm)
    bias = raw.copy() if args.bias == "initial" else np.zeros(6)

    def check(expected, timestep):
        error = replay_error(model, data, expected)
        limits = np.array([args.qpos_tolerance, args.qvel_tolerance, 1e-5])
        if not np.isfinite(error).all() or np.any(error > limits):
            raise RuntimeError(
                f"{source.name}, state {timestep}: replay diverged. "
                f"Max position/velocity/time errors: {error.tolist()}, limits: {limits.tolist()}. "
                "Match the source robosuite/MuJoCo versions and controller settings. "
                "No completed output will be written. Do not hide this with per-step resets.")
        return error

    success = False
    for t, action in enumerate(actions):
        errors.append(check(states[t], t))
        corrected = raw - bias
        write_observation(obs_group, obs, corrected, t, length)
        raw_wrenches.append(raw.copy())
        corrected_wrenches.append(corrected.copy())
        recorded_states[t] = env.get_state()["states"]
        outcome = env.step(action)
        next_obs, reward, _, _ = outcome
        next_raw = wrist_wrench(env, args.arm)
        if t + 1 < len(states):
            post_errors.append(check(states[t + 1], t + 1))
        current_success = bool(env.is_success().get("task", False))
        success |= current_success
        rewards.append(float(reward))
        dones.append(int(current_success or t == length - 1))
        if next_group is not None:
            write_observation(next_group, next_obs, next_raw - bias, t, length)
        obs, raw = next_obs, next_raw

    corrected_wrenches = np.asarray(corrected_wrenches, dtype=np.float32)
    labels, valid, counts = future_contact_labels(
        corrected_wrenches, args.horizon, args.contact_threshold)
    for key, value in (("rewards", rewards), ("dones", dones), ("contact_label", labels),
                       ("contact_label_valid", valid), ("contact_future_count", counts)):
        write_array(target, key, value)
    c = target.create_group("contact")
    write_array(c, "wrist_wrench_raw", raw_wrenches)
    write_array(c, "wrist_bias", bias)
    write_array(c, "replay_error", errors)
    write_array(c, "post_action_replay_error", np.asarray(post_errors).reshape(-1, 3))
    # The final post-action force is real simulated data, not a repeated last row.
    write_array(c, "terminal_wrist_wrench_raw", raw)
    write_array(c, "terminal_state", env.get_state()["states"])
    c.attrs["wrist_units"] = json.dumps(["N"] * 3 + ["N*m"] * 3)
    c.attrs["wrist_frame"] = "native end-effector sensor frame"
    c.attrs["first_wrench_is_reset_estimate"] = True
    c.attrs["force_bias_mode"] = args.bias
    c.attrs["replay_error_columns"] = json.dumps(["max_tangent_qpos", "max_qvel", "abs_time"])
    target.attrs["success"] = int(success)
    target.attrs["contact_label_horizon"] = args.horizon
    target.attrs["contact_label_threshold"] = args.contact_threshold
    target.attrs["contact_label_desc"] = "any available force norm > threshold in t+1..t+H"
    stats = dict(samples=length, success=success,
                 max_replay_error=np.max(np.vstack(errors + post_errors), axis=0).tolist(),
                 peak_force_N=float(np.linalg.norm(corrected_wrenches[:, :3], axis=1).max()),
                 positive_future_labels=int(labels.sum()),
                 checked_final_transition=(len(states) == length + 1))
    target.attrs["replay_report"] = json.dumps(stats)
    return stats


def rebuild(args):
    import mujoco
    import robosuite
    from robomimic.utils import env_utils as EnvUtils

    src = Path(args.dataset).expanduser().resolve()
    dst = Path(args.output).expanduser().resolve()
    if not src.is_file() or not h5py.is_hdf5(src):
        raise ValueError(f"Input is not an HDF5 file: {src}")
    if src == dst or dst.exists():
        raise FileExistsError(f"Output must be a new path: {dst}")
    if args.n is not None and args.n < 1:
        raise ValueError("--n must be positive")
    if args.image_size is None:
        args.image_size = TASKS[args.task][2]
    if args.horizon < 1 or args.image_size < 1:
        raise ValueError("Horizon and image size must be positive")
    for key in ("qpos_tolerance", "qvel_tolerance"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not np.isfinite(args.contact_threshold) or args.contact_threshold < 0:
        raise ValueError("Invalid contact threshold")
    camera_names = [] if args.low_dim else (args.camera_names or TASKS[args.task][1])
    env = None
    temporary = None
    try:
        with h5py.File(src, "r") as source:
            if "data" not in source or "env_args" not in source["data"].attrs:
                raise ValueError("Need a robomimic HDF5 with data.attrs['env_args']")
            metadata = json.loads(as_text(source["data"].attrs["env_args"]))
            if metadata.get("env_name") != TASKS[args.task][0]:
                raise ValueError(f"Task mismatch: {metadata.get('env_name')} vs {TASKS[args.task][0]}")
            print(f"Source robosuite: {metadata.get('env_version', 'unspecified')}; "
                  f"installed: {robosuite.__version__}; MuJoCo: {mujoco.__version__}", flush=True)
            print("Replaying stored actions; force readings are generated in this simulator.", flush=True)
            env = EnvUtils.create_env_for_data_processing(
                env_meta=deepcopy(metadata), camera_names=camera_names,
                camera_height=args.image_size, camera_width=args.image_size,
                reward_shaping=False, use_depth_obs=False)
            demos = sorted(source["data"], key=lambda name: int(name.rsplit("_", 1)[1]))
            if args.n is not None:
                demos = demos[:args.n]
            if not demos:
                raise ValueError("No demonstrations")
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=dst.name + ".", suffix=".partial", dir=dst.parent)
            os.close(fd)
            summary = {}
            with h5py.File(temporary, "w") as output:
                output.attrs["source_dataset"] = str(src)
                output.attrs["source_env_args"] = json.dumps(metadata, default=json_default)
                output.attrs["rebuild_settings"] = json.dumps(dict(vars(args), cameras_used=camera_names,
                    robosuite=robosuite.__version__, mujoco=mujoco.__version__), default=json_default)
                output.attrs["force_provenance"] = "continuous_action_replay; original force recovery not guaranteed"
                output.attrs["complete"] = False
                group = output.create_group("data")
                for name in demos:
                    summary[name] = process_demo(env, source["data"][name],
                                                 group.create_group(name), args)
                    print(name + ": " + json.dumps(summary[name]), flush=True)
                    output.flush()
                group.attrs["env_args"] = json.dumps(env.serialize(), default=json_default)
                group.attrs["total"] = sum(x["samples"] for x in summary.values())
                copy_masks(source, output, set(demos))
                output.attrs["rebuild_report"] = json.dumps(summary)
                output.attrs["complete"] = True
            # Atomic publication without overwriting an existing output.
            os.link(temporary, dst)
            os.unlink(temporary)
            temporary = None
        print(f"Saved {len(summary)} demonstrations to {dst}", flush=True)
    finally:
        if env is not None:
            close = getattr(env, "close", None) or getattr(env.env, "close", None)
            if close:
                close()
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--n", type=int, help="Process only the first N demos")
    p.add_argument("--low-dim", action="store_true", help="No rendered camera images")
    p.add_argument("--camera-names", nargs="+", help="Override task-specific camera defaults")
    p.add_argument("--image-size", type=int, help="Defaults: Square 84; Tool Hang 240")
    p.add_argument("--include-next-obs", action="store_true", help="Also save next_obs; uses more disk space")
    p.add_argument("--arm", default="right")
    p.add_argument("--bias", choices=("initial", "none"), default="initial")
    p.add_argument("--horizon", type=int, default=10, help="Match CaMI snippet_horizon")
    p.add_argument("--contact-threshold", type=float, default=10., help="Force magnitude threshold in N")
    p.add_argument("--qpos-tolerance", type=float, default=1e-3)
    p.add_argument("--qvel-tolerance", type=float, default=1e-2)
    return p


def download_demos(download_dir, tasks):
    """Fetch public PH raw demos without importing this branch's algorithms."""
    from huggingface_hub import hf_hub_download
    from robomimic import DATASET_REGISTRY, HF_REPO_ID

    base = Path(download_dir).expanduser().resolve()
    for task in tasks:
        filename = DATASET_REGISTRY[task]["ph"]["raw"]["url"]
        if not filename or "://" in filename or not filename.endswith(".hdf5"):
            raise ValueError("Expected a raw HDF5 path in the branch's Hugging Face registry")
        target = base / task / "ph" / Path(filename).name
        if target.exists():
            if not h5py.is_hdf5(target):
                raise ValueError(f"Existing file is not HDF5: {target}")
            print(f"Using existing {target}", flush=True)
            continue
        print(f"Downloading {task} PH raw demonstrations...", flush=True)
        cached = hf_hub_download(repo_id=HF_REPO_ID, filename=filename, repo_type="dataset")
        if not h5py.is_hdf5(cached):
            raise ValueError(f"Downloaded file is not HDF5: {cached}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, partial = tempfile.mkstemp(prefix=target.name + ".", suffix=".partial", dir=target.parent)
        os.close(fd)
        try:
            shutil.copyfile(cached, partial)
            os.link(partial, target)
        finally:
            os.unlink(partial)
        print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["download"]:
        p = argparse.ArgumentParser(description="Download public proficient-human raw demos")
        p.add_argument("--download-dir", default="./datasets")
        p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
        a = p.parse_args(sys.argv[2:])
        download_demos(a.download_dir, a.tasks)
    else:
        rebuild(parser().parse_args())
