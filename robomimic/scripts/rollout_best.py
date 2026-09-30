"""Evaluate a saved policy and keep successful videos plus all rollout paths.

Example (run from the repository root):
    python -m robomimic.scripts.rollout_best \
        --run-dir trained_models/square_cami_continuous --n-rollouts 50

Selection uses the trainer's *_success_<rate>.pth filenames. Checkpoint
variable_state.best_success_rate is a historical maximum, not necessarily
the performance of the weights in that file, so it must not rank last.pth.
"""
import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


def select_checkpoint(run_dir, metric_key=None):
    """Find the best success-tagged checkpoint in one timestamped run.

    An experiment directory selects its latest timestamped run, never a
    mixture of runs. Multiple evaluation datasets require --metric-key.
    Equal success rates prefer the earlier epoch, without a return tie-break.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if run_dir.name == "models":
        run_dir = run_dir.parent
    if not (run_dir / "models").is_dir():
        runs = sorted(p for p in run_dir.glob("*")
                      if p.is_dir() and re.fullmatch(r"\d{14}", p.name)
                      and (p / "models").is_dir())
        if not runs:
            raise ValueError("No timestamped training run found in {}".format(run_dir))
        run_dir = runs[-1]

    # Dataset keys can contain underscores. Parse return and success tags in
    # order so a return tag never absorbs a preceding success tag.
    pattern = re.compile(
        r"_(?P<key>.+?)_(?P<kind>success|return)_(?P<rate>-?\d+(?:\.\d+)?)"
        r"(?=_|$)"
    )
    candidates = []
    for path in sorted((run_dir / "models").glob("model_epoch_*.pth")):
        epoch_match = re.match(r"model_epoch_(\d+)", path.stem)
        if epoch_match is None:
            continue
        suffix = path.stem[epoch_match.end():]
        suffix = re.sub(r"^_best_validation_-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", "", suffix)
        for match in pattern.finditer(suffix):
            if match["kind"] == "success":
                candidates.append((match["key"], float(match["rate"]),
                                   int(epoch_match[1]), path))
    keys = sorted({key for key, _, _, _ in candidates})
    if metric_key is None and len(keys) > 1:
        raise ValueError("Multiple rollout metrics found; use --metric-key: {}".format(keys))
    candidates = [c for c in candidates if metric_key is None or c[0] == metric_key]
    if not candidates:
        raise ValueError("No matching success-tagged checkpoint in {}. "
                         "Wait for evaluation, or pass --checkpoint explicitly.".format(run_dir))
    key, rate, epoch, path = max(candidates, key=lambda c: (c[1], -c[2]))
    return path, {"metric_key": key, "training_success_rate": rate, "epoch": epoch}


def collect_episode(policy, env, horizon, camera_names, video_skip):
    """Collect aligned pre-action states, post-action states, and video frames.

    Reset recurrent policy state for each independent trial. Success uses the
    environment's task criterion and stops the trial at its first success,
    matching the Square training template. No force injection is required.
    """
    policy.start_episode()
    obs = env.reset()
    initial_state = env.get_state()
    state = initial_state
    trajectory = {k: [] for k in ("states", "next_states", "actions", "rewards", "dones")}
    frames = []
    success = False
    error = None
    goal = env.get_goal() if getattr(policy.policy.global_config, "use_goals", False) else None
    try:
        for step in range(horizon):
            with torch.no_grad():
                action = policy(ob=obs, goal=goal)
            obs, reward, done, _ = env.step(action)
            next_state = env.get_state()
            for key, value in (("states", state["states"]),
                               ("next_states", next_state["states"]),
                               ("actions", action), ("rewards", reward), ("dones", done)):
                trajectory[key].append(np.array(value, copy=True))
            success = success or bool(env.is_success()["task"])
            # Always include the success / terminal frame even when skipped.
            if step % video_skip == 0 or success or done or step == horizon - 1:
                frames.append(np.concatenate([
                    env.render(mode="rgb_array", height=512, width=512, camera_name=camera)
                    for camera in camera_names
                ], axis=1))
            state = next_state
            if done or success:
                break
    except env.rollout_exceptions as exc:
        # Record simulator errors separately; they remain in the denominator.
        error = "{}: {}".format(type(exc).__name__, exc)
        success = False
    stats = {"Return": float(np.sum(trajectory["rewards"])),
             "Horizon": len(trajectory["actions"]), "Success_Rate": float(success),
             "error": error}
    return stats, {k: np.asarray(v) for k, v in trajectory.items()}, initial_state, frames


def save_trajectory(data, name, trajectory, initial_state, stats, seed):
    """Store both successful and unsuccessful paths in robomimic HDF5 format."""
    group = data.create_group(name)
    for key, values in trajectory.items():
        group.create_dataset(key, data=values)
    group.attrs["num_samples"] = stats["Horizon"]
    group.attrs["success"] = bool(stats["Success_Rate"])
    group.attrs["seed"] = seed
    group.attrs["return"] = stats["Return"]
    if stats["error"]:
        group.attrs["rollout_error"] = stats["error"]
    if "model" in initial_state:
        group.attrs["model_file"] = initial_state["model"]
    if "ep_meta" in initial_state:
        group.attrs["ep_meta"] = initial_state["ep_meta"]


def run(args):
    """Restore the policy once and run a fixed, reproducible trial budget."""
    selection = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    else:
        checkpoint, selection = select_checkpoint(args.run_dir, args.metric_key)
    device = TorchUtils.get_torch_device(try_to_use_cuda=not args.cpu)
    # Load on CPU first, avoiding optimizer tensors occupying the GPU.
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt)
    horizon = args.horizon or config.experiment.rollout.horizon
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Rollout horizon must be a positive integer")
    if isinstance(ckpt["env_metadata"], list):
        raise ValueError("This helper requires a single-environment checkpoint")
    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
              checkpoint.parent.parent / "successful_rollouts" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    # A fresh output directory prevents overwriting earlier evidence.
    output.mkdir(parents=True, exist_ok=False)
    print("Checkpoint: {}".format(checkpoint), flush=True)
    if selection:
        print("Training evaluation success: {:.1%} | epoch {}".format(
            selection["training_success_rate"], selection["epoch"]), flush=True)
    print("Device: {} | output: {}".format(device, output), flush=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt, device=device, verbose=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt, render=False,
                                          render_offscreen=True, verbose=False)
    records = []
    masks = {"successful": [], "failed": [], "errors": []}
    summary = {"checkpoint": str(checkpoint), "selection": selection,
               "n_rollouts_requested": args.n_rollouts, "horizon": horizon,
               "seed": args.seed, "camera_names": args.camera_names,
               "video_skip": args.video_skip, "fps": args.fps, "episodes": records}
    try:
        with h5py.File(output / "rollouts.hdf5", "x") as dataset:
            data = dataset.create_group("data")
            data.attrs["env_args"] = json.dumps(env.serialize())
            data.attrs["checkpoint"] = str(checkpoint)
            data.attrs["total"] = 0
            for index in range(args.n_rollouts):
                seed = args.seed + index
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                stats, trajectory, initial, frames = collect_episode(
                    policy, env, horizon, args.camera_names, args.video_skip)
                name = "demo_{}".format(index)
                save_trajectory(data, name, trajectory, initial, stats, seed)
                data.attrs.modify("total", int(data.attrs["total"]) + stats["Horizon"])
                status = "errors" if stats["error"] else (
                    "successful" if stats["Success_Rate"] else "failed")
                masks[status].append(name)
                video = None
                if frames and (stats["Success_Rate"] or args.keep_failures):
                    video = "{}_trial_{:03d}_seed_{}.mp4".format(status, index, seed)
                    # Encode after classification: unsuccessful frames need not
                    # create temporary videos that would then require deleting.
                    with imageio.get_writer(str(output / video), fps=args.fps) as writer:
                        for frame in frames:
                            writer.append_data(frame)
                records.append(dict(stats, trial=index, seed=seed, status=status,
                                    trajectory=name, video=video))
                dataset.flush()
                summary.update(n_rollouts_completed=len(records),
                               num_success=len(masks["successful"]),
                               num_errors=len(masks["errors"]),
                               success_rate=len(masks["successful"]) / len(records))
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print("Trial {}/{} | {} | steps={} | success so far={:.1%}".format(
                    index + 1, args.n_rollouts, status, stats["Horizon"],
                    summary["success_rate"]), flush=True)
                del frames
            for key, names in masks.items():
                dataset.create_dataset("mask/" + key, data=np.asarray(names, dtype="S"))
    finally:
        # EnvBase has no close method; robosuite's wrapped simulator does.
        base = getattr(env, "unwrapped", env)
        simulator = getattr(base, "env", base)
        close = getattr(simulator, "close", None)
        if callable(close):
            close()
    print("Success: {}/{} ({:.1%}); simulator errors: {}".format(
        summary["num_success"], len(records), summary["success_rate"],
        summary["num_errors"]), flush=True)
    print("Videos, all trajectories, and summary saved in {}".format(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Timestamped run, models folder, or experiment folder (latest run)")
    source.add_argument("--checkpoint", help="Explicit .pth file instead of automatic best selection")
    parser.add_argument("--metric-key", help="Dataset metric suffix when a run has multiple metrics")
    parser.add_argument("--output-dir", help="New output directory; must not exist")
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--horizon", type=int, help="Default: checkpoint rollout horizon")
    parser.add_argument("--seed", type=int, default=10000, help="Trial i uses seed+i")
    parser.add_argument("--camera-names", nargs="+", default=["agentview"])
    parser.add_argument("--video-skip", type=int, default=1)
    parser.add_argument("--fps", type=float, default=20, help="Use control_freq/video_skip for natural playback")
    parser.add_argument("--keep-failures", action="store_true", help="Also save videos of failures/errors")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.n_rollouts < 1 or args.video_skip < 1 or args.fps <= 0 or args.seed < 0:
        parser.error("n-rollouts, video-skip and fps must be positive; seed must be nonnegative")
    if args.horizon is not None and args.horizon < 1:
        parser.error("horizon must be positive")
    run(args)


if __name__ == "__main__":
    main()
