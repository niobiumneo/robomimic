"""
Extended rollout evaluation script for robomimic.

New features:
- evaluate multiple models in one run
- run multiple trials per model
- compute per-trial success-rate distributions
- save trial-level and rollout-level CSV files for box plots / repeatability analysis

Example:
python run_trained_agent_multi_eval.py \
    --agents /path/model1.pth /path/model2.pth \
    --agent_names BC CaMI \
    --n_trials 10 \
    --rollouts_per_trial 50 \
    --horizon 400 \
    --seed 0 \
    --results_dir /path/to/results
"""

import argparse
import json
import os
import csv
import h5py
import imageio
import numpy as np
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


def safe_makedirs(path):
    if path is not None:
        os.makedirs(path, exist_ok=True)


def seed_everything(seed, env=None):
    if seed is None:
        return

    np.random.seed(seed)
    torch.manual_seed(seed)

    # best-effort environment seeding
    try:
        if hasattr(env, "seed"):
            env.seed(seed)
    except Exception:
        pass

    try:
        if hasattr(env, "base_env") and hasattr(env.base_env, "seed"):
            env.base_env.seed(seed)
    except Exception:
        pass


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5,
            return_obs=False, camera_names=None):
    """
    Run one rollout episode.
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # robosuite determinism hack
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0
    total_reward = 0.0
    success = False

    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))

    try:
        for step_i in range(horizon):
            act = policy(ob=obs)
            next_obs, r, done, _ = env.step(act)

            total_reward += r
            success = env.is_success()["task"]

            if render:
                env.render(mode="human", camera_name=camera_names[0])

            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        frame = env.render(
                            mode="rgb_array",
                            height=512,
                            width=512,
                            camera_name=cam_name,
                        )
                        video_img.append(frame)
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                video_count += 1

            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])

            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            if done or success:
                break

            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print(f"WARNING: got rollout exception {e}")

    stats = dict(
        Return=float(total_reward),
        Horizon=int(step_i + 1),
        Success_Rate=float(success),
    )

    if return_obs:
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def create_env_and_policy(ckpt_path, device, args, env_name=None):
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path,
        device=device,
        verbose=True,
    )

    rollout_horizon = args.horizon
    if rollout_horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=env_name,
        render=args.render,
        render_offscreen=(args.video_path is not None),
        verbose=True,
    )

    return policy, env, rollout_horizon


def write_trial_dataset(dataset_path, trajectories, env):
    with h5py.File(dataset_path, "w") as data_writer:
        data_grp = data_writer.create_group("data")
        total_samples = 0

        for i, traj in enumerate(trajectories):
            ep_data_grp = data_grp.create_group(f"demo_{i}")
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))

            if "obs" in traj:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset(f"obs/{k}", data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset(f"next_obs/{k}", data=np.array(traj["next_obs"][k]))

            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]

            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]

        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)


def evaluate_one_model(ckpt_path, model_name, args, device):
    env_names = args.envs if args.envs is not None else [None]

    all_env_trial_rows = []
    all_env_rollout_rows = []

    for env_name in env_names:
        policy, env, rollout_horizon = create_env_and_policy(
            ckpt_path, device, args, env_name=env_name
        )

        env_tag = env_name if env_name is not None else "checkpoint_env"
        model_env_dir = os.path.join(args.results_dir, model_name, env_tag) if args.results_dir else None
        safe_makedirs(model_env_dir)

        per_rollout_rows = []
        per_trial_rows = []

        for trial_idx in range(args.n_trials):
            trial_seed = None if args.seed is None else args.seed + trial_idx
            seed_everything(trial_seed, env=env)

            print(f"\n=== Model: {model_name} | Env: {env_tag} | Trial {trial_idx + 1}/{args.n_trials} | Seed: {trial_seed} ===")

            trajectories = []
            trial_rollout_stats = []

            for rollout_idx in range(args.rollouts_per_trial):
                stats, traj = rollout(
                    policy=policy,
                    env=env,
                    horizon=rollout_horizon,
                    render=args.render,
                    video_writer=None,
                    video_skip=args.video_skip,
                    return_obs=(args.dataset_obs and args.save_datasets),
                    camera_names=args.camera_names,
                )

                trial_rollout_stats.append(stats)
                trajectories.append(traj)

                per_rollout_rows.append({
                    "model": model_name,
                    "env": env_tag,
                    "checkpoint": ckpt_path,
                    "trial": trial_idx,
                    "trial_seed": trial_seed,
                    "rollout": rollout_idx,
                    "return": stats["Return"],
                    "horizon": stats["Horizon"],
                    "success": stats["Success_Rate"],
                })

            stats_dict = TensorUtils.list_of_flat_dict_to_dict_of_list(trial_rollout_stats)
            trial_summary = {
                "model": model_name,
                "env": env_tag,
                "checkpoint": ckpt_path,
                "trial": trial_idx,
                "trial_seed": trial_seed,
                "num_rollouts": args.rollouts_per_trial,
                "success_rate_mean": float(np.mean(stats_dict["Success_Rate"])),
                "num_success": int(np.sum(stats_dict["Success_Rate"])),
                "return_mean": float(np.mean(stats_dict["Return"])),
                "return_std": float(np.std(stats_dict["Return"])),
                "horizon_mean": float(np.mean(stats_dict["Horizon"])),
                "horizon_std": float(np.std(stats_dict["Horizon"])),
            }
            per_trial_rows.append(trial_summary)

        if model_env_dir is not None:
            write_csv(os.path.join(model_env_dir, "trial_results.csv"), per_trial_rows)
            write_csv(os.path.join(model_env_dir, "rollout_results.csv"), per_rollout_rows)

        all_env_trial_rows.extend(per_trial_rows)
        all_env_rollout_rows.extend(per_rollout_rows)

    return all_env_trial_rows, all_env_rollout_rows


def write_csv(path, rows):
    if len(rows) == 0:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_multi_eval(args):
    assert not (args.render and args.video_path is not None), \
        "Choose either on-screen rendering or video output, not both."

    if args.render:
        assert len(args.camera_names) == 1, \
            "On-screen rendering supports only one camera."

    safe_makedirs(args.results_dir)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    agent_paths = args.agents
    if args.agent_names is None:
        model_names = [os.path.splitext(os.path.basename(p))[0] for p in agent_paths]
    else:
        assert len(args.agent_names) == len(agent_paths), \
            "--agent_names must match number of --agents"
        model_names = args.agent_names

    all_trial_rows = []
    all_rollout_rows = []

    for ckpt_path, model_name in zip(agent_paths, model_names):
        trial_rows, rollout_rows = evaluate_one_model(
            ckpt_path=ckpt_path,
            model_name=model_name,
            args=args,
            device=device,
        )
        all_trial_rows.extend(trial_rows)
        all_rollout_rows.extend(rollout_rows)

    if args.results_dir is not None:
        trial_csv = os.path.join(args.results_dir, "trial_results.csv")
        rollout_csv = os.path.join(args.results_dir, "rollout_results.csv")
        write_csv(trial_csv, all_trial_rows)
        write_csv(rollout_csv, all_rollout_rows)
        print(f"\nSaved trial-level results to: {trial_csv}")
        print(f"Saved rollout-level results to: {rollout_csv}")

    # print global summary per model
    print("\n=== Overall Summary Per Model ===")
    unique_models = sorted(set([r["model"] for r in all_trial_rows]))
    for model_name in unique_models:
        rows = [r for r in all_trial_rows if r["model"] == model_name]
        success_rates = [r["success_rate_mean"] for r in rows]
        returns = [r["return_mean"] for r in rows]

        summary = {
            "model": model_name,
            "n_trials": len(rows),
            "success_rate_mean_over_trials": float(np.mean(success_rates)),
            "success_rate_std_over_trials": float(np.std(success_rates)),
            "return_mean_over_trials": float(np.mean(returns)),
            "return_std_over_trials": float(np.std(returns)),
        }
        print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        required=True,
        help="one or more checkpoint paths",
    )

    parser.add_argument(
        "--agent_names",
        type=str,
        nargs="+",
        default=None,
        help="optional display names for the models",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="number of repeated trials per model",
    )

    parser.add_argument(
        "--rollouts_per_trial",
        type=int,
        default=50,
        help="number of rollout episodes per trial",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="override maximum rollout horizon from checkpoint",
    )

    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=None,
        help="optional list of env names to evaluate on; if omitted, uses checkpoint env",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="render on screen",
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="if set, videos are saved per model/trial into results dir",
    )

    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="write video every n frames",
    )

    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["agentview"],
        help="camera name(s) for rendering",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="multi_eval_results",
        help="directory to save csv summaries and optional artifacts",
    )

    parser.add_argument(
        "--save_datasets",
        action="store_true",
        help="save one hdf5 rollout dataset per model/trial",
    )

    parser.add_argument(
        "--dataset_obs",
        action="store_true",
        help="include observations when saving datasets",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base seed; trial i uses seed + i",
    )

    args = parser.parse_args()
    run_multi_eval(args)