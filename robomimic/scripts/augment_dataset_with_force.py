import os
import json
import shutil
import argparse
import h5py
import numpy as np
import robosuite as suite


def make_env_from_env_info(env_info):
    meta = json.loads(env_info) if isinstance(env_info, str) else dict(env_info)

    if "env_kwargs" in meta:
        env_name = meta.get("env_name", None)
        env_kwargs = dict(meta["env_kwargs"])
        if env_name is not None and "env_name" not in env_kwargs:
            env_kwargs["env_name"] = env_name
    else:
        env_kwargs = dict(meta)

    env_name = env_kwargs.pop("env_name", None)
    if env_name is None:
        raise ValueError(f"Could not find env_name in env metadata. Keys: {list(meta.keys())}")

    if isinstance(env_kwargs.get("robots", None), str):
        env_kwargs["robots"] = [env_kwargs["robots"]]

    # remove metadata keys
    for k in ["type", "env_version", "repository_version", "env_lang"]:
        env_kwargs.pop(k, None)

    # remove runtime keys that we will set explicitly
    for k in [
        "has_renderer",
        "has_offscreen_renderer",
        "ignore_done",
        "use_camera_obs",
        "reward_shaping",
        "control_freq",
    ]:
        env_kwargs.pop(k, None)

    print("env_name:", env_name)
    print("env_kwargs keys:", sorted(env_kwargs.keys()))

    env = suite.make(
        env_name,
        **env_kwargs,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    return env

def reset_env_to_demo(env, model_xml_str):
    env.reset()
    xml = env.edit_model_xml(model_xml_str)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.forward()

def read_robot_ft_from_env(env):
    """
    Read end-effector force / torque from robosuite robot object directly.
    Returns concatenated single-arm 3D force and 3D torque.
    """
    robot = env.robots[0]
    arms = robot.arms

    force_vals = []
    torque_vals = []

    for arm in arms:
        force_vals.append(np.asarray(robot.ee_force[arm], dtype=np.float32).reshape(-1)[:3])
        torque_vals.append(np.asarray(robot.ee_torque[arm], dtype=np.float32).reshape(-1)[:3])

    F = np.concatenate(force_vals, axis=0) if len(force_vals) > 1 else force_vals[0]
    T = np.concatenate(torque_vals, axis=0) if len(torque_vals) > 1 else torque_vals[0]

    return F, T

def get_ft_sensor_slices(env,
                         force_sensor_name="gripper0_right_force_ee",
                         torque_sensor_name="gripper0_right_torque_ee"):
    sim = env.sim

    f_id = sim.model.sensor_name2id(force_sensor_name)
    t_id = sim.model.sensor_name2id(torque_sensor_name)

    adr_f = int(sim.model.sensor_adr[f_id])
    dim_f = int(sim.model.sensor_dim[f_id])
    adr_t = int(sim.model.sensor_adr[t_id])
    dim_t = int(sim.model.sensor_dim[t_id])

    return adr_f, dim_f, adr_t, dim_t


def read_raw_ft_from_sim(env, sensor_slices):
    adr_f, dim_f, adr_t, dim_t = sensor_slices
    sim = env.sim

    F_raw = np.asarray(sim.data.sensordata[adr_f:adr_f + dim_f], dtype=np.float32).reshape(-1)[:3]
    T_raw = np.asarray(sim.data.sensordata[adr_t:adr_t + dim_t], dtype=np.float32).reshape(-1)[:3]

    return F_raw, T_raw


def compute_force_sequences_with_bias(
    env,
    states_arr,
    bias_alpha=0.1,
    force_sensor_name="gripper0_right_force_ee",
    torque_sensor_name="gripper0_right_torque_ee",
    obs_force_key="robot0_ee_force",
    obs_torque_key="robot0_ee_torque",
):
    """
    Returns two unscaled force sequences of shape (T, 6):

    force_rawbias:
        raw mujoco sensor retrieval + EMA bias when ncon == 0

    force_obsbias:
        robosuite obs retrieval + EMA bias when ncon == 0

    Both are stored as:
        [Fx, Fy, Fz, Tx, Ty, Tz]
    """
    T = states_arr.shape[0]
    force_rawbias = np.zeros((T, 6), dtype=np.float32)
    force_obsbias = np.zeros((T, 6), dtype=np.float32)

    sensor_slices = get_ft_sensor_slices(
        env,
        force_sensor_name=force_sensor_name,
        torque_sensor_name=torque_sensor_name,
    )

    bias_raw_F = None
    bias_raw_T = None
    bias_obs_F = None
    bias_obs_T = None

    for t in range(T):
        env.sim.set_state_from_flattened(states_arr[t])
        env.sim.forward()

        # ----- option 1: raw Mujoco sensors -----
        F_raw, T_raw = read_raw_ft_from_sim(env, sensor_slices)

        if bias_raw_F is None:
            bias_raw_F = F_raw.copy()
            bias_raw_T = T_raw.copy()

        # ----- option 2: robot API FT retrieval -----
        # First try observation keys if present, otherwise fall back to robot.ee_force / ee_torque
        obs = env._get_observations()

        if obs_force_key in obs and obs_torque_key in obs:
            F_obs = np.asarray(obs[obs_force_key], dtype=np.float32).reshape(-1)[:3]
            T_obs = np.asarray(obs[obs_torque_key], dtype=np.float32).reshape(-1)[:3]
        else:
            F_obs, T_obs = read_robot_ft_from_env(env)

        if bias_obs_F is None:
            bias_obs_F = F_obs.copy()
            bias_obs_T = T_obs.copy()

        # bias update gate copied from collection script
        ncon = int(env.sim.data.ncon)
        if ncon == 0:
            a = float(bias_alpha)

            bias_raw_F = (1.0 - a) * bias_raw_F + a * F_raw
            bias_raw_T = (1.0 - a) * bias_raw_T + a * T_raw

            bias_obs_F = (1.0 - a) * bias_obs_F + a * F_obs
            bias_obs_T = (1.0 - a) * bias_obs_T + a * T_obs

        F_raw_corr = F_raw - bias_raw_F
        T_raw_corr = T_raw - bias_raw_T

        F_obs_corr = F_obs - bias_obs_F
        T_obs_corr = T_obs - bias_obs_T

        force_rawbias[t, :3] = F_raw_corr
        force_rawbias[t, 3:] = T_raw_corr

        force_obsbias[t, :3] = F_obs_corr
        force_obsbias[t, 3:] = T_obs_corr

    return force_rawbias, force_obsbias

def compute_future_contact_labels(force_seq, snippet_horizon=10, contact_threshold=10.0):
    """
    force_seq: (T, 6) array
    uses only translational force [:, :3]

    Returns:
        contact_label: (T, 1) float32 array
        where label[t] = 1 if any future step in t+1 ... t+H exceeds threshold
    """
    T = force_seq.shape[0]
    labels = np.zeros((T, 1), dtype=np.float32)

    force_mag = np.linalg.norm(force_seq[:, :3], axis=1)  # (T,)

    for t in range(T):
        start = t + 1
        end = min(t + 1 + snippet_horizon, T)

        if start >= end:
            future_mag = force_mag[t:t+1]
        else:
            future_mag = force_mag[start:end]

        labels[t, 0] = 1.0 if np.any(future_mag > contact_threshold) else 0.0

    return labels

def safe_write_dataset(group, name, data):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, compression="gzip", compression_opts=4)


def augment_dataset_with_force(
    raw_dataset_path,
    extracted_dataset_path,
    output_dataset_path,
    bias_alpha=0.1,
    force_sensor_name="gripper0_right_force_ee",
    torque_sensor_name="gripper0_right_torque_ee",
    make_force_alias=True,
    snippet_horizon=10,
    contact_threshold=10.0
):
    if os.path.abspath(extracted_dataset_path) != os.path.abspath(output_dataset_path):
        shutil.copy2(extracted_dataset_path, output_dataset_path)

    with h5py.File(raw_dataset_path, "r") as f_raw, h5py.File(output_dataset_path, "r+") as f_out:
        raw_data_grp = f_raw["data"]
        out_data_grp = f_out["data"]

        env_info = raw_data_grp.attrs["env_args"] if "env_args" in raw_data_grp.attrs else raw_data_grp.attrs["env_info"]
        env = make_env_from_env_info(env_info)

        demo_keys = sorted(raw_data_grp.keys())

        for demo_key in demo_keys:
            print(f"\nProcessing {demo_key}...")

            raw_demo = raw_data_grp[demo_key]
            out_demo = out_data_grp[demo_key]

            if "states" not in raw_demo:
                raise KeyError(f"{demo_key} in raw dataset has no 'states'")
            if "model_file" not in raw_demo.attrs:
                raise KeyError(f"{demo_key} in raw dataset has no model_file attr")

            states_arr = np.asarray(raw_demo["states"])
            model_xml_str = raw_demo.attrs["model_file"]

            reset_env_to_demo(env, model_xml_str)

            force_rawbias, force_obsbias = compute_force_sequences_with_bias(
                env=env,
                states_arr=states_arr,
                bias_alpha=bias_alpha,
                force_sensor_name=force_sensor_name,
                torque_sensor_name=torque_sensor_name,
            )

            contact_label = compute_future_contact_labels(
                force_seq=force_rawbias,
                snippet_horizon=snippet_horizon,
                contact_threshold=contact_threshold,
            )

            obs_force_rawbias = force_rawbias
            obs_force_obsbias = force_obsbias

            next_obs_force_rawbias = np.concatenate([force_rawbias[1:], force_rawbias[-1:]], axis=0)
            next_obs_force_obsbias = np.concatenate([force_obsbias[1:], force_obsbias[-1:]], axis=0)

            if "obs" not in out_demo:
                out_demo.create_group("obs")
            if "next_obs" not in out_demo:
                out_demo.create_group("next_obs")

            obs_grp = out_demo["obs"]
            next_obs_grp = out_demo["next_obs"]

            safe_write_dataset(obs_grp, "force_rawbias", obs_force_rawbias)
            safe_write_dataset(next_obs_grp, "force_rawbias", next_obs_force_rawbias)

            safe_write_dataset(obs_grp, "force_obsbias", obs_force_obsbias)
            safe_write_dataset(next_obs_grp, "force_obsbias", next_obs_force_obsbias)

            safe_write_dataset(out_demo, "contact_label", contact_label)

            safe_write_dataset(obs_grp, "contact_label", contact_label)
            next_obs_contact_label = np.concatenate([contact_label[1:], contact_label[-1:]], axis=0)
            safe_write_dataset(next_obs_grp, "contact_label", next_obs_contact_label)

            if make_force_alias:
                safe_write_dataset(obs_grp, "force", obs_force_rawbias)
                safe_write_dataset(next_obs_grp, "force", next_obs_force_rawbias)

            # store some metadata
            out_demo.attrs["force_bias_alpha"] = float(bias_alpha)
            out_demo.attrs["force_sensor_name"] = force_sensor_name
            out_demo.attrs["torque_sensor_name"] = torque_sensor_name
            out_demo.attrs["force_rawbias_desc"] = "raw Mujoco sensor retrieval + EMA bias when ncon == 0, unscaled"
            out_demo.attrs["force_obsbias_desc"] = "robosuite obs retrieval + EMA bias when ncon == 0, unscaled"
            out_demo.attrs["contact_label_desc"] = "binary future-contact label derived from translational force norm over window t+1..t+H"
            out_demo.attrs["contact_label_horizon"] = int(snippet_horizon)
            out_demo.attrs["contact_label_threshold"] = float(contact_threshold)

            print(f"  obs/force_rawbias      {obs_force_rawbias.shape}")
            print(f"  next_obs/force_rawbias {next_obs_force_rawbias.shape}")
            print(f"  obs/force_obsbias      {obs_force_obsbias.shape}")
            print(f"  next_obs/force_obsbias {next_obs_force_obsbias.shape}")
            if make_force_alias:
                print(f"  obs/force alias -> force_rawbias")

        env.close()

    print(f"\nDone. Augmented dataset saved to:\n{output_dataset_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset", type=str, required=True)
    parser.add_argument("--extracted_dataset", type=str, required=True)
    parser.add_argument("--output_dataset", type=str, required=True)
    parser.add_argument("--bias_alpha", type=float, default=0.1)
    parser.add_argument("--force_sensor_name", type=str, default="gripper0_right_force_ee")
    parser.add_argument("--torque_sensor_name", type=str, default="gripper0_right_torque_ee")
    parser.add_argument("--no_force_alias", action="store_true")
    parser.add_argument("--snippet_horizon", type=int, default=10)
    parser.add_argument("--contact_threshold", type=float, default=10.0)
    args = parser.parse_args()

    augment_dataset_with_force(
        raw_dataset_path=args.raw_dataset,
        extracted_dataset_path=args.extracted_dataset,
        output_dataset_path=args.output_dataset,
        bias_alpha=args.bias_alpha,
        force_sensor_name=args.force_sensor_name,
        torque_sensor_name=args.torque_sensor_name,
        make_force_alias=(not args.no_force_alias),
        snippet_horizon=args.snippet_horizon,
        contact_threshold=args.contact_threshold
    )


if __name__ == "__main__":
    main()