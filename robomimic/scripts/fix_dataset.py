import json
import h5py
import numpy as np
import robosuite as suite


def reconstruct_obs_from_states(model_xml_str, env_info_dict, states_arr):
    """
    Replay each MuJoCo state and query robosuite observations to get:
      - force:         (T, 6) = [Fx, Fy, Fz, Tx, Ty, Tz]
      - robot0_eef_pos (T, 3)
      - robot0_eef_quat (T, 4)

    Returns:
        force_arr, eef_pos_arr, eef_quat_arr
    """
    env_cfg = dict(env_info_dict)

    if isinstance(env_cfg.get("robots", None), str):
        env_cfg["robots"] = [env_cfg["robots"]]

    env = suite.make(
        **env_cfg,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env.reset()
    xml = env.edit_model_xml(model_xml_str)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.forward()

    T = states_arr.shape[0]

    force_arr = np.zeros((T, 6), dtype=np.float32)
    eef_pos_arr = np.zeros((T, 3), dtype=np.float32)
    eef_quat_arr = np.zeros((T, 4), dtype=np.float32)

    printed_keys = False

    for i in range(T):
        env.sim.set_state_from_flattened(states_arr[i])
        env.sim.forward()

        obs = env._get_observations()

        if not printed_keys:
            print("Observation keys available during replay:")
            print(list(obs.keys()))
            printed_keys = True

        if "robot0_ee_force" not in obs:
            raise KeyError(f"robot0_ee_force not found. Keys: {list(obs.keys())}")
        if "robot0_ee_torque" not in obs:
            raise KeyError(f"robot0_ee_torque not found. Keys: {list(obs.keys())}")
        if "robot0_eef_pos" not in obs:
            raise KeyError(f"robot0_eef_pos not found. Keys: {list(obs.keys())}")
        if "robot0_eef_quat" not in obs:
            raise KeyError(f"robot0_eef_quat not found. Keys: {list(obs.keys())}")

        force_arr[i, :3] = np.asarray(obs["robot0_ee_force"], dtype=np.float32).reshape(-1)[:3]
        force_arr[i, 3:] = np.asarray(obs["robot0_ee_torque"], dtype=np.float32).reshape(-1)[:3]
        eef_pos_arr[i] = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3]
        eef_quat_arr[i] = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)[:4]

    env.close()
    return force_arr, eef_pos_arr, eef_quat_arr


def make_next_obs(arr):
    return np.concatenate([arr[1:], arr[-1:]], axis=0)


path = "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robosuite_datasets/table_wiping/1772931257_163442/demo.hdf5"

with h5py.File(path, "r+") as f:
    data_grp = f["data"]

    # Get env config once from dataset-level attrs
    if "env_info" in data_grp.attrs:
        env_info_raw = data_grp.attrs["env_info"]
    elif "env_args" in data_grp.attrs:
        env_args = json.loads(data_grp.attrs["env_args"])
        env_info_raw = env_args["env_kwargs"]
    else:
        raise KeyError("Could not find data.attrs['env_info'] or data.attrs['env_args'].")

    if isinstance(env_info_raw, bytes):
        env_info_raw = env_info_raw.decode("utf-8")
    env_info_dict = json.loads(env_info_raw) if isinstance(env_info_raw, str) else dict(env_info_raw)

    for demo_name in data_grp.keys():
        demo = data_grp[demo_name]

        if "states" not in demo:
            print(f"{demo_name}: no states, skipping")
            continue

        if "model_file" not in demo.attrs:
            print(f"{demo_name}: no model_file attr, skipping")
            continue

        if "obs" not in demo:
            demo.create_group("obs")
        if "next_obs" not in demo:
            demo.create_group("next_obs")

        obs_grp = demo["obs"]
        next_obs_grp = demo["next_obs"]

        states_ext = demo["states"][()].astype(np.float32)
        state_dim_original = int(demo.attrs["state_dim_original"])

        # IMPORTANT: replay only the original MuJoCo state, not appended FT dims
        states_orig = states_ext[:, :state_dim_original]

        model_xml_str = demo.attrs["model_file"]
        if isinstance(model_xml_str, bytes):
            model_xml_str = model_xml_str.decode("utf-8")

        try:
            force_arr, eef_pos_arr, eef_quat_arr = reconstruct_obs_from_states(
                model_xml_str=model_xml_str,
                env_info_dict=env_info_dict,
                states_arr=states_orig,
            )
        except Exception as e:
            print(f"{demo_name}: reconstruction failed -> {e}")
            continue

        next_force_arr = make_next_obs(force_arr)
        next_eef_pos_arr = make_next_obs(eef_pos_arr)
        next_eef_quat_arr = make_next_obs(eef_quat_arr)

        write_items = {
            "force": force_arr,
            "robot0_eef_pos": eef_pos_arr,
            "robot0_eef_quat": eef_quat_arr,
        }

        write_next_items = {
            "force": next_force_arr,
            "robot0_eef_pos": next_eef_pos_arr,
            "robot0_eef_quat": next_eef_quat_arr,
        }

        for key, arr in write_items.items():
            if key in obs_grp:
                del obs_grp[key]
            obs_grp.create_dataset(key, data=arr)

        for key, arr in write_next_items.items():
            if key in next_obs_grp:
                del next_obs_grp[key]
            next_obs_grp.create_dataset(key, data=arr)

        T = demo["actions"].shape[0] if "actions" in demo else states_orig.shape[0]
        demo.attrs["num_samples"] = int(T)

        print(f"{demo_name}:")
        print(f"  wrote obs/force            {force_arr.shape}")
        print(f"  wrote obs/robot0_eef_pos  {eef_pos_arr.shape}")
        print(f"  wrote obs/robot0_eef_quat {eef_quat_arr.shape}")
        print(f"  set num_samples = {T}")