# import json
# import h5py

# path = "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robosuite_datasets/table_wiping/1772931257_163442/demo.hdf5"

# with h5py.File(path, "r+") as f:
#     data_grp = f["data"]

#     raw = None
#     if "env_args" in data_grp.attrs:
#         raw = data_grp.attrs["env_args"]
#     elif "env_info" in data_grp.attrs:
#         raw = data_grp.attrs["env_info"]
#     else:
#         raise KeyError("Neither 'env_args' nor 'env_info' found.")

#     if isinstance(raw, bytes):
#         raw = raw.decode("utf-8")

#     env_info = json.loads(raw)

#     # If already in robomimic-style format, keep it
#     if "env_kwargs" in env_info:
#         env_args = env_info
#     else:
#         # Wrap your existing config into robomimic's expected structure
#         env_args = {
#             "type": 1,
#             "env_name": env_info["env_name"],
#             "env_kwargs": env_info,
#         }

#     data_grp.attrs["env_args"] = json.dumps(env_args)
#     print("Wrote robomimic-compatible env_args:")
#     print(json.dumps(env_args, indent=2))

import h5py
import numpy as np

path = "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robosuite_datasets/table_wiping/1772931257_163442/demo.hdf5"

with h5py.File(path, "r+") as f:
    data_grp = f["data"]

    for demo_name in data_grp.keys():
        demo = data_grp[demo_name]

        if "states" not in demo:
            print(f"{demo_name}: no states, skipping")
            continue

        if "obs" not in demo:
            demo.create_group("obs")
        if "next_obs" not in demo:
            demo.create_group("next_obs")

        obs_grp = demo["obs"]
        next_obs_grp = demo["next_obs"]

        states = demo["states"][()].astype(np.float32)

        state_dim_original = int(demo.attrs["state_dim_original"])
        state_dim_ft = int(demo.attrs["state_dim_ft"])

        if state_dim_ft < 6:
            raise ValueError(f"{demo_name}: expected at least 6 FT dims, got {state_dim_ft}")

        force = states[:, state_dim_original:state_dim_original + 6].astype(np.float32)

        # next_obs version: shifted by one, last repeated
        next_force = np.concatenate([force[1:], force[-1:]], axis=0)

        if "force" in obs_grp:
            del obs_grp["force"]
        obs_grp.create_dataset("force", data=force)

        if "force" in next_obs_grp:
            del next_obs_grp["force"]
        next_obs_grp.create_dataset("force", data=next_force)

        if "actions" in demo:
            T = demo["actions"].shape[0]
        elif "states" in demo:
            T = demo["states"].shape[0]
        else:
            raise KeyError(f"{demo_name}: neither actions nor states found")

        demo.attrs["num_samples"] = int(T)
        print(f"{demo_name}: set num_samples = {T}")

        print(f"{demo_name}: wrote obs/force and next_obs/force with shape {force.shape}")