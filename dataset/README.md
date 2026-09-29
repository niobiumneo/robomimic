# Using an existing CaMI dataset

`contact-state` is the Git branch. Put the data inside `dataset/` at the root
of your robomimic checkout, beside `setup.py` and `rebuild_cami_dataset.py`.
The templates use these paths:

| Task | HDF5 path relative to the checkout | Continuous CaMI config |
| --- | --- | --- |
| Square | `dataset/square/ph/cami_with_force.hdf5` | `robomimic/exps/templates/bc_cami_square.json` |
| Tool Hang | `dataset/tool_hang/ph/cami_with_force.hdf5` | `robomimic/exps/templates/bc_cami.json` |

The file can have another name or live elsewhere. Set `train.data` to a list
of objects containing `path`, or use the trainer's `--dataset` override:

```json
"data": [{"path": "/absolute/path/to/your_dataset.hdf5"}]
```

Existing paths relative to the working directory still work. If a relative
path is not found there, CaMI also checks relative to the robomimic checkout.
The resolved absolute paths are stored in the run configuration.

## Required data

A downloaded raw demonstration file containing only states and actions needs
processing before these image-based training configs can use it. Moving or
renaming it does not add observations or force measurements.

| HDF5 field | Required contents |
| --- | --- |
| `data.attrs["env_args"]` | JSON with `env_name`, `type`, and `env_kwargs` |
| `data/demo_N.attrs["num_samples"]` | Episode length `T` |
| `data/demo_N/actions` | Numeric actions, shape `[T, action_dim]` |
| `data/demo_N/obs/robot0_eef_pos` | End-effector position at each timestep |
| `data/demo_N/obs/robot0_eef_quat` | End-effector quaternion at each timestep |
| `data/demo_N/obs/robot0_gripper_qpos` | Gripper joint positions at each timestep |
| `data/demo_N/obs/object` | Task object observations at each timestep |
| `data/demo_N/obs/<camera>_image` | Channel-last uint8 images, `[T,H,W,3]` |
| `data/demo_N/obs/force` | Stored wrench aligned with `obs[t]` and `actions[t]` |
| `mask/train` | Demonstration names selected for training by these templates |

Square uses `agentview_image` and `robot0_eye_in_hand_image`, with a 76x76
crop for the usual 84x84 images. Tool Hang uses `sideview_image` and
`robot0_eye_in_hand_image`, with a 216x216 crop. Both image dimensions must
exceed the configured crop. Adjust the observation keys and crop in your
config if your data uses different cameras or resolutions.

Continuous CaMI accepts `[T,3]` force or `[T,6]` force/torque, ordered
`[Fx,Fy,Fz,Tx,Ty,Tz]`. Its comparison uses only the first three channels,
in newtons. The LCP templates require all six channels. Values must be finite
and have the same number of timesteps as actions. Sequence order and force
timing must already be correct in the file.

If an augmented file stores the desired signal as `obs/force_rawbias` or
`obs/force_obsbias`, set its exact key in
`algo.cami.continuous_contact.force_dataset_key` for CaMI, or
`algo.cami.lcp.force_dataset_key` for LCP. Choose the intended bias treatment
explicitly. Training adds this key to `train.dataset_keys` automatically.
Force and labels must stay outside policy observation modalities.

## Supported configurations

| Config | Algorithm | Additional supervision |
| --- | --- | --- |
| `bc_cami.json` / `bc_cami_square.json` | Continuous force-weighted CaMI | Force; binary labels are unnecessary |
| `bc_cami_lcp_v1.json` | LCP violation objective | Six-channel wrench |
| `bc_cami_lcp_v2.json` | CaNCE with regime negatives | Six-channel wrench and `contact_label` |
| `bc_cami_lcp_v2_marginal.json` | CaNCE with marginal negatives | Six-channel wrench |

The LCP templates currently use Tool Hang's path and cameras. To use Square,
copy the Square template's dataset path, camera keys, crop dimensions, and
400-step rollout horizon into the chosen LCP config.

Binary CaMI (`continuous_contact.enabled=false`) also requires
`data/demo_N/contact_label`, with one 0/1 value per timestep, shaped `[T]` or
`[T,1]`. Its key can be configured with `algo.cami.contact_label_key`.
The regime CaNCE template reads the root `contact_label` key.
Existing future-contact labels retain their original meaning; loading a file
does not convert those labels into instantaneous contact measurements.

LCP sequence padding is disabled so its consecutive pairs contain recorded
timesteps. Its selected demonstrations must contain at least `seq_length`
samples. Continuous CaMI retains the padding validity mask for future force
windows and requires `seq_length >= snippet_horizon + 1`.

## Training split and force scaling

The templates select `mask/train`. If the file intentionally has no split,
set `train.hdf5_filter_key` to `null` to use all demonstrations. To use an
existing subset such as `20_percent_train`, set that mask name instead.
Do not include held-out demonstrations when fitting or training.

Continuous CaMI's `force_scale: null` fits `std(||F_xyz||) + 1e-6` only on
the selected training demonstrations, respecting per-dataset filters and
demo limits. Torque and validation demonstrations are excluded. An explicit
positive scale is preserved. The fitted number is saved with the run config.
An automatic fit with constant force magnitudes fails with an explanation.

Before training, the code checks paths, selected masks, required fields,
signal lengths, finite actions/force, observation dimensions, and image
crops. Incomplete replay files are rejected. These checks verify the data
interface; they do not establish the physical accuracy of replayed forces
or guarantee improved policy performance.

## Start training

From the checkout root in your existing robomimic environment:

```bash
# Tool Hang
python -m robomimic.scripts.train --config robomimic/exps/templates/bc_cami.json

# Square
python -m robomimic.scripts.train --config robomimic/exps/templates/bc_cami_square.json
```

Results go into `trained_models/` in the checkout. WandB logging is disabled
by default; add `--wandb --wandb-project cami-contact-state` to enable it.
See the [W&B setup guide](../docs/tutorials/cami_wandb.md) for login, account
selection, metrics, and offline syncing. The JSON setting
`experiment.logging.log_wandb=true` is also supported.
Rollouts remain enabled and require a compatible robosuite/MuJoCo environment.
Set `experiment.rollout.enabled=false` for an offline training check.

The HDF5 files are ignored by Git in this directory. Keep your source data
and backups separately; these code changes do not upload or alter datasets.
For regeneration, see [the replay guide](../docs/datasets/cami_regeneration.md).
