# CAMI pipeline with the TIAGo Pro (`TiagoProRight`)

`TiagoProRight` (robosuite fork, `robosuite/models/robots/manipulators/tiago_pro_right_robot.py`) is the right arm of the
PAL TIAGo Pro with the PAL Pro gripper, everything else frozen. It uses the single-arm API, so it is a drop-in for
`Panda` in the CAMI pipeline:

| | Panda | TiagoProRight |
|---|---|---|
| action dim (NutAssemblySquare, ToolHang) | 7 (OSC_POSE delta 6 + gripper 1) | 7 (Wipe: 6, `WipingGripper` has no DoF) |
| `robot0_eef_pos`, `robot0_eef_quat` | yes | yes (same shapes) |
| `robot0_gripper_qpos` | 2-D | **8-D** (one commanded joint + 7 mimic joints; robomimic reads the shape from the dataset) |
| `robot0_eye_in_hand_image` | yes | yes (wrist camera, fovy 75, beside the gripper housing, see `tools/tiago_pro/README.md`) |
| F/T sensors | `gripper0_right_force_ee`, `gripper0_right_torque_ee` | same names, same defaults of `augment_dataset_with_force.py` |

The observation keys in the TiagoProRight templates are byte-identical to the Panda ones; the only differences from
`bc_cami*.json` are `experiment.name` and `train.output_dir` (suffix `_tiagoproright`). Datasets must be generated with
`TiagoProRight` (a Panda dataset has 2-D `robot0_gripper_qpos` and different image/F/T statistics), never mixed.

Templates: `exps/templates/bc_cami_tiagopro.json`, `bc_cami_lcp_v1_tiagopro.json`, `bc_cami_lcp_v2_tiagopro.json`,
`bc_cami_lcp_v2_marginal_tiagopro.json`. (`rollout.horizon` etc. are the ToolHang values of the originals.)

## Commands (training / rollouts need torch: not run in the development container)

```bash
# 1. collect (needs a device; robosuite fork). env_args.env_kwargs.robots == ["TiagoProRight"], controller = default_tiagoproright.json
python robosuite/scripts/collect_human_demonstrations_force_vision.py --environment ToolHang \
    --robots TiagoProRight --camera sideview robot0_eye_in_hand --device spacemouse
#    no device? scripted smoke test of the same code path: python tools/tiago_pro/scripted_collect_smoke.py

# 2. states -> observations (84x84 images; use the camera names of the template)
python robomimic/scripts/dataset_states_to_obs.py --done_mode 2 --dataset <demo.hdf5> \
    --output_name image_84.hdf5 --camera_names sideview robot0_eye_in_hand --camera_height 84 --camera_width 84

# 3. force + contact labels (sensor names default to gripper0_right_force_ee / gripper0_right_torque_ee)
python robomimic/scripts/augment_dataset_with_force.py --raw_dataset <demo.hdf5> \
    --extracted_dataset image_84.hdf5 --output_dataset image_84_force.hdf5

# 4. train (set train.data to image_84_force.hdf5)
python robomimic/scripts/train.py --config robomimic/exps/templates/bc_cami_tiagopro.json --dataset image_84_force.hdf5
```

## Things to know before trusting the labels

* **mujoco 3.5.0 needs the robosuite fork's `MjSim.step2` fix.** With 3.5.0 (this repo's pin) and robosuite's default
  `lite_physics`, `mj_step1/mj_step2` left the force/torque sensors at their reset-time value (live collection and the
  `force` observable were constant, for Panda too). The fork now clears the lazy `flg_rnepost` flag in `step2`; datasets
  collected on 3.5.0 with an older robosuite checkout have a frozen live force column.

* **State replay is not the live force.** `augment_dataset_with_force.py` does `set_state_from_flattened(states[t]);
  forward()` and reads the sensor. `data.ctrl` is not part of the stored state, so the replay uses a stale ctrl, and
  the wrist force depends on qacc. On a scripted TiagoProRight grasp (tools/tiago_pro/scripted_collect_smoke.py) the
  replayed force differs from the live one by up to 36 N (median 1.9 N) and 8.8% of the steps flip the 10 N label.
  Re-stepping the stored actions from `states[0]` (what robomimic's `reset_to` + `step` does) reproduces the states
  exactly and the force to < 0.2 N (solver warm start). The script itself is unchanged here.
* **The bias is taken at reset time** (`NutAssembly._reset_internal`, first replayed state in the augment script), which
  is a transient, not the weight of the gripper. With TiagoProRight (0.81 kg gripper, 8.0 N) a motionless robot reads
  `|F - bias| = 2.6 N` (Panda: 1.9 N).
* **Gravity re-projection.** After a wrist tilt of 20 deg the bias-corrected force changes by 2.7 N without any contact
  (30 deg: 4.1 N, 45 deg: 6.1 N); yaw about the approach axis does not matter. The init pose has the approach axis
  within 5 deg of vertical (incl. the default joint init noise).
* **Free-space dynamics.** At full-amplitude oscillating motions the free-space `|F - F_settled|` reaches 11.6 N on this
  robot (Panda: 4.6 N, same script), above the 10 N threshold; at half amplitude 4.1 N. Re-tune
  `contact_threshold` (10 N) after the real data distribution is known.
* **Reach.** With the base at the table edge the arm reaches the nut / tool / frame regions of NutAssemblySquare and
  ToolHang with the gripper pointing down, but only about 6-12% of the NutAssemblySquare peg region (peg 1 at x = 0.23 m).
