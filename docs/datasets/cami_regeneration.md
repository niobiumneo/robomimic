# Regenerate Square and Tool Hang datasets for CaMI

Run `rebuild_cami_dataset.py` from the root of `niobiumneo/robomimic`, branch
`contact-state`, in your existing CaMI Python environment. In the container
with `~/robomimic` mounted at `/project`, start with `cd /project`.

The script replays a complete demonstration's recorded actions continuously,
regenerates camera and robot observations, records the simulated wrist wrench,
and creates future-contact labels. It checks position, velocity, and time
against the recorded states throughout replay.

## Required source data

The input is a robomimic HDF5 with:

- `data.attrs["env_args"]`: environment and controller configuration.
- `data/demo_*/actions`: the original actions.
- `data/demo_*/states`: simulation states at each action, with an optional
  additional terminal state.
- `data/demo_*.attrs["model_file"]`: the episode's model XML.
- Optional per-episode `ep_meta` and dataset split masks.

If the processed dataset was deleted, use the remaining raw HDF5. If all
HDF5 files are missing, download the public proficient-human demonstrations
below. Recovering custom deleted demonstrations requires their original
recordings or a backup; videos alone do not supply the required simulator data.

## 1. Download public raw demonstrations

Skip this step when you already have raw files, and substitute their paths in
the replay commands.

```bash
cd /project

python rebuild_cami_dataset.py download \
  --download-dir ./datasets \
  --tasks square tool_hang
```

The command uses `robomimic.DATASET_REGISTRY` and `huggingface_hub` to download
the PH raw files. With this branch's current registry, their paths are:

```text
datasets/square/ph/demo_v15.hdf5
datasets/tool_hang/ph/demo_v15.hdf5
```

Existing HDF5 files are reused. A file with the expected name that is not
HDF5 causes an error instead of being overwritten. Dataset downloading and
replay do not directly import `robomimic.algo` or `bc_cami.py`.

## 2. Check one complete demonstration from each task

This first pass skips camera rendering.

```bash
for task in square tool_hang; do
  python rebuild_cami_dataset.py \
    --task "$task" \
    --dataset "datasets/$task/ph/demo_v15.hdf5" \
    --output "datasets/$task/ph/cami_check.hdf5" \
    --low-dim \
    --n 1 || break
done
```

Check that both commands succeed. If replay diverges, match the source
robosuite/MuJoCo versions, controller configuration, action convention, and
model XML before proceeding. Recorded states do not retain all controller
memory or solver state. Increasing tolerances can conceal an incompatible
replay and does not restore the original forces.

Output paths must be new. On another successful check, choose a different
output name or deliberately remove the previous check file yourself.

## 3. Rebuild the complete image datasets

```bash
export MUJOCO_GL=egl

for task in square tool_hang; do
  python rebuild_cami_dataset.py \
    --task "$task" \
    --dataset "datasets/$task/ph/demo_v15.hdf5" \
    --output "datasets/$task/ph/cami_with_force.hdf5" \
    --horizon 10 \
    --contact-threshold 10.0 || break
done
```

EGL rendering requires a functioning GPU-enabled container and graphics
drivers. Use your environment's supported MuJoCo rendering backend if EGL
is unavailable. `--low-dim` omits image rendering entirely.

| Task | Environment | Default cameras | Image size |
| --- | --- | --- | --- |
| Square insertion | `NutAssemblySquare` | `agentview`, `robot0_eye_in_hand` | 84 x 84 |
| Tool Hang | `ToolHang` | `sideview`, `robot0_eye_in_hand` | 240 x 240 |

The Tool Hang resolution accommodates the 216-pixel crop in the branch's
`robomimic/exps/templates/bc_cami.json`. For Square, use camera keys matching
the table and a crop that fits within 84 x 84. `--camera-names` and
`--image-size` override the defaults.

## Stored fields and timing

Each `data/demo_*` contains:

| Field | Meaning |
| --- | --- |
| `obs/force` | Six channels `[Fx, Fy, Fz, Tx, Ty, Tz]` with the selected bias treatment |
| `contact/wrist_wrench_raw` | Unmodified simulated wrist readings, in N and N*m |
| `contact/wrist_bias` | The six-channel offset subtracted from the raw readings |
| `contact_label` | Binary future-contact label, shape `[T, 1]` |
| `contact_label_valid` | Whether at least one future sample exists |
| `contact_future_count` | Number of available future samples, up to `H` |
| `states` | Replayed pre-action simulation states |
| `reference_states` | Source states retained for comparison |
| `contact/replay_error` | Pre-action position, velocity, and time errors |
| `contact/post_action_replay_error` | Post-action errors where a reference state exists |
| `contact/terminal_state` | Final simulated state after the last action |
| `contact/terminal_wrist_wrench_raw` | Final simulated wrist reading |
| `obs/*`, `actions`, `rewards`, `dones` | Regenerated observations, original actions, and recomputed rewards/dones |

`obs[t]`, `states[t]`, and `obs/force[t]` precede `actions[t]`. The first
wrist reading is a forward-dynamics estimate after reset. Subsequent readings
come from the previous action's last solved physics substep. If the source
does not include a terminal state, the final transition cannot be compared
against a reference; this is recorded in the replay report.

The default `--bias initial` subtracts the first wrist wrench throughout the
episode. This is a fixed baseline correction and is not gravity compensation.
Use `--bias none` for unmodified measurements in `obs/force`. Raw measurements
are retained in either mode. Wrist readings can include gravity, inertia, and
loads from multiple contacts; they do not isolate nut/peg or tool/frame contact.

`contact_label[t]` is one when an available sample among `t+1` through `t+H`
has `norm(force[:3]) > contact_threshold`. Torque is excluded from this norm.
Partial windows use only available samples. The final label is zero with
`contact_label_valid=0`. Downstream code must explicitly load and use that
validity field to mask it; adding it to the HDF5 alone does not change the
existing trainer's handling of labels.

Match `--horizon` to the training configuration's `snippet_horizon`.
Force-based continuous CaMI can load `obs/force` as auxiliary supervision via
`train.dataset_keys`; it should stay outside the policy's observation keys.
The dataset also provides the root `contact_label` used by binary CaMI.
Loss settings such as Huber remain training configuration choices.

`next_obs` is omitted by default, matching CaMI's `hdf5_load_next_obs=false`.
Add `--include-next-obs` when needed. Its final force comes from the actual
last replayed action, rather than duplicating the previous force row.

## Output integrity and limits

The script keeps source files unchanged, writes images incrementally, and
publishes the output path only after every selected demonstration succeeds.
It removes its partial output on an ordinary Python exception and preserves
split masks, filtered to the selected demonstrations. The HDF5 stores replay
settings, simulator versions, error summaries, and force provenance.

These measurements are generated in the replay simulator. Passing trajectory
checks does not prove exact recovery of the original forces, since the saved
states do not include every quantity needed to reproduce the dynamics.

## Verification

```bash
python -m pytest -q tests/test_rebuild_cami_dataset.py
```

The six focused tests use a small native MuJoCo model to check force/action
alignment, future-label boundaries, bias handling, drift rejection, output
preservation, and dataset downloading without training-module imports. They
do not execute the complete public Square or Tool Hang demonstrations or
validate GPU rendering. Run the one-demonstration checks in your container.
