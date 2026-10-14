# Train the full ContactNets-inspired CaMI model

Use `bc_cami_contactNets` in both registries and the experiment JSON. The
registered config is `BCCaMIContactNetsConfig` in
`robomimic/config/bc_cami_configContactNetsv2.py`; the policy implementation is
`BC_CaMI_ContactNets` in `robomimic/algo/bc_cami_contactNets.py`.

Older force-only checkpoints tagged `bc_cami_contactNets` still select the
original force-CaMI architecture when their complete saved config lacks the
`cami.continuous` descriptor block. New training uses the full physics template.

This is BC-RNN with a physical descriptor used to weight contrastive negatives.
It does not implement the ContactNets dynamics model or its optimization loss.
The descriptor is signed gap in metres, normal impulse in N*s, and tangential
impulse magnitude in N*s. The latter sums friction impulse magnitudes across
contact points and physics substeps, so opposing loads do not cancel. It uses
no tangential speed. Huber distance uses normalized descriptors with delta 0.1.

## 1. Prepare the labels once

An HDF5 containing only `obs/force` does not contain the full descriptor. You
must either replay original recorded actions with their simulator states and
per-demo XML, or use directly recorded gap and impulse channels. The preparer
copies the original to a **new** output and preserves its images, observations,
actions and existing train/valid masks. It never replaces the source file.
An existing output is rejected; use a new filename to prepare another copy.

From the repository root on the training machine:

```bash
conda activate robomimic2
cd ~/robomimic

python -m robomimic.scripts.prepare_cami_c_dataset \
  --dataset dataset/square_image_84_with_force.hdf5 \
  --output dataset/square_image_84_contactnets.hdf5 \
  --mode physics \
  --pairs robomimic/exps/cami_c/square_pairs.json \
  --horizon 10 \
  --quiet
```

The Square pair specification selects SquareNut collision geometries against
the insertion peg on body `peg1`. The resolver requires actual matches and
records the resolved geometry names in the output. For a different environment
or asset version, inspect its collision names and adjust the pair JSON:

```bash
python -m robomimic.scripts.inspect_cami_c_geoms \
  --dataset dataset/square_image_84_with_force.hdf5
```

Replay requires native MuJoCo with `mj_geomDistance`, the original robosuite
version, controller and simulation settings, and per-demo `states` and
`model_file`. If the image HDF5 omits states or XML, add
`--raw-dataset /path/to/the/original_demonstrations.hdf5`. The raw and image
files must have identical demo names and actions. Replay checks qpos drift
against the recorded states and aborts on a mismatch; it does not teleport
between action steps in the default `--replay-mode action`. Matching qpos is
a consistency check and does not prove that the original contact forces have
been recovered.

If strict replay diverges, an explicit alternative is state-aligned
reconstruction. This restores the recorded qpos/qvel at every action boundary,
refreshes controller goals, then simulates that action and integrates contact
forces across its physics substeps. It prevents accumulated trajectory drift
from misaligning the labels with the source observations. The gap comes from
the recorded configuration; the impulses are **state-conditioned simulator
estimates**, with the current model, solver, and reset controller state. This
does not recover unrecorded controller memory or original contact impulses.
It is a different reconstruction assumption, not a larger replay tolerance.

```bash
python -m robomimic.scripts.prepare_cami_c_dataset \
  --dataset dataset/square_image_84_with_force.hdf5 \
  --output dataset/square_image_84_contactnets_state.hdf5 \
  --mode physics \
  --pairs robomimic/exps/cami_c/square_pairs.json \
  --replay-mode state \
  --horizon 10 \
  --quiet
```

The preparer reports one-step qpos drift and stores its maximum, the count of
transitions exceeding the unchanged tolerance, simulator versions, and the
impulse-estimation semantics in `data.attrs['cami_c']['reconstruction']`.
Recorded action intervals must still match simulated durations, and nonfinite
states still fail. Training preserves this provenance in the resolved config.
Use the output above as the training command's `--dataset` when choosing this
mode. In a paper, describe the labels as simulator-estimated impulses from
state-aligned action intervals, rather than recorded impulses.

For already recorded `contact/physics` rows, each with feature order
`[gap, normal_impulse, tangential_impulse_magnitude]` and `contact/dt`:

```bash
python -m robomimic.scripts.prepare_cami_c_dataset \
  --dataset dataset/recorded_physics.hdf5 \
  --output dataset/square_image_84_contactnets.hdf5 \
  --mode physics --contact-key contact/physics --horizon 10 --quiet
```

Store the feature names as a JSON `feature_names` attribute on that source
dataset, or explicitly pass `--contact-features gap normal_impulse
tangential_impulse_magnitude` for unmarked rows. A recorded `contact/valid`
mask is used automatically. `--valid-key` selects another validity source.
For separate channels, use `--gap-key`, `--normal-impulse-key`,
`--tangential-impulse-key` and `--impulse-dt SECONDS`. Converting force columns
with `--normal-force-key` / `--tangential-force-key` and `--impulse-dt` is only
a force-times-duration estimate, not an integrated substep measurement.

The output stores `contact_state` and `contact_mask` with shape `[T,10,3]`,
with offsets `t+1` through `t+10`. Future steps past a demo boundary have zero
validity. `data.attrs['cami_c']` declares the schema, feature order, units,
source provenance, horizon, action interval and split names.

## 2. Train one model

```bash
python -m robomimic.scripts.train \
  --config robomimic/exps/templates/bc_cami_square_contactNetsv2.json \
  --dataset dataset/square_image_84_contactnets.hdf5 \
  --name square_cami_contactnets_lr_decay \
  --wandb-project cami-square-lr-decay \
  --quiet
```

Use your existing `WANDB_ENTITY` setting to select the shared W&B team.
The config enables held-out validation with `train` and `valid` masks, trains
for 2,000 epochs, runs 50 evaluation episodes every 50 epochs, and saves regular
and best-success checkpoints. All four online optimizers start at 1e-4 and
halve their rates after epoch 1,000. JSON settings control these values.
The latest template's seed (1), batch size (16), and policy architecture are
preserved. The previous CaMI coefficient defaults are explicit: state 0.01,
trajectory 1.0, and EMA momentum 0.995.

The three active normalization scales default to null. Training fits std plus
1e-6 on the **selected training demos**, counting each valid future measurement
once. Validation is excluded. The resolved scales and fitted demo names are
written to the run config. Explicit positive scales are kept if supplied.

Training checks descriptor shapes, units, horizon, masks, finite values,
nonnegative impulse magnitudes, observations and disjoint splits before
opening output directories. It fails with a preparation instruction on an
unprepared force-only HDF5. Neither force nor the physical labels enter the
policy or goal modalities. Rollouts need only the policy's normal observations.

Results are under:

```text
~/robomimic/trained_models/square_cami_contactnets_lr_decay/<timestamp>/
```

Existing `train_trials.py` can use this same JSON and prepared dataset. It
trains one model; `--n-trials` controls the subsequent repeated evaluations.

## Verification

```bash
python -m pytest -q tests/test_contactnets_integration.py tests/test_contactnets_physics.py
```

The fixtures are synthetic and test interfaces and dynamics. Passing them does
not establish Square success, replay compatibility with your recorded dataset,
or an improvement over a previously trained force-only checkpoint.
