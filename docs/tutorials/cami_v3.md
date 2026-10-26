# CaMI v3: portable objectives and separate experiments

v3 adds new modules, configs and a launcher. The existing ContactNets algorithm,
v2 config, preparation script and experiment JSON are preserved. The shared
registries and data-loader dispatch have small additions for the new name
`bc_cami_v3`; existing algorithm names retain their original behavior.

These are experiments intended to improve training. Passing the included
synthetic tests does not establish higher Square success rates.

## Start training

Use the **already prepared** state-aligned dataset. No replay or recollection
is needed when keeping the existing ten-step contact horizon.

```bash
conda activate robomimic2
cd ~/robomimic
git switch contact-state
git pull --ff-only origin contact-state

python -m robomimic.scripts.train_cami_v3 \
  --variant warmup \
  --dataset dataset/square_image_84_contactnets_state.hdf5 \
  --name square_cami_v3_warmup \
  --wandb-project cami-square-v3 \
  --quiet
```

The `warmup` variant uses batch size 32, five preceding observation steps before
the CaMI decision time and ten future action/contact steps. Its 100-epoch
state-loss warmup has since been removed from every variant (`warmup_epochs` is
0), so it now matches `history`. The prediction head is disabled. To train with
both gaps and impulses shaping the weights, use the `contact` variant
([below](#gaps-and-impulses-the-contact-variant)). Training remains
2,000 epochs, with 50 rollout episodes every 50 epochs, validation enabled,
a constant learning rate of 1e-4, and best-success checkpoint saving.
All bundled v3 variants use seed 0. This is not a guarantee of GPU determinism.

For the matching no-CaMI baseline, run the same command with:

```bash
python -m robomimic.scripts.train_cami_v3 \
  --variant baseline \
  --dataset dataset/square_image_84_contactnets_state.hdf5 \
  --name square_bc_rnn_v3_baseline \
  --wandb-project cami-square-v3 \
  --quiet
```

For the optional gap-prediction experiment, run:

```bash
python -m robomimic.scripts.train_cami_v3 \
  --variant prediction \
  --dataset dataset/square_image_84_contactnets_state.hdf5 \
  --name square_cami_v3_gap_prediction \
  --wandb-project cami-square-v3 \
  --quiet
```

`--name` selects the W&B run and local experiment name. These examples have
different names, so their checkpoints and logs belong to separate runs. The
launcher delegates to the existing trainer and also accepts `--config`,
`--debug`, and `--resume`. Debug mode performs two short training epochs and,
if enabled, short rollouts; it is not a performance evaluation.

`train_trials` takes the same options plus `--variant`, and evaluates the best
checkpoint over repeated trials after training; see `cami_trials.md`:

```bash
python -m robomimic.scripts.train_trials \
  --variant faithful \
  --dataset dataset/square_image_84_contactnets_state.hdf5 \
  --name square_cami_v3_faithful --n-trials 10 \
  --wandb-project cami-square-v3 --quiet
```

## Separate variants

| `--variant` | Batch | Sequence | Context before anchor | Warmup | Prediction |
|---|---:|---:|---:|---:|---|
| `baseline_reference` | 16 | 11 | N/A | N/A | No CaMI modules |
| `reference` | 16 | 11 | 0 | 0 | Off |
| `batch32` | 32 | 11 | 0 | 0 | Off |
| `batch64` | 64 | 11 | 0 | 0 | Off |
| `history` | 32 | 16 | 5 | 0 | Off |
| `warmup` | 32 | 16 | 5 | 0 | Off |
| `prediction` | 16 | 16 | 5 | 0 | Future signed gap (5,000 epochs) |
| `prediction_full` | 32 | 16 | 5 | 0 | Future gap and both impulses |
| `contact` | 32 | 16 | 5 | 0 | Future gap and both impulses; see below |
| `baseline` | 32 | 16 | N/A | N/A | No CaMI modules |
| `faithful` | 16 | 11 | 0 | 0 | Off, and see below |
| `baseline_faithful` | 16 | 11 | N/A | N/A | No CaMI modules |

`reference` is a v3 control with the old batch size, anchor time and loss weights.
It is not claimed to reproduce v2 exactly: v3 groups the auxiliary parameters
under one optimizer and clips that group together. The original v2 remains
available for an exact comparison with previously trained models.

The baseline uses the same deterministic BC-RNN policy, image augmentation,
observations and BC objective as its matching v3 setting. It excludes padded
steps from BC loss, constructs no CaMI network or auxiliary optimizer, and
requires no contact labels. `baseline` matches the batch and sequence settings
of `history`, `warmup`, `prediction_full` and `contact`; `baseline_reference`
matches `reference`. `prediction` now trains with batch 16 for 5,000 epochs, so
neither baseline matches it: compare it with a copy of `baseline` set to the same
batch size and epochs.

### `faithful` and `baseline_faithful`

`faithful` follows the October 2026 formulation report's loss, weights, EMA update,
anchor, snippet horizon and numbers, but **not its descriptor**: it weights negatives by the
three reconstructed channels (gap, normal and tangential impulse), not by recorded force
magnitude. For the report's own descriptor use the force-based `bc_cami` path. `reference` already uses
every number the report gives (state weight 0.01, trajectory weight 1, EMA
momentum 0.995, ten future actions, Huber delta 0.1, contact temperature 0.05,
gamma 1, MSE-only BC, a 1024-d policy feature, a 1024-128-128 state projector, a
one-layer action LSTM of 256 and a 256-128-128 trajectory head), and `faithful`
sets them explicitly. It differs from `reference` in three things:

- the descriptor channels are averaged (weights 1/3, summing to 1) rather than
  summed, so that a pair differing by x in every channel gets the report's weight
  curve (0.0247, 0.5934 and 0.9798 for 0.05, 0.5 and 2.0) and not the weight of
  three times that distance;
- the rollout hidden state is reset every `rnn.horizon` = 11 steps, which is also
  the training sequence length (`reference` trains on 11 and resets every 10);
- every demonstration is used, with no validation split, so that the pair trains on
  the same data as a plain BC-RNN run.

Like `reference`, it anchors at the first policy feature, keeps the state weight
constant from epoch one, trains no prediction head, and uses batch size 16 and
seed 0.

`baseline_faithful` is its twin without CaMI: a BC-RNN run through the v3 adapter.
The two configs differ only in `algo.cami.enabled` and `train.dataset_keys`, and a
test keeps it that way. Compare `faithful` with `baseline_faithful`. A `train.py`
BC-RNN run is a looser comparison: besides any config differences, the v3 adapter
excludes padded steps from the BC loss, which a config comparison cannot show.
To see the config differences with an earlier `train.py` baseline:

```bash
python -m robomimic.scripts.compare_checkpoint_configs \
  your_train_py_baseline.pth baseline_faithful.pth
```

Match its `train.seq_length` and `algo.rnn.horizon` (11), seed (0) and
`train.max_grad_norm` where they differ, if you want it closer.

## Differences from the formulation report

What the v3 code does that the report (October 2026, "Continuous force-weighted
CaMI") does not say, per setting. `faithful` removes the rows marked *removed*;
the others are intended extensions or open points.

| Topic | Report | v3 |
|---|---|---|
| Descriptor | Scalar force magnitude `‖F_xyz‖ / sigma_F`, torque and direction excluded | Three simulator-reconstructed channels per offset: signed gap (m, capped at `--max-gap`, 2 cm by default), normal impulse and tangential impulse magnitude (N s). The impulses are state-conditioned simulator estimates, not recorded forces. Each channel is divided by its training standard deviation, as in the report |
| Distance | Mean over common valid offsets of the Huber penalty of one scalar difference | Same, summed over channels with `feature_weights`. Unit weights triple the distance; `faithful` uses 1/3 each |
| Weights | `W = (1 - exp(-D / tau_c))^gamma`, fixed, zero on the diagonal and without common offsets | Same, with the same defaults (`tau_c` 0.05, gamma 1, delta 0.1) |
| Loss | Full-batch mean of `-log(exp(S_ii) / (exp(S_ii) + sum W exp(S)))` for the state and trajectory branches against one shared EMA key | Same. The log-sum-exp form handles rows without negatives |
| Targets | EMA of the LSTM and head after the optimizer step, none for the state projector, untouched in validation | Same |
| Gradients | The policy receives BC and state-loss gradients | Same for `reference`, `faithful`, `history` and `warmup`; `prediction*` and `contact` also send the prediction loss into the policy features |
| Anchor | The first policy feature | `reference`, `batch32`, `batch64`, `faithful`: the first feature. `history`, `warmup`, `prediction*`, `contact`: the feature after five steps of context. The report leaves open how longer history enters the anchor (section 8) |
| State-loss weight | Constant 0.01 | Constant 0.01 in every variant (the earlier 100-epoch ramp is set to 0) |
| Missing channels | Not specified | A channel missing in either sample drops only itself; the rest are rescaled by `sum(w) / sum(observed w)` (`missing_features: "feature"`). `faithful` keeps the earlier rule, which drops the whole timestep |
| Prediction head | None (it names a prediction head as a separate probe, section 9) | `prediction*` and `contact` add a head (weight 0.01) whose input is the policy feature and a detached action embedding *(removed in `faithful`)* |
| Padding | Left to be specified | BC loss ignores padded steps. Snippets are packed to their real length, and anchors with no future action have zero CaMI loss |
| Rollout hidden state | Not specified | Reset every `rnn.horizon` steps. Variants train on 11 or 16 steps but reset every 10 *(aligned in `faithful`)* |
| Learning rate, validation | Not specified | Constant learning rate of 1e-4 in every variant. Except the `faithful` pair, they train on the `train` split and validate on `valid` |
| Policy | BC-RNN with one latent per sample | Deterministic BC-RNN (no GMM); the feature is the last actor layer |

For the report's own descriptor (force magnitude from `obs/force`), use the
existing force-based `bc_cami` path (`bc_cami_square.json`); it follows every
clause the report fixes.

The larger batches are real joint contrastive batches. Gradient accumulation
alone does not expand their negatives. Batch size 32 is the starting point;
64 is an explicit optional experiment. Random sampling does not guarantee a
particular contact-phase mixture. Check the effective-negative and weight
diagnostics before concluding that a larger batch provides more useful pairs.

Increasing batch or sequence length increases examples processed per optimizer
step. For a compute- or sample-budget comparison, match the relevant budget as
well as seeds and hyperparameters. Compare one change at a time, use matched
rollout starts for development, and evaluate selected checkpoints on fresh
starts. A single training seed does not establish robustness across training.

## Alignment and loss

Each stored `contact_state[t]` contains offsets `t+1,...,t+10`. With five context
steps, the adapter selects policy features at index 5 and pairs them with
`actions[:,6:16]` and `contact_state[:,5]`. A causal RNN can use observations
through index 5, but none after it. Padding masks exclude unavailable actions
and labels. Samples too close to a demonstration end can still contribute BC
loss while contributing zero CaMI loss.

Within a window, row k holds the gap in the recorded state before action t+k+1
and the impulse integrated while that action runs. So row k describes the same
step as `actions[t+k+1]`, the k-th action of the snippet. The anchor's own action
`actions[t]` is the BC target and is in neither the snippet nor the labels.

The objective is:

```text
L = L_IL
    + ramp(epoch) * lambda_state * L_state
    + lambda_traj * L_traj
    + ramp(epoch) * lambda_prediction * L_prediction
```

`ramp(epoch) = clamp((epoch - 1) / warmup_epochs, 0, 1)`. Every bundled variant
sets `warmup_epochs` to 0, which makes the ramp one from epoch 1. The default
coefficients are `lambda_state=0.01`, `lambda_traj=1.0`, and, when enabled,
`lambda_prediction=0.01`.
The trajectory loss trains the snippet encoder and its projection; its effect
on the policy is through the target used by state loss.

## Optional prediction head

The head receives causal policy features plus a detached embedding of the
demonstrated future action snippet. It predicts the normalized descriptor
sequence for those actions with masked Huber loss. Conditioning on the actions
distinguishes possible future interactions from the same observation. The head
updates itself and the policy features; this extra loss does not update the
snippet encoder. No future actions or descriptors enter the policy inputs.

The head, snippet encoder and contrastive projections are unused when selecting
rollout actions. Gap prediction is the first experiment because state-aligned
impulses are simulator estimates. `prediction_full` is a separate ablation of
the three channels, not an assumption that reconstructed impulses are accurate.
Training-only normalization and reconstruction provenance are saved in the
checkpoint config, using the existing descriptor checks.

## Gaps and impulses: the `contact` variant

The labels describe only the nut against the insertion peg (five geometry pairs,
`peg1` against `SquareNut_g0..g4`). Gripper-nut and nut-table contact never reach
them: "free" includes a nut held firmly in the gripper, and impulses measure only
the peg's load on the nut. Recording grasp force would need new pairs and a new
preparation.

An audit of `square_image_84_contactnets_state.hdf5` found the labels sound but
their use in the weights lossy:

- Gaps reproduce exactly from the recorded states, and re-simulating gives the
  stored impulses exactly. Recorded insertions contain overlaps down to -2 cm,
  where the robot presses the nut onto the peg at 40-90 N; MuJoCo 3.5 reproduces
  that motion, so these are real pressing events, not replay artifacts.
- An unknown gap used to drop its whole timestep, impulses included: 25% of the
  rows with a measured impulse never reached `W`.
- With weights `[1,1,1]` and `tau_c` 0.05, almost every pair of contact windows had
  `W` about 0.98. Impulses then pushed all contact windows apart equally instead of
  separating light contact from pressing. Tangential impulse nearly repeats normal
  impulse (correlation 0.93), and the deepest rows inflate the impulse scale.

Three changes address this.

1. **Per-channel masking.** `algo.cami.v3.missing_features` defaults to
   `"feature"`. A channel missing in either sample drops only itself; the
   observed channels are rescaled by `sum(w) / sum(observed w)`. A timestep counts
   when any weighted channel is observed in both samples. Without missing data
   both rules give the same distance. `faithful` keeps `"timestep"`.
2. **Per-pair gap rule.** A known CCD failure makes `mj_geomDistance` report 0
   for separated geoms. The gap used to be unknown whenever any of the five pairs
   failed. It now stays known when another pair is validly penetrating, because
   that pair is then the minimum. New preparations use this rule (`gap_rule` in
   the metadata). For an existing dataset, recompute the gap channel from the
   recorded states, with no stepping and impulses unchanged:

   ```bash
   python -m robomimic.scripts.recompute_cami_gaps \
     --dataset dataset/square_image_84_contactnets_state.hdf5 \
     --output dataset/square_image_84_contactnets_state_gapfix.hdf5
   ```

   On the Square dataset this lowers unknown gaps from 7.2% to 3.1% of rows.
   Every previously valid gap is unchanged, which the script checks.
3. **Descriptor transforms.** `algo.cami.v3.descriptor.gap_min` clips penetration
   below that gap (m). `impulse_log_scale` `s` replaces each impulse `I` by
   `s * log1p(I / s)`, which is linear below `s` N*s and logarithmic above it. Both
   default to null. When set, they are applied before scaling, and the scales are
   fitted on the transformed values.

The `contact` variant uses all three: weights 1/3 each, `gap_min` -0.005,
`impulse_log_scale` 0.1, prediction of all three channels, and the gap-fixed
dataset. On that dataset, the inspector reports:

- pairs with `W > 0.99`: 10% (34% for `prediction`);
- contact-vs-contact pairs: mean `W` 0.86 instead of about 0.98;
- free-vs-contact pairs: mean `W` 0.94.

These are properties of the labels; compare `contact` with `baseline` in
training before concluding it helps.

```bash
python -m robomimic.scripts.train_cami_v3 \
  --variant contact \
  --dataset dataset/square_image_84_contactnets_state_gapfix.hdf5 \
  --name square_cami_v3_contact \
  --wandb-project cami-square-v3 \
  --quiet
```

Training uses 16-step sequences, but rollout resets the hidden state every
`rnn.horizon` = 10 steps, so BC positions 10-15 train memory lengths rollout never
uses. The anchor (six observations) is within the rollout range, and the
`baseline` arms match, so comparisons stay fair.

## Porting CaMI

`robomimic/CAMI/cami_v3.py` has no dependency on BC, policy classes, recurrent
hidden states, or action sampling. It accepts:

| Input | Shape | Meaning |
|---|---|---|
| `features` | `[B,D]` | Host features at the declared decision time, with gradients attached |
| `expert_actions` | `[B,H,A]` | Demonstrated future action snippet |
| `lengths` | `[B]` | Number of valid future actions |
| `contact` | `[B,H,F]` | Corresponding privileged physical labels |
| `contact_mask` | `[B,H,F]` | Valid physical measurements |

The host keeps its original imitation objective and adds the returned losses.
The host is responsible for causal alignment, optimizer steps, and calling
`update_targets()` after a training update, never after validation. Physical
labels and EMA keys are detached; policy features must not be detached.

This release includes deterministic BC-RNN and BC-MLP adapters. Set
`algo.rnn.enabled=false` in a copied v3 JSON to select the MLP adapter. The MLP
does not gain temporal memory from the adapter. Diffusion, Transformer, GMM and
other learners still need their own feature adapters; they are not silently
mapped onto BC-RNN. The internal LSTM encodes training action snippets only.

## Check the labels before training

The state loss can only separate anchors that the descriptors separate. Look at
what the prepared labels give the weights, on any machine and without training:

```bash
python -m robomimic.scripts.inspect_cami_v3_dataset \
  --dataset dataset/square_image_84_contactnets_state.hdf5 \
  --variant prediction
```

It draws anchors the way the loader does (every training timestep starts a
sequence and the anchor sits `context_steps` in, so `reference` and
`prediction` give different pools), then reports:

- the reconstruction provenance and any transitions above the replay tolerance;
- per-channel statistics on training demos: valid fraction, fitted scale, the
  share of gaps sitting at the 2 cm cap or penetrating, the share of exactly
  zero impulses, and the 90th/99th percentiles in scale units;
- the share of anchor windows that are `free` (every gap at the cap, no impulse),
  `near` (anything between) and `contact` (a normal impulse above the binary
  threshold), the offset between each gap-closing and its impulse onset (over
  observed gap rows), and how many rows with a measured impulse have no valid gap.
  With `missing_features: "timestep"` (`faithful`), a pair distance drops a whole
  timestep when any weighted channel is missing, so those rows add no impulse to
  any weight while the gap weight is positive, and the report warns about them.
  The default `"feature"` keeps their impulses;
- for random batches of the configured size, the fraction of pairs with
  `W < 0.01` and `W > 0.99`, the mean `sum_j W_ij`, the share of anchors with
  any negative, and `init loss = mean_i log(1 + sum_j W_ij)`, the state loss an
  embedding with equal logits would score;
- a table of which window classes contribute the negative weight, and the same
  statistics under the configured weights divided by their sum (identical to a
  contact temperature multiplied by that sum), with only the gap channel and with
  only the two impulse channels, each keeping its configured weight.

Free-versus-free pairs have `W = 0` exactly, so `weight/near_zero_fraction` in
the W&B metrics is close to the square of the free share printed here. A state
loss that starts near `init loss` and falls far below it shows that the state
projector can tell an anchor's own future action key from the negatives with
nonzero weight. It does not show that the policy improves. The mass table says
what those negatives are: if almost all of it is contact-versus-free, the loss
mainly asks whether contact is coming, not how hard it will be.

To compare a v3 variant with another run, diff the saved `config.json` of both
runs. Compare v3 variants with the v3 `baseline` or `baseline_reference`, which
share their batch size, sequence length, seed and validation split, and compare
`faithful` with `baseline_faithful`. `baseline_reference` has a validation split
and resets the hidden state every 10 steps, so it is the wrong control for
`faithful`. Runs from `train.py` with another config also differ in those.

## Check how the nut ends up on the post

robosuite counts a Square episode as a success when the nut's centre is within 3 cm
of the post in x and y and less than 5 cm above the table, and the gripper has let
go. It never checks that the post passes through the hole or that the nut is flat,
so a nut jammed against the post or dragged into it counts. To see how often that
happens, replay the episodes of a demonstration file (it needs `states` and
`model_file`, and the simulator it was recorded in):

```bash
python -m robomimic.scripts.inspect_square_insertions \
  --dataset dataset/square_image_84_contactnets_state.hdf5 \
  --csv square_insertions.csv
```

That measures the demonstrations. To measure what a policy does, give it the
`rollouts.hdf5` that `rollout_best.py` writes next to its videos:

```bash
python -m robomimic.scripts.rollout_best --checkpoint <model_epoch_N.pth> \
  --n-rollouts 50 --output-dir square_rollouts
python -m robomimic.scripts.inspect_square_insertions \
  --dataset square_rollouts/rollouts.hdf5 --csv rollout_insertions.csv
```

`rollout_best.py --insertion-report` does both in one command; with `--n-trials` it
averages the insertion numbers over trials, and with `--wandb-project` it sends the
success rate, each trial's average, the insertion numbers and the videos to W&B; see
[Weights & Biases](cami_wandb.md).

`train_trials.py` runs the same evaluation for every trial, so each attempt writes
`<group>/evaluations/<trial>/<timestamp>/rollouts.hdf5` (a trial that was retried has
one folder per attempt). A rollout stops at the action after which the task counts
as done, so the state that satisfies the success rule is stored only in
`next_states`; the tool replays it as well, and it prints how many episodes it
replays a different success from the one the file records. The rule is evaluated on
the restored poses after a forward pass, so it can fire one step before the state
the rollout stopped at. Rollouts that failed before their first action have no
states and are listed as skipped. A rollout also ends at its first success, so
"seated at the last step" is not reported for it (the report says how many
episodes this leaves out); judge rollouts by the post in the hole and the tilt at
the success, which are measured at that step.

Other kinds of rollouts do not work as well. The videos that `train.py` writes hold
no simulator state, so they cannot be analyzed. A file written by
`run_trained_agent.py --dataset_path` has `states` and `model_file`, but only the
state before each action, with no `next_states` and no `success` attribute: the
state that satisfies the success rule is missing, so its successes can read as
never succeeded. Evaluate those checkpoints again with `rollout_best.py`.

For every episode it restores each recorded state and reads the poses and contacts
there, then prints how many successes have the post really inside the hole, the
nut's height above lying flat and its tilt at the first success, whether the nut
was lifted or slid along the table while held, whether it was above the post when
the post last came inside its outline before the success, how long it was in
contact with the post and with what force, and how often the fingers, hand and arm
touch the table. robosuite drops the nut onto the table from 6 cm at the start of
every episode, so lifting is measured from the moment the nut lands. Sliding is
counted from the first lift above `--lift-height` (from the landing if the nut is
never lifted), so shifting the nut while the fingers close on it does not count.
Forces come from the restored pose with no actuator force, so compare them between
episodes; they are not what the robot applied. Episodes get flags:
`success_without_post_in_hole`, `tilted_at_success` (past `--max-tilt-deg`),
`dragged_on_table`, `arrived_beside_post` and `long_contact_with_post`. When the
dataset records replay drift, it also compares the drift of flagged and unflagged
episodes.

Fingertips on the table while they grasp a handle that lies there are normal, so
the table counts are for comparing episodes and not a reason to reject them; the
hand and arm touching it are the interesting ones. The report also gives the step
at which both fingers first hold the nut, and how many steps and how hard the
fingertips pressed on the table before that, which separates a gentle touch on
the way down from pushing the nut around. A finger counts once however many of its
collision bodies touch the nut, and both fingers must touch it for a grasp. The CaMI
descriptors read only the nut and the post, so none of this table contact appears in
the labels.

To train on the episodes that do not look jammed, add them as a filter key and name
it in the config's `train.hdf5_filter_key`:

```bash
python -m robomimic.scripts.inspect_square_insertions --dataset <demos.hdf5> --write-mask no_jam
```

By default only the jam flags (`success_without_post_in_hole`,
`tilted_at_success`, `arrived_beside_post`) keep an episode out of the mask;
`--mask-flags` names others, or `all`. Look at the report first: with the default
limits a long press on the post, or a few steps of sliding after the lift, can flag
a large share of good demonstrations. The tool refuses to write a mask when
episodes were skipped, since they would fall out of it unjudged, and it decides the
mask before writing the CSV or JSON, so a refusal leaves no files behind.

The descriptor scale was fitted on the training split when the dataset was
prepared, and filtering afterwards does not refit it. The tool refuses to replace
a mask or an output file that exists. It replays every step, which takes a couple
of seconds per episode, and `--write-mask` needs every episode, so it refuses
`--limit`.

The thresholds are options (`--max-tilt-deg`, `--seat-tolerance`,
`--min-drag-steps`, `--min-post-contact-steps`, `--drag-speed`, `--lift-height`),
and the `--json` file records the ones it used under `thresholds`. A nut held by its handle hangs 10 to 20
degrees off flat, so the default tilt limit is 30. The `--json` file is strict JSON:
values that are not numbers are `null`. Nothing here reads a policy or a label, so
the report is the same for every variant.

## Diagnostics and customization

W&B/TensorBoard include raw losses, effective weights and:

- `CaMI_V3/gradient/BC_Policy_Norm`
- `CaMI_V3/gradient/State_Policy_Norm`
- `CaMI_V3/gradient/Weighted_State_to_BC_Ratio`
- `CaMI_V3/gradient/State_BC_Cosine`
- Corresponding prediction-gradient metrics when the head is enabled
- `CaMI_V3/weight/p10`, `p50`, `p90`, and near-zero / near-one fractions
- `CaMI_V3/state/effective_negatives` and `eligible_anchor_fraction`

Separate gradient measurements run every 100 optimizer updates by default,
before total-loss backward and clipping. They do not implement automatic loss
balancing or gradient surgery. Set `algo.cami.v3.gradient_diagnostics_every=0`
to disable them. Nearly unit negative weights can indicate weak contact-based
distinction; mostly zero weights can indicate little contrastive supervision.
Inspect these with rollout failures before changing contact temperature.

Copy a JSON to customize its batch size, seed, coefficients or settings. The
standard trainer also works directly:

```bash
python -m robomimic.scripts.train \
  --config robomimic/exps/templates/bc_cami_square_contactNetsv3_prediction.json \
  --dataset dataset/square_image_84_contactnets_state.hdf5 \
  --name square_cami_v3_custom \
  --wandb-project cami-square-v3 \
  --quiet
```

To generate fresh copies in another directory, use:

```bash
python -m robomimic.scripts.generate_cami_v3_configs --output-dir /tmp/cami_v3_configs
```

The generator refuses to replace any existing destination config.
