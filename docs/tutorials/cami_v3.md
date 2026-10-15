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

The recommended starting variant uses batch size 32, five preceding observation
steps before the CaMI decision time, ten future action/contact steps, and a
100-epoch state-loss warmup. The prediction head is disabled. Training remains
2,000 epochs, with 50 rollout episodes every 50 epochs, validation enabled,
learning rates of 1e-4 halved after epoch 1,000, and best-success checkpoint saving.
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

## Separate variants

| `--variant` | Batch | Sequence | Context before anchor | Warmup | Prediction |
|---|---:|---:|---:|---:|---|
| `baseline_reference` | 16 | 11 | N/A | N/A | No CaMI modules |
| `reference` | 16 | 11 | 0 | 0 | Off |
| `batch32` | 32 | 11 | 0 | 0 | Off |
| `batch64` | 64 | 11 | 0 | 0 | Off |
| `history` | 32 | 16 | 5 | 0 | Off |
| `warmup` | 32 | 16 | 5 | 100 | Off |
| `prediction` | 32 | 16 | 5 | 100 | Future signed gap |
| `prediction_full` | 32 | 16 | 5 | 100 | Future gap and both impulses |
| `baseline` | 32 | 16 | N/A | N/A | No CaMI modules |

`reference` is a v3 control with the old batch size, anchor time and loss weights.
It is not claimed to reproduce v2 exactly: v3 groups the auxiliary parameters
under one optimizer and clips that group together. The original v2 remains
available for an exact comparison with previously trained models.

The baseline uses the same deterministic BC-RNN policy, image augmentation,
observations and BC objective as its matching v3 setting. It excludes padded
steps from BC loss, constructs no CaMI network or auxiliary optimizer, and
requires no contact labels. `baseline` matches the batch and sequence settings
of `history`, `warmup` and `prediction`; `baseline_reference` matches `reference`.

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

## Alignment and warmup

Each stored `contact_state[t]` contains offsets `t+1,...,t+10`. With five context
steps, the adapter selects policy features at index 5 and pairs them with
`actions[:,6:16]` and `contact_state[:,5]`. A causal RNN can use observations
through index 5, but none after it. Padding masks exclude unavailable actions
and labels. Samples too close to a demonstration end can still contribute BC
loss while contributing zero CaMI loss.

The objective is:

```text
L = L_IL
    + ramp(epoch) * lambda_state * L_state
    + lambda_traj * L_traj
    + ramp(epoch) * lambda_prediction * L_prediction
```

`ramp(epoch) = clamp((epoch - 1) / 100, 0, 1)` for the warmup variants. Epoch 1
therefore trains BC and the action-snippet representation; epoch 101 uses the
full requested state and prediction weights. The default coefficients are
`lambda_state=0.01`, `lambda_traj=1.0`, and, when enabled,
`lambda_prediction=0.01`. Disabling warmup sets the ramp to one immediately.
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
