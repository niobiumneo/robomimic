# Ten CaMI training runs and mean metrics

Use the same environment and Hisham's robosuite fork that already run your
single-seed CaMI training. The new launcher starts a fresh Python process for
each seed, sequentially. No robosuite edits are required.

## Run

From the robomimic repository root:

```bash
wandb login
python -m robomimic.scripts.train_trials \
  --config robomimic/exps/templates/bc_cami_square.json \
  --dataset dataset/square_image_84_with_force.hdf5 \
  --n-trials 10 \
  --epochs 2000 \
  --seed-start 1 \
  --group square-cami-c-10-seeds \
  --wandb-project cami
```

This starts seeds 1 through 10 with the same dataset, split, and hyperparameters.
The Square template defines an epoch as 500 gradient steps. Each trial therefore
performs 2,000 of those epochs. The launcher uses absolute output paths under
`trained_models/square-cami-c-10-seeds/` by default.

The template evaluates 50 episodes every 50 epochs, and the trainer also
evaluates the final epoch. The launcher enables saving best-success checkpoints.
The template's periodic checkpoints and rolling `last.pth`/`last_bak.pth` also
remain available. Training checkpoint files are weights; rollout paths contain
states, actions, rewards, and episode outcomes.

After each training run, `rollout_best.py` selects the checkpoint with the highest
**recorded training-evaluation success rate**. Ties prefer the earlier epoch.
It evaluates that checkpoint on 50 fresh episodes, seeds 10000 through 10049,
using the same seeds for every model. It saves all trajectories in `rollouts.hdf5`
and saves successful videos. Add `--keep-failures` to keep failure/error videos,
too. These are new evaluations of the best weights. They do not reconstruct
the original training-evaluation paths, which the old training loop did not save.

Python, NumPy, PyTorch, and Hisham's robosuite `default_rng` are seeded. GPU
kernels can still be nondeterministic. Resuming uses the existing trainer's
optimizer/checkpoint resume behavior; it does not restore the complete RNG state
of an uninterrupted run.

## W&B

There are ten `job_type=training` runs in the named group and one additional
`job_type=aggregate` run named `square-cami-c-10-seeds-mean` after completion.
Trial configs include `trial_seed`, `trial_index`, and `trial_group`.

For a mean learning curve in the W&B workspace:

1. Filter to the named group and `job_type=training`.
2. Set X to `epoch` and Y to a raw scalar metric, for example
   `Rollout/Success_Rate/square_image_84_with_force`.
3. Enable grouping by `group`, set aggregation to Mean, and set the shaded
   range to Std Dev or Std Err. Turn off “Use only latest run per group”.

Use the actual dataset key displayed in your dashboard. Existing metric suffixes
such as `/mean` are running statistics *within one training run*. The raw
`Rollout/Success_Rate/<dataset>` measures that epoch's evaluation success.

The additional mean run logs every scalar metric captured by the trainer:

| Prefix | Meaning |
| --- | --- |
| `Mean/<metric>` | Mean across available training seeds at that exact epoch |
| `Std/<metric>` | Sample SD across seeds, denominator n minus 1 |
| `SE/<metric>` | SD divided by sqrt(n) |
| `N/<metric>` | Number of finite observations contributing to that point |

Metrics are aligned by epoch without interpolation. Rollout statistics only
exist at evaluation epochs. Non-finite/missing observations are excluded and
their lower n remains visible. An incomplete planned trial stops the experiment;
the launcher does not silently label a partial group a ten-run result.

For ten finite measurements m_i, mean = sum(m_i)/10,
sample SD = sqrt(sum((m_i - mean)^2)/9), and SE = SD/sqrt(10).

The mean run also has `metric_summary` and `trial_summary` tables. Its summary
fields use `Summary/<scope>/<metric>/mean`, `/std`, `/se`, and `/n`:

| Scope | Which value each seed contributes |
| --- | --- |
| `Final` | Metric at epoch 2000 |
| `BestCheckpoint` | Metric at that seed's selected checkpoint epoch |
| `Evaluation` | Mean over the 50 fresh episodes of the selected checkpoint |
| `Selection/Epoch` | Epoch of the selected checkpoint |

For the best models' overall task performance, look at
`Summary/Evaluation/Success_Rate/mean` and `/std`. Here n is 10 independently
trained models. Each model contributes its own 50-episode mean, including failed
episodes and simulator errors in the denominator. Those 500 episodes are not
treated as 500 independent training trials. Training, BC/CaMI losses, rollout
returns/success/horizon, timing, RAM, and other logged scalar metrics are captured
automatically. Validation metrics are included if validation is enabled in the
template. The current Square template has validation disabled.

## Saved outputs

Under `trained_models/<group>/`:

- `manifest.json`: seeds, trial status, chosen checkpoints, and evaluation paths.
- `configs/`: the exact input config for each training seed.
- `trials/trial_XX_seed_N/<timestamp>/`: checkpoints and scalar `logs/metrics.jsonl`.
- `evaluations/trial_XX_seed_N/<timestamp>/`: HDF5 paths, videos, and episode summary.
- `aggregate/curves.csv`: exact mean, sample SD, SE, and n for every metric and epoch.
- `aggregate/summary.csv` and `summary.json`: final, selected-checkpoint, and fresh evaluation metrics.

CSV/JSON results are saved before W&B upload. The aggregate run also uploads
these as an evaluation artifact. No W&B history sampling is used to compute them.

## Resume, repeat aggregation, or test briefly

If a trial fails, fix the cause and rerun the same training command with
`--resume`. Completed trials are skipped. A trial with an existing `last.pth`
resumes training. A trial whose training completed retries only its evaluation.
The saved plan must match the resumed command. Each evaluation retry writes a
new folder, preserving earlier files.

To regenerate or upload the completed group's mean without training again:

```bash
python -m robomimic.scripts.train_trials \
  --group square-cami-c-10-seeds \
  --aggregate-only \
  --wandb-project cami
```

Use `--output-dir` again if you used a custom output directory. Aggregating
again creates a separate W&B aggregate run; filter `job_type=training` when
grouping the original ten learning curves.

A quick check of the actual dataset and simulation pipeline:

```bash
python -m robomimic.scripts.train_trials \
  --config robomimic/exps/templates/bc_cami_square.json \
  --dataset dataset/square_image_84_with_force.hdf5 \
  --n-trials 2 --debug --group cami-smoke --wandb-project cami
```

Debug uses two epochs, three gradient steps per epoch, and two short rollouts.
Use a new group for the full experiment. For offline logging, add
`--wandb-mode offline`, then use the usual `wandb sync` on the saved W&B folders.
For local-only checks, add `--wandb-mode disabled`.

## Validation

```bash
python -m unittest discover -s tests -p test_train_trials.py -v
```

These checks exercise the ten independent 2,000-epoch job configurations with
synthetic child outputs, exact statistics, failure handling/resume, stable RNG
references, and W&B logging. They do not run neural-network training or MuJoCo.
The existing `tests/test_rollout_best.py` covers real checkpoint restore and video
export in an environment containing the full simulation/training dependencies.

W&B grouping reference:
https://docs.wandb.ai/models/app/features/panels/line-plot/reference
