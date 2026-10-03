# Train one CaMI model and evaluate its best checkpoint over repeated trials

This follows `robomimic/scripts/run_trained_agent_multi_eval.py`: train **one**
model, then evaluate its best checkpoint several times. A *rollout* is one
episode. A *trial* is one evaluation of `--rollouts-per-trial` rollouts.

Use the same environment and Hisham's robosuite fork that already run your
single-seed CaMI training. The launcher starts a fresh Python process for the
training run and for each trial, one after another. No robosuite edits are
required.

## Run

From the robomimic repository root:

```bash
wandb login
python -m robomimic.scripts.train_trials \
  --config robomimic/exps/templates/bc_cami_square.json \
  --dataset dataset/square_image_84_with_force.hdf5 \
  --n-trials 10 \
  --rollouts-per-trial 50 \
  --epochs 2000 \
  --group square-cami-c \
  --wandb-project cami
```

1. **Train one model** with the template's `train.seed` (1), or `--seed N`. The
   Square template defines an epoch as 500 gradient steps, so this is 2,000
   epochs of 500 steps. It evaluates 50 episodes every 50 epochs and at the
   final epoch, and keeps best-success checkpoints. The template's periodic
   checkpoints and rolling `last.pth`/`last_bak.pth` remain available.
2. **Choose the checkpoint once.** It is the one with the highest recorded
   training-evaluation success rate; ties prefer the earlier epoch. Every trial
   evaluates these same weights, and the launcher stops if a trial reports any
   others.
3. **Run the trials.** Rollout *j* of trial *t* (both counted from 0) uses seed
   `eval_seed + t * rollouts_per_trial + j`. With the defaults that is 10000 to
   10499, all different, so no two rollouts share a start state and any one of
   them can be reproduced. Python, NumPy, PyTorch, CUDA, and Hisham's robosuite
   `default_rng` are seeded for each rollout. GPU kernels can still be
   nondeterministic. Each trial saves every trajectory, videos of the
   successful rollouts, and a summary.
4. **Write the results:** per-epoch curves, trial and rollout CSVs, summary
   statistics, a manifest, and the W&B runs. The evaluation run in W&B is
   updated after each trial, so you can follow the trials as they finish.

By default the videos use `--camera-names agentview` and `--video-skip 5` at
`--fps 4`, which plays at natural speed for 20 Hz control. See "Videos" below
for 60 fps, failure videos, and the joined `SUCCESSFUL_ALL.mp4` and
`FAILED_ALL.mp4`. Trajectories of all rollouts are saved either way.

## Videos

Each successful rollout gets its own video, `successful_rollout_<NNN>_seed_<S>.mp4`.
With `--keep-failures`, failed rollouts and simulator errors get one too
(`failed_rollout_...`, `errors_rollout_...`). Each trial folder also has one
video per outcome, made by joining those clips in rollout order:

- `SUCCESSFUL_ALL.mp4`: every successful rollout of the trial.
- `FAILED_ALL.mp4`: every failed rollout and simulator error of the trial. It
  exists only when failure videos are kept.

`SUCCESSFUL_ALL.csv` and `FAILED_ALL.csv` sit next to them. Each row gives the
clip number, rollout, seed, outcome, file name, and the start time and duration of
that clip in the joined video, so you can tell which rollout is on screen. The
joined videos are the clips copied one after another without re-encoding, and
the frame count of each is checked against its clips. If joining fails, a message
is printed and the trial carries on with its individual videos. `--no-stitch`
skips the joining; the joined videos take as much disk space as the clips
themselves.

For every control step at 60 fps, with failures kept:

```bash
python -m robomimic.scripts.train_trials \
  ... \
  --video-skip 1 --fps 60 --keep-failures
```

`--video-skip N` records every Nth control step and `--fps` is how fast the frames
play back. The environment steps at 20 Hz and a frame is rendered after each
step, so a video has at most 20 frames per simulated second. Playing every step at
60 fps therefore runs **3 times faster than the simulation**; `--fps 20` plays it
in real time, and the default `--video-skip 5 --fps 4` does too, with fewer
frames. Each trial prints which one you have, for example `Videos: every 1
control step(s) at 60 fps = 3x real time`. Rendering every step instead of every
fifth takes noticeably longer per trial.

Videos are settings of the experiment: a group that was started with other video
options cannot be resumed with these (use a new `--group`).

## Evaluate a model you already have

To run the same trials on a model you trained earlier, skip training and give
either the checkpoint file or its run folder:

```bash
# One checkpoint, for example the epoch-1000 model that reached 90% in training
python -m robomimic.scripts.train_trials \
  --checkpoint trained_models/<group>/trials/trial_02_seed_2/<timestamp>/models/model_epoch_1000_<dataset>_success_0.9.pth \
  --n-trials 10 --rollouts-per-trial 50 \
  --group square-cami-trial2-best \
  --wandb-project cami

# Or let the launcher pick the best-success checkpoint of a run
python -m robomimic.scripts.train_trials \
  --run-dir trained_models/<group>/trials/trial_02_seed_2 \
  --n-trials 10 --group square-cami-trial2-best --wandb-project cami
```

`--run-dir` takes an experiment folder (its latest timestamped run), a timestamped
run, or a `models` folder. It chooses the highest recorded training success rate,
ties preferring the earlier epoch. List a run's candidates with
`ls <run>/models | grep success`.

Nothing is trained, so `--config`, `--dataset`, `--epochs`, `--seed`,
`--rollouts`, and `--rollout-rate` are rejected, and `--wandb-project` is
required unless you pass `--wandb-mode disabled` (there is no template to take
it from). Everything else works as above: the same seed blocks and outputs,
`--resume`, `--aggregate-only`, `--keep-failures`, `--no-stitch`,
`--camera-names`, `--video-skip`, and `--fps`. `--horizon N` overrides the horizon stored in the checkpoint
(400 for Square, 700 for Tool Hang) for the trials only. The W&B group gets just
the `<group>-eval` run, because no training run is created.

What the report contains depends on what sits next to the checkpoint:

- The epoch and the in-training success rate come from the trainer's file name,
  `model_epoch_<N>_<dataset>_success_<rate>.pth`. A renamed file still works, but
  its epoch and rate are then unknown and left empty.
- A run made by this repository's trainer keeps `logs/metrics.jsonl` next to
  `models/`. When it covers the checkpoint's epoch, the report also has
  `curves.csv` and the `Final` and `BestCheckpoint` values. `last.pth` in the run
  folder is matched to that journal too, but a copy under any other name is not,
  because it could belong to a different run. Without a usable journal the report
  has the `Evaluation` statistics (and `Selection/Epoch` when the epoch is known).
  Runs that only wrote TensorBoard logs have no journal.

The checkpoint has to load with this code and a robosuite that can build its
environment. Its in-training success rate (for example 0.9) is the best of many
evaluations and is optimistic. The trials are a fresh measurement of the same
weights, so expect a lower number. A model picked as the best of several seeds is
also better than the method's expected result; report the other seeds too.

## Which numbers to report

| Summary field | What it is |
| --- | --- |
| `Evaluation/Success_Rate` | Fresh evaluation of the selected checkpoint. Each trial contributes the mean of its own rollouts, including failures and simulator errors in the denominator. The statistics are over trials (n = number of trials), so selection of the checkpoint cannot inflate them. |
| `Final/Rollout/Success_Rate/<dataset>/max` | The highest in-training success rate over the whole run. This is the "maximum success rate over training" metric that the robomimic study paper uses for baselines (see `docs/tutorials/viewing_results.md`). It is chosen from the same evaluations it is measured on, so it is optimistic. One value, n = 1. |
| `BestCheckpoint/Rollout/Success_Rate/<dataset>` | In-training success at the selected epoch. Same optimism. n = 1. |

Use the dataset key shown in your dashboard, for example
`square_image_84_with_force`. `Final` and `BestCheckpoint` metrics are single
measurements from the one training run, so their SD and SE are empty and n is 1.

For ten finite trial measurements m_i, mean = sum(m_i)/10, sample
SD = sqrt(sum((m_i - mean)^2)/9), and SE = SD/sqrt(10). Missing or non-finite
values are excluded and the lower n stays visible.

**What the spread means.** The trials repeat the evaluation of one trained
model, so the SD shows how much the success rate moves with the start states.
It does not include variation between training runs: another seed could give a
different model. Treat a difference between two methods trained once each with
that in mind. See "More seeds and the BC-RNN baseline" below.

## How this differs from `run_trained_agent_multi_eval.py`

- The CSV columns are the same, so `trial_results.csv` can be concatenated with
  that script's output and read by `CAMI/cami_eval_plots_dist.py`.
- That script seeds NumPy and PyTorch once per trial. Robomimic's environment
  wrappers define no `seed()` method, so it cannot reseed a robosuite
  `default_rng`. Here every rollout has its own seed and its own robosuite
  generator state, so rollouts are reproducible one by one.
- Trial and rollout numbers in `trial_results.csv` and `rollout_results.csv`
  start at 1 for trials (matching the `trial_NN` folders) and 0 for rollouts
  (matching `demo_N` in `rollouts.hdf5`). The within-trial `return_std` and
  `horizon_std` are population SDs, as in that script. The across-trial SD in
  the summary is the sample SD (n minus 1).
- This launcher also trains the model, keeps all trajectories and success
  videos, and writes a manifest and the W&B runs.

To plot with Hisham's scripts, name the group like his model names, for example
`square_100_percent_cami` and `square_100_percent_baseline`. The `model` column
is the group name, and that script reads the method and data fraction from it.

## W&B

The group holds two runs:

- `<group>-train` (`job_type=training`): the trainer's per-epoch scalars with
  x = epoch, including `Rollout/Success_Rate/<dataset>` and its `/mean`, `/max`,
  `/min`, and `/std` variants. The config has `train_seed`. This is where the
  per-epoch curves live.
- `<group>-eval` (`job_type=evaluation`): opened before the first trial and
  updated as each trial completes, with one point per trial for each metric as
  `Trial/<metric>` (x = `trial`) and, beside it, `Running/<metric>/mean` and
  `/se` over the trials so far, which shows the estimate settling as trials
  come in. After the last trial it gets the fields
  `Summary/<scope>/<metric>/mean`, `/std`, `/se`, and `/n`, the tables
  `metric_summary` and `trial_summary`, and an artifact with the CSV and JSON
  results. Its config records the checkpoint, its epoch, the training run's ID,
  and the video settings.

The evaluation run's ID is kept in `manifest.json` (`wandb_eval_id`), so `--resume`
continues the same run instead of starting another, and first logs any trial
that was finished before W&B was following it. A trial that fails marks the run
as failed until you resume. W&B problems never stop the evaluation: one warning
is printed, the trials carry on, and the complete results are uploaded to a new
run at the end. Videos are not uploaded; they stay in the evaluation folders.

| Scope | Which value it uses |
| --- | --- |
| `Final` | Metric at the last epoch |
| `BestCheckpoint` | Metric at the selected checkpoint's epoch |
| `Evaluation` | Mean over each trial's rollouts; statistics over trials |
| `Selection/Epoch` | Epoch of the selected checkpoint |

Results are written to CSV and JSON before W&B upload. No W&B history sampling is
used to compute them.

## Saved outputs

Under `trained_models/<group>/`:

```text
manifest.json                         seed, chosen checkpoint, status and results of every trial
config.json                           the exact training config
training/<group>-train/<timestamp>/   checkpoints, last.pth, logs/metrics.jsonl (exact per-epoch scalars)
evaluations/trial_NN/<timestamp>/     rollouts.hdf5, summary.json, one video per kept rollout,
                                      SUCCESSFUL_ALL.mp4/.csv, FAILED_ALL.mp4/.csv
results/curves.csv                    epoch, metric, value (exact, nothing averaged)
results/trial_results.csv             one row per trial (multi-eval columns)
results/rollout_results.csv           one row per rollout, with its seed and outcome
results/summary.csv, summary.json     mean, sample SD, SE, and n for every metric
```

`rollouts.hdf5` holds actions, rewards, dones, simulator states before and after
each action, and model XML for every rollout. The masks `successful`, `failed`,
and `errors` select outcomes. Replay them with
`robomimic.scripts.playback_dataset --filter_key successful`; see
`docs/tutorials/cami_wandb.md`.

## Resume, rebuild, or test briefly

If the training run or a trial fails, fix the cause and rerun the same command
with `--resume`. Finished work is kept, so completed trials are not run again.
Running the command a second time without `--resume` stops with a message,
because its group folder already exists: add `--resume` to continue it, or use
another `--group` (or delete the folder) to start over. Resuming needs the same
options as the first attempt, so a changed `--n-trials` or `--run-dir` is refused.
If a trial fails again, the child's traceback is printed just above the
"Stopped at" line. An interrupted training run with a
`last.pth` resumes with the trainer's optimizer and checkpoint resume (it does
not restore the complete RNG state of an uninterrupted run), and the same W&B
training run continues, as does the evaluation run. A trial that failed is evaluated again into a new
folder, so earlier files are preserved. The saved plan must match the resumed
command. A manifest from the earlier ten-seed launcher is rejected; use a new
`--group`.

To regenerate or upload the results of a completed group without running
anything:

```bash
python -m robomimic.scripts.train_trials \
  --group square-cami-c \
  --aggregate-only \
  --wandb-project cami
```

Use `--output-dir` again if you used a custom output directory. Uploading again
creates a separate W&B evaluation run that holds every trial.

A quick check of the actual dataset and simulation pipeline:

```bash
python -m robomimic.scripts.train_trials \
  --config robomimic/exps/templates/bc_cami_square.json \
  --dataset dataset/square_image_84_with_force.hdf5 \
  --n-trials 2 --debug --group cami-smoke --wandb-project cami
```

Debug uses two epochs, three gradient steps per epoch, and two short rollouts in
each trial. Use a new group for the full experiment. For offline logging, add
`--wandb-mode offline`, then use the usual `wandb sync` on the saved W&B folders.
For local-only checks, add `--wandb-mode disabled`.

## More seeds and the BC-RNN baseline

Run the launcher again with another `--seed` and a new `--group` to add a
training seed. Each group is a separate model, and the CSV files can be
concatenated.

For a baseline, run the plain BC-RNN config through the same launcher with the
same `--eval-seed`. Both policies then start every rollout from the same seeds.
No Square `bc` template is included. Copy `bc_cami_square.json`, set `algo_name`
to `bc`, remove the `cami` block and the extra optimizers, and set
`train.dataset_keys` to `["actions"]`, as `tests/test_rollout_best.py` does.

## Validation

```bash
python -m unittest discover -s tests -p test_train_trials.py -v
```

These checks exercise the single training job and the repeated trials with
synthetic child outputs: the shared checkpoint, disjoint seeds, exact
statistics, failure and resume handling, the multi-eval CSV columns, the W&B
calls (including the live per-trial logging and its failure handling), and the
evaluation of an existing checkpoint or run. They do not run neural-network
training or MuJoCo. The existing `tests/test_rollout_best.py` covers real
checkpoint restore and video export, including the 60 fps joined videos, which it
encodes and decodes for real, in an environment containing the full simulation
and training dependencies.
