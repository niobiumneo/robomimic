# Track CaMI training and evaluation in Weights & Biases

Run these commands from the repository root on `contact-state`, using the
same Python environment as training. W&B is optional and is disabled in the
shared templates until you enable it.

## Install and authenticate

```bash
python -m pip install -e '.[wandb]'
wandb login
```

Use the API key from your W&B account in the terminal login prompt. Keep it
out of JSON configs, committed Python files, and chat messages. See the
[W&B login documentation](https://docs.wandb.ai/models/ref/cli/wandb-login).

Choose the team or account that owns your project:

```bash
export WANDB_ENTITY="your-wandb-team-or-account"
export WANDB_MODE=online
```

Replace the example entity with the exact slug from your W&B project URL:
`https://wandb.ai/ENTITY/PROJECT`. You need write access there. If you omit
the entity, W&B uses its configured default. Existing `WANDB_ENTITY` and
`WANDB_API_KEY` entries in `robomimic/macros_private.py` remain supported as
fallbacks; a private macros file is not required for W&B.

## Check the Square run

For the robosuite 1.5.2 Square dataset, install the simulator versions recorded
in this branch's `environment.yml`. Robosuite's own dependency allows newer
MuJoCo releases, which can fail its joint-type check during environment setup:

```bash
python -m pip install -r requirements-cami-sim.txt
```

For an existing `dataset/square_image_84_with_force.hdf5`:

```bash
export MUJOCO_GL=egl

python -m robomimic.scripts.train \
  --config robomimic/exps/templates/bc_cami_square.json \
  --dataset dataset/square_image_84_with_force.hdf5 \
  --name square_cami_wandb_check \
  --wandb --wandb-project cami-contact-state \
  --quiet --debug
```

This runs two short epochs and short rollouts. The logger prints the run URL
after initialization. Metrics arrive at epoch boundaries, so the graphs can
remain empty during the first epoch. A failed simulator setup occurs before
the logger is created and will not produce a W&B run.

The full training command is:

```bash
python -m robomimic.scripts.train \
  --config robomimic/exps/templates/bc_cami_square.json \
  --dataset dataset/square_image_84_with_force.hdf5 \
  --name square_cami_continuous \
  --wandb --wandb-project cami-contact-state \
  --quiet
```

`--wandb-project` also enables logging by itself. JSON users can set
`experiment.logging.log_wandb=true` and
`experiment.logging.wandb_proj_name="cami-contact-state"`. The CLI overrides
these settings. `--name` selects the experiment output directory; this branch
generates the W&B display name from the algorithm, dataset, and timestamp.

## Evaluate an existing checkpoint over 50 rollouts

Use this command for a single evaluation of a model that is already trained.
Replace the checkpoint path with the actual `.pth` file from your seed-42 run:

```bash
python -m robomimic.scripts.run_trained_agent_multi_eval \
  --agents /full/path/to/your_checkpoint.pth \
  --agent_names CaMI \
  --n_trials 1 \
  --rollouts_per_trial 50 \
  --horizon 400 \
  --seed 42 \
  --results_dir results/cami_best_seed42_50 \
  --wandb-project cami-contact-state \
  --wandb-name square-best-eval50-seed42
```

This restores one checkpoint and runs exactly 50 episodes without training.
Use a 700-step horizon for Tool Hang. The evaluation gets a separate W&B run
with `job_type=evaluation`; its config records the checkpoint, evaluation seed,
rollout budget, resolved horizon, and seed schedule.

| W&B metric | Meaning |
| --- | --- |
| `Rollout/Success_Rate` | Success (0 or 1) of each episode, with rollout number on the x-axis |
| `Running/Success_Rate` | Fraction of successful episodes so far; the last point is the final result |
| `Rollout/Return`, `Rollout/Horizon` | Reward and duration of each episode |
| `Trial/Success_Rate` | Mean success rate of the complete 50-episode batch |
| `Evaluation/Success_Rate` | Final success rate in the run summary |
| `Evaluation/Num_Success`, `Evaluation/Num_Rollouts` | Successful and completed episode counts |

The run also contains `trial_results` and `rollout_results` tables and an
`evaluation` artifact containing the result CSVs. The same CSVs remain in
`--results_dir`, including the existing per-model/environment folders.

### Per-trial maximums and variability

`run_trained_agent_multi_eval.py` logs the following when each trial completes.
Plot these metrics against `trial` to compare repeated evaluations of the same
checkpoint. This happens in the evaluation loop automatically when W&B is
enabled; no separate statistics command is needed for new evaluations.
Existing CSV columns are retained, with the new statistics added.
Per-model/environment CSVs are saved after every completed trial, so completed
trial results remain available if a later trial stops.

| W&B metric | Meaning |
| --- | --- |
| `Trial/Success_Rate` | Final success fraction over the complete trial budget |
| `Trial/Peak_Running_Success_Rate` | Maximum running success fraction within that trial |
| `Trial/Peak_Running_Num_Rollouts`, `Trial/Peak_Running_Num_Success` | Rollouts and successes at the earliest occurrence of that peak |
| `Trial/Return_Min`, `Trial/Return_Max`, `Trial/Return_Std` | Minimum, maximum and population SD of rollout returns |
| `Trial/Horizon_Min`, `Trial/Horizon_Max`, `Trial/Horizon_Std` | Minimum, maximum and population SD of rollout lengths |

The peak starts at the first rollout and resets for each trial. A successful
first rollout therefore gives a peak of 100% after one rollout. To compare
performance over a fixed budget, use the final trial rates and
`Evaluation/Success_Rate_Max_Over_Trials`, the best completed trial rate.
The run Summary also contains the minimum, mean, population SD, sample SD and
SE over completed trial rates, plus `Evaluation/Best_Trial` and its seed.
Sample SD and SE are absent for a single trial. W&B summary reducers retain
min/max/mean for the main `Trial/` metrics as each trial finishes.

While a trial is in progress, the evaluator logs `TrialRunning/Success_Rate`,
`TrialRunning/Peak_Running_Success_Rate`, the counts at the peak, and
`TrialRunning/Return_*` and `TrialRunning/Horizon_*` statistics after every
rollout. These series use the global `rollout` x-axis and reset their accumulated
values at each trial boundary; `trial` and `episode_in_trial` identify the
active trial and its local progress. The completed-trial summary, including
`Evaluation/Success_Rate_Max_Over_Trials` and `Evaluation/Best_Trial`, is updated
immediately after each trial completes, so it is available before the entire
evaluation finishes. `Trial/*` metrics always use the full trial budget.

An already-running evaluator keeps its loaded logger code. After it finishes,
upload the additional statistics from its existing rollout CSV without running
the policy or simulator again:

```bash
python -m robomimic.scripts.log_eval_trial_statistics \
  --csv results/cami_best_eval_seed0_50_videos/CaMI/checkpoint_env/rollout_results.csv \
  --rollouts-per-trial 50 \
  --wandb-project cami-contact-state \
  --wandb-name square-best-eval50-seed0-trial-stats
```

This creates a separate W&B analysis run with charts, a trial table and result
artifacts. Local output is written beside the input CSV in `trial_statistics/`.
Use `--wandb-mode disabled` for local-only calculations or `offline` for later
sync. The uploader accepts either evaluator's rollout CSV; it recognizes
zero-based multi-eval trials and one-based `train_trials` trials. If a CSV
contains only later trials and numbering is ambiguous, set
`--trial-index-base 0` or `--trial-index-base 1`. Output CSV trial IDs are
normalized to zero-based; W&B displays the corresponding one-based trial IDs.
Incomplete, duplicated or mixed-model/environment/checkpoint inputs are
rejected before anything is logged.

The existing root-level `stitch_rollout_videos.py` combines the recorded clips
across trials with separate success/failure videos and CSV timestamp indexes:

```bash
python stitch_rollout_videos.py \
  results/cami_best_eval_seed0_50_videos/CaMI/checkpoint_env/videos
```

Compilations go in `videos/stitched/`; `--output-dir` chooses another location.
Use `--overwrite` to replace an earlier compilation. Only labeled rollout
clips are included; existing compilations are skipped.

`--wandb-project` enables tracking; `--wandb` enables it with the default
`cami-contact-state` project. `--wandb-entity` selects an account/team and
`--wandb-group` groups comparisons. Multiple checkpoints or environments create
separate W&B runs with the same metric names. Use `--wandb-mode offline` to
record locally and sync later, or `--wandb-mode disabled` for CSV-only evaluation.
Missing dependencies or initialization errors are reported before the rollout
batch. If streaming later fails, evaluation continues and preserves its CSVs.

The seed handling follows the original evaluator: NumPy and PyTorch are seeded
once before each trial with `seed + trial_index`. With one trial and `--seed 42`,
the random streams advance across the 50 episodes; they are not reset to 42
for every episode. W&B calls preserve the global RNG states. The original
best-effort environment seeding is retained, so a simulator using an independent
generator without a wrapper `seed()` method may still randomize placements.
Matching a training seed does not reconstruct the episodes that produced a
training-time checkpoint score. Use the same simulator and evaluation settings
for model comparisons, and report the newly measured success rate.

### Record the same evaluation episodes

Add `--save-videos` to record every successful and failed episode in this same
evaluation loop. Add `--wandb-videos` to also upload each clip under
`Rollout/Video` in W&B; that flag enables recording too. For a seed-0 Square
evaluation with every simulation step recorded:

```bash
python -m robomimic.scripts.run_trained_agent_multi_eval \
  --agents /full/path/to/your_checkpoint.pth \
  --agent_names CaMI \
  --n_trials 1 --rollouts_per_trial 50 --horizon 400 --seed 0 \
  --results_dir results/cami_best_eval_seed0_50_videos \
  --save-videos --video_skip 1 \
  --wandb-project cami-contact-state \
  --wandb-name square-best-eval50-seed0-videos \
  --wandb-videos
```

From the repository root, those clips are saved under:

```text
results/cami_best_eval_seed0_50_videos/CaMI/checkpoint_env/videos/trial_01_seed_0/
```

The directory contains `successful_rollout_000.mp4`, `failed_rollout_001.mp4`,
and the corresponding outcome name for each episode. Filenames keep the
zero-based rollout index used in the CSV. The recorder prints the video folder
before the batch and each clip path after that episode finishes. The
`rollout_results.csv` and W&B table gain a `video` column with the absolute clip
path. Clips are captured from the same policy execution used for the reported
metrics, with seed 0 applied once before the batch.

The default playback rate is the environment's `control_freq / video_skip`:
20 fps when every step is recorded in Square's 20 Hz environment. Use
`--video-fps 60` for faster playback or `--camera_names agentview robot0_eye_in_hand`
for side-by-side cameras. `--video_path /path/to/clips` selects a different
video root, with separate model/environment/trial subfolders. Choose a new
`--results_dir` or video root for another invocation; existing video batches
are protected from overwriting.

Recording is optional. The command above the recording section saves numeric
results and CSVs only. A process started without recording does not acquire
video capture when you pull this update; unrecorded episodes cannot be turned
into their original videos afterwards. Start a new evaluation with recording
enabled. `rollout_best.py` below also exports simulator trajectories and uses
its own per-episode seeding protocol.

## Compact terminal output

Add `--quiet` to keep the W&B run link, live batch progress bars, short loss
summaries, rollout success rates, and checkpoint completion status. It omits
the large configuration, observation, dataset, model, and per-epoch JSON dumps.
Dataset loading progress and warnings/errors remain visible. The W&B SDK also
uses its quiet setting, which retains warnings and errors.

All scalar metrics still go to W&B/TensorBoard when enabled, and the effective
configuration is still saved in `config.json`. Quiet mode only changes terminal
verbosity. It does not change training, evaluation, or checkpoint contents.

Remove `--debug` for the full configuration. The Square template runs 2,000
epochs with 500 training batches per epoch, and evaluates 50 rollouts every
50 epochs with a 400-step horizon. With `--debug`, those become 2 epochs,
3 batches per epoch, and 2 rollouts per epoch with a 10-step horizon. Starting
without `--debug` creates a fresh full run unless you explicitly use `--resume`.
You can keep `--quiet` in either mode.

## What appears in W&B

| Metric | Meaning |
| --- | --- |
| `Train/Loss` | Combined training objective |
| `Train/BC_Action_Loss` | Action imitation loss |
| `Train/State_CaMI_Loss` | State-query contrastive loss |
| `Train/Traj_CaMI_Loss` | Trajectory-query contrastive loss |
| `Train/state_*`, `Train/traj_*` | CaMI retrieval, negative weighting, and validity diagnostics |
| `Rollout/Success_Rate/square_image_84_with_force` | Fraction of evaluation rollouts completing the task with the dataset filename used above |
| `Rollout/Return/square_image_84_with_force` | Mean sum of rewards over the evaluation rollouts |
| `Rollout/Horizon/square_image_84_with_force` | Mean number of steps before success, termination, or timeout |
| `Valid/*` | Held-out losses when validation is enabled |

The effective training configuration, including dataset split, seed, model
settings, and fitted continuous force scale, is saved in the run config.
Rollout metric suffixes use the dataset filename without `.hdf5`, or its explicit
`train.data[].key` when configured.
In the full Square template, rollout metrics are normally generated every
50 epochs using 50 episodes. Debug rollouts are too short to assess policy
quality. Validation is disabled by default; enable `experiment.validate` and
set `train.hdf5_validation_filter_key="valid"` to use the held-out mask.

Checkpoint files and rollout videos follow the local trainer settings.
This logger does not upload them as W&B artifacts or automatically produce
contact-estimation accuracy measurements. It records the metrics the
algorithm and simulator actually compute.

Online initialization errors stop training with the original exception and
login/account guidance. The logger does not silently switch to offline mode.
For intentional offline logging, set `WANDB_MODE=offline`; afterwards upload
the directory printed by W&B with `wandb sync /path/to/offline-run-directory`.

## If robosuite has no `__version__`

An error such as `module 'robosuite' has no attribute '__version__'` occurs
while constructing the simulator, before W&B logging starts. Check the module
that the training environment imports:

```bash
python - <<'PY'
import sys
import robosuite
print("Python:", sys.executable)
print("Module file:", getattr(robosuite, "__file__", None))
print("Search paths:", list(getattr(robosuite, "__path__", [])))
print("Version:", getattr(robosuite, "__version__", "MISSING"))
print("Has make:", callable(getattr(robosuite, "make", None)))
PY
```

The official robosuite 1.5.2 package defines both `__version__` and `make` in
its `robosuite/__init__.py`. Missing attributes indicate an incomplete,
modified, or shadowing import. A local `robosuite.py` or `robosuite/` can
override the intended installation. Preserve any local changes before
repairing that installation. If using a source checkout, install from the
checkout root containing its `setup.py` or `pyproject.toml`, not from the inner
Python package directory. Do not fix this by inventing a version attribute.

## If `get_joint_qpos_addr` raises `AssertionError`

This failure occurs while constructing the simulator, before optimization or
W&B initialization. Robosuite 1.5.2 checks whether each robot joint is a hinge
or slide using a tuple of MuJoCo enum values. We reproduced the assertion with
MuJoCo 3.14.0: a valid hinge's NumPy integer fails the enum tuple membership
test. The same check passes with MuJoCo 3.5.0. This is a binding compatibility
problem and does not by itself indicate a bad dataset or CaMI loss.
With robosuite 1.5.2 and NumPy 1.26.4 held fixed, changing MuJoCo from 3.14.0
to 3.5.0 also allowed the Square/Panda environment to construct, reset, and
complete one physics step in an isolated Python 3.12 check with rendering
disabled. Your GPU rendering and full training still need the debug check.

First record the versions in the active training environment:

```bash
python - <<'PY'
import sys
from importlib.metadata import version
print("Python:", sys.executable)
for package in ("robosuite", "mujoco", "numpy"):
    print(package, version(package))
PY
```

Install the simulator requirements from the repository root:

```bash
python -m pip install -r requirements-cami-sim.txt
python -m pip check
```

These pins match the simulator versions recorded in `environment.yml`. They
apply to this branch's robosuite 1.5.2 experiments, not every historical
robomimic dataset. Do not remove robosuite's assertion or modify the dataset
metadata to hide a version mismatch.

Check simulator construction, reset, and a single step using the actual
dataset metadata before rerunning training:

```bash
python - <<'PY'
import json
import h5py
import numpy as np
import robosuite

with h5py.File("dataset/square_image_84_with_force.hdf5", "r") as dataset:
    metadata = json.loads(dataset["data"].attrs["env_args"])

kwargs = dict(metadata["env_kwargs"])
# Test physics setup without needing a display or creating camera images.
kwargs.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
env = robosuite.make(metadata["env_name"], **kwargs)
try:
    env.reset()
    env.step(np.zeros(env.action_dim))
    print("Simulator construction, reset, and step passed.")
finally:
    env.close()
PY
```

The debug training command above then checks the image-rendering path and W&B.
If the assertion remains with the pinned versions, include the printed
versions, `robosuite.__file__`, `mujoco.__file__`, and the new traceback when
reporting it. Missing private macros, optional robot models, and GR1's optional
whole-body IK warnings do not cause this Panda/OSC_POSE joint-type assertion.

## Repeated `failed to inject step force` messages during rollouts

Older versions of `run_rollout` attempted to inject force after every reset
and step by unwrapping the environment and calling `_read_raw_ft_sensor()`.
Standard robosuite `NutAssemblySquare` does not provide that custom method.
These caught exceptions printed repeatedly during evaluation, normally first
at epoch 50 in a full run. The obsolete injection has been removed.

Continuous CaMI still reads the stored `obs/force` sequences during training
to weight its contrastive objective. At deployment, both the current CaMI
policy and the BC-RNN baseline use their configured visual and proprioceptive
observations. Environment-provided observations are passed through unchanged.
Force-based policies must obtain force through their environment wrapper.

The current CaMI policy learns a contact-informed representation, but does not
output an explicit contact-state prediction. Task success rates and contrastive
metrics therefore do not directly measure contact-estimation accuracy. Such
an evaluation needs a defined prediction/probe and separate contact ground
truth; runtime diagnostic force readings may be recorded without becoming
policy inputs.

Pull the fix and restart the original training command with `--resume` and the
same `--name`, configuration, and dataset. Resuming uses the latest saved epoch;
an interruption during epoch-50 evaluation normally resumes from epoch 49.
This trainer starts a new W&B run on resume. Pulling changes alone does not
update the code in an already-running Python process.

## Rollout plots, best checkpoint, and successful videos

Plot `Rollout/Success_Rate/square_image_84_with_force` against W&B's Step
(the training epoch). A value of 0.8 means 40 of 50 evaluation rollouts succeeded.
Use the plain metric: its `/mean`, `/max`, `/min`, and `/std` variants summarize
the history of evaluation checkpoints, not the current 50 rollouts. Set chart
smoothing to zero when locating the actual highest evaluation point.
`Rollout/Return/...` and `Rollout/Horizon/...` provide reward and duration
context; short episodes can reflect either early success or termination, so
interpret horizon alongside success rate.

The Square template saves numbered checkpoints every 50 epochs, including
epochs with failures. New best success rates add a `_success_<rate>` suffix.
`last.pth` and `last_bak.pth` are latest weights for resuming. These are policy
parameters, not individual episode paths. Checkpoint `best_success_rate`
metadata is a historical maximum and does not identify the quality of the
current weights. The template has `render_video=false`, and training does
not write individual episode state/action trajectories.

The helper below selects the highest success-tagged checkpoint from the
latest timestamped run in the experiment folder. Pass a timestamped run
directory to evaluate an older run. It works with BC-RNN and CaMI checkpoints.

```bash
export MUJOCO_GL=egl
python -m robomimic.scripts.rollout_best \
  --run-dir trained_models/square_cami_continuous \
  --n-rollouts 50 \
  --camera-names agentview robot0_eye_in_hand
```

For the BC-RNN baseline, use `trained_models/square_bc_rnn` instead.
Use `--checkpoint /path/to/model.pth` to choose weights explicitly, or
`--metric-key square_image_84_with_force` if multiple dataset metrics occur
in checkpoint filenames. Automatic selection requires at least one completed
evaluation with best-success checkpoint saving enabled; it will not silently
substitute `last.pth`. A checkpoint with multiple environment metadata entries
requires the existing evaluation tools and is rejected by this helper.

Each execution is one evaluation trial of `--n-rollouts` rollouts. It creates a
new folder under the run's `successful_rollouts/`:

- `successful_rollout_...mp4`: one video per successful rollout. Add
  `--keep-failures` to also save `failed_rollout_...mp4` and `errors_rollout_...mp4`.
- `SUCCESSFUL_ALL.mp4` and `FAILED_ALL.mp4`: those clips joined in rollout order,
  one video per outcome (failures only with `--keep-failures`), each with a
  `.csv` saying which rollout plays when. `--no-stitch` skips them.
- `rollouts.hdf5`: all rollouts, with actions, rewards, dones, simulator states
  before and after each action, model XML, episode metadata, and success flags.
  Masks `successful`, `failed`, and `errors` select the outcomes.
- `rollouts.csv`: one row per rollout with its seed, outcome, return, length,
  seconds, any simulator error and its video.
- `summary.json`: checkpoint path, original training evaluation score,
  rollout seeds, outcomes, return, length, video filenames, and the new success
  rate across every requested rollout, with its 95% confidence interval (Wilson
  score). Simulator errors count in its denominator. The same numbers are printed
  at the end.

These are fresh rollouts from the selected policy. Old training rollouts cannot
be recovered exactly from weights alone. The default seeds start at 10000;
pass a different `--seed` for a different set. Success videos are selected
examples, while the summary reports the whole evaluation budget. To train
one model and repeat this evaluation over several trials, with a manifest and
W&B summary, use the [trial runner](cami_trials.md).

### Sending an evaluation to Weights & Biases

W&B is opt-in, as in `train.py`: `--wandb`, `--wandb-project NAME` (which also
enables it) or `--wandb-mode online|offline|disabled`. The project defaults to the
checkpoint's `experiment.logging.wandb_proj_name`. It never overwrites the
training metrics: an evaluation is its own run (`job_type=rollouts`), named after
`--name` or the output folder, and placed in `--group` if you give one. For
example, one evaluation with failure videos kept and the output folder named:

```bash
python -m robomimic.scripts.rollout_best \
  --run-dir trained_models/square_cami_continuous \
  --n-rollouts 50 --keep-failures --insertion-report \
  --output-dir square_rollouts_cami \
  --wandb-project cami-square-v3 --name square_rollouts_cami --group rollouts
```

The run gets:

- each rollout as it finishes: `Rollout/Success`, `Rollout/Return`,
  `Rollout/Horizon`, `Rollout/Seconds`, `Rollout/Simulator_Error` (x = `rollout`,
  matching `demo_N`), and beside them `Running/Success_Rate/mean` and `/se`, which
  show the estimate settling;
- at the end, `Summary/...` fields (counts, and mean, SD, SE and n of every
  metric, plus `Summary/Success_Rate/ci95_low` and `ci95_high`), a `rollouts`
  table, and an artifact with `summary.json`, `rollouts.csv` and the joined-video
  indexes;
- the joined videos `SUCCESSFUL_ALL` and `FAILED_ALL` (failures need
  `--keep-failures`). `--wandb-videos all` also puts each kept clip in the table,
  and `none` sends no video. `rollouts.hdf5` is not uploaded.

A W&B problem prints one warning and never stops the evaluation; everything is
still saved in the output folder. A rollout loop that crashes marks the run as
failed. `summary.json` records the run's project, ID and URL under `wandb`.

`--insertion-report` is for the Square task. After the rollouts it replays them and
reports how the nut ended up on the post (whether it was jammed, tilted or
dragged, and how the fingertips touched the table; see
[Check how the nut ends up on the post](cami_v3.md)). It writes `insertions.csv` and
`insertions.json`, prints the report, and sends the numbers to W&B as
`Insertion/...` summary fields (including `jam_flag_share`) and an `insertions`
table. A failure there is printed and does not stop the evaluation. It needs the
same simulator as the rollouts.

The horizon defaults to the checkpoint setting (400 for full Square runs).
Videos default to 20 fps with every step recorded, matching Square's 20 Hz.
If using `--video-skip 5`, use `--fps 4` for approximately natural speed. A
higher `--fps` plays the same frames faster than the simulation: every step at
`--fps 60` is 3 times real time, which the helper prints at the start.
Simulator paths can be replayed without rerunning the policy:

```bash
python -m robomimic.scripts.playback_dataset \
  --dataset /path/to/successful_rollouts/RUN/rollouts.hdf5 \
  --filter_key successful \
  --render_image_names agentview robot0_eye_in_hand \
  --video_path /path/to/replayed_successes.mp4 --video_skip 1
```

State playback uses the stored model and states to avoid drift from action
playback. The original MP4 includes the final post-action success frame;
the standard playback script renders the stored pre-action states. The
helper also preserves final post-action states in `next_states`. Observations
and force/contact supervision are not exported by this utility, so its output
is for trajectory review, not a replacement CaMI training dataset.
