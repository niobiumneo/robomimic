# Track CaMI training in Weights & Biases

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
- `summary.json`: checkpoint path, original training evaluation score,
  rollout seeds, outcomes, return, length, video filenames, and the new success
  rate across every requested rollout. Simulator errors count in its denominator.

These are fresh rollouts from the selected policy. Old training rollouts cannot
be recovered exactly from weights alone. The default seeds start at 10000;
pass a different `--seed` for a different set. Success videos are selected
examples, while the summary reports the whole evaluation budget. The helper
does not upload its outputs or overwrite the training metrics in W&B. To train
one model and repeat this evaluation over several trials, with a manifest and
W&B summary, use the [trial runner](cami_trials.md).

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
