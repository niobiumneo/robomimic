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

For an existing `dataset/square_image_84_with_force.hdf5`:

```bash
export MUJOCO_GL=egl

python -m robomimic.scripts.train \
  --config robomimic/exps/templates/bc_cami_square.json \
  --dataset dataset/square_image_84_with_force.hdf5 \
  --name square_cami_wandb_check \
  --wandb --wandb-project cami-contact-state \
  --debug
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
  --wandb --wandb-project cami-contact-state
```

`--wandb-project` also enables logging by itself. JSON users can set
`experiment.logging.log_wandb=true` and
`experiment.logging.wandb_proj_name="cami-contact-state"`. The CLI overrides
these settings. `--name` becomes the displayed W&B run name.

## What appears in W&B

| Metric | Meaning |
| --- | --- |
| `Train/Loss` | Combined training objective |
| `Train/BC_Action_Loss` | Action imitation loss |
| `Train/State_CaMI_Loss` | State-query contrastive loss |
| `Train/Traj_CaMI_Loss` | Trajectory-query contrastive loss |
| `Train/state_*`, `Train/traj_*` | CaMI retrieval, negative weighting, and validity diagnostics |
| `Rollout/Success_Rate/NutAssemblySquare` | Fraction of evaluation rollouts completing the task |
| `Valid/*` | Held-out losses when validation is enabled |

The effective training configuration, including dataset split, seed, model
settings, and fitted continuous force scale, is saved in the run config.
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
