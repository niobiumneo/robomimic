"""List the settings in which trained checkpoints differ.

Every robomimic checkpoint stores the resolved config it trained with, so two
runs can be compared without their run directories or launch commands:

    python -m robomimic.scripts.compare_checkpoint_configs \
        train_py_baseline.pth v3_baseline.pth v3_prediction.pth

Column [0] is the reference. Settings that exist only for CaMI (``algo.cami.*``
and the extra optimizers) are counted rather than listed when some checkpoint has
CaMI switched off, because they do nothing there; when every checkpoint trains
with CaMI they are listed, since they differ between active runs. ``--show-cami``
lists them always. A few bookkeeping settings are not compared (experiment name,
logging, output directory, ``train.cuda``, ``train.num_data_workers`` and
``train.hdf5_cache_mode``); differences in them are counted in the output, and
``--compare-all`` includes them.

Checkpoints are loaded with ``weights_only=True`` plus the numpy array types a
trainer checkpoint holds. A file that needs anything else is refused unless
``--trust-checkpoints`` is passed, because full unpickling runs code from the file.
"""
import argparse
import json
import os
import pickle
import re
import sys

import torch

CAMI_ONLY = re.compile(r"^algo\.(cami\.(?!enabled$)|optim_params\.(cami|state_encoder|snippet_encoder|key_proj)\.)")
DEFAULT_IGNORE = (r"^experiment\.name$", r"^experiment\.logging\.", r"^train\.output_dir$", r"^train\.cuda$",
                  r"^train\.num_data_workers$", r"^train\.hdf5_cache_mode$")
ABSENT = "<absent>"
MAX_SHOWN = 120


def _numpy_types():
    """The numpy array, scalar and dtype types that trainer checkpoints store; plain data, no code."""
    import numpy as np
    try:
        from numpy._core import multiarray
    except ImportError:
        from numpy.core import multiarray
    types = [np.dtype, np.ndarray, multiarray.scalar, multiarray._reconstruct]
    return types + [getattr(np.dtypes, name) for name in dir(getattr(np, "dtypes", object)) if name.endswith("DType")]


def load_checkpoint(path, trust=False):
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise SystemExit("Checkpoint not found: {}".format(path))
    try:
        with torch.serialization.safe_globals(_numpy_types()):
            return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        message = str(error)
        if "Unsupported global" not in message and "allowlisted" not in message and "Can only build" not in message:
            raise SystemExit("{} is not a readable checkpoint ({})".format(path, message.splitlines()[0])) from None
        if not trust:
            raise SystemExit("{} needs full unpickling, which can run code stored in the file. Re-run with "
                             "--trust-checkpoints if you trust its source.".format(path)) from None
    except (RuntimeError, EOFError, OSError) as error:
        raise SystemExit("{} is not a readable checkpoint ({})".format(path, str(error).splitlines()[0])) from None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise SystemExit("{} could not be loaded ({}: {})".format(path, type(error).__name__, str(error).splitlines()[0])) from None


def stored_config(checkpoint, path):
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if config is None:
        raise SystemExit("{} has no stored config".format(path))
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            raise SystemExit("{} stores a config that is not JSON".format(path)) from None
    if not isinstance(config, dict):
        raise SystemExit("{} stores a config that is not a dictionary".format(path))
    return config


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, prefix + "." + str(key) if prefix else str(key)))
        return out
    return {prefix: value}


def describe(checkpoint, config):
    state = checkpoint.get("variable_state") or {}
    epoch = state.get("epoch")
    steps = config.get("experiment", {}).get("epoch_every_n_steps")
    info = dict(algo=checkpoint.get("algo_name") or config.get("algo_name"), epoch=epoch,
                gradient_steps=epoch * steps if isinstance(epoch, int) and isinstance(steps, int) else None)
    for key in ("best_success_rate", "best_return", "best_valid_loss"):
        if isinstance(state.get(key), (int, float)):
            info[key] = state[key]
        elif isinstance(state.get(key), dict) and state[key]:
            info[key] = {k: v for k, v in state[key].items()}
    return info


def compare(configs, ignore=DEFAULT_IGNORE, show_cami=False):
    """Return (rows, hidden_cami_count, identical_count, ignored_differences) for flattened configs.

    CaMI-only settings are hidden only when some config has CaMI off; between runs that all use
    CaMI they are real differences. ``ignored_differences`` lists the ignored settings that differ.
    """
    flat = [flatten(c) for c in configs]
    patterns = [re.compile(p) for p in ignore]
    all_active = all(f.get("algo.cami.enabled") is True for f in flat)
    rows, hidden, identical, ignored = [], 0, 0, []
    for key in sorted(set().union(*flat)):
        values = [f.get(key, ABSENT) for f in flat]
        same = all(v == values[0] for v in values[1:])
        if any(p.search(key) for p in patterns):
            if not same:
                ignored.append(key)
            continue
        if same:
            identical += 1
            continue
        if not show_cami and not all_active and CAMI_ONLY.search(key):
            hidden += 1
            continue
        rows.append((key, values))
    return rows, hidden, identical, ignored


def shorten(value):
    """One-line JSON of a setting; a long list or string is cut with its length."""
    if value == ABSENT:
        return ABSENT
    text = json.dumps(value)
    if len(text) <= MAX_SHOWN:
        return text
    size = " ({} items)".format(len(value)) if isinstance(value, (list, dict)) else ""
    return text[:MAX_SHOWN - 3] + "..." + size


def render(names, infos, rows, hidden, identical, ignored=()):
    lines = ["Checkpoints"]
    for i, (name, info) in enumerate(zip(names, infos)):
        extra = ", ".join("{}={}".format(k, v) for k, v in info.items() if v is not None and k != "algo")
        lines.append("  [{}] {} | algo={}{}".format(i, name, info["algo"], " | " + extra if extra else ""))
    lines.append("")
    lines.append("Differing settings: {} ({} identical, {} CaMI-only hidden)".format(len(rows), identical, hidden))
    for key, values in rows:
        lines.append("  " + key)
        for i, value in enumerate(values):
            marker = "=" if i and value == values[0] else " "
            lines.append("      [{}]{} {}".format(i, marker, shorten(value)))
    if ignored:
        lines.append("")
        lines.append("Not compared by default, but different: {}. Use --compare-all to list them.".format(", ".join(ignored)))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint files; the first is the reference")
    parser.add_argument("--ignore", action="append", default=[], help="Extra regex of settings to skip")
    parser.add_argument("--show-cami", action="store_true", help="List CaMI-only settings too")
    parser.add_argument("--compare-all", action="store_true", help="Also compare the settings skipped by default")
    parser.add_argument("--trust-checkpoints", action="store_true", help="Allow full unpickling of the checkpoints")
    args = parser.parse_args(argv)
    if len(args.checkpoints) < 2:
        parser.error("Pass at least two checkpoints to compare")
    configs, infos = [], []
    for path in args.checkpoints:
        checkpoint = load_checkpoint(path, args.trust_checkpoints)
        config = stored_config(checkpoint, path)
        configs.append(config)
        infos.append(describe(checkpoint, config))
    ignore = tuple(args.ignore) if args.compare_all else DEFAULT_IGNORE + tuple(args.ignore)
    rows, hidden, identical, ignored = compare(configs, ignore, args.show_cami)
    print(render([os.path.basename(p) for p in args.checkpoints], infos, rows, hidden, identical, ignored))
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
