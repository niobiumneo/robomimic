"""List the settings in which trained checkpoints differ.

Every robomimic checkpoint stores the resolved config it trained with, so two
runs can be compared without their run directories or launch commands:

    python -m robomimic.scripts.compare_checkpoint_configs \
        train_py_baseline.pth v3_baseline.pth v3_prediction.pth

Column [0] is the reference. Settings that exist only for CaMI (``algo.cami.*``
and the extra optimizers) are counted rather than listed unless
``--show-cami`` is passed. Checkpoints are loaded with ``weights_only=True``;
a checkpoint that needs full unpickling is refused unless ``--trust-checkpoints``
is passed, because unpickling runs code from the file.
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


def load_checkpoint(path, trust=False):
    path = os.path.expanduser(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        if not trust:
            raise SystemExit("{} needs full unpickling, which can run code stored in the file. Re-run with "
                             "--trust-checkpoints if you trust its source.\n({})".format(path, str(error).splitlines()[0]))
    return torch.load(path, map_location="cpu", weights_only=False)


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
    """Return (rows, hidden_cami_count, identical_count) for flattened configs."""
    flat = [flatten(c) for c in configs]
    patterns = [re.compile(p) for p in ignore]
    rows, hidden, identical = [], 0, 0
    for key in sorted(set().union(*flat)):
        if any(p.search(key) for p in patterns):
            continue
        values = [f.get(key, ABSENT) for f in flat]
        if all(v == values[0] for v in values[1:]):
            identical += 1
            continue
        if not show_cami and CAMI_ONLY.search(key):
            hidden += 1
            continue
        rows.append((key, values))
    return rows, hidden, identical


def render(names, infos, rows, hidden, identical):
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
            lines.append("      [{}]{} {}".format(i, marker, json.dumps(value) if value != ABSENT else ABSENT))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint files; the first is the reference")
    parser.add_argument("--ignore", action="append", default=[], help="Extra regex of settings to skip")
    parser.add_argument("--show-cami", action="store_true", help="List CaMI-only settings too")
    parser.add_argument("--trust-checkpoints", action="store_true", help="Allow full unpickling of the checkpoints")
    args = parser.parse_args(argv)
    if len(args.checkpoints) < 2:
        parser.error("Pass at least two checkpoints to compare")
    configs, infos = [], []
    for path in args.checkpoints:
        checkpoint = load_checkpoint(path, args.trust_checkpoints)
        if "config" not in checkpoint:
            raise SystemExit("{} has no stored config".format(path))
        config = checkpoint["config"]
        config = json.loads(config) if isinstance(config, str) else config
        configs.append(config)
        infos.append(describe(checkpoint, config))
    rows, hidden, identical = compare(configs, DEFAULT_IGNORE + tuple(args.ignore), args.show_cami)
    print(render([os.path.basename(p) for p in args.checkpoints], infos, rows, hidden, identical))
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
