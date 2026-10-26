"""Recompute the gap channel of a prepared physics dataset with the per-pair gap rule.

Gaps are measured in the recorded states, so this needs no stepping and leaves the
impulse channels, contact_label and every other field untouched. Only
contact_state[..., 0], contact_mask[..., 0] and the descriptor metadata change.
The source file is never modified; the output must be a new path:

    python -m robomimic.scripts.recompute_cami_gaps \
        --dataset dataset/square_image_84_contactnets_state.hdf5 \
        --output dataset/square_image_84_contactnets_state_gapfix.hdf5

The earlier rule marked a whole row's gap unknown when any one nut-peg pair hit a
known CCD failure. That failure reports separated geoms, so when another pair is
validly penetrating the row gap is still known. Gaps that were valid before must
come out identical, which is checked for every demo.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

from robomimic.CAMI.continuous_contact import FEATURE_SETS, future_windows
from robomimic.CAMI.continuous_mujoco import (
    ContinuousContactProbe, make_replay_env, native_model_data, reset_demo,
)


def recompute_gaps(env, demo, pairs, max_gap, length):
    """[T] gaps and validity of the recorded states that start each of a demo's T actions.

    states may hold T+1 rows (the state after the last action); labels only describe the first T.
    """
    reset_demo(env, demo)
    model, data = native_model_data(env.env.sim)
    probe = ContinuousContactProbe(model, data, [tuple(pair) for pair in pairs], max_gap)
    states = np.asarray(demo["states"])
    if len(states) not in (length, length + 1):
        raise ValueError("{}: states must have T or T+1 rows for T={} label rows".format(demo.name, length))
    states = states[:length]
    gaps, valid = np.zeros(len(states)), np.zeros(len(states), dtype=bool)
    for t, state in enumerate(states):
        env.reset_to({"states": state})  # sets time, qpos and qvel, then runs the forward pass
        gaps[t], valid[t] = probe.gap(), probe.gap_valid
    return gaps, valid, probe.GAP_RULE


def recompute_dataset(dataset, output, max_gap=None, quiet=False):
    source, destination = Path(dataset).expanduser().resolve(), Path(output).expanduser().resolve()
    if not source.is_file() or destination.exists() or source == destination:
        raise ValueError("Input must exist; output must be a new, different path")
    fd, tmp = tempfile.mkstemp(prefix=".recompute_gaps_", suffix=".hdf5", dir=destination.parent)
    os.close(fd)
    env = None
    try:
        shutil.copyfile(source, tmp)
        shutil.copymode(source, tmp)  # mkstemp creates the file private
        with h5py.File(tmp, "r+") as file:
            report = json.loads(file["data"].attrs.get("cami_c", "{}"))
            if report.get("feature_names") != FEATURE_SETS["physics"] or report.get("future_start") != 1:
                raise ValueError("{} is not a prepared physics dataset".format(source))
            cap = float(report.get("gap_cap", .02) if max_gap is None else max_gap)
            if cap != report.get("gap_cap", cap):
                raise ValueError("--max-gap {} differs from the dataset's gap_cap {}; keep one cap per dataset"
                                 .format(cap, report["gap_cap"]))
            semantics = {d.get("gap_semantics") for d in report.get("demos", {}).values()}
            if semantics != {"recorded_state_geometry"}:
                raise ValueError("Gaps here were not all measured in the recorded states ({}); recomputing them from "
                                 "the recorded states would change their meaning. Prepare with --replay-mode state "
                                 "instead".format(sorted(map(str, semantics))))
            horizon = report["horizon"]
            train = set(report.get("splits", {}).get("train", []))
            env = make_replay_env(file["data"])
            before = after = rows = 0
            fitted = []
            rule = None
            for name, demo in file["data"].items():
                details = report["demos"].get(name)
                if not details or not details.get("pairs"):
                    raise ValueError("{}: the metadata records no resolved geometry pairs".format(name))
                state, mask = demo["contact_state"][()], demo["contact_mask"][()]
                gaps, valid, rule = recompute_gaps(env, demo, details["pairs"], cap, len(state))
                windows, window_mask = future_windows(gaps[:, None], horizon, valid[:, None])
                old = mask[..., 0].astype(bool)
                if (old & ~window_mask[..., 0].astype(bool)).any():
                    raise ValueError("{}: a gap valid under the earlier rule is now unknown".format(name))
                if not np.allclose(state[..., 0][old], windows[..., 0][old], rtol=0, atol=1e-6):
                    raise ValueError("{}: recomputed gaps differ from the stored valid gaps".format(name))
                state[..., 0], mask[..., 0] = windows[..., 0], window_mask[..., 0]
                demo["contact_state"][...] = state
                demo["contact_mask"][...] = mask
                live = np.arange(len(state)) + 1 < len(state)  # rows with any future offset
                before += int((~old[live, 0]).sum())
                after += int((~mask[live, 0, 0].astype(bool)).sum())
                rows += int(live.sum())
                details.update(gap_rule=rule, valid_fraction=float(mask.mean()))
                if name in train:
                    fitted.append(gaps[valid])
            values = np.concatenate(fitted)
            normalization = report.get("normalization")
            if normalization:
                # The gap statistics follow the recomputed values; impulse statistics are unchanged.
                normalization["mean"][0] = float(values.mean())
                normalization["std"][0] = float(values.std())
                normalization["scale"][0] = float(values.std() + normalization.get("epsilon", 1e-6))
                if "count" in normalization:
                    normalization["count"][0] = int(len(values))
            report["gap_rule"] = rule
            report["gap_recomputed_from"] = str(source)
            for demo in file["data"].values():
                demo.attrs["cami_c"] = json.dumps({k: v for k, v in report.items() if k != "demos"})
            file["data"].attrs["cami_c"] = json.dumps(report)
        os.link(tmp, destination)
        os.unlink(tmp)
    finally:
        if env is not None:
            close = getattr(env, "close", None) or getattr(getattr(env, "env", None), "close", None)
            if close:
                close()
        if os.path.exists(tmp):
            os.unlink(tmp)
    if not quiet:
        print("Unknown gaps at offset 1: {}/{} ({:.1%}) before, {}/{} ({:.1%}) after".format(
            before, rows, before / max(1, rows), after, rows, after / max(1, rows)), flush=True)
    return dict(before=before, after=after, rows=rows, gap_rule=rule)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Prepared physics dataset (not modified)")
    p.add_argument("--output", required=True, help="New output path")
    p.add_argument("--max-gap", type=float, help="Default: the dataset's gap_cap")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    result = recompute_dataset(args.dataset, args.output, args.max_gap, args.quiet)
    print("Created {} | gap rule {}".format(args.output, result["gap_rule"]))


if __name__ == "__main__":
    main()
