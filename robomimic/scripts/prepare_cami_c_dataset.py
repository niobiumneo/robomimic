"""Build schema-v2 future impulse labels, episode splits and training-only scales."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import h5py
import numpy as np
from robomimic.CAMI.continuous_contact import SCHEMA_VERSION, FEATURE_SETS, future_windows, force_magnitude


def _read(demo, key):
    if key is None or key not in demo:
        raise KeyError("Missing source {} in {}".format(key, demo.name))
    return demo[key][()]


def _scalar(demo, key):
    value = np.asarray(_read(demo, key), dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError("{} must be scalar [T] or [T,1]".format(key))
    return value


def _split(file, demos, fraction, seed):
    masks = file.require_group("mask")
    decode = lambda key: [x.decode() if isinstance(x, bytes) else str(x) for x in masks[key][()]]
    if "valid" in masks and "train" not in masks:
        raise ValueError("Existing valid split needs a disjoint train mask")
    if "train" in masks:
        train, valid = decode("train"), decode("valid") if "valid" in masks else []
        if not train or set(train) & set(valid) or not (set(train) | set(valid)) <= set(demos):
            raise ValueError("Invalid or overlapping train/valid masks")
    else:
        order = np.random.default_rng(seed).permutation(demos).tolist()
        nvalid = min(len(demos)-1, max(1, round(len(demos)*fraction))) if len(demos)>1 and fraction>0 else 0
        valid, train = sorted(order[:nvalid]), sorted(order[nvalid:])
        masks.create_dataset("train", data=np.asarray(train, dtype="S"))
        if valid:
            masks.create_dataset("valid", data=np.asarray(valid, dtype="S"))
    return train, valid


def prepare_dataset(dataset, output, mode="physics", horizon=10, force_key="obs/force",
                    gap_key=None, normal_impulse_key=None, tangential_impulse_key=None,
                    normal_force_key=None, tangential_force_key=None, impulse_dt=None,
                    contact_key=None, contact_features=None, valid_key=None, raw_dataset=None,
                    pairs=None, max_gap=.02, replay_tolerance=1e-3, binary_threshold=10.,
                    validation_fraction=.1, split_seed=1, quiet=False):
    source, destination = Path(dataset).expanduser().resolve(), Path(output).expanduser().resolve()
    if not source.is_file() or destination.exists() or source == destination:
        raise ValueError("Input must exist; output must be a new, different path")
    if (mode not in FEATURE_SETS or not isinstance(horizon, int) or horizon < 1
            or not math.isfinite(binary_threshold) or binary_threshold < 0
            or not math.isfinite(max_gap) or max_gap <= 0
            or not math.isfinite(replay_tolerance) or replay_tolerance < 0
            or not 0 <= validation_fraction < 1):
        raise ValueError("Invalid descriptor mode, horizon, threshold, gap cap, or split")
    if impulse_dt is not None and (not math.isfinite(impulse_dt) or impulse_dt <= 0):
        raise ValueError("impulse_dt must be finite and positive")
    names = FEATURE_SETS[mode]
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    raw, env = None, None
    report = dict(schema_version=SCHEMA_VERSION, feature_names=names, horizon=horizon,
        future_start=1, units=["N"] if mode == "force" else ["m", "N*s", "N*s"],
        gap_cap=max_gap, binary_threshold_force_N=binary_threshold, demos={})
    moments = []
    common_dt = None
    try:
        shutil.copy2(source, tmp)
        with h5py.File(tmp, "r+") as file:
            if raw_dataset:
                raw = h5py.File(Path(raw_dataset).expanduser(), "r")
            if pairs:
                from robomimic.CAMI.continuous_mujoco import make_replay_env
                if mode != "physics":
                    raise ValueError("Use physics mode for pair-based contact replay")
                env = make_replay_env(raw["data"] if raw else file["data"])
            demos = sorted(file["data"], key=lambda x: int(x.split("_")[-1]))
            if not demos:
                raise ValueError("Dataset has no demonstrations")
            train, valid_demos = _split(file, demos, validation_fraction, split_seed)
            report["splits"] = dict(train=train, valid=valid_demos, seed=split_seed)
            for name in demos:
                demo = file["data"][name]
                T = len(demo["actions"])
                if T != demo.attrs["num_samples"] or T < 1:
                    raise ValueError("Invalid action length in {}".format(name))
                details, dt, source_valid = {}, impulse_dt, None
                if pairs:
                    from robomimic.CAMI.continuous_mujoco import replay_descriptors
                    src = raw["data"][name] if raw else demo
                    if src["actions"].shape != demo["actions"].shape or not np.allclose(
                            src["actions"][()], demo["actions"][()], rtol=0, atol=1e-6):
                        raise ValueError("Raw/extracted action mismatch in {}".format(name))
                    values, details = replay_descriptors(env, src, pairs, max_gap, replay_tolerance)
                    dt = details["interval_dt"]
                    source_valid = details.pop("_valid")
                    if not np.allclose(details["interval_dt_range"], dt, rtol=1e-6, atol=1e-9):
                        raise ValueError("Variable action intervals are not supported in one physics dataset")
                elif contact_key:
                    values = np.asarray(_read(demo, contact_key), dtype=np.float32)
                    declared = json.loads(demo[contact_key].attrs.get("feature_names", "null"))
                    if declared is not None and declared != names:
                        raise ValueError("Legacy or incompatible descriptor: recreate impulse labels using replay or recollect")
                    if declared is None and contact_features != names:
                        raise ValueError("Unmarked descriptor: explicitly provide --contact-features with the actual feature order")
                    if "contact/dt" in demo and contact_key == "contact/physics":
                        intervals = np.asarray(demo["contact/dt"])
                        dt = float(intervals[0])
                        if intervals.shape != (T,) or not np.allclose(intervals, dt, rtol=1e-6, atol=1e-9):
                            raise ValueError("Invalid or variable recorded action intervals")
                    if valid_key is None and contact_key == "contact/physics" and "contact/valid" in demo:
                        source_valid = _read(demo, "contact/valid")
                    details = dict(source=contact_key)
                elif mode == "force":
                    values = force_magnitude(_read(demo, force_key))
                    details = dict(source=force_key, force_semantics="translational_wrench_magnitude")
                else:
                    if normal_impulse_key or tangential_impulse_key:
                        if normal_force_key or tangential_force_key:
                            raise ValueError("Choose impulse columns OR force columns")
                        keys = [gap_key, normal_impulse_key, tangential_impulse_key]
                        values = np.concatenate([_scalar(demo, key) for key in keys], axis=1)
                        details = dict(source=keys, impulse_semantics="user_supplied")
                    else:
                        keys = [gap_key, normal_force_key, tangential_force_key]
                        values = np.concatenate([_scalar(demo, key) for key in keys], axis=1)
                        if dt is None:
                            raise ValueError("Force conversion requires --impulse-dt in seconds")
                        values[:, 1:] *= dt
                        details = dict(source=keys, impulse_semantics="force_times_dt_estimate")
                if mode == "physics":
                    if dt is None or not math.isfinite(dt) or dt <= 0:
                        raise ValueError("Impulse sources require the action interval: --impulse-dt or contact/dt")
                    if common_dt is not None and not np.isclose(dt, common_dt, rtol=1e-6, atol=1e-9):
                        raise ValueError("Do not mix different impulse interval durations in one dataset")
                    common_dt = dt
                if values.shape != (T, len(names)):
                    raise ValueError("{} descriptor shape {}, expected {}".format(name, values.shape, (T, len(names))))
                mask = np.asarray(_read(demo, valid_key)) if valid_key else (source_valid if source_valid is not None else np.ones_like(values))
                if mask.shape != values.shape or not np.all((mask == 0) | (mask == 1)):
                    raise ValueError("Validity source must be binary [T,F]")
                mask = mask.astype(bool)
                for i, feature in enumerate(names):
                    if feature != "gap" and (values[:, i][mask[:, i]] < 0).any():
                        raise ValueError("{} must be nonnegative".format(feature))
                windows, future_mask = future_windows(values, horizon, mask)
                threshold = binary_threshold * dt if mode == "physics" else binary_threshold
                idx = 1 if mode == "physics" else 0
                labels = ((windows[..., idx] > threshold) & future_mask[..., idx].astype(bool)).any(1)[:, None]
                # No future data at the final anchor: zero label, zero mask.
                for key, array in (("contact_state", windows), ("contact_mask", future_mask),
                                   ("contact_label", labels.astype(np.float32))):
                    if key in demo:
                        del demo[key]
                    demo.create_dataset(key, data=array, compression="gzip", compression_opts=4)
                if name in train:
                    safe = np.where(mask, values, 0.).astype(np.float64)
                    moments.append((mask.sum(0), safe.sum(0), (safe**2).sum(0)))
                report["demos"][name] = dict(details, samples=T, interval_dt=dt,
                    valid_fraction=float(future_mask.mean()), binary_fraction=float(labels.mean()))
                if not quiet:
                    print("Prepared {}: {} samples".format(name, T), flush=True)
            count, sums, squares = np.asarray(moments).sum(0)
            if (count == 0).any():
                raise ValueError("Each descriptor needs valid observations in the training split")
            mean = sums / count
            std = np.sqrt(np.maximum(0., squares / count - mean**2))
            # Centering cancels in every pairwise difference. Store means for
            # provenance; divide by (training std + epsilon) in the loss.
            report["normalization"] = dict(method="training_std", mean=mean.tolist(),
                std=std.tolist(), scale=(std + 1e-6).tolist(), epsilon=1e-6,
                fit_demos=train, count=count.astype(int).tolist())
            report.update(interval_dt=common_dt,
                binary_feature="normal_impulse" if mode == "physics" else "force",
                binary_threshold=binary_threshold * common_dt if mode == "physics" else binary_threshold,
                binary_threshold_units="N*s" if mode == "physics" else "N")
            for demo in file["data"].values():
                demo.attrs["cami_c"] = json.dumps({k:v for k,v in report.items() if k != "demos"})
            file["data"].attrs["cami_c"] = json.dumps(report)
        os.link(tmp, destination)
        os.unlink(tmp)
    finally:
        if raw is not None:
            raw.close()
        if env is not None:
            close = getattr(env, "close", None) or getattr(getattr(env, "env", None), "close", None)
            if close:
                close()
        if os.path.exists(tmp):
            os.unlink(tmp)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True); p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=FEATURE_SETS, default="physics")
    p.add_argument("--horizon", type=int, default=10); p.add_argument("--force-key", default="obs/force")
    for name in ("gap", "normal-impulse", "tangential-impulse", "normal-force", "tangential-force", "contact", "valid"):
        p.add_argument("--" + name + "-key")
    p.add_argument("--contact-features", nargs="+")
    p.add_argument("--impulse-dt", type=float, help="Duration in seconds represented by each impulse row")
    p.add_argument("--raw-dataset"); p.add_argument("--pairs", help="JSON geometry groups, enables action replay")
    p.add_argument("--max-gap", type=float, default=.02); p.add_argument("--replay-tolerance", type=float, default=1e-3)
    p.add_argument("--binary-threshold", type=float, default=10., help="Force threshold N; converted to N*s in physics mode")
    p.add_argument("--validation-fraction", type=float, default=.1); p.add_argument("--split-seed", type=int, default=1)
    p.add_argument("--quiet", action="store_true", help="Print only the final preparation summary")
    args = vars(p.parse_args())
    if args["pairs"]:
        args["pairs"] = json.loads(Path(args["pairs"]).read_text())
    report = prepare_dataset(**args)
    print("Created {} | features={} | horizon={} | train={} | valid={}".format(
        args["output"], report["feature_names"], report["horizon"],
        len(report["splits"]["train"]), len(report["splits"]["valid"])))


if __name__ == "__main__":
    main()
