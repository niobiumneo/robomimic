"""Validate prepared physics descriptors and fit scales on training demos only."""

import json

import h5py
import numpy as np

from robomimic.CAMI.continuous_contact import (
    FEATURE_SETS, SCHEMA_VERSION, contact_observation_keys,
)
from robomimic.utils.cami_dataset_utils import (
    _blocks, _dataset, resolve_dataset_path, selected_demos,
)


SCALE_KEYS = ("gap_scale", "normal_impulse_scale", "tangential_impulse_scale")


def prepare_contactnets_datasets(config, *, require_rnn=True):
    """Check the exact feature order, future offsets, masks and SI units.

    A null scale fits std + 1e-6 on the selected training demonstrations. Use
    only offset 1 of each future window so an observation is counted once,
    rather than H times. Validation descriptors never enter these statistics.
    Resolved scales and fitted demo names are saved in the run's config.
    """
    c, train = config.algo.cami.continuous, config.train
    names = list(c.feature_names)
    horizon = config.algo.cami.snippet_horizon
    if c.formulation_version != 7 or names != FEATURE_SETS["physics"]:
        raise ValueError("ContactNets requires formulation 7 and [gap, normal_impulse, tangential_impulse_magnitude]")
    if not isinstance(horizon, int) or horizon < 1 or train.seq_length < horizon + 1 or train.frame_stack != 1:
        raise ValueError("ContactNets requires frame_stack=1 and seq_length>=snippet_horizon+1")
    if require_rnn and (not config.algo.rnn.enabled or config.algo.rnn.open_loop
                        or config.algo.rnn.kwargs.get("bidirectional", False)):
        raise ValueError("ContactNets requires a causal, closed-loop BC-RNN")
    if config.algo.cami.continuous_contact.enabled:
        raise ValueError("Remove the force-only continuous_contact.enabled setting; ContactNets uses cami.continuous")
    weights = np.asarray(c.feature_weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any() or not weights.sum() > 0:
        raise ValueError("continuous.feature_weights must be three finite, nonnegative weights, not all zero")
    fit = [c[key] is None for key in SCALE_KEYS]
    for key, fitting in zip(SCALE_KEYS, fit):
        if not fitting and (not np.isfinite(c[key]) or c[key] <= 0):
            raise ValueError("continuous.{} must be null or finite and positive".format(key))

    entries = train.data
    if isinstance(entries, str):
        entries = [{"path": entries}]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Set train.data to a nonempty list of prepared physics datasets")
    entries = [dict(entry, path=resolve_dataset_path(entry.get("path"))) for entry in entries]
    if set(train.action_keys) != {"actions"}:
        raise ValueError("ContactNets requires action_keys=['actions'] to align prepared labels")
    keys = list(train.dataset_keys)
    for key in ("actions", "contact_state", "contact_mask"):
        if key not in keys:
            keys.append(key)
    with config.values_unlocked():
        train.data = entries
        train.dataset_keys = keys

    forbidden = set(c.privileged_obs_keys)
    forbidden.update(k for k in config.all_obs_keys if k.lower().startswith("contact_")
                     or any(part in k.lower() for part in ("force", "torque", "impulse")))
    count = np.zeros(3, dtype=np.int64)
    mean, m2 = np.zeros(3), np.zeros(3)
    fit_demos, shapes, common_dt = [], {}, None
    reconstruction = []
    for entry in entries:
        with h5py.File(entry["path"], "r") as hdf5:
            if "data" not in hdf5 or "env_args" not in hdf5["data"].attrs:
                raise ValueError("{}: missing data group or env_args".format(entry["path"]))
            if not bool(hdf5.attrs.get("complete", True)):
                raise ValueError("{}: dataset is marked incomplete".format(entry["path"]))
            env_meta = json.loads(hdf5["data"].attrs["env_args"])
            if not all(key in env_meta for key in ("env_name", "type", "env_kwargs")):
                raise ValueError("{}: incomplete environment metadata".format(entry["path"]))
            meta = json.loads(hdf5["data"].attrs.get("cami_c", "{}"))
            if "reconstruction" in meta:
                reconstruction.append(dict(meta["reconstruction"], dataset=entry["path"]))
            if (meta.get("schema_version") != SCHEMA_VERSION or meta.get("feature_names") != names
                    or meta.get("horizon") != horizon or meta.get("future_start") != 1
                    or meta.get("units") != ["m", "N*s", "N*s"]):
                raise ValueError("{}: missing or incompatible physics labels. Run python -m "
                                 "robomimic.scripts.prepare_cami_c_dataset --mode physics "
                                 "using recorded descriptors or action replay; obs/force alone is insufficient."
                                 .format(entry["path"]))
            dt = meta.get("interval_dt")
            if dt is None or not np.isfinite(dt) or dt <= 0:
                raise ValueError("Physics labels must declare a positive action interval in seconds")
            if common_dt is not None and not np.isclose(common_dt, dt, rtol=1e-6, atol=1e-9):
                raise ValueError("Do not mix different impulse interval durations")
            common_dt = dt
            forbidden.update(contact_observation_keys(meta))
            if forbidden.intersection(config.all_obs_keys):
                raise ValueError("Privileged force/contact descriptor sources must not be policy or goal observations")
            training = selected_demos(hdf5, entry, train.hdf5_filter_key)
            validation = []
            if config.experiment.validate:
                if train.hdf5_filter_key is None or train.hdf5_validation_filter_key is None:
                    raise ValueError("Validation requires explicit train and valid filter keys")
                validation = selected_demos(hdf5, entry, train.hdf5_validation_filter_key)
                if set(training).intersection(validation):
                    raise ValueError("Training and validation demonstrations overlap; remove per-dataset filter_key")
            for name in sorted(set(training).union(validation)):
                demo = hdf5["data/" + name]
                length = int(demo.attrs.get("num_samples", 0))
                if length < 1 or (not train.pad_seq_length and length < train.seq_length):
                    raise ValueError("{}: invalid num_samples or sequence length".format(demo.name))
                actions = _dataset(demo, "actions", length)
                if actions.ndim != 2:
                    raise ValueError("{}: actions must be [T,A]".format(demo.name))
                for _ in _blocks(actions):
                    pass
                if shapes.setdefault("actions", actions.shape[1:]) != actions.shape[1:]:
                    raise ValueError("Inconsistent action dimensions")
                for prefix in (["obs", "next_obs"] if train.hdf5_load_next_obs else ["obs"]):
                    for modality, obs_keys in config.observation.modalities.obs.items():
                        for key in obs_keys:
                            obs = _dataset(demo, prefix + "/" + key, length)
                            if shapes.setdefault(key, obs.shape[1:]) != obs.shape[1:]:
                                raise ValueError("{}: inconsistent observation shape".format(obs.name))
                            if modality == "rgb":
                                if obs.ndim != 4 or obs.shape[-1] != 3 or obs.dtype != np.uint8:
                                    raise ValueError("{}: RGB must be uint8 [T,H,W,3]".format(obs.name))
                                encoder = config.observation.encoder.rgb
                                if encoder.obs_randomizer_class == "CropRandomizer":
                                    crop = encoder.obs_randomizer_kwargs
                                    if crop.crop_height >= obs.shape[1] or crop.crop_width >= obs.shape[2]:
                                        raise ValueError("{}: configured crop must be smaller than image".format(obs.name))
                            elif modality == "low_dim":
                                for _ in _blocks(obs):
                                    pass
                state, mask = [_dataset(demo, key, length) for key in ("contact_state", "contact_mask")]
                if state.shape != (length, horizon, 3) or mask.shape != state.shape:
                    raise ValueError("{}: contact_state and contact_mask must be [T,H,3]".format(demo.name))
                for start in range(0, length, 4096):
                    stop = min(length, start + 4096)
                    values, valid = np.asarray(state[start:stop], dtype=np.float64), np.asarray(mask[start:stop])
                    if not np.isin(valid, [0, 1]).all():
                        raise ValueError("{}: contact_mask must be binary".format(demo.name))
                    valid = valid.astype(bool)
                    future = np.arange(start, stop)[:, None] + np.arange(1, horizon + 1)[None] < length
                    if (valid & ~future[..., None]).any():
                        raise ValueError("{}: contact_mask marks padded future actions valid".format(demo.name))
                    if not np.isfinite(values[valid]).all():
                        raise ValueError("{}: nonfinite valid contact data".format(demo.name))
                    if (values[..., 1:][valid[..., 1:]] < 0).any():
                        raise ValueError("{}: impulse magnitudes must be nonnegative".format(demo.name))
                    if name in training and any(fit):
                        for feature in range(3):
                            sample = values[:, 0, feature][valid[:, 0, feature]]
                            n = len(sample)
                            if not n:
                                continue
                            block_mean = sample.mean()
                            delta, total = block_mean - mean[feature], count[feature] + n
                            m2[feature] += ((sample - block_mean) ** 2).sum() + delta ** 2 * count[feature] * n / total
                            mean[feature] += delta * n / total
                            count[feature] = total
                if name in training:
                    fit_demos.append(entry["path"] + "#" + name)
    with config.values_unlocked():
        c.privileged_obs_keys = sorted(forbidden)
        c.dataset_reconstruction = reconstruction
        if any(fit):
            for i, (key, fitting) in enumerate(zip(SCALE_KEYS, fit)):
                if fitting:
                    if count[i] == 0:
                        raise ValueError("No valid training measurements for " + names[i])
                    c[key] = float(np.sqrt(max(0., m2[i] / count[i])) + 1e-6)
            c.normalization_fit_demos = fit_demos
