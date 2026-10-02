"""Validate CaMI's stored supervision before constructing loaders or models.

These checks verify the HDF5 interface, not the physical accuracy of a replay.
Force must already be recorded and aligned with the observations and actions.
"""

import json
from pathlib import Path

import h5py
import numpy as np


CAMI_ALGOS = {"bc_cami", "bc_cami_lcp", "bc_cami_cance"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_dataset_path(path):
    """Keep existing working-directory paths; also accept checkout-relative paths."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Set train.data to a list containing {'path': 'dataset/...hdf5'}")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and not candidate.is_file():
        candidate = REPO_ROOT / candidate
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {candidate}. Put the HDF5 under this checkout's "
            "dataset/ directory or set train.data[0].path to its absolute path."
        )
    return str(candidate.resolve())


def selected_demos(hdf5, entry, filter_key):
    """Use the same filter override, numeric ordering, and limit as SequenceDataset."""
    key = entry.get("filter_key", filter_key)
    if key is None:
        demos = list(hdf5["data"])
    else:
        if f"mask/{key}" not in hdf5:
            raise ValueError(f"{hdf5.filename}: missing split mask/{key}")
        demos = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                 for x in hdf5[f"mask/{key}"][()]]
    if len(set(demos)) != len(demos) or any(x not in hdf5["data"] for x in demos):
        raise ValueError(f"{hdf5.filename}: split contains duplicate or missing demonstrations")
    try:
        if any(not name.startswith("demo_") for name in demos):
            raise ValueError("Unexpected demonstration name")
        demos = sorted(demos, key=lambda name: int(name[5:]))
    except ValueError as error:
        raise ValueError("Demonstrations must be named demo_0, demo_1, ...") from error
    limit = entry.get("demo_limit")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("demo_limit must be a positive integer")
        demos = demos[:limit]
    if not demos:
        raise ValueError(f"{hdf5.filename}: selected demonstration split is empty")
    return demos


def _dataset(group, key, length):
    """Require real stored values, so missing supervision cannot become zero labels."""
    if key not in group or not isinstance(group[key], h5py.Dataset):
        raise ValueError(f"{group.file.filename}: missing {group.name}/{key}")
    value = group[key]
    if value.ndim < 1 or value.shape[0] != length or value.dtype.kind not in "buif":
        raise ValueError(f"{value.name}: expected a numeric array with {length} timesteps")
    return value


def _blocks(dataset):
    """Check numeric signals in bounded chunks without loading camera sequences."""
    for start in range(0, len(dataset), 4096):
        values = np.asarray(dataset[start:start + 4096], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{dataset.file.filename}: {dataset.name} contains NaN or infinity")
        yield values


def prepare_cami_datasets(config):
    """Resolve paths, load privileged keys, validate selected demos, and fit scale.

    A null continuous-contact force_scale requests std(||F_xyz||) + 1e-6,
    fitted only on the selected training demonstrations. Explicit positive
    scales are preserved. The resolved value is saved in the run configuration.
    """
    if config.algo_name not in CAMI_ALGOS:
        return
    train = config.train
    cami = config.algo.cami
    lcp = config.algo_name != "bc_cami"
    continuous = (not lcp and cami.get("enabled", False)
                  and cami.get("continuous_contact", {}).get("enabled", False))
    force_key = (cami.lcp.force_dataset_key if lcp else
                 cami.continuous_contact.force_dataset_key if continuous else None)
    label_key = None
    if config.algo_name == "bc_cami" and cami.get("enabled", False) and not continuous:
        label_key = cami.get("contact_label_key", "contact_label")
    elif config.algo_name == "bc_cami_cance" and cami.lcp.negative_mode == "regime":
        label_key = "contact_label"

    if not config.algo.rnn.enabled or config.algo.rnn.open_loop:
        raise ValueError("CaMI requires rnn.enabled=true and rnn.open_loop=false")
    minimum = cami.snippet_horizon + 1 if continuous else 2 if lcp else 1
    if train.seq_length < minimum or train.frame_stack != 1:
        raise ValueError(f"This CaMI configuration requires frame_stack=1 and seq_length>={minimum}")
    if lcp and train.pad_seq_length:
        raise ValueError("LCP variants require pad_seq_length=false to keep consecutive pairs unpadded")

    entries = train.data
    if isinstance(entries, str):
        entries = [{"path": entries}]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Set train.data to [{'path': 'dataset/tool_hang/ph/cami_with_force.hdf5'}]")
    entries = [dict(entry, path=resolve_dataset_path(entry.get("path"))) for entry in entries]
    keys = list(train.dataset_keys)
    for key in (force_key, label_key):
        if key and key not in keys:
            keys.append(key)
    # Existing legacy LCP configs requested a nonexistent root 'force' dataset.
    # Remove that redundant request once the explicit stored wrench key is used.
    if lcp and force_key != "force" and "force" in keys:
        keys.remove("force")
    with config.values_unlocked():
        train.data = entries
        train.dataset_keys = keys

    forbidden = {"force", "force_rawbias", "force_obsbias", "contact_label"}
    forbidden.update(key.split("/")[-1] for key in (force_key, label_key) if key)
    if forbidden.intersection(config.all_obs_keys):
        raise ValueError("Force and contact labels must be auxiliary dataset_keys, not policy observations")
    fit_scale = continuous and cami.continuous_contact.force_scale is None
    if continuous and not fit_scale:
        scale = float(cami.continuous_contact.force_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("continuous_contact.force_scale must be null or finite and positive")

    # Merge per-chunk variance statistics instead of concatenating whole datasets.
    count, mean, m2 = 0, 0.0, 0.0
    shapes = {}
    for entry in entries:
        with h5py.File(entry["path"], "r") as hdf5:
            if "data" not in hdf5 or "env_args" not in hdf5["data"].attrs:
                raise ValueError(f"{entry['path']}: missing data group or data.attrs['env_args']")
            metadata = json.loads(hdf5["data"].attrs["env_args"])
            if not all(key in metadata for key in ("env_name", "type", "env_kwargs")):
                raise ValueError(f"{entry['path']}: incomplete environment metadata")
            if not bool(hdf5.attrs.get("complete", True)):
                raise ValueError(f"{entry['path']}: replay output is marked incomplete")
            training = selected_demos(hdf5, entry, train.hdf5_filter_key)
            validation = []
            if config.experiment.validate:
                if train.hdf5_filter_key is None or train.hdf5_validation_filter_key is None:
                    raise ValueError("Validation requires explicit train and validation filter keys")
                validation = selected_demos(hdf5, entry, train.hdf5_validation_filter_key)
                if set(training).intersection(validation):
                    raise ValueError("Training and validation demonstrations overlap; check per-dataset filter_key")
            training = set(training)
            for name in sorted(training.union(validation)):
                demo = hdf5[f"data/{name}"]
                length = int(demo.attrs.get("num_samples", 0))
                if length < 1 or (not train.pad_seq_length and length < train.seq_length):
                    raise ValueError(f"{demo.name}: num_samples is missing, empty, or shorter than seq_length")
                for key in train.action_keys:
                    action = _dataset(demo, key, length)
                    if action.ndim != 2:
                        raise ValueError(f"{action.name}: expected [T, action_dim]")
                    for _ in _blocks(action):
                        pass
                    shape_key = ("action", key)
                    if shapes.setdefault(shape_key, action.shape[1:]) != action.shape[1:]:
                        raise ValueError(f"{action.name}: inconsistent action dimensions")
                for prefix in (["obs", "next_obs"] if train.hdf5_load_next_obs else ["obs"]):
                    for modality, obs_keys in config.observation.modalities.obs.items():
                        for key in obs_keys:
                            obs = _dataset(demo, f"{prefix}/{key}", length)
                            shape_key = ("observation", key)
                            if shapes.setdefault(shape_key, obs.shape[1:]) != obs.shape[1:]:
                                raise ValueError(f"{obs.name}: inconsistent observation shape")
                            if modality == "rgb":
                                if obs.ndim != 4 or obs.shape[-1] != 3 or obs.dtype != np.uint8:
                                    raise ValueError(f"{obs.name}: RGB observations must be uint8 [T,H,W,3]")
                                encoder = config.observation.encoder.rgb
                                if encoder.obs_randomizer_class == "CropRandomizer":
                                    crop = encoder.obs_randomizer_kwargs
                                    if crop.crop_height >= obs.shape[1] or crop.crop_width >= obs.shape[2]:
                                        raise ValueError(f"{obs.name}: configured crop must be smaller than the image")
                            elif modality == "low_dim":
                                for _ in _blocks(obs):
                                    pass
                if force_key:
                    force = _dataset(demo, force_key, length)
                    dims = (cami.lcp.force_dim,) if lcp else (3, 6)
                    if force.ndim != 2 or force.shape[1] not in dims:
                        raise ValueError(f"{force.name}: expected [T,D] with D in {dims}")
                    for values in _blocks(force):
                        if fit_scale and name in training:
                            magnitudes = np.linalg.norm(values[:, :3], axis=1)
                            n = len(magnitudes)
                            block_mean = float(magnitudes.mean())
                            delta = block_mean - mean
                            total = count + n
                            m2 += float(((magnitudes - block_mean) ** 2).sum()) + delta ** 2 * count * n / total
                            mean += delta * n / total
                            count = total
                checked_labels = {key for key in (label_key, "contact_label" if "contact_label" in keys else None) if key}
                for key in checked_labels:
                    labels = _dataset(demo, key, length)
                    if labels.shape not in ((length,), (length, 1)):
                        raise ValueError(f"{labels.name}: expected one binary contact label per timestep")
                    for values in _blocks(labels):
                        if not np.isin(values, [0, 1]).all():
                            raise ValueError(f"{labels.name}: contact labels must be 0 or 1")
    if fit_scale:
        if count == 0 or m2 <= 0:
            raise ValueError("Cannot fit force_scale: training force magnitudes have no variation")
        with config.values_unlocked():
            cami.continuous_contact.force_scale = float(np.sqrt(m2 / count) + 1e-6)
