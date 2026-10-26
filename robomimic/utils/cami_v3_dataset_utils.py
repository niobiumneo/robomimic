"""v3 dispatch using the existing, already prepared physics dataset."""

import math

from robomimic.CAMI.continuous_contact import check_descriptor_transform
from robomimic.utils.cami_dataset_utils import resolve_dataset_path


def prepare_cami_v3_datasets(config):
    v3 = config.algo.cami.v3
    if not isinstance(v3.context_steps, int) or v3.context_steps < 0:
        raise ValueError("v3.context_steps must be a nonnegative integer")
    if not isinstance(v3.warmup_epochs, int) or v3.warmup_epochs < 0:
        raise ValueError("v3.warmup_epochs must be a nonnegative integer")
    if v3.missing_features not in ("timestep", "feature"):
        raise ValueError("v3.missing_features must be timestep or feature")
    check_descriptor_transform(v3.descriptor.gap_min, v3.descriptor.impulse_log_scale)
    if config.train.frame_stack != 1:
        raise ValueError("v3 adapters currently require frame_stack=1")
    if config.algo.cami.enabled:
        if config.train.seq_length < v3.context_steps + config.algo.cami.snippet_horizon + 1:
            raise ValueError("v3 seq_length must cover context_steps + anchor + snippet_horizon")
        prediction = v3.prediction
        if (not math.isfinite(prediction.loss_weight) or prediction.loss_weight < 0
                or not math.isfinite(prediction.huber_delta) or prediction.huber_delta <= 0):
            raise ValueError("Prediction weight must be nonnegative and Huber delta positive")
        from robomimic.utils.contactnets_dataset_utils import prepare_contactnets_datasets
        transform = dict(gap_min=v3.descriptor.gap_min, impulse_log_scale=v3.descriptor.impulse_log_scale)
        return prepare_contactnets_datasets(config, require_rnn=False, transform=transform)
    # A true no-CaMI run does not read, fit, or require physical supervision.
    entries = config.train.data
    if isinstance(entries, str):
        entries = [{"path": entries}]
    with config.values_unlocked():
        config.train.data = [dict(entry, path=resolve_dataset_path(entry.get("path"))) for entry in entries]
        config.train.dataset_keys = [key for key in config.train.dataset_keys
                                     if key not in ("contact_state", "contact_mask")]
