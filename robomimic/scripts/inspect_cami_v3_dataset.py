"""Report what a prepared physics dataset gives the CaMI v3 negative weights.

Run it on the dataset a v3 variant trains on, with no GPU and no training:

    python -m robomimic.scripts.inspect_cami_v3_dataset \
        --dataset dataset/square_image_84_contactnets_state.hdf5 --variant prediction

It answers, from the labels alone: how much of the data is in contact, whether the
three descriptor channels saturate, and how the weights W_ij of a batch distribute
(the same distances and weights the training module computes, from the same
functions). Anchors are drawn the way the training loader draws them: every
timestep of the training demonstrations is a sequence start, the anchor sits
``context_steps`` into the sequence, and anchors without a real future action
have no eligible negatives.

Every number is a property of the labels. Nothing here measures whether the state
loss helps the policy; read the W&B ``CaMI_V3/...`` metrics for that.
"""
import argparse
import copy
import json
import os
import sys

import h5py
import numpy as np
import torch

from robomimic.CAMI.continuous_contact import contact_distance, negative_weights, transform_descriptors
from robomimic.config import config_factory
from robomimic.utils.cami_dataset_utils import resolve_dataset_path

CHANNELS = ("gap", "normal_impulse", "tangential_impulse")
CLASSES = ("free", "near", "contact")


def load_config(variant=None, path=None):
    if (variant is None) == (path is None):
        raise ValueError("Pass exactly one of --variant or --config")
    if path is not None:
        with open(path) as f:
            ext = json.load(f)
    else:
        from robomimic.scripts.generate_cami_v3_configs import VARIANT_CONFIGS, make_configs
        if variant not in VARIANT_CONFIGS:
            raise ValueError("Unknown variant {!r}; choose from {}".format(variant, sorted(VARIANT_CONFIGS)))
        ext = make_configs()[VARIANT_CONFIGS[variant]]
    config = config_factory(ext["algo_name"])
    with config.values_unlocked():
        config.update(copy.deepcopy(ext))
    return config


def dataset_path(config, override=None):
    """The one dataset the report describes. Several entries, per-entry filters or demo limits are refused:
    the trainer fits scales over every entry and applies each entry's own filter, which this report does not."""
    entries = config.train.data
    if isinstance(entries, str):
        entries = [{"path": entries}]
    entries = list(entries or [])
    if len(entries) != 1 or any(k in entries[0] for k in ("filter_key", "demo_limit")):
        raise ValueError("The report describes a single dataset entry without its own filter_key or demo_limit; "
                         "this config has {} entries (or per-entry settings)".format(len(entries)))
    return str(resolve_dataset_path(override if override else entries[0]["path"]))


def _names(hdf5, key):
    if "mask" not in hdf5 or key not in hdf5["mask"]:
        return None
    return [x.decode() if isinstance(x, bytes) else str(x) for x in hdf5["mask"][key][()]]


def load_pool(path, config):
    """Return per-anchor windows for the training demonstrations, plus provenance."""
    train, algo = config.train, config.algo.cami
    horizon, context = algo.snippet_horizon, algo.v3.context_steps
    with h5py.File(path, "r") as f:
        meta = json.loads(f["data"].attrs.get("cami_c", "{}"))
        if meta.get("feature_names") != ["gap", "normal_impulse", "tangential_impulse_magnitude"]:
            raise ValueError("{} is not a prepared physics dataset (run prepare_cami_c_dataset --mode physics)".format(path))
        if meta.get("horizon") != horizon:
            raise ValueError("Dataset horizon {} differs from the config's snippet_horizon {}".format(meta.get("horizon"), horizon))
        key = train.hdf5_filter_key
        names = _names(f, key) if key else None
        if key and names is None:
            raise ValueError("Filter key {!r} is not in {}".format(key, path))
        names = names if names is not None else sorted(f["data"], key=lambda x: int(x.split("_")[-1]))
        windows, masks, lengths, demo_ids, steps, series = [], [], [], [], [], {}
        for index, name in enumerate(names):
            demo = f["data"][name]
            state, mask = demo["contact_state"][()], demo["contact_mask"][()].astype(bool)
            length = state.shape[0]
            if length < 1 or (not train.pad_seq_length and length < train.seq_length):
                raise ValueError("{}: invalid num_samples or sequence length".format(demo.name))
            series[name] = (state, mask)
            # pad_seq_length lets every timestep start a sequence; otherwise starts stop seq_length-1 early.
            count = length if train.pad_seq_length else length - train.seq_length + 1
            for start in range(count):
                t = start + context
                valid = max(0, min(horizon, length - 1 - t)) if t < length else 0
                windows.append(state[t] if t < length else np.zeros_like(state[0]))
                masks.append(mask[t] if t < length else np.zeros_like(mask[0]))
                lengths.append(valid)
                demo_ids.append(index)
                steps.append(t)
    pool = dict(windows=np.asarray(windows, dtype=np.float64), mask=np.asarray(masks), lengths=np.asarray(lengths),
                demo=np.asarray(demo_ids), step=np.asarray(steps), series=series, names=names, meta=meta)
    action_valid = np.arange(horizon)[None] < pool["lengths"][:, None]
    pool["mask"] = pool["mask"] & action_valid[..., None]
    return pool


def descriptor_transform(config):
    """The v3 descriptor transform the trainer applies before scaling."""
    d = config.algo.cami.v3.descriptor
    return dict(gap_min=d.gap_min, impulse_log_scale=d.impulse_log_scale)


def fit_scales(pool, config):
    """Training-demo std + 1e-6 of offset-1 values after the descriptor transform, as the trainer fits them
    (or the config's explicit scales)."""
    c = config.algo.cami.continuous
    explicit = [c.gap_scale, c.normal_impulse_scale, c.tangential_impulse_scale]
    transform = descriptor_transform(config)
    scales = []
    for i, given in enumerate(explicit):
        if given is not None:
            scales.append(float(given))
            continue
        values = np.concatenate([transform_descriptors(s[0][:, 0].astype(np.float64), **transform)[:, i][s[1][:, 0, i]]
                                 for s in pool["series"].values()])
        scales.append(float(values.std() + 1e-6))
    return scales


def channel_statistics(pool, scales, cap, transform=None):
    """Physical statistics of each channel; the normalized quantiles are what the loss sees, after the transform."""
    rows = []
    values = [np.concatenate([s[0][:, 0, i][s[1][:, 0, i]] for s in pool["series"].values()]).astype(np.float64)
              for i in range(3)]
    total = sum(len(s[0]) for s in pool["series"].values())
    for i, (name, x) in enumerate(zip(CHANNELS, values)):
        q = np.quantile(x, [.5, .9, .99]) if len(x) else np.zeros(3)
        # Both transforms are monotonic, so they map quantiles to quantiles.
        n = transform_descriptors(np.tile(q[:, None], (1, 3)), **(transform or {}))[:, i] / scales[i]
        rows.append(dict(
            channel=name, valid_fraction=len(x) / max(1, total), scale=scales[i], mean=float(x.mean()), std=float(x.std()),
            at_cap=float((x >= cap - 1e-6).mean()) if i == 0 else None,
            penetrating=float((x <= 0).mean()) if i == 0 else None,
            exactly_zero=float((x == 0).mean()) if i else None,
            p50=float(q[0]), p90=float(q[1]), p99=float(q[2]),
            normalized_p50=float(n[0]), normalized_p90=float(n[1]), normalized_p99=float(n[2])))
    return rows


def unobserved_gap_rows(pool):
    """Rows with a measured normal impulse but no valid gap. With missing_features="timestep" a pair distance drops
    a whole timestep when any weighted channel is missing, so these rows contribute no impulse to any W_ij while the
    gap weight is positive; "feature" keeps their impulses."""
    rows = impulse_rows = 0
    for state, mask in pool["series"].values():
        measured = mask[:, 0, 1] & (state[:, 0, 1] > 0)
        impulse_rows += int(measured.sum())
        rows += int((measured & ~mask[:, 0, 0]).sum())
    return dict(rows=rows, impulse_rows=impulse_rows, fraction=rows / impulse_rows if impulse_rows else 0.)


def classify(pool, cap, threshold):
    """free: every valid gap sits at the cap and impulses are zero; contact: a normal impulse above threshold."""
    w, m = pool["windows"], pool["mask"]
    gap_free = np.where(m[..., 0], w[..., 0] >= cap - 1e-6, True).all(1)
    quiet = np.where(m[..., 1], w[..., 1] <= 0, True).all(1) & np.where(m[..., 2], w[..., 2] <= 0, True).all(1)
    contact = np.where(m[..., 1], w[..., 1] > threshold, False).any(1)
    label = np.where(contact, 2, np.where(gap_free & quiet, 0, 1))
    return np.where(pool["lengths"] > 0, label, -1)


def onset_lags(pool, tolerance=3):
    """Offset between each gap<=0 run and the nearest normal-impulse onset, in descriptor rows.

    Rows hold [gap at the start state of an action, impulse during that action]. A contact that
    begins inside an action therefore shows its impulse one row before its gap closes (-1); a contact
    that exists at the start state shows both in the same row (0).

    A row whose gap is unobserved is skipped: it neither ends a touching run nor starts one, so the run
    starts at the first observed gap<=0 after an observed gap>0 (or at the first observed row). Impulse
    onsets come from the impulse mask alone. When the gap is unobserved at the true closing, the lag is
    measured from the first observed row, so a lead of more than one row can partly be unobserved rows.
    """
    counts = {"<=-2": 0, "-1": 0, "0": 0, "+1": 0, ">=+2": 0, "no impulse nearby": 0}
    for state, mask in pool["series"].values():
        gap, impulse = state[:, 0, 0], state[:, 0, 1]
        on = (impulse > 1e-9) & mask[:, 0, 1]
        onsets = [k for k in range(len(on)) if on[k] and (k == 0 or not on[k - 1])]
        touching_before = False
        for j in range(len(gap)):
            if not mask[j, 0, 0]:
                continue
            touching = bool(gap[j] <= 0)
            if touching and not touching_before:
                window = [k for k in onsets if abs(k - j) <= tolerance]
                if not window:
                    counts["no impulse nearby"] += 1
                else:
                    lag = min(window, key=lambda k: (abs(k - j), k)) - j
                    counts["<=-2" if lag <= -2 else "-1" if lag == -1 else "0" if lag == 0 else "+1" if lag == 1 else ">=+2"] += 1
            touching_before = touching
    return counts


def weight_regime(pool, scales, config, labels, batch_size, batches, seed, overrides=None, index_batches=None):
    """Weights of random batches (or of the given index batches, for tests against the training module)."""
    if batch_size > len(pool["lengths"]):
        raise ValueError("batch size {} exceeds the {} anchors training would draw from".format(batch_size, len(pool["lengths"])))
    c = config.algo.cami.continuous
    settings = dict(feature_weights=list(c.feature_weights), contact_temperature=c.contact_temperature, gamma=c.gamma,
                    huber_delta=c.huber_delta, distance_penalty=c.distance_penalty, chunk_size=c.chunk_size,
                    missing_features=config.algo.cami.v3.missing_features)
    settings.update(overrides or {})
    rng = np.random.default_rng(seed)
    scale = torch.as_tensor(scales, dtype=torch.float64)
    masks = torch.as_tensor(pool["mask"])
    windows = transform_descriptors(torch.where(masks, torch.as_tensor(pool["windows"]), torch.zeros(())),
                                    **descriptor_transform(config))
    lengths = torch.as_tensor(pool["lengths"])
    near_zero, near_one, eff, loss, active, mid = [], [], [], [], [], []
    pair_count, pair_mass = np.zeros((3, 3)), np.zeros((3, 3))
    for step in range(batches if index_batches is None else len(index_batches)):
        idx = torch.as_tensor(rng.choice(len(windows), size=batch_size, replace=False) if index_batches is None
                              else np.asarray(index_batches[step]))
        mask, length = masks[idx], lengths[idx]
        normalized = torch.where(mask, windows[idx], torch.zeros(())) / scale
        distance, valid = contact_distance(normalized, mask, settings["feature_weights"], settings["huber_delta"],
                                           settings["chunk_size"], penalty=settings["distance_penalty"],
                                           missing=settings["missing_features"])
        valid &= (length > 0)[:, None] & (length > 0)[None]
        w = negative_weights(distance, valid, settings["contact_temperature"], settings["gamma"])
        eye = torch.eye(batch_size, dtype=torch.bool)
        eligible = valid & ~eye
        if eligible.any():
            values = w[eligible]
            near_zero.append(float((values < .01).double().mean()))
            near_one.append(float((values > .99).double().mean()))
            mid.append(values.median().item())
        eff.append(float(w.sum(1).mean()))
        loss.append(float(torch.log1p(w.sum(1)).mean()))  # all logits equal: what an uninformed embedding scores
        active.append(float((w > 0).any(1).double().mean()))
        lab = labels[idx.numpy()]
        for i in range(3):
            for j in range(3):
                sel = eligible.numpy() & (lab[:, None] == i) & (lab[None] == j)
                pair_count[i, j] += sel.sum()
                pair_mass[i, j] += w.numpy()[sel].sum()
    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return dict(near_zero_fraction=mean(near_zero), near_one_fraction=mean(near_one), median_weight=mean(mid),
                effective_negatives=mean(eff), anchors_with_any_negative=mean(active),
                uninformed_state_loss=mean(loss), pair_count=pair_count, pair_mass=pair_mass)


def build_report(args):
    config = load_config(args.variant, args.config)
    path = dataset_path(config, args.dataset)
    pool = load_pool(path, config)
    meta = pool["meta"]
    scales = fit_scales(pool, config)
    cap = meta.get("gap_cap", .02)
    threshold = meta.get("binary_threshold", 0.)
    labels = classify(pool, cap, threshold)
    batch_size = args.batch_size or config.train.batch_size
    eligible = pool["lengths"] > 0
    report = dict(dataset=str(path), variant=args.variant or args.config, batch_size=batch_size, anchor_offset=config.algo.cami.v3.context_steps,
                  horizon=config.algo.cami.snippet_horizon, train_demos=len(pool["names"]), anchors=int(len(labels)),
                  eligible_anchor_fraction=float(eligible.mean()), gap_cap=cap, binary_threshold=threshold,
                  reconstruction=meta.get("reconstruction"), channels=channel_statistics(pool, scales, cap, descriptor_transform(config)),
                  classes={name: float((labels == i).sum() / max(1, eligible.sum())) for i, name in enumerate(CLASSES)},
                  onset_lag=onset_lags(pool), unobserved_gap=unobserved_gap_rows(pool), settings={}, regimes={},
                  explicit_scales=[v for v in (config.algo.cami.continuous.gap_scale, config.algo.cami.continuous.normal_impulse_scale,
                                               config.algo.cami.continuous.tangential_impulse_scale) if v is not None])
    c = config.algo.cami.continuous
    weights = np.asarray(c.feature_weights, dtype=float)
    cases = [("config as trained", {}),
             ("weights / sum (= contact_temperature x sum)", dict(feature_weights=(weights / weights.sum()).tolist())),
             ("gap channel only", dict(feature_weights=[weights[0], 0., 0.])),
             ("impulse channels only", dict(feature_weights=[0., weights[1], weights[2]]))]
    report["settings"] = dict(feature_weights=weights.tolist(), contact_temperature=c.contact_temperature, gamma=c.gamma,
                              huber_delta=c.huber_delta, distance_penalty=c.distance_penalty, scales=scales,
                              missing_features=config.algo.cami.v3.missing_features,
                              **descriptor_transform(config))
    for label, overrides in cases:
        report["regimes"][label] = weight_regime(pool, scales, config, labels, batch_size, args.batches, args.seed, overrides)
    return report


def flags(report):
    out = []
    classes = report["classes"]
    if report["reconstruction"] and report["reconstruction"].get("transitions_over_tolerance"):
        out.append("WARN {} transitions exceeded the replay tolerance; labels may lag the observations".format(
            report["reconstruction"]["transitions_over_tolerance"]))
    if classes["contact"] < .05:
        out.append("WARN only {:.1%} of anchors have a window with an impulse above the binary threshold".format(classes["contact"]))
    unobserved = report["unobserved_gap"]
    if (unobserved["fraction"] > .05 and report["settings"]["feature_weights"][0] > 0
            and report["settings"]["missing_features"] == "timestep"):
        out.append("WARN {:.0%} of the rows with a measured impulse have no valid gap; every pair distance drops them "
                   "while the gap weight is positive".format(unobserved["fraction"]))
    current = report["regimes"]["config as trained"]
    total = current["pair_count"].sum()
    free_free = current["pair_count"][0, 0] / total if total else 0.
    if current["near_zero_fraction"] > .5:
        out.append("NOTE {:.0%} of pairs have W<0.01 ({:.0%} are free-vs-free pairs, which have W=0)".format(
            current["near_zero_fraction"], free_free))
    if current["near_one_fraction"] > .5:
        out.append("NOTE {:.0%} of pairs have W>0.99: the weights barely discriminate between negatives".format(
            current["near_one_fraction"]))
    if current["anchors_with_any_negative"] < .5:
        out.append("NOTE only {:.0%} of anchors have a negative with W>0 in a batch of {}".format(
            current["anchors_with_any_negative"], report["batch_size"]))
    return out


def print_report(report):
    p = lambda *a: print(*a, flush=True)
    p("Dataset      :", report["dataset"])
    p("Variant      : {} | batch {} | anchor offset {} | horizon {}".format(
        report["variant"], report["batch_size"], report["anchor_offset"], report["horizon"]))
    p("Training demos: {} | anchors {} | with a real future action {:.1%}".format(
        report["train_demos"], report["anchors"], report["eligible_anchor_fraction"]))
    r = report["reconstruction"]
    if r:
        p("Reconstruction: {} | mujoco {} | robosuite {} | max qpos drift {:.3g} | over tolerance {}/{}".format(
            r.get("replay_mode"), r.get("mujoco_version"), r.get("robosuite_version"), r.get("max_qpos_replay_error", float("nan")),
            r.get("transitions_over_tolerance"), r.get("checked_transitions")))
    explicit = report["explicit_scales"]
    p("\nChannels (offset-1 values of training demos; scale = {})".format(
        "std + 1e-6, or the configured value where one is set" if explicit else "std + 1e-6"))
    p("  {:<20}{:>8}{:>11}{:>9}{:>9}{:>9}{:>9}{:>9}".format("channel", "valid", "scale", "at cap", "<=0", "==0", "n.p90", "n.p99"))
    fmt = lambda x: "-" if x is None else "{:.1%}".format(x)
    for ch in report["channels"]:
        p("  {:<20}{:>8}{:>11.4g}{:>9}{:>9}{:>9}{:>9.2f}{:>9.2f}".format(
            ch["channel"], fmt(ch["valid_fraction"]), ch["scale"], fmt(ch["at_cap"]), fmt(ch["penetrating"]),
            fmt(ch["exactly_zero"]), ch["normalized_p90"], ch["normalized_p99"]))
    c = report["classes"]
    p("\nAnchor windows (cap {:.3g} m, binary threshold {:.3g} N*s)".format(report["gap_cap"], report["binary_threshold"]))
    p("  free {:.1%} | near (gap below the cap or small impulses) {:.1%} | contact {:.1%}".format(c["free"], c["near"], c["contact"]))
    p("  impulse onset row - gap<=0 row (rows; -1 = contact began inside the action; observed gaps only): " + ", ".join(
        "{} {}".format(k, v) for k, v in report["onset_lag"].items()))
    u = report["unobserved_gap"]
    p("  rows with a measured impulse but no valid gap: {} of {} ({:.1%})".format(u["rows"], u["impulse_rows"], u["fraction"]))
    s = report["settings"]
    p("\nWeights W_ij of random batches of {} (weights {}, tau_c {}, gamma {}, {})".format(
        report["batch_size"], [round(w, 4) for w in s["feature_weights"]], s["contact_temperature"], s["gamma"],
        "huber delta {}".format(s["huber_delta"]) if s["distance_penalty"] == "huber" else s["distance_penalty"]))
    p("  missing channels drop the {} | gap_min {} | impulse_log_scale {}".format(
        s["missing_features"], s["gap_min"], s["impulse_log_scale"]))
    p("  {:<42}{:>9}{:>9}{:>9}{:>10}{:>10}{:>11}".format("case", "W<.01", "W>.99", "median", "sum_j W", "any neg", "init loss"))
    for label, r in report["regimes"].items():
        p("  {:<42}{:>9.1%}{:>9.1%}{:>9.3f}{:>10.2f}{:>10.1%}{:>11.3f}".format(
            label, r["near_zero_fraction"], r["near_one_fraction"], r["median_weight"], r["effective_negatives"],
            r["anchors_with_any_negative"], r["uninformed_state_loss"]))
    p("  init loss = mean_i log(1 + sum_j W_ij): the state loss an embedding with all logits equal would show.")
    current = report["regimes"]["config as trained"]
    total = current["pair_mass"].sum()
    p("\nWhere the negative weight comes from, as trained (share of sum W over pairs; mean W in parentheses)")
    p("  {:<10}".format("") + "".join("{:>20}".format(name) for name in CLASSES))
    for i, name in enumerate(CLASSES):
        cells = []
        for j in range(3):
            n = current["pair_count"][i, j]
            cells.append("{:.1%} ({:.2f})".format(current["pair_mass"][i, j] / total if total else 0., current["pair_mass"][i, j] / n if n else 0.))
        p("  {:<10}".format(name) + "".join("{:>20}".format(x) for x in cells))
    for line in flags(report):
        p(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="Prepared physics dataset (default: the config's train.data path)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--variant", help="v3 variant name, e.g. prediction")
    source.add_argument("--config", help="Path to a v3 config JSON")
    parser.add_argument("--batch-size", type=int, help="Override the config's batch size")
    parser.add_argument("--batches", type=int, default=200, help="Random batches to average over")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="Also write the full report to this new file")
    args = parser.parse_args(argv)
    if args.variant is None and args.config is None:
        args.variant = "prediction"
    if args.batches < 1:
        parser.error("--batches must be positive")
    if args.json and os.path.exists(args.json):
        parser.error("--json refuses to replace an existing file: " + args.json)
    report = build_report(args)
    print_report(report)
    if args.json:
        with open(args.json, "x") as f:
            json.dump(report, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
