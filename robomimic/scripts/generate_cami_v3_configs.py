"""Generate separate v3 ablation configs without replacing existing files."""

import argparse
from copy import deepcopy
import json
from pathlib import Path


VARIANT_CONFIGS = {
    "baseline": "bc_rnn_square_v3_baseline.json",
    "baseline_reference": "bc_rnn_square_v3_baseline_reference.json",
    "reference": "bc_cami_square_contactNetsv3_reference.json",
    "batch32": "bc_cami_square_contactNetsv3_batch32.json",
    "batch64": "bc_cami_square_contactNetsv3_batch64.json",
    "history": "bc_cami_square_contactNetsv3_history.json",
    "warmup": "bc_cami_square_contactNetsv3.json",
    "prediction": "bc_cami_square_contactNetsv3_prediction.json",
    "prediction_full": "bc_cami_square_contactNetsv3_prediction_full.json",
    "contact": "bc_cami_square_contactNetsv3_contact.json",
    "faithful": "bc_cami_square_contactNetsv3_faithful.json",
    "baseline_faithful": "bc_rnn_square_v3_baseline_faithful.json",
}


def _faithful(config):
    """Settings of the formulation report (October 2026) that the report leaves to the experiment.

    The report's loss, weights, EMA update, anchor, snippet horizon and example sizes are used; its
    descriptor is not: the three reconstructed channels (gap, normal and tangential impulse) replace the
    recorded force magnitude, so this is the report's method on a different descriptor. The anchor is the
    first policy feature (offset 0), so the sequence holds the anchor and its ten future actions. The state
    loss weight is constant, with no prediction head. The report's values are set for every number it gives:
    lambda_s 0.01, lambda_t 1, EMA momentum 0.995, snippet horizon 10, Huber delta 0.1, contact temperature
    0.05, gamma 1, MSE-only BC, the example encoder sizes (1024-d policy feature, state projector
    1024-128-128, one-layer LSTM of 256 and a 256-128-128 trajectory head). Descriptor channels are averaged
    (weights 1/3) so a pair that differs by x in every channel gets the weight of the report's single-channel
    curve, not three times its distance. The remaining choices keep the pair a plain BC-RNN run through the
    v3 adapter plus the CaMI terms: every demonstration used, no validation split, and the training sequence
    length is also the rollout hidden-state reset period. A missing gap drops its whole timestep, as v3 did
    before per-channel masking, rather than only the gap channel.
    """
    train, cami, algo = config["train"], config["algo"]["cami"], config["algo"]
    train.update(seq_length=cami["snippet_horizon"] + 1, batch_size=16, seed=0, hdf5_filter_key=None,
                 hdf5_validation_filter_key=None)
    config["experiment"]["validate"] = False
    algo["rnn"]["horizon"] = train["seq_length"]
    cami.update(state_loss_weight=.01, traj_loss_weight=1., momentum=.995, use_momentum_target=True,
                snippet_horizon=10, temperature=.07, normalize_embeddings=True, contrastive_dim=128,
                query_proj_layers=[128], key_proj_layers=[128], snippet_hidden_dim=256, snippet_num_layers=1)
    cami["continuous"].update(feature_weights=[1. / 3] * 3, huber_delta=.1, contact_temperature=.05,
                              gamma=1., distance_penalty="huber", objective="weighted")
    cami["v3"].update(context_steps=0, warmup_epochs=0, missing_features="timestep")
    cami["v3"]["prediction"]["enabled"] = False
    algo["loss"].update(l2_weight=1, l1_weight=0, cos_weight=0)


def make_configs():
    root = Path(__file__).resolve().parents[1] / "exps/templates"
    base = json.loads((root / "bc_cami_square_contactNetsv2.json").read_text())
    base["algo_name"] = "bc_cami_v3"
    base["train"].update(batch_size=32, seq_length=16, seed=0)
    base["train"]["data"] = [{"path": "dataset/square_image_84_contactnets_state.hdf5"}]
    optim = base["algo"]["optim_params"]
    base["algo"]["optim_params"] = {"policy": deepcopy(optim["policy"]), "cami": deepcopy(optim["policy"])}
    # v3 trains at a constant learning rate; the v2 template halves it at epoch 1000.
    for name in ("policy", "cami"):
        base["algo"]["optim_params"][name]["learning_rate"].update(decay_factor=.1, epoch_schedule=[])
    base["algo"]["cami"]["v3"] = {
        "context_steps": 5, "warmup_epochs": 0, "gradient_diagnostics_every": 100,
        "missing_features": "feature",
        "descriptor": {"gap_min": None, "impulse_log_scale": None},
        "prediction": {"enabled": False, "features": ["gap"], "hidden_dims": [128],
                       "loss_weight": .01, "huber_delta": .1},
    }
    configs = {}
    for variant, filename in VARIANT_CONFIGS.items():
        config = deepcopy(base)
        cami, train = config["algo"]["cami"], config["train"]
        config["experiment"]["name"] = "square_cami_v3_" + variant
        if variant in ("baseline_reference", "reference", "batch32", "batch64"):
            train["seq_length"] = 11
            cami["v3"].update(context_steps=0, warmup_epochs=0)
        if variant in ("baseline_reference", "reference", "prediction"):
            train["batch_size"] = 16
        if variant == "prediction":
            train["num_epochs"] = 5000
        if variant == "batch64":
            train["batch_size"] = 64
        if variant == "history":
            cami["v3"]["warmup_epochs"] = 0
        if variant.endswith("faithful"):
            _faithful(config)
        if variant.startswith("baseline"):
            cami["enabled"] = False
            train["dataset_keys"] = ["actions"]
            config["experiment"]["name"] = "square_bc_rnn_v3_" + variant
        if variant.startswith("prediction"):
            cami["v3"]["prediction"]["enabled"] = True
        if variant in ("prediction_full", "contact"):
            cami["v3"]["prediction"]["enabled"] = True
            cami["v3"]["prediction"]["features"] = list(cami["continuous"]["feature_names"])
        if variant == "contact":
            # Gaps and impulses both shape the weights: equal shares (sum 1, so pairs that differ in
            # contact do not all saturate at W ~ 1), deep penetration clipped at 5 mm, impulses
            # log-compressed above 0.1 N*s.
            cami["continuous"]["feature_weights"] = [1. / 3] * 3
            cami["v3"]["descriptor"].update(gap_min=-.005, impulse_log_scale=.1)
            # Gaps recomputed with the per-pair rule (robomimic/scripts/recompute_cami_gaps.py).
            train["data"] = [{"path": "dataset/square_image_84_contactnets_state_gapfix.hdf5"}]
        configs[filename] = config
    return configs


def write_configs(output_dir):
    output_dir = Path(output_dir)
    configs = make_configs()
    existing = [str(output_dir / name) for name in configs if (output_dir / name).exists()]
    if existing:
        raise FileExistsError("Refusing to replace existing configs: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        with (output_dir / name).open("x") as output:
            json.dump(config, output, indent=4)
            output.write("\n")
    return list(configs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True,
                        help="New destination directory; existing config files are never replaced")
    args = parser.parse_args()
    for name in write_configs(args.output_dir):
        print(name)
