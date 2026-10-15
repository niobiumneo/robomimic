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
}


def make_configs():
    root = Path(__file__).resolve().parents[1] / "exps/templates"
    base = json.loads((root / "bc_cami_square_contactNetsv2.json").read_text())
    base["algo_name"] = "bc_cami_v3"
    base["train"].update(batch_size=32, seq_length=16, seed=0)
    base["train"]["data"] = [{"path": "dataset/square_image_84_contactnets_state.hdf5"}]
    optim = base["algo"]["optim_params"]
    base["algo"]["optim_params"] = {"policy": deepcopy(optim["policy"]), "cami": deepcopy(optim["policy"])}
    base["algo"]["cami"]["v3"] = {
        "context_steps": 5, "warmup_epochs": 100, "gradient_diagnostics_every": 100,
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
        if variant in ("baseline_reference", "reference"):
            train["batch_size"] = 16
        if variant == "batch64":
            train["batch_size"] = 64
        if variant == "history":
            cami["v3"]["warmup_epochs"] = 0
        if variant.startswith("baseline"):
            cami["enabled"] = False
            train["dataset_keys"] = ["actions"]
            config["experiment"]["name"] = "square_bc_rnn_v3_" + variant
        if variant.startswith("prediction"):
            cami["v3"]["prediction"]["enabled"] = True
        if variant == "prediction_full":
            cami["v3"]["prediction"]["features"] = list(cami["continuous"]["feature_names"])
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
