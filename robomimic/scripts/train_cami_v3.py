"""Select a v3 experiment while retaining robomimic's standard trainer."""

import argparse
from pathlib import Path

from robomimic.scripts.generate_cami_v3_configs import VARIANT_CONFIGS
from robomimic.scripts.train import main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANT_CONFIGS, default="warmup")
    parser.add_argument("--config", help="Optional custom v3 JSON instead of a bundled variant")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = str(Path(__file__).resolve().parents[1] / "exps/templates" / VARIANT_CONFIGS[args.variant])
    args.algo = "bc_cami_v3"
    return args


if __name__ == "__main__":
    main(parse_args())
