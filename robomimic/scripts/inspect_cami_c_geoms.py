"""List exact collision geometry names for a recorded robosuite environment."""
import argparse
import json
import h5py
from robomimic.CAMI.continuous_mujoco import make_replay_env, reset_demo, native_model_data, geometry_catalog


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--demo", default="demo_0")
    args = p.parse_args()
    with h5py.File(args.dataset, "r") as file:
        env = make_replay_env(file["data"])
        try:
            reset_demo(env, file["data"][args.demo])
            model, _ = native_model_data(env.env.sim)
            print(json.dumps(geometry_catalog(model), indent=2))
        finally:
            close = getattr(env, "close", None) or getattr(env.env, "close", None)
            if close:
                close()


if __name__ == "__main__":
    main()
