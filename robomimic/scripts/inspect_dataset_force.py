import h5py

path = "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/cami_datasets/square_image_84_with_force.hdf5"

bad = []

with h5py.File(path, "r") as f:
    for ep in f["data"].keys():
        grp = f[f"data/{ep}"]
        num_samples = int(grp.attrs["num_samples"])

        has_force = "force" in grp["obs"]

        if not has_force:
            bad.append((ep, "missing obs/force", num_samples, None))
            continue

        force_shape = grp["obs"]["force"].shape

        # Check that it is a 2D array (time, features)
        if len(force_shape) != 2:
            bad.append((ep, "obs/force wrong rank", num_samples, force_shape))
            continue

        # Check that the number of timesteps matches num_samples
        if force_shape[0] != num_samples:
            bad.append((ep, "obs/force length mismatch", num_samples, force_shape))
            continue

        # Check that the force vector dimension is exactly 6 (Fx, Fy, Fz, Tx, Ty, Tz)
        if force_shape[1] != 6:
            bad.append((ep, f"obs/force wrong vector dimension (expected 6, got {force_shape[1]})", num_samples, force_shape))