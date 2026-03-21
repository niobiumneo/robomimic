import h5py

path = "/home/hisham246/uwaterloo/ME780_Collaborative_Robotics/robomimic_datasets/square/ph/image_224_square_force_v15.hdf5"

with h5py.File(path, "r") as f:
    demo = list(f["data"].keys())[0]
    print("demo:", demo)
    print("obs keys:", list(f["data"][demo]["obs"].keys()))
    print("next_obs keys:", list(f["data"][demo]["next_obs"].keys()))