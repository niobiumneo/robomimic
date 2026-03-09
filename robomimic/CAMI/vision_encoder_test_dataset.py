import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from vision_encoder import DinoV2Core

HDF5_PATH = "/home/hisham246/uwaterloo/robosuite_datasets/table_wiping/1772931257_163442/demo.hdf5"
IMAGE_KEY = "agentview_image"
DEMO_NAME = None                # None = first demo automatically
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with h5py.File(HDF5_PATH, "r") as f:
    demo_names = sorted(list(f["data"].keys()))
    print("Demos:", demo_names[:10])

    if DEMO_NAME is None:
        DEMO_NAME = demo_names[0]

    print("Using demo:", DEMO_NAME)
    print("Obs keys:", list(f["data"][DEMO_NAME]["obs"].keys()))

    imgs = f["data"][DEMO_NAME]["obs"][IMAGE_KEY][:]
    print("Raw image array shape:", imgs.shape, imgs.dtype)

# expected usually [T, H, W, C]
T = imgs.shape[0]

# show a few frames
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, idx in zip(axes, [0, T // 2, T - 1]):
    ax.imshow(imgs[idx])
    ax.set_title(f"frame {idx}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# convert to torch [T, C, H, W]
x = torch.from_numpy(imgs)
if x.ndim != 4:
    raise ValueError(f"Expected 4D images, got {x.shape}")

if x.shape[-1] == 3:
    x = x.permute(0, 3, 1, 2).contiguous()

print("Torch image tensor shape:", x.shape, x.dtype)

encoder = DinoV2Core(
    input_shape=tuple(x.shape[1:]),
    model_name="facebook/dinov2-base",
    feature_dimension=128,
    freeze_backbone=True,
    use_cls_token=True,
).to(DEVICE)

encoder.eval()

with torch.no_grad():
    z = encoder(x[:20].to(DEVICE)).cpu()

print("Embedding shape:", z.shape)
print("Mean embedding norm:", z.norm(dim=1).mean().item())
print("Min embedding norm:", z.norm(dim=1).min().item())
print("Max embedding norm:", z.norm(dim=1).max().item())

# similarity check
sim_adj = F.cosine_similarity(z[:-1], z[1:], dim=1).mean().item()
perm = torch.randperm(z.shape[0])
sim_rand = F.cosine_similarity(z, z[perm], dim=1).mean().item()

print("Mean adjacent-frame cosine similarity:", sim_adj)
print("Mean random-pair cosine similarity:", sim_rand)

with torch.no_grad():
    z_begin = encoder(x[:10].to(DEVICE)).cpu()     # first 10
    z_end   = encoder(x[-10:].to(DEVICE)).cpu()    # last 10

def mean_pairwise_cosine(A, B):
    sims = []
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            sims.append(F.cosine_similarity(
                A[i:i+1], B[j:j+1], dim=1
            ).item())
    return np.mean(sims)

sim_begin_begin = mean_pairwise_cosine(z_begin, z_begin)
sim_end_end = mean_pairwise_cosine(z_end, z_end)
sim_begin_end = mean_pairwise_cosine(z_begin, z_end)

print("begin-begin similarity:", sim_begin_begin)
print("end-end similarity:", sim_end_end)
print("begin-end similarity:", sim_begin_end)