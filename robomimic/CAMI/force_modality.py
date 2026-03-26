import numpy as np
import torch
from robomimic.utils.obs_utils import Modality


class ForceModality(Modality):
    name = "force"

    # fill these from dataset stats
    FORCE_MEAN = None
    FORCE_STD = None
    EPS = 1e-6
    CLIP = 10.0

    @classmethod
    def set_normalization_stats(cls, mean, std, clip=5.0):
        cls.FORCE_MEAN = np.asarray(mean, dtype=np.float32)
        cls.FORCE_STD = np.asarray(std, dtype=np.float32)
        cls.CLIP = clip

    @classmethod
    def _default_obs_processor(cls, obs):
        if cls.FORCE_MEAN is None or cls.FORCE_STD is None:
            raise RuntimeError("ForceModality normalization stats were not set.")

        if isinstance(obs, np.ndarray):
            out = (obs.astype(np.float32) - cls.FORCE_MEAN) / (cls.FORCE_STD + cls.EPS)
            return np.clip(out, -cls.CLIP, cls.CLIP)

        # torch tensor case
        mean = torch.as_tensor(cls.FORCE_MEAN, device=obs.device, dtype=torch.float32)
        std = torch.as_tensor(cls.FORCE_STD, device=obs.device, dtype=torch.float32)
        out = (obs.float() - mean) / (std + cls.EPS)
        return torch.clamp(out, -cls.CLIP, cls.CLIP)

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        if cls.FORCE_MEAN is None or cls.FORCE_STD is None:
            return obs

        if isinstance(obs, np.ndarray):
            return obs * (cls.FORCE_STD + cls.EPS) + cls.FORCE_MEAN

        mean = torch.as_tensor(cls.FORCE_MEAN, device=obs.device, dtype=torch.float32)
        std = torch.as_tensor(cls.FORCE_STD, device=obs.device, dtype=torch.float32)
        return obs * (std + cls.EPS) + mean