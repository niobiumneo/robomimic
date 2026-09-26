"""
Config for BC + Contrastive Contact-Aware Noise Contrastive Estimation
(C-CaNCE).

Same pattern as bc_cami_lcp_config.py, but declares the fields
BC_CaMI_CaNCE actually reads (see bc_cami_cance.py): "beta" (softmax
temperature, Eq. 4.24) and "w_mag" (scale-anchoring term, Eq. 4.29)
replace the LCP version's "w_comp", and "negative_mode" selects between
the "marginal" and "regime" negative-sampling ablation conditions
discussed alongside the derivation.

ALGO_NAME must match register_algo_factory_func("bc_cami_cance") in
bc_cami_cance.py, and this file must be importable wherever config_factory
discovers algo configs.
"""

from robomimic.config.bc_config import BCConfig
from copy import deepcopy


class BCCaMICaNCEConfig(BCConfig):
    ALGO_NAME = "bc_cami_cance"

    def train_config(self):
        super(BCCaMICaNCEConfig, self).train_config()
        # Same reasoning as BCCaMILCPConfig: pair harvesting only needs
        # the sequence window already loaded (obs_encoding[:, t, :] for
        # t = 0..T-1), never a separate next_obs batch entry.
        self.train.hdf5_load_next_obs = False

    def observation_config(self):
        super(BCCaMICaNCEConfig, self).observation_config()

        self.observation.modalities.obs.rgb = ["agentview_image", "robot0_eye_in_hand_image"]
        self.observation.modalities.obs.low_dim = [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"
        ]

        self.observation.modalities.goal.rgb = []
        self.observation.modalities.goal.low_dim = []

    def algo_config(self):
        super(BCCaMICaNCEConfig, self).algo_config()

        # Same optimizer names as BC_CaMI_LCP -- BC_CaMI_CaNCE builds the
        # identical two-network structure (gap_encoder, impulse_encoder),
        # just trains them with a different loss.
        self.algo.optim_params.gap_encoder = deepcopy(self.algo.optim_params.policy)
        self.algo.optim_params.impulse_encoder = deepcopy(self.algo.optim_params.policy)

        self.algo.cami.enabled = True

        self.algo.cami.lcp.loss_weight = 0.01     # w_nce, Eq. 4.30
        self.algo.cami.lcp.beta = 1.0              # softmax temperature, Eq. 4.24 -- UNTUNED, sweep this first
        self.algo.cami.lcp.w_pen = 100.0           # non-penetration penalty, Eq. 4.19
        self.algo.cami.lcp.w_mag = 1.0             # scale-anchoring term, Eq. 4.29

        # "marginal": negatives = any different-sequence pair in the batch.
        # "regime": negatives = different-sequence AND opposite contact
        # regime (requires per-timestep contact_label; see bc_cami_cance.py
        # process_batch_for_training for the broadcast fallback if the
        # dataset only has one label per trajectory).
        self.algo.cami.lcp.negative_mode = "regime"

        self.algo.cami.lcp.num_contacts = 1
        self.algo.cami.lcp.force_dim = 6

        self.algo.cami.lcp.gap_hidden_dims = (256, 256)
        self.algo.cami.lcp.impulse_hidden_dims = (128,)