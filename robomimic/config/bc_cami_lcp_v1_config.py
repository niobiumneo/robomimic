"""
Config for BC + Continuous Contact-Aware Mutual Information (LCP formulation).

Mirrors bc_cami_config.py's structure for the discrete BC_CaMI algorithm,
but declares the algo.cami.lcp.* fields that BC_CaMI_LCP._create_networks
and _compute_lcp_loss actually read (see bc_cami_lcp.py), and registers
optimizer entries for gap_encoder/impulse_encoder instead of the discrete
branch's state_encoder/snippet_encoder/key_proj.

ALGO_NAME must match the string passed to register_algo_factory_func in
bc_cami_lcp.py ("bc_cami_lcp"), and this file must be imported somewhere
config_factory can discover it (the same way bc_cami_config.py needed to be
importable for "bc_cami" to resolve), or config_factory("bc_cami_lcp") will
not find this class.
"""

from robomimic.config.bc_config import BCConfig
from copy import deepcopy


class BCCaMILCPConfig(BCConfig):
    ALGO_NAME = "bc_cami_lcp"

    def train_config(self):
        super(BCCaMILCPConfig, self).train_config()
        # Unlike the discrete branch, we do NOT need hdf5_load_next_obs:
        # phi_hat(t+1) comes from the next timestep already present in the
        # loaded sequence window (obs_encoding[:, 1, :]), not a separate
        # "next_obs" batch entry.
        self.train.hdf5_load_next_obs = False

    def observation_config(self):
        super(BCCaMILCPConfig, self).observation_config()

        self.observation.modalities.obs.rgb = ["agentview_image", "robot0_eye_in_hand_image"]
        self.observation.modalities.obs.low_dim = [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"
        ]

        self.observation.modalities.goal.rgb = []
        self.observation.modalities.goal.low_dim = []

    def algo_config(self):
        super(BCCaMILCPConfig, self).algo_config()

        # Optimizer entries for the LCP branch's two new trainable modules.
        # BC_CaMI_LCP._train_step requires exactly these three optimizer
        # names to exist ("policy", "gap_encoder", "impulse_encoder"); the
        # discrete branch's state_encoder/snippet_encoder/key_proj entries
        # are NOT created here since BC_CaMI_LCP builds no such networks.
        self.algo.optim_params.gap_encoder = deepcopy(self.algo.optim_params.policy)
        self.algo.optim_params.impulse_encoder = deepcopy(self.algo.optim_params.policy)

        self.algo.cami.enabled = True

        # Nested "lcp" sub-config read by BC_CaMI_LCP._create_networks and
        # _compute_lcp_loss (Eq. 4.16/4.20 in the thesis derivation).
        self.algo.cami.lcp.loss_weight = 0.01     # w_lcp, Eq. 4.21
        self.algo.cami.lcp.w_comp = 0.01          # complementarity term weight, Eq. 4.20
        self.algo.cami.lcp.w_pen = 100.0          # non-penetration penalty weight, Eq. 4.20

        self.algo.cami.lcp.num_contacts = 1       # p, number of contact points
        self.algo.cami.lcp.force_dim = 6          # raw wrench dimensionality [fx,fy,fz,tx,ty,tz]

        self.algo.cami.lcp.gap_hidden_dims = (256, 256)   # E_v MLP hidden layers
        self.algo.cami.lcp.impulse_hidden_dims = (128,)   # E_f MLP hidden layers