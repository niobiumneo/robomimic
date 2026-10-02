"""
Configuration for BC-RNN with binary or continuous force-weighted CaMI.
The experiment JSON supplies task-specific cameras and dataset paths.
"""

from copy import deepcopy
from robomimic.config.bc_config import BCConfig


class BCCaMIConfig(BCConfig):
    ALGO_NAME = "bc_cami"

    def train_config(self):
        super().train_config()
        self.train.hdf5_load_next_obs = False
        # Default horizon is 10: one anchor plus ten future steps.
        self.train.seq_length = 11

    def observation_config(self):
        super().observation_config()
        self.observation.modalities.obs.rgb = [
            "agentview_image", "robot0_eye_in_hand_image",
        ]
        self.observation.modalities.obs.low_dim = [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object",
        ]
        self.observation.modalities.goal.rgb = []
        self.observation.modalities.goal.low_dim = []
        # Force and contact labels are auxiliary data, not policy inputs.

    def algo_config(self):
        super().algo_config()
        self.algo.rnn.enabled = True
        for name in ("state_encoder", "snippet_encoder", "key_proj"):
            self.algo.optim_params[name] = deepcopy(self.algo.optim_params.policy)

        self.algo.cami.enabled = True
        # Original fallback: lambda_state=loss_weight, lambda_traj=1.0.
        # Preserve explicit coefficients in any existing experiment JSON.
        self.algo.cami.loss_weight = 0.01
        self.algo.cami.temperature = 0.07
        self.algo.cami.loss_type = "paired_infonce"
        self.algo.cami.snippet_horizon = 10

        # Legacy fields retained for config compatibility. The in-batch
        # loss uses all eligible negatives, not a num_negatives subsample.
        self.algo.cami.num_negatives = 1
        self.algo.cami.opposite_contact_negatives_only = True
        self.algo.cami.contact_threshold = 10.0

        # Match the current branch's experiment JSON: targets use EMA.
        self.algo.cami.use_momentum_target = True
        self.algo.cami.target_tau = 0.005

        # Legacy fusion fields remain readable by existing configurations.
        # The current anchor is policy latent -> state_encoder.
        self.algo.cami.image_feature_dim = 128
        self.algo.cami.force_feature_dim = 128
        self.algo.cami.fused_feature_dim = 256
        self.algo.cami.anchor_fusion_layers = (256, 256)

        self.algo.cami.contrastive_dim = 128
        self.algo.cami.policy_latent_dim = 1024
        self.algo.cami.query_proj_layers = (128,)
        self.algo.cami.key_proj_layers = (128,)
        self.algo.cami.snippet_encoder_type = "lstm"
        self.algo.cami.snippet_hidden_dim = 256
        self.algo.cami.snippet_num_layers = 1
        self.algo.cami.image_obs_key = [
            "agentview_image", "robot0_eye_in_hand_image",
        ]
        self.algo.cami.force_obs_key = "force"
        self.algo.cami.contact_label_key = "contact_label"
        self.algo.cami.normalize_embeddings = True

        # Binary behavior remains the default. The continuous experiment
        # JSON explicitly enables the force-weighted loss.
        self.algo.cami.continuous_contact.enabled = False
        self.algo.cami.continuous_contact.force_dataset_key = "obs/force"

        # Null fits std(||F_xyz||) + 1e-6 from selected training demos only.
        # A positive number explicitly preserves a previously fitted scale.
        self.algo.cami.continuous_contact.force_scale = None

        # Huber transition after force normalization.
        self.algo.cami.continuous_contact.huber_delta = 0.1

        # Weight W_ij = (1 - exp(-D_ij / contact_temperature)) ** gamma.
        self.algo.cami.continuous_contact.contact_temperature = 0.05
        self.algo.cami.continuous_contact.gamma = 1.0
