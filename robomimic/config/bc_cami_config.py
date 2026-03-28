"""
Config for BC + CaMI algorithm.
"""

from robomimic.config.bc_config import BCConfig
from copy import deepcopy


class BCCaMIConfig(BCConfig):
    ALGO_NAME = "bc_cami"

    def train_config(self):
        super(BCCaMIConfig, self).train_config()
        self.train.hdf5_load_next_obs = False

    def observation_config(self):
        super(BCCaMIConfig, self).observation_config()

        # # keep your image key
        # if "agentview_image" not in self.observation.modalities.obs.rgb:
        #     self.observation.modalities.obs.rgb.append("agentview_image")

        # # add synthetic force key as low_dim obs
        # if "force" not in self.observation.modalities.obs.low_dim:
        #     self.observation.modalities.obs.low_dim.append("force")

        self.observation.modalities.obs.rgb = ["agentview_image", "robot0_eye_in_hand_image"]
        self.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object", "force"]

        self.observation.modalities.goal.rgb = []
        self.observation.modalities.goal.low_dim = []


    def algo_config(self):
        super(BCCaMIConfig, self).algo_config()

        # make optimizer entries for new CaMI trainable modules
        self.algo.optim_params.state_encoder = deepcopy(self.algo.optim_params.policy)
        self.algo.optim_params.snippet_encoder = deepcopy(self.algo.optim_params.policy)
        self.algo.optim_params.key_proj = deepcopy(self.algo.optim_params.policy)

        self.algo.cami.enabled = True
        self.algo.cami.loss_weight = 0.01
        self.algo.cami.temperature = 0.07
        self.algo.cami.loss_type = "paired_infonce"

        self.algo.cami.snippet_horizon = 10
        self.algo.cami.num_negatives = 1
        self.algo.cami.opposite_contact_negatives_only = True
        self.algo.cami.contact_threshold = 10.0

        self.algo.cami.use_momentum_target = False
        self.algo.cami.target_tau = 0.005

        self.algo.cami.image_feature_dim = 128
        self.algo.cami.force_feature_dim = 128
        self.algo.cami.fused_feature_dim = 256
        self.algo.cami.contrastive_dim = 128
        self.algo.cami.policy_latent_dim = 1024

        self.algo.cami.anchor_fusion_layers = (256, 256)
        self.algo.cami.query_proj_layers = (128,)
        self.algo.cami.key_proj_layers = (128,)

        self.algo.cami.snippet_encoder_type = "lstm"
        self.algo.cami.snippet_hidden_dim = 256
        self.algo.cami.snippet_num_layers = 1

        self.algo.cami.image_obs_key = ["agentview_image", "robot0_eye_in_hand_image"]
        self.algo.cami.force_obs_key = "force"
        self.algo.cami.contact_label_key = "contact_label"

        self.algo.cami.normalize_embeddings = True


# class BCCaMIConfig(BaseConfig):
#     ALGO_NAME = "bc_cami"

#     def train_config(self):
#         """
#         CaMI needs future observations / snippets, so keep next_obs enabled.
#         """
#         super(BCCaMIConfig, self).train_config()
#         self.train.hdf5_load_next_obs = True

#     def algo_config(self):
#         """
#         Populate config.algo for BC + CaMI.
#         """

#         # ------------------------------------------------------------------ #
#         # Standard BC optimization
#         # ------------------------------------------------------------------ #
#         self.algo.optim_params.policy.optimizer_type = "adam"
#         self.algo.optim_params.policy.learning_rate.initial = 1e-4
#         self.algo.optim_params.policy.learning_rate.decay_factor = 0.1
#         self.algo.optim_params.policy.learning_rate.epoch_schedule = []
#         self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep"
#         self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
#         self.algo.optim_params.policy.regularization.L2 = 0.00

#         # ------------------------------------------------------------------ #
#         # Standard BC loss
#         # ------------------------------------------------------------------ #
#         self.algo.loss.l2_weight = 1.0
#         self.algo.loss.l1_weight = 0.0
#         self.algo.loss.cos_weight = 0.0

#         # ------------------------------------------------------------------ #
#         # Standard BC actor network
#         # ------------------------------------------------------------------ #
#         self.algo.actor_layer_dims = (1024, 1024)

#         # ------------------------------------------------------------------ #
#         # Keep policy simple in version 1
#         # ------------------------------------------------------------------ #
#         self.algo.gaussian.enabled = False
#         self.algo.gaussian.fixed_std = False
#         self.algo.gaussian.init_std = 0.1
#         self.algo.gaussian.min_std = 0.01
#         self.algo.gaussian.std_activation = "softplus"
#         self.algo.gaussian.low_noise_eval = True

#         self.algo.gmm.enabled = False
#         self.algo.gmm.num_modes = 5
#         self.algo.gmm.min_std = 0.0001
#         self.algo.gmm.std_activation = "softplus"
#         self.algo.gmm.low_noise_eval = True

#         self.algo.vae.enabled = False
#         self.algo.vae.latent_dim = 14
#         self.algo.vae.latent_clip = None
#         self.algo.vae.kl_weight = 1.0

#         self.algo.vae.decoder.is_conditioned = True
#         self.algo.vae.decoder.reconstruction_sum_across_elements = False

#         self.algo.vae.prior.learn = False
#         self.algo.vae.prior.is_conditioned = False
#         self.algo.vae.prior.use_gmm = False
#         self.algo.vae.prior.gmm_num_modes = 10
#         self.algo.vae.prior.gmm_learn_weights = False
#         self.algo.vae.prior.use_categorical = False
#         self.algo.vae.prior.categorical_dim = 10
#         self.algo.vae.prior.categorical_gumbel_softmax_hard = False
#         self.algo.vae.prior.categorical_init_temp = 1.0
#         self.algo.vae.prior.categorical_temp_anneal_step = 0.001
#         self.algo.vae.prior.categorical_min_temp = 0.3

#         self.algo.vae.encoder_layer_dims = (300, 400)
#         self.algo.vae.decoder_layer_dims = (300, 400)
#         self.algo.vae.prior_layer_dims = (300, 400)

#         self.algo.rnn.enabled = False
#         self.algo.rnn.horizon = 10
#         self.algo.rnn.hidden_dim = 400
#         self.algo.rnn.rnn_type = "LSTM"
#         self.algo.rnn.num_layers = 2
#         self.algo.rnn.open_loop = False
#         self.algo.rnn.kwargs.bidirectional = False
#         self.algo.rnn.kwargs.do_not_lock_keys()

#         self.algo.transformer.enabled = False
#         self.algo.transformer.context_length = 10
#         self.algo.transformer.embed_dim = 512
#         self.algo.transformer.num_layers = 6
#         self.algo.transformer.num_heads = 8
#         self.algo.transformer.emb_dropout = 0.1
#         self.algo.transformer.attn_dropout = 0.1
#         self.algo.transformer.block_output_dropout = 0.1
#         self.algo.transformer.sinusoidal_embedding = False
#         self.algo.transformer.activation = "gelu"
#         self.algo.transformer.supervise_all_steps = False
#         self.algo.transformer.nn_parameter_for_timesteps = True
#         self.algo.transformer.pred_future_acs = False

#         # ------------------------------------------------------------------ #
#         # CaMI-specific settings
#         # ------------------------------------------------------------------ #
#         self.algo.cami.enabled = True

#         # overall CaMI contribution
#         self.algo.cami.loss_weight = 0.1
#         self.algo.cami.temperature = 0.07
#         self.algo.cami.loss_type = "paired_infonce"

#         # anchor / future snippet setup
#         self.algo.cami.snippet_horizon = 10
#         self.algo.cami.num_negatives = 1
#         self.algo.cami.opposite_contact_negatives_only = True

#         # momentum target encoder
#         self.algo.cami.use_momentum_target = True
#         self.algo.cami.target_tau = 0.005

#         # embedding sizes
#         self.algo.cami.image_feature_dim = 128
#         self.algo.cami.force_feature_dim = 128
#         self.algo.cami.fused_feature_dim = 256
#         self.algo.cami.contrastive_dim = 128

#         # fusion and projection MLPs
#         self.algo.cami.anchor_fusion_layers = (256, 256)
#         self.algo.cami.query_proj_layers = (128,)
#         self.algo.cami.key_proj_layers = (128,)

#         # future snippet temporal encoder
#         self.algo.cami.snippet_encoder_type = "gru"
#         self.algo.cami.snippet_hidden_dim = 256
#         self.algo.cami.snippet_num_layers = 1

#         # modality / label keys in batch
#         self.algo.cami.image_obs_key = "agentview_image"
#         self.algo.cami.force_obs_key = "force"
#         self.algo.cami.contact_label_key = "contact_label"

#         # final embedding normalization before contrastive loss
#         self.algo.cami.normalize_embeddings = True