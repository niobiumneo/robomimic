"""BC-RNN with the full gap / normal / tangential impulse CaMI descriptor."""

from robomimic.config.bc_cami_config import BCCaMIConfig


class BCCaMIContactNetsConfig(BCCaMIConfig):
    ALGO_NAME = "bc_cami_contactNets"

    def experiment_config(self):
        super().experiment_config()
        self.experiment.validate = True
        self.experiment.rollout.rate = 50
        self.experiment.rollout.n = 50

    def train_config(self):
        super().train_config()
        self.train.frame_stack = 1
        self.train.seq_length = 11
        self.train.dataset_keys = ["actions", "contact_state", "contact_mask"]
        self.train.action_keys = ["actions"]
        self.train.action_config = {"actions": {"normalization": None}}
        self.train.hdf5_filter_key = "train"
        self.train.hdf5_validation_filter_key = "valid"
        self.train.num_epochs = 2000
        self.train.max_grad_norm = 1.

    def algo_config(self):
        super().algo_config()
        self.algo.rnn.enabled = True
        self.algo.rnn.open_loop = False
        self.algo.rnn.kwargs.bidirectional = False
        # Preserve the previous force-CaMI coefficient defaults explicitly.
        self.algo.cami.state_loss_weight = .01
        self.algo.cami.traj_loss_weight = 1.
        self.algo.cami.use_momentum_target = True
        # EMA: target <- momentum * target + (1 - momentum) * online.
        self.algo.cami.momentum = .995
        self.algo.cami.continuous_contact.enabled = False
        for name in ("policy", "state_encoder", "snippet_encoder", "key_proj"):
            lr = self.algo.optim_params[name].learning_rate
            lr.initial = 1e-4
            lr.decay_factor = .5
            lr.epoch_schedule = [1000]
            lr.scheduler_type = "multistep"
            lr.step_every_batch = False

        c = self.algo.cami.continuous
        c.formulation_version = 7
        c.feature_names = ["gap", "normal_impulse", "tangential_impulse_magnitude"]
        c.feature_weights = [1., 1., 1.]
        # Null requests training-only scale fitting before model construction.
        c.gap_scale = None
        c.force_scale = 50.  # Unused by the physics descriptor.
        c.normal_impulse_scale = None
        c.tangential_impulse_scale = None
        c.normalization_fit_demos = []
        c.normalization_method = "selected_training_std"
        # Populated from prepared HDF5 metadata and saved with the trained model.
        c.dataset_reconstruction = []
        c.binary_feature = "normal_impulse"
        c.binary_threshold = 0.
        c.contact_temperature = .05
        c.gamma = 1.
        c.distance_penalty = "huber"
        c.huber_delta = .1
        c.chunk_size = 64
        c.objective = "weighted"
        c.privileged_obs_keys = []
