"""Separate v3 settings; the ContactNets v2 config remains unchanged."""

from copy import deepcopy

from robomimic.config.bc_cami_configContactNetsv2 import BCCaMIContactNetsConfig


class BCCaMIV3Config(BCCaMIContactNetsConfig):
    ALGO_NAME = "bc_cami_v3"

    def train_config(self):
        super().train_config()
        self.train.seq_length = 16  # five preceding steps, anchor, ten future steps
        self.train.batch_size = 32
        self.train.seed = 0

    def algo_config(self):
        super().algo_config()
        # v3 trains at a constant learning rate; the v2 default halves it at epoch 1000.
        policy_rate = self.algo.optim_params.policy.learning_rate
        policy_rate.decay_factor = .1
        policy_rate.epoch_schedule = []
        # One optimizer owns the reusable module, including an optional head.
        self.algo.optim_params.cami = deepcopy(self.algo.optim_params.policy)
        for name in ("state_encoder", "snippet_encoder", "key_proj"):
            del self.algo.optim_params[name]
        self.algo.cami.v3.context_steps = 5
        self.algo.cami.v3.warmup_epochs = 0
        self.algo.cami.v3.gradient_diagnostics_every = 100
        # "feature": an unobserved channel drops only itself from a pair distance, not the timestep.
        self.algo.cami.v3.missing_features = "feature"
        # Optional descriptor compression before scaling; scales are fitted after it.
        self.algo.cami.v3.descriptor.gap_min = None            # clip penetration below this gap (m)
        self.algo.cami.v3.descriptor.impulse_log_scale = None  # impulses I -> s * log1p(I / s) (N*s)
        self.algo.cami.v3.prediction.enabled = False
        self.algo.cami.v3.prediction.features = ["gap"]
        self.algo.cami.v3.prediction.hidden_dims = [128]
        self.algo.cami.v3.prediction.loss_weight = .01
        self.algo.cami.v3.prediction.huber_delta = .1
