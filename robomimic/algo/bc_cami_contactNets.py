from collections import OrderedDict
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from robomimic.algo import register_algo_factory_func
from robomimic.algo.bc import BC_RNN
from robomimic.algo.bc_cami import BC_CaMI, build_mlp
from robomimic.CAMI.continuous_contact import (
    normalize_contact, contact_distance, negative_weights, embedding_logits,
    weighted_infonce,
)
import robomimic.utils.tensor_utils as TensorUtils


@register_algo_factory_func("bc_cami_c")
def algo_config_to_class(algo_config):
    return BC_CaMI_C, {}


def _privileged_key(name):
    return name.startswith("contact_") or "force" in name.lower() or "torque" in name.lower() or "impulse" in name.lower()


class BC_CaMI_C(BC_CaMI):
    def _create_networks(self):
        cfg = self.algo_config
        c = cfg.cami.continuous
        if c.get("formulation_version", None) != 7:
            raise ValueError("Generate a version 7 CaMI-C config before training")
        if c.distance_penalty not in ("l1", "huber"):
            raise ValueError("distance_penalty must be l1 or huber")
        if c.distance_penalty == "huber" and (not math.isfinite(c.huber_delta) or c.huber_delta <= 0):
            raise ValueError("Huber delta must be finite and positive")
        if cfg.cami.get("subspace", {}).get("enabled", False):
            raise ValueError("This is CaMI-C only. Generate a new force or physics config.")
        if c.get("rank_weight", 0) or c.get("prediction_weight", 0):
            raise ValueError("This formulation has no auxiliary losses. Generate a fresh config.")
        for value in (cfg.cami.state_loss_weight, cfg.cami.traj_loss_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError("CaMI loss weights must be finite and nonnegative")
        if cfg.cami.snippet_horizon < 1:
            raise ValueError("snippet_horizon must be positive")
        if not math.isfinite(c.binary_threshold) or c.binary_threshold < 0:
            raise ValueError("binary_threshold must be finite and nonnegative")
        if not cfg.rnn.enabled or cfg.rnn.open_loop or cfg.rnn.kwargs.get("bidirectional", False):
            raise ValueError("CaMI-C requires a causal, closed-loop BC-RNN")
        if cfg.gmm.enabled or cfg.gaussian.enabled or cfg.vae.enabled or cfg.transformer.enabled:
            raise ValueError("CaMI-C currently supports deterministic BC-RNN only")
        if self.global_config.train.frame_stack != 1:
            raise ValueError("CaMI-C requires frame_stack=1 for anchor alignment")
        if self.global_config.train.seq_length < cfg.cami.snippet_horizon + 1:
            raise ValueError("seq_length must be at least snippet_horizon + 1")
        if any(_privileged_key(k) or k in c.privileged_obs_keys
               for k in list(self.obs_shapes) + list(self.goal_shapes)):
            raise ValueError("Contact, force, and torque must not be policy or goal observations")
        if c.objective not in ("weighted", "binary", "paired"):
            raise ValueError("Unknown continuous contact objective")
        if not 0 <= cfg.cami.momentum <= 1:
            raise ValueError("momentum must lie in [0,1]")

        # Infer the true post-RNN, post-MLP feature width instead of assuming 1024.
        BC_RNN._create_networks(self)
        latent_dim = (cfg.actor_layer_dims[-1] if len(cfg.actor_layer_dims) else cfg.rnn.hidden_dim)
        state_layers = cfg.cami.get("state_proj_layers", cfg.cami.query_proj_layers)
        self.nets["state_encoder"] = build_mlp(latent_dim, state_layers, cfg.cami.contrastive_dim)
        self.nets["snippet_encoder"] = nn.LSTM(
            self.ac_dim, cfg.cami.snippet_hidden_dim, cfg.cami.snippet_num_layers, batch_first=True)
        self.nets["key_proj"] = build_mlp(
            cfg.cami.snippet_hidden_dim, cfg.cami.key_proj_layers, cfg.cami.contrastive_dim)
        for name in ("snippet_encoder", "key_proj"):
            self.nets[name + "_target"] = copy.deepcopy(self.nets[name])
            self.nets[name + "_target"].requires_grad_(False)
        self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        H = self.algo_config.cami.snippet_horizon
        Fdim = len(self.algo_config.cami.continuous.feature_names)
        actions = batch["actions"]
        if actions.ndim != 3 or actions.shape[1] < H + 1:
            raise ValueError("actions must be [B,T,A], T >= snippet_horizon+1")
        if "pad_mask" not in batch:
            raise KeyError("Missing pad_mask: use the updated train_utils / SequenceDataset(get_pad_mask=True)")
        result = {"actions": actions, "pad_mask": batch["pad_mask"], "goal_obs": batch.get("goal_obs")}
        result["obs"] = {k: batch["obs"][k] for k in self.obs_shapes}
        for key in ("contact_state", "contact_mask"):
            if key not in batch:
                raise KeyError("Missing {}. Run prepare_cami_c_dataset and add both contact keys to dataset_keys".format(key))
            value = batch[key]
            if value.ndim != 4 or value.shape[:2] != actions.shape[:2] or value.shape[2:] != (H, Fdim):
                raise ValueError("{} must be [B,T,{},{}], got {}".format(key, H, Fdim, tuple(value.shape)))
            result[key] = value[:, 0]
        cm = result["contact_mask"]
        if not torch.isfinite(cm).all() or not torch.all((cm == 0) | (cm == 1)):
            raise ValueError("contact_mask must contain only zero or one")
        pad = result["pad_mask"]
        if pad.shape != (*actions.shape[:2], 1) or not torch.all((pad == 0) | (pad == 1)):
            raise ValueError("pad_mask must be binary [B,T,1]")
        if not pad[:, 0].all():
            raise ValueError("Every anchor must contain a real initial action")
        if (pad[:, 1:] > pad[:, :-1]).any():
            raise ValueError("Only right-padded sequences are supported")
        future_valid = pad[:, 1:H+1].bool()
        if (cm.bool() & ~future_valid).any():
            raise ValueError("Contact labels mark padded actions as valid; rebuild the dataset")
        if not torch.isfinite(result["contact_state"][cm.bool()]).all():
            raise ValueError("Nonfinite valid contact data")
        # Missing descriptors are explicitly masked, never sent into the encoder.
        result["contact_state"] = torch.where(cm.bool(), result["contact_state"], 0.)
        result["snippet_lengths"] = future_valid.sum(1).squeeze(-1)
        return TensorUtils.to_float(TensorUtils.to_device(result, self.device))

    def _forward_policy_with_latent(self, obs_dict, goal_dict=None):
        if goal_dict is not None:
            length = next(iter(obs_dict.values())).shape[1]
            goal_dict = TensorUtils.unsqueeze_expand_at(goal_dict, size=length, dim=1)
        raw, feats = self.nets["policy"].forward_with_features(obs=obs_dict, goal=goal_dict)
        # The inherited feature API returns raw decoder values. Match rollout.
        return torch.tanh(raw["action"]), feats

    def _snippet_embedding(self, actions, lengths, target=False):
        suffix = "_target" if target else ""
        packed = pack_padded_sequence(actions, lengths.clamp_min(1).long().cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.nets["snippet_encoder" + suffix](packed)
        return self.nets["key_proj" + suffix](hidden[-1])

    def _forward_training(self, batch):
        actions, features = self._forward_policy_with_latent(batch["obs"], batch["goal_obs"])
        state = self.nets["state_encoder"](features[:, 0])
        H = self.algo_config.cami.snippet_horizon
        snippet = batch["actions"][:, 1:H+1]
        query = self._snippet_embedding(snippet, batch["snippet_lengths"])
        with torch.no_grad():
            key = self._snippet_embedding(snippet, batch["snippet_lengths"], target=True)
        pred = OrderedDict(actions=actions, policy_latent=features[:, 0], state_query_embedding=state,
                           traj_query_embedding=query, target_key_embedding=key)
        return pred

    def _compute_losses(self, predictions, batch):
        cfg, c = self.algo_config, self.algo_config.cami.continuous
        actions, target = predictions["actions"], batch["actions"]
        valid = batch["pad_mask"].squeeze(-1).bool()
        a, y = actions[valid], target[valid]
        losses = OrderedDict(l2_loss=F.mse_loss(a, y), l1_loss=F.smooth_l1_loss(a, y),
                             cos_loss=(1 - F.cosine_similarity(a[..., :3], y[..., :3], dim=-1)).mean())
        losses["bc_action_loss"] = sum(cfg.loss[name + "_weight"] * losses[name + "_loss"]
                                        for name in ("l2", "l1", "cos"))
        zero = actions.sum() * 0.
        for name in ("state_cami_loss", "traj_cami_loss"):
            losses[name] = zero
        if cfg.cami.enabled:
            contact = batch["contact_state"]
            mask = batch["contact_mask"].bool()
            normalized = normalize_contact(contact, c.feature_names, c.gap_scale, c.force_scale, c.normal_impulse_scale, c.tangential_impulse_scale)
            # Privileged physics labels only set detached negative weights.
            # L1 is the default rho; neither labels nor impulses enter the policy.
            distance, pair_valid = contact_distance(normalized, mask, c.feature_weights,
                c.huber_delta, c.chunk_size, penalty=c.distance_penalty)
            # No valid future actions -> neither anchor nor negative candidate.
            future = batch["snippet_lengths"] > 0
            pair_valid &= future[:, None] & future[None]
            if c.objective == "weighted":
                weights = negative_weights(distance, pair_valid, c.contact_temperature, c.gamma)
            elif c.objective == "binary":
                idx = list(c.feature_names).index(c.binary_feature)
                labels = ((contact[..., idx] > c.binary_threshold) & mask[..., idx]).any(1)
                weights = ((labels[:, None] != labels[None]) & pair_valid).float()
            else:
                weights = pair_valid.float()
                weights.fill_diagonal_(0.)
            for branch in ("state", "traj"):
                logits = embedding_logits(predictions[branch + "_query_embedding"],
                    predictions["target_key_embedding"], cfg.cami.temperature, cfg.cami.normalize_embeddings)
                loss, info = weighted_infonce(logits, weights)
                losses[branch + "_cami_loss"] = loss
                for name, value in info.items():
                    losses[branch + "_" + name] = value
            offdiag = pair_valid & ~torch.eye(len(contact), dtype=torch.bool, device=contact.device)
            losses["contact_distance_mean"] = distance[offdiag].mean() if offdiag.any() else zero.detach()
        losses["action_loss"] = (losses["bc_action_loss"] +
            cfg.cami.state_loss_weight * losses["state_cami_loss"] +
            cfg.cami.traj_loss_weight * losses["traj_cami_loss"])
        return losses

    def _train_step(self, losses):
        names = ["policy", "state_encoder", "snippet_encoder", "key_proj"]
        for name in names:
            self.optimizers[name].zero_grad(set_to_none=True)
        losses["action_loss"].backward()
        info = OrderedDict()
        for name in names:
            max_norm = self.global_config.train.max_grad_norm
            norm = torch.nn.utils.clip_grad_norm_(self.nets[name].parameters(),
                float("inf") if max_norm is None else max_norm, error_if_nonfinite=True)
            info[name + "_grad_norm"] = float(norm)
        for name in names:
            self.optimizers[name].step()
        return info

    @torch.no_grad()
    def _update_target_networks(self):
        # With EMA off, hard-copy the online encoder after each training step.
        # Never silently leave a random target frozen forever.
        cfg = self.algo_config.cami
        m = cfg.momentum if cfg.use_momentum_target else 0.
        for name in ("snippet_encoder", "key_proj"):
            for online, target in zip(self.nets[name].parameters(), self.nets[name + "_target"].parameters()):
                target.mul_(m).add_(online, alpha=1. - m)

    def log_info(self, info):
        log = BC_RNN.log_info(self, info)
        for name, value in info["losses"].items():
            log["CaMI_C/" + name] = value.item()
        for name, value in info.items():
            if name.endswith("_grad_norm"):
                log["CaMI_C/" + name] = value
        return log