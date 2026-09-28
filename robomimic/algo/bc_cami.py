"""
BC-RNN with binary or continuous force-weighted CaMI.

The policy latent is projected into a state query. An online LSTM encodes
future expert actions into a trajectory query, and an EMA target LSTM
produces the keys shared by both contrastive objectives.

Continuous mode uses recorded future force magnitudes to weight negatives.
Force is training supervision and is not a policy observation.

The weighted objective is an adaptation of CaMI. An unchanged mutual-
information lower-bound guarantee is not assumed.
"""

from collections import OrderedDict
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import register_algo_factory_func
from robomimic.algo.bc import BC_RNN


TRAINABLE_NETWORKS = ("policy", "state_encoder", "snippet_encoder", "key_proj")
DIAGNOSTICS = (
    "valid_anchor_count", "valid_anchor_fraction", "pos_logit_mean",
    "neg_logit_mean", "retrieval_acc", "avg_valid_negatives",
    "soft_scale_mean", "negative_weight_mean", "effective_negatives",
)


@register_algo_factory_func("bc_cami")
def algo_config_to_class(algo_config):
    return BC_CaMI, {}


def build_mlp(input_dim, hidden_dims, output_dim):
    """Build the state or trajectory projection network."""
    layers = []
    for width in hidden_dims:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class BC_CaMI(BC_RNN):
    """Preserve the BC-RNN policy and add two contact-aware objectives."""

    def _cami_enabled(self):
        return self.algo_config.cami.get("enabled", False)

    def _continuous_enabled(self):
        return self.algo_config.cami.get(
            "continuous_contact", {}
        ).get("enabled", False)

    def _create_networks(self):
        cami = self.algo_config.cami
        if not math.isfinite(float(cami.temperature)) or cami.temperature <= 0:
            raise ValueError("cami.temperature must be finite and positive")
        if int(cami.snippet_horizon) != cami.snippet_horizon or cami.snippet_horizon < 1:
            raise ValueError("snippet_horizon must be a positive integer")

        # Keep force and labels outside the policy in both contact modes.
        excluded = {"force", "force_rawbias", "force_obsbias", "contact_label"}
        if self._continuous_enabled():
            excluded.add(cami.continuous_contact.force_dataset_key.split("/", 1)[-1])
        if excluded.intersection(self.obs_shapes):
            raise ValueError("Remove privileged force/contact labels from policy modalities")

        if self._cami_enabled() and self._continuous_enabled():
            cc = cami.continuous_contact
            for name in ("force_scale", "huber_delta", "contact_temperature", "gamma"):
                value = float(cc[name])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("continuous_contact." + name + " must be finite and positive")
            if self.global_config.train.frame_stack != 1:
                raise ValueError("Continuous CaMI requires frame_stack=1")
            if self.global_config.train.seq_length < cami.snippet_horizon + 1:
                raise ValueError("seq_length must be at least snippet_horizon + 1")
            if cc.force_dataset_key not in self.global_config.train.dataset_keys:
                raise ValueError("Add " + cc.force_dataset_key + " to train.dataset_keys")
            if not cami.get("use_momentum_target", True):
                raise ValueError("Continuous CaMI requires use_momentum_target=true")

        # This must run outside the continuous-mode conditional, including
        # when training the binary baseline.
        super()._create_networks()
        if self._rnn_is_open_loop:
            raise ValueError("This CaMI implementation requires rnn.open_loop=false")

        state_hidden = cami.get("state_proj_layers", cami.get("query_proj_layers", []))
        self.nets["state_encoder"] = build_mlp(
            cami.policy_latent_dim, list(state_hidden), cami.contrastive_dim
        )
        self.nets["snippet_encoder"] = nn.LSTM(
            input_size=self.ac_dim,
            hidden_size=cami.snippet_hidden_dim,
            num_layers=cami.snippet_num_layers,
            batch_first=True,
        )
        self.nets["key_proj"] = build_mlp(
            cami.snippet_hidden_dim, list(cami.key_proj_layers), cami.contrastive_dim
        )

        # Retain the original module names for checkpoint compatibility.
        for online, target in (
            ("snippet_encoder", "snippet_encoder_target"),
            ("key_proj", "key_proj_target"),
        ):
            self.nets[target] = copy.deepcopy(self.nets[online])
            for parameter in self.nets[target].parameters():
                parameter.requires_grad = False
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """Keep full BC sequences and extract aligned contact supervision."""
        result = {
            "obs": {key: batch["obs"][key] for key in self.obs_shapes},
            "goal_obs": batch.get("goal_obs"),
            "actions": batch["actions"],
        }

        # Disabling CaMI should not require contact labels or force.
        if self._cami_enabled():
            if self._continuous_enabled():
                key = self.algo_config.cami.continuous_contact.force_dataset_key
                if key not in batch:
                    raise KeyError("Missing auxiliary force dataset key: " + key)
                wrench = batch[key]
                if wrench.ndim != 3 or wrench.shape[-1] not in (3, 6):
                    raise ValueError("Stored wrench must be [B,T,3] or [B,T,6]")
                if wrench.shape[:2] != batch["actions"].shape[:2]:
                    raise ValueError("Wrench and action sequence lengths must match")
                if "pad_mask" not in batch or batch["pad_mask"].shape != wrench.shape[:2] + (1,):
                    raise ValueError("Continuous CaMI requires get_pad_mask=True")

                horizon = self.algo_config.cami.snippet_horizon
                if wrench.shape[1] < horizon + 1:
                    raise ValueError("Batch needs one anchor plus H future timesteps")

                # Exactly the same future offsets as the action snippet.
                # Torque is excluded because it has different units.
                result["force_sequence"] = wrench[:, 1:horizon + 1, :3].detach()
                result["force_valid"] = batch["pad_mask"][:, 1:horizon + 1, 0].bool()
            else:
                label = batch.get("contact_label", batch["obs"].get("contact_label"))
                if label is None:
                    raise KeyError("Binary CaMI requires contact_label")
                if label.ndim > 1:
                    label = label[:, 0]
                result["contact_label"] = label.reshape(-1).float()

        return TensorUtils.to_float(TensorUtils.to_device(result, self.device))

    def train_on_batch(self, batch, epoch, validate=False):
        """Update online networks first, then their momentum targets."""
        info = super().train_on_batch(batch=batch, epoch=epoch, validate=validate)
        if not validate:
            self._update_target_networks()
        return info

    def _select_future_snippet(self, batch):
        """Return expert actions at offsets 1..H, preserving original padding."""
        horizon = self.algo_config.cami.snippet_horizon
        actions = batch["actions"]
        snippet = actions[:, 1:min(horizon + 1, actions.shape[1])]
        if snippet.shape[1] == 0:
            snippet = actions[:, :1].repeat(1, horizon, 1)
        elif snippet.shape[1] < horizon:
            padding = snippet[:, -1:].repeat(1, horizon - snippet.shape[1], 1)
            snippet = torch.cat([snippet, padding], dim=1)
        return {"actions": snippet}

    def _make_snippet_tensor(self, snippet_dict):
        return snippet_dict["actions"]

    def _encode_anchor_state(self, anchor_latent):
        """State query z_i = psi(h_i), where h_i is the policy latent."""
        embedding = self.nets["state_encoder"](anchor_latent)
        if self.algo_config.cami.get("normalize_embeddings", False):
            embedding = F.normalize(embedding, dim=-1)
        return anchor_latent, embedding

    def _encode_snippet(self, obs_seq, use_target=False):
        """Encode future actions with the online or frozen target branch."""
        actions = self._make_snippet_tensor(obs_seq)
        if use_target:
            with torch.no_grad():
                _, (hidden, _) = self.nets["snippet_encoder_target"](actions)
                feature = hidden[-1]
                embedding = self.nets["key_proj_target"](feature)
        else:
            _, (hidden, _) = self.nets["snippet_encoder"](actions)
            feature = hidden[-1]
            embedding = self.nets["key_proj"](feature)
        if self.algo_config.cami.get("normalize_embeddings", False):
            embedding = F.normalize(embedding, dim=-1)
        return feature, embedding

    def _forward_training(self, batch):
        predictions = OrderedDict()
        actions, features = self._forward_policy_with_latent(
            obs_dict=batch["obs"], goal_dict=batch["goal_obs"]
        )
        predictions["actions"] = actions
        if not self._cami_enabled():
            return predictions

        # The anchor is the first state in each sampled sequence.
        anchor = features[:, 0, :]
        if anchor.shape[-1] != self.algo_config.cami.policy_latent_dim:
            raise ValueError("policy_latent_dim does not match policy feature size")
        snippet = self._select_future_snippet(batch)
        _, predictions["state_query_embedding"] = self._encode_anchor_state(anchor)
        (
            predictions["online_snippet_feat"],
            predictions["traj_query_embedding"],
        ) = self._encode_snippet(snippet, use_target=False)
        (
            predictions["target_snippet_feat"],
            predictions["target_key_embedding"],
        ) = self._encode_snippet(snippet, use_target=True)
        return predictions

    def _forward_policy_with_latent(self, obs_dict, goal_dict=None):
        """Use the existing branch's action/feature forward path unchanged."""
        policy = self.nets["policy"]
        if not hasattr(policy, "forward_with_features"):
            raise AttributeError("Policy needs forward_with_features")
        output = policy.forward_with_features(obs=obs_dict, goal=goal_dict)
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Expected (actions, features) from the policy")
        actions, features = output
        if isinstance(actions, dict):
            actions = actions.get("action", actions.get("actions"))
        if not torch.is_tensor(actions) or not torch.is_tensor(features):
            raise RuntimeError("Policy actions and features must be tensors")
        if features.ndim != 3:
            raise RuntimeError("Policy features must have shape [B,T,D]")
        return actions, features

    def _compute_contact_inbatch_cami_loss(
        self, query_embedding, key_embedding, contact_label,
        force_sequence=None, force_valid=None,
    ):
        """
        For either query branch:
            ell_ij = dot(query_i, key_j) / temperature
            L_i = log(exp(ell_ii) + sum_j W_ij exp(ell_ij)) - ell_ii

        Binary: W_ij = 1[C_i != C_j].
        Continuous: W_ij comes from future force-profile differences.
        The diagonal is always the paired positive, never a negative.
        """
        cami = self.algo_config.cami
        if cami.get("normalize_embeddings", False):
            query_embedding = F.normalize(query_embedding, dim=-1)
            key_embedding = F.normalize(key_embedding, dim=-1)
        logits = (query_embedding @ key_embedding.T) / cami.temperature
        positive = logits.diag()
        batch_size = logits.shape[0]
        continuous = self._continuous_enabled()
        weights = None

        if continuous:
            cc = cami.continuous_contact
            if (
                force_sequence is None or force_valid is None
                or force_sequence.ndim != 3
                or force_sequence.shape[0] != batch_size
                or force_sequence.shape[-1] != 3
                or force_valid.shape != force_sequence.shape[:2]
            ):
                raise ValueError("Expected future force [B,H,3] and validity [B,H]")

            # Recorded force is fixed supervision: no gradients into weights.
            with torch.no_grad():
                valid = force_valid.detach().bool()
                force = force_sequence.detach().float()
                if not torch.isfinite(force[valid]).all():
                    raise ValueError("Recorded force contains nonfinite measured values")
                force = force.masked_fill(~valid[..., None], 0.0)
                magnitude = torch.linalg.vector_norm(force, dim=-1) / cc.force_scale

                # Each [i,j,h] entry compares the same future offset h.
                left, right = torch.broadcast_tensors(
                    magnitude[:, None, :], magnitude[None, :, :]
                )
                rho = F.huber_loss(left, right, reduction="none", delta=cc.huber_delta)
                common = valid[:, None, :] & valid[None, :, :]
                count = common.sum(-1)
                distance = rho.masked_fill(~common, 0.0).sum(-1) / count.clamp_min(1)
                if not torch.isfinite(distance).all():
                    raise ValueError("Nonfinite contact distances; check force units and scale")

                # W_ij = (1 - exp(-D_ij / tau_c))^gamma.
                # Similar profiles are weak negatives; different profiles
                # are stronger negatives. expm1 is stable for small D_ij.
                weights = (
                    -torch.expm1(-distance / cc.contact_temperature)
                ).clamp(0, 1).pow(cc.gamma)
                weights = weights.masked_fill(count == 0, 0.0)
                weights.fill_diagonal_(0.0)
                negative_mask = weights > 0
        else:
            if contact_label is None:
                raise ValueError("Binary CaMI requires contact_label")
            label = contact_label.long().view(-1)
            if label.numel() != batch_size:
                raise ValueError("Expected one contact label per anchor")
            negative_mask = label[:, None] != label[None, :]

        negative_count = negative_mask.sum(1)
        valid_anchor = negative_count > 0
        if not valid_anchor.any():
            # Keep a graph connection so backward remains well-defined.
            zero = logits.sum() * 0.0
            return zero, {name: zero.detach() for name in DIAGNOSTICS}

        negative_logits = logits.masked_fill(~negative_mask, float("-inf"))
        if continuous:
            # Multiplying exp(ell_ij) by W_ij becomes adding log(W_ij).
            log_weights = weights.masked_fill(~negative_mask, 1.0).log()
            negative_logits = negative_logits + log_weights
        denominator = torch.cat([positive[:, None], negative_logits], dim=1)
        per_anchor = torch.logsumexp(denominator, dim=1) - positive

        # Preserve the upstream optional embedding-dependent binary variant.
        # Continuous mode already uses physical pair weights.
        soft_scale = logits.new_zeros(())
        if cami.get("soft_variant", False) and not continuous:
            scales = logits.new_ones(batch_size)
            for index in range(batch_size):
                if valid_anchor[index]:
                    distance = torch.norm(
                        key_embedding[index:index + 1]
                        - key_embedding[negative_mask[index]], dim=1
                    ).mean()
                    scales[index] = distance.clamp_min(1.0)
            per_anchor = per_anchor * scales
            soft_scale = scales[valid_anchor].mean()

        # Preserve the original binary reduction. Continuous anchors with
        # no measured negatives contribute zero to the full-batch average.
        loss = per_anchor.mean() if continuous else per_anchor[valid_anchor].mean()

        with torch.no_grad():
            diagnostic_weights = weights if continuous else negative_mask.float()
            info = {
                "valid_anchor_count": valid_anchor.float().sum(),
                "valid_anchor_fraction": valid_anchor.float().mean(),
                "pos_logit_mean": positive[valid_anchor].mean(),
                "neg_logit_mean": logits[negative_mask].mean(),
                # Continuous retrieval uses WEIGHTED negative scores.
                # It is not directly comparable to binary retrieval accuracy.
                "retrieval_acc": (
                    positive[valid_anchor]
                    > negative_logits.max(1).values[valid_anchor]
                ).float().mean(),
                "avg_valid_negatives": negative_count[valid_anchor].float().mean(),
                "soft_scale_mean": soft_scale.detach(),
                "negative_weight_mean": (
                    diagnostic_weights.sum() / max(batch_size * (batch_size - 1), 1)
                ),
                # This is sum-of-weights per anchor, not a statistical ESS.
                "effective_negatives": diagnostic_weights.sum(1).mean(),
            }
        return loss, info

    def _compute_losses(self, predictions, batch):
        """L_total = L_BC + lambda_state*L_state + lambda_traj*L_traj."""
        actions, target = predictions["actions"], batch["actions"]
        losses = OrderedDict()
        losses["l2_loss"] = F.mse_loss(actions, target)
        losses["l1_loss"] = F.smooth_l1_loss(actions, target)
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], target[..., :3])
        loss_cfg = self.algo_config.loss
        losses["bc_action_loss"] = (
            loss_cfg.l2_weight * losses["l2_loss"]
            + loss_cfg.l1_weight * losses["l1_loss"]
            + loss_cfg.cos_weight * losses["cos_loss"]
        )
        losses["state_cami_loss"] = actions.new_zeros(())
        losses["traj_cami_loss"] = actions.new_zeros(())
        losses["action_loss"] = losses["bc_action_loss"]

        if self._cami_enabled():
            cami = self.algo_config.cami
            label = None if self._continuous_enabled() else batch["contact_label"].view(-1)
            for prefix in ("state", "traj"):
                # Both branches receive the same aligned force supervision.
                value, diagnostics = self._compute_contact_inbatch_cami_loss(
                    query_embedding=predictions[prefix + "_query_embedding"],
                    key_embedding=predictions["target_key_embedding"],
                    contact_label=label,
                    force_sequence=batch.get("force_sequence"),
                    force_valid=batch.get("force_valid"),
                )
                losses[prefix + "_cami_loss"] = value
                for name, diagnostic in diagnostics.items():
                    losses[prefix + "_" + name] = diagnostic

            state_weight = (
                cami.state_loss_weight if "state_loss_weight" in cami else cami.loss_weight
            )
            traj_weight = cami.get("traj_loss_weight", 1.0)
            losses["action_loss"] = (
                losses["bc_action_loss"]
                + state_weight * losses["state_cami_loss"]
                + traj_weight * losses["traj_cami_loss"]
            )
        return losses

    def _train_step(self, losses):
        """Optimize online modules only; target modules never get gradients."""
        names = TRAINABLE_NETWORKS if self._cami_enabled() else ("policy",)
        missing = [name for name in names if name not in self.optimizers]
        if missing:
            raise KeyError("Missing algo.optim_params entries: " + str(missing))
        for name in names:
            self.optimizers[name].zero_grad()
        losses["action_loss"].backward()

        info = OrderedDict()
        limit = self.global_config.train.max_grad_norm
        for name in names:
            norm = torch.nn.utils.clip_grad_norm_(
                self.nets[name].parameters(), limit if limit is not None else 1e9
            )
            info[name + "_grad_norm"] = float(norm)
            self.optimizers[name].step()
        return info

    @torch.no_grad()
    def _update_target_networks(self):
        """EMA target = m*target + (1-m)*online, after each training step."""
        cami = self.algo_config.cami
        if not self._cami_enabled() or not cami.get("use_momentum_target", True):
            return
        momentum = cami.get("momentum")
        if momentum is None:
            momentum = 1.0 - cami.get("target_tau", 0.05)
        if not math.isfinite(float(momentum)) or not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must be finite and within [0,1]")
        for online, target in (
            ("snippet_encoder", "snippet_encoder_target"),
            ("key_proj", "key_proj_target"),
        ):
            for source, destination in zip(
                self.nets[online].parameters(), self.nets[target].parameters()
            ):
                destination.mul_(momentum).add_(source, alpha=1.0 - momentum)

    def log_info(self, info):
        log = super().log_info(info)
        losses = info["losses"]
        for label, key in (
            ("Loss", "action_loss"), ("BC_Action_Loss", "bc_action_loss"),
            ("State_CaMI_Loss", "state_cami_loss"), ("Traj_CaMI_Loss", "traj_cami_loss"),
            ("L2_Loss", "l2_loss"), ("L1_Loss", "l1_loss"), ("Cosine_Loss", "cos_loss"),
        ):
            log[label] = losses[key].item()
        for prefix in ("state", "traj"):
            for name in DIAGNOSTICS:
                key = prefix + "_" + name
                if key in losses:
                    log[key] = losses[key].item()
        for name in TRAINABLE_NETWORKS:
            key = name + "_grad_norm"
            if key in info:
                log[key] = info[key]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """Roll out with declared policy observations; no future force needed."""
        assert not self.nets.training
        missing = [key for key in self.obs_shapes if key not in obs_dict]
        if missing:
            raise KeyError("Missing rollout observations: " + str(missing))
        filtered = {key: obs_dict[key] for key in self.obs_shapes}
        return super().get_action(filtered, goal_dict=goal_dict)
