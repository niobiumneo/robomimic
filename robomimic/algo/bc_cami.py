"""
BC_RNN + CaMI for robomimic

This version is aligned to robomimic's BC_RNN training flow and also follows
the SaMI repo pattern of:
    - online encoder for query
    - target encoder for keys
    - EMA / Polyak update for the target encoder

Implemented losses:
    1) PDF-style CAMI:
        z_i     = psi(s_i)
        e_i^+   = phi*(tau_i)
        e_j^-   = phi*(tau_j), j in N_i
        N_i     = {j | k_j != k_i}

    2) SaMI-style trajectory momentum CAMI:
        q_i     = phi(tau_i)
        e_i^+   = phi*(tau_i)
        e_j^-   = phi*(tau_j), j in N_i

The second term is included so the online snippet encoder actually receives
gradient, similar to how SaMI trains its online encoder against a momentum
target encoder.
"""

from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func
from robomimic.algo.bc import BC_RNN


@register_algo_factory_func("bc_cami")
def algo_config_to_class(algo_config):
    return BC_CaMI, {}


def build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class BC_CaMI(BC_RNN):
    """
    BC_RNN with Contact-aware Mutual Information regularization.

    Policy:
        standard BC_RNN policy over full observation sequences

    CAMI:
        - state encoder psi(s_i) for the PDF anchor query
        - online snippet encoder phi(tau_i)
        - momentum target snippet encoder phi*(tau_i)
        - contact-aware negatives only
    """

    def _create_networks(self):
        super(BC_CaMI, self)._create_networks()

        contrastive_dim = self.algo_config.cami.contrastive_dim
        state_hidden = list(
            getattr(
                self.algo_config.cami,
                "state_proj_layers",
                getattr(self.algo_config.cami, "query_proj_layers", []),
            )
        )
        key_hidden = list(self.algo_config.cami.key_proj_layers)

        # state encoder input size from current obs s_i
        state_input_dim = 0
        for k, shape in self.obs_shapes.items():
            if k == "contact_label":
                continue
            d = 1
            for s in shape:
                d *= s
            state_input_dim += d

        self.nets["state_encoder"] = build_mlp(
            input_dim=state_input_dim,
            hidden_dims=state_hidden,
            output_dim=contrastive_dim,
        )

        # snippet input size from tau_i
        snippet_input_dim = (
            self.obs_shapes["force"][0]
            + self.obs_shapes["robot0_eef_pos"][0]
            + self.obs_shapes["robot0_eef_quat"][0]
            + self.obs_shapes["robot0_gripper_qpos"][0]
        )

        self.nets["snippet_encoder"] = nn.LSTM(
            input_size=snippet_input_dim,
            hidden_size=self.algo_config.cami.snippet_hidden_dim,
            num_layers=self.algo_config.cami.snippet_num_layers,
            batch_first=True,
        )

        self.nets["key_proj"] = build_mlp(
            input_dim=self.algo_config.cami.snippet_hidden_dim,
            hidden_dims=key_hidden,
            output_dim=contrastive_dim,
        )

        # momentum target branch
        self.nets["snippet_encoder_target"] = copy.deepcopy(self.nets["snippet_encoder"])
        self.nets["key_proj_target"] = copy.deepcopy(self.nets["key_proj"])

        for p in self.nets["snippet_encoder_target"].parameters():
            p.requires_grad = False
        for p in self.nets["key_proj_target"].parameters():
            p.requires_grad = False

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Keep BC_RNN-compatible full sequences.
        Also pull out a contact label for each sequence.
        """
        input_batch = dict()
        input_batch["obs"] = {
            k: batch["obs"][k]
            for k in batch["obs"]
            if k != "contact_label"
        }
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"]

        # Contact label k_i
        if "contact_label" in batch:
            cl = batch["contact_label"]
        elif "contact_label" in batch["obs"]:
            cl = batch["obs"]["contact_label"]
        else:
            cl = None

        if cl is not None:
            if cl.ndim > 1:
                cl = cl[:, 0]
            if cl.ndim > 1:
                cl = cl.squeeze(-1)
            input_batch["contact_label"] = cl
        else:
            # derive from current force magnitude
            force_t0 = batch["obs"]["force"][:, 0, :3]
            force_mag = torch.norm(force_t0, dim=-1)
            threshold = getattr(self.algo_config.cami, "contact_threshold", 10.0)
            input_batch["contact_label"] = (force_mag > threshold).float()

        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Run BC_RNN train step, then update target encoders.
        """
        info = super(BC_CaMI, self).train_on_batch(
            batch=batch,
            epoch=epoch,
            validate=validate,
        )
        if not validate:
            self._update_target_networks()
        return info

    def _select_anchor_obs(self, obs_seq):
        """
        Current state s_i from timestep 0.
        """
        return {k: obs_seq[k][:, 0] for k in obs_seq}

    def _select_future_snippet(self, obs_seq):
        """
        Future snippet tau_i from timesteps 1..H, padded if needed.
        """
        H = self.algo_config.cami.snippet_horizon
        snippet_keys = ["force", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

        out = {}
        for k in snippet_keys:
            seq = obs_seq[k]
            T = seq.shape[1]
            end = min(H + 1, T)
            snippet = seq[:, 1:end]

            if snippet.shape[1] < H:
                if snippet.shape[1] == 0:
                    snippet = seq[:, 0:1].repeat(1, H, *([1] * (seq.ndim - 2)))
                else:
                    pad_count = H - snippet.shape[1]
                    repeat_shape = [1, pad_count] + [1] * (snippet.ndim - 2)
                    pad = snippet[:, -1:].repeat(*repeat_shape)
                    snippet = torch.cat([snippet, pad], dim=1)

            out[k] = snippet
        return out

    def _make_state_tensor(self, anchor_obs):
        pieces = []
        for k in anchor_obs:
            v = anchor_obs[k]
            pieces.append(v.reshape(v.shape[0], -1))
        return torch.cat(pieces, dim=-1)

    def _make_snippet_tensor(self, obs_seq):
        return torch.cat(
            [
                obs_seq["force"],
                obs_seq["robot0_eef_pos"],
                obs_seq["robot0_eef_quat"],
                obs_seq["robot0_gripper_qpos"],
            ],
            dim=-1,
        )

    def _encode_anchor_state(self, anchor_obs):
        """
        z_i = psi(s_i)
        """
        s = self._make_state_tensor(anchor_obs)
        z = self.nets["state_encoder"](s)
        if getattr(self.algo_config.cami, "normalize_embeddings", False):
            z = F.normalize(z, dim=-1)
        return s, z

    def _encode_snippet(self, obs_seq, use_target=False):
        """
        Encode tau_i with online or target trajectory encoder.
        """
        x = self._make_snippet_tensor(obs_seq)

        if use_target:
            with torch.no_grad():
                _, (h_n, _) = self.nets["snippet_encoder_target"](x)
                feat = h_n[-1]
                emb = self.nets["key_proj_target"](feat)
        else:
            _, (h_n, _) = self.nets["snippet_encoder"](x)
            feat = h_n[-1]
            emb = self.nets["key_proj"](feat)

        if getattr(self.algo_config.cami, "normalize_embeddings", False):
            emb = F.normalize(emb, dim=-1)

        return feat, emb

    def _forward_training(self, batch):
        """
        Forward pass for BC_RNN + CAMI.
        """
        predictions = OrderedDict()

        # BC_RNN policy branch
        actions = self.nets["policy"](
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )
        predictions["actions"] = actions

        anchor_obs = self._select_anchor_obs(batch["obs"])
        future_snippet = self._select_future_snippet(batch["obs"])

        # PDF-style state query
        _, state_query = self._encode_anchor_state(anchor_obs)
        predictions["state_query_embedding"] = state_query

        # online snippet query, SaMI-style
        online_snippet_feat, online_snippet_query = self._encode_snippet(
            future_snippet,
            use_target=False,
        )
        predictions["online_snippet_feat"] = online_snippet_feat
        predictions["traj_query_embedding"] = online_snippet_query

        # target momentum key
        target_snippet_feat, target_key_embedding = self._encode_snippet(
            future_snippet,
            use_target=True,
        )
        predictions["target_snippet_feat"] = target_snippet_feat
        predictions["target_key_embedding"] = target_key_embedding

        return predictions

    def _compute_contact_inbatch_cami_loss(self, query_embedding, key_embedding, contact_label):
        """
        Contact-aware InfoNCE:

            L_i = -log exp(q_i·k_i / beta)
                       ---------------------------------------------
                       exp(q_i·k_i / beta) + sum_{j in N_i} exp(q_i·k_j / beta)

        where N_i = {j | k_j != k_i}
        """
        beta = self.algo_config.cami.temperature

        if getattr(self.algo_config.cami, "normalize_embeddings", False):
            query_embedding = F.normalize(query_embedding, dim=-1)
            key_embedding = F.normalize(key_embedding, dim=-1)

        contact_label = contact_label.long().view(-1)
        B = query_embedding.shape[0]

        logits = torch.matmul(query_embedding, key_embedding.T) / beta
        pos_logits = logits.diag()

        neg_mask = contact_label.unsqueeze(1) != contact_label.unsqueeze(0)
        valid_neg_count = neg_mask.sum(dim=1)
        valid_anchor_mask = valid_neg_count > 0

        if valid_anchor_mask.sum() == 0:
            zero = logits.sum() * 0.0
            info = {
                "valid_anchor_count": zero.detach(),
                "valid_anchor_fraction": zero.detach(),
                "pos_logit_mean": zero.detach(),
                "neg_logit_mean": zero.detach(),
                "retrieval_acc": zero.detach(),
                "avg_valid_negatives": zero.detach(),
                "soft_scale_mean": zero.detach(),
            }
            return zero, info

        neg_logits_masked = logits.masked_fill(~neg_mask, float("-inf"))
        denom_inputs = torch.cat([pos_logits.unsqueeze(1), neg_logits_masked], dim=1)
        log_denom = torch.logsumexp(denom_inputs, dim=1)
        per_anchor_loss = -(pos_logits - log_denom)

        soft_scale_mean = torch.zeros((), device=query_embedding.device)
        if getattr(self.algo_config.cami, "soft_variant", False):
            scales = torch.ones(B, device=query_embedding.device, dtype=query_embedding.dtype)
            for i in range(B):
                if valid_anchor_mask[i]:
                    e_pos_i = key_embedding[i]
                    e_neg_i = key_embedding[neg_mask[i]]
                    dist_mean = torch.norm(e_pos_i.unsqueeze(0) - e_neg_i, dim=1).mean()
                    scales[i] = torch.clamp(dist_mean, min=1.0)
            per_anchor_loss = per_anchor_loss * scales
            soft_scale_mean = scales[valid_anchor_mask].mean()

        loss = per_anchor_loss[valid_anchor_mask].mean()

        with torch.no_grad():
            valid_pos_logits = pos_logits[valid_anchor_mask]
            pos_logit_mean = valid_pos_logits.mean()
            neg_logit_mean = logits[neg_mask].mean() if neg_mask.any() else torch.zeros((), device=logits.device)
            max_neg_logits = neg_logits_masked.max(dim=1).values
            retrieval_acc = (
                (pos_logits[valid_anchor_mask] > max_neg_logits[valid_anchor_mask]).float().mean()
            )

            info = {
                "valid_anchor_count": valid_anchor_mask.float().sum(),
                "valid_anchor_fraction": valid_anchor_mask.float().mean(),
                "pos_logit_mean": pos_logit_mean,
                "neg_logit_mean": neg_logit_mean,
                "retrieval_acc": retrieval_acc,
                "avg_valid_negatives": valid_neg_count[valid_anchor_mask].float().mean(),
                "soft_scale_mean": soft_scale_mean.detach(),
            }

        return loss, info

    def _compute_losses(self, predictions, batch):
        """
        Total loss:
            L_total = L_BC
                    + lambda_state * L_CaMI_state
                    + lambda_traj  * L_CaMI_traj
        """
        losses = OrderedDict()

        a_target = batch["actions"]
        actions = predictions["actions"]

        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        bc_action_loss = (
            self.algo_config.loss.l2_weight * losses["l2_loss"]
            + self.algo_config.loss.l1_weight * losses["l1_loss"]
            + self.algo_config.loss.cos_weight * losses["cos_loss"]
        )
        losses["bc_action_loss"] = bc_action_loss

        zero = torch.zeros((), device=actions.device, dtype=actions.dtype)

        state_cami_loss = zero
        traj_cami_loss = zero

        if getattr(self.algo_config.cami, "enabled", False):
            contact_label = batch["contact_label"].long().view(-1)
            target_key_embedding = predictions["target_key_embedding"]

            # PDF-style CAMI: psi(s_i) against phi*(tau_i)
            state_query_embedding = predictions["state_query_embedding"]
            state_cami_loss, state_info = self._compute_contact_inbatch_cami_loss(
                query_embedding=state_query_embedding,
                key_embedding=target_key_embedding,
                contact_label=contact_label,
            )
            losses["state_cami_loss"] = state_cami_loss

            # SaMI-style momentum CAMI: phi(tau_i) against phi*(tau_i)
            traj_query_embedding = predictions["traj_query_embedding"]
            traj_cami_loss, traj_info = self._compute_contact_inbatch_cami_loss(
                query_embedding=traj_query_embedding,
                key_embedding=target_key_embedding,
                contact_label=contact_label,
            )
            losses["traj_cami_loss"] = traj_cami_loss

            # log state loss diagnostics
            losses["state_valid_anchor_count"] = state_info["valid_anchor_count"]
            losses["state_valid_anchor_fraction"] = state_info["valid_anchor_fraction"]
            losses["state_pos_logit_mean"] = state_info["pos_logit_mean"]
            losses["state_neg_logit_mean"] = state_info["neg_logit_mean"]
            losses["state_retrieval_acc"] = state_info["retrieval_acc"]
            losses["state_avg_valid_negatives"] = state_info["avg_valid_negatives"]
            losses["state_soft_scale_mean"] = state_info["soft_scale_mean"]

            # log traj loss diagnostics
            losses["traj_valid_anchor_count"] = traj_info["valid_anchor_count"]
            losses["traj_valid_anchor_fraction"] = traj_info["valid_anchor_fraction"]
            losses["traj_pos_logit_mean"] = traj_info["pos_logit_mean"]
            losses["traj_neg_logit_mean"] = traj_info["neg_logit_mean"]
            losses["traj_retrieval_acc"] = traj_info["retrieval_acc"]
            losses["traj_avg_valid_negatives"] = traj_info["avg_valid_negatives"]
            losses["traj_soft_scale_mean"] = traj_info["soft_scale_mean"]

        else:
            losses["state_cami_loss"] = zero
            losses["traj_cami_loss"] = zero

        lambda_state = getattr(self.algo_config.cami, "state_loss_weight", self.algo_config.cami.loss_weight)
        lambda_traj = getattr(self.algo_config.cami, "traj_loss_weight", 1.0)

        losses["action_loss"] = (
            bc_action_loss
            + lambda_state * state_cami_loss
            + lambda_traj * traj_cami_loss
        )

        return losses

    def _train_step(self, losses):
        """
        Backprop through:
            - policy
            - state_encoder
            - snippet_encoder
            - key_proj
        """
        required_optimizers = ["policy", "state_encoder", "snippet_encoder", "key_proj"]
        missing = [k for k in required_optimizers if k not in self.optimizers]
        if len(missing) > 0:
            raise KeyError(
                "Missing optimizer entries for {}. Add them under algo.optim_params in your config.".format(missing)
            )

        info = OrderedDict()

        for name in required_optimizers:
            self.optimizers[name].zero_grad()

        losses["action_loss"].backward()

        max_grad_norm = self.global_config.train.max_grad_norm
        for name in required_optimizers:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.nets[name].parameters(),
                max_grad_norm if max_grad_norm is not None else 1e9,
            )
            info[f"{name}_grad_norm"] = float(grad_norm)

        for name in required_optimizers:
            self.optimizers[name].step()

        return info

    @torch.no_grad()
    def _update_target_networks(self):
        """
        EMA / Polyak update of target trajectory encoder.
        """
        if not getattr(self.algo_config.cami, "enabled", False):
            return
        if not getattr(self.algo_config.cami, "use_momentum_target", True):
            return

        m = getattr(self.algo_config.cami, "momentum", None)
        if m is None:
            tau = getattr(self.algo_config.cami, "target_tau", 0.05)
            m = 1.0 - tau

        for online_param, target_param in zip(
            self.nets["snippet_encoder"].parameters(),
            self.nets["snippet_encoder_target"].parameters(),
        ):
            target_param.data.mul_(m)
            target_param.data.add_((1.0 - m) * online_param.data)

        for online_param, target_param in zip(
            self.nets["key_proj"].parameters(),
            self.nets["key_proj_target"].parameters(),
        ):
            target_param.data.mul_(m)
            target_param.data.add_((1.0 - m) * online_param.data)

    def log_info(self, info):
        log = super(BC_CaMI, self).log_info(info)
        losses = info["losses"]

        log["Loss"] = losses["action_loss"].item()
        log["BC_Action_Loss"] = losses["bc_action_loss"].item()
        log["State_CaMI_Loss"] = losses["state_cami_loss"].item()
        log["Traj_CaMI_Loss"] = losses["traj_cami_loss"].item()

        if "l2_loss" in losses:
            log["L2_Loss"] = losses["l2_loss"].item()
        if "l1_loss" in losses:
            log["L1_Loss"] = losses["l1_loss"].item()
        if "cos_loss" in losses:
            log["Cosine_Loss"] = losses["cos_loss"].item()

        for key in [
            "state_valid_anchor_count",
            "state_valid_anchor_fraction",
            "state_pos_logit_mean",
            "state_neg_logit_mean",
            "state_retrieval_acc",
            "state_avg_valid_negatives",
            "state_soft_scale_mean",
            "traj_valid_anchor_count",
            "traj_valid_anchor_fraction",
            "traj_pos_logit_mean",
            "traj_neg_logit_mean",
            "traj_retrieval_acc",
            "traj_avg_valid_negatives",
            "traj_soft_scale_mean",
        ]:
            if key in losses:
                log[key] = losses[key].item()

        for key in [
            "policy_grad_norm",
            "state_encoder_grad_norm",
            "snippet_encoder_grad_norm",
            "key_proj_grad_norm",
        ]:
            if key in info:
                log[key] = info[key]

        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Preserve BC_RNN rollout behavior.
        """
        assert not self.nets.training

        if "force" not in obs_dict:
            if ("robot0_ee_force" in obs_dict) and ("robot0_ee_torque" in obs_dict):
                obs_dict = dict(obs_dict)
                obs_dict["force"] = torch.cat(
                    [obs_dict["robot0_ee_force"], obs_dict["robot0_ee_torque"]],
                    dim=-1,
                )

        expected_keys = list(self.obs_shapes.keys())
        filtered_obs_dict = {k: obs_dict[k] for k in expected_keys if k in obs_dict}
        missing_keys = [k for k in expected_keys if k not in filtered_obs_dict]
        if len(missing_keys) > 0:
            raise KeyError(f"Missing required rollout observation keys: {missing_keys}")

        return super(BC_CaMI, self).get_action(filtered_obs_dict, goal_dict=goal_dict)
