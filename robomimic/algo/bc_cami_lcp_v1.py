"""
BC_RNN + Continuous Contact-Aware Mutual Information (LCP formulation)
for robomimic.

This is the continuous extension of the discrete BC_CaMI algorithm
(Sec 4.1.5 of the thesis), replacing the binary-contact InfoNCE objective
with a violation-based implicit loss derived from the Linear
Complementarity Problem (LCP) formulation of unilateral rigid-body contact.

Physical constraint (Eq. 4.15 / 4.17), per contact point i = 1..p:

    lambda_hat_i(t) >= 0,  phi_hat_i(t+1) >= 0,  lambda_hat_i(t) * phi_hat_i(t+1) = 0

Two encoders:
    E_v : o_t^v  -> phi_hat(t)    (visual/proprio obs encoding -> estimated gap)
    E_f : o_t^f  -> lambda_hat(t) (force/torque signal -> estimated impulse, >= 0)

Loss (Eq. 4.16 / 4.18-4.20):

    L_LCP = w_comp * mean_j [ sum_i lambda_hat_i(t) * phi_hat_i(t+1) ]      # complementarity
          + w_pen  * mean_j [ sum_i min(0, phi_hat_i(t+1))^2 ]             # non-penetration

Total objective (Eq. 4.21):

    L_total = L_BC + w_lcp * L_LCP

Design notes carried over from discussion:
    - E_v consumes the PRE-RNN, per-timestep multimodal observation encoding
      (the output of the policy's ObservationGroupEncoder, i.e. "obs_encoding"
      below), NOT the post-RNN policy latent h_i used by the discrete branch.
      This requires the policy's forward_with_features to expose that tensor
      via a `return_obs_encoding` flag -- see the accompanying patch to
      RNN_MIMO_MLP.forward_with_features in base_nets.py.
    - phi_hat is left UNCONSTRAINED at the encoder (no clamping activation),
      so the non-penetration penalty has nonzero gradient when violated.
    - lambda_hat is constrained via softplus, since negative impulse has no
      physical meaning and there is no loss term that needs to see negative
      lambda_hat to be useful.
    - No momentum/target encoder, no negative sampling, no snippet horizon:
      L_LCP only needs phi_hat(t+1) and lambda_hat(t), so all of that
      discrete-CaMI machinery is removed.
    - contact_label, if present in the batch, is used ONLY as a diagnostic
      (calibration_rho below) to detect the degenerate collapse failure
      mode (lambda_hat -> 0 everywhere trivially satisfies complementarity
      without learning anything about contact). It never enters any loss.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils

from robomimic.algo import register_algo_factory_func
from robomimic.algo.bc import BC_RNN


@register_algo_factory_func("bc_cami_lcp")
def algo_config_to_class(algo_config):
    return BC_CaMI_LCP, {}


def build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class _PolicyLatentMixin:
    """
    Shared helpers that are identical between discrete BC_CaMI and
    continuous BC_CaMI_LCP: exposing the policy's internal features during
    training, and filtering rollout observations at inference time.
    Both algo classes should inherit this alongside BC_RNN.
    """

    def _forward_policy_with_latent(self, obs_dict, goal_dict=None, return_obs_encoding=False):
        """
        Forward policy and return action outputs, per-step post-RNN latent
        features (h_i, Eq. 4.4), and optionally the pre-RNN multimodal
        observation encoding (o_t^v (+proprio), needed for E_v).
        """
        if not hasattr(self.nets["policy"], "forward_with_features"):
            raise AttributeError(
                "Policy network does not expose forward_with_features."
            )

        out = self.nets["policy"].forward_with_features(
            obs=obs_dict,
            goal=goal_dict,
            return_obs_encoding=return_obs_encoding,
        )

        if return_obs_encoding:
            if not isinstance(out, tuple) or len(out) != 3:
                raise RuntimeError(
                    "policy.forward_with_features(return_obs_encoding=True) must "
                    "return (actions, feats, obs_encoding), but got type {} len {}".format(
                        type(out), len(out) if isinstance(out, tuple) else "n/a"
                    )
                )
            actions, feats, obs_encoding = out
        else:
            if not isinstance(out, tuple) or len(out) != 2:
                raise RuntimeError(
                    "policy.forward_with_features must return (actions, feats), "
                    "but got type {}".format(type(out))
                )
            actions, feats = out
            obs_encoding = None

        if isinstance(actions, dict):
            if "action" in actions:
                actions = actions["action"]
            elif "actions" in actions:
                actions = actions["actions"]
            else:
                raise RuntimeError(
                    "Policy returned dict outputs but no 'action' or 'actions' key was found. "
                    f"Available keys: {list(actions.keys())}"
                )

        if not torch.is_tensor(actions):
            raise RuntimeError(
                "Expected policy actions to be a tensor, got {}".format(type(actions))
            )
        if not torch.is_tensor(feats):
            raise RuntimeError(
                "Expected policy feats to be a tensor, got {}".format(type(feats))
            )
        if feats.ndim != 3:
            raise RuntimeError(
                "Expected policy feats with shape [B, T, D], got shape {}".format(tuple(feats.shape))
            )
        if return_obs_encoding:
            if not torch.is_tensor(obs_encoding):
                raise RuntimeError(
                    "Expected obs_encoding to be a tensor, got {}".format(type(obs_encoding))
                )
            if obs_encoding.ndim != 3:
                raise RuntimeError(
                    "Expected obs_encoding with shape [B, T, D], got shape {}".format(
                        tuple(obs_encoding.shape)
                    )
                )
            return actions, feats, obs_encoding

        return actions, feats

    def get_action(self, obs_dict, goal_dict=None):
        """
        Preserve BC_RNN rollout behavior. Force and any privileged-only
        keys are never part of self.obs_shapes, so filtering against
        expected_keys is sufficient to keep them out of rollout.
        """
        assert not self.nets.training

        expected_keys = list(self.obs_shapes.keys())
        filtered_obs_dict = {k: obs_dict[k] for k in expected_keys if k in obs_dict}
        missing_keys = [k for k in expected_keys if k not in filtered_obs_dict]
        if len(missing_keys) > 0:
            raise KeyError(f"Missing required rollout observation keys: {missing_keys}")

        return super(_PolicyLatentMixin, self).get_action(filtered_obs_dict, goal_dict=goal_dict)


class BC_CaMI_LCP(_PolicyLatentMixin, BC_RNN):
    """
    BC_RNN with continuous, LCP-based Contact-Aware Mutual Information
    regularization (Sec 4.1.5 extension).

    Policy:
        standard BC_RNN policy over full observation sequences

    LCP branch:
        - gap encoder E_v(o_t^v) -> phi_hat(t), unconstrained
        - impulse encoder E_f(o_t^f) -> lambda_hat(t), >= 0 via softplus
        - violation loss combining complementarity slackness and
          non-penetration penalty (Eq. 4.16/4.20)
    """

    def _create_networks(self):
        super(BC_CaMI_LCP, self)._create_networks()

        lcp_cfg = self.algo_config.cami.lcp

        num_contacts = lcp_cfg.num_contacts
        force_dim = lcp_cfg.force_dim
        gap_hidden = list(lcp_cfg.gap_hidden_dims)
        impulse_hidden = list(lcp_cfg.impulse_hidden_dims)

        # Dimension of the policy's pre-RNN multimodal observation encoding
        # (visual + proprioceptive, concatenated). Computed dynamically from
        # the already-built policy encoder rather than hardcoded in config,
        # so it always matches whatever obs_shapes/encoder_kwargs actually
        # produced -- avoids silent mismatches if the obs config changes.
        obs_encoding_dim = self.nets["policy"].nets["encoder"].output_shape()[0]

        self.nets["gap_encoder"] = build_mlp(
            input_dim=obs_encoding_dim,
            hidden_dims=gap_hidden,
            output_dim=num_contacts,
        )
        self.nets["impulse_encoder"] = build_mlp(
            input_dim=force_dim,
            hidden_dims=impulse_hidden,
            output_dim=num_contacts,
        )

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Keep BC_RNN-compatible full sequences.

        Force is kept (unlike the discrete BC_CaMI, which drops it after
        computing offline contact labels) because E_f needs raw force at
        training time. Force is still excluded from the policy's own obs
        dict, so it is never seen at inference (privileged-only, per the
        thesis's force-usage constraint from Sec 4.1.2).

        contact_label, if present, is kept ONLY as an optional diagnostic
        (see calibration_rho in _compute_losses) -- it is not required.
        """
        input_batch = dict()
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"]

        if "force" in batch["obs"]:
            force = batch["obs"]["force"]
        elif "force" in batch:
            force = batch["force"]
        else:
            raise KeyError(
                "BC_CaMI_LCP requires raw force/torque in batch['obs']['force'] "
                "(shape [B, T, D_f]). This is different from discrete BC_CaMI, "
                "which only needs a precomputed binary contact_label."
            )
        input_batch["force"] = force

        contact_label = None
        if "contact_label" in batch:
            contact_label = batch["contact_label"]
        elif "contact_label" in batch["obs"]:
            contact_label = batch["obs"]["contact_label"]

        if contact_label is not None:
            if contact_label.ndim > 1:
                contact_label = contact_label[:, 0]
            if contact_label.ndim > 1:
                contact_label = contact_label.squeeze(-1)
            input_batch["contact_label"] = contact_label.float()
        else:
            input_batch["contact_label"] = None

        # Policy obs must stay clean: exclude force and contact_label
        input_batch["obs"] = {
            k: batch["obs"][k]
            for k in batch["obs"]
            if k not in ["force", "contact_label"]
        }

        if not hasattr(self, "_debug_printed_batch_stats"):
            self._debug_printed_batch_stats = False
        if not self._debug_printed_batch_stats:
            print("\n[BC_CaMI_LCP DEBUG] process_batch_for_training")
            print("  actions shape       :", tuple(batch["actions"].shape))
            print("  force shape         :", tuple(input_batch["force"].shape))
            print("  contact_label avail :", input_batch["contact_label"] is not None)
            print("  policy obs keys     :", list(input_batch["obs"].keys()))
            self._debug_printed_batch_stats = True

        out = TensorUtils.to_device(input_batch, self.device)
        # to_float would choke on the None contact_label entry, so convert
        # selectively instead of blanket TensorUtils.to_float(out).
        out["actions"] = out["actions"].float()
        out["force"] = out["force"].float()
        out["obs"] = TensorUtils.to_float(out["obs"])
        if out["contact_label"] is not None:
            out["contact_label"] = out["contact_label"].float()
        return out

    def train_on_batch(self, batch, epoch, validate=False):
        """
        No target-network update needed (no momentum encoder in the
        continuous formulation), so this is just the standard BC_RNN step.
        """
        return super(BC_CaMI_LCP, self).train_on_batch(
            batch=batch,
            epoch=epoch,
            validate=validate,
        )

    def _forward_training(self, batch):
        predictions = OrderedDict()

        policy_actions, policy_feats, obs_encoding = self._forward_policy_with_latent(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            return_obs_encoding=True,
        )
        predictions["actions"] = policy_actions

        if obs_encoding.shape[1] < 2:
            raise RuntimeError(
                "BC_CaMI_LCP needs at least 2 timesteps to form phi_hat(t+1); "
                "check algo_config.train.seq_length (currently T={}).".format(
                    obs_encoding.shape[1]
                )
            )

        # phi_hat(t+1): gap estimate from NEXT-step visual/proprio encoding.
        # Only t+1 is needed -- Eq. 4.18/4.19 never reference phi_hat(t).
        obs_encoding_tp1 = obs_encoding[:, 1, :]
        phi_hat_tp1 = self.nets["gap_encoder"](obs_encoding_tp1)

        # lambda_hat(t): impulse estimate from CURRENT-step force.
        force_t = batch["force"][:, 0, :]
        lambda_hat_t = F.softplus(self.nets["impulse_encoder"](force_t))

        predictions["phi_hat_tp1"] = phi_hat_tp1
        predictions["lambda_hat_t"] = lambda_hat_t

        if not hasattr(self, "_debug_printed_lcp_stats"):
            self._debug_printed_lcp_stats = False
        if not self._debug_printed_lcp_stats:
            print("\n[BC_CaMI_LCP DEBUG] _forward_training")
            print("  obs_encoding shape   :", tuple(obs_encoding.shape))
            print("  policy_feats shape   :", tuple(policy_feats.shape))
            print("  phi_hat_tp1 shape    :", tuple(phi_hat_tp1.shape))
            print("  lambda_hat_t shape   :", tuple(lambda_hat_t.shape))
            print("  phi_hat_tp1 sample   :", phi_hat_tp1[0].detach().cpu())
            print("  lambda_hat_t sample  :", lambda_hat_t[0].detach().cpu())
            self._debug_printed_lcp_stats = True

        return predictions

    def _compute_lcp_loss(self, phi_hat_tp1, lambda_hat_t):
        """
        Eq. 4.18-4.20:
            l_comp^(j) = sum_i lambda_hat_i(t) * phi_hat_i(t+1)
            l_pen^(j)  = sum_i [min(0, phi_hat_i(t+1))]^2
            L_LCP      = w_comp * mean_j l_comp^(j) + w_pen * mean_j l_pen^(j)
        """
        complementarity = (lambda_hat_t * phi_hat_tp1).sum(dim=-1)          # [B]
        penetration = torch.clamp(phi_hat_tp1, max=0.0).pow(2).sum(dim=-1)  # [B]

        l_comp = complementarity.mean()
        l_pen = penetration.mean()

        w_comp = self.algo_config.cami.lcp.w_comp
        w_pen = self.algo_config.cami.lcp.w_pen
        l_lcp = w_comp * l_comp + w_pen * l_pen

        info = {"l_comp": l_comp.detach(), "l_pen": l_pen.detach()}
        return l_lcp, info

    def _compute_losses(self, predictions, batch):
        """
        Total loss (Eq. 4.21):
            L_total = L_BC + w_lcp * L_LCP
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

        lcp_enabled = (
            self.algo_config.cami.enabled
            if "enabled" in self.algo_config.cami
            else False
        )

        zero = torch.zeros((), device=actions.device, dtype=actions.dtype)
        lcp_loss = zero

        if lcp_enabled:
            lcp_loss, lcp_info = self._compute_lcp_loss(
                predictions["phi_hat_tp1"], predictions["lambda_hat_t"]
            )
            losses["lcp_loss"] = lcp_loss
            losses["l_comp"] = lcp_info["l_comp"]
            losses["l_pen"] = lcp_info["l_pen"]

            # Diagnostics: collapse detection + optional calibration check.
            # NEITHER of these feeds into any loss -- inspection only.
            with torch.no_grad():
                losses["lambda_hat_mean"] = predictions["lambda_hat_t"].mean()
                losses["phi_hat_mean"] = predictions["phi_hat_tp1"].mean()

                contact_label = batch.get("contact_label", None)
                if contact_label is not None:
                    # mean gap per sample across contact points, then sign
                    phi_sign = (predictions["phi_hat_tp1"].mean(dim=-1) < 0).float()
                    if phi_sign.std() > 0 and contact_label.std() > 0:
                        stacked = torch.stack([phi_sign, contact_label])
                        losses["calibration_rho"] = torch.corrcoef(stacked)[0, 1]
                    else:
                        # degenerate: no variance in one of the two signals
                        # (e.g. all-contact or all-free-space batch, or
                        # phi_hat collapsed to one sign) -- corrcoef is
                        # undefined, report NaN rather than a misleading 0.
                        losses["calibration_rho"] = torch.tensor(
                            float("nan"), device=actions.device
                        )
        else:
            losses["lcp_loss"] = zero
            losses["l_comp"] = zero
            losses["l_pen"] = zero

        w_lcp = self.algo_config.cami.lcp.loss_weight
        losses["action_loss"] = bc_action_loss + w_lcp * lcp_loss

        return losses

    def _train_step(self, losses):
        """
        Backprop through:
            - policy
            - gap_encoder
            - impulse_encoder
        """
        required_optimizers = ["policy", "gap_encoder", "impulse_encoder"]
        missing = [k for k in required_optimizers if k not in self.optimizers]
        if len(missing) > 0:
            raise KeyError(
                "Missing optimizer entries for {}. Add them under algo.optim_params "
                "in your config.".format(missing)
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

    def log_info(self, info):
        log = super(BC_CaMI_LCP, self).log_info(info)
        losses = info["losses"]

        log["Loss"] = losses["action_loss"].item()
        log["BC_Action_Loss"] = losses["bc_action_loss"].item()
        log["LCP_Loss"] = losses["lcp_loss"].item()
        log["Complementarity_Loss"] = losses["l_comp"].item()
        log["Penetration_Loss"] = losses["l_pen"].item()

        if "l2_loss" in losses:
            log["L2_Loss"] = losses["l2_loss"].item()
        if "l1_loss" in losses:
            log["L1_Loss"] = losses["l1_loss"].item()
        if "cos_loss" in losses:
            log["Cosine_Loss"] = losses["cos_loss"].item()

        if "lambda_hat_mean" in losses:
            log["Lambda_Hat_Mean"] = losses["lambda_hat_mean"].item()
        if "phi_hat_mean" in losses:
            log["Phi_Hat_Mean"] = losses["phi_hat_mean"].item()
        if "calibration_rho" in losses:
            val = losses["calibration_rho"].item()
            log["Calibration_Rho"] = val  # may be NaN; see _compute_losses note

        for key in ["policy_grad_norm", "gap_encoder_grad_norm", "impulse_encoder_grad_norm"]:
            if key in info:
                log[key] = info[key]

        return log