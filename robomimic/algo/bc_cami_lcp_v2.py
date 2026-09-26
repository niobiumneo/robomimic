"""
BC_RNN + Contrastive Contact-Aware Noise Contrastive Estimation (C-CaNCE)
for robomimic.

This reformulates the continuous LCP violation loss (BC_CaMI_LCP,
Eq. 4.16/4.20) as a noise-contrastive estimator over a complementarity
energy, unifying it with the discrete CaMI branch's InfoNCE formulation
(Eq. 4.10-4.11) as two instances of the same discriminative estimation
principle, differing only in their score function.

--------------------------------------------------------------------------
Derivation summary (full derivation in accompanying thesis notes)
--------------------------------------------------------------------------

Energy function, per contact point i = 1..p (Eq. 4.22):
    E(lambda_hat, phi_hat) = lambda_hat^T phi_hat = sum_i lambda_hat_i * phi_hat_i
    E >= 0, with equality iff complementary slackness (Eq. 4.17) holds.

Boltzmann density (Eq. 4.23):
    p_theta(phi | lambda) = exp(-E(lambda,phi)/beta) / Z_theta(lambda)
    Z_theta is intractable -- this is the unnormalized-model setting NCE
    (Gutmann & Hyvarinen, 2010) was designed for.

Per-anchor contrastive loss (Eq. 4.24, generalized to a harvested pair
pool in Eq. 4.31-4.32):
    L_C-CaNCE^(k) = -log[ exp(-E_kk/beta) / (exp(-E_kk/beta) + sum_{l in N_k} exp(-E_kl/beta)) ]
    where E_kl = lambda_hat_k^T phi_hat_l, and N_k is anchor k's negative set.

Key results from the derivation (see thesis notes for full proofs):
    - Eq. 4.25: L_C-CaNCE = (1/beta) * l_comp + log-sum-exp normalizer.
      The normalizer is the ONLY thing added relative to the old L_LCP's
      complementarity term -- and it is exactly what prevents the
      lambda_hat -> 0 collapse (Eq. 4.26): collapse pushes every logit to
      0, giving L -> log(1+|N_k|), which is the loss's CEILING, not a
      minimizer.
    - Eq. 4.28: the non-penetration term CANNOT be folded into the
      softmax score -- doing so gives phi_hat_i a gradient that actively
      REWARDS penetration for the positive pair. It must stay as an
      explicit penalty (L_pen below) outside the contrastive score.
    - The energy is bilinear in (lambda_hat, phi_hat), so uniformly
      scaling lambda_hat sharpens the softmax and drives the loss toward
      zero without learning anything (scale degeneracy). Eq. 4.29 fixes
      this by anchoring ||lambda_hat||_1 to the MEASURED force magnitude
      (force is privileged/available at training time), rather than by
      normalizing embeddings as generic contrastive learning would.

--------------------------------------------------------------------------
Negative sampling: two modes, same loss shape (Eq. 4.31)
--------------------------------------------------------------------------

    "marginal": negatives = any pair from a DIFFERENT source sequence in
        the batch, regardless of contact regime. Keeps a clean I(Lambda;Phi)
        lower-bound claim (no contact-label dependence in the loss), but
        negatives are frequently "easy" (unrelated task phases).

    "regime": negatives = pairs from a different source sequence AND the
        OPPOSITE contact regime (mirrors discrete CaMI's Eq. 4.8 negative
        construction). Harder, more informative negatives and consistent
        with the "contact-aware" framing of the rest of the method, but
        the quantity estimated is then a contact-CONDITIONAL discrimination
        rather than the marginal I(Lambda;Phi) -- state this precisely in
        the thesis rather than citing the plain InfoNCE bound. Requires
        contact_label (per-timestep) in the batch; "marginal" does not.

Both modes are implemented here behind algo_config.cami.lcp.negative_mode
so the same class produces both ablation conditions from one config field.

--------------------------------------------------------------------------
Pair harvesting (Eq. 4.31, avoiding both under-use of the batch and
false negatives from near-duplicate adjacent frames)
--------------------------------------------------------------------------

Naively using only the pair (t=0, t+1=1) per sequence wastes the T-1
other consecutive pairs already available in every loaded sequence (the
RNN already processes the full sequence for L_BC). This implementation
harvests ALL consecutive pairs (t, t+1) for t = 0..T-2 from every
sequence in the batch, flattening to a pool of size N = B*(T-1) anchors,
which enlarges the negative pool (K) essentially for free.

To avoid treating adjacent-timestep pairs from the SAME sequence as
negatives of each other (they are usually near-duplicates physically,
which would inject false negatives), the negative mask always excludes
same-sequence pairs, in addition to whatever contact-regime condition
negative_mode adds.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils

from robomimic.algo import register_algo_factory_func
from robomimic.algo.bc import BC_RNN

@register_algo_factory_func("bc_cami_cance")
def algo_config_to_class(algo_config):
    return BC_CaMI_CaNCE, {}

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

class BC_CaMI_CaNCE(_PolicyLatentMixin, BC_RNN):
    """
    BC_RNN with Contrastive Contact-Aware Noise Contrastive Estimation
    (C-CaNCE): the NCE-reformulated successor to BC_CaMI_LCP.

    Policy:
        standard BC_RNN policy over full observation sequences

    C-CaNCE branch:
        - gap encoder E_v(o_t^v) -> phi_hat(t), unconstrained, applied at
          EVERY timestep (needed to build the harvested pair pool)
        - impulse encoder E_f(o_t^f) -> lambda_hat(t), >= 0 via softplus,
          applied at every timestep
        - contrastive complementarity loss (Eq. 4.24/4.32) in place of
          the bare violation term
        - explicit non-penetration penalty (Eq. 4.19), kept OUTSIDE the
          softmax per the Eq. 4.28 gradient analysis
        - magnitude-anchoring term (Eq. 4.29) against measured force,
          fixing the bilinear scale degeneracy
    """

    def _create_networks(self):
        super(BC_CaMI_CaNCE, self)._create_networks()

        lcp_cfg = self.algo_config.cami.lcp

        num_contacts = lcp_cfg.num_contacts
        force_dim = lcp_cfg.force_dim
        gap_hidden = list(lcp_cfg.gap_hidden_dims)
        impulse_hidden = list(lcp_cfg.impulse_hidden_dims)

        negative_mode = (
            lcp_cfg.negative_mode if "negative_mode" in lcp_cfg else "regime"
        )
        if negative_mode not in ("marginal", "regime"):
            raise ValueError(
                "algo_config.cami.lcp.negative_mode must be 'marginal' or "
                "'regime', got '{}'.".format(negative_mode)
            )
        self._negative_mode = negative_mode

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
        Same as BC_CaMI_LCP, EXCEPT contact_label is kept per-timestep
        ([B, T]) rather than squeezed to one label per sequence, since the
        "regime" negative mode needs the contact regime AT EACH anchor
        timestep in the harvested pair pool, not just one label for the
        whole window.

        Backward-compat fallback: if contact_label arrives as [B] (one
        label per sequence, as the discrete BC_CaMI's pipeline produces),
        it is broadcast across the time dimension with a one-time warning,
        since that is a coarser signal than true per-timestep labels.
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
                "BC_CaMI_CaNCE requires raw force/torque in batch['obs']['force'] "
                "(shape [B, T, D_f])."
            )
        input_batch["force"] = force
        T = force.shape[1]

        contact_label = None
        if "contact_label" in batch:
            contact_label = batch["contact_label"]
        elif "contact_label" in batch["obs"]:
            contact_label = batch["obs"]["contact_label"]

        if contact_label is not None:
            if contact_label.ndim == 3:
                contact_label = contact_label.squeeze(-1)  # [B, T, 1] -> [B, T]
            if contact_label.ndim == 1:
                contact_label = contact_label.unsqueeze(0)  # defensive
            if contact_label.shape[1] == 1 and T > 1:
                if not hasattr(self, "_warned_broadcast_contact_label"):
                    self._warned_broadcast_contact_label = False
                if not self._warned_broadcast_contact_label:
                    print(
                        "[BC_CaMI_CaNCE WARNING] contact_label has only 1 timestep "
                        "but sequences have T={}; broadcasting the single label "
                        "across the window. Per-timestep labels are preferred for "
                        "the 'regime' negative_mode.".format(T)
                    )
                    self._warned_broadcast_contact_label = True
                contact_label = contact_label.expand(-1, T)
            input_batch["contact_label"] = contact_label.float()
        else:
            input_batch["contact_label"] = None

        if self._negative_mode == "regime" and input_batch["contact_label"] is None:
            raise KeyError(
                "negative_mode='regime' requires contact_label in the batch "
                "(per-timestep preferred). Use negative_mode='marginal' if "
                "contact labels are unavailable."
            )

        input_batch["obs"] = {
            k: batch["obs"][k]
            for k in batch["obs"]
            if k not in ["force", "contact_label"]
        }

        if not hasattr(self, "_debug_printed_batch_stats"):
            self._debug_printed_batch_stats = False
        if not self._debug_printed_batch_stats:
            print("\n[BC_CaMI_CaNCE DEBUG] process_batch_for_training")
            print("  negative_mode       :", self._negative_mode)
            print("  actions shape       :", tuple(batch["actions"].shape))
            print("  force shape         :", tuple(input_batch["force"].shape))
            print("  contact_label avail :", input_batch["contact_label"] is not None)
            if input_batch["contact_label"] is not None:
                print("  contact_label shape :", tuple(input_batch["contact_label"].shape))
            print("  policy obs keys     :", list(input_batch["obs"].keys()))
            self._debug_printed_batch_stats = True

        out = TensorUtils.to_device(input_batch, self.device)
        out["actions"] = out["actions"].float()
        out["force"] = out["force"].float()
        out["obs"] = TensorUtils.to_float(out["obs"])
        if out["contact_label"] is not None:
            out["contact_label"] = out["contact_label"].float()
        return out

    def train_on_batch(self, batch, epoch, validate=False):
        return super(BC_CaMI_CaNCE, self).train_on_batch(
            batch=batch, epoch=epoch, validate=validate,
        )

    def _forward_training(self, batch):
        predictions = OrderedDict()

        policy_actions, policy_feats, obs_encoding = self._forward_policy_with_latent(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            return_obs_encoding=True,
        )
        predictions["actions"] = policy_actions

        B, T = obs_encoding.shape[0], obs_encoding.shape[1]
        if T < 2:
            raise RuntimeError(
                "BC_CaMI_CaNCE needs at least 2 timesteps per sequence to form "
                "consecutive pairs; check algo_config.train.seq_length (T={}).".format(T)
            )

        # Apply E_v and E_f at EVERY timestep (not just one pair) so the
        # full pool of T-1 consecutive pairs per sequence can be harvested.
        phi_hat_all = TensorUtils.time_distributed(obs_encoding, self.nets["gap_encoder"])
        lambda_hat_all = F.softplus(
            TensorUtils.time_distributed(batch["force"], self.nets["impulse_encoder"])
        )

        predictions["phi_hat_all"] = phi_hat_all         # [B, T, p]
        predictions["lambda_hat_all"] = lambda_hat_all   # [B, T, p]

        if not hasattr(self, "_debug_printed_cance_stats"):
            self._debug_printed_cance_stats = False
        if not self._debug_printed_cance_stats:
            print("\n[BC_CaMI_CaNCE DEBUG] _forward_training")
            print("  obs_encoding shape  :", tuple(obs_encoding.shape))
            print("  phi_hat_all shape   :", tuple(phi_hat_all.shape))
            print("  lambda_hat_all shape:", tuple(lambda_hat_all.shape))
            print("  pairs per batch (N) :", B * (T - 1))
            self._debug_printed_cance_stats = True

        return predictions

    def _harvest_pairs(self, phi_hat_all, lambda_hat_all, force, contact_label):
        """
        Flatten [B, T, ...] tensors into the pair pool P (Eq. 4.31):
            anchor k = (b, t),  t = 0..T-2
            lambda_hat_k = lambda_hat(t; b)        (impulse at anchor time)
            phi_hat_k    = phi_hat(t+1; b)          (gap at successor time, the positive)
        Also returns seq_id (for same-sequence exclusion) and, if available,
        the per-anchor contact label and force magnitude.
        """
        B, T, p = phi_hat_all.shape

        lambda_anchor = lambda_hat_all[:, :-1, :].reshape(B * (T - 1), p)
        phi_positive = phi_hat_all[:, 1:, :].reshape(B * (T - 1), p)
        force_anchor = force[:, :-1, :].reshape(B * (T - 1), force.shape[-1])

        seq_id = (
            torch.arange(B, device=phi_hat_all.device)
            .unsqueeze(1)
            .expand(B, T - 1)
            .reshape(-1)
        )

        contact_anchor = None
        if contact_label is not None:
            contact_anchor = contact_label[:, :-1].reshape(-1)

        return lambda_anchor, phi_positive, force_anchor, seq_id, contact_anchor

    def _compute_cance_loss(self, lambda_anchor, phi_positive, seq_id, contact_anchor):
        """
        Eq. 4.24/4.32, with the negative mask (Eq. 4.31) built from
        same-sequence exclusion (always) intersected with opposite-regime
        exclusion (only when negative_mode == "regime").
        """
        N = lambda_anchor.shape[0]
        beta = self.algo_config.cami.lcp.beta

        # Energy matrix E[k, l] = lambda_hat_k^T phi_hat_l  (Eq. 4.22)
        E = torch.matmul(lambda_anchor, phi_positive.T)  # [N, N]
        logits = -E / beta  # low violation -> high logit
        pos_logits = logits.diag()

        diff_seq = seq_id.unsqueeze(1) != seq_id.unsqueeze(0)
        neg_mask = diff_seq

        if self._negative_mode == "regime":
            diff_regime = contact_anchor.unsqueeze(1) != contact_anchor.unsqueeze(0)
            neg_mask = neg_mask & diff_regime

        valid_neg_count = neg_mask.sum(dim=1)
        valid_anchor_mask = valid_neg_count > 0

        if not hasattr(self, "_debug_printed_cance_neg_stats"):
            self._debug_printed_cance_neg_stats = False
        if not self._debug_printed_cance_neg_stats:
            print("\n[BC_CaMI_CaNCE DEBUG] _compute_cance_loss")
            print("  N (pool size)        :", N)
            print("  valid_neg_count min/max/mean:",
                  valid_neg_count.min().item(), valid_neg_count.max().item(),
                  valid_neg_count.float().mean().item())
            print("  valid_anchor_fraction:", valid_anchor_mask.float().mean().item())
            self._debug_printed_cance_neg_stats = True

        if valid_anchor_mask.sum() == 0:
            zero = logits.sum() * 0.0
            info = {
                "retrieval_acc": zero.detach(),
                "avg_valid_negatives": zero.detach(),
                "valid_anchor_fraction": zero.detach(),
                "collapse_ceiling": zero.detach(),
                "pos_logit_mean": zero.detach(),
                "neg_logit_mean": zero.detach(),
            }
            return zero, info

        neg_logits_masked = logits.masked_fill(~neg_mask, float("-inf"))
        denom_inputs = torch.cat([pos_logits.unsqueeze(1), neg_logits_masked], dim=1)
        log_denom = torch.logsumexp(denom_inputs, dim=1)
        per_anchor_loss = -(pos_logits - log_denom)
        loss = per_anchor_loss[valid_anchor_mask].mean()

        with torch.no_grad():
            max_neg_logits = neg_logits_masked.max(dim=1).values
            retrieval_acc = (
                pos_logits[valid_anchor_mask] > max_neg_logits[valid_anchor_mask]
            ).float().mean()
            # Per-anchor collapse ceiling: L^(k) -> log(1 + |N_k|) if the
            # model collapses (lambda_hat -> 0 uniformly). Comparing the
            # running loss against this each batch is a useful sanity
            # check: loss should sit meaningfully BELOW this ceiling.
            collapse_ceiling = torch.log1p(
                valid_neg_count[valid_anchor_mask].float()
            ).mean()
            avg_valid_negatives = valid_neg_count[valid_anchor_mask].float().mean()
            pos_logit_mean = pos_logits[valid_anchor_mask].mean()
            neg_logit_mean = (
                logits[neg_mask].mean() if neg_mask.any()
                else torch.zeros((), device=logits.device)
            )

        info = {
            "retrieval_acc": retrieval_acc,
            "avg_valid_negatives": avg_valid_negatives,
            "valid_anchor_fraction": valid_anchor_mask.float().mean(),
            "collapse_ceiling": collapse_ceiling,
            "pos_logit_mean": pos_logit_mean,
            "neg_logit_mean": neg_logit_mean,
        }
        return loss, info

    def _compute_pen_loss(self, phi_positive):
        """Eq. 4.19, kept OUTSIDE the softmax per the Eq. 4.28 gradient analysis."""
        return torch.clamp(phi_positive, max=0.0).pow(2).sum(dim=-1).mean()

    def _compute_mag_loss(self, lambda_anchor, force_anchor):
        """
        Eq. 4.29: anchor total estimated impulse magnitude to the measured
        TRANSLATIONAL force magnitude, fixing the bilinear scale
        degeneracy of the contrastive energy. force_anchor's first 3
        channels are assumed to be [fx, fy, fz] of the 6D wrench.
        """
        f_mag = force_anchor[..., :3].norm(dim=-1)
        lambda_mag = lambda_anchor.sum(dim=-1)  # already >= 0 via softplus, so sum == L1 norm
        return F.mse_loss(lambda_mag, f_mag)

    def _compute_losses(self, predictions, batch):
        """
        Total loss (Eq. 4.30):
            L_total = L_BC + w_nce * L_C-CaNCE + w_pen * L_pen + w_mag * L_mag
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

        cami_enabled = (
            self.algo_config.cami.enabled if "enabled" in self.algo_config.cami else False
        )

        zero = torch.zeros((), device=actions.device, dtype=actions.dtype)
        nce_loss = zero
        pen_loss = zero
        mag_loss = zero

        if cami_enabled:
            lambda_anchor, phi_positive, force_anchor, seq_id, contact_anchor = self._harvest_pairs(
                predictions["phi_hat_all"],
                predictions["lambda_hat_all"],
                batch["force"],
                batch["contact_label"],
            )

            nce_loss, nce_info = self._compute_cance_loss(
                lambda_anchor, phi_positive, seq_id, contact_anchor
            )
            pen_loss = self._compute_pen_loss(phi_positive)
            mag_loss = self._compute_mag_loss(lambda_anchor, force_anchor)

            losses["nce_loss"] = nce_loss
            losses["pen_loss"] = pen_loss
            losses["mag_loss"] = mag_loss
            for key, val in nce_info.items():
                losses[key] = val
        else:
            losses["nce_loss"] = zero
            losses["pen_loss"] = zero
            losses["mag_loss"] = zero

        w_nce = self.algo_config.cami.lcp.loss_weight
        w_pen = self.algo_config.cami.lcp.w_pen
        w_mag = self.algo_config.cami.lcp.w_mag

        losses["action_loss"] = (
            bc_action_loss + w_nce * nce_loss + w_pen * pen_loss + w_mag * mag_loss
        )

        return losses

    def _train_step(self, losses):
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
        log = super(BC_CaMI_CaNCE, self).log_info(info)
        losses = info["losses"]

        log["Loss"] = losses["action_loss"].item()
        log["BC_Action_Loss"] = losses["bc_action_loss"].item()
        log["NCE_Loss"] = losses["nce_loss"].item()
        log["Pen_Loss"] = losses["pen_loss"].item()
        log["Mag_Loss"] = losses["mag_loss"].item()
        log["Negative_Mode"] = self._negative_mode

        if "l2_loss" in losses:
            log["L2_Loss"] = losses["l2_loss"].item()
        if "l1_loss" in losses:
            log["L1_Loss"] = losses["l1_loss"].item()
        if "cos_loss" in losses:
            log["Cosine_Loss"] = losses["cos_loss"].item()

        for key in [
            "retrieval_acc", "avg_valid_negatives", "valid_anchor_fraction",
            "collapse_ceiling", "pos_logit_mean", "neg_logit_mean",
        ]:
            if key in losses:
                log[key] = losses[key].item()

        for key in ["policy_grad_norm", "gap_encoder_grad_norm", "impulse_encoder_grad_norm"]:
            if key in info:
                log[key] = info[key]

        return log