"""v3 adapters for deterministic BC-MLP and BC-RNN.

Only these adapters know how their host exposes features. The CaMI module
operates on tensors and can be reused by other imitation-learning adapters.
Existing CaMI classes, names, checkpoints, and configs are untouched.
"""

from collections import OrderedDict
import math

import torch
from torch.nn import functional as F

from robomimic.algo import PolicyAlgo, register_algo_factory_func
from robomimic.algo.bc import BC, BC_RNN
from robomimic.CAMI.cami_v3 import CaMIV3, warmup_factor
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils


@register_algo_factory_func("bc_cami_v3")
def algo_config_to_class(algo_config):
    if (algo_config.gmm.enabled or algo_config.gaussian.enabled or algo_config.vae.enabled
            or algo_config.transformer.enabled):
        raise ValueError("v3 ships deterministic BC-MLP / BC-RNN adapters. Other policies need a feature adapter.")
    return (BCRNNCaMIV3 if algo_config.rnn.enabled else BCMLPCaMIV3), {}


class _CaMIV3Adapter:
    def _create_networks(self):
        cfg = self.algo_config
        if cfg.rnn.enabled and (cfg.rnn.open_loop or cfg.rnn.kwargs.get("bidirectional", False)):
            raise ValueError("The v3 recurrent adapter requires a causal, closed-loop policy")
        forbidden = set(cfg.cami.continuous.privileged_obs_keys)
        keys = list(self.obs_shapes) + list(self.goal_shapes)
        if any(k in forbidden or k.lower().startswith("contact_")
               or any(part in k.lower() for part in ("force", "torque", "impulse")) for k in keys):
            raise ValueError("Keep privileged contact, force, torque and impulses outside policy observations")
        super()._create_networks()
        self._epoch, self._gradient_steps = 1, 0
        if not cfg.cami.enabled:
            return
        if self.global_config.train.seq_length < cfg.cami.v3.context_steps + cfg.cami.snippet_horizon + 1:
            raise ValueError("seq_length must cover context + anchor + future snippet")
        for value in (cfg.cami.state_loss_weight, cfg.cami.traj_loss_weight,
                      cfg.cami.v3.prediction.loss_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError("CaMI loss weights must be finite and nonnegative")
        if (not isinstance(cfg.cami.v3.gradient_diagnostics_every, int)
                or cfg.cami.v3.gradient_diagnostics_every < 0):
            raise ValueError("gradient_diagnostics_every must be a nonnegative integer")
        c, p = cfg.cami.continuous, cfg.cami.v3.prediction
        names = list(c.feature_names)
        if any(name not in names for name in p.features):
            raise ValueError("Prediction features must occur in continuous.feature_names")
        feature_dim = (cfg.actor_layer_dims[-1] if len(cfg.actor_layer_dims)
                       else cfg.rnn.hidden_dim if cfg.rnn.enabled
                       else self.nets["policy"].nets["encoder"].output_shape()[0])
        settings = dict(
            snippet_horizon=cfg.cami.snippet_horizon,
            temperature=cfg.cami.temperature, normalize_embeddings=cfg.cami.normalize_embeddings,
            contrastive_dim=cfg.cami.contrastive_dim,
            query_proj_layers=list(cfg.cami.query_proj_layers), key_proj_layers=list(cfg.cami.key_proj_layers),
            snippet_hidden_dim=cfg.cami.snippet_hidden_dim, snippet_num_layers=cfg.cami.snippet_num_layers,
            momentum=cfg.cami.momentum, use_momentum_target=cfg.cami.use_momentum_target,
            feature_scales=[c.gap_scale, c.normal_impulse_scale, c.tangential_impulse_scale],
            feature_weights=list(c.feature_weights), contact_temperature=c.contact_temperature,
            gamma=c.gamma, huber_delta=c.huber_delta, chunk_size=c.chunk_size,
            distance_penalty=c.distance_penalty, missing_features=cfg.cami.v3.missing_features,
            gap_min=cfg.cami.v3.descriptor.gap_min, impulse_log_scale=cfg.cami.v3.descriptor.impulse_log_scale,
            prediction_enabled=p.enabled,
            prediction_indices=[names.index(name) for name in p.features],
            prediction_hidden_dims=list(p.hidden_dims), prediction_huber_delta=p.huber_delta,
        )
        self.nets["cami"] = CaMIV3(feature_dim, self.ac_dim, settings)
        self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        actions, pad = batch["actions"], batch.get("pad_mask")
        if actions.ndim != 3 or pad is None or pad.shape != (*actions.shape[:2], 1):
            raise ValueError("v3 requires actions [B,T,A] and pad_mask [B,T,1]")
        if (not torch.all((pad == 0) | (pad == 1)) or not pad[:, 0].all()
                or (pad[:, 1:] > pad[:, :-1]).any()):
            raise ValueError("v3 requires nonempty, right-padded binary masks")
        result = dict(actions=actions, pad_mask=pad, goal_obs=batch.get("goal_obs"),
                      obs={key: batch["obs"][key] for key in self.obs_shapes})
        if self.algo_config.cami.enabled:
            h = self.algo_config.cami.snippet_horizon
            for key in ("contact_state", "contact_mask"):
                if key not in batch or batch[key].shape != (*actions.shape[:2], h, 3):
                    raise ValueError(key + " must be [B,T,H,3]; use the prepared physics dataset")
                result[key] = batch[key]
            if not torch.all((result["contact_mask"] == 0) | (result["contact_mask"] == 1)):
                raise ValueError("contact_mask must be binary")
        return TensorUtils.to_float(TensorUtils.to_device(result, self.device))

    def _forward_training(self, batch):
        actions, features = self._policy_training_outputs(batch["obs"], batch["goal_obs"])
        output = OrderedDict(actions=actions)
        if self.algo_config.cami.enabled:
            anchor, h = self.algo_config.cami.v3.context_steps, self.algo_config.cami.snippet_horizon
            if actions.shape[1] < anchor + h + 1:
                raise ValueError("Batch sequence is too short for context, anchor and snippet")
            future_valid = batch["pad_mask"][:, anchor + 1:anchor + h + 1, 0].bool()
            # A real future action implies a real causal context under right padding.
            lengths = future_valid.sum(-1)
            mask = batch["contact_mask"][:, anchor].bool()
            if (mask & ~future_valid[..., None]).any():
                raise ValueError("Contact labels mark padded future actions valid")
            losses, stats = self.nets["cami"](
                features[:, anchor], batch["actions"][:, anchor + 1:anchor + h + 1],
                lengths, batch["contact_state"][:, anchor], mask)
            output["cami_losses"], output["cami_stats"] = losses, stats
        return output

    def _compute_losses(self, predictions, batch):
        valid = batch["pad_mask"].squeeze(-1).bool()
        actions, target = predictions["actions"][valid], batch["actions"][valid]
        losses = OrderedDict(l2_loss=F.mse_loss(actions, target),
                             l1_loss=F.smooth_l1_loss(actions, target),
                             cos_loss=(1 - F.cosine_similarity(actions[..., :3], target[..., :3], dim=-1)).mean())
        losses["bc_action_loss"] = sum(self.algo_config.loss[k + "_weight"] * losses[k + "_loss"]
                                        for k in ("l2", "l1", "cos"))
        zero = predictions["actions"].sum() * 0.
        for name in ("state", "traj", "prediction"):
            losses[name + "_cami_loss"] = predictions.get("cami_losses", {}).get(name + "_loss", zero)
        cfg = self.algo_config.cami
        factor = warmup_factor(self._epoch, cfg.v3.warmup_epochs)
        state_weight = factor * cfg.state_loss_weight if cfg.enabled else 0.
        traj_weight = cfg.traj_loss_weight if cfg.enabled else 0.
        pred_weight = factor * cfg.v3.prediction.loss_weight if cfg.enabled and cfg.v3.prediction.enabled else 0.
        for name, value in (("state_weight", state_weight), ("traj_weight", traj_weight),
                            ("prediction_weight", pred_weight), ("warmup_factor", factor)):
            losses[name] = zero.detach().new_tensor(value)
        losses["action_loss"] = (losses["bc_action_loss"]
            + state_weight * losses["state_cami_loss"]
            + traj_weight * losses["traj_cami_loss"]
            + pred_weight * losses["prediction_cami_loss"])
        return losses

    def _gradient_diagnostics(self, losses):
        """Measure separate policy gradients before total-loss backward/clipping."""
        params = [p for p in self.nets["policy"].parameters() if p.requires_grad]
        gradients = {}
        names = ["bc_action", "state_cami"]
        if self.algo_config.cami.v3.prediction.enabled:
            names.append("prediction_cami")
        for name in names:
            gradients[name] = torch.autograd.grad(losses[name + "_loss"], params,
                                                  retain_graph=True, allow_unused=True)
        norms = {name: sum((g.detach().square().sum() for g in grad if g is not None),
                           losses["action_loss"].new_zeros(())).sqrt()
                 for name, grad in gradients.items()}
        info = {"gradient/BC_Policy_Norm": norms["bc_action"].item()}
        for name, weight in (("state_cami", "state_weight"), ("prediction_cami", "prediction_weight")):
            if name not in gradients:
                continue
            dot = sum((a.detach().mul(b.detach()).sum()
                       for a, b in zip(gradients["bc_action"], gradients[name])
                       if a is not None and b is not None), losses["action_loss"].new_zeros(()))
            label = "State" if name == "state_cami" else "Prediction"
            info["gradient/" + label + "_Policy_Norm"] = norms[name].item()
            info["gradient/" + label + "_BC_Cosine"] = (dot / (norms[name] * norms["bc_action"]).clamp_min(1e-12)).item()
            info["gradient/Weighted_" + label + "_to_BC_Ratio"] = (
                losses[weight] * norms[name] / norms["bc_action"].clamp_min(1e-12)).item()
        return info

    def train_on_batch(self, batch, epoch, validate=False):
        self._epoch = epoch
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = PolicyAlgo.train_on_batch(self, batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)
            info["losses"] = TensorUtils.detach(losses)
            info["cami_stats"] = TensorUtils.detach(predictions.get("cami_stats", {}))
            if not validate:
                interval = self.algo_config.cami.v3.gradient_diagnostics_every
                if self.algo_config.cami.enabled and interval and self._gradient_steps % interval == 0:
                    info.update(self._gradient_diagnostics(losses))
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                losses["action_loss"].backward()
                for name, optimizer in self.optimizers.items():
                    limit = self.global_config.train.max_grad_norm
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.nets[name].parameters(), float("inf") if limit is None else limit,
                        error_if_nonfinite=True)
                    info[name + "_grad_norm"] = float(norm)
                    optimizer.step()
                if self.algo_config.cami.enabled:
                    self.nets["cami"].update_targets()
                self._gradient_steps += 1
        return info

    def log_info(self, info):
        log = BC.log_info(self, info)
        for label, name in (("BC_Action_Loss", "bc_action_loss"),
                            ("State_CaMI_Loss", "state_cami_loss"),
                            ("Traj_CaMI_Loss", "traj_cami_loss"),
                            ("Prediction_CaMI_Loss", "prediction_cami_loss")):
            log[label] = info["losses"][name].item()
        for name in ("state_weight", "traj_weight", "prediction_weight", "warmup_factor"):
            log["CaMI_V3/" + name] = info["losses"][name].item()
        log.update(("CaMI_V3/" + name, value.item()) for name, value in info["cami_stats"].items())
        log.update(("CaMI_V3/" + name, value) for name, value in info.items()
                   if name.startswith("gradient/") or name.endswith("_grad_norm"))
        return log

    def serialize(self):
        model = super().serialize()
        model["cami_v3_gradient_steps"] = self._gradient_steps
        return model

    def deserialize(self, model_dict, load_optimizers=False):
        super().deserialize(model_dict, load_optimizers=load_optimizers)
        self._gradient_steps = int(model_dict.get("cami_v3_gradient_steps", 0))


class BCRNNCaMIV3(_CaMIV3Adapter, BC_RNN):
    def _policy_training_outputs(self, obs, goal):
        return self.nets["policy"].forward_with_features(obs=obs, goal=goal)


class BCMLPCaMIV3(_CaMIV3Adapter, BC):
    def _policy_training_outputs(self, obs, goal):
        b, t = next(iter(obs.values())).shape[:2]
        flat_obs = {k: value.reshape(b * t, *value.shape[2:]) for k, value in obs.items()}
        flat_goal = None
        if goal:
            flat_goal = {k: value[:, None].expand(b, t, *value.shape[1:]).reshape(b * t, *value.shape[1:])
                         for k, value in goal.items()}
        actions, features = self.nets["policy"].forward_with_features(flat_obs, flat_goal)
        return actions.reshape(b, t, -1), features.reshape(b, t, -1)
