"""
Custom implementation of BC with CaMI (Contact-Aware Mutual Information).
"""
from collections import OrderedDict
import copy

import torch
import torch.nn as nn

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func
from robomimic.algo.bc import BC

@register_algo_factory_func("bc_cami")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC_CaMI algo class.
    """
    return BC_CaMI, {}


def build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

class BC_CaMI(BC):
    """
    Behavioral Cloning with Contact-aware Mutual Information (CaMI) regularization.

    Version 1:
        - Keeps standard BC policy training behavior
        - Adds placeholders for future CaMI modules and losses
        - Adds momentum target update hook
    """

    def _create_networks(self):
        """
        Create networks and place them into @self.nets.

        For the first scaffold:
            - keep standard BC policy
            - add placeholder query / snippet modules
            - add target snippet encoder for future momentum updates
        """
        super(BC_CaMI, self)._create_networks()

        # ------------------------------------------------------------------ #
        # Placeholder CaMI modules
        # These will be replaced later with:
        #   - anchor fusion encoder
        #   - query projection head
        #   - future snippet encoder
        #   - key projection head
        # ------------------------------------------------------------------ #
        anchor_dim = self.algo_config.actor_layer_dims[-1]
        proj_hidden = list(self.algo_config.cami.query_proj_layers)
        contrastive_dim = self.algo_config.cami.contrastive_dim

        self.nets["query_proj"] = build_mlp(
            input_dim=anchor_dim,
            hidden_dims=proj_hidden,
            output_dim=contrastive_dim,
        )
        self.nets["snippet_encoder"] = nn.Identity()
        self.nets["key_proj"] = nn.Identity()

        # Momentum / target copy for future snippet encoder branch
        self.nets["snippet_encoder_target"] = copy.deepcopy(self.nets["snippet_encoder"])
        self.nets["key_proj_target"] = copy.deepcopy(self.nets["key_proj"])

        for p in self.nets["snippet_encoder_target"].parameters():
            p.requires_grad = False
        for p in self.nets["key_proj_target"].parameters():
            p.requires_grad = False

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Process dataloader batch.

        For now:
            - keep standard BC inputs
            - optionally pull in future CaMI fields if present
        """
        input_batch = super(BC_CaMI, self).process_batch_for_training(batch)

        print("obs keys:", batch["obs"].keys())
        print("force shape before processing:", batch["obs"]["force"].shape)

        # Optional CaMI fields - only included if dataset already provides them
        if "contact_label" in batch:
            input_batch["contact_label"] = TensorUtils.to_device(
                batch["contact_label"], self.device
            )

        if "pos_future_obs" in batch:
            input_batch["pos_future_obs"] = TensorUtils.to_float(
                TensorUtils.to_device(batch["pos_future_obs"], self.device)
            )

        if "neg_future_obs" in batch:
            input_batch["neg_future_obs"] = TensorUtils.to_float(
                TensorUtils.to_device(batch["neg_future_obs"], self.device)
            )

        print("force shape after processing:", input_batch["obs"]["force"].shape)
        return input_batch

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Same structure as BC, but with a hook for momentum target updates after
        optimization when not validating.
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)
                self._update_target_networks()

        return info

    def _forward_training(self, batch):
        """
        Forward pass during training.

        Version 1:
            - run normal BC action prediction
            - create placeholder CaMI outputs
        """
        predictions = OrderedDict()

        actions, anchor_feat = self.nets["policy"].forward_with_features(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        query_embedding = self.nets["query_proj"](anchor_feat)

        if getattr(self.algo_config.cami, "normalize_embeddings", False):
            query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1)

        predictions["actions"] = actions
        predictions["anchor_feat"] = anchor_feat
        predictions["query_embedding"] = query_embedding
        predictions["cami_loss"] = torch.tensor(0.0, device=self.device)

        if not hasattr(self, "_printed_anchor_debug"):
            print("[BC_CaMI DEBUG] anchor_feat shape:", tuple(anchor_feat.shape))
            print("[BC_CaMI DEBUG] query_embedding shape:", tuple(query_embedding.shape))
            print("[BC_CaMI DEBUG] actions shape:", tuple(actions.shape))
            print("[BC_CaMI DEBUG] target actions shape:", tuple(batch["actions"].shape))
            self._printed_anchor_debug = True

        return predictions
    
    def _compute_losses(self, predictions, batch):
        """
        Compute BC loss + CaMI loss.

        Version 1:
            - action_loss is exactly BC
            - cami_loss is placeholder zero
        """
        losses = OrderedDict()

        a_target = batch["actions"]
        actions = predictions["actions"]

        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)

        # cosine direction loss on eef delta position
        # keeps BC behavior aligned with original BC implementation
        import robomimic.utils.loss_utils as LossUtils
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        bc_action_loss = sum(action_losses)

        cami_loss = predictions["cami_loss"]
        losses["cami_loss"] = cami_loss

        if getattr(self.algo_config.cami, "enabled", False):
            losses["action_loss"] = bc_action_loss + self.algo_config.cami.loss_weight * cami_loss
        else:
            losses["action_loss"] = bc_action_loss

        return losses

    def _train_step(self, losses):
        """
        Backpropagation step.

        Version 1:
            - single optimizer on policy net, like BC
            - later we can split or expand when explicit CaMI modules become trainable
        """
        info = OrderedDict()

        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    @torch.no_grad()
    def _update_target_networks(self):
        """
        EMA update for target snippet encoder branch.

        Version 1:
            harmless placeholder update since modules are identities.
            Later this will update the actual future snippet encoder + key projection head.
        """
        if not getattr(self.algo_config.cami, "enabled", False):
            return
        if not getattr(self.algo_config.cami, "use_momentum_target", False):
            return

        tau = self.algo_config.cami.target_tau

        for online_param, target_param in zip(
            self.nets["snippet_encoder"].parameters(),
            self.nets["snippet_encoder_target"].parameters(),
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * online_param.data)

        for online_param, target_param in zip(
            self.nets["key_proj"].parameters(),
            self.nets["key_proj_target"].parameters(),
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * online_param.data)

    def log_info(self, info):
        """
        Process info dictionary for logging.
        """
        log = super(BC_CaMI, self).log_info(info)

        if "cami_loss" in info["losses"]:
            log["CaMI_Loss"] = info["losses"]["cami_loss"].item()

        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)
