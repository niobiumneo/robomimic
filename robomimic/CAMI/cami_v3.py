"""Policy-independent CaMI objectives.

The host supplies causal features [B,D], demonstrated future actions [B,H,A],
physical labels [B,H,F], and masks. No BC, recurrence, action distribution, or
policy sampling API is assumed. The host must keep features in its autograd
graph. Only labels and momentum keys are detached.
"""

from collections import OrderedDict
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from robomimic.CAMI.continuous_contact import (
    contact_distance, embedding_logits, negative_weights, weighted_infonce,
)


def _mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


def warmup_factor(epoch, warmup_epochs):
    """Epoch 1 has zero state pressure; epoch N+1 reaches the requested weight."""
    if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
        raise ValueError("warmup_epochs must be a nonnegative integer")
    return 1. if warmup_epochs == 0 else min(1., max(0., (epoch - 1) / warmup_epochs))


class CaMIV3(nn.Module):
    """Reusable contrastive module with an optional future-contact predictor.

    An LSTM encodes expert action snippets internally. This does not constrain
    the host policy architecture. The optional head predicts normalized future
    descriptors, conditioned on the demonstrated action snippet. Its action
    embedding is detached so its gradients shape the policy features and head,
    without changing the snippet objective. It is unused during action inference.
    """

    def __init__(self, feature_dim, action_dim, settings):
        super().__init__()
        self.settings = deepcopy(dict(settings))
        s = self.settings
        self.horizon = s["snippet_horizon"]
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("snippet_horizon must be a positive integer")
        for name in ("temperature", "contact_temperature", "gamma", "huber_delta"):
            if not math.isfinite(s[name]) or s[name] <= 0:
                raise ValueError(name + " must be finite and positive")
        scales = torch.as_tensor(s["feature_scales"], dtype=torch.float32)
        weights = torch.as_tensor(s["feature_weights"], dtype=torch.float32)
        if (scales.ndim != 1 or not len(scales) or not torch.isfinite(scales).all()
                or (scales <= 0).any()):
            raise ValueError("feature_scales must be fitted, finite and positive")
        if (weights.shape != scales.shape or not torch.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("feature_weights must match scales and be nonnegative, not all zero")
        if not 0 <= s["momentum"] <= 1:
            raise ValueError("momentum must lie in [0,1]")
        if not math.isfinite(s["prediction_huber_delta"]) or s["prediction_huber_delta"] <= 0:
            raise ValueError("prediction_huber_delta must be finite and positive")
        self.register_buffer("feature_scales", scales)
        self.feature_dim, self.action_dim = feature_dim, action_dim
        dim = s["contrastive_dim"]
        self.state_projector = _mlp(feature_dim, s["query_proj_layers"], dim)
        self.snippet_encoder = nn.LSTM(action_dim, s["snippet_hidden_dim"],
                                       s["snippet_num_layers"], batch_first=True)
        self.key_projector = _mlp(s["snippet_hidden_dim"], s["key_proj_layers"], dim)
        self.snippet_target = deepcopy(self.snippet_encoder).requires_grad_(False)
        self.key_target = deepcopy(self.key_projector).requires_grad_(False)
        self.prediction_indices = tuple(s["prediction_indices"])
        self.prediction_head = None
        if s["prediction_enabled"]:
            if (not self.prediction_indices or len(set(self.prediction_indices)) != len(self.prediction_indices)
                    or any(i < 0 or i >= len(scales) for i in self.prediction_indices)):
                raise ValueError("prediction_indices must select distinct descriptor channels")
            self.prediction_head = _mlp(feature_dim + dim, s["prediction_hidden_dims"],
                                        self.horizon * len(self.prediction_indices))

    def _embed_actions(self, actions, lengths, target=False):
        packed = pack_padded_sequence(actions, lengths.clamp_min(1).long().cpu(),
                                      batch_first=True, enforce_sorted=False)
        encoder = self.snippet_target if target else self.snippet_encoder
        projector = self.key_target if target else self.key_projector
        _, (hidden, _) = encoder(packed)
        return projector(hidden[-1])

    def forward(self, features, expert_actions, lengths, contact, contact_mask):
        b = len(features)
        f = len(self.feature_scales)
        if not b or features.shape != (b, self.feature_dim) or not torch.isfinite(features).all():
            raise ValueError("features must be finite [B,feature_dim]")
        if expert_actions.shape != (b, self.horizon, self.action_dim):
            raise ValueError("expert_actions must be [B,H,action_dim]")
        if (lengths.shape != (b,) or not torch.isfinite(lengths).all()
                or (lengths != lengths.long()).any() or (lengths < 0).any()
                or (lengths > self.horizon).any()):
            raise ValueError("lengths must be integer counts in [0,H]")
        if contact.shape != (b, self.horizon, f) or contact_mask.shape != contact.shape:
            raise ValueError("contact and mask must be [B,H,F]")
        if not torch.all((contact_mask == 0) | (contact_mask == 1)):
            raise ValueError("contact_mask must be binary")
        action_valid = torch.arange(self.horizon, device=features.device)[None] < lengths[:, None]
        mask = contact_mask.bool() & action_valid[..., None]
        if not torch.isfinite(contact[mask]).all() or not torch.isfinite(expert_actions[action_valid]).all():
            raise ValueError("Nonfinite valid actions or descriptors")
        actions = expert_actions.masked_fill(~action_valid[..., None], 0.)
        normalized = torch.where(mask, contact.detach(), 0.) / self.feature_scales
        state_query = self.state_projector(features)
        traj_query = self._embed_actions(actions, lengths)
        with torch.no_grad():
            key = self._embed_actions(actions, lengths, target=True)
            distance, pair_valid = contact_distance(
                normalized, mask, self.settings["feature_weights"],
                self.settings["huber_delta"], self.settings["chunk_size"],
                penalty=self.settings["distance_penalty"])
            pair_valid &= (lengths > 0)[:, None] & (lengths > 0)[None]
            weights = negative_weights(distance, pair_valid, self.settings["contact_temperature"],
                                       self.settings["gamma"])

        losses, stats = OrderedDict(), OrderedDict()
        for name, query in (("state", state_query), ("traj", traj_query)):
            logits = embedding_logits(query, key, self.settings["temperature"],
                                       self.settings["normalize_embeddings"])
            losses[name + "_loss"], diagnostics = weighted_infonce(logits, weights)
            stats.update((name + "/" + k, v) for k, v in diagnostics.items())

        losses["prediction_loss"] = features.sum() * 0.
        if self.prediction_head is not None:
            predicted = self.prediction_head(torch.cat((features, traj_query.detach()), -1))
            predicted = predicted.reshape(b, self.horizon, len(self.prediction_indices))
            target = normalized[..., self.prediction_indices]
            valid = mask[..., self.prediction_indices]
            # A masked mean, including the all-missing case, avoids padding targets.
            error = F.huber_loss(predicted, target, reduction="none",
                                  delta=self.settings["prediction_huber_delta"])
            losses["prediction_loss"] = error.masked_fill(~valid, 0.).sum() / valid.sum().clamp_min(1)
            stats["prediction/valid_fraction"] = valid.float().mean()

        with torch.no_grad():
            eligible = pair_valid & ~torch.eye(b, dtype=torch.bool, device=features.device)
            for name, values in (("weight", weights[eligible]), ("distance", distance[eligible])):
                quantiles = torch.quantile(values, values.new_tensor([.1, .5, .9])) if len(values) else features.new_zeros(3)
                for label, value in zip(("p10", "p50", "p90"), quantiles):
                    stats[name + "/" + label] = value
            stats["weight/near_zero_fraction"] = (weights[eligible] < .01).float().mean() if eligible.any() else features.new_zeros(())
            stats["weight/near_one_fraction"] = (weights[eligible] > .99).float().mean() if eligible.any() else features.new_zeros(())
            stats["eligible_anchor_fraction"] = (lengths > 0).float().mean()
            stats["contrastive_batch_size"] = features.new_tensor(b)
        return losses, stats

    @torch.no_grad()
    def update_targets(self):
        momentum = self.settings["momentum"] if self.settings["use_momentum_target"] else 0.
        for online, target in ((self.snippet_encoder, self.snippet_target),
                               (self.key_projector, self.key_target)):
            for source, dest in zip(online.parameters(), target.parameters()):
                dest.mul_(momentum).add_(source, alpha=1. - momentum)
