"""CaMI-C mechanics and loss, with physical labels detached from autograd.

D_ij = mean_h sum_f w_f rho(c_ihf / scale_f - c_jhf / scale_f)
W_ij = (1 - exp(-D_ij / tau_c)) ** gamma, i != j
L = mean_i log(1 + sum_{j != i} W_ij exp(s_ij - s_ii))

rho(x) = abs(x) by default (L1). Huber is an explicit optional ablation,
not a requirement of ContactNets. B is the number of anchors in the minibatch,
H the future horizon, and F the number of descriptor features.

Only jointly observed, complete timesteps enter a pair's horizon average.
Disabled features (weight zero) do not affect timestep validity. Zero-negative
rows contribute zero to the batch mean, exactly as in the supplied formulation.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

SCHEMA_VERSION = 2
FEATURE_SETS = {"force": ["force"], "physics": ["gap", "normal_impulse", "tangential_impulse_magnitude"]}


def contact_observation_keys(metadata):
    """Resolve explicitly recorded privileged observation source names."""
    result = set()
    for demo in metadata.get("demos", {}).values():
        sources = demo.get("source", [])
        sources = [sources] if isinstance(sources, str) else sources
        for name in sources:
            if isinstance(name, str) and name.startswith("obs/"):
                result.add(name[4:])
    return result


def future_windows(values, horizon, valid=None):
    """Create [T,H,F] labels for t+1,...,t+H, without crossing demos."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if (values.ndim != 2 or not len(values) or values.shape[1] < 1
            or not isinstance(horizon, int) or horizon < 1):
        raise ValueError("Expected nonempty [T,F] values and a positive integer horizon")
    valid = np.ones_like(values, dtype=bool) if valid is None else np.asarray(valid)
    if valid.shape != values.shape or not np.all((valid == 0) | (valid == 1)):
        raise ValueError("Validity must be binary and match values")
    valid = valid.astype(bool)
    if not np.isfinite(values[valid]).all():
        raise ValueError("Valid contact values must be finite")
    values = np.where(valid, values, 0.)
    indices = np.arange(len(values))[:, None] + np.arange(1, horizon + 1)[None]
    clipped = np.minimum(indices, len(values) - 1)
    return values[clipped], (valid[clipped] & (indices < len(values))[..., None]).astype(np.float32)


def force_magnitude(wrench):
    """Nonnegative scalar or norm of Fx,Fy,Fz in N. Never mix in torque."""
    wrench = np.asarray(wrench, dtype=np.float32)
    if wrench.ndim == 1:
        result = wrench[:, None]
    elif wrench.ndim == 2 and wrench.shape[1] == 1:
        result = wrench
    elif wrench.ndim == 2 and wrench.shape[1] in (3, 6):
        result = np.linalg.norm(wrench[:, :3], axis=1, keepdims=True)
    else:
        raise ValueError("Force source must be [T], [T,1], [T,3], or [T,6]")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Force magnitudes must be finite and nonnegative")
    return result


def normalize_contact(contact, feature_names, gap_scale=.02, force_scale=50., normal_impulse_scale=2.5, tangential_impulse_scale=2.5):
    """Divide SI quantities by training std + epsilon; do not clip large forces.

    Scales make metres and N*s comparable without clipping their differences.
    They must be fixed before validation/test, never batch-fitted.
    Mask unavailable values to zero before calling this function.
    """
    scales = {"gap": gap_scale, "force": force_scale, "normal_impulse": normal_impulse_scale, "tangential_impulse_magnitude": tangential_impulse_scale}
    names = list(feature_names)
    if names not in list(FEATURE_SETS.values()):
        raise ValueError("Use [force] or [gap, normal_impulse, tangential_impulse_magnitude] in that order")
    if any(not math.isfinite(x) or x <= 0 for x in scales.values()):
        raise ValueError("All contact scales must be finite and positive")
    if contact.ndim != 3 or contact.shape[-1] != len(names):
        raise ValueError("contact must be [B,H,len(feature_names)]")
    c = contact.detach().float()
    if not torch.isfinite(c).all():
        raise ValueError("Mask unavailable descriptors to zero before normalization")
    for i, name in enumerate(names):
        if name != "gap" and (c[..., i] < 0).any():
            raise ValueError("Force and impulse magnitudes must be nonnegative")
    return c / c.new_tensor([scales[name] for name in names])


@torch.no_grad()
def contact_distance(normalized, mask, feature_weights, delta=.1, chunk_size=64, *, penalty="l1"):
    """Return [B,B] distances and validity; sum features, average time.

    With no missing data this is exactly (1/H) sum_h sum_f w_f rho(diff).
    With missing data, exclude timesteps missing any enabled feature from either
    sample and divide by the remaining timestep count. No overlap => invalid.
    rho(x)=abs(x) for L1; Huber uses 0.5*x**2 below delta and
    delta*(abs(x)-0.5*delta) above it. delta is ignored for L1.
    """
    if penalty not in ("l1", "huber"):
        raise ValueError("penalty must be l1 or huber")
    if penalty == "huber" and (not math.isfinite(delta) or delta <= 0):
        raise ValueError("Huber delta must be finite and positive")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("integer chunk_size must be positive")
    c = normalized.detach().float()
    if c.ndim != 3 or min(c.shape) < 1 or mask.shape != c.shape:
        raise ValueError("Contact and mask must be matching nonempty [B,H,F]")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Contact mask must be binary")
    mask = mask.detach().to(device=c.device, dtype=torch.bool)
    w = c.new_tensor(feature_weights)
    if w.shape != (c.shape[-1],) or not torch.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Feature weights must be finite, nonnegative, not all zero, and match F")
    if not torch.isfinite(c[mask]).all():
        raise ValueError("Nonfinite valid contact labels")
    c = torch.where(mask, c, 0.)
    complete = (mask | (w == 0)).all(-1)
    distances, validity = [], []
    for start in range(0, len(c), chunk_size):
        diff = (c[start:start + chunk_size, None] - c[None]).abs()
        rho = diff if penalty == "l1" else torch.where(
            diff <= delta, .5 * diff.square(), delta * (diff - .5 * delta))
        common = complete[start:start + chunk_size, None] & complete[None]
        count = common.sum(-1)
        per_time = (rho * w).sum(-1)
        distances.append(per_time.masked_fill(~common, 0.).sum(-1) / count.clamp_min(1))
        validity.append(count > 0)
    return torch.cat(distances), torch.cat(validity)


@torch.no_grad()
def negative_weights(distance, pair_valid, contact_temperature=.05, gamma=1.):
    """Stable evaluation of (1-exp(-D/tau_c))**gamma, with exact zero weights."""
    if not math.isfinite(contact_temperature) or contact_temperature <= 0:
        raise ValueError("contact_temperature must be finite and positive")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if (distance.ndim != 2 or distance.shape[0] != distance.shape[1]
            or pair_valid.shape != distance.shape or not torch.isfinite(distance).all()
            or (distance < 0).any()):
        raise ValueError("Distance must be finite, nonnegative, square, and match pair_valid")
    w = (-torch.expm1(-distance.detach() / contact_temperature)).clamp(0, 1).pow(gamma)
    eye = torch.eye(len(w), dtype=torch.bool, device=w.device)
    return w.masked_fill(~pair_valid.bool() | eye, 0.)


def embedding_logits(query, key, temperature=.07, normalize=True):
    """Query receives gradients; momentum keys do not."""
    if (not math.isfinite(temperature) or temperature <= 0 or query.ndim != 2
            or query.shape != key.shape or min(query.shape) < 1):
        raise ValueError("Expected matching nonempty [B,D] embeddings and positive temperature")
    q, k = query.float(), key.detach().float()
    if normalize:
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    return q @ k.T / temperature


def weighted_infonce(logits, weights):
    """Paired InfoNCE averaged over the whole batch, including zero-loss rows.

    For a negative j, dL_i/ds_ij = W_ij exp(s_ij) / denominator_i.
    This reweights competition; it does not force similar-contact pairs close
    or prove an InfoNCE mutual-information bound for this modified objective.
    """
    if (logits.ndim != 2 or logits.shape[0] != logits.shape[1]
            or len(logits) == 0 or weights.shape != logits.shape):
        raise ValueError("logits and weights must be matching nonempty square matrices")
    if not torch.isfinite(logits).all():
        raise ValueError("Embedding logits must be finite")
    # Float32 for low precision inputs, retain double for numerical tests.
    scores = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    w = weights.detach().to(scores)
    if not torch.isfinite(w).all() or (w < 0).any() or (w > 1).any():
        raise ValueError("Negative weights must be finite and in [0,1]")
    eye = torch.eye(len(w), device=w.device, dtype=torch.bool)
    w = w.masked_fill(eye, 0.)
    nonzero = w > 0
    active = nonzero.any(1)
    logw = w.masked_fill(~nonzero, 1.).log().masked_fill(~nonzero, -torch.inf)
    relative = scores - scores.diagonal()[:, None]
    competition = (relative + logw).masked_fill(~nonzero, -torch.inf)
    # Including zero (the positive relative logit) makes the all-zero row exact
    # and avoids overflow in exp even with large unnormalized input logits.
    per_anchor = torch.logsumexp(torch.cat([scores.new_zeros(len(w), 1), competition], 1), 1)
    # Divide by B, including anchors with no eligible negatives (zero loss).
    # The diagonal s_ii is each anchor's own future-action positive key.
    loss = per_anchor.mean()
    with torch.no_grad():
        zero = scores.new_zeros(())
        eligible = (torch.eye(len(w), device=w.device, dtype=torch.bool) | nonzero.any(0)[None])
        info = {
            "valid_anchor_fraction": active.float().mean(),
            "negative_weight_mean": w[~eye].mean() if len(w) > 1 else zero,
            "effective_negatives": w.sum(1).mean(),
            "pos_logit_mean": scores.diagonal().mean(),
            "retrieval_acc": ((competition[active].max(1).values < 0).float().mean() if active.any() else zero),
            "unweighted_retrieval_acc": ((scores.masked_fill(~eligible, -torch.inf).argmax(1)[active]
                == torch.arange(len(w), device=w.device)[active]).float().mean() if active.any() else zero),
        }
    return loss, info


@torch.no_grad()
def contact_summary(normalized, mask):
    """Min, mean and max for each channel, used only in offline analysis."""
    valid = mask.bool()
    observed = valid.any(1)
    mean = normalized.masked_fill(~valid, 0.).sum(1) / valid.sum(1).clamp_min(1)
    low = normalized.masked_fill(~valid, torch.inf).min(1).values
    high = normalized.masked_fill(~valid, -torch.inf).max(1).values
    return torch.cat([torch.where(observed, x, 0.) for x in (low, mean, high)], -1), observed.repeat(1, 3)
