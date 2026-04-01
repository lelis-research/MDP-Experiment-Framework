"""
Feature extraction from VQOptionCritic rollout records.

Each record has the shape:
    {
        "option_id":    int,
        "observations": {
            "onehot_image":     np.ndarray (T, H, W, C)  float16
            "onehot_direction": np.ndarray (T, D)          float16
            "onehot_carrying":  np.ndarray (T, K)          float16
        },
        "actions": np.ndarray (T-1,) int16
    }

Each feature function returns a features_dict: {key: tensor} whose shapes
can be passed directly to prepare_network_config as input_dims.

Available features
------------------
sf               : forward-discounted sum   gamma^t      * obs[t]
delta_sf         : forward-discounted sum   gamma^t      * (obs[t] - obs[0])
reverse_sf       : reverse-discounted sum   gamma^{T-1-t} * obs[t]
delta_reverse_sf : reverse-discounted sum   gamma^{T-1-t} * (obs[t] - obs[0])
last             : obs[-1]
delta_last       : obs[-1] - obs[0]

*_enc variants   : same operations performed in encoder latent space.
                   encoder(**obs_batch) -> (T, enc_dim) tensor.
                   Output features_dict has a single key "x".
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers — raw observation space
# ---------------------------------------------------------------------------

def _apply_weights(obs: dict, w1d: np.ndarray) -> dict:
    """Weighted sum of all obs keys. w1d shape: (T,), already normalised."""
    img   = obs["onehot_image"].astype(np.float32)    # (T, H, W, C)
    dir_  = obs["onehot_direction"].astype(np.float32) # (T, D)
    carry = obs["onehot_carrying"].astype(np.float32)  # (T, K)
    return {
        "onehot_image":     torch.from_numpy((w1d[:, None, None, None] * img  ).sum(0)),
        "onehot_direction": torch.from_numpy((w1d[:, None]             * dir_ ).sum(0)),
        "onehot_carrying":  torch.from_numpy((w1d[:, None]             * carry).sum(0)),
    }


def _apply_weights_delta(obs: dict, w1d: np.ndarray) -> dict:
    """Weighted sum of (obs[t] - obs[0]) for all obs keys."""
    img   = obs["onehot_image"].astype(np.float32)
    dir_  = obs["onehot_direction"].astype(np.float32)
    carry = obs["onehot_carrying"].astype(np.float32)
    img   = img   - img[0][None, ...]
    dir_  = dir_  - dir_[0][None, :]
    carry = carry - carry[0][None, :]
    return {
        "onehot_image":     torch.from_numpy((w1d[:, None, None, None] * img  ).sum(0)),
        "onehot_direction": torch.from_numpy((w1d[:, None]             * dir_ ).sum(0)),
        "onehot_carrying":  torch.from_numpy((w1d[:, None]             * carry).sum(0)),
    }


# ---------------------------------------------------------------------------
# Helpers — encoder latent space
# ---------------------------------------------------------------------------

def _encode_batch(obs: dict, encoder) -> torch.Tensor:
    """Run all T timesteps through encoder in one batch. Returns (T, enc_dim)."""
    batch = {k: torch.from_numpy(v.astype(np.float32)) for k, v in obs.items()}
    with torch.no_grad():
        return encoder(**batch)  # (T, enc_dim)


def _apply_weights_enc(obs: dict, w1d: np.ndarray, encoder) -> dict:
    """Weighted sum in latent space. Returns {"x": (enc_dim,)}."""
    z = _encode_batch(obs, encoder)                      # (T, enc_dim)
    w = torch.from_numpy(w1d).unsqueeze(1)               # (T, 1)
    return {"x": (w * z).sum(0)}


def _apply_weights_delta_enc(obs: dict, w1d: np.ndarray, encoder) -> dict:
    """Weighted sum of (z[t] - z[0]) in latent space. Returns {"x": (enc_dim,)}."""
    z = _encode_batch(obs, encoder)                      # (T, enc_dim)
    z = z - z[0:1]                                       # delta in latent space
    w = torch.from_numpy(w1d).unsqueeze(1)               # (T, 1)
    return {"x": (w * z).sum(0)}


# ---------------------------------------------------------------------------
# Feature functions — raw observation space
# ---------------------------------------------------------------------------

def sf(record: dict, gamma: float = 0.99) -> dict:
    """features_dict: forward-discounted sum  gamma^t * obs[t]."""
    obs = record["observations"]
    T   = obs["onehot_image"].shape[0]
    t   = np.arange(T, dtype=np.float32)
    w   = gamma ** t
    w  /= w.sum() + 1e-8
    return _apply_weights(obs, w)


def delta_sf(record: dict, gamma: float = 0.99) -> dict:
    """features_dict: forward-discounted sum  gamma^t * (obs[t] - obs[0])."""
    obs = record["observations"]
    T   = obs["onehot_image"].shape[0]
    t   = np.arange(T, dtype=np.float32)
    w   = gamma ** t
    w  /= w.sum() + 1e-8
    return _apply_weights_delta(obs, w)


def reverse_sf(record: dict, gamma: float = 0.99) -> dict:
    """features_dict: reverse-discounted sum  gamma^{T-1-t} * obs[t]."""
    obs = record["observations"]
    T   = obs["onehot_image"].shape[0]
    t   = np.arange(T, dtype=np.float32)
    w   = gamma ** (T - 1 - t)
    w  /= w.sum() + 1e-8
    return _apply_weights(obs, w)


def delta_reverse_sf(record: dict, gamma: float = 0.99) -> dict:
    """features_dict: reverse-discounted sum  gamma^{T-1-t} * (obs[t] - obs[0])."""
    obs = record["observations"]
    T   = obs["onehot_image"].shape[0]
    t   = np.arange(T, dtype=np.float32)
    w   = gamma ** (T - 1 - t)
    w  /= w.sum() + 1e-8
    return _apply_weights_delta(obs, w)


def last(record: dict, gamma: float = 0.99) -> dict:
    """features_dict: last observation obs[-1]."""
    obs = record["observations"]
    return {
        "onehot_image":     torch.from_numpy(obs["onehot_image"][-1].astype(np.float32)),
        "onehot_direction": torch.from_numpy(obs["onehot_direction"][-1].astype(np.float32)),
        "onehot_carrying":  torch.from_numpy(obs["onehot_carrying"][-1].astype(np.float32)),
    }


def delta_last(record: dict, gamma: float = 0.99) -> dict:
    """features_dict: (obs[-1] - obs[0]) per key."""
    obs = record["observations"]
    return {
        "onehot_image":     torch.from_numpy((obs["onehot_image"][-1]     - obs["onehot_image"][0]    ).astype(np.float32)),
        "onehot_direction": torch.from_numpy((obs["onehot_direction"][-1] - obs["onehot_direction"][0]).astype(np.float32)),
        "onehot_carrying":  torch.from_numpy((obs["onehot_carrying"][-1]  - obs["onehot_carrying"][0] ).astype(np.float32)),
    }


# ---------------------------------------------------------------------------
# Feature functions — encoder latent space
# ---------------------------------------------------------------------------

def sf_enc(record: dict, gamma: float = 0.99, encoder=None) -> dict:
    """features_dict: forward-discounted sum  gamma^t * z[t]  in latent space."""
    obs = record["observations"]
    T   = obs["onehot_image"].shape[0]
    t   = np.arange(T, dtype=np.float32)
    w   = gamma ** t
    w  /= w.sum() + 1e-8
    return _apply_weights_enc(obs, w, encoder)


def delta_sf_enc(record: dict, gamma: float = 0.99, encoder=None) -> dict:
    """features_dict: forward-discounted sum  gamma^t * (z[t] - z[0])  in latent space."""
    obs = record["observations"]
    T   = obs["onehot_image"].shape[0]
    t   = np.arange(T, dtype=np.float32)
    w   = gamma ** t
    w  /= w.sum() + 1e-8
    return _apply_weights_delta_enc(obs, w, encoder)


def reverse_sf_enc(record: dict, gamma: float = 0.99, encoder=None) -> dict:
    """features_dict: reverse-discounted sum  gamma^{T-1-t} * z[t]  in latent space."""
    obs = record["observations"]
    T   = obs["onehot_image"].shape[0]
    t   = np.arange(T, dtype=np.float32)
    w   = gamma ** (T - 1 - t)
    w  /= w.sum() + 1e-8
    return _apply_weights_enc(obs, w, encoder)


def delta_reverse_sf_enc(record: dict, gamma: float = 0.99, encoder=None) -> dict:
    """features_dict: reverse-discounted sum  gamma^{T-1-t} * (z[t] - z[0])  in latent space."""
    obs = record["observations"]
    T   = obs["onehot_image"].shape[0]
    t   = np.arange(T, dtype=np.float32)
    w   = gamma ** (T - 1 - t)
    w  /= w.sum() + 1e-8
    return _apply_weights_delta_enc(obs, w, encoder)


def last_enc(record: dict, gamma: float = 0.99, encoder=None) -> dict:
    """features_dict: z[-1]  in latent space."""
    obs   = record["observations"]
    batch = {k: torch.from_numpy(v[-1:].astype(np.float32)) for k, v in obs.items()}
    with torch.no_grad():
        z = encoder(**batch)  # (1, enc_dim)
    return {"x": z.squeeze(0)}


def delta_last_enc(record: dict, gamma: float = 0.99, encoder=None) -> dict:
    """features_dict: z[-1] - z[0]  in latent space."""
    obs   = record["observations"]
    first = {k: torch.from_numpy(v[0:1].astype(np.float32)) for k, v in obs.items()}
    last_ = {k: torch.from_numpy(v[-1:].astype(np.float32)) for k, v in obs.items()}
    with torch.no_grad():
        z0 = encoder(**first)  # (1, enc_dim)
        z1 = encoder(**last_)  # (1, enc_dim)
    return {"x": (z1 - z0).squeeze(0)}


# ---------------------------------------------------------------------------

FEATURE_FN_DICT = {
    "sf":                    sf,
    "delta_sf":              delta_sf,
    "reverse_sf":            reverse_sf,
    "delta_reverse_sf":      delta_reverse_sf,
    "last":                  last,
    "delta_last":            delta_last,
    # encoder latent-space variants
    "sf_enc":                sf_enc,
    "delta_sf_enc":          delta_sf_enc,
    "reverse_sf_enc":        reverse_sf_enc,
    "delta_reverse_sf_enc":  delta_reverse_sf_enc,
    "last_enc":              last_enc,
    "delta_last_enc":        delta_last_enc,
}
