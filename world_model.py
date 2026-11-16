
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


OBS_NAMES = [
    "SOC_e",
    "SOC_NH3",
    "SOC_HNO3",
    "SOC_NH4NO3",
    "SOC_F",
    "price_el",
    "load_el",
    "load_th",
    "wt_el",
    "pvt_el",
    "pvt_th",
    "q_na_ref",
    "q_an_ref",
    "q_f_ref",
]

# Key inventory state indices in the observation vector.
KEY_STATE_IDXS = [0, 1, 2, 3, 4]
KEY_STATE_NAMES = [OBS_NAMES[i] for i in KEY_STATE_IDXS]


@dataclass
class WorldModelConfig:
    hidden_dim: int = 256
    depth: int = 3
    dropout: float = 0.02
    lr: float = 3e-4
    batch_size: int = 512
    epochs: int = 180
    val_ratio: float = 0.2
    seed: int = 42
    delta_loss_weight: float = 1.0
    reward_loss_weight: float = 0.2
    key_state_loss_weight: float = 2.5
    grad_clip: float = 1.0


class StandardScaler:
    """Small numpy/torch scaler used for world-model training."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None, eps: float = 1e-6):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.eps = eps

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float32)
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = (x.std(axis=0) + self.eps).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) * self.std + self.mean).astype(np.float32)

    def transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def inverse_transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * std + mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std, "eps": np.array(self.eps, dtype=np.float32)}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)
        self.eps = float(state.get("eps", 1e-6))


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedWorldModel(nn.Module):
    """Predicts normalized state delta and normalized reward from normalized state + raw action.

    Input shape:  obs_dim + act_dim
    Output:       delta_norm(obs_dim), reward_norm(1)
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: WorldModelConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.backbone = MLP(
            in_dim=obs_dim + act_dim,
            out_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )
        self.delta_head = nn.Linear(cfg.hidden_dim, obs_dim)
        self.reward_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, obs_norm: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_norm, action], dim=-1)
        h = self.backbone(x)
        delta_norm = self.delta_head(h)
        reward_norm = self.reward_head(h).squeeze(-1)
        return delta_norm, reward_norm


@torch.no_grad()
def predict_next_obs_and_reward(
    model: LearnedWorldModel,
    obs: np.ndarray,
    action: np.ndarray,
    obs_scaler: StandardScaler,
    delta_scaler: StandardScaler,
    reward_scaler: StandardScaler,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    """One-step prediction in original, unnormalized units."""
    model.eval()
    obs_np = np.asarray(obs, dtype=np.float32)
    action_np = np.asarray(action, dtype=np.float32)
    obs_norm_np = obs_scaler.transform(obs_np[None, :])
    obs_norm = torch.as_tensor(obs_norm_np, dtype=torch.float32, device=device)
    action_t = torch.as_tensor(action_np[None, :], dtype=torch.float32, device=device)
    delta_norm, reward_norm = model(obs_norm, action_t)
    delta = delta_scaler.inverse_transform(delta_norm.cpu().numpy())[0]
    reward = reward_scaler.inverse_transform(reward_norm.cpu().numpy().reshape(-1, 1))[0, 0]
    next_obs = obs_np + delta
    # Keep normalized SOC/inventory states in a valid range for rollout sanity.
    next_obs[:5] = np.clip(next_obs[:5], 0.0, 1.0)
    return next_obs.astype(np.float32), float(reward)
