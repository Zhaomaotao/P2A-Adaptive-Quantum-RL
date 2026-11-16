

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from world_model import (
    KEY_STATE_IDXS,
    KEY_STATE_NAMES,
    OBS_NAMES,
    LearnedWorldModel,
    StandardScaler,
    WorldModelConfig,
)


DATASET_PATH = os.path.join("world_model_data", "wm_transitions.npz")
MODEL_DIR = "models_world"
FIG_DIR = "figs_world"
MODEL_PATH = os.path.join(MODEL_DIR, "world_model.pt")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def train_val_split(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def weighted_delta_loss(pred_delta, true_delta, cfg: WorldModelConfig):
    base = ((pred_delta - true_delta) ** 2).mean()
    key = ((pred_delta[:, KEY_STATE_IDXS] - true_delta[:, KEY_STATE_IDXS]) ** 2).mean()
    return cfg.delta_loss_weight * base + cfg.key_state_loss_weight * key


def evaluate(model, loader, cfg, device) -> Dict[str, float]:
    model.eval()
    delta_losses = []
    reward_losses = []
    total_losses = []
    with torch.no_grad():
        for obs_norm, act, delta_norm, rew_norm in loader:
            obs_norm = obs_norm.to(device)
            act = act.to(device)
            delta_norm = delta_norm.to(device)
            rew_norm = rew_norm.to(device)
            pred_delta, pred_rew = model(obs_norm, act)
            dl = weighted_delta_loss(pred_delta, delta_norm, cfg)
            rl = ((pred_rew - rew_norm.squeeze(-1)) ** 2).mean()
            loss = dl + cfg.reward_loss_weight * rl
            delta_losses.append(float(dl.cpu()))
            reward_losses.append(float(rl.cpu()))
            total_losses.append(float(loss.cpu()))
    return {
        "total": float(np.mean(total_losses)),
        "delta": float(np.mean(delta_losses)),
        "reward": float(np.mean(reward_losses)),
    }


def plot_losses(history: Dict[str, list]):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_total"], label="train total")
    plt.plot(history["val_total"], label="val total")
    plt.plot(history["val_delta"], label="val transition")
    plt.plot(history["val_reward"], label="val reward")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("World Model Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "world_model_loss.png"), dpi=300)
    plt.show()


def plot_one_step_predictions(model, arrays, scalers, device, max_points: int = 300):
    obs, actions, rewards, next_obs = arrays
    obs_scaler, delta_scaler, reward_scaler = scalers
    model.eval()
    n = min(max_points, len(obs))
    obs_sub = obs[:n]
    act_sub = actions[:n]
    next_sub = next_obs[:n]
    rew_sub = rewards[:n]

    obs_norm = torch.as_tensor(obs_scaler.transform(obs_sub), dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act_sub, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_delta_norm, pred_rew_norm = model(obs_norm, act_t)
    pred_delta = delta_scaler.inverse_transform(pred_delta_norm.cpu().numpy())
    pred_next = obs_sub + pred_delta
    pred_next[:, :5] = np.clip(pred_next[:, :5], 0.0, 1.0)
    pred_rew = reward_scaler.inverse_transform(pred_rew_norm.cpu().numpy().reshape(-1, 1)).squeeze(-1)

    x = np.arange(n)
    for idx, name in zip(KEY_STATE_IDXS, KEY_STATE_NAMES):
        plt.figure(figsize=(8, 4))
        plt.plot(x, next_sub[:, idx], label=f"true {name}", linewidth=1.4)
        plt.plot(x, pred_next[:, idx], label=f"pred {name}", linewidth=1.4)
        plt.xlabel("Sample index")
        plt.ylabel(name)
        plt.title(f"One-step World Model Prediction: {name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"wm_pred_{name}.png"), dpi=300)
        plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(x, rew_sub, label="true reward", linewidth=1.2)
    plt.plot(x, pred_rew, label="pred reward", linewidth=1.2)
    plt.xlabel("Sample index")
    plt.ylabel("Reward")
    plt.title("One-step World Model Prediction: Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "wm_pred_reward.png"), dpi=300)
    plt.show()


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("World model dataset not found. Run python collect_world_model_data.py first.")

    cfg = WorldModelConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load(DATASET_PATH)
    obs = data["obs"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    rewards = data["rewards"].astype(np.float32).reshape(-1, 1)
    next_obs = data["next_obs"].astype(np.float32)
    delta = (next_obs - obs).astype(np.float32)

    obs_dim = obs.shape[1]
    act_dim = actions.shape[1]
    print(f"Loaded dataset: {len(obs)} transitions, obs_dim={obs_dim}, act_dim={act_dim}")

    train_idx, val_idx = train_val_split(len(obs), cfg.val_ratio, cfg.seed)

    obs_scaler = StandardScaler().fit(obs[train_idx])
    delta_scaler = StandardScaler().fit(delta[train_idx])
    reward_scaler = StandardScaler().fit(rewards[train_idx])

    obs_norm = obs_scaler.transform(obs)
    delta_norm = delta_scaler.transform(delta)
    reward_norm = reward_scaler.transform(rewards)

    train_ds = TensorDataset(
        torch.as_tensor(obs_norm[train_idx], dtype=torch.float32),
        torch.as_tensor(actions[train_idx], dtype=torch.float32),
        torch.as_tensor(delta_norm[train_idx], dtype=torch.float32),
        torch.as_tensor(reward_norm[train_idx], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.as_tensor(obs_norm[val_idx], dtype=torch.float32),
        torch.as_tensor(actions[val_idx], dtype=torch.float32),
        torch.as_tensor(delta_norm[val_idx], dtype=torch.float32),
        torch.as_tensor(reward_norm[val_idx], dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = LearnedWorldModel(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history = {"train_total": [], "train_delta": [], "train_reward": [], "val_total": [], "val_delta": [], "val_reward": []}
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_total = []
        train_delta = []
        train_reward = []
        for obs_b, act_b, delta_b, rew_b in train_loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            delta_b = delta_b.to(device)
            rew_b = rew_b.to(device)

            pred_delta, pred_rew = model(obs_b, act_b)
            dl = weighted_delta_loss(pred_delta, delta_b, cfg)
            rl = ((pred_rew - rew_b.squeeze(-1)) ** 2).mean()
            loss = dl + cfg.reward_loss_weight * rl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            train_total.append(float(loss.detach().cpu()))
            train_delta.append(float(dl.detach().cpu()))
            train_reward.append(float(rl.detach().cpu()))
        scheduler.step()

        val_metrics = evaluate(model, val_loader, cfg, device)
        history["train_total"].append(float(np.mean(train_total)))
        history["train_delta"].append(float(np.mean(train_delta)))
        history["train_reward"].append(float(np.mean(train_reward)))
        history["val_total"].append(val_metrics["total"])
        history["val_delta"].append(val_metrics["delta"])
        history["val_reward"].append(val_metrics["reward"])

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "obs_scaler": obs_scaler.state_dict(),
                    "delta_scaler": delta_scaler.state_dict(),
                    "reward_scaler": reward_scaler.state_dict(),
                    "obs_names": OBS_NAMES,
                    "key_state_idxs": KEY_STATE_IDXS,
                },
                MODEL_PATH,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | "
                f"train {history['train_total'][-1]:.5f} | "
                f"val {val_metrics['total']:.5f} | "
                f"val_delta {val_metrics['delta']:.5f} | val_reward {val_metrics['reward']:.5f}"
            )

    np.savez(os.path.join(MODEL_DIR, "world_model_training_history.npz"), **{k: np.asarray(v) for k, v in history.items()})
    plot_losses(history)

    # Load best checkpoint for prediction plots.
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    plot_one_step_predictions(
        model,
        arrays=(obs[val_idx], actions[val_idx], rewards[val_idx].squeeze(-1), next_obs[val_idx]),
        scalers=(obs_scaler, delta_scaler, reward_scaler),
        device=device,
    )
    print(f"Saved best world model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
