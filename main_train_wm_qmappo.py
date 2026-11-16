

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from energy_env import EnergyEnv
from main_train_qmappo import (
    AGENT_NAMES,
    AGENT_OBS_IDXS,
    QMAPPOPolicy,
    RunningMeanStd,
    TrainConfig,
    compute_gae,
)
from world_model import (
    LearnedWorldModel,
    StandardScaler,
    WorldModelConfig,
    predict_next_obs_and_reward,
)


DATA_PATH = "energy_data.npz"
NON_ADAPTIVE_MODEL_PATH = os.path.join("models_non_adaptive", "non_adaptive_energy_model.pt")
WORLD_MODEL_PATH = os.path.join("models_world", "world_model.pt")
MODEL_DIR = "models_wm_qmappo"
LOG_DIR = "logs_wm_qmappo"
FIG_DIR = "figs_wm_qmappo"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class WMQMAPPOConfig(TrainConfig):
    # The real rollout still dominates. Imagined rollouts are an auxiliary signal.
    total_timesteps: int = 200_000
    rollout_steps: int = 2048
    ppo_epochs: int = 6
    minibatch_size: int = 256

    # World-model imagination settings.
    imagination_horizon: int = 3
    imagined_starts_per_update: int = 384
    imagined_weight: float = 0.20
    imagined_actor_weight: float = 0.10
    imagined_critic_weight: float = 0.25
    imagined_warmup_updates: int = 4
    imagined_ramp_updates: int = 20
    imagined_state_clip_std: float = 5.0
    imagined_reward_abs_clip: float = 20.0

    # Keep the real-trajectory PPO backbone identical across comparison methods.
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    clip_range: float = 0.18
    ent_coef: float = 0.02

    # Save/check settings.
    save_every_updates: int = 10


class WorldModelBundle:
    def __init__(self, path: str, device: torch.device):
        if not os.path.exists(path):
            raise FileNotFoundError("world_model.pt not found. Run python train_world_model.py first.")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        wm_cfg = WorldModelConfig()
        if isinstance(ckpt.get("cfg"), dict):
            for k, v in ckpt["cfg"].items():
                if hasattr(wm_cfg, k):
                    setattr(wm_cfg, k, v)
        self.model = LearnedWorldModel(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"], cfg=wm_cfg).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.obs_scaler = StandardScaler(); self.obs_scaler.load_state_dict(ckpt["obs_scaler"])
        self.delta_scaler = StandardScaler(); self.delta_scaler.load_state_dict(ckpt["delta_scaler"])
        self.reward_scaler = StandardScaler(); self.reward_scaler.load_state_dict(ckpt["reward_scaler"])
        self.device = device

    def predict(self, obs_raw: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        return predict_next_obs_and_reward(
            model=self.model,
            obs=obs_raw,
            action=action,
            obs_scaler=self.obs_scaler,
            delta_scaler=self.delta_scaler,
            reward_scaler=self.reward_scaler,
            device=self.device,
        )


def update_policy_weighted(policy: QMAPPOPolicy, batch: Dict[str, np.ndarray], cfg: WMQMAPPOConfig):
    """PPO-style update with optional sample weights.

    Real samples get weight 1.0. Imagined samples get a smaller weight so that the
    model-generated data improves learning without dominating real interaction.
    """
    device = policy.device
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    actor_weights = torch.as_tensor(batch["actor_weights"], dtype=torch.float32, device=device)
    critic_weights = torch.as_tensor(batch["critic_weights"], dtype=torch.float32, device=device)

    n = obs.shape[0]
    idxs = np.arange(n)
    last_losses = {}

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(idxs)
        for start in range(0, n, cfg.minibatch_size):
            mb = idxs[start:start + cfg.minibatch_size]
            mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)
            obs_mb = obs[mb_t]
            act_mb = actions[mb_t]
            old_lp_mb = old_log_probs[mb_t]
            adv_mb = advantages[mb_t]
            ret_mb = returns[mb_t]
            aw_mb = actor_weights[mb_t]
            cw_mb = critic_weights[mb_t]

            actor_loss_total = 0.0
            entropy_total = 0.0
            for agent_i, name in enumerate(AGENT_NAMES):
                local = policy._local_obs(obs_mb, name)
                action_i = act_mb[:, agent_i:agent_i + 1]
                new_lp, entropy = policy.actors[name].evaluate_actions(local, action_i)
                mean = policy.actors[name].action_mean(local).squeeze(-1)
                ratio = torch.exp(new_lp - old_lp_mb[:, agent_i])
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_mb

                weight_sum = aw_mb.sum() + 1e-8
                ppo_term = -torch.min(surr1, surr2)
                actor_loss = (aw_mb * ppo_term).sum() / weight_sum
                weighted_entropy = (aw_mb * entropy).sum() / weight_sum
                actor_loss = actor_loss - cfg.ent_coef * weighted_entropy

                # Apply the same boundary regularization with the same source weights.
                if name == "Battery":
                    boundary = torch.relu(torch.abs(mean) - 0.92)
                else:
                    boundary = torch.relu(0.03 - mean) + torch.relu(mean - 0.97)
                mean_reg = cfg.action_mean_reg * (aw_mb * boundary.pow(2)).sum() / weight_sum
                eb_reg = 0.0
                if name == "EB":
                    eb_vec = torch.relu(mean - cfg.eb_mean_threshold).pow(2)
                    eb_reg = cfg.eb_mean_high_reg * (aw_mb * eb_vec).sum() / weight_sum
                p2a_neg_reg = 0.0
                if name == "P2A":
                    p2a_vec = torch.relu(-mean).pow(2)
                    p2a_neg_reg = cfg.p2a_negative_reg * (aw_mb * p2a_vec).sum() / weight_sum
                actor_loss = actor_loss + mean_reg + eb_reg + p2a_neg_reg

                policy.actor_opts[name].zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(policy.actors[name].parameters(), cfg.max_grad_norm)
                policy.actor_opts[name].step()
                actor_loss_total += float(actor_loss.detach().cpu())
                entropy_total += float(weighted_entropy.detach().cpu())

            value_pred = policy.critic(obs_mb)
            value_loss_vec = 0.5 * ((value_pred - ret_mb) ** 2)
            value_loss = (cw_mb * value_loss_vec).sum() / (cw_mb.sum() + 1e-8)
            policy.critic_opt.zero_grad(set_to_none=True)
            (cfg.vf_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(policy.critic.parameters(), cfg.max_grad_norm)
            policy.critic_opt.step()

            last_losses = {
                "actor_loss": actor_loss_total / len(AGENT_NAMES),
                "critic_loss": float(value_loss.detach().cpu()),
                "entropy": entropy_total / len(AGENT_NAMES),
            }
    return last_losses


def generate_imagined_batch(
    policy: QMAPPOPolicy,
    wm: WorldModelBundle,
    start_obs_raw: np.ndarray,
    obs_rms: RunningMeanStd,
    cfg: WMQMAPPOConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Generate short imagined rollouts from real states.

    The world model is only trusted for short horizons. Each imagined trajectory
    starts from a real state, then repeatedly uses the current policy and the
    learned world model to predict the next state and reward.
    """
    if len(start_obs_raw) == 0:
        raise ValueError("No start states for imagined rollout.")

    n_starts = min(cfg.imagined_starts_per_update, len(start_obs_raw))
    start_idxs = rng.choice(len(start_obs_raw), size=n_starts, replace=False)

    obs_buf, action_buf, logp_buf, reward_buf, done_buf, value_buf = [], [], [], [], [], []
    traj_slices: List[Tuple[int, int, float]] = []

    for idx in start_idxs:
        begin = len(reward_buf)
        obs_raw = start_obs_raw[idx].copy().astype(np.float32)

        for _h in range(cfg.imagination_horizon):
            obs_norm = obs_rms.normalize(obs_raw)
            action, log_probs, value = policy.act(obs_norm, deterministic=False)
            next_obs_raw, pred_reward = wm.predict(obs_raw, action)

            # Reject numerical outliers and keep imagined states inside the
            # empirical normalization envelope. This reduces harmful model bias
            # without replacing real environment interaction.
            next_obs_raw = np.where(np.isfinite(next_obs_raw), next_obs_raw, obs_raw)
            state_std = np.sqrt(np.maximum(obs_rms.var, 1e-8)).astype(np.float32)
            state_low = obs_rms.mean.astype(np.float32) - cfg.imagined_state_clip_std * state_std
            state_high = obs_rms.mean.astype(np.float32) + cfg.imagined_state_clip_std * state_std
            next_obs_raw = np.clip(next_obs_raw, state_low, state_high)
            next_obs_raw[:5] = np.clip(next_obs_raw[:5], 0.0, 1.0)
            pred_reward = float(np.clip(pred_reward, -cfg.imagined_reward_abs_clip, cfg.imagined_reward_abs_clip))

            obs_buf.append(obs_norm.astype(np.float32))
            action_buf.append(action.astype(np.float32))
            logp_buf.append(log_probs.astype(np.float32))
            reward_buf.append(float(pred_reward))
            done_buf.append(0.0)
            value_buf.append(float(value))
            obs_raw = next_obs_raw.astype(np.float32)

        obs_norm_last = obs_rms.normalize(obs_raw)
        with torch.no_grad():
            last_value = float(policy.critic(torch.as_tensor(obs_norm_last, dtype=torch.float32, device=policy.device).unsqueeze(0)).cpu().numpy()[0])
        end = len(reward_buf)
        traj_slices.append((begin, end, last_value))

    obs_arr = np.asarray(obs_buf, dtype=np.float32)
    actions_arr = np.asarray(action_buf, dtype=np.float32)
    logp_arr = np.asarray(logp_buf, dtype=np.float32)
    rewards_arr = np.asarray(reward_buf, dtype=np.float32)
    dones_arr = np.asarray(done_buf, dtype=np.float32)
    values_arr = np.asarray(value_buf, dtype=np.float32)

    adv_all = np.zeros_like(rewards_arr, dtype=np.float32)
    ret_all = np.zeros_like(rewards_arr, dtype=np.float32)
    for begin, end, last_value in traj_slices:
        adv, ret = compute_gae(
            rewards_arr[begin:end],
            dones_arr[begin:end],
            values_arr[begin:end],
            last_value,
            cfg.gamma,
            cfg.gae_lambda,
        )
        adv_all[begin:end] = adv
        ret_all[begin:end] = ret

    return {
        "obs": obs_arr,
        "actions": actions_arr,
        "log_probs": logp_arr,
        "advantages": adv_all,
        "returns": ret_all,
    }


def _standardize_advantages(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


def imagination_scale(update_idx: int, cfg: WMQMAPPOConfig) -> float:
    """Warm up on real trajectories and then gradually introduce imagination."""
    if update_idx < cfg.imagined_warmup_updates:
        return 0.0
    progress = update_idx - cfg.imagined_warmup_updates + 1
    return float(min(1.0, progress / max(1, cfg.imagined_ramp_updates)))


def combine_real_and_imagined(
    real_batch: Dict[str, np.ndarray],
    imagined_batch: Dict[str, np.ndarray],
    cfg: WMQMAPPOConfig,
    scale: float = 1.0,
):
    real_n = len(real_batch["obs"])
    imag_n = len(imagined_batch["obs"])

    # Normalize advantages separately so predicted rewards cannot rescale the
    # real-policy gradient merely because their numerical distribution differs.
    real_local = dict(real_batch)
    imag_local = dict(imagined_batch)
    real_local["advantages"] = _standardize_advantages(real_local["advantages"])
    imag_local["advantages"] = _standardize_advantages(imag_local["advantages"])

    out = {}
    for key in ["obs", "actions", "log_probs", "advantages", "returns"]:
        out[key] = np.concatenate([real_local[key], imag_local[key]], axis=0).astype(np.float32)

    scale = float(np.clip(scale, 0.0, 1.0))
    out["weights"] = np.concatenate([
        np.ones(real_n), np.ones(imag_n) * cfg.imagined_weight * scale
    ]).astype(np.float32)
    out["actor_weights"] = np.concatenate([
        np.ones(real_n), np.ones(imag_n) * cfg.imagined_actor_weight * scale
    ]).astype(np.float32)
    out["critic_weights"] = np.concatenate([
        np.ones(real_n), np.ones(imag_n) * cfg.imagined_critic_weight * scale
    ]).astype(np.float32)
    return out


def plot_training(episode_rewards: List[float], out_path: str):
    if not episode_rewards:
        return
    x = np.arange(1, len(episode_rewards) + 1)
    y = np.array(episode_rewards, dtype=np.float32)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=1.2)
    if len(y) >= 10:
        w = min(20, max(5, len(y) // 20))
        ma = np.convolve(y, np.ones(w) / w, mode="valid")
        plt.plot(np.arange(w, len(y) + 1), ma, linewidth=2.0, label=f"moving avg ({w})")
        plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("World-Model-Assisted Quantum Training Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def main():
    cfg = WMQMAPPOConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("energy_data.npz not found. Run python data_generator.py first.")
    if not os.path.exists(WORLD_MODEL_PATH):
        raise FileNotFoundError("world_model.pt not found. Run python train_world_model.py first.")

    env = EnergyEnv(data_path=DATA_PATH)
    obs_dim = env.observation_space.shape[0]
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    policy = QMAPPOPolicy(obs_dim=obs_dim, cfg=cfg, device=device)

    if os.path.exists(NON_ADAPTIVE_MODEL_PATH):
        policy.load(NON_ADAPTIVE_MODEL_PATH, obs_rms)
        print(f"Loaded Non-Adaptive checkpoint from {NON_ADAPTIVE_MODEL_PATH}")
    else:
        print("Non-Adaptive checkpoint not found. Training the world-model stage from scratch.")

    wm = WorldModelBundle(WORLD_MODEL_PATH, device)
    print(f"Loaded world model from {WORLD_MODEL_PATH}")

    obs, _ = env.reset(seed=cfg.seed)
    obs_rms.update(obs)
    episode_rewards: List[float] = []
    current_ep_reward = 0.0
    total_steps = 0
    update_idx = 0

    while total_steps < cfg.total_timesteps:
        rollout = {"obs": [], "obs_raw": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []}

        for _ in range(cfg.rollout_steps):
            obs_rms.update(obs)
            obs_raw = obs.copy().astype(np.float32)
            obs_norm = obs_rms.normalize(obs)
            action, log_probs, value = policy.act(obs_norm, deterministic=False)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)

            rollout["obs"].append(obs_norm)
            rollout["obs_raw"].append(obs_raw)
            rollout["actions"].append(action)
            rollout["log_probs"].append(log_probs)
            rollout["rewards"].append(float(reward))
            rollout["dones"].append(float(done))
            rollout["values"].append(float(value))

            current_ep_reward += float(reward)
            total_steps += 1
            obs = next_obs

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                obs, _ = env.reset()
                obs_rms.update(obs)

            if total_steps >= cfg.total_timesteps:
                break

        obs_norm = obs_rms.normalize(obs)
        with torch.no_grad():
            last_value = float(policy.critic(torch.as_tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy()[0])

        rewards = np.asarray(rollout["rewards"], dtype=np.float32)
        dones = np.asarray(rollout["dones"], dtype=np.float32)
        values = np.asarray(rollout["values"], dtype=np.float32)
        advantages, returns = compute_gae(rewards, dones, values, last_value, cfg.gamma, cfg.gae_lambda)

        real_batch = {
            "obs": np.asarray(rollout["obs"], dtype=np.float32),
            "actions": np.asarray(rollout["actions"], dtype=np.float32),
            "log_probs": np.asarray(rollout["log_probs"], dtype=np.float32),
            "advantages": advantages.astype(np.float32),
            "returns": returns.astype(np.float32),
        }

        imagined_batch = generate_imagined_batch(
            policy=policy,
            wm=wm,
            start_obs_raw=np.asarray(rollout["obs_raw"], dtype=np.float32),
            obs_rms=obs_rms,
            cfg=cfg,
            rng=rng,
        )
        imag_scale = imagination_scale(update_idx, cfg)
        batch = combine_real_and_imagined(real_batch, imagined_batch, cfg, scale=imag_scale)
        losses = update_policy_weighted(policy, batch, cfg)
        update_idx += 1

        recent = np.mean(episode_rewards[-20:]) if episode_rewards else np.nan
        print(
            f"Update {update_idx:04d} | steps {total_steps:7d} | "
            f"recent_ep_reward {recent:9.3f} | real {len(real_batch['obs'])} | imag {len(imagined_batch['obs'])} | "
            f"actor_loss {losses.get('actor_loss', np.nan): .4f} | "
            f"critic_loss {losses.get('critic_loss', np.nan): .4f} | "
            f"entropy {losses.get('entropy', np.nan): .4f} | imag_scale {imag_scale:.2f}"
        )

        if update_idx % cfg.save_every_updates == 0:
            policy.save(os.path.join(MODEL_DIR, "wm_qmappo_energy_model.pt"), obs_rms, extra={"total_steps": total_steps})
            np.save(os.path.join(LOG_DIR, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))

    policy.save(os.path.join(MODEL_DIR, "wm_qmappo_energy_model.pt"), obs_rms, extra={"total_steps": total_steps})
    np.save(os.path.join(LOG_DIR, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    plot_training(episode_rewards, os.path.join(FIG_DIR, "wm_qmappo_training_curve.png"))
    print(f"Saved world-model-assisted quantum model to {os.path.join(MODEL_DIR, 'wm_qmappo_energy_model.pt')}")


if __name__ == "__main__":
    main()
