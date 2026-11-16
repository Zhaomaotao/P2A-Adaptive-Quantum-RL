

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from energy_env import EnergyEnv
from main_train_qmappo import RunningMeanStd, compute_gae
from main_train_wm_qmappo import (
    WorldModelBundle,
    generate_imagined_batch,
    combine_real_and_imagined,
    imagination_scale,
    update_policy_weighted,
    plot_training,
)
from main_train_wm_qmappo_qmaml import WMQMAPPOQMAMLConfig
from quantum_tuning import apply_qmaml_initialization
from dqas_tuning import load_dqas_architectures, DQASTunedQMAPPOPolicy

DATA_PATH = "energy_data.npz"
WORLD_MODEL_PATH = os.path.join("models_world", "world_model.pt")
QMAML_INIT_PATH = os.path.join("models_qtuning", "qmaml_initializer.pt")
DQAS_ARCH_PATH = os.path.join("models_qtuning", "dqas_architectures.json")
MODEL_DIR = "models_proposed"
LOG_DIR = "logs_proposed"
FIG_DIR = "figs_proposed"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class WMQMAPPOQMAMLDQASConfig(WMQMAPPOQMAMLConfig):
    total_timesteps: int = 200_000
    rollout_steps: int = 2048
    ppo_epochs: int = 6
    minibatch_size: int = 256

    # The comparison uses an identical real-trajectory PPO backbone.
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    ent_coef: float = 0.02
    clip_range: float = 0.18

    imagination_horizon: int = 3
    imagined_starts_per_update: int = 384
    imagined_weight: float = 0.20
    imagined_actor_weight: float = 0.10
    imagined_critic_weight: float = 0.25
    imagined_warmup_updates: int = 4
    imagined_ramp_updates: int = 20

    save_every_updates: int = 10


def main():
    cfg = WMQMAPPOQMAMLDQASConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for path, msg in [
        (DATA_PATH, "Run python data_generator.py first."),
        (WORLD_MODEL_PATH, "Run python train_world_model.py first."),
        (QMAML_INIT_PATH, "Run python train_qmaml_initializer.py first."),
        (DQAS_ARCH_PATH, "Run python train_dqas_selector.py first."),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}. {msg}")

    dqas_payload = load_dqas_architectures(DQAS_ARCH_PATH)
    print("Proposed method: loaded adaptive agent-specific PQC architectures:")
    for name, arch in dqas_payload["selected"].items():
        print(
            f"  {name:7s} | score={arch['score']:.3f} | {arch['arch_name']:6s} | "
            f"qubits={arch['n_qubits']} depth={arch['pqc_depth']} reupload={arch['reupload']}"
        )

    env = EnergyEnv(data_path=DATA_PATH)
    obs_dim = env.observation_space.shape[0]
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    policy = DQASTunedQMAPPOPolicy(obs_dim=obs_dim, cfg=cfg, device=device, dqas_payload=dqas_payload)

    init_dict = apply_qmaml_initialization(policy, QMAML_INIT_PATH, device, verbose=True)
    wm = WorldModelBundle(WORLD_MODEL_PATH, device)
    print(f"Loaded world model from {WORLD_MODEL_PATH}")

    obs, _ = env.reset(seed=cfg.seed)
    obs_rms.update(obs)
    episode_rewards: List[float] = []
    current_ep_reward = 0.0
    total_steps = 0
    update_idx = 0
    best_recent_reward = -np.inf

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
        if np.isfinite(recent) and recent > best_recent_reward:
            best_recent_reward = float(recent)
            policy.save(
                os.path.join(MODEL_DIR, "proposed_energy_model.pt"),
                obs_rms,
                extra={
                    "total_steps": total_steps,
                    "best_recent_reward": best_recent_reward,
                    "qmaml_init": init_dict,
                    "dqas": dqas_payload,
                    "dqas_payload": dqas_payload,
                },
            )
        print(
            f"Update {update_idx:04d} | steps {total_steps:7d} | "
            f"recent_ep_reward {recent:9.3f} | real {len(real_batch['obs'])} | imag {len(imagined_batch['obs'])} | "
            f"actor_loss {losses.get('actor_loss', np.nan): .4f} | "
            f"critic_loss {losses.get('critic_loss', np.nan): .4f} | "
            f"entropy {losses.get('entropy', np.nan): .4f} | imag_scale {imag_scale:.2f}"
        )

        if update_idx % cfg.save_every_updates == 0:
            policy.save(
                os.path.join(MODEL_DIR, "proposed_latest_model.pt"),
                obs_rms,
                extra={"total_steps": total_steps, "qmaml_init": init_dict, "dqas": dqas_payload},
            )
            np.save(os.path.join(LOG_DIR, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))

    policy.save(
        os.path.join(MODEL_DIR, "proposed_latest_model.pt"),
        obs_rms,
        extra={"total_steps": total_steps, "qmaml_init": init_dict, "dqas": dqas_payload},
    )
    np.save(os.path.join(LOG_DIR, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    plot_training(episode_rewards, os.path.join(FIG_DIR, "proposed_training_curve.png"))
    print(f"Saved best Proposed checkpoint to {os.path.join(MODEL_DIR, 'proposed_energy_model.pt')}")


if __name__ == "__main__":
    main()
