

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch

from energy_env import EnergyEnv
from main_train_qmappo import QMAPPOPolicy, RunningMeanStd, TrainConfig


DATA_PATH = "energy_data.npz"
QMAPPO_MODEL_PATH = os.path.join("models_non_adaptive", "non_adaptive_energy_model.pt")
OUT_DIR = "world_model_data"
OUT_PATH = os.path.join(OUT_DIR, "wm_transitions.npz")

os.makedirs(OUT_DIR, exist_ok=True)


def sample_valid_random_action(rng: np.random.Generator) -> np.ndarray:
    """Sample actions in the physical support of each device."""
    return np.asarray([
        rng.uniform(-1.0, 1.0),  # Battery
        rng.uniform(0.0, 1.0),   # P2A
        rng.uniform(0.0, 1.0),   # CHP
        rng.uniform(0.0, 1.0),   # GB
        rng.uniform(0.0, 1.0),   # EB
    ], dtype=np.float32)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("energy_data.npz not found. Run python data_generator.py first.")

    cfg = TrainConfig()
    rng = np.random.default_rng(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use day_index=None so episodes can start from random days if EnergyEnv supports it.
    env = EnergyEnv(data_path=DATA_PATH)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = None
    obs_rms = None
    if os.path.exists(QMAPPO_MODEL_PATH):
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        policy = QMAPPOPolicy(obs_dim=obs_dim, cfg=cfg, device=device)
        policy.load(QMAPPO_MODEL_PATH, obs_rms)
        print(f"Loaded Non-Adaptive policy from {QMAPPO_MODEL_PATH}")
    else:
        print("Non-Adaptive model not found. Collecting fully random data.")

    obs_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []
    reward_list: List[float] = []
    next_obs_list: List[np.ndarray] = []
    done_list: List[float] = []

    # Roughly 500 episodes * 24 hours = 12k transitions, enough for first world model.
    num_episodes = 500
    random_action_prob = 0.35
    action_noise_std = 0.12

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            if policy is not None and rng.random() > random_action_prob:
                obs_norm = obs_rms.normalize(obs)
                action, _lp, _v = policy.act(obs_norm, deterministic=True)
                action = action + rng.normal(0.0, action_noise_std, size=act_dim).astype(np.float32)
                action[0] = np.clip(action[0], -1.0, 1.0)
                action[1:] = np.clip(action[1:], 0.0, 1.0)
                action = action.astype(np.float32)
            else:
                action = sample_valid_random_action(rng)

            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)

            obs_list.append(obs.astype(np.float32))
            action_list.append(action.astype(np.float32))
            reward_list.append(float(reward))
            next_obs_list.append(next_obs.astype(np.float32))
            done_list.append(float(done))
            ep_reward += float(reward)
            obs = next_obs

        if (ep + 1) % 50 == 0:
            print(f"Collected episode {ep + 1:4d}/{num_episodes}, last ep reward = {ep_reward: .3f}")

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    action_arr = np.asarray(action_list, dtype=np.float32)
    reward_arr = np.asarray(reward_list, dtype=np.float32)
    next_obs_arr = np.asarray(next_obs_list, dtype=np.float32)
    done_arr = np.asarray(done_list, dtype=np.float32)

    np.savez_compressed(
        OUT_PATH,
        obs=obs_arr,
        actions=action_arr,
        rewards=reward_arr,
        next_obs=next_obs_arr,
        dones=done_arr,
    )
    print(f"Saved {len(obs_arr)} transitions to {OUT_PATH}")
    print(f"obs shape={obs_arr.shape}, actions shape={action_arr.shape}, reward mean={reward_arr.mean():.3f}")


if __name__ == "__main__":
    main()
