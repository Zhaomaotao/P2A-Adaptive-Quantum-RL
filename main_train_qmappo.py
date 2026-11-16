

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


DATA_PATH = "energy_data.npz"
MODEL_DIR = "models_non_adaptive"
LOG_DIR = "logs_non_adaptive"
FIG_DIR = "figs_non_adaptive"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


AGENT_NAMES = ["Battery", "P2A", "CHP", "GB", "EB"]
ACTION_MODES = {"Battery": "bipolar", "P2A": "unipolar", "CHP": "unipolar", "GB": "unipolar", "EB": "unipolar"}

# Observation vector in EnergyEnv:
# 0 SOC_e, 1 SOC_NH3, 2 SOC_HNO3, 3 SOC_NH4NO3, 4 SOC_F,
# 5 price_el, 6 load_el, 7 load_th, 8 wt_el, 9 pvt_el, 10 pvt_th,
# 11 q_na_ref, 12 q_an_ref, 13 q_f_ref
AGENT_OBS_IDXS: Dict[str, List[int]] = {
    "Battery": [0, 5, 6, 8, 9],
    "P2A": [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13],
    "CHP": [1, 5, 6, 7, 8, 9, 10],
    "GB": [1, 5, 7, 10],
    "EB": [0, 5, 6, 7, 8, 9, 10],
}


@dataclass
class TrainConfig:
    total_timesteps: int = 200_000
    rollout_steps: int = 2048
    ppo_epochs: int = 6
    minibatch_size: int = 256
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.18
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Quantum-inspired actors are a little more sensitive than MLP actors.
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    seed: int = 42
    hidden_dim: int = 128
    n_qubits: int = 6
    pqc_depth: int = 2
    reupload: int = 2
    action_mean_reg: float = 0.004
    eb_mean_high_reg: float = 0.03
    eb_mean_threshold: float = 0.65
    p2a_negative_reg: float = 0.0
    save_every_updates: int = 10


class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(y, -clip, clip).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "var": self.var, "count": np.array(self.count)}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = float(state["count"])


class QuantumInspiredPQC(nn.Module):
    """A lightweight differentiable PQC-like feature layer.

    This is a simulation-friendly quantum-inspired layer for algorithm development.
    It mimics three PQC ideas without requiring a quantum simulator:
      1) angle encoding / data re-uploading,
      2) trainable parameterized rotations,
      3) ring-like entanglement mixing between qubit features.

    Later DQAS/Q-MAML stages can tune the depth, re-uploading, entanglement pattern,
    input scaling, and initialization of this module.
    """

    def __init__(self, obs_dim: int, n_qubits: int = 6, depth: int = 3, reupload: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.reupload = reupload

        self.encoder = nn.Linear(obs_dim, n_qubits)
        self.input_scale = nn.Parameter(torch.ones(n_qubits) * 0.7)

        # Trainable rotation-like parameters: [reupload, depth, qubit]
        self.theta_rx = nn.Parameter(0.05 * torch.randn(reupload, depth, n_qubits))
        self.theta_ry = nn.Parameter(0.05 * torch.randn(reupload, depth, n_qubits))
        self.theta_rz = nn.Parameter(0.05 * torch.randn(reupload, depth, n_qubits))

        # Ring entanglement strength per layer. It is learnable but bounded by tanh in forward.
        self.ent_strength = nn.Parameter(0.1 * torch.ones(reupload, depth, n_qubits))
        self.norm = nn.LayerNorm(n_qubits)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Angle encoding. Tanh keeps angles in a stable range.
        base_angles = torch.tanh(self.encoder(obs)) * self.input_scale
        state = torch.zeros_like(base_angles)

        for r in range(self.reupload):
            # Data re-uploading: inject the encoded observation repeatedly.
            state = state + base_angles
            for d in range(self.depth):
                # Rotation-like nonlinear transformation.
                rx = torch.sin(state + self.theta_rx[r, d])
                ry = torch.cos(state + self.theta_ry[r, d])
                rz = torch.sin(0.5 * state + self.theta_rz[r, d])
                state = (rx + ry + rz) / 3.0

                # Ring entanglement-like mixing: each qubit receives information from the next qubit.
                rolled = torch.roll(state, shifts=1, dims=-1)
                ent = torch.tanh(self.ent_strength[r, d])
                state = state + ent * rolled
                state = self.norm(state)

        # Measurement-like features in [-1, 1].
        return torch.tanh(state)


class QuantumActor(nn.Module):
    """Fixed-architecture hybrid quantum-classical bounded actor.

    The Non-Adaptive scheme uses the same PQC structure for all agents. A
    compact classical residual branch improves gradient flow, while the latent
    Gaussian policy is smoothly mapped to the device-specific action interval.
    """

    def __init__(
        self,
        obs_dim: int,
        n_qubits: int = 6,
        depth: int = 2,
        reupload: int = 2,
        hidden_dim: int = 64,
        action_mode: str = "unipolar",
    ):
        super().__init__()
        self.action_mode = action_mode
        self.pqc = QuantumInspiredPQC(obs_dim=obs_dim, n_qubits=n_qubits, depth=depth, reupload=reupload)
        self.classical_residual = nn.Sequential(nn.Linear(obs_dim, n_qubits), nn.Tanh())
        self.fusion_gate = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.fusion_norm = nn.LayerNorm(n_qubits)
        self.head = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output_scale = nn.Parameter(torch.tensor([0.65], dtype=torch.float32))
        self.log_std = nn.Parameter(torch.tensor([-1.35], dtype=torch.float32))
        self._eps = 1e-6

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        q = self.pqc(obs)
        c = self.classical_residual(obs)
        gate = torch.sigmoid(self.fusion_gate)
        return torch.tanh(self.fusion_norm(q + gate * c))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        features = self._features(obs)
        scale = torch.clamp(self.output_scale, 0.35, 1.20)
        latent_mean = 2.5 * torch.tanh(scale * self.head(features) / 2.5)
        std = torch.exp(torch.clamp(self.log_std, -2.5, -0.55)).expand_as(latent_mean)
        return torch.distributions.Normal(latent_mean, std)

    def _transform(self, latent: torch.Tensor) -> torch.Tensor:
        if self.action_mode == "bipolar":
            return torch.tanh(latent)
        return torch.sigmoid(latent)

    def _inverse_transform(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_mode == "bipolar":
            a = torch.clamp(action, -1.0 + self._eps, 1.0 - self._eps)
            return 0.5 * (torch.log1p(a) - torch.log1p(-a))
        a = torch.clamp(action, self._eps, 1.0 - self._eps)
        return torch.log(a) - torch.log1p(-a)

    def _log_abs_det_jacobian(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_mode == "bipolar":
            return torch.log(torch.clamp(1.0 - action.pow(2), min=self._eps))
        return torch.log(torch.clamp(action * (1.0 - action), min=self._eps))

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self._transform(self.dist(obs).mean)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        base_dist = self.dist(obs)
        latent = base_dist.mean if deterministic else base_dist.rsample()
        action = self._transform(latent)
        log_prob = (
            base_dist.log_prob(latent) - self._log_abs_det_jacobian(action)
        ).sum(dim=-1)
        entropy = base_dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        base_dist = self.dist(obs)
        latent = self._inverse_transform(actions)
        action_safe = self._transform(latent)
        log_prob = (
            base_dist.log_prob(latent) - self._log_abs_det_jacobian(action_safe)
        ).sum(dim=-1)
        entropy = base_dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class QMAPPOPolicy:
    def __init__(self, obs_dim: int, cfg: TrainConfig, device: torch.device):
        self.obs_dim = obs_dim
        self.cfg = cfg
        self.device = device
        self.actors = nn.ModuleDict({
            name: QuantumActor(
                obs_dim=len(AGENT_OBS_IDXS[name]),
                n_qubits=cfg.n_qubits,
                depth=cfg.pqc_depth,
                reupload=cfg.reupload,
                hidden_dim=64,
                action_mode=ACTION_MODES[name],
            ).to(device)
            for name in AGENT_NAMES
        })
        self.critic = CentralizedCritic(obs_dim, cfg.hidden_dim).to(device)
        # Mild agent-specific initialization keeps the initial policy away from
        # physically undesirable boundaries without prescribing a final schedule.
        with torch.no_grad():
            for name in AGENT_NAMES:
                last = self.actors[name].head[-1]
                if name == "P2A":
                    last.bias.fill_(0.10)
                elif name == "EB":
                    last.bias.fill_(-0.08)
                elif name == "CHP":
                    last.bias.fill_(0.05)
                else:
                    last.bias.zero_()

        self.actor_opts = {
            name: optim.Adam(self.actors[name].parameters(), lr=cfg.actor_lr)
            for name in AGENT_NAMES
        }
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def _local_obs(self, obs_tensor: torch.Tensor, name: str) -> torch.Tensor:
        idx = torch.as_tensor(AGENT_OBS_IDXS[name], dtype=torch.long, device=obs_tensor.device)
        return obs_tensor.index_select(dim=-1, index=idx)

    @torch.no_grad()
    def act(self, obs_norm: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = []
        log_probs = []
        for name in AGENT_NAMES:
            local = self._local_obs(obs_t, name)
            a, lp, _ = self.actors[name].act(local, deterministic=deterministic)
            actions.append(a.cpu().numpy()[0, 0])
            log_probs.append(lp.cpu().numpy()[0])
        value = self.critic(obs_t).cpu().numpy()[0]
        return np.array(actions, dtype=np.float32), np.array(log_probs, dtype=np.float32), float(value)

    def save(self, path: str, obs_rms: RunningMeanStd, extra: Dict | None = None) -> None:
        payload = {
            "actors": {name: self.actors[name].state_dict() for name in AGENT_NAMES},
            "critic": self.critic.state_dict(),
            "obs_rms": obs_rms.state_dict(),
            "agent_obs_idxs": AGENT_OBS_IDXS,
            "agent_names": AGENT_NAMES,
            "cfg": self.cfg.__dict__,
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load(self, path: str, obs_rms: RunningMeanStd) -> Dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        for name in AGENT_NAMES:
            self.actors[name].load_state_dict(payload["actors"][name])
        self.critic.load_state_dict(payload["critic"])
        obs_rms.load_state_dict(payload["obs_rms"])
        return payload.get("extra", {})


def compute_gae(rewards, dones, values, last_value, gamma, gae_lambda):
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def update_policy(policy: QMAPPOPolicy, batch: Dict[str, np.ndarray], cfg: TrainConfig):
    device = policy.device
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

            actor_loss_total = 0.0
            entropy_total = 0.0
            for agent_i, name in enumerate(AGENT_NAMES):
                local = policy._local_obs(obs_mb, name)
                action_i = act_mb[:, agent_i:agent_i + 1]
                new_lp, entropy = policy.actors[name].evaluate_actions(local, action_i)
                mean = policy.actors[name].action_mean(local)
                ratio = torch.exp(new_lp - old_lp_mb[:, agent_i])
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_mb

                # QMAPPO stabilization terms:
                # 1) avoid early saturation of quantum-inspired actor means;
                # 2) discourage EB mean from staying near the upper boundary;
                # 3) retain a compatible regularization hook for the P2A policy mean.
                if name == "Battery":
                    boundary = torch.relu(torch.abs(mean) - 0.92)
                else:
                    boundary = torch.relu(0.03 - mean) + torch.relu(mean - 0.97)
                mean_reg = cfg.action_mean_reg * (boundary ** 2).mean()
                eb_reg = 0.0
                if name == "EB":
                    eb_reg = cfg.eb_mean_high_reg * (torch.relu(mean - cfg.eb_mean_threshold) ** 2).mean()
                p2a_neg_reg = 0.0
                if name == "P2A":
                    p2a_neg_reg = cfg.p2a_negative_reg * (torch.relu(-mean) ** 2).mean()

                actor_loss = (
                    -torch.min(surr1, surr2).mean()
                    - cfg.ent_coef * entropy.mean()
                    + mean_reg
                    + eb_reg
                    + p2a_neg_reg
                )

                policy.actor_opts[name].zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(policy.actors[name].parameters(), cfg.max_grad_norm)
                policy.actor_opts[name].step()
                actor_loss_total += float(actor_loss.detach().cpu())
                entropy_total += float(entropy.mean().detach().cpu())

            value_pred = policy.critic(obs_mb)
            value_loss = 0.5 * ((value_pred - ret_mb) ** 2).mean()
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
    plt.title("Non-Adaptive Training Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def main():
    cfg = TrainConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("energy_data.npz not found. Run python data_generator.py first.")

    env = EnergyEnv(data_path=DATA_PATH)
    obs_dim = env.observation_space.shape[0]
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    policy = QMAPPOPolicy(obs_dim=obs_dim, cfg=cfg, device=device)

    obs, _ = env.reset(seed=cfg.seed)
    obs_rms.update(obs)
    episode_rewards: List[float] = []
    current_ep_reward = 0.0
    total_steps = 0
    update_idx = 0
    best_recent_reward = -np.inf

    while total_steps < cfg.total_timesteps:
        rollout = {"obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []}

        for _ in range(cfg.rollout_steps):
            obs_rms.update(obs)
            obs_norm = obs_rms.normalize(obs)
            action, log_probs, value = policy.act(obs_norm, deterministic=False)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)

            rollout["obs"].append(obs_norm)
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

        batch = {
            "obs": np.asarray(rollout["obs"], dtype=np.float32),
            "actions": np.asarray(rollout["actions"], dtype=np.float32),
            "log_probs": np.asarray(rollout["log_probs"], dtype=np.float32),
            "advantages": advantages.astype(np.float32),
            "returns": returns.astype(np.float32),
        }
        losses = update_policy(policy, batch, cfg)
        update_idx += 1

        recent = np.mean(episode_rewards[-20:]) if episode_rewards else np.nan
        if np.isfinite(recent) and recent > best_recent_reward:
            best_recent_reward = float(recent)
            policy.save(
                os.path.join(MODEL_DIR, "non_adaptive_energy_model.pt"),
                obs_rms,
                extra={"total_steps": total_steps, "best_recent_reward": best_recent_reward},
            )
        print(
            f"Update {update_idx:04d} | steps {total_steps:7d} | "
            f"recent_ep_reward {recent:9.3f} | "
            f"actor_loss {losses.get('actor_loss', np.nan): .4f} | "
            f"critic_loss {losses.get('critic_loss', np.nan): .4f} | "
            f"entropy {losses.get('entropy', np.nan): .4f}"
        )

        if update_idx % cfg.save_every_updates == 0:
            policy.save(os.path.join(MODEL_DIR, "non_adaptive_latest_model.pt"), obs_rms, extra={"total_steps": total_steps})
            np.save(os.path.join(LOG_DIR, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))

    policy.save(os.path.join(MODEL_DIR, "non_adaptive_latest_model.pt"), obs_rms, extra={"total_steps": total_steps})
    np.save(os.path.join(LOG_DIR, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    plot_training(episode_rewards, os.path.join(FIG_DIR, "non_adaptive_training_curve.png"))
    print(f"Saved best Non-Adaptive checkpoint to {os.path.join(MODEL_DIR, 'non_adaptive_energy_model.pt')}")


if __name__ == "__main__":
    main()
