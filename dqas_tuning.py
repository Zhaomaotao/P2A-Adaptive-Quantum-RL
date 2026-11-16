from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from main_train_qmappo import (
    AGENT_NAMES,
    AGENT_OBS_IDXS,
    ACTION_MODES,
    QuantumActor,
    CentralizedCritic,
    RunningMeanStd,
    TrainConfig,
)
from quantum_tuning import build_agent_features


# Candidate operation/architecture pool. This is the simplified DQAS search space.
# A full DQAS super-circuit would select operations at every placeholder; this
# practical version selects one architecture candidate per heterogeneous agent.
ARCH_POOL: Dict[str, Dict] = {
    "small": {
        "n_qubits": 4,
        "pqc_depth": 1,
        "reupload": 1,
        "entanglement": "weak_ring",
        "description": "shallow PQC for weakly coupled or simple agents",
    },
    "medium": {
        "n_qubits": 6,
        "pqc_depth": 2,
        "reupload": 2,
        "entanglement": "ring",
        "description": "balanced PQC for moderately coupled agents",
    },
    "large": {
        "n_qubits": 8,
        "pqc_depth": 3,
        "reupload": 2,
        "entanglement": "strong_ring",
        "description": "deeper PQC for strongly coupled material-energy agents",
    },
}


def compute_dqas_complexity_scores(data_path: str = "world_model_data/wm_transitions.npz") -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Compute an agent-specific task complexity score in [0, 1].

    The score is based on the same task descriptor used for Q-MAML, including:
      - prior task complexity,
      - action sensitivity estimated from collected transition data,
      - material/heat/storage coupling indicators.

    This makes the DQAS selection explainable: P2A is usually large because it
    affects NH3, downstream HNO3/NH4NO3/Fertilizer inventories, and electric load.
    """
    features, details = build_agent_features(data_path)
    scores: Dict[str, float] = {}
    for i, name in enumerate(AGENT_NAMES):
        f = features[i]
        prior = float(f[6])
        sensitivity = float(f[7])
        material = float(f[8])
        heat = float(f[9])
        storage = float(f[10])
        score = 0.28 * prior + 0.27 * sensitivity + 0.22 * material + 0.13 * heat + 0.10 * storage
        scores[name] = float(np.clip(score, 0.0, 1.0))
        details[name]["dqas_complexity"] = scores[name]
    return scores, details


def select_architecture_name(score: float) -> str:
    if score < 0.46:
        return "small"
    if score < 0.72:
        return "medium"
    return "large"


def build_dqas_architectures(data_path: str = "world_model_data/wm_transitions.npz") -> Dict:
    scores, details = compute_dqas_complexity_scores(data_path)
    selected: Dict[str, Dict] = {}
    for name in AGENT_NAMES:
        arch_name = select_architecture_name(scores[name])
        selected[name] = {
            "arch_name": arch_name,
            "score": float(scores[name]),
            **ARCH_POOL[arch_name],
        }
    return {
        "arch_pool": ARCH_POOL,
        "selected": selected,
        "details": details,
        "agent_names": AGENT_NAMES,
        "agent_obs_idxs": AGENT_OBS_IDXS,
    }


def save_dqas_architectures(payload: Dict, path: str = "models_qtuning/dqas_architectures.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_dqas_architectures(path: str = "models_qtuning/dqas_architectures.json") -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DQAS architecture file not found: {path}. Run python train_dqas_selector.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_dqas_selection(payload: Dict, fig_dir: str = "figs_qtuning") -> None:
    os.makedirs(fig_dir, exist_ok=True)
    names = payload["agent_names"]
    scores = [payload["selected"][n]["score"] for n in names]
    arch_labels = [payload["selected"][n]["arch_name"] for n in names]
    arch_level = {"small": 1, "medium": 2, "large": 3}
    levels = [arch_level[a] for a in arch_labels]

    plt.figure(figsize=(8, 4.5))
    plt.bar(names, scores)
    for i, (s, a) in enumerate(zip(scores, arch_labels)):
        plt.text(i, s + 0.02, a, ha="center", va="bottom", fontsize=10)
    plt.ylim(0, 1.05)
    plt.ylabel("World-model-informed complexity score")
    plt.title("DQAS-Inspired Agent Complexity and Selected PQC Architecture")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "dqas_complexity_scores.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.bar(names, levels)
    plt.yticks([1, 2, 3], ["small", "medium", "large"])
    plt.ylabel("Selected PQC candidate")
    plt.title("DQAS-Inspired PQC Architecture Selection")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "dqas_selected_architectures.png"), dpi=300)
    plt.show()


class DQASTunedQMAPPOPolicy:
    """adaptive quantum policy with DQAS-selected heterogeneous PQC actor structures.

    It preserves the same interface as QMAPPOPolicy so it can reuse the existing
    WM-QMAPPO update functions:
      - act()
      - _local_obs()
      - save() / load()
      - actors, actor_opts, critic, critic_opt
    """

    def __init__(self, obs_dim: int, cfg: TrainConfig, device: torch.device, dqas_payload: Dict):
        self.obs_dim = obs_dim
        self.cfg = cfg
        self.device = device
        self.dqas_payload = dqas_payload

        actors = {}
        for name in AGENT_NAMES:
            arch = dqas_payload["selected"][name]
            actors[name] = QuantumActor(
                obs_dim=len(AGENT_OBS_IDXS[name]),
                n_qubits=int(arch["n_qubits"]),
                depth=int(arch["pqc_depth"]),
                reupload=int(arch["reupload"]),
                hidden_dim=64,
                action_mode=ACTION_MODES[name],
            ).to(device)
        self.actors = nn.ModuleDict(actors)
        self.critic = CentralizedCritic(obs_dim, cfg.hidden_dim).to(device)

        # Mild agent-specific bias remains useful even with Q-MAML.
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

        self.actor_opts = {name: optim.Adam(self.actors[name].parameters(), lr=cfg.actor_lr) for name in AGENT_NAMES}
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
            "dqas_payload": self.dqas_payload,
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load(self, path: str, obs_rms: RunningMeanStd) -> Dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        for name in AGENT_NAMES:
            self.actors[name].load_state_dict(payload["actors"][name])
        self.critic.load_state_dict(payload["critic"])
        obs_rms.load_state_dict(payload["obs_rms"])
        self.dqas_payload = payload.get("dqas_payload", self.dqas_payload)
        return payload.get("extra", {})
