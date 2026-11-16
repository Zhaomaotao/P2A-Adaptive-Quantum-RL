

from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from energy_env import EnergyEnv
from main_train_qmappo import QMAPPOPolicy, RunningMeanStd, TrainConfig


DATA_PATH = "energy_data.npz"
MODEL_PATH = os.path.join("models_non_adaptive", "non_adaptive_energy_model.pt")
FIG_DIR = "figs_non_adaptive"
os.makedirs(FIG_DIR, exist_ok=True)



def save_records_npz(records, path: str) -> None:
    if not records:
        return
    keys = [k for k in records[0] if k not in {"action"}]
    payload = {k: np.asarray([r[k] for r in records]) for k in keys}
    payload["actions"] = np.asarray([r["action"] for r in records], dtype=np.float32)
    np.savez_compressed(path, **payload)


def plot_records(records, fig_dir: str):
    t = np.array([r["t"] for r in records])

    plt.figure(figsize=(9, 5))
    for key, label in [
        ("soc_e", "Battery SOC"),
        ("soc_nh3", "NH3 SOC"),
        ("soc_hno3", "HNO3 SOC"),
        ("soc_nh4no3", "NH4NO3 SOC"),
        ("soc_f", "Fertilizer SOC"),
    ]:
        plt.plot(t, [r[key] for r in records], label=label, linewidth=1.5)
    plt.xlabel("Hour")
    plt.ylabel("SOC / inventory level")
    plt.title("Non-Adaptive: Storage and Industrial Inventory States")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "non_adaptive_test_storage_states.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(9, 5))
    for key, label in [
        ("q_na", "Nitric acid unit"),
        ("q_an", "Ammonium nitrate unit"),
        ("q_f", "Fertilizer finishing"),
    ]:
        plt.plot(t, [r[key] for r in records], label=label, linewidth=1.5)
    plt.xlabel("Hour")
    plt.ylabel("Equivalent process rate")
    plt.title("Non-Adaptive: Internal Industrial Process Rates")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "non_adaptive_test_industrial_rates.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(t, [r["p_grid"] for r in records], label="Grid power (+buy, -sell)", linewidth=1.5)
    plt.plot(t, [r["heat_unmet"] for r in records], label="Unmet heat", linewidth=1.5)
    plt.xlabel("Hour")
    plt.ylabel("kW")
    plt.title("Non-Adaptive: Grid Exchange and Heat Unmet")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "non_adaptive_test_grid_heat.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(9, 5))
    action_names = ["a_batt", "a_p2a", "a_chp", "a_gb", "a_eb"]
    actions = np.array([r["action"] for r in records])
    for i, name in enumerate(action_names):
        plt.plot(t, actions[:, i], label=name, linewidth=1.5)
    plt.xlabel("Hour")
    plt.ylabel("Action")
    plt.title("Non-Adaptive: Agent Actions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "non_adaptive_test_actions.png"), dpi=300)
    plt.show()


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("energy_data.npz not found. Run python data_generator.py first.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Non-Adaptive model not found. Run python main_train_qmappo.py first.")

    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = EnergyEnv(data_path=DATA_PATH, day_index=0)
    obs_dim = env.observation_space.shape[0]
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    policy = QMAPPOPolicy(obs_dim=obs_dim, cfg=cfg, device=device)
    policy.load(MODEL_PATH, obs_rms)

    obs, _ = env.reset(seed=123)
    records = []
    total_reward = 0.0

    for step in range(env.cfg.episode_hours):
        obs_norm = obs_rms.normalize(obs)
        action, _log_probs, _value = policy.act(obs_norm, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)

        records.append({
            "t": step,
            "reward": float(reward),
            "action": action.copy(),
            "p_grid": info.get("p_grid", np.nan),
            "p_gen": info.get("p_gen", np.nan),
            "p_consume": info.get("p_consume", np.nan),
            "h_supply": info.get("h_supply", np.nan),
            "h_demand": info.get("h_demand", np.nan),
            "p_batt_charge": info.get("p_batt_charge", np.nan),
            "p_batt_dis": info.get("p_batt_dis", np.nan),
            "p_p2a": info.get("p_p2a", np.nan),
            "nh3_prod": info.get("nh3_prod", np.nan),
            "chp_el": info.get("chp_el", np.nan),
            "chp_th": info.get("chp_th", np.nan),
            "gb_th": info.get("gb_th", np.nan),
            "p_eb_el": info.get("p_eb_el", np.nan),
            "eb_th": info.get("eb_th", np.nan),
            "p_ind": info.get("p_ind", np.nan),
            "h_ind": info.get("h_ind", np.nan),
            "heat_unmet": info.get("heat_unmet", np.nan),
            "soc_e": info.get("soc_e", np.nan),
            "soc_nh3": info.get("soc_nh3", np.nan),
            "soc_hno3": info.get("soc_hno3", np.nan),
            "soc_nh4no3": info.get("soc_nh4no3", np.nan),
            "soc_f": info.get("soc_f", np.nan),
            "q_na": info.get("q_na", np.nan),
            "q_an": info.get("q_an", np.nan),
            "q_f": info.get("q_f", np.nan),
            "fert_prod": info.get("fert_prod", np.nan),
            "wt_el": info.get("wt_el", np.nan),
            "pvt_el": info.get("pvt_el", np.nan),
            "pvt_th": info.get("pvt_th", np.nan),
            "load_el": info.get("load_el", np.nan),
            "load_th": info.get("load_th", np.nan),
            "price_el": info.get("price_el", np.nan),
            "industrial_stop": info.get("industrial_stop", 0.0),
        })

        obs = next_obs
        if done:
            break

    print(f"Total eval reward over {len(records)} steps: {total_reward:.4f}")
    record_path = os.path.join(FIG_DIR, "non_adaptive_test_records.npz")
    save_records_npz(records, record_path)
    print(f"Saved test records to {record_path}")
    plot_records(records, FIG_DIR)


if __name__ == "__main__":
    main()
