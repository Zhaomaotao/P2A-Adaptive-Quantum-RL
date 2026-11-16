
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from dqas_tuning import DQASTunedQMAPPOPolicy, load_dqas_architectures
from energy_env import EnergyEnv
from main_train_mappo import MAPPOPolicy, RunningMeanStd as MAPRunningMeanStd, TrainConfig as MAPPOConfig
from main_train_qmappo import QMAPPOPolicy, RunningMeanStd, TrainConfig
from main_train_wm_qmappo_qmaml_dqas import WMQMAPPOQMAMLDQASConfig

DATA_PATH = "energy_data.npz"
RESULT_DIR = "results_compare"
AGENT_ACTION_NAMES = ["a_batt", "a_p2a", "a_chp", "a_gb", "a_eb"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    model_path: str
    cfg_factory: Callable
    policy_type: str
    dqas_arch_path: Optional[str] = None
    log_path: Optional[str] = None


METHODS: List[MethodSpec] = [
    MethodSpec(
        name="MAPPO",
        model_path=os.path.join("models_mappo", "mappo_energy_model.pt"),
        cfg_factory=MAPPOConfig,
        policy_type="mappo",
        log_path=os.path.join("logs_mappo", "episode_rewards.npy"),
    ),
    MethodSpec(
        name="Non-Adaptive",
        model_path=os.path.join("models_non_adaptive", "non_adaptive_energy_model.pt"),
        cfg_factory=TrainConfig,
        policy_type="fixed_quantum",
        log_path=os.path.join("logs_non_adaptive", "episode_rewards.npy"),
    ),
    MethodSpec(
        name="Proposed",
        model_path=os.path.join("models_proposed", "proposed_energy_model.pt"),
        cfg_factory=WMQMAPPOQMAMLDQASConfig,
        policy_type="proposed",
        dqas_arch_path=os.path.join("models_qtuning", "dqas_architectures.json"),
        log_path=os.path.join("logs_proposed", "episode_rewards.npy"),
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the three paper methods on identical days.")
    p.add_argument("--start-day", type=int, default=0)
    p.add_argument("--num-days", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026, help="Base seed for identical initial states.")
    p.add_argument("--result-dir", type=str, default=RESULT_DIR)
    p.add_argument(
        "--allow-unordered",
        action="store_true",
        help="Write diagnostics even when the required mean reward order is not satisfied.",
    )
    return p.parse_args()


def _cfg_from_checkpoint(spec: MethodSpec, device: torch.device):
    payload = torch.load(spec.model_path, map_location=device, weights_only=False)
    cfg = spec.cfg_factory()
    for key, value in payload.get("cfg", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg, payload


def build_policy(spec: MethodSpec, obs_dim: int, device: torch.device):
    cfg, payload = _cfg_from_checkpoint(spec, device)
    if spec.policy_type == "mappo":
        obs_rms = MAPRunningMeanStd(shape=(obs_dim,))
        policy = MAPPOPolicy(obs_dim=obs_dim, cfg=cfg, device=device)
    elif spec.policy_type == "fixed_quantum":
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        policy = QMAPPOPolicy(obs_dim=obs_dim, cfg=cfg, device=device)
    elif spec.policy_type == "proposed":
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        dqas_payload = payload.get("dqas_payload") or payload.get("dqas")
        if dqas_payload is None:
            if not spec.dqas_arch_path or not os.path.exists(spec.dqas_arch_path):
                raise FileNotFoundError("DQAS architecture metadata is missing from checkpoint and JSON file.")
            dqas_payload = load_dqas_architectures(spec.dqas_arch_path)
        policy = DQASTunedQMAPPOPolicy(obs_dim=obs_dim, cfg=cfg, device=device, dqas_payload=dqas_payload)
    else:
        raise ValueError(f"Unknown policy type: {spec.policy_type}")
    policy.load(spec.model_path, obs_rms)
    return policy, obs_rms


def evaluate_policy(spec: MethodSpec, day_index: int, initial_seed: int, device: torch.device) -> Dict[str, float | str]:
    env = EnergyEnv(data_path=DATA_PATH, day_index=day_index)
    obs_dim = env.observation_space.shape[0]
    policy, obs_rms = build_policy(spec, obs_dim, device)
    obs, _ = env.reset(seed=initial_seed)

    series: Dict[str, List[float]] = {
        "reward": [], "p_grid": [], "heat_unmet": [], "industrial_stop": [],
        "industrial_short": [], "soc_nh3": [], "soc_an": [], "soc_f": [],
        "wt_el": [], "pvt_el": [], "load_el": [], "p_p2a": [], "chp_el": [],
    }
    actions: List[np.ndarray] = []

    for _ in range(env.cfg.episode_hours):
        obs_norm = obs_rms.normalize(obs)
        action, _, _ = policy.act(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        series["reward"].append(float(reward))
        series["p_grid"].append(float(info.get("p_grid", np.nan)))
        series["heat_unmet"].append(float(info.get("heat_unmet", np.nan)))
        series["industrial_stop"].append(float(info.get("industrial_stop", 0.0) > 0.0))
        series["industrial_short"].append(float(info.get("industrial_short", 0.0)))
        series["soc_nh3"].append(float(info.get("soc_nh3", np.nan)))
        series["soc_an"].append(float(info.get("soc_nh4no3", np.nan)))
        series["soc_f"].append(float(info.get("soc_f", np.nan)))
        series["wt_el"].append(float(info.get("wt_el", np.nan)))
        series["pvt_el"].append(float(info.get("pvt_el", np.nan)))
        series["load_el"].append(float(info.get("load_el", np.nan)))
        series["p_p2a"].append(float(info.get("p_p2a", np.nan)))
        series["chp_el"].append(float(info.get("chp_el", np.nan)))
        actions.append(np.asarray(action, dtype=np.float32))

        if bool(terminated or truncated):
            break

    a = np.asarray(actions, dtype=np.float64)
    r = np.asarray(series["reward"], dtype=np.float64)
    grid = np.asarray(series["p_grid"], dtype=np.float64)
    heat = np.asarray(series["heat_unmet"], dtype=np.float64)
    renewable = np.asarray(series["wt_el"], dtype=np.float64) + np.asarray(series["pvt_el"], dtype=np.float64)
    load = np.asarray(series["load_el"], dtype=np.float64)

    mean_action_variation = float(np.mean(np.abs(np.diff(a, axis=0)))) if a.shape[0] > 1 else 0.0
    renewable_used_proxy = np.minimum(np.maximum(load, 0.0), np.maximum(renewable, 0.0))
    renewable_utilization = float(renewable_used_proxy.sum() / max(renewable.sum(), 1e-8))

    result: Dict[str, float | str] = {
        "method": spec.name,
        "day_index": float(day_index),
        "initial_seed": float(initial_seed),
        "total_reward": float(r.sum()),
        "mean_step_reward": float(r.mean()),
        "grid_purchase_MWh": float(np.maximum(grid, 0.0).sum() / 10.0),
        "grid_export_MWh": float(np.maximum(-grid, 0.0).sum() / 10.0),
        "peak_grid_purchase_MW": float(np.maximum(grid, 0.0).max() / 10.0),
        "total_unmet_heat_MWh": float(heat.sum() / 10.0),
        "industrial_stop_hours": float(np.sum(series["industrial_stop"])),
        "industrial_shortage": float(np.sum(series["industrial_short"])),
        "mean_action_variation_pu": mean_action_variation,
        "min_nh3_soc": float(np.nanmin(series["soc_nh3"])),
        "min_nh4no3_soc": float(np.nanmin(series["soc_an"])),
        "final_fertilizer_soc": float(series["soc_f"][-1]),
        "renewable_utilization_ratio": renewable_utilization,
        "mean_p2a_power_MW": float(np.nanmean(series["p_p2a"]) / 10.0),
        "mean_chp_electric_power_MW": float(np.nanmean(series["chp_el"]) / 10.0),
    }
    for idx, name in enumerate(AGENT_ACTION_NAMES):
        result[f"{name}_mean"] = float(a[:, idx].mean())
        result[f"{name}_std"] = float(a[:, idx].std())
    return result


def summarize(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    metrics = [k for k in rows[0] if k not in {"method", "day_index", "initial_seed"}]
    output: List[Dict[str, float | str]] = []
    for spec in METHODS:
        sub = [r for r in rows if r["method"] == spec.name]
        if not sub:
            continue
        out: Dict[str, float | str] = {"method": spec.name, "n_episodes": float(len(sub))}
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in sub], dtype=np.float64)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(out)
    return output


def write_csv(path: str, rows: List[Dict[str, float | str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def check_reward_order(summary: List[Dict[str, float | str]]) -> tuple[bool, Dict[str, float]]:
    reward_map = {str(r["method"]): float(r["total_reward_mean"]) for r in summary}
    required = {"MAPPO", "Non-Adaptive", "Proposed"}
    passed = required.issubset(reward_map) and reward_map["Proposed"] > reward_map["Non-Adaptive"] > reward_map["MAPPO"]
    return passed, reward_map


def main() -> None:
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("energy_data.npz not found. Run python data_generator.py first.")
    missing = [m.model_path for m in METHODS if not os.path.exists(m.model_path)]
    if missing:
        raise FileNotFoundError("Missing trained checkpoints:\n  " + "\n  ".join(missing))

    data = np.load(DATA_PATH)
    total_days = len(data["price_el"]) // 24
    start = int(np.clip(args.start_day, 0, max(0, total_days - 1)))
    days = list(range(start, min(total_days, start + max(1, args.num_days))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {len(days)} identical days with device={device}.")

    rows: List[Dict[str, float | str]] = []
    for day in days:
        initial_seed = args.seed + day
        for spec in METHODS:
            row = evaluate_policy(spec, day, initial_seed, device)
            rows.append(row)
            print(
                f"{spec.name:12s} day={day:03d} reward={float(row['total_reward']):9.3f} "
                f"heat={float(row['total_unmet_heat_MWh']):7.3f} MWh "
                f"stop={float(row['industrial_stop_hours']):4.1f} h"
            )

    summary = summarize(rows)
    write_csv(os.path.join(args.result_dir, "metrics_by_episode.csv"), rows)
    write_csv(os.path.join(args.result_dir, "metrics_summary.csv"), summary)

    passed, reward_map = check_reward_order(summary)
    status = {
        "required_order": "Proposed > Non-Adaptive > MAPPO",
        "passed": passed,
        "mean_total_reward": reward_map,
        "note": "Values are measured outputs and are never altered by this script.",
    }
    with open(os.path.join(args.result_dir, "ranking_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print("\nRequired mean-reward order: Proposed > Non-Adaptive > MAPPO")
    print("PASS" if passed else "FAIL — no figure values were changed.")
    print(f"Saved evaluation outputs to {args.result_dir}")

    if not passed and not args.allow_unordered:
        raise RuntimeError(
            "Required reward order is not satisfied. The strict submission gate stopped here. "
            "Inspect training stability/world-model validation, retrain all methods fairly, then rerun evaluation."
        )


if __name__ == "__main__":
    main()
