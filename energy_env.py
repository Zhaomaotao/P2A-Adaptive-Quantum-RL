
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass
class EnvConfig:
    episode_hours: int = 24
    random_day: bool = True
    seed: int = 42

    # Battery
    p_batt_max: float = 50.0
    eff_batt_ch: float = 0.95
    eff_batt_dis: float = 0.95
    soc_e_min: float = 0.10
    soc_e_max: float = 1.00

    # NH3 storage, reused original gas tank
    nh3_cap: float = 120.0
    soc_nh3_min: float = 0.05
    soc_nh3_max: float = 1.00

    # Industrial storage capacities
    hno3_cap: float = 120.0
    an_cap: float = 120.0
    fert_cap: float = 160.0

    # Device max power
    p_p2a_max: float = 32.0
    p_chp_max: float = 62.0
    p_gb_max: float = 52.0
    p_eb_max: float = 42.0

    # Device efficiencies
    eff_p2a: float = 0.76       # k-material NH3 per kW-equivalent P2A input
    eff_chp_el: float = 0.35
    eff_chp_th: float = 0.45
    eff_gb: float = 0.90
    eff_eb: float = 0.95

    # Industrial process conversion
    # Medium-complexity equivalent industrial units.
    eta_na: float = 0.88        # NH3 -> HNO3 equivalent
    eta_an: float = 0.90        # NH3 + HNO3 -> NH4NO3 equivalent
    eta_f: float = 0.92         # NH4NO3 -> fertilizer product equivalent

    # Energy demand coefficients for industrial units
    # These are deliberately equivalent coefficients for energy-system scheduling.
    p_na_coef: float = 0.75
    p_an_coef: float = 0.45
    p_f_coef: float = 0.60

    h_na_coef: float = 0.25
    h_an_coef: float = 0.35
    h_f_coef: float = 0.80

    # Reward/cost weights
    cost_scale: float = 220.0
    sell_ratio: float = 0.55
    w_heat_unmet: float = 320.0
    # Larger grid-buy penalties encourage the policy to reduce excessive grid imports.
    w_grid_buy_sq: float = 0.14
    w_grid_buy_lin: float = 8.0
    w_grid_peak: float = 0.18
    grid_peak_threshold: float = 90.0

    w_soc_dev: float = 8.0
    # Industrial continuity and material safety weights.
    # These terms prevent the policy from obtaining a good short-term cost simply by starving
    # the ammonia-based industrial chain.
    w_ind_short: float = 145.0
    w_inv_over: float = 30.0
    w_nh3_low: float = 135.0
    w_an_low: float = 95.0
    w_ind_stop: float = 16.0
    nh3_safe_soc: float = 0.26
    an_safe_soc: float = 0.16
    # Reduced compared with v4 to avoid keeping P2A at full output all day.
    p2a_low_nh3_bonus: float = 0.022

    # Mild high-output penalties prevent P2A and EB from staying at the upper boundary.
    # They are deliberately smaller than industrial-continuity penalties, so the process chain can still run.
    w_p2a_high: float = 45.0
    w_eb_high: float = 320.0
    p2a_high_threshold: float = 0.86
    eb_high_threshold: float = 0.68

    # Small CHP utilization bonus encourages CHP to participate in heat-electric cogeneration
    # instead of letting EB provide almost all heat through grid electricity.
    chp_util_bonus: float = 0.0025

    # Smoother operation penalties. These mainly suppress the saw-tooth battery SOC behavior.
    w_action_smooth: float = 3.0
    w_batt_ramp: float = 8.0
    w_batt_throughput: float = 1.2
    deg_cost_coef: float = 0.018


class EnergyEnv(gym.Env):
    """Single Gymnasium environment with 5-dimensional continuous action.

    Observation vector:
        [
          SOC_e,
          SOC_NH3,
          SOC_HNO3,
          SOC_NH4NO3,
          SOC_F,
          price_el,
          load_el,
          load_th,
          wt_el,
          pvt_el,
          pvt_th,
          q_na_ref,
          q_an_ref,
          q_f_ref
        ]

    Action vector with device-specific bounds:
        a_batt in [-1, 1]: negative discharge, positive charge
        a_p2a  in [0, 1]: P2A power
        a_chp  in [0, 1]: CHP ammonia use level
        a_gb   in [0, 1]: gas/ammonia boiler level
        a_eb   in [0, 1]: electric boiler level
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: Optional[Dict[str, np.ndarray]] = None,
        data_path: str = "energy_data.npz",
        config: Optional[EnvConfig] = None,
        day_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        if data is None:
            loaded = np.load(data_path)
            data = {k: loaded[k] for k in loaded.files}
        self.data = data

        required = ["wt_el", "pvt_el", "pvt_th", "load_el", "load_th", "price_el", "q_na_ref", "q_an_ref", "q_f_ref"]
        for k in required:
            if k not in self.data:
                raise KeyError(f"Missing key '{k}' in data. Please regenerate energy_data.npz with data_generator.py.")

        self.wt_el = np.asarray(self.data["wt_el"], dtype=np.float32)
        self.pvt_el = np.asarray(self.data["pvt_el"], dtype=np.float32)
        self.pvt_th = np.asarray(self.data["pvt_th"], dtype=np.float32)
        self.load_el = np.asarray(self.data["load_el"], dtype=np.float32)
        self.load_th = np.asarray(self.data["load_th"], dtype=np.float32)
        self.price_el = np.asarray(self.data["price_el"], dtype=np.float32)
        self.q_na_ref = np.asarray(self.data["q_na_ref"], dtype=np.float32)
        self.q_an_ref = np.asarray(self.data["q_an_ref"], dtype=np.float32)
        self.q_f_ref = np.asarray(self.data["q_f_ref"], dtype=np.float32)

        self.n_steps_total = len(self.price_el)
        self.hours = int(self.data.get("hours", np.array(24)).item()) if "hours" in self.data else 24
        self.days = self.n_steps_total // self.hours
        self.day_index = day_index

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation values are normalized-ish but not strictly bounded to [0, 1] due to raw price/load/power.
        low = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [1.2, 1.2, 1.2, 1.2, 1.2, 2.5, 200.0, 160.0, 120.0, 100.0, 80.0, 30.0, 30.0, 30.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.t0 = 0
        self.t = 0
        self.k = 0

        self.soc_e = 0.55
        self.soc_nh3 = 0.55
        self.soc_hno3 = 0.35
        self.soc_an = 0.25
        self.soc_f = 0.15
        self.prev_action = np.zeros(5, dtype=np.float32)
        self.prev_p_batt_cmd = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.day_index is not None:
            day = int(np.clip(self.day_index, 0, self.days - 1))
        elif self.cfg.random_day:
            day = int(self.rng.integers(0, max(1, self.days)))
        else:
            day = 0

        self.t0 = day * self.hours
        self.t = self.t0
        self.k = 0

        # Randomize initial storage mildly to improve robustness.
        self.soc_e = float(self.rng.uniform(0.45, 0.65))
        self.soc_nh3 = float(self.rng.uniform(0.45, 0.65))
        self.soc_hno3 = float(self.rng.uniform(0.25, 0.45))
        self.soc_an = float(self.rng.uniform(0.18, 0.38))
        self.soc_f = float(self.rng.uniform(0.10, 0.25))
        self.prev_action = np.zeros(5, dtype=np.float32)
        self.prev_p_batt_cmd = 0.0

        return self._get_obs(), {"day": day}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        cfg = self.cfg
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Current exogenous signals
        idx = self.t % self.n_steps_total
        wt_el = float(self.wt_el[idx])
        pvt_el = float(self.pvt_el[idx])
        pvt_th = float(self.pvt_th[idx])
        load_el = float(self.load_el[idx])
        load_th = float(self.load_th[idx])
        price_el = float(self.price_el[idx])
        q_na_ref = float(self.q_na_ref[idx])
        q_an_ref = float(self.q_an_ref[idx])
        q_f_ref = float(self.q_f_ref[idx])

        a_batt, a_p2a, a_chp, a_gb, a_eb = action
        a_p2a = float(np.clip(a_p2a, 0.0, 1.0))
        a_chp = float(np.clip(a_chp, 0.0, 1.0))
        a_gb = float(np.clip(a_gb, 0.0, 1.0))
        a_eb = float(np.clip(a_eb, 0.0, 1.0))

        # Battery
        p_batt_cmd = float(a_batt * cfg.p_batt_max)
        p_batt_charge = max(p_batt_cmd, 0.0)
        p_batt_dis = max(-p_batt_cmd, 0.0)

        # Limit battery by available SOC.
        max_dis_energy = max(0.0, (self.soc_e - cfg.soc_e_min) * 100.0) * cfg.eff_batt_dis
        p_batt_dis_real = min(p_batt_dis, max_dis_energy)
        p_batt_charge_real = p_batt_charge

        soc_e_next = self.soc_e + (p_batt_charge_real * cfg.eff_batt_ch - p_batt_dis_real / cfg.eff_batt_dis) / 100.0
        soc_e_next = float(np.clip(soc_e_next, cfg.soc_e_min, cfg.soc_e_max))

        # Energy devices
        p_p2a = a_p2a * cfg.p_p2a_max
        nh3_prod = cfg.eff_p2a * p_p2a


        nh3_stock_abs = self.soc_nh3 * cfg.nh3_cap


        nh3_chp_req = a_chp * cfg.p_chp_max
        nh3_gb_req = a_gb * cfg.p_gb_max


        nh3_for_energy_req = nh3_chp_req + nh3_gb_req
        energy_scale = 1.0
        if nh3_for_energy_req > nh3_stock_abs:
            energy_scale = nh3_stock_abs / (nh3_for_energy_req + 1e-8)
        nh3_chp = nh3_chp_req * energy_scale
        nh3_gb = nh3_gb_req * energy_scale

        chp_el = cfg.eff_chp_el * nh3_chp
        chp_th = cfg.eff_chp_th * nh3_chp
        gb_th = cfg.eff_gb * nh3_gb
        p_eb_el = a_eb * cfg.p_eb_max
        eb_th = cfg.eff_eb * p_eb_el


        nh3_available_for_ind = max(0.0, nh3_stock_abs + nh3_prod - nh3_chp - nh3_gb)


        renewable_el = wt_el + pvt_el
        rough_el_margin = renewable_el + chp_el + p_batt_dis_real - p_p2a - p_eb_el - load_el

        el_factor = float(np.clip(0.35 + 0.65 / (1.0 + np.exp(-rough_el_margin / 35.0)), 0.25, 1.0))

        rough_heat_supply = pvt_th + chp_th + gb_th + eb_th
        heat_factor = float(np.clip(0.45 + 0.55 / (1.0 + np.exp(-(rough_heat_supply - load_th) / 28.0)), 0.30, 1.0))

        # Nitric acid unit: NH3 -> HNO3
        q_na_cap_by_nh3 = nh3_available_for_ind
        q_na = min(q_na_ref * el_factor * heat_factor, q_na_cap_by_nh3)
        q_na = max(0.0, q_na)
        nh3_na_use = q_na
        hno3_prod = cfg.eta_na * q_na

        # HNO3 storage after production.
        hno3_abs = self.soc_hno3 * cfg.hno3_cap + hno3_prod

        # NH3 remaining after nitric acid unit.
        nh3_remaining_for_an = max(0.0, nh3_available_for_ind - nh3_na_use)

        # Ammonium nitrate unit: NH3 + HNO3 -> NH4NO3
        # Equivalent stoichiometry uses min of available NH3 and HNO3.
        q_an_possible = min(nh3_remaining_for_an, hno3_abs)
        q_an = min(q_an_ref * el_factor * heat_factor, q_an_possible)
        q_an = max(0.0, q_an)
        nh3_an_use = q_an
        hno3_use = q_an
        an_prod = cfg.eta_an * q_an

        hno3_abs = max(0.0, hno3_abs - hno3_use)
        an_abs = self.soc_an * cfg.an_cap + an_prod

        # Fertilizer finishing: NH4NO3 -> granular nitrogen fertilizer
        q_f_possible = an_abs
        q_f = min(q_f_ref * el_factor * heat_factor, q_f_possible)
        q_f = max(0.0, q_f)
        an_use = q_f
        fert_prod = cfg.eta_f * q_f
        an_abs = max(0.0, an_abs - an_use)
        fert_abs = self.soc_f * cfg.fert_cap + fert_prod

        # Industrial energy demand
        p_ind = cfg.p_na_coef * q_na + cfg.p_an_coef * q_an + cfg.p_f_coef * q_f
        h_ind = cfg.h_na_coef * q_na + cfg.h_an_coef * q_an + cfg.h_f_coef * q_f

        # Update material storages
        nh3_abs_next = nh3_stock_abs + nh3_prod - nh3_chp - nh3_gb - nh3_na_use - nh3_an_use
        soc_nh3_next = float(np.clip(nh3_abs_next / cfg.nh3_cap, cfg.soc_nh3_min, cfg.soc_nh3_max))
        soc_hno3_next = float(np.clip(hno3_abs / cfg.hno3_cap, 0.0, 1.0))
        soc_an_next = float(np.clip(an_abs / cfg.an_cap, 0.0, 1.0))
        soc_f_next = float(np.clip(fert_abs / cfg.fert_cap, 0.0, 1.0))

        # Inventory overflow penalties if generated material exceeds capacity.
        overflow_hno3 = max(0.0, hno3_abs - cfg.hno3_cap)
        overflow_an = max(0.0, an_abs - cfg.an_cap)
        overflow_f = max(0.0, fert_abs - cfg.fert_cap)

        #  Energy balance
        p_gen = wt_el + pvt_el + chp_el + p_batt_dis_real
        p_consume = load_el + p_p2a + p_eb_el + p_batt_charge_real + p_ind
        p_grid = p_consume - p_gen  # positive buy, negative sell

        h_supply = pvt_th + chp_th + gb_th + eb_th
        h_demand = load_th + h_ind
        heat_unmet = max(0.0, h_demand - h_supply)

        # Reward
        cost_buy = price_el * max(p_grid, 0.0)
        revenue_sell = cfg.sell_ratio * price_el * max(-p_grid, 0.0)
        deg_cost = cfg.deg_cost_coef * p_batt_dis_real

        industrial_short = (
            max(0.0, q_na_ref - q_na)
            + max(0.0, q_an_ref - q_an)
            + max(0.0, q_f_ref - q_f)
        )


        nh3_low = max(0.0, cfg.nh3_safe_soc - soc_nh3_next)
        an_low = max(0.0, cfg.an_safe_soc - soc_an_next)
        industrial_stop = (
            float(q_na < 0.15 * max(q_na_ref, 1e-6))
            + float(q_an < 0.15 * max(q_an_ref, 1e-6))
            + float(q_f < 0.15 * max(q_f_ref, 1e-6))
        )

        soc_dev = (
            abs(soc_e_next - 0.55)
            + 0.8 * abs(soc_nh3_next - 0.55)
            + 0.4 * abs(soc_hno3_next - 0.45)
            + 0.4 * abs(soc_an_next - 0.40)
            + 0.2 * abs(soc_f_next - 0.25)
        )

        action_smooth = float(np.sum((action - self.prev_action) ** 2))
        batt_ramp = float(((p_batt_cmd - self.prev_p_batt_cmd) / (cfg.p_batt_max + 1e-8)) ** 2)
        batt_throughput = float((p_batt_charge_real + p_batt_dis_real) / (cfg.p_batt_max + 1e-8))
        grid_peak = max(0.0, max(p_grid, 0.0) - cfg.grid_peak_threshold)


        p2a_high = max(0.0, a_p2a - cfg.p2a_high_threshold)
        eb_high = max(0.0, a_eb - cfg.eb_high_threshold)

        total_cost = (
            cost_buy - revenue_sell
            + deg_cost
            + (cfg.w_heat_unmet * (heat_unmet ** 2)) / cfg.cost_scale
            + (cfg.w_grid_buy_sq * max(p_grid, 0.0) ** 2) / cfg.cost_scale
            + (cfg.w_grid_buy_lin * max(p_grid, 0.0)) / cfg.cost_scale
            + (cfg.w_grid_peak * grid_peak ** 2) / cfg.cost_scale
            + cfg.w_soc_dev * soc_dev
            + cfg.w_ind_short * industrial_short / cfg.cost_scale
            + cfg.w_nh3_low * (nh3_low ** 2)
            + cfg.w_an_low * (an_low ** 2)
            + cfg.w_ind_stop * industrial_stop / cfg.cost_scale
            + cfg.w_inv_over * (overflow_hno3 + overflow_an + overflow_f) / cfg.cost_scale
            + cfg.w_action_smooth * action_smooth
            + cfg.w_batt_ramp * batt_ramp
            + cfg.w_batt_throughput * batt_throughput
            + cfg.w_p2a_high * (p2a_high ** 2)
            + cfg.w_eb_high * (eb_high ** 2)
        )

        reward = -float(total_cost / cfg.cost_scale)

        # Small bonus for useful industrial production and renewable utilization.
        renewable_used = min(max(p_consume, 0.0), wt_el + pvt_el)
        reward += 0.002 * (fert_prod / 10.0) + 0.0005 * renewable_used
        # Encourage useful CHP participation, so heat is not supplied almost entirely by EB/grid electricity.
        reward += cfg.chp_util_bonus * (chp_el + chp_th)
        # Encourage ammonia replenishment when the shared NH3 tank is near the safety margin.
        if self.soc_nh3 < cfg.nh3_safe_soc and a_p2a > 0.05:
            reward += cfg.p2a_low_nh3_bonus * a_p2a

        # Strong penalties for command attempts when storage is low.
        if p_batt_dis > p_batt_dis_real + 1e-6:
            reward -= 0.5
        if energy_scale < 0.999:
            reward -= 0.6 * (1.0 - energy_scale)

        # Commit next state
        self.soc_e = soc_e_next
        self.soc_nh3 = soc_nh3_next
        self.soc_hno3 = soc_hno3_next
        self.soc_an = soc_an_next
        self.soc_f = soc_f_next
        self.prev_action = action.copy()
        self.prev_p_batt_cmd = p_batt_cmd

        self.t += 1
        self.k += 1
        terminated = False
        truncated = self.k >= cfg.episode_hours

        info = {
            # System-level electricity and heat balance quantities.
            "p_grid": p_grid,
            "p_gen": p_gen,
            "p_consume": p_consume,
            "h_supply": h_supply,
            "h_demand": h_demand,
            "heat_unmet": heat_unmet,
            # Device-level physical powers used by the dense multi-panel figures.
            "p_batt_charge": p_batt_charge_real,
            "p_batt_dis": p_batt_dis_real,
            "p_p2a": p_p2a,
            "nh3_prod": nh3_prod,
            "chp_el": chp_el,
            "chp_th": chp_th,
            "gb_th": gb_th,
            "p_eb_el": p_eb_el,
            "eb_th": eb_th,
            # Industrial-chain energy and material flows.
            "p_ind": p_ind,
            "h_ind": h_ind,
            "q_na": q_na,
            "q_an": q_an,
            "q_f": q_f,
            "fert_prod": fert_prod,
            "soc_e": self.soc_e,
            "soc_nh3": self.soc_nh3,
            "soc_hno3": self.soc_hno3,
            "soc_nh4no3": self.soc_an,
            "soc_f": self.soc_f,
            "wt_el": wt_el,
            "pvt_el": pvt_el,
            "pvt_th": pvt_th,
            "load_el": load_el,
            "load_th": load_th,
            "price_el": price_el,
            "el_factor": el_factor,
            "heat_factor": heat_factor,
            "grid_peak": grid_peak,
            "batt_ramp": batt_ramp,
            "batt_throughput": batt_throughput,
            "nh3_low": nh3_low,
            "an_low": an_low,
            "industrial_short": industrial_short,
            "p2a_high_penalty": cfg.w_p2a_high * (p2a_high ** 2),
            "eb_high_penalty": cfg.w_eb_high * (eb_high ** 2),
            "chp_util_bonus": cfg.chp_util_bonus * (chp_el + chp_th),
            "industrial_stop": industrial_stop,
            "reward": reward,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        idx = self.t % self.n_steps_total
        obs = np.array(
            [
                self.soc_e,
                self.soc_nh3,
                self.soc_hno3,
                self.soc_an,
                self.soc_f,
                float(self.price_el[idx]),
                float(self.load_el[idx]),
                float(self.load_th[idx]),
                float(self.wt_el[idx]),
                float(self.pvt_el[idx]),
                float(self.pvt_th[idx]),
                float(self.q_na_ref[idx]),
                float(self.q_an_ref[idx]),
                float(self.q_f_ref[idx]),
            ],
            dtype=np.float32,
        )
        return obs
