import numpy as np


def daily_profile(hours: int, peak_hour: float, width: float) -> np.ndarray:
    """Smooth bell-shaped daily profile normalized to [0, 1]."""
    h = np.arange(hours)
    dist = np.minimum(np.abs(h - peak_hour), hours - np.abs(h - peak_hour))
    y = np.exp(-(dist ** 2) / (2 * width ** 2))
    return y / (y.max() + 1e-8)


def generate_energy_data(days: int = 300, hours: int = 24, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    T = days * hours
    hour = np.tile(np.arange(hours), days)
    day = np.repeat(np.arange(days), hours)

    solar_shape = np.sin(np.pi * (hour - 6) / 12)
    solar_shape = np.clip(solar_shape, 0, None)

    pvt_el = 45 * solar_shape * rng.uniform(0.75, 1.15, T)
    pvt_th = 35 * solar_shape * rng.uniform(0.75, 1.15, T)

    wind_daily = 0.65 + 0.25 * np.sin(2 * np.pi * (hour + 3) / 24)
    wind_season = 1.0 + 0.15 * np.sin(2 * np.pi * day / 365)
    wind = 35 * wind_daily * wind_season * rng.uniform(0.65, 1.35, T)
    wind = np.clip(wind, 0, 65)

    load_el = (
        70
        + 18 * np.sin(2 * np.pi * (hour - 8) / 24)
        + 10 * np.sin(4 * np.pi * (hour - 7) / 24)
        + rng.normal(0, 5, T)
    )
    load_el = np.clip(load_el, 40, 120)

    load_th = (
        55
        + 16 * np.sin(2 * np.pi * (hour - 5) / 24)
        + 8 * np.cos(2 * np.pi * day / 365)
        + rng.normal(0, 4, T)
    )
    load_th = np.clip(load_th, 30, 90)

    load_gas = 25 + 6 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 2, T)
    load_gas = np.clip(load_gas, 10, 45)

    price_el = np.zeros(T, dtype=float)
    for h in range(hours):
        if 0 <= h <= 6:
            price_el[hour == h] = 0.55
        elif 7 <= h <= 10:
            price_el[hour == h] = 0.95
        elif 11 <= h <= 16:
            price_el[hour == h] = 0.75
        elif 17 <= h <= 21:
            price_el[hour == h] = 1.35
        else:
            price_el[hour == h] = 0.85
    price_el *= rng.uniform(0.9, 1.1, T)
    price_el = np.clip(price_el, 0.4, 1.8)

    price_gas = 0.4 + 0.6 * (price_el - price_el.min()) / (price_el.max() - price_el.min() + 1e-8)
    price_gas = np.clip(price_gas, 0.4, 1.0)

    # Internal industrial reference rates, not extra agent actions.
    ind_shape = 0.65 + 0.25 * daily_profile(hours, peak_hour=14, width=5)[hour]
    weekly = 0.9 + 0.1 * np.sin(2 * np.pi * day / 7)

    q_na_ref = 8.0 * ind_shape * weekly * rng.uniform(0.9, 1.1, T)   # desired HNO3 production rate
    q_an_ref = 7.0 * ind_shape * weekly * rng.uniform(0.9, 1.1, T)   # desired NH4NO3 synthesis rate
    q_f_ref = 6.0 * ind_shape * weekly * rng.uniform(0.9, 1.1, T)    # desired fertilizer finishing rate
    fertilizer_demand = 4.5 * ind_shape * weekly * rng.uniform(0.85, 1.15, T)

    np.savez(
        "energy_data.npz",
        wt_el=wind.astype(np.float32),
        wind=wind.astype(np.float32),
        pvt_el=pvt_el.astype(np.float32),
        pvt_th=pvt_th.astype(np.float32),
        load_el=load_el.astype(np.float32),
        load_th=load_th.astype(np.float32),
        load_gas=load_gas.astype(np.float32),
        price_el=price_el.astype(np.float32),
        price_gas=price_gas.astype(np.float32),
        q_na_ref=q_na_ref.astype(np.float32),
        q_an_ref=q_an_ref.astype(np.float32),
        q_f_ref=q_f_ref.astype(np.float32),
        fertilizer_demand=fertilizer_demand.astype(np.float32),
    )

    print(f"Saved energy_data.npz with {T} hourly steps.")
    print("Keys include original energy signals plus q_na_ref, q_an_ref, q_f_ref, fertilizer_demand.")


if __name__ == "__main__":
    generate_energy_data(days=300, hours=24, seed=42)
