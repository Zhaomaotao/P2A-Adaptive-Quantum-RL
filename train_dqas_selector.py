

from __future__ import annotations

import os
import json

from dqas_tuning import build_dqas_architectures, save_dqas_architectures, plot_dqas_selection

DATA_PATH = "world_model_data/wm_transitions.npz"
OUT_PATH = os.path.join("models_qtuning", "dqas_architectures.json")


def main():
    payload = build_dqas_architectures(DATA_PATH)
    save_dqas_architectures(payload, OUT_PATH)
    plot_dqas_selection(payload, "figs_qtuning")

    print(f"Saved DQAS-inspired architecture selection to {OUT_PATH}")
    print("Selected architectures:")
    for name, arch in payload["selected"].items():
        print(
            f"  {name:7s} | score={arch['score']:.3f} | {arch['arch_name']:6s} | "
            f"qubits={arch['n_qubits']} depth={arch['pqc_depth']} reupload={arch['reupload']} ent={arch['entanglement']}"
        )

    os.makedirs("logs_qtuning", exist_ok=True)
    with open(os.path.join("logs_qtuning", "dqas_architecture_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
