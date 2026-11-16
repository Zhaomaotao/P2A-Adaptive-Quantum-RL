
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from quantum_tuning import (
    QMetaInitConfig,
    INIT_KEYS,
    build_agent_features,
    build_target_initializers,
    train_meta_initializer,
    save_meta_initializer,
    initializer_outputs_to_dict,
    AGENT_NAMES,
)

MODEL_DIR = "models_qtuning"
FIG_DIR = "figs_qtuning"
OUT_PATH = os.path.join(MODEL_DIR, "qmaml_initializer.pt")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, details = build_agent_features("world_model_data/wm_transitions.npz")
    targets = build_target_initializers(features)
    cfg = QMetaInitConfig(feature_dim=features.shape[1], hidden_dim=64, init_dim=len(INIT_KEYS), seed=42)
    model, losses = train_meta_initializer(features, targets, cfg=cfg, epochs=2000, lr=2e-3, device=device)
    save_meta_initializer(OUT_PATH, model, cfg, features, targets, details)

    with torch.no_grad():
        pred = model(torch.as_tensor(features, dtype=torch.float32, device=device)).cpu().numpy()
    init_dict = initializer_outputs_to_dict(pred)

    print(f"Saved Q-MAML-inspired initializer to {OUT_PATH}")
    print("Agent complexity descriptors:")
    for name in AGENT_NAMES:
        print(f"  {name}: {details[name]}")
    print("Generated initializers:")
    for name in AGENT_NAMES:
        vals = ", ".join([f"{k}={v:.3f}" for k, v in init_dict[name].items()])
        print(f"  {name}: {vals}")

    plt.figure(figsize=(8, 5))
    plt.plot(losses, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Q-MAML-Inspired Initializer Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "qmaml_initializer_loss.png"), dpi=300)
    plt.show()

    x = np.arange(len(AGENT_NAMES))
    width = 0.13
    plt.figure(figsize=(10, 5))
    for j, key in enumerate(INIT_KEYS):
        plt.bar(x + (j - len(INIT_KEYS)/2) * width + width/2, pred[:, j], width=width, label=key)
    plt.xticks(x, AGENT_NAMES)
    plt.ylabel("Generated value")
    plt.title("Q-MAML-Inspired Initializer Outputs by Agent")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "qmaml_initializer_outputs.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
