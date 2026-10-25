"""
plot_training.py

Reads fold log CSVs and plots training curves for all 5 folds.

USAGE:
    python plot_training.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

CKPT_DIR = "checkpoints"
N_FOLDS  = 5
OUT_FILE = "training_curves.png"


def plot():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "2.5D EfficientNet-B0 — 5-Fold CV Training\n"
        "Tumor Resectability | UCSF-PDGM (GTR vs STR)\n"
        "Focal Loss (α=0.50) | Patience=20 | WeightedSampler",
        fontsize=13, fontweight="bold"
    )

    colors = cm.tab10(np.linspace(0, 0.5, N_FOLDS))
    loaded = 0

    for fold in range(1, N_FOLDS + 1):
        log_path = os.path.join(CKPT_DIR, f"log_fold{fold}.csv")
        if not os.path.isfile(log_path):
            print(f"Fold {fold} log not found — skipping")
            continue

        df    = pd.read_csv(log_path)
        color = colors[fold - 1]
        loaded += 1

        axes[0, 0].plot(df["epoch"], df["val_loss"],
                        label=f"Fold {fold}", color=color, linewidth=1.5)
        axes[0, 1].plot(df["epoch"], df["val_auc"],
                        label=f"Fold {fold}", color=color, linewidth=1.5)
        axes[1, 0].plot(df["epoch"], df["val_sensitivity"],
                        label=f"Fold {fold}", color=color, linewidth=1.5)
        axes[1, 1].plot(df["epoch"], df["val_specificity"],
                        label=f"Fold {fold}", color=color, linewidth=1.5)

    if loaded == 0:
        print("No fold logs found. Run train.py first.")
        return

    for ax, title, ylabel, ylim in zip(
        axes.flat,
        ["Val Loss (Focal)", "Val AUC-ROC", "Val Sensitivity", "Val Specificity"],
        ["Loss", "AUC", "Sensitivity", "Specificity"],
        [None, (0, 1), (0, 1), (0, 1)],
    ):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150)
    plt.show()
    print(f"Saved → '{OUT_FILE}'")

    cv_path = os.path.join(CKPT_DIR, "cv_results.csv")
    if os.path.isfile(cv_path):
        cv_df = pd.read_csv(cv_path)
        print(f"\nCV Summary:")
        for _, row in cv_df.iterrows():
            print(f"  Fold {int(row['fold'])}: AUC = {row['val_auc']:.4f}")
        print(f"  Mean : {cv_df['val_auc'].mean():.4f}")
        print(f"  Std  : {cv_df['val_auc'].std():.4f}")


if __name__ == "__main__":
    plot()
