"""
train.py

2.5D EfficientNet-B0 with 5-fold cross validation.

Changes in this version:
  - Patience: 10 → 20  (allows full LR restart cycles)
  - Focal alpha: 0.65 → 0.50  (equal weight, fixes GTR bias)

USAGE:
    python train.py --reset
    python train.py          (resume — skips completed folds)
"""

import os
import csv
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse

from dataset import TumorResectabilityDataset
from model   import build_model

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join("Data", "step7_normalized_copy")
LABELS_CSV   = "labels.csv"
CKPT_DIR     = "checkpoints"
LOG_CSV      = "training_log.csv"

N_FOLDS      = 5
EPOCHS       = 80
BATCH_SIZE   = 8
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 20          # increased from 10 → allows full LR restart cycles
NUM_WORKERS  = 0
SEED         = 42
TEST_RATIO   = 0.15

# Focal Loss — alpha=0.50 gives equal weight to GTR and STR
# Previously 0.65 was biasing toward GTR causing low specificity
FOCAL_ALPHA  = 0.50
FOCAL_GAMMA  = 2.0
# ─────────────────────────────────────────────────────────────────────────────


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss.
    alpha=0.50 treats GTR and STR equally.
    gamma=2.0 focuses training on hard/uncertain examples.
    """
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        at  = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (at * (1.0 - pt) ** self.gamma * bce).mean()


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sensitivity  = tp / (tp + fn + 1e-8)
    specificity  = tn / (tn + fp + 1e-8)
    balanced_acc = (sensitivity + specificity) / 2.0
    return {
        "auc":          round(auc,          4),
        "balanced_acc": round(balanced_acc, 4),
        "sensitivity":  round(sensitivity,  4),
        "specificity":  round(specificity,  4),
    }


def train_fold(train_csv, val_csv, fold, epochs, batch_size):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_ds = TumorResectabilityDataset(train_csv, DATA_DIR, augment=True)
    val_ds   = TumorResectabilityDataset(val_csv,   DATA_DIR, augment=False)

    # WeightedRandomSampler — balanced batches every epoch
    sampler = WeightedRandomSampler(
        weights     = train_ds.get_sample_weights(),
        num_samples = len(train_ds),
        replacement = True,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=use_amp)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=use_amp)

    model     = build_model(pretrained=True, dropout_p=0.4).to(device)
    criterion = BinaryFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimiser = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # CosineAnnealingWarmRestarts stepped per batch
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser,
        T_0    = steps_per_epoch * 10,
        T_mult = 1,
        eta_min= 1e-6,
    )
    scaler = GradScaler(enabled=use_amp)

    best_ckpt     = os.path.join(CKPT_DIR, f"best_fold{fold}.pth")
    best_auc      = 0.0
    epochs_no_imp = 0
    log_rows      = []

    print(f"\n{'='*60}")
    print(f"FOLD {fold}/{N_FOLDS} | Train:{len(train_ds)} Val:{len(val_ds)} | "
          f"device:{device} | patience:{PATIENCE} | alpha:{FOCAL_ALPHA}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss, train_probs, train_labels = 0.0, [], []
        train_bar = tqdm(train_loader,
                         desc=f"F{fold} Ep{epoch:03d}/{epochs} [Train]",
                         leave=False, unit="batch")
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimiser.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            train_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            train_probs.extend(probs.tolist())
            train_labels.extend(labels.cpu().numpy().flatten().tolist())
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_ds)
        train_m     = compute_metrics(
            np.array(train_labels, dtype=int), np.array(train_probs)
        )

        # Validate
        model.eval()
        val_loss, val_probs, val_labels = 0.0, [], []
        val_bar = tqdm(val_loader,
                       desc=f"F{fold} Ep{epoch:03d}/{epochs} [Val]  ",
                       leave=False, unit="batch")
        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                with autocast(enabled=use_amp):
                    logits = model(images)
                    loss   = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_probs.extend(probs.tolist())
                val_labels.extend(labels.cpu().numpy().flatten().tolist())
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss /= len(val_ds)
        val_m     = compute_metrics(
            np.array(val_labels, dtype=int), np.array(val_probs)
        )
        elapsed = time.time() - t0
        lr_now  = optimiser.param_groups[0]["lr"]

        print(
            f"F{fold} Ep{epoch:03d}/{epochs} | "
            f"Train loss:{train_loss:.4f} AUC:{train_m['auc']:.3f} | "
            f"Val loss:{val_loss:.4f} AUC:{val_m['auc']:.3f} "
            f"Sens:{val_m['sensitivity']:.3f} Spec:{val_m['specificity']:.3f} | "
            f"LR:{lr_now:.2e} | {elapsed:.0f}s"
        )

        log_rows.append({
            "fold":               fold,
            "epoch":              epoch,
            "train_loss":         round(train_loss,            6),
            "train_auc":          train_m["auc"],
            "train_sensitivity":  train_m["sensitivity"],
            "train_specificity":  train_m["specificity"],
            "train_balanced_acc": train_m["balanced_acc"],
            "val_loss":           round(val_loss,              6),
            "val_auc":            val_m["auc"],
            "val_sensitivity":    val_m["sensitivity"],
            "val_specificity":    val_m["specificity"],
            "val_balanced_acc":   val_m["balanced_acc"],
            "lr":                 round(lr_now,                8),
        })

        if val_m["auc"] > best_auc:
            best_auc      = val_m["auc"]
            epochs_no_imp = 0
            torch.save({
                "fold":        fold,
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     best_auc,
            }, best_ckpt)
            print(f"  ✓ Saved best fold {fold} model (val AUC: {best_auc:.4f})")
        else:
            epochs_no_imp += 1
            print(f"  No improvement ({epochs_no_imp}/{PATIENCE})")

        fold_log = os.path.join(CKPT_DIR, f"log_fold{fold}.csv")
        with open(fold_log, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

        if epochs_no_imp >= PATIENCE:
            print(f"\n⏹ Early stopping fold {fold} — "
                  f"no improvement for {PATIENCE} epochs.")
            break

    print(f"\nFold {fold} complete. Best val AUC: {best_auc:.4f}")
    return best_auc


def run_cv(epochs, batch_size, reset):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    df = pd.read_csv(LABELS_CSV)

    if reset or not os.path.isfile("test_split.csv"):
        trainval_ids, test_ids = train_test_split(
            df["patient_id"].tolist(),
            test_size    = TEST_RATIO,
            stratify     = df["label"].tolist(),
            random_state = SEED,
        )
        trainval_df = df[df["patient_id"].isin(trainval_ids)].reset_index(drop=True)
        test_df     = df[df["patient_id"].isin(test_ids)]
        trainval_df.to_csv("trainval_split.csv", index=False)
        test_df.to_csv("test_split.csv",         index=False)
        print(f"Test set held out: {len(test_df)} patients "
              f"(GTR={int(test_df['label'].sum())}, "
              f"STR={len(test_df)-int(test_df['label'].sum())})")
        print(f"⚠  test_split.csv is held out — do NOT touch until all folds complete.\n")
    else:
        trainval_df = pd.read_csv("trainval_split.csv")
        print(f"Reusing existing test split.\n")

    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    patients  = trainval_df["patient_id"].values
    labels    = trainval_df["label"].values
    fold_aucs = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(patients, labels), start=1):
        ckpt_path = os.path.join(CKPT_DIR, f"best_fold{fold}.pth")
        if not reset and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            print(f"Fold {fold} already complete — "
                  f"best AUC: {ckpt['val_auc']:.4f} (skipping)")
            fold_aucs.append(ckpt["val_auc"])
            continue

        train_df_fold = trainval_df.iloc[tr_idx]
        val_df_fold   = trainval_df.iloc[vl_idx]
        train_df_fold.to_csv(f"fold{fold}_train.csv", index=False)
        val_df_fold.to_csv(f"fold{fold}_val.csv",     index=False)

        best_auc = train_fold(
            train_csv  = f"fold{fold}_train.csv",
            val_csv    = f"fold{fold}_val.csv",
            fold       = fold,
            epochs     = epochs,
            batch_size = batch_size,
        )
        fold_aucs.append(best_auc)

    print(f"\n{'='*60}")
    print(f"5-FOLD CV COMPLETE")
    for i, auc in enumerate(fold_aucs, start=1):
        print(f"  Fold {i}: AUC = {auc:.4f}")
    print(f"  Mean AUC : {np.mean(fold_aucs):.4f}")
    print(f"  Std  AUC : {np.std(fold_aucs):.4f}")
    print(f"\nNext step → run: python test.py")

    pd.DataFrame({
        "fold":    list(range(1, N_FOLDS + 1)),
        "val_auc": fold_aucs,
    }).to_csv(os.path.join(CKPT_DIR, "cv_results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,  default=EPOCHS)
    parser.add_argument("--batch_size", type=int,  default=BATCH_SIZE)
    parser.add_argument("--reset",      action="store_true")
    args = parser.parse_args()
    run_cv(args.epochs, args.batch_size, args.reset)
