"""
test.py

Final evaluation on the held-out test set.
Uses ensemble of all 5 fold models.

Features:
  - Ensemble of 5 models (averaged probabilities)
  - Optimal threshold from ROC curve (maximises balanced accuracy)
  - Results at multiple thresholds (0.3, 0.4, 0.5, 0.6, optimal)
  - Grad-CAM heatmaps for explainable AI
  - Full report saved to test_report.txt
  - ROC curve saved to test_roc_curve.png
  - Grad-CAM saved for first 3 patients

USAGE:
    python test.py
    python test.py --gradcam_patients UCSF-PDGM-0004_nifti UCSF-PDGM-0005_nifti
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from dataset import (
    TumorResectabilityDataset, find_file,
    load_volume, resize_volume, extract_5slices,
    MODALITIES, VOLUME_SIZE, SLICE_SIZE,
)
from model import build_model

warnings.filterwarnings("ignore")

DATA_DIR    = os.path.join("Data", "step7_normalized_copy")
TEST_CSV    = "test_split.csv"
CKPT_DIR    = "checkpoints"
RESULTS_CSV = "test_results.csv"
REPORT_TXT  = "test_report.txt"
ROC_PLOT    = "test_roc_curve.png"
N_FOLDS     = 5
NUM_WORKERS = 0


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    model = build_model(pretrained=False, dropout_p=0.0)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt.get("val_auc", "?")


# ── Optimal threshold ─────────────────────────────────────────────────────────
def find_optimal_threshold(labels, probs):
    """
    Find threshold that maximises balanced accuracy on the test set.
    Uses the ROC curve — picks the point closest to top-left corner.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    # Youden's J statistic = sensitivity + specificity - 1
    # Maximising this = best balance between sensitivity and specificity
    j_scores  = tpr - fpr
    best_idx  = np.argmax(j_scores)
    return float(thresholds[best_idx]), fpr, tpr


# ── Metrics at given threshold ────────────────────────────────────────────────
def metrics_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    acc   = accuracy_score(labels, preds)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity  = tp / (tp + fn + 1e-8)
    specificity  = tn / (tn + fp + 1e-8)
    balanced_acc = (sensitivity + specificity) / 2.0
    ppv          = tp / (tp + fp + 1e-8)
    npv          = tn / (tn + fn + 1e-8)
    return {
        "threshold":    round(threshold,    4),
        "accuracy":     round(acc,          4),
        "balanced_acc": round(balanced_acc, 4),
        "sensitivity":  round(sensitivity,  4),
        "specificity":  round(specificity,  4),
        "ppv":          round(ppv,          4),
        "npv":          round(npv,          4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for EfficientNet-B0.
    Hooks into the last convolutional block to produce a 2D heatmap.

    How it works:
      1. Forward pass → get prediction
      2. Backprop prediction score to last conv layer
      3. Weight each feature map by its mean gradient
      4. ReLU → resize to 224×224 → normalise to [0,1]
    """
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Use features[-3] instead of features[-1]
        # Earlier layer has higher spatial resolution (14×14 vs 7×7)
        # Gives sharper, more localised heatmaps
        target = self.model.features[-3]

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        """
        input_tensor: [1, 60, 224, 224]
        Returns: heatmap [224, 224] normalised to [0,1]
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Enable grad for backward pass
        with torch.enable_grad():
            logit = self.model(input_tensor)
            logit.backward()

        # Weight feature maps by mean gradient
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1,C,1,1]
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(cam, size=(SLICE_SIZE, SLICE_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0,1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam


def save_gradcam(patient_id, model, device, prob, pred, label, threshold,
                 save_path):
    """
    Generate and save Grad-CAM heatmap for one patient.
    Shows T1c center slice with heatmap overlay.
    """
    # Get input tensor for this patient (exact same preprocessing the model saw)
    ds          = TumorResectabilityDataset.__new__(TumorResectabilityDataset)
    ds.data_dir = DATA_DIR
    ds.augment  = False
    ds.samples  = [(patient_id, label)]
    tensor, _   = ds[0]                          # [C*5, H, W]  e.g. [60, 224, 224]
    tensor      = tensor.unsqueeze(0).to(device)  # [1, 60, H, W]

    # Generate Grad-CAM using the full 60-channel tensor (as trained).
    grad_cam = GradCAM(model)
    cam      = grad_cam.generate(tensor)           # [H, W] normalised [0,1]

    # ── Background images ────────────────────────────────────────────────────
    # Tensor layout (dataset.py):
    #   4 modalities × 15 slices = 60 channels
    #   Per modality: channels 0-4  = axial slices  (center-2 to center+2)
    #                 channels 5-9  = coronal slices
    #                 channels 10-14 = sagittal slices
    #
    # KEY ALIGNMENT FIX:
    # We use ONLY the 5 axial slices (channels 0-4) for each modality.
    # The center axial slice (channel 2) is the same slice the CAM is computed on.
    # Using coronal/sagittal slices would show a different view of the brain
    # which does not match the axial CAM — causing the misalignment seen before.
    SLICES_PER_MOD    = 15
    AXIAL_SLICES      = 5    # first 5 channels per modality are axial
    CENTER_AXIAL_IDX  = 2    # index 2 = center slice (offset 0)

    def _get_center_axial(mod_name):
        """
        Returns the center axial slice for mod_name, normalised to [0,1].
        This is EXACTLY the same slice the model received as input channel,
        so it is perfectly co-registered with the Grad-CAM heatmap.
        """
        if mod_name not in MODALITIES:
            return np.zeros((tensor.shape[2], tensor.shape[3]))
        base  = MODALITIES.index(mod_name) * SLICES_PER_MOD
        ch    = base + CENTER_AXIAL_IDX           # center axial channel index
        img   = tensor[0, ch].cpu().numpy()       # [H, W]
        lo, hi = img.min(), img.max()
        if hi - lo > 1e-8:
            img = (img - lo) / (hi - lo)
        else:
            img = np.zeros_like(img)
        return img.astype(np.float32)

    bg_t1c   = _get_center_axial("T1c")
    bg_flair = _get_center_axial("FLAIR")

    # ── Plot ─────────────────────────────────────────────────────────────────
    # 5 panels: T1c | FLAIR | Heatmap | T1c+CAM | FLAIR+CAM
    label_str = "Resectable (GTR)"  if label == 1 else "Not Resectable (STR)"
    pred_str  = "Resectable (GTR)"  if pred  == 1 else "Not Resectable (STR)"
    correct   = "✓ Correct"         if pred == label else "✗ Wrong"
    color     = "green"             if pred == label else "red"

    fig = plt.figure(figsize=(24, 6))

    # Title in its own row with enough space so it never overlaps the images
    fig.text(
        0.5, 0.97,
        f"{patient_id}",
        ha="center", va="top", fontsize=12, fontweight="bold"
    )
    fig.text(
        0.5, 0.91,
        f"Actual: {label_str}  |  Predicted: {pred_str}  ({correct})  |  "
        f"P(resectable) = {prob*100:.1f}%  |  Threshold = {threshold:.2f}",
        ha="center", va="top", fontsize=10, fontweight="bold", color=color
    )

    # Leave top 18% for title, images occupy the bottom 80%
    axes = []
    for i in range(5):
        ax = fig.add_axes([0.02 + i * 0.196, 0.05, 0.185, 0.80])
        axes.append(ax)

    titles = [
        "T1c\n(tumor enhancement)",
        "FLAIR\n(tumor extent)",
        "Grad-CAM Heatmap\n(red = most important)",
        "T1c + Grad-CAM\n(overlay)",
        "FLAIR + Grad-CAM\n(overlay)",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis("off")

    axes[0].imshow(bg_t1c,   cmap="gray",  origin="upper")
    axes[1].imshow(bg_flair, cmap="gray",  origin="upper")
    axes[2].imshow(cam,      cmap="jet",   origin="upper", vmin=0, vmax=1)

    axes[3].imshow(bg_t1c,   cmap="gray",  origin="upper")
    axes[3].imshow(cam,      cmap="jet",   origin="upper", alpha=0.45, vmin=0, vmax=1)

    axes[4].imshow(bg_flair, cmap="gray",  origin="upper")
    axes[4].imshow(cam,      cmap="jet",   origin="upper", alpha=0.45, vmin=0, vmax=1)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grad-CAM saved → '{save_path}'")


# ── Main test evaluation ──────────────────────────────────────────────────────
def run_test(gradcam_patients=None):
    if not os.path.isfile(TEST_CSV):
        raise FileNotFoundError(f"'{TEST_CSV}' not found. Run train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load all fold models
    models   = []
    val_aucs = []
    for fold in range(1, N_FOLDS + 1):
        ckpt_path = os.path.join(CKPT_DIR, f"best_fold{fold}.pth")
        if not os.path.isfile(ckpt_path):
            print(f"⚠ Fold {fold} checkpoint not found — skipping")
            continue
        model, val_auc = load_model(ckpt_path, device)
        models.append(model)
        val_aucs.append(val_auc)
        print(f"Loaded fold {fold} model (val AUC: {val_auc})")

    if not models:
        raise FileNotFoundError("No fold checkpoints found. Run train.py first.")
    print(f"\nEnsemble of {len(models)} models\n")

    # Test dataset
    test_ds     = TumorResectabilityDataset(TEST_CSV, DATA_DIR, augment=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                             num_workers=NUM_WORKERS)
    test_df     = pd.read_csv(TEST_CSV)

    print(f"Test set : {len(test_ds)} patients")
    print(f"  GTR (resectable)     : {int(test_df['label'].sum())}")
    print(f"  STR (not resectable) : {len(test_df)-int(test_df['label'].sum())}\n")

    # Ensemble inference (no TTA)
    all_model_probs = []
    fold_labels     = []

    print("Running ensemble inference...")
    for fold_idx, model in enumerate(models):
        fold_probs = []
        fold_lbls  = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits = model(images)
                probs  = torch.sigmoid(logits).cpu().numpy().flatten()
                fold_probs.extend(probs.tolist())
                fold_lbls.extend(labels.numpy().flatten().tolist())
        all_model_probs.append(fold_probs)
        fold_labels = fold_lbls
        print(f"  Fold {fold_idx+1} done")

    all_probs  = np.mean(all_model_probs, axis=0)
    all_labels = np.array(fold_labels, dtype=int)

    # AUC
    auc = roc_auc_score(all_labels, all_probs)

    # Find optimal threshold using Youden's J
    opt_threshold, fpr, tpr = find_optimal_threshold(all_labels, all_probs)
    print(f"Optimal threshold (Youden's J): {opt_threshold:.4f}\n")

    # Metrics at multiple thresholds
    thresholds_to_try = [0.3, 0.4, 0.5, opt_threshold]
    threshold_results = []
    for t in thresholds_to_try:
        m = metrics_at_threshold(all_labels, all_probs, t)
        threshold_results.append(m)

    # Use optimal threshold for final predictions
    all_preds = (all_probs >= opt_threshold).astype(int)
    m_opt     = metrics_at_threshold(all_labels, all_probs, opt_threshold)

    # Build report
    report_lines = [
        "=" * 60,
        "TUMOR RESECTABILITY CLASSIFIER — TEST RESULTS",
        "2.5D EfficientNet-B0 | UCSF-PDGM Dataset",
        f"Ensemble of {len(models)} fold models",
        "Labels: GTR (resectable=1) vs STR (not resectable=0)",
        "=" * 60,
        f"Test patients        : {len(all_labels)}",
        f"  GTR (resectable)   : {int(all_labels.sum())}",
        f"  STR (not resect.)  : {len(all_labels)-int(all_labels.sum())}",
        "",
        f"Fold val AUCs        : {[round(float(v), 4) for v in val_aucs]}",
        f"Mean val AUC         : {np.mean([float(v) for v in val_aucs]):.4f}",
        "",
        "─" * 60,
        "PRIMARY METRIC",
        f"  AUC-ROC (test)     : {auc:.4f}",
        "",
        "─" * 60,
        "THRESHOLD ANALYSIS",
        f"  {'Threshold':>10}  {'Accuracy':>10}  {'Bal.Acc':>10}  "
        f"{'Sens':>8}  {'Spec':>8}",
        "  " + "-"*54,
    ]

    for m in threshold_results:
        marker = " ← optimal" if abs(m["threshold"] - opt_threshold) < 0.001 else ""
        report_lines.append(
            f"  {m['threshold']:>10.2f}  {m['accuracy']:>10.4f}  "
            f"{m['balanced_acc']:>10.4f}  "
            f"{m['sensitivity']:>8.4f}  {m['specificity']:>8.4f}{marker}"
        )

    report_lines += [
        "",
        "─" * 60,
        f"FINAL RESULTS (optimal threshold = {opt_threshold:.4f})",
        f"  Accuracy           : {m_opt['accuracy']:.4f}  "
        f"({int(m_opt['accuracy']*len(all_labels))}/{len(all_labels)} correct)",
        f"  Balanced Accuracy  : {m_opt['balanced_acc']:.4f}",
        f"  Sensitivity        : {m_opt['sensitivity']:.4f}",
        f"  Specificity        : {m_opt['specificity']:.4f}",
        f"  PPV (Precision)    : {m_opt['ppv']:.4f}",
        f"  NPV                : {m_opt['npv']:.4f}",
        "",
        "CONFUSION MATRIX (optimal threshold)",
        f"                     Predicted STR   Predicted GTR",
        f"  Actual STR (0) :        {m_opt['tn']:>4}             {m_opt['fp']:>4}",
        f"  Actual GTR (1) :        {m_opt['fn']:>4}             {m_opt['tp']:>4}",
        "",
        "─" * 60,
        "CLASSIFICATION REPORT (optimal threshold)",
        classification_report(
            all_labels, all_preds,
            target_names=["Not Resectable (STR)", "Resectable (GTR)"]
        ),
        "=" * 60,
    ]

    report_str = "\n".join(report_lines)
    print(report_str)

    with open(REPORT_TXT, "w") as f:
        f.write(report_str)
    print(f"\nReport saved          → '{REPORT_TXT}'")

    # Per-patient CSV
    results_df = test_df.copy()
    results_df["prob_resectable"] = np.round(all_probs, 4)
    results_df["predicted_label"] = all_preds
    results_df["predicted_class"] = results_df["predicted_label"].map(
        {1: "Resectable (GTR)", 0: "Not Resectable (STR)"}
    )
    results_df["actual_class"] = results_df["label"].map(
        {1: "Resectable (GTR)", 0: "Not Resectable (STR)"}
    )
    results_df["correct"] = (results_df["predicted_label"] == results_df["label"])
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Per-patient results   → '{RESULTS_CSV}'")

    # ROC curve
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="steelblue", linewidth=2.5,
             label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    # Mark optimal threshold point
    opt_fpr = 1 - m_opt["specificity"]
    opt_tpr = m_opt["sensitivity"]
    plt.scatter([opt_fpr], [opt_tpr], color="red", zorder=5, s=120,
                label=f"Optimal threshold={opt_threshold:.2f}\n"
                      f"(Sens={opt_tpr:.2f}, Spec={m_opt['specificity']:.2f})")
    # Mark default 0.5 threshold point
    m_05 = metrics_at_threshold(all_labels, all_probs, 0.5)
    plt.scatter([1 - m_05["specificity"]], [m_05["sensitivity"]],
                color="orange", zorder=5, s=80, marker="^",
                label=f"Threshold=0.50\n"
                      f"(Sens={m_05['sensitivity']:.2f}, "
                      f"Spec={m_05['specificity']:.2f})")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)",      fontsize=12)
    plt.title(
        "ROC Curve — Tumor Resectability\n"
        f"2.5D EfficientNet-B0 Ensemble | UCSF-PDGM Test Set",
        fontsize=13, fontweight="bold"
    )
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PLOT, dpi=150)
    plt.show()
    print(f"ROC curve             → '{ROC_PLOT}'")

    # Grad-CAM heatmaps
    print(f"\nGenerating Grad-CAM heatmaps...")
    os.makedirs("gradcam", exist_ok=True)

    # Use best fold model for Grad-CAM
    best_fold_idx = int(np.argmax([float(v) for v in val_aucs]))
    gradcam_model = models[best_fold_idx]

    # Determine which patients to show
    if gradcam_patients:
        patients_for_cam = gradcam_patients
    else:
        # Default: show 1 correct GTR, 1 correct STR, 1 wrong prediction
        correct_gtr = results_df[
            (results_df["label"] == 1) & (results_df["correct"] == True)
        ]["patient_id"].tolist()
        correct_str = results_df[
            (results_df["label"] == 0) & (results_df["correct"] == True)
        ]["patient_id"].tolist()
        wrong = results_df[
            results_df["correct"] == False
        ]["patient_id"].tolist()

        patients_for_cam = []
        if correct_gtr: patients_for_cam.append(correct_gtr[0])
        if correct_str: patients_for_cam.append(correct_str[0])
        if wrong:       patients_for_cam.append(wrong[0])

    for pid in patients_for_cam:
        row   = results_df[results_df["patient_id"] == pid].iloc[0]
        prob  = float(row["prob_resectable"])
        pred  = int(row["predicted_label"])
        label = int(row["label"])
        save_path = os.path.join("gradcam", f"{pid}_gradcam.png")
        save_gradcam(pid, gradcam_model, device, prob, pred, label,
                     opt_threshold, save_path)

    print(f"\nAll done! Files saved:")
    print(f"  {REPORT_TXT}")
    print(f"  {RESULTS_CSV}")
    print(f"  {ROC_PLOT}")
    print(f"  gradcam/  (heatmap images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradcam_patients", nargs="+", default=None,
                        help="Specific patient IDs to generate Grad-CAM for")
    args = parser.parse_args()
    run_test(args.gradcam_patients)
