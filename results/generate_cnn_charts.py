"""
results/generate_cnn_charts.py

Generates publication-quality charts for the sugarcane leaf-condition CNN
(cnn/dataset.py, cnn/model.py, cnn/config.py -- all imported as-is, NOT
modified).

Run from the project root:
    python -m results.generate_cnn_charts

Note: the project ships cnn/dataset.py and cnn/model.py but no training
loop, so this script includes its own `train_and_evaluate()` (uses
cnn.dataset.build_dataloaders and cnn.model.build_model unmodified) rather
than assuming a checkpoint already exists. If you already have a trained
checkpoint you trust, skip retraining -- see main(retrain=False).

Charts produced:
  1. class_distribution     -- image counts per condition class (dataset-level;
                                shows the class imbalance created by
                                cnn/config.py's RAW_TO_CONDITION mapping)
  2. training_curves        -- train/val loss & accuracy vs. epoch
  3. confusion_matrix        -- raw counts + row-normalized (recall) confusion
                                matrix on the held-out test split
  4. per_class_metrics       -- precision / recall / F1 per condition class
  5. roc_curves              -- one-vs-rest ROC curve + AUC per class, plus
                                micro-average
  6. confidence_histogram    -- predicted-class softmax confidence, correct
                                vs. incorrect predictions (calibration sanity
                                check)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize

from cnn import config as cnn_config
from cnn.dataset import build_dataloaders, ConditionRemappedDataset
from cnn.model import build_model

from results.plot_utils import savefig, save_table, COLORS

CONDITION_COLORS = [COLORS["healthy"], COLORS["moderate_stress"], COLORS["severe_stress"]]


# --------------------------------------------------------------------------
# Chart 1: dataset class distribution (no training needed for this one)
# --------------------------------------------------------------------------

def chart_class_distribution(data_root=None):
    """Counts images per condition class after cnn/config.py's
    RAW_TO_CONDITION remapping (5 raw disease folders -> 3 condition
    levels). Worth plotting on its own: e.g. if 'moderate_stress' pools
    several raw classes (Mosaic/Rust/Yellow) into one bucket while
    'severe_stress' is just RedRot, the 3-class problem is imbalanced in a
    way that isn't obvious from the raw folder listing, and that imbalance
    is exactly what chart 4 (per-class recall) needs to be read against."""
    root = data_root or cnn_config.RAW_DATA_DIR
    ds = ConditionRemappedDataset(root, transform=None)
    raw_counts = pd.Series(ds.base.targets).value_counts().sort_index()
    raw_names = ds.base.classes

    condition_counts = {c: 0 for c in cnn_config.CONDITION_CLASSES}
    for raw_idx, count in raw_counts.items():
        condition_counts[cnn_config.RAW_TO_CONDITION[raw_names[raw_idx]]] += count

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(raw_names, [raw_counts.get(i, 0) for i in range(len(raw_names))], color="#adb5bd")
    axes[0].set_title("Raw dataset folders", fontsize=9)
    axes[0].set_ylabel("Image count")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(cnn_config.CONDITION_CLASSES, [condition_counts[c] for c in cnn_config.CONDITION_CLASSES],
                color=CONDITION_COLORS)
    axes[1].set_title("Remapped to RL crop-condition levels", fontsize=9)
    axes[1].set_ylabel("Image count")

    fig.suptitle("Leaf-condition dataset class distribution")
    fig.tight_layout()
    savefig(fig, "cnn_01_class_distribution")

    df = pd.DataFrame({"class": list(condition_counts.keys()), "count": list(condition_counts.values())})
    save_table(df, "cnn_01_class_distribution")
    return condition_counts


# --------------------------------------------------------------------------
# Self-contained train + test-set evaluation
# --------------------------------------------------------------------------

def _run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            n += images.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def train_and_evaluate(num_epochs=None, data_root=None, pretrained=True):
    """Trains cnn.model.build_model() on cnn.dataset.build_dataloaders()'s
    train/val/test split (same 70/15/15-ish split config.py defines),
    logging per-epoch train/val loss+accuracy, then runs one pass over the
    held-out test set collecting per-sample true label, predicted label,
    and full softmax probability vector (needed for the ROC-curve chart)."""
    num_epochs = num_epochs or cnn_config.NUM_EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(root=data_root)
    model = build_model(pretrained=pretrained).to(device)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cnn_config.LEARNING_RATE)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(num_epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, device, optimizer=None)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        print(f"  epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    os.makedirs(cnn_config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cnn_config.CHECKPOINT_DIR, "leaf_cnn.pt"))

    model.eval()
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = softmax(logits, dim=1).cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(probs.argmax(axis=1).tolist())
            y_proba.extend(probs.tolist())

    test_results = {"y_true": np.array(y_true), "y_pred": np.array(y_pred), "y_proba": np.array(y_proba)}
    return model, pd.DataFrame(history), test_results


# --------------------------------------------------------------------------
# Chart 2: training curves
# --------------------------------------------------------------------------

def chart_training_curves(history_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train", color=COLORS["train"], marker="o", markersize=3)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation", color=COLORS["val"], marker="o", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss"); axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="Train", color=COLORS["train"], marker="o", markersize=3)
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="Validation", color=COLORS["val"], marker="o", markersize=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy"); axes[1].legend()

    fig.suptitle("Leaf-condition CNN: training curves")
    fig.tight_layout()
    savefig(fig, "cnn_02_training_curves")
    save_table(history_df, "cnn_02_training_curves")


# --------------------------------------------------------------------------
# Chart 3: confusion matrix
# --------------------------------------------------------------------------

def chart_confusion_matrix(test_results):
    y_true, y_pred = test_results["y_true"], test_results["y_pred"]
    classes = cnn_config.CONDITION_CLASSES
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title, fmt in zip(axes, [cm, cm_norm], ["Raw counts", "Row-normalized (recall)"], ["d", ".2f"]):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = mat[i, j]
                text = f"{val:d}" if fmt == "d" else f"{val:.2f}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if val > mat.max() / 2 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)

    accuracy = (y_true == y_pred).mean()
    fig.suptitle(f"Test-set confusion matrix (accuracy={accuracy:.3f}, n={len(y_true)})")
    fig.tight_layout()
    savefig(fig, "cnn_03_confusion_matrix")

    df = pd.DataFrame(cm, index=classes, columns=classes)
    save_table(df.reset_index().rename(columns={"index": "true_class"}), "cnn_03_confusion_matrix")


# --------------------------------------------------------------------------
# Chart 4: per-class precision/recall/F1
# --------------------------------------------------------------------------

def chart_per_class_metrics(test_results):
    y_true, y_pred = test_results["y_true"], test_results["y_pred"]
    classes = cnn_config.CONDITION_CLASSES
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(classes)); width = 0.25
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_title("Per-class test-set metrics"); ax.legend()
    fig.tight_layout()
    savefig(fig, "cnn_04_per_class_metrics")

    df = pd.DataFrame({"class": classes, "precision": precision, "recall": recall, "f1": f1, "support": support})
    save_table(df, "cnn_04_per_class_metrics")


# --------------------------------------------------------------------------
# Chart 5: multi-class ROC curves
# --------------------------------------------------------------------------

def chart_roc_curves(test_results):
    """One-vs-rest ROC curve per condition class, plus a micro-averaged
    curve pooling all classes -- standard for a 3-class softmax classifier
    where no single class is the 'positive' class by default."""
    y_true, y_proba = test_results["y_true"], test_results["y_proba"]
    classes = cnn_config.CONDITION_CLASSES
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    rows = []
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_proba[:, c])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CONDITION_COLORS[c], linewidth=1.5, label=f"{classes[c]} (AUC={roc_auc:.2f})")
        rows.extend({"class": classes[c], "fpr": f, "tpr": t} for f, t in zip(fpr, tpr))

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color="black", linewidth=2, linestyle="--", label=f"micro-average (AUC={auc_micro:.2f})")

    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Leaf-condition CNN: one-vs-rest ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    savefig(fig, "cnn_05_roc_curves")
    save_table(pd.DataFrame(rows), "cnn_05_roc_curves")


# --------------------------------------------------------------------------
# Chart 6: prediction confidence histogram
# --------------------------------------------------------------------------

def chart_confidence_histogram(test_results):
    """Distribution of the model's confidence (max softmax probability) in
    its predicted class, split by whether that prediction was correct. A
    well-calibrated, genuinely discriminative model should show correct
    predictions clustered near confidence=1.0 and incorrect ones spread
    lower/flatter -- a big overlap is a sign of overconfident wrong guesses,
    worth knowing about before this feeds irrigation decisions downstream."""
    y_true, y_pred, y_proba = test_results["y_true"], test_results["y_pred"], test_results["y_proba"]
    confidence = y_proba.max(axis=1)
    correct_mask = y_true == y_pred

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(confidence[correct_mask], bins=bins, alpha=0.7, label="Correct predictions", color=COLORS["healthy"])
    ax.hist(confidence[~correct_mask], bins=bins, alpha=0.7, label="Incorrect predictions", color=COLORS["severe_stress"])
    ax.set_xlabel("Predicted-class confidence (max softmax probability)")
    ax.set_ylabel("Number of test images")
    ax.set_title("Prediction confidence: correct vs. incorrect")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "cnn_06_confidence_histogram")

    df = pd.DataFrame({"confidence": confidence, "correct": correct_mask})
    save_table(df, "cnn_06_confidence_histogram")


def main(num_epochs=None, retrain=True, data_root=None, pretrained=True):
    chart_class_distribution(data_root=data_root)

    if retrain:
        print("Training leaf-condition CNN (this needs the dataset at "
              f"{data_root or cnn_config.RAW_DATA_DIR} and, if pretrained=True, internet access "
              "to download ImageNet weights the first time)...")
        model, history_df, test_results = train_and_evaluate(
            num_epochs=num_epochs, data_root=data_root, pretrained=pretrained)
    else:
        raise NotImplementedError(
            "retrain=False needs a saved (history_df, test_results) pair from a prior "
            "train_and_evaluate() call -- this project doesn't persist y_proba to disk "
            "by default since it's per-test-image and can be large. Re-run with retrain=True, "
            "or adapt train_and_evaluate() to pickle test_results if you need to reuse it."
        )

    print("Chart 2/6: training curves"); chart_training_curves(history_df)
    print("Chart 3/6: confusion matrix"); chart_confusion_matrix(test_results)
    print("Chart 4/6: per-class metrics"); chart_per_class_metrics(test_results)
    print("Chart 5/6: ROC curves"); chart_roc_curves(test_results)
    print("Chart 6/6: confidence histogram"); chart_confidence_histogram(test_results)
    print("\nAll CNN charts written to results/figures/, data to results/tables/.")


if __name__ == "__main__":
    main()
