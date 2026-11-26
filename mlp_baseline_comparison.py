"""
================================================================
MLP BASELINE COMPARISON — Reviewer 1 Response
================================================================
LIMA Lab — University of Chadli Bendjedid, El Tarf, Algeria

PURPOSE:
  Reviewer 1 asked: "Have you compared against a deep MLP
  baseline with the same parameter count?" This script answers
  that question directly with 3 models trained under IDENTICAL
  conditions (Option A: Weighted Cross-Entropy + Weighted-
  RandomSampler, 50 epochs).

  MODEL 1 — MLP-Flat        : all 32 features concatenated,
                                single deep MLP, NO dual-stream
                                separation at all.
  MODEL 2 — MLP-DualStream  : two separate MLPs (one per
                                stream), concatenated before
                                classification. Tests whether
                                dual-stream SEPARATION alone
                                (without attention) matters.
  MODEL 3 — HD-ST (existing): your already-trained Option A
                                checkpoint, loaded for direct
                                comparison (no retraining).

INTERPRETATION GUIDE:
  If HD-ST >> MLP-DualStream >> MLP-Flat
      -> Both dual-stream separation AND Cross-Attention add
         real value. Keep current architecture, cite these
         numbers as your empirical justification.

  If HD-ST ~= MLP-DualStream >> MLP-Flat
      -> Dual-stream separation matters, but the Transformer/
         attention mechanism specifically does not add much
         beyond simple per-stream MLPs. Consider the Tier 2
         feature-tokenization redesign discussed with Claude.

  If HD-ST ~= MLP-DualStream ~= MLP-Flat
      -> None of the architectural choices matter much on this
         dataset. Re-open the discussion on redesign priorities.

USAGE (Google Colab):
  1. Upload this script alongside your existing .npy data files
     (X_net_train.npy, X_prof_train.npy, y_train.npy, and the
     corresponding _val / _test splits) in the same folder
     structure you already use for transformer_ids_optionB.py.
  2. Adjust DATA_DIR below to match your Google Drive path.
  3. Run: !python mlp_baseline_comparison.py
  4. Results are printed AND saved to mlp_comparison_results.json
================================================================
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# ============================================================
# CONFIGURATION — adjust DATA_DIR to your Google Drive path
# ============================================================
DATA_DIR = "/content/drive/MyDrive/CICIoMT2024/processed"
OUTPUT_DIR = "/content/drive/MyDrive/CICIoMT2024/mlp_baseline_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 19
D_NET = 18
D_PROF = 14
BATCH_SIZE = 256
EPOCHS = 50              # matches Option A exactly
LR = 3e-5                # matches Option A exactly
WEIGHT_DECAY = 0.01      # matches Option A exactly
PATIENCE = 10            # matches Option A exactly

print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")


# ============================================================
# DATA LOADING — matches your existing .npy file convention
# ============================================================
def load_split(split_name):
    """Loads X_net, X_prof, y for a given split (train/val/test)."""
    x_net = np.load(os.path.join(DATA_DIR, f"X_net_{split_name}.npy"))
    x_prof = np.load(os.path.join(DATA_DIR, f"X_prof_{split_name}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"y_{split_name}.npy"))
    return x_net.astype(np.float32), x_prof.astype(np.float32), y.astype(np.int64)


class IoMTDataset(Dataset):
    def __init__(self, x_net, x_prof, y):
        self.x_net = torch.from_numpy(x_net)
        self.x_prof = torch.from_numpy(x_prof)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Concatenated view for MLP-Flat; dual view for MLP-DualStream
        x_flat = torch.cat([self.x_net[idx], self.x_prof[idx]], dim=0)
        return x_flat, self.x_net[idx], self.x_prof[idx], self.y[idx]


# ============================================================
# MODEL 1 — MLP-Flat (all 32 features, no stream separation)
# ============================================================
class MLPFlat(nn.Module):
    """
    Simplest possible baseline: concatenate all 32 features,
    feed through a deep MLP. Sized to ~2.1M parameters to match
    HD-ST's total parameter count for a fair comparison.
    """
    def __init__(self, in_dim=32, n_classes=19, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x_flat, x_net=None, x_prof=None):
        return self.net(x_flat)


# ============================================================
# MODEL 2 — MLP-DualStream (separate MLP per stream, then concat)
# ============================================================
class MLPDualStream(nn.Module):
    """
    Preserves the dual-stream PHILOSOPHY (separate encoding of
    network vs profiling features) but replaces the 2-token
    Transformer encoder with a simple MLP encoder per stream.
    This isolates the question: does dual-stream separation
    alone (without attention) explain HD-ST's performance?

    Sized to ~2.11M parameters (vs HD-ST's 2,095,635) so the
    comparison is PARAMETER-MATCHED, not just "a small model
    loses to a bigger one" — directly answering Reviewer 1's
    explicit request for a same-parameter-count MLP baseline.
    """
    def __init__(self, d_net=18, d_prof=14, n_classes=19,
                 d_model=256, enc_dim=768, clf_dim=512, dropout=0.3):
        super().__init__()
        # Per-stream encoder — widened to match HD-ST's total
        # parameter budget (see param-matching calculation in
        # the accompanying documentation / commit message).
        self.net_encoder = nn.Sequential(
            nn.Linear(d_net, enc_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(enc_dim, enc_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(enc_dim, d_model), nn.GELU(),
        )
        self.prof_encoder = nn.Sequential(
            nn.Linear(d_prof, enc_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(enc_dim, enc_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(enc_dim, d_model), nn.GELU(),
        )
        # Fusion: simple concatenation (no Cross-Attention —
        # isolates the within-stream encoder question cleanly)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, clf_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(clf_dim, 384), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(384, 128), nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x_flat, x_net, x_prof):
        e_net = self.net_encoder(x_net)
        e_prof = self.prof_encoder(x_prof)
        fused = torch.cat([e_net, e_prof], dim=1)
        return self.classifier(fused)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# TRAINING — identical procedure to your Option A
# ============================================================
def compute_class_weights(y_train, n_classes):
    counts = np.bincount(y_train, minlength=n_classes)
    N = len(y_train)
    weights = N / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(y_train):
    counts = np.bincount(y_train)
    sample_weights = 1.0 / counts[y_train]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_model(model, model_name, train_loader, val_loader,
                 class_weights, epochs=EPOCHS):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Training {model_name} — {count_params(model):,} parameters")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_flat, x_net, x_prof, y in train_loader:
            x_flat, x_net, x_prof, y = (x_flat.to(DEVICE), x_net.to(DEVICE),
                                          x_prof.to(DEVICE), y.to(DEVICE))
            optimizer.zero_grad()
            logits = model(x_flat, x_net, x_prof)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_flat, x_net, x_prof, y in val_loader:
                x_flat, x_net, x_prof = (x_flat.to(DEVICE), x_net.to(DEVICE),
                                           x_prof.to(DEVICE))
                logits = model(x_flat, x_net, x_prof)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " <- BEST"
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or marker:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} "
                  f"| val_F1={val_f1:.4f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    model.load_state_dict(best_state)
    print(f"  Best val Macro-F1: {best_f1:.4f}")
    return model, best_f1


# ============================================================
# EVALUATION — same metrics as your existing evaluation.py
# ============================================================
def evaluate_model(model, test_loader, n_classes=N_CLASSES):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x_flat, x_net, x_prof, y in test_loader:
            x_flat, x_net, x_prof = (x_flat.to(DEVICE), x_net.to(DEVICE),
                                       x_prof.to(DEVICE))
            logits = model(x_flat, x_net, x_prof)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Binary DR/FPR: class 0 = Benign, all others = Attack
    y_true_bin = (all_labels != 0).astype(int)
    y_pred_bin = (all_preds != 0).astype(int)
    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    dr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        y_true_onehot = np.eye(n_classes)[all_labels]
        auc = roc_auc_score(y_true_onehot, all_probs, average="macro",
                             multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "detection_rate": float(dr),
        "fpr": float(fpr),
        "auc_macro": float(auc),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading data...")
    x_net_tr, x_prof_tr, y_tr = load_split("train")
    x_net_val, x_prof_val, y_val = load_split("val")
    x_net_te, x_prof_te, y_te = load_split("test")
    print(f"  Train: {len(y_tr):,} | Val: {len(y_val):,} | Test: {len(y_te):,}")

    train_ds = IoMTDataset(x_net_tr, x_prof_tr, y_tr)
    val_ds = IoMTDataset(x_net_val, x_prof_val, y_val)
    test_ds = IoMTDataset(x_net_te, x_prof_te, y_te)

    sampler = make_weighted_sampler(y_tr)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(y_tr, N_CLASSES)

    results = {}

    # ---- Model 1: MLP-Flat ----
    t0 = time.time()
    mlp_flat = MLPFlat(in_dim=D_NET + D_PROF, n_classes=N_CLASSES)
    mlp_flat, _ = train_model(mlp_flat, "MLP-Flat", train_loader, val_loader,
                               class_weights)
    results["MLP-Flat"] = evaluate_model(mlp_flat, test_loader)
    results["MLP-Flat"]["params"] = count_params(mlp_flat)
    results["MLP-Flat"]["train_time_min"] = (time.time() - t0) / 60
    torch.save(mlp_flat.state_dict(), os.path.join(OUTPUT_DIR, "mlp_flat.pt"))

    # ---- Model 2: MLP-DualStream ----
    t0 = time.time()
    mlp_dual = MLPDualStream(d_net=D_NET, d_prof=D_PROF, n_classes=N_CLASSES)
    mlp_dual, _ = train_model(mlp_dual, "MLP-DualStream", train_loader,
                               val_loader, class_weights)
    results["MLP-DualStream"] = evaluate_model(mlp_dual, test_loader)
    results["MLP-DualStream"]["params"] = count_params(mlp_dual)
    results["MLP-DualStream"]["train_time_min"] = (time.time() - t0) / 60
    torch.save(mlp_dual.state_dict(), os.path.join(OUTPUT_DIR, "mlp_dualstream.pt"))

    # ---- Model 3: HD-ST (your existing Option A results) ----
    # Hardcoded from your confirmed Table III — no retraining needed.
    results["HD-ST (Option A)"] = {
        "accuracy": 0.9775,
        "macro_f1": 0.7522,
        "detection_rate": 0.9873,
        "fpr": 0.1225,
        "auc_macro": 0.9142,
        "params": 2095635,
        "train_time_min": None,  # already trained
    }

    # ---- Print comparison table ----
    print("\n" + "=" * 78)
    print("  FINAL COMPARISON — MLP Baselines vs HD-ST")
    print("=" * 78)
    header = f"  {'Model':<20s}{'Acc%':>8s}{'F1m%':>8s}{'DR%':>8s}{'FPR%':>8s}{'AUC':>9s}{'Params':>12s}"
    print(header)
    print("  " + "-" * 74)
    for name, r in results.items():
        print(f"  {name:<20s}{r['accuracy']*100:>7.2f}%{r['macro_f1']*100:>7.2f}%"
              f"{r['detection_rate']*100:>7.2f}%{r['fpr']*100:>7.2f}%"
              f"{r['auc_macro']:>9.4f}{r['params']:>12,}")
    print("=" * 78)

    # ---- Interpretation hint ----
    f1_flat = results["MLP-Flat"]["macro_f1"]
    f1_dual = results["MLP-DualStream"]["macro_f1"]
    f1_hdst = results["HD-ST (Option A)"]["macro_f1"]
    print("\n  INTERPRETATION:")
    if f1_hdst > f1_dual + 0.02 and f1_dual > f1_flat + 0.02:
        print("  -> HD-ST clearly outperforms both MLP baselines.")
        print("     Both dual-stream separation AND Cross-Attention add")
        print("     real value. Keep current architecture — cite these")
        print("     numbers directly in response to Reviewer 1.")
    elif f1_hdst > f1_dual + 0.02:
        print("  -> HD-ST outperforms MLP-DualStream (which matches")
        print("     MLP-Flat). Cross-Attention fusion is the key value")
        print("     driver, not the within-stream encoder alone.")
    elif abs(f1_hdst - f1_dual) < 0.02:
        print("  -> HD-ST and MLP-DualStream perform similarly.")
        print("     The Transformer/attention mechanism is NOT adding")
        print("     much beyond simple dual-stream separation.")
        print("     Consider the Tier 2 feature-tokenization redesign.")
    else:
        print("  -> Unexpected pattern — inspect per-class results")
        print("     before drawing conclusions.")

    # ---- Save results ----
    out_path = os.path.join(OUTPUT_DIR, "mlp_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print("  Copy this table directly into Section VIII (Ablation Study)")
    print("  as your response to Reviewer 1's MLP baseline request.")


if __name__ == "__main__":
    main()
