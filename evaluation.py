"""
=============================================================================
EVALUATION — Hybrid Transformer IDS — CICIoMT2024
=============================================================================
Calcule :
  - Detection Rate (DR) et FPR global et par classe
  - F1-score macro/weighted par classe
  - Matrice de confusion
  - Courbes loss/F1 d'entraînement

Inputs  : processed_data/  +  model_outputs/best_model.pt
Outputs : evaluation_outputs/
=============================================================================
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION — adapter si nécessaire
# ─────────────────────────────────────────────────────────────────
CFG = {
    "data_dir"    : "./processed_data",
    "model_path"  : "./model_outputs/best_model.pt",
    "history_path": "./model_outputs/training_history.json",
    "output_dir"  : "./evaluation_outputs",
    "batch_size"  : 1024,
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",
}


# ═══════════════════════════════════════════════════════════════════
#  ARCHITECTURE (copie depuis transformer_ids.py)
# ═══════════════════════════════════════════════════════════════════

class InputProjection(nn.Module):
    def __init__(self, d_in, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_model * 2), nn.LayerNorm(d_model * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
        )
    def forward(self, x): return self.proj(x)

class StreamEncoder(nn.Module):
    def __init__(self, d_in, d_model=128, n_heads=8,
                 n_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.input_proj = InputProjection(d_in, d_model, dropout)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        b = x.size(0)
        xp  = self.input_proj(x).unsqueeze(1)
        cls = self.cls_token.expand(b, -1, -1)
        seq = self.norm(self.transformer(torch.cat([cls, xp], dim=1)))
        return seq[:, 0, :], seq

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        kw = dict(embed_dim=d_model, num_heads=n_heads,
                  dropout=dropout, batch_first=True)
        self.c_n2p = nn.MultiheadAttention(**kw)
        self.c_p2n = nn.MultiheadAttention(**kw)
        self.ln_n  = nn.LayerNorm(d_model)
        self.ln_p  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
    def forward(self, en, sn, ep, sp):
        an, _ = self.c_n2p(en.unsqueeze(1), sp, sp)
        enf   = self.ln_n(en + self.drop(an.squeeze(1)))
        ap, _ = self.c_p2n(ep.unsqueeze(1), sn, sn)
        epf   = self.ln_p(ep + self.drop(ap.squeeze(1)))
        return torch.cat([enf, epf], dim=-1)

class MLPClassifier(nn.Module):
    def __init__(self, d_in, n_classes, hidden_dims, dropout=0.1):
        super().__init__()
        layers, prev = [], d_in
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.LayerNorm(dim),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class HybridTransformerIDS(nn.Module):
    def __init__(self, d_net, d_prof, n_classes, d_model=128,
                 n_heads=8, n_layers=4, d_ff=512, dropout=0.1,
                 mlp_dims=[512, 256, 128]):
        super().__init__()
        self.net_enc  = StreamEncoder(d_net,  d_model, n_heads, n_layers, d_ff, dropout)
        self.prof_enc = StreamEncoder(d_prof, d_model, n_heads, n_layers, d_ff, dropout)
        self.fusion   = CrossAttentionFusion(d_model, n_heads, dropout)
        self.clf      = MLPClassifier(2*d_model, n_classes, mlp_dims, dropout)
    def forward(self, xn, xp):
        en, sn = self.net_enc(xn)
        ep, sp = self.prof_enc(xp)
        return self.clf(self.fusion(en, sn, ep, sp))


# ═══════════════════════════════════════════════════════════════════
#  CHARGEMENT MODÈLE
# ═══════════════════════════════════════════════════════════════════

def load_model(model_path, device):
    ck  = torch.load(model_path, map_location=device)
    cfg = ck["model_config"]
    m   = HybridTransformerIDS(**cfg).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    idx2label    = {int(k): v for k, v in ck["idx2label"].items()}
    label_mapping = ck["label_mapping"]
    print(f"  ✅ Modèle chargé — best F1={ck.get('best_f1',0):.4f} "
          f"@ epoch {ck.get('best_epoch','?')}")
    print(f"     d_net={cfg['d_net']} | d_prof={cfg['d_prof']} "
          f"| n_classes={cfg['n_classes']}")
    return m, idx2label, label_mapping


# ═══════════════════════════════════════════════════════════════════
#  INFÉRENCE
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, X_net, X_prof, batch_size, device):
    ds = TensorDataset(
        torch.tensor(X_net,  dtype=torch.float32),
        torch.tensor(X_prof, dtype=torch.float32),
    )
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_preds, all_probs = [], []
    for xn, xp in ld:
        xn, xp = xn.to(device), xp.to(device)
        logits  = model(xn, xp)
        probs   = torch.softmax(logits, dim=-1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_probs)


# ═══════════════════════════════════════════════════════════════════
#  MÉTRIQUES DR / FPR / F1
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, idx2label, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    per_class = {}
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        dr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        per_class[idx2label[i]] = {
            "class_id"  : i,
            "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn),
            "DR"        : round(float(dr),  4),
            "FPR"       : round(float(fpr), 4),
            "Precision" : round(float(pre), 4),
            "Recall"    : round(float(dr),  4),
        }

    # Métriques globales (Benign = classe 0, attaques = 1..N)
    attack_idx = list(range(1, n_classes))
    tp_g = sum(cm[i, i]             for i in attack_idx)
    fn_g = sum(cm[i,:].sum()-cm[i,i] for i in attack_idx)
    fp_g = cm[0, 1:].sum()
    tn_g = cm[0, 0]

    dr_g  = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
    fpr_g = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0

    global_metrics = {
        "accuracy"    : round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro"    : round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "f1_weighted" : round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "DR_global"   : round(float(dr_g),  4),
        "FPR_global"  : round(float(fpr_g), 4),
        "TP_total"    : int(tp_g),
        "FN_total"    : int(fn_g),
        "FP_total"    : int(fp_g),
        "TN_total"    : int(tn_g),
    }
    return cm, per_class, global_metrics


# ═══════════════════════════════════════════════════════════════════
#  RAPPORT TEXTE
# ═══════════════════════════════════════════════════════════════════

def print_report(global_metrics, per_class, idx2label, n_classes,
                 y_true, y_pred):
    print(f"\n{'='*70}")
    print(f"  RÉSULTATS GLOBAUX")
    print(f"{'='*70}")
    print(f"  Accuracy        : {global_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Macro        : {global_metrics['f1_macro']*100:.2f}%")
    print(f"  F1 Weighted     : {global_metrics['f1_weighted']*100:.2f}%")
    print(f"  Detection Rate  : {global_metrics['DR_global']*100:.2f}%")
    print(f"  False Pos. Rate : {global_metrics['FPR_global']*100:.2f}%")

    target_names = [idx2label[i] for i in range(n_classes)]
    cr = classification_report(y_true, y_pred,
                                target_names=target_names,
                                digits=4, zero_division=0,
                                output_dict=True)

    print(f"\n{'='*70}")
    print(f"  MÉTRIQUES PAR CLASSE")
    print(f"{'='*70}")
    print(f"  {'Classe':35s} {'DR':>7} {'FPR':>7} {'Prec':>7} {'F1':>7} "
          f"{'TP':>8} {'FP':>7} {'FN':>7}")
    print(f"  {'-'*80}")
    for i in range(n_classes):
        cls  = idx2label[i]
        m    = per_class[cls]
        f1_c = cr.get(cls, {}).get("f1-score", 0)
        print(f"  {cls:35s} "
              f"{m['DR']*100:>6.2f}% "
              f"{m['FPR']*100:>6.2f}% "
              f"{m['Precision']*100:>6.2f}% "
              f"{f1_c*100:>6.2f}% "
              f"{m['TP']:>8,} "
              f"{m['FP']:>7,} "
              f"{m['FN']:>7,}")

    print(f"\n  Classification Report sklearn :\n")
    print(classification_report(y_true, y_pred,
                                 target_names=target_names,
                                 digits=4, zero_division=0))


# ═══════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, idx2label, n_classes, output_dir):
    labels   = [idx2label[i] for i in range(n_classes)]
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm  = cm.astype(float) / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle("Hybrid Transformer IDS — Confusion Matrix",
                 fontsize=13, fontweight="bold")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=axes[0],
                cbar_kws={"label": "Recall (row-normalized)"})
    axes[0].set_title("Normalized"); axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis='x', rotation=45, labelsize=7)
    axes[0].tick_params(axis='y', rotation=0,  labelsize=7)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=axes[1],
                cbar_kws={"label": "Count"})
    axes[1].set_title("Raw Counts"); axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis='x', rotation=45, labelsize=7)
    axes[1].tick_params(axis='y', rotation=0,  labelsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 confusion_matrix.png")


def plot_training_curves(history_path, output_dir):
    if not os.path.exists(history_path):
        print(f"  ⚠️  training_history.json non trouvé — courbes ignorées")
        return
    with open(history_path) as f:
        history = json.load(f)

    epochs  = [h["epoch"]   for h in history]
    loss_tr = [h["loss_tr"] for h in history]
    loss_te = [h["loss_te"] for h in history]
    acc_tr  = [h["acc_tr"]  for h in history]
    acc_te  = [h["acc_te"]  for h in history]
    f1_te   = [h["f1_te"]   for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, loss_tr, "b-o", ms=3, label="Train")
    axes[0].plot(epochs, loss_te, "r-o", ms=3, label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in acc_tr], "b-o", ms=3, label="Train")
    axes[1].plot(epochs, [a*100 for a in acc_te], "r-o", ms=3, label="Val")
    axes[1].set_title("Accuracy (%)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    best_ep = epochs[f1_te.index(max(f1_te))]
    axes[2].plot(epochs, [f*100 for f in f1_te], "g-o", ms=3, label="Val F1")
    axes[2].axvline(x=best_ep, color="red", ls="--", alpha=0.6,
                    label=f"Best ep={best_ep}")
    axes[2].set_title("F1 Macro (%)"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 training_curves.png")


def plot_per_class_metrics(per_class, output_dir):
    classes  = list(per_class.keys())
    dr_vals  = [per_class[c]["DR"]        * 100 for c in classes]
    fpr_vals = [per_class[c]["FPR"]       * 100 for c in classes]
    pre_vals = [per_class[c]["Precision"] * 100 for c in classes]

    x = np.arange(len(classes)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(16, len(classes)*0.9), 6))

    b1 = ax.bar(x - w,   dr_vals,  w, label="Detection Rate (%)",     color="#2196F3", alpha=0.85)
    b2 = ax.bar(x,       fpr_vals, w, label="False Positive Rate (%)", color="#F44336", alpha=0.85)
    b3 = ax.bar(x + w,   pre_vals, w, label="Precision (%)",           color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rate (%)"); ax.set_ylim(0, 115)
    ax.set_title("Per-Class Metrics — CICIoMT2024",
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(b1, dr_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=6, color="#1565C0")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 per_class_metrics.png")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("  EVALUATION — Hybrid Transformer IDS — CICIoMT2024")
    print("="*70)

    os.makedirs(CFG["output_dir"], exist_ok=True)
    device = torch.device(CFG["device"])
    print(f"\n  Device : {device}")

    # ── Chargement données test ───────────────────────────────────
    D = CFG["data_dir"]
    print(f"\n📂 Chargement données depuis {D}/")
    X_net_te  = np.load(f"{D}/X_net_test.npy")
    X_prof_te = np.load(f"{D}/X_prof_test.npy")
    y_test    = np.load(f"{D}/y_test.npy")

    with open(f"{D}/label_mapping.json") as f:
        label_mapping = json.load(f)

    n_classes = len(label_mapping)
    idx2label = {int(v): k for k, v in label_mapping.items()}

    print(f"  X_net_test  : {X_net_te.shape}")
    print(f"  X_prof_test : {X_prof_te.shape}")
    print(f"  n_classes   : {n_classes}")

    # Distribution du test set
    print(f"\n  Distribution test set :")
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    [{u:2d}] {idx2label[u]:40s} : {c:>8,}")

    # ── Chargement modèle ─────────────────────────────────────────
    print(f"\n📂 Chargement modèle depuis {CFG['model_path']}")
    model, idx2label_ck, _ = load_model(CFG["model_path"], device)

    # ── Inférence ─────────────────────────────────────────────────
    print(f"\n⚙️  Inférence sur {len(y_test):,} exemples...")
    y_pred, y_probs = run_inference(
        model, X_net_te, X_prof_te, CFG["batch_size"], device)

    # ── Métriques ─────────────────────────────────────────────────
    print(f"\n⚙️  Calcul des métriques...")
    cm, per_class, global_metrics = compute_metrics(
        y_test, y_pred, idx2label, n_classes)

    # ── Affichage ─────────────────────────────────────────────────
    print_report(global_metrics, per_class, idx2label, n_classes,
                 y_test, y_pred)

    # ── Figures ───────────────────────────────────────────────────
    print(f"\n📊 Génération des figures...")
    plot_confusion_matrix(cm, idx2label, n_classes, CFG["output_dir"])
    plot_training_curves(CFG["history_path"], CFG["output_dir"])
    plot_per_class_metrics(per_class, CFG["output_dir"])

    # ── Sauvegarde rapport ────────────────────────────────────────
    O = CFG["output_dir"]
    target_names = [idx2label[i] for i in range(n_classes)]
    report_str   = classification_report(
        y_test, y_pred, target_names=target_names,
        digits=4, zero_division=0)

    # TXT
    with open(f"{O}/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("EVALUATION — Hybrid Transformer IDS — CICIoMT2024\n")
        f.write("="*70 + "\n\n")
        f.write("GLOBAL METRICS\n" + "-"*40 + "\n")
        for k, v in global_metrics.items():
            f.write(f"  {k:20s} : {v}\n")
        f.write("\n\nPER-CLASS METRICS\n" + "-"*40 + "\n")
        f.write(f"{'Class':35s} {'DR':>8} {'FPR':>8} {'Prec':>8} "
                f"{'TP':>9} {'FP':>8} {'FN':>8}\n")
        for cls, m in per_class.items():
            f.write(f"{cls:35s} "
                    f"{m['DR']*100:>7.2f}% "
                    f"{m['FPR']*100:>7.2f}% "
                    f"{m['Precision']*100:>7.2f}% "
                    f"{m['TP']:>9,} "
                    f"{m['FP']:>8,} "
                    f"{m['FN']:>8,}\n")
        f.write("\n\nCLASSIFICATION REPORT\n" + "-"*40 + "\n")
        f.write(report_str)

    # JSON
    report_json = {
        "global_metrics" : global_metrics,
        "per_class"      : per_class,
        "label_mapping"  : label_mapping,
        "sklearn_report" : classification_report(
            y_test, y_pred, target_names=target_names,
            digits=4, zero_division=0, output_dict=True),
    }
    with open(f"{O}/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    # ── Résumé ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ✅ ÉVALUATION TERMINÉE")
    print(f"{'='*70}")
    print(f"""
  RÉSULTATS CLÉS :
    Accuracy        : {global_metrics['accuracy']*100:.2f}%
    F1 Macro        : {global_metrics['f1_macro']*100:.2f}%
    Detection Rate  : {global_metrics['DR_global']*100:.2f}%
    False Pos. Rate : {global_metrics['FPR_global']*100:.2f}%

  FICHIERS GÉNÉRÉS dans {O}/ :
    evaluation_report.txt   ← rapport complet (article)
    evaluation_report.json  ← données structurées
    confusion_matrix.png    ← figure IEEE
    training_curves.png     ← figure IEEE
    per_class_metrics.png   ← figure IEEE
""")


if __name__ == "__main__":
    main()
