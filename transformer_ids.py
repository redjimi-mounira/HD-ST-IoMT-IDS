"""
=============================================================================
TRANSFORMER IDS — Hybrid Dual-Stream Architecture
CICIoMT2024 — Hybrid Intrusion Detection System
=============================================================================

ARCHITECTURE :
  X_net  (n, d_net)  →  StreamEncoder  →  e_net  (128)  ─┐
                                          seq_net  (2,128)  ├→ CrossAttention → f(256) → MLP → ŷ
  X_prof (n, d_prof) →  StreamEncoder  →  e_prof (128)  ─┘
                                          seq_prof (2,128)

INPUTS (depuis preprocessing_ciciomt2024.py) :
  processed_data/X_net_train.npy   X_net_test.npy
  processed_data/X_prof_train.npy  X_prof_test.npy
  processed_data/y_train.npy       y_test.npy
  processed_data/label_mapping.json

OUTPUTS :
  model_outputs/best_model.pt
  model_outputs/training_history.json
  model_outputs/classification_report.txt
  model_outputs/classification_report.json
=============================================================================
"""

import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score)
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir"    : "./processed_data",
    "output_dir"  : "./model_outputs",
    "d_model"     : 128,
    "n_heads"     : 8,
    "n_layers"    : 4,
    "d_ff"        : 512,
    "dropout"     : 0.1,
    "mlp_dims"    : [512, 256, 128],
    "batch_size"  : 512,
    "epochs"      : 50,
    "lr"          : 1e-4,
    "weight_decay": 1e-5,
    "label_smooth": 0.1,
    "patience"    : 10,
    "min_delta"   : 1e-4,
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers" : 0,
}


# ═══════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════

class IoMTDataset(Dataset):
    def __init__(self, X_net, X_prof, y):
        self.X_net  = torch.tensor(X_net,  dtype=torch.float32)
        self.X_prof = torch.tensor(X_prof, dtype=torch.float32)
        self.y      = torch.tensor(y,      dtype=torch.long)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X_net[idx], self.X_prof[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════
#  BLOCS ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

class InputProjection(nn.Module):
    """Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm"""
    def __init__(self, d_in, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
    def forward(self, x): return self.proj(x)


class StreamEncoder(nn.Module):
    """
    Encode un stream de features.
    Input  : (batch, d_in)
    Output : cls_emb (batch, d_model), seq (batch, 2, d_model)
    """
    def __init__(self, d_in, d_model=128, n_heads=8,
                 n_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.input_proj = InputProjection(d_in, d_model, dropout)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, x):
        batch   = x.size(0)
        x_proj  = self.input_proj(x).unsqueeze(1)              # (B, 1, d)
        cls     = self.cls_token.expand(batch, -1, -1)         # (B, 1, d)
        seq     = torch.cat([cls, x_proj], dim=1)              # (B, 2, d)
        seq_out = self.norm(self.transformer(seq))             # (B, 2, d)
        return seq_out[:, 0, :], seq_out                       # cls, full_seq


class CrossAttentionFusion(nn.Module):
    """
    Fusion bidirectionnelle :
      ẽ_net  = LN(e_net  + MHA(e_net,  S_prof, S_prof))
      ẽ_prof = LN(e_prof + MHA(e_prof, S_net,  S_net ))
      f = [ẽ_net ‖ ẽ_prof]
    """
    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        kw = dict(embed_dim=d_model, num_heads=n_heads,
                  dropout=dropout, batch_first=True)
        self.cross_n2p  = nn.MultiheadAttention(**kw)
        self.cross_p2n  = nn.MultiheadAttention(**kw)
        self.norm_net   = nn.LayerNorm(d_model)
        self.norm_prof  = nn.LayerNorm(d_model)
        self.drop       = nn.Dropout(dropout)

    def forward(self, e_net, seq_net, e_prof, seq_prof):
        # network interroge profiling
        a_net,  _ = self.cross_n2p(e_net.unsqueeze(1),  seq_prof, seq_prof)
        e_net_f   = self.norm_net( e_net  + self.drop(a_net.squeeze(1)))

        # profiling interroge network
        a_prof, _ = self.cross_p2n(e_prof.unsqueeze(1), seq_net,  seq_net)
        e_prof_f  = self.norm_prof(e_prof + self.drop(a_prof.squeeze(1)))

        return torch.cat([e_net_f, e_prof_f], dim=-1)   # (B, 2*d_model)


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


# ═══════════════════════════════════════════════════════════════════
#  MODÈLE COMPLET
# ═══════════════════════════════════════════════════════════════════

class HybridTransformerIDS(nn.Module):
    """
    Hybrid Dual-Stream Transformer IDS.

    X_net  → StreamEncoder → e_net,  seq_net  ─┐
                                                ├→ CrossAttentionFusion → f → MLP → ŷ
    X_prof → StreamEncoder → e_prof, seq_prof ─┘
    """
    def __init__(self, d_net, d_prof, n_classes,
                 d_model=128, n_heads=8, n_layers=4,
                 d_ff=512, dropout=0.1, mlp_dims=[512,256,128]):
        super().__init__()
        self.net_encoder  = StreamEncoder(d_net,  d_model, n_heads, n_layers, d_ff, dropout)
        self.prof_encoder = StreamEncoder(d_prof, d_model, n_heads, n_layers, d_ff, dropout)
        self.fusion       = CrossAttentionFusion(d_model, n_heads, dropout)
        self.classifier   = MLPClassifier(2*d_model, n_classes, mlp_dims, dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x_net, x_prof):
        e_net,  seq_net  = self.net_encoder(x_net)
        e_prof, seq_prof = self.prof_encoder(x_prof)
        f = self.fusion(e_net, seq_net, e_prof, seq_prof)
        return self.classifier(f)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════
#  EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = None
        self.counter    = 0
        self.best_epoch = 0

    def step(self, value, epoch):
        if self.best is None or value > self.best + self.min_delta:
            self.best = value; self.best_epoch = epoch; self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ═══════════════════════════════════════════════════════════════════
#  TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for Xn, Xp, y in loader:
        Xn, Xp, y = Xn.to(device), Xp.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(Xn, Xp)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct  += (logits.argmax(1) == y).sum().item()
        n        += len(y)
    return loss_sum/n, correct/n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    preds_all, labels_all = [], []
    for Xn, Xp, y in loader:
        Xn, Xp, y = Xn.to(device), Xp.to(device), y.to(device)
        logits = model(Xn, Xp)
        loss   = criterion(logits, y)
        preds  = logits.argmax(1)
        loss_sum += loss.item() * len(y)
        correct  += (preds == y).sum().item()
        n        += len(y)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return loss_sum/n, correct/n, f1, preds_all, labels_all


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print("  HYBRID TRANSFORMER IDS — CICIoMT2024")
    print("  Dual-Stream Network + Profiling | Cross-Attention Fusion")
    print("="*65)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = torch.device(CONFIG["device"])
    print(f"\n  Device: {device}")

    # ── Données ───────────────────────────────────────────────────
    D = CONFIG["data_dir"]
    print(f"\n📂 Chargement depuis {D}/")
    X_net_tr  = np.load(f"{D}/X_net_train.npy")
    X_net_te  = np.load(f"{D}/X_net_test.npy")
    X_prof_tr = np.load(f"{D}/X_prof_train.npy")
    X_prof_te = np.load(f"{D}/X_prof_test.npy")
    y_tr      = np.load(f"{D}/y_train.npy")
    y_te      = np.load(f"{D}/y_test.npy")
    with open(f"{D}/label_mapping.json") as f:
        label_mapping = json.load(f)

    # Validation set (généré par preprocessing)
    if os.path.exists(f"{D}/X_net_val.npy"):
        X_net_val  = np.load(f"{D}/X_net_val.npy")
        X_prof_val = np.load(f"{D}/X_prof_val.npy")
        y_val      = np.load(f"{D}/y_val.npy")
        print(f"  ✅ Validation set chargé : {X_net_val.shape}")
    else:
        X_net_val, X_prof_val, y_val = X_net_te, X_prof_te, y_te
        print(f"  ⚠️  Pas de val set → relancer preprocessing pour le générer")

    # Sample weights pour WeightedRandomSampler
    sample_weights_arr = np.load(f"{D}/sample_weights.npy") if os.path.exists(f"{D}/sample_weights.npy") else None
    class_weights_arr  = np.load(f"{D}/class_weights.npy")  if os.path.exists(f"{D}/class_weights.npy")  else None

    n_classes = len(label_mapping)
    d_net     = X_net_tr.shape[1]
    d_prof    = X_prof_tr.shape[1]
    idx2label = {int(v): k for k, v in label_mapping.items()}

    print(f"  X_net_train  : {X_net_tr.shape}")
    print(f"  X_prof_train : {X_prof_tr.shape}")
    print(f"  X_net_val    : {X_net_val.shape}")
    print(f"  n_classes    : {n_classes}")

    # ── DataLoaders ───────────────────────────────────────────────
    # Train : WeightedRandomSampler pour équilibrer les classes rares
    if sample_weights_arr is not None:
        sampler  = WeightedRandomSampler(
            weights=torch.tensor(sample_weights_arr, dtype=torch.float32),
            num_samples=len(y_tr), replacement=True)
        train_ld = DataLoader(
            IoMTDataset(X_net_tr, X_prof_tr, y_tr),
            batch_size=CONFIG["batch_size"], sampler=sampler,
            num_workers=CONFIG["num_workers"], pin_memory=(device.type=="cuda"))
        print(f"  ✅ WeightedRandomSampler activé")
    else:
        train_ld = DataLoader(
            IoMTDataset(X_net_tr, X_prof_tr, y_tr),
            batch_size=CONFIG["batch_size"], shuffle=True,
            num_workers=CONFIG["num_workers"], pin_memory=(device.type=="cuda"))

    # Validation loader (early stopping ici, PAS sur le test)
    val_ld = DataLoader(
        IoMTDataset(X_net_val, X_prof_val, y_val),
        batch_size=CONFIG["batch_size"]*2, shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=(device.type=="cuda"))

    # Test loader (évaluation finale uniquement)
    test_ld = DataLoader(
        IoMTDataset(X_net_te, X_prof_te, y_te),
        batch_size=CONFIG["batch_size"]*2, shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=(device.type=="cuda"))

    # ── Poids de classes (weighted CrossEntropyLoss) ──────────────
    if class_weights_arr is not None:
        cw_full = class_weights_arr.astype(np.float32)
        print(f"  ✅ class_weights chargés depuis preprocessing")
    else:
        classes_present = np.unique(y_tr)
        cw_vals = compute_class_weight("balanced", classes=classes_present, y=y_tr)
        cw_full = np.ones(n_classes, dtype=np.float32)
        for i, cls in enumerate(classes_present): cw_full[cls] = cw_vals[i]
    cw = torch.tensor(cw_full, dtype=torch.float32).to(device)

    # ── Modèle ───────────────────────────────────────────────────
    model = HybridTransformerIDS(
        d_net=d_net, d_prof=d_prof, n_classes=n_classes,
        d_model=CONFIG["d_model"], n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"], d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"], mlp_dims=CONFIG["mlp_dims"],
    ).to(device)

    print(f"\n  Paramètres : {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss(weight=cw,
                    label_smoothing=CONFIG["label_smooth"])
    optimizer = torch.optim.AdamW(model.parameters(),
                    lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"],
                    eta_min=CONFIG["lr"]*0.01)
    stopper   = EarlyStopping(CONFIG["patience"], CONFIG["min_delta"])

    # ── Boucle entraînement ───────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {'Ep':>3} | {'L_tr':>7} | {'L_te':>7} | "
          f"{'Acc_tr':>6} | {'Acc_te':>6} | {'F1_te':>6} | {'s':>4}")
    print(f"  {'-'*60}")

    history, best_f1, best_state = [], 0.0, None

    for ep in range(1, CONFIG["epochs"]+1):
        t0 = time.time()
        l_tr, a_tr = train_epoch(model, train_ld, optimizer, criterion, device)
        l_te, a_te, f1, preds, labels = evaluate(model, val_ld,  criterion, device)
        scheduler.step()
        dt = time.time()-t0

        tag = ""
        if f1 > best_f1:
            best_f1   = f1
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            tag = " ✅"

        print(f"  {ep:>3} | {l_tr:>7.4f} | {l_te:>7.4f} | "
              f"{a_tr:>6.4f} | {a_te:>6.4f} | {f1:>6.4f} | {dt:>4.1f}s{tag}")

        history.append({"epoch":ep,"loss_tr":round(l_tr,6),"loss_te":round(l_te,6),
                         "acc_tr":round(a_tr,6),"acc_te":round(a_te,6),
                         "f1_te":round(f1,6)})

        if stopper.step(f1, ep):
            print(f"\n  ⏹ Early stop @ ep{ep} | best ep{stopper.best_epoch} F1={stopper.best:.4f}")
            break

    # ── Évaluation finale ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ÉVALUATION FINALE — best F1={best_f1:.4f}")
    print(f"{'='*65}")

    model.load_state_dict(best_state)
    _, acc, f1, preds, labels = evaluate(model, test_ld, criterion, device)

    target_names = [idx2label[i] for i in range(n_classes)]
    report_str  = classification_report(labels, preds,
                    target_names=target_names, digits=4, zero_division=0)
    report_dict = classification_report(labels, preds,
                    target_names=target_names, digits=4, zero_division=0,
                    output_dict=True)

    print(f"\n  Accuracy : {acc:.4f}  |  F1 Macro : {f1:.4f}")
    print(f"\n{report_str}")

    # DR et FPR
    cm       = confusion_matrix(labels, preds)
    tp       = np.diag(cm)[1:].sum()
    fn       = (cm[1:].sum(1) - np.diag(cm)[1:]).sum()
    fp       = cm[0, 1:].sum()
    tn       = cm[0, 0]
    dr       = tp/(tp+fn) if (tp+fn)>0 else 0
    fpr_rate = fp/(fp+tn) if (fp+tn)>0 else 0
    print(f"  Detection Rate (DR) : {dr*100:.2f}%")
    print(f"  False Positive Rate : {fpr_rate*100:.2f}%")

    # ── Sauvegarde ────────────────────────────────────────────────
    O = CONFIG["output_dir"]

    torch.save({
        "model_state_dict": best_state,
        "model_config": dict(d_net=d_net, d_prof=d_prof, n_classes=n_classes,
                             d_model=CONFIG["d_model"], n_heads=CONFIG["n_heads"],
                             n_layers=CONFIG["n_layers"], d_ff=CONFIG["d_ff"],
                             dropout=CONFIG["dropout"], mlp_dims=CONFIG["mlp_dims"]),
        "label_mapping": label_mapping,
        "idx2label"    : idx2label,
        "best_f1"      : best_f1,
        "best_epoch"   : stopper.best_epoch,
        "config"       : CONFIG,
    }, f"{O}/best_model.pt")

    with open(f"{O}/training_history.json","w") as f:
        json.dump(history, f, indent=2)

    with open(f"{O}/classification_report.txt","w") as f:
        f.write(f"Accuracy : {acc:.4f}\nF1 Macro : {f1:.4f}\n"
                f"DR       : {dr*100:.2f}%\nFPR      : {fpr_rate*100:.2f}%\n\n")
        f.write(report_str)

    report_dict.update({"detection_rate": round(dr,6),
                        "false_positive_rate": round(fpr_rate,6)})
    with open(f"{O}/classification_report.json","w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\n  ✅ Fichiers sauvegardés dans {O}/")
    print(f"     best_model.pt | training_history.json")
    print(f"     classification_report.txt | .json")


# ═══════════════════════════════════════════════════════════════════
#  INFÉRENCE
# ═══════════════════════════════════════════════════════════════════

def load_model_for_inference(model_path, device="cpu"):
    """
    Charge le modèle pour l'inférence en production.

    Exemple :
        model, idx2label = load_model_for_inference("model_outputs/best_model.pt")
        x_net  = torch.tensor(x_net_scaled,  dtype=torch.float32).unsqueeze(0)
        x_prof = torch.tensor(x_prof_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(x_net, x_prof)
            pred   = logits.argmax(1).item()
        print(idx2label[pred])
    """
    ck  = torch.load(model_path, map_location=device)
    cfg = ck["model_config"]
    m   = HybridTransformerIDS(**cfg)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    idx2label = {int(k): v for k,v in ck["idx2label"].items()}
    return m, idx2label


if __name__ == "__main__":
    main()
