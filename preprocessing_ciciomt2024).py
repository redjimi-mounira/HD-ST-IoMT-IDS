"""
=============================================================================
PREPROCESSING — CICIoMT2024
Hybrid Transformer IDS — Dual Stream Architecture
=============================================================================

STRATÉGIE DATASET :
  ┌─────────────────────────────────────────────────────────────────┐
  │  ATTACKS  → Split ORIGINAL des auteurs préservé                │
  │    attacks/csv/train/ → TRAIN (tel quel)                       │
  │    attacks/csv/test/  → TEST  (tel quel)                       │
  │                                                                 │
  │  PROFILING → Split 70/30 stratifié par scenario                │
  │    profiling/csv/ → 70% ajouté dans TRAIN                      │
  │                  → 30% ajouté dans TEST                        │
  │                                                                 │
  │  FINAL :                                                        │
  │    TRAIN = attacks_train (original) + profiling_train (70%)    │
  │    TEST  = attacks_test  (original) + profiling_test  (30%)    │
  └─────────────────────────────────────────────────────────────────┘

LABELS :
  "Benign"           → 0   (attacks Benign + TOUT le profiling)
  "DDoS-ICMP_Flood"  → 1
  "DDoS-SYN_Flood"   → 2
  "DDoS-TCP_Flood"   → 3
  "DDoS-UDP_Flood"   → 4
  "DoS-ICMP_Flood"   → 5
  "DoS-SYN_Flood"    → 6
  "DoS-TCP_Flood"    → 7
  "DoS-UDP_Flood"    → 8
  "Recon-OS_Scan"    → 9
  "Recon-Ping_Sweep" → 10
  "Recon-Port_Scan"  → 11
  "Recon-Vuln_Scan"  → 12
  "Spoofing-ARP"     → 13
  "MQTT-Malformed"   → 14
  "MQTT-DoS_Connect" → 15
  "MQTT-DDoS_Connect"→ 16
  "MQTT-DoS_Publish" → 17
  "MQTT-DDoS_Publish"→ 18

DUAL STREAM :
  NETWORK_COLS   → NetworkEncoder   (features trafic/attaque)
  PROFILING_COLS → ProfilingEncoder (features comportement/timing)
=============================================================================
"""

import os, json, glob, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATASET_ROOT        = os.path.join("WiFi_and_MQTT")
OUTPUT_DIR          = "processed_data"
RANDOM_SEED         = 42
PROFILING_TEST_SIZE = 0.30

EXCLUDE_COLS = [
    "label","Label","LABEL","Flow ID","flow_id","flowid",
    "Src IP","src_ip","Source IP","Dst IP","dst_ip","Destination IP",
    "Src Port","src_port","Source Port","Dst Port","dst_port","Destination Port",
    "Protocol","protocol","Timestamp","timestamp","Time",
]

# ─────────────────────────────────────────────────────────────────
#  SÉPARATION DUAL STREAM — Basée sur les VRAIS noms de colonnes
#  Vérifiés sur CICIoMT2024 (attacks: 45 cols, profiling: 39 cols)
#
#  NETWORK_COLS  (26) : signature des ATTAQUES
#    → flags TCP, débits, protocoles, features exclusives attacks
#  PROFILING_COLS (20) : signature du COMPORTEMENT normal IoMT
#    → timing IAT, volume, distribution, protocoles comportementaux
#
#  Après normalize_cols() les noms sont en lowercase + underscores :
#    'Protocol Type' → 'protocol_type'
#    'Tot sum'       → 'tot_sum'
#    'Time_To_Live'  → 'time_to_live'
# ─────────────────────────────────────────────────────────────────

NETWORK_COLS_EXACT = [
    # Flags TCP — discriminent directement le type d'attaque
    'syn_flag_number',   # SYN Flood : valeur extrêmement élevée
    'fin_flag_number',   # Port Scan : FIN élevé
    'rst_flag_number',   # DoS / Scan : RST storm
    'psh_flag_number',   # Command Injection : PSH élevé
    'ack_flag_number',   # ACK Flood DDoS
    'ece_flag_number',   # Signalement congestion
    'cwr_flag_number',   # Congestion window reduced
    # Compteurs de flags — version cumulée
    'syn_count',
    'ack_count',
    'fin_count',
    'rst_count',
    # Débit — signature des floods
    'rate',              # Taux global (Rate)
    'srate',             # Source rate   ← exclusif attacks
    'drate',             # Destination rate ← exclusif attacks
    # Durée du flux — SYN Flood = durée très courte
    'duration',          # ← exclusif attacks
    # Protocoles applicatifs — révèlent le type d'attaque
    'http', 'https',     # DDoS HTTP
    'dns',               # DNS Amplification
    'tcp', 'udp', 'icmp',# DDoS/DoS par protocole
    'arp',               # ARP Spoofing
    # Features statistiques exclusives attacks
    'weight',            # ← exclusif attacks
    'magnitue',          # Magnitude du flux ← exclusif attacks (typo dataset)
    'radius',            # ← exclusif attacks
    'covariance',        # ← exclusif attacks
]

PROFILING_COLS_EXACT = [
    # En-tête et protocole bas niveau — comportement normal
    'header_length',     # Idle : headers courts (keepalives)
    'protocol_type',     # Power : protocoles spécifiques au boot
    # TTL — exclusif profiling (révèle les sauts réseau normaux)
    'time_to_live',      # ← exclusif profiling
    # IAT — cœur du comportement IoMT
    'iat',               # Idle : IAT très long (peu de trafic)
                         # Active : IAT court (envoi régulier mesures)
                         # Power : IAT variable (initialisation)
    # Volume et taille — pattern de chaque scénario
    'tot_sum',           # Volume total
    'tot_size',          # Taille totale des paquets
    'number',            # Nombre de paquets
    # Distribution statistique du trafic
    'min',               # Taille min paquets
    'max',               # Taille max paquets
    'avg',               # Taille moyenne
    'std',               # Écart-type : Idle=faible, Active=modéré
    'variance',          # Variance temporelle du comportement
    # Protocoles comportementaux IoMT
    'ssh',               # Interaction : sessions admin
    'telnet',            # Interaction : accès distant
    'dhcp',              # Power : requête DHCP au démarrage
    'igmp',              # Multicast — protocoles réseau normaux
    'ipv',               # Couche IP
    'llc',               # Liaison de données
    'smtp',              # Email — comportement applicatif
    'irc',               # IRC — comportement applicatif
]


# ═══════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ═══════════════════════════════════════════════════════════════════

def log(msg, level=0):
    print("  "*level + msg)

def normalize_label(label):
    """
    Regroupe les variantes numerotees d'une meme attaque.
    TCP_IP-DDoS-ICMP1 -> TCP_IP-DDoS-ICMP
    TCP_IP-DoS-SYN3   -> TCP_IP-DoS-SYN
    TCP_IP-DDoS-UDP5  -> TCP_IP-DDoS-UDP
    """
    import re
    normalized = re.sub(r'(TCP_IP-(?:D?DoS)-(?:ICMP|SYN|TCP|UDP))\d+$',
                        r'\1', label)
    return normalized

def normalize_cols(df):
    """Normalise les noms de colonnes : lowercase + underscores."""
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_", regex=False)
                  .str.replace("/", "_per_", regex=False)
                  .str.replace("-", "_", regex=False))
    return df

def extract_label(filepath):
    """
    Extrait le label depuis le nom de fichier.
    Exemples :
      'DDoS-SYN_Flood_train.pcap.csv' → 'DDoS-SYN_Flood'
      'Benign_train.csv'               → 'Benign'
    """
    name = Path(filepath).stem          # supprime .csv
    # Supprimer éventuel .pcap résiduel
    if name.endswith(".pcap"):
        name = name[:-5]
    # Supprimer suffixes _train / _test
    for s in ["_train","_test","_Train","_Test"]:
        if name.endswith(s):
            name = name[:-len(s)]; break
    return name

def read_csv_safe(fpath, label, source, scenario=None):
    try:
        df = pd.read_csv(fpath, low_memory=False)
        df = normalize_cols(df)
        drop = [c for c in df.columns if c in
                [x.lower().replace(" ","_") for x in EXCLUDE_COLS]]
        df = df.drop(columns=drop, errors="ignore")
        # Normaliser le label (regrouper variantes numérotées)
        label = normalize_label(label)

        df["label_str"] = label
        df["source"]    = source
        df["scenario"]  = scenario or source
        return df
    except Exception as e:
        log(f"⚠️  Erreur {fpath}: {e}", 2); return None


# ═══════════════════════════════════════════════════════════════════
#  ÉTAPE 1 — ATTACKS (split original préservé)
# ═══════════════════════════════════════════════════════════════════

def load_attacks(split):
    folder = os.path.join(DATASET_ROOT, "attacks", "csv", split)
    folder = os.path.normpath(folder)
    files  = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"Aucun CSV dans : {folder}")

    log(f"\n📂 attacks/{split}/ — {len(files)} fichiers", 1)
    dfs = []
    for f in files:
        lbl = extract_label(f)
        df  = read_csv_safe(f, label=lbl, source="network")
        if df is not None and len(df) > 0:
            dfs.append(df)
            log(f"✓ {Path(f).name:45s} {len(df):>8,} lignes  '{lbl}'", 2)

    out = pd.concat(dfs, ignore_index=True)
    log(f"→ Total {split}: {len(out):,} lignes | {out['label_str'].nunique()} classes", 1)
    return out


# ═══════════════════════════════════════════════════════════════════
#  ÉTAPE 2 — PROFILING (label = Benign)
# ═══════════════════════════════════════════════════════════════════

def load_profiling():
    """
    Charge les CSV de profiling.
    Supporte deux structures :
      A) profiling/Power/*.csv  profiling/Idle/*.csv  ...  (sous-dossiers)
      B) profiling/*.csv                                    (fichiers directs)
    """
    scenarios = ["Power", "Idle", "Active", "Interaction"]

    # Racines possibles à tester
    roots = [
        os.path.normpath(os.path.join(DATASET_ROOT, "profiling", "csv")),
        os.path.normpath(os.path.join(DATASET_ROOT, "profiling")),
    ]

    root = None
    for r in roots:
        if os.path.isdir(r):
            root = r
            break

    if root is None:
        print(f"\n❌ ERREUR : Dossier profiling introuvable.")
        print(f"   Répertoire courant : {os.getcwd()}")
        print(f"   DATASET_ROOT       : {DATASET_ROOT}")
        import sys; sys.exit(1)

    log(f"\n📂 profiling/ (racine: {root})", 1)
    dfs = []

    # ── CAS A : sous-dossiers Power/Idle/Active/Interaction ──────
    has_subdirs = any(
        os.path.isdir(os.path.join(root, sc)) for sc in scenarios
    )

    if has_subdirs:
        log(f"  Structure détectée : sous-dossiers par scénario", 1)
        for sc in scenarios:
            folder = os.path.join(root, sc)
            if not os.path.isdir(folder):
                log(f"⚠️  Absent : {folder}", 2); continue
            files = sorted(glob.glob(os.path.join(folder, "*.csv")))
            if not files:
                log(f"⚠️  Aucun CSV dans {sc}/", 2); continue
            sc_dfs = [read_csv_safe(f, label="Benign", source="profiling",
                                    scenario=sc) for f in files]
            sc_dfs = [d for d in sc_dfs if d is not None and len(d) > 0]
            if sc_dfs:
                df_sc = pd.concat(sc_dfs, ignore_index=True)
                dfs.append(df_sc)
                log(f"✓ {sc:15s} {len(df_sc):>8,} lignes", 2)

    # ── CAS B : CSV directement dans le dossier (pas de sous-dossiers) ──
    else:
        log(f"  Structure détectée : CSV directement dans profiling/", 1)
        files = sorted(glob.glob(os.path.join(root, "*.csv")))
        if not files:
            print(f"\n❌ ERREUR : Aucun CSV dans {root}")
            import sys; sys.exit(1)

        log(f"  {len(files)} fichiers CSV trouvés", 1)
        for f in files:
            # Déduire le scénario depuis le nom de fichier si possible
            fname = Path(f).stem.lower()
            if   "power"       in fname: sc = "Power"
            elif "idle"        in fname: sc = "Idle"
            elif "active"      in fname: sc = "Active"
            elif "interact"    in fname: sc = "Interaction"
            else:                        sc = "Profiling"   # inconnu

            df = read_csv_safe(f, label="Benign", source="profiling",
                               scenario=sc)
            if df is not None and len(df) > 0:
                dfs.append(df)
                log(f"✓ {Path(f).name:45s} {len(df):>8,} lignes  scenario='{sc}'", 2)

    if not dfs:
        print(f"\n❌ ERREUR : Aucun CSV de profiling chargé depuis {root}")
        import sys; sys.exit(1)

    out = pd.concat(dfs, ignore_index=True)
    log(f"→ Total profiling: {len(out):,} lignes | scénarios: {out['scenario'].unique().tolist()}", 1)
    return out


# ═══════════════════════════════════════════════════════════════════
#  ÉTAPE 3 — SPLIT PROFILING 70/30
# ═══════════════════════════════════════════════════════════════════

def split_profiling(df):
    log(f"\n⚙️  Split profiling 70/30", 1)

    unique_sc = df["scenario"].nunique()

    if unique_sc > 1:
        # Stratifier sur scenario pour représenter tous les scénarios
        tr, te = train_test_split(df, test_size=PROFILING_TEST_SIZE,
                                  random_state=RANDOM_SEED,
                                  stratify=df["scenario"])
        log(f"  Stratifié sur 'scenario' ({unique_sc} scénarios)", 1)
    else:
        # Un seul scénario → split simple
        tr, te = train_test_split(df, test_size=PROFILING_TEST_SIZE,
                                  random_state=RANDOM_SEED)
        log(f"  Split simple (1 seul scénario)", 1)

    log(f"  profiling_train: {len(tr):,}  |  profiling_test: {len(te):,}", 1)
    return tr, te


# ═══════════════════════════════════════════════════════════════════
#  ÉTAPE 4 — FUSION FINALE
# ═══════════════════════════════════════════════════════════════════

def build_splits(att_tr, att_te, prof_tr, prof_te):
    log(f"\n⚙️  Fusion finale", 1)

    # TRAIN = attacks_train (original) + profiling_train (70%)
    train = pd.concat([att_tr, prof_tr], ignore_index=True)
    train = train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # TEST  = attacks_test  (original) + profiling_test  (30%)
    test  = pd.concat([att_te, prof_te], ignore_index=True)
    test  = test.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    log(f"  TRAIN: {len(train):,}  (attacks_train:{len(att_tr):,} + profiling_train:{len(prof_tr):,})", 1)
    log(f"  TEST : {len(test):,}   (attacks_test:{len(att_te):,}  + profiling_test:{len(prof_te):,})", 1)
    return train, test


# ═══════════════════════════════════════════════════════════════════
#  ÉTAPE 5 — NETTOYAGE + SÉPARATION DUAL STREAM
# ═══════════════════════════════════════════════════════════════════


def select_features(df_train, df_test):
    log(f"\n⚙️  Sélection features + Feature Engineering (fit sur TRAIN)", 1)
    META = ["label_str","source","scenario","label"]

    num_tr = df_train.select_dtypes(include=[np.number]).columns.tolist()
    num_tr = [c for c in num_tr if c not in META]
    log(f"  Colonnes numériques disponibles ({len(num_tr)}) : {num_tr}", 1)

    X_tr = df_train[num_tr].replace([np.inf,-np.inf], np.nan)
    X_te = df_test.reindex(columns=num_tr, fill_value=0).replace([np.inf,-np.inf], np.nan)

    # ── FEATURE ENGINEERING ───────────────────────────────────────
    log(f"\n  ⚙️  Feature Engineering", 1)

    def add_engineered(X):
        X = X.copy()
        eps = 1e-8

        # Ratio SYN/(ACK+1) — signature SYN Flood
        if "syn_count" in X.columns and "ack_count" in X.columns:
            X["fe_syn_ratio"] = X["syn_count"] / (X["ack_count"] + eps)

        # Total flags = somme de tous les flags → intensité d'activité TCP
        flag_cols = [c for c in ["syn_flag_number","fin_flag_number",
                                  "rst_flag_number","psh_flag_number",
                                  "ack_flag_number","ece_flag_number",
                                  "cwr_flag_number"] if c in X.columns]
        if len(flag_cols) >= 2:
            X["fe_total_flags"] = X[flag_cols].sum(axis=1)

        # RST ratio — signature Port Scan
        if "rst_count" in X.columns and "syn_count" in X.columns:
            X["fe_rst_ratio"] = X["rst_count"] / (X["syn_count"] + eps)

        # Bytes/packet = tot_size / (number + 1)
        if "tot_size" in X.columns and "number" in X.columns:
            X["fe_bytes_per_pkt"] = X["tot_size"] / (X["number"] + eps)

        # IAT × Rate — interaction entre timing et débit
        if "iat" in X.columns and "rate" in X.columns:
            X["fe_iat_rate"] = X["iat"] * X["rate"]

        # Spread = max - min → dispersion taille paquets
        if "max" in X.columns and "min" in X.columns:
            X["fe_pkt_spread"] = X["max"] - X["min"]

        # Flood indicator = rate / (iat + 1) → élevé pour les floods
        if "rate" in X.columns and "iat" in X.columns:
            X["fe_flood_score"] = X["rate"] / (X["iat"] + eps)

        return X

    X_tr = add_engineered(X_tr)
    X_te = add_engineered(X_te)

    fe_cols = [c for c in X_tr.columns if c.startswith("fe_")]
    log(f"  Features dérivées ajoutées ({len(fe_cols)}) : {fe_cols}", 1)

    # ── FILTRE 1 : variance strictement nulle ────────────────────
    ok = X_tr.std()[lambda s: s > 1e-10].index.tolist()
    log(f"  Filtre variance=0  : {len(X_tr.columns)-len(ok):>4} supprimées → {len(ok)} restantes", 1)
    X_tr = X_tr[ok]
    X_te = X_te.reindex(columns=ok, fill_value=0)

    # ── FILTRE 2 : zeros dominants avec rescue statistique ────────
    # Avant de supprimer une feature quasi-nulle, vérifier si elle
    # est discriminante pour au moins une classe (ANOVA + MI).
    # Si oui → la garder même à 97%+ de zéros.
    log(f"\n  ⚙️  Filtre zeros + rescue statistique (ANOVA / Mutual Information)", 1)

    y_arr = df_train["label_str"].values  # labels string pour MI/ANOVA

    zero_pct   = (X_tr == 0).mean()
    candidates = zero_pct[zero_pct > 0.95].index.tolist()   # candidats à suppression
    keep_raw   = zero_pct[zero_pct <= 0.95].index.tolist()  # toujours gardées

    rescued = []
    dropped = []

    if candidates:
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import LabelEncoder as _LE

        # Encoder les labels pour mutual_info_classif
        _le  = _LE()
        y_enc = _le.fit_transform(y_arr)

        # Imputer les NaN temporairement pour le calcul MI
        X_cand = X_tr[candidates].fillna(0).values.astype(np.float32)
        mi_scores = mutual_info_classif(
            X_cand, y_enc,
            discrete_features=False,
            random_state=RANDOM_SEED,
            n_neighbors=3,
        )

        # Rescue : garder si MI > seuil OU si la feature est active
        # (non-nulle) dans au moins une classe avec mean > 3× global
        MI_THRESHOLD = 0.01   # information mutuelle minimale

        for col, mi in zip(candidates, mi_scores):
            rescue = False
            reason = ""

            # Critère 1 : MI significatif
            if mi > MI_THRESHOLD:
                rescue = True
                reason = f"MI={mi:.4f} > {MI_THRESHOLD}"

            # Critère 2 : feature très active dans une petite classe
            if not rescue:
                global_mean = X_tr[col].mean()
                for cls in np.unique(y_arr):
                    cls_mean = X_tr[col][y_arr == cls].mean()
                    if cls_mean > 3 * global_mean + 1e-8:
                        rescue = True
                        reason = f"active pour classe '{cls}' (mean={cls_mean:.4f} vs global={global_mean:.4f})"
                        break

            if rescue:
                rescued.append(col)
                log(f"    💾 RESCUE {col:30s} zeros={zero_pct[col]*100:.1f}%  {reason}", 1)
            else:
                dropped.append(col)

    ok2 = keep_raw + rescued
    log(f"  Zeros>95%% candidats : {len(candidates):>3}", 1)
    log(f"    → Supprimées       : {len(dropped):>3}  {dropped}", 1)
    log(f"    → Rescuées (MI/cls): {len(rescued):>3}  {rescued}", 1)
    log(f"  Features après filtre: {len(ok2)}", 1)

    X_tr = X_tr[ok2]
    X_te = X_te.reindex(columns=ok2, fill_value=0)

    # ── FILTRE 3 : corrélation > 0.95 ────────────────────────────
    log(f"\n  ⚙️  Filtre corrélation (seuil > 0.95, fit sur TRAIN)", 1)
    corr_matrix = X_tr.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    # Pour chaque paire corrélée, supprimer celle avec le MI le plus bas
    # (si MI disponible) ou la seconde par ordre alphabétique
    to_drop_corr = []
    for col in upper.columns:
        corr_partners = upper.index[upper[col] > 0.95].tolist()
        if corr_partners:
            to_drop_corr.append(col)
            for partner in corr_partners:
                log(f"    corr({col}, {partner}) = {upper.loc[partner,col]:.3f} → supprime {col}", 1)

    ok3 = [c for c in ok2 if c not in to_drop_corr]
    log(f"  Filtre corr>0.95   : {len(to_drop_corr):>3} supprimées", 1)
    log(f"  Features finales   : {len(ok3)} → {ok3}", 1)

    X_tr = X_tr[ok3]
    X_te = X_te.reindex(columns=ok3, fill_value=0)

    # ── SÉPARATION DUAL STREAM ────────────────────────────────────
    present   = set(ok3)
    net_cols  = [c for c in NETWORK_COLS_EXACT  if c in present]
    prof_cols = [c for c in PROFILING_COLS_EXACT if c in present]

    # Features engineered → network (liées au trafic)
    fe_present = [c for c in ok3 if c.startswith("fe_")]
    net_cols  += fe_present

    # Features non catégorisées → network
    assigned = set(net_cols) | set(prof_cols)
    other    = [c for c in ok3 if c not in assigned]
    if other:
        log(f"  ⚠️  {len(other)} non catégorisées → network : {other}", 1)
        net_cols += other

    # Log absences
    net_missing  = [c for c in NETWORK_COLS_EXACT  if c not in present]
    prof_missing = [c for c in PROFILING_COLS_EXACT if c not in present]
    if net_missing:
        log(f"  ⚠️  NETWORK  absentes : {net_missing}", 1)
    if prof_missing:
        log(f"  ⚠️  PROFILING absentes: {prof_missing}", 1)

    if len(net_cols) < 2:
        raise ValueError(f"NETWORK stream vide. Colonnes dispo: {list(present)}")
    if len(prof_cols) < 2:
        raise ValueError(f"PROFILING stream vide. Colonnes dispo: {list(present)}")

    log(f"  NETWORK_COLS   : {len(net_cols)}  {net_cols}", 1)
    log(f"  PROFILING_COLS : {len(prof_cols)}  {prof_cols}", 1)

    meta_tr = df_train[["label_str","source","scenario"]].reset_index(drop=True)
    meta_te = df_test[["label_str","source","scenario"]].reset_index(drop=True)

    return X_tr, X_te, meta_tr, meta_te, net_cols, prof_cols


# ═══════════════════════════════════════════════════════════════════
#  ÉTAPE 5b — VALIDATION SET STRATIFIÉ (extrait du TRAIN)
# ═══════════════════════════════════════════════════════════════════

def make_validation_set(Xn_tr, Xp_tr, y_tr, val_size=0.15):
    """
    Extrait un validation set stratifié depuis le TRAIN.
    Split : TRAIN 85% / VALID 15%

    Stratifié par classe pour conserver la proportion
    de chaque type d'attaque dans le validation set.

    Retourne :
      Xn_tr, Xp_tr, y_tr    (train réduit  ~85%)
      Xn_val, Xp_val, y_val (validation    ~15%)
    """
    log(f"\n⚙️  Création validation set stratifié ({int(val_size*100)}% du TRAIN)", 1)

    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size,
                                  random_state=RANDOM_SEED)
    train_idx, val_idx = next(sss.split(Xn_tr, y_tr))

    Xn_val = Xn_tr[val_idx];  Xn_tr = Xn_tr[train_idx]
    Xp_val = Xp_tr[val_idx];  Xp_tr = Xp_tr[train_idx]
    y_val  = y_tr[val_idx];   y_tr  = y_tr[train_idx]

    log(f"  TRAIN  : {len(y_tr):>9,} lignes", 1)
    log(f"  VALID  : {len(y_val):>9,} lignes  (stratifié par classe)", 1)

    # Vérifier que toutes les classes sont dans le validation set
    classes_tr  = set(np.unique(y_tr))
    classes_val = set(np.unique(y_val))
    missing_in_val = classes_tr - classes_val
    if missing_in_val:
        log(f"  ⚠️  Classes absentes du val set : {missing_in_val}", 1)
    else:
        log(f"  ✅ Toutes les classes représentées dans le val set", 1)

    return Xn_tr, Xp_tr, y_tr, Xn_val, Xp_val, y_val


# ═══════════════════════════════════════════════════════════════════
#  ÉTAPE 5c — POIDS DE CLASSES (déséquilibre)
# ═══════════════════════════════════════════════════════════════════

def compute_class_weights_and_sampler_weights(y_tr, n_classes):
    """
    Calcule :
      1. class_weights (pour CrossEntropyLoss weighted)
      2. sample_weights (pour WeightedRandomSampler)

    Méthode : poids inversement proportionnel à la fréquence
      w_c = N_total / (n_classes × N_c)
    """
    log(f"\n⚙️  Calcul poids de classes (déséquilibre)", 1)

    from sklearn.utils.class_weight import compute_class_weight

    classes_present = np.unique(y_tr)
    cw_vals = compute_class_weight("balanced",
                                    classes=classes_present,
                                    y=y_tr)

    # Vecteur complet taille n_classes (classes absentes → poids 1.0)
    class_weights = np.ones(n_classes, dtype=np.float32)
    for cls, w in zip(classes_present, cw_vals):
        class_weights[cls] = w

    # Poids par sample pour WeightedRandomSampler
    sample_weights = class_weights[y_tr]

    log(f"  Top 5 classes les plus rares (poids élevé) :", 1)
    sorted_idx = np.argsort(class_weights)[::-1]
    for i in sorted_idx[:5]:
        log(f"    classe {i:2d} : poids={class_weights[i]:.2f}  "
            f"(n={int((y_tr==i).sum())})", 1)

    log(f"  Top 5 classes les plus fréquentes (poids faible) :", 1)
    for i in sorted_idx[-5:]:
        log(f"    classe {i:2d} : poids={class_weights[i]:.4f}  "
            f"(n={int((y_tr==i).sum())})", 1)

    return class_weights, sample_weights


def impute_scale(X_tr, X_te, net_cols, prof_cols):
    log(f"\n⚙️  Imputation médiane + RobustScaler (fit sur TRAIN)", 1)

    def process(cols):
        imp = SimpleImputer(strategy="median")
        sc  = RobustScaler()
        tr  = sc.fit_transform(imp.fit_transform(X_tr[cols].values.astype(np.float32)))
        te  = sc.transform(imp.transform(X_te[cols].values.astype(np.float32)))
        return tr.astype(np.float32), te.astype(np.float32), sc, imp

    Xn_tr, Xn_te, sc_n, imp_n = process(net_cols)
    Xp_tr, Xp_te, sc_p, imp_p = process(prof_cols)

    log(f"  Network  : train{Xn_tr.shape}  test{Xn_te.shape}", 1)
    log(f"  Profiling: train{Xp_tr.shape}  test{Xp_te.shape}", 1)
    return Xn_tr, Xn_te, Xp_tr, Xp_te, sc_n, sc_p, imp_n, imp_p


# ═══════════════════════════════════════════════════════════════════
#  ÉTAPE 7 — ENCODAGE LABELS
# ═══════════════════════════════════════════════════════════════════

def encode_labels(meta_tr, meta_te):
    log(f"\n⚙️  Encodage labels (Benign=0 garanti, labels dynamiques)", 1)

    all_lbl = sorted(set(meta_tr["label_str"]) | set(meta_te["label_str"]))

    # Garantir Benign = 0
    benign_candidates = [l for l in all_lbl if "benign" in l.lower()]
    other_lbl = [l for l in all_lbl if "benign" not in l.lower()]

    if benign_candidates:
        all_lbl = sorted(benign_candidates) + sorted(other_lbl)
    else:
        log(f"  ⚠️  Aucun label 'Benign' trouvé — premier label = 0", 1)

    le = LabelEncoder()
    le.classes_ = np.array(all_lbl)

    y_tr = le.transform(meta_tr["label_str"]).astype(np.int64)
    y_te = le.transform(meta_te["label_str"]).astype(np.int64)

    mapping = {c: int(i) for i, c in enumerate(le.classes_)}
    log(f"  {len(mapping)} classes détectées :", 1)
    for cls, idx in mapping.items():
        tr_n = (y_tr==idx).sum()
        te_n = (y_te==idx).sum()
        log(f"    [{idx:2d}] {cls:45s} train:{tr_n:>8,}  test:{te_n:>8,}", 1)

    return y_tr, y_te, le, mapping


# ═══════════════════════════════════════════════════════════════════
#  ÉTAPE 8 — SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════

def save_all(Xn_tr, Xn_te, Xp_tr, Xp_te, y_tr, y_te,
             sc_n, sc_p, imp_n, imp_p, le, mapping,
             net_cols, prof_cols, summary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log(f"\n💾 Sauvegarde → {OUTPUT_DIR}/", 1)

    np.save(f"{OUTPUT_DIR}/X_net_train.npy",  Xn_tr)
    np.save(f"{OUTPUT_DIR}/X_net_test.npy",   Xn_te)
    np.save(f"{OUTPUT_DIR}/X_prof_train.npy", Xp_tr)
    np.save(f"{OUTPUT_DIR}/X_prof_test.npy",  Xp_te)
    np.save(f"{OUTPUT_DIR}/y_train.npy",      y_tr)
    np.save(f"{OUTPUT_DIR}/y_test.npy",       y_te)

    for name, obj in [("scaler_net.pkl",sc_n),("scaler_prof.pkl",sc_p),
                      ("imputer_net.pkl",imp_n),("imputer_prof.pkl",imp_p),
                      ("label_encoder.pkl",le)]:
        with open(f"{OUTPUT_DIR}/{name}","wb") as f: pickle.dump(obj,f)

    for name, obj in [("network_cols.json",net_cols),
                      ("profiling_cols.json",prof_cols),
                      ("label_mapping.json",mapping),
                      ("preprocessing_summary.json",summary)]:
        with open(f"{OUTPUT_DIR}/{name}","w") as f: json.dump(obj,f,indent=2)

    log(f"  ✅ 6 arrays .npy + 5 .pkl + 4 .json sauvegardés", 1)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print("  PREPROCESSING CICIoMT2024 — Hybrid Transformer IDS")
    print("="*65)

    # ── Vérification chemin dataset ───────────────────────────────
    attacks_train = os.path.normpath(
        os.path.join(DATASET_ROOT, "attacks", "csv", "train"))
    if not os.path.exists(attacks_train):
        print(f"\n❌ ERREUR : Dossier introuvable :")
        print(f"   {attacks_train}")
        print(f"\n   Vérifiez que DATASET_ROOT pointe vers le bon dossier.")
        print(f"   Structure attendue :")
        print(f"   CICIoMT2024/")
        print(f"   └── WiFi_and_MQTT/")
        print(f"       ├── attacks/csv/train/  ← doit exister")
        print(f"       ├── attacks/csv/test/")
        print(f"       └── profiling/csv/Power/Idle/Active/Interaction/")
        print(f"\n   Votre répertoire courant : {os.getcwd()}")
        print(f"   Modifiez DATASET_ROOT dans le script.")
        import sys; sys.exit(1)

    # 1. Attacks (splits originaux)
    att_tr = load_attacks("train")
    att_te = load_attacks("test")

    # 2. Profiling → label Benign
    profiling = load_profiling()

    # 3. Split profiling 70/30
    prof_tr, prof_te = split_profiling(profiling)

    # 4. Fusion
    #    TRAIN = attacks_train (original) + profiling_train (70%)
    #    TEST  = attacks_test  (original) + profiling_test  (30%)
    df_train, df_test = build_splits(att_tr, att_te, prof_tr, prof_te)

    # 5. Sélection features + Feature Engineering + Corrélation
    X_tr, X_te, meta_tr, meta_te, net_cols, prof_cols = \
        select_features(df_train, df_test)

    # 6. Imputation + normalisation
    Xn_tr, Xn_te, Xp_tr, Xp_te, sc_n, sc_p, imp_n, imp_p = \
        impute_scale(X_tr, X_te, net_cols, prof_cols)

    # 7. Labels
    y_tr, y_te, le, mapping = encode_labels(meta_tr, meta_te)

    # 8. Validation set stratifié (extrait du TRAIN)
    n_classes = len(mapping)
    Xn_tr, Xp_tr, y_tr, Xn_val, Xp_val, y_val = \
        make_validation_set(Xn_tr, Xp_tr, y_tr, val_size=0.15)

    # Ajuster Xn_te/Xp_te/y_te restent inchangés (test set original)

    # 9. Poids de classes pour WeightedRandomSampler + CrossEntropyLoss
    class_weights, sample_weights = \
        compute_class_weights_and_sampler_weights(y_tr, n_classes)

    # Sauvegarder poids + val set
    np.save(f"{OUTPUT_DIR}/X_net_val.npy",   Xn_val)
    np.save(f"{OUTPUT_DIR}/X_prof_val.npy",  Xp_val)
    np.save(f"{OUTPUT_DIR}/y_val.npy",       y_val)
    np.save(f"{OUTPUT_DIR}/class_weights.npy", class_weights)
    np.save(f"{OUTPUT_DIR}/sample_weights.npy", sample_weights)

    # 10. Sauvegarde
    summary = {
        "strategy": {
            "attacks"  : "split original auteurs (train/test séparés, non mélangés)",
            "profiling": "split 70/30 stratifié → ajouté dans train et test"
        },
        "composition_train": {
            "attacks_train_original": len(att_tr),
            "profiling_train_70pct" : len(prof_tr),
            "total"                 : len(df_train)
        },
        "composition_test": {
            "attacks_test_original": len(att_te),
            "profiling_test_30pct" : len(prof_te),
            "total"                : len(df_test)
        },
        "shapes": {
            "X_net_train" : list(Xn_tr.shape),
            "X_net_test"  : list(Xn_te.shape),
            "X_prof_train": list(Xp_tr.shape),
            "X_prof_test" : list(Xp_te.shape),
        },
        "n_classes"            : len(mapping),
        "n_train"              : int(len(y_tr)),
        "n_val"                : int(len(y_val)),
        "n_test"               : int(len(y_te)),
        "label_mapping"        : mapping,
        "n_network_features"   : len(net_cols),
        "n_profiling_features" : len(prof_cols),
        "network_cols"         : net_cols,
        "profiling_cols"       : prof_cols,
    }
    save_all(Xn_tr, Xn_te, Xp_tr, Xp_te, y_tr, y_te,
             sc_n, sc_p, imp_n, imp_p, le, mapping,
             net_cols, prof_cols, summary)

    print(f"""
{"="*65}
  ✅ PREPROCESSING TERMINÉ
{"="*65}

  TRAIN : {len(df_train):>10,} lignes
    ├── attacks_train  (ORIGINAL) : {len(att_tr):>8,}
    └── profiling_train (70%)     : {len(prof_tr):>8,}

  TEST  : {len(df_test):>10,} lignes
    ├── attacks_test   (ORIGINAL) : {len(att_te):>8,}
    └── profiling_test  (30%)     : {len(prof_te):>8,}

  DUAL STREAM :
    X_net_train  : {str(Xn_tr.shape):>15}  (network features)
    X_prof_train : {str(Xp_tr.shape):>15}  (profiling features)

  CLASSES : {len(mapping)} (Benign=0 + {len(mapping)-1} types d'attaques)

  SPLITS :
    TRAIN  : {len(y_tr):>9,} lignes  (85% du train original)
    VALID  : {len(y_val):>9,} lignes  (15% stratifié)
    TEST   : {len(y_te):>9,} lignes  (original, jamais touché)

  DÉSÉQUILIBRE :
    Classe la plus rare   : {int(min(c for c in np.bincount(y_tr.astype(int)) if c > 0))} lignes
    Classe la plus fréq.  : {int(np.max(np.bincount(y_tr.astype(int))))} lignes
    Ratio déséquilibre    : {int(np.max(np.bincount(y_tr.astype(int)))) // max(1, int(min(c for c in np.bincount(y_tr.astype(int)) if c > 0)))}x
    → WeightedRandomSampler activé dans transformer_ids.py
""")


if __name__ == "__main__":
    main()
