"""
Baseline Models for Ablation Study
====================================

Implements non-GNN and standard-GNN baselines for fair comparison
in the IEEE paper's experimental section.

Models:
    1. XGBoostBaseline  — Gradient boosted trees on flat 32 features (no graph)
    2. MLPBaseline      — 3-layer neural network on flat 32 features (no graph)
    3. StandardGAT      — PyG GATConv (homogeneous, no temporal/hetero)

Usage:
    python scripts/baselines.py                # Train all baselines
    python scripts/baselines.py --model xgb    # Train only XGBoost
"""

import numpy as np
import pandas as pd
import pickle
import logging
import argparse
import time
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURE_COLS = [
    'TransactionAmt', 'TransactionDT', 'ProductCD', 'dist1',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
    'C1', 'C2', 'C5', 'C6', 'C11', 'C13', 'C14',
    'D1', 'D2', 'D4', 'D10', 'D15',
    'V127', 'V130', 'V307', 'V310',
    'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo'
]


def prepare_flat_features(df, feature_mean=None, feature_std=None):
    """Prepare flat feature matrix (no graph structure)."""
    X = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

    X['TransactionAmt'] = np.log1p(np.maximum(X['TransactionAmt'].values, 0.0))
    for v_col in ['V127', 'V130', 'V307', 'V310']:
        if v_col in X.columns:
            X[v_col] = np.log1p(np.maximum(X[v_col].values, 0.0))

    X_vals = X.values.astype(np.float32)

    if feature_mean is None:
        feature_mean = X_vals.mean(axis=0)
        feature_std = X_vals.std(axis=0)
        feature_std[feature_std == 0] = 1.0

    X_norm = (X_vals - feature_mean) / feature_std
    y = df['isFraud'].values.astype(np.float32)

    return X_norm, y, feature_mean, feature_std


def evaluate_predictions(name, targets, predictions):
    """Compute all metrics for a model."""
    auc = roc_auc_score(targets, predictions)
    ap = average_precision_score(targets, predictions)
    binary = (predictions >= 0.5).astype(int)
    f1 = f1_score(targets, binary, zero_division=0)
    prec = precision_score(targets, binary, zero_division=0)
    rec = recall_score(targets, binary, zero_division=0)

    results = {
        'name': name, 'auc_roc': auc, 'pr_auc': ap,
        'f1': f1, 'precision': prec, 'recall': rec,
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"  {name}")
    logger.info(f"  AUC-ROC: {auc:.4f} | PR-AUC: {ap:.4f} | F1: {f1:.4f}")
    logger.info(f"  Precision: {prec:.4f} | Recall: {rec:.4f}")
    logger.info(f"{'='*50}")

    return results


# ===================================================================
# 1. XGBoost Baseline
# ===================================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost on flat features (no graph)."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.error("xgboost not installed. pip install xgboost")
        return None, None

    logger.info("Training XGBoost baseline...")
    t0 = time.time()

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric='auc',
        use_label_encoder=False,
        tree_method='hist',
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    preds = model.predict_proba(X_val)[:, 1]
    t_elapsed = time.time() - t0
    logger.info(f"XGBoost trained in {t_elapsed:.1f}s")
    
    # Save the model
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    bst = model
    bst.save_model(models_dir / "xgboost_model.json")
    logger.info("Saved XGBoost model to models/xgboost_model.json")

    results = evaluate_predictions("XGBoost (no graph)", y_val, preds)
    return results, preds


# ===================================================================
# 2. MLP Baseline
# ===================================================================

def train_mlp(X_train, y_train, X_val, y_val):
    """Train a 3-layer MLP on flat features (no graph)."""
    import torch
    import torch.nn as nn

    logger.info("Training MLP baseline...")
    t0 = time.time()

    class MLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'

    model = MLP(X_train.shape[1]).to(device)

    x_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_vl = torch.tensor(X_val, dtype=torch.float32).to(device)

    n_pos = y_tr.sum().item()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_auc = 0
    best_preds = None

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = torch.sigmoid(model(x_vl)).cpu().numpy()
                auc = roc_auc_score(y_val, val_preds)
                if auc > best_auc:
                    best_auc = auc
                    best_preds = val_preds.copy()
                logger.info(f"  MLP Epoch {epoch+1}/50 - Loss: {loss.item():.4f} - Val AUC: {auc:.4f}")

    elapsed = time.time() - t0
    logger.info(f"MLP trained in {elapsed:.1f}s")

    if best_preds is None:
        model.eval()
        with torch.no_grad():
            best_preds = torch.sigmoid(model(x_vl)).cpu().numpy()

    results = evaluate_predictions("MLP (no graph)", y_val, best_preds)
    return results, best_preds


# ===================================================================
# 3. Standard GAT Baseline
# ===================================================================

def train_standard_gat(X_train, y_train, X_val, y_val,
                       train_edge_index, val_edge_index):
    """Train standard PyG GATConv (homogeneous, no temporal/hetero)."""
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATConv

    logger.info("Training standard GAT baseline...")
    t0 = time.time()

    class StandardGAT(nn.Module):
        def __init__(self, in_dim, hidden=128, heads=4, dropout=0.2):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
            self.conv3 = GATConv(hidden, hidden // 2, heads=1, concat=False, dropout=dropout)
            self.pred = nn.Sequential(
                nn.Linear(hidden // 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
            )

        def forward(self, x, edge_index, **kwargs):
            h = torch.relu(self.conv1(x, edge_index))
            h = torch.relu(self.conv2(h, edge_index))
            h = torch.relu(self.conv3(h, edge_index))
            return self.pred(h).squeeze(-1)

    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'

    model = StandardGAT(X_train.shape[1]).to(device)

    x_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_vl = torch.tensor(X_val, dtype=torch.float32).to(device)
    ei_tr = train_edge_index.to(device)
    ei_vl = val_edge_index.to(device)

    n_pos = y_tr.sum().item()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    best_auc = 0
    best_preds = None

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tr, ei_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 29:
            model.eval()
            with torch.no_grad():
                val_preds = torch.sigmoid(model(x_vl, ei_vl)).cpu().numpy()
                auc = roc_auc_score(y_val, val_preds)
                if auc > best_auc:
                    best_auc = auc
                    best_preds = val_preds.copy()
                logger.info(f"  GAT Epoch {epoch+1}/30 - Loss: {loss.item():.4f} - Val AUC: {auc:.4f}")

    elapsed = time.time() - t0
    logger.info(f"Standard GAT trained in {elapsed:.1f}s")

    if best_preds is None:
        model.eval()
        with torch.no_grad():
            best_preds = torch.sigmoid(model(x_vl, ei_vl)).cpu().numpy()

    results = evaluate_predictions("GAT (homogeneous)", y_val, best_preds)
    return results, best_preds


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['xgb', 'mlp', 'gat', 'rgcn', 'all'], default='all')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets...")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")

    X_train, y_train, feat_mean, feat_std = prepare_flat_features(train_df)
    X_val, y_val, _, _ = prepare_flat_features(val_df, feat_mean, feat_std)

    all_results = {}

    # XGBoost
    if args.model in ('xgb', 'all'):
        xgb_results, xgb_preds = train_xgboost(X_train, y_train, X_val, y_val)
        if xgb_results:
            all_results['xgboost'] = xgb_results
            np.savez(results_dir / "xgboost_predictions.npz",
                     predictions=xgb_preds, targets=y_val)

    # MLP
    if args.model in ('mlp', 'all'):
        mlp_results, mlp_preds = train_mlp(X_train, y_train, X_val, y_val)
        if mlp_results:
            all_results['mlp'] = mlp_results
            np.savez(results_dir / "mlp_predictions.npz",
                     predictions=mlp_preds, targets=y_val)

    # Standard GAT
    if args.model in ('gat', 'all'):
        import torch
        # Build simple homogeneous edges (same as original train_gnn.py)
        card_groups = train_df.groupby(['card1', 'card2']).indices
        edges = []
        for _, idxs in card_groups.items():
            n = len(idxs)
            if 1 < n < 60:
                limit = min(n, 12)
                for i in range(limit):
                    for j in range(i + 1, min(limit, i + 5)):
                        edges.append((idxs[i], idxs[j]))
                        edges.append((idxs[j], idxs[i]))
        for i in range(len(train_df)):
            edges.append((i, i))
        train_ei = torch.tensor(edges, dtype=torch.long).t()

        n_val = len(val_df)
        val_ei = torch.arange(n_val, dtype=torch.long).repeat(2, 1)

        gat_results, gat_preds = train_standard_gat(
            X_train, y_train, X_val, y_val, train_ei, val_ei
        )
        if gat_results:
            all_results['gat'] = gat_results
            np.savez(results_dir / "gat_predictions.npz",
                     predictions=gat_preds, targets=y_val)

    # RGCN Baseline
    if args.model in ('rgcn', 'all'):
        from rgcn_baseline import train_rgcn
        rgcn_results, rgcn_preds = train_rgcn(
            X_train, y_train, X_val, y_val, train_df, val_df, feat_mean, feat_std
        )
        if rgcn_results:
            all_results['rgcn'] = rgcn_results
            np.savez(results_dir / "rgcn_predictions.npz",
                     predictions=rgcn_preds, targets=y_val)

    # Save all baseline results
    with open(results_dir / "baseline_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)

    # Summary table
    logger.info(f"\n{'='*70}")
    logger.info("  BASELINE COMPARISON SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Model':<25s} {'AUC-ROC':>8s} {'PR-AUC':>8s} {'F1':>8s} {'Prec':>8s} {'Recall':>8s}")
    logger.info(f"  {'-'*67}")
    for name, res in all_results.items():
        logger.info(f"  {res['name']:<25s} {res['auc_roc']:>8.4f} {res['pr_auc']:>8.4f} "
                    f"{res['f1']:>8.4f} {res['precision']:>8.4f} {res['recall']:>8.4f}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
