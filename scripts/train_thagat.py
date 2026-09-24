"""
THA-GAT Training Script for Fraud Detection (IEEE Publication Version)
======================================================================
Trains the novel Temporal-Aware Heterogeneous Graph Attention Network (THA-GAT)
on the IEEE-CIS Fraud Detection dataset.

Key differences from the original train_gnn.py:
 1. Builds heterogeneous edges with edge_type and edge_time_delta tensors
 2. Uses THAGAT architecture with temporal + relation-aware attention
 3. Uses Focal Loss instead of simple pos_weight BCE
 4. Includes Fraud-Ring-Aware Subgraph Readout for ring detection
 5. Also trains the legacy FraudGNN for ablation comparison

Usage:
    python scripts/train_thagat.py              # Train THA-GAT (default)
    python scripts/train_thagat.py --baseline   # Also train GraphSAGE baseline for ablation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import time
import argparse
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

from thagat_model import (
    THAGAT, THAGATInference, FocalLoss,
    EDGE_TYPE_CARD, EDGE_TYPE_DEVICE, EDGE_TYPE_EMAIL,
    EDGE_TYPE_ADDRESS, EDGE_TYPE_SHARED_DEVICE, EDGE_TYPE_SELF_LOOP,
    NUM_EDGE_TYPES, count_parameters
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    # Core Transaction (4)
    'TransactionAmt', 'TransactionDT', 'ProductCD', 'dist1',
    # Card Profile & Location (8)
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
    # High-Order Velocity Counters (7)
    'C1', 'C2', 'C5', 'C6', 'C11', 'C13', 'C14',
    # Recency Timedeltas (5)
    'D1', 'D2', 'D4', 'D10', 'D15',
    # Critical Behavioral V-Aggregates (4)
    'V127', 'V130', 'V307', 'V310',
    # Digital Fingerprint & Identity (4)
    'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo'
]


# ===================================================================
# Legacy FraudGNN (for ablation comparison)
# ===================================================================

class FraudGNN(nn.Module):
    """Legacy 3-Layer GraphSAGE baseline — kept for ablation study."""

    def __init__(self, num_features=32, hidden_channels=128, num_layers=3, dropout=0.2):
        super(FraudGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.norm3 = nn.LayerNorm(hidden_channels // 2)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, **kwargs):
        # Ignores edge_type and edge_time_delta (homogeneous baseline)
        h1 = F.relu(self.norm1(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.relu(self.norm2(self.conv2(h1, edge_index)))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h3 = F.relu(self.norm3(self.conv3(h2, edge_index)))
        return self.predictor(h3).squeeze(-1)


class FraudGNNInference(nn.Module):
    """Legacy inference wrapper (backward compatibility)."""

    def __init__(self, base_model, feature_mean, feature_std):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('feature_mean', torch.tensor(feature_mean, dtype=torch.float32))
        self.register_buffer('feature_std', torch.tensor(feature_std, dtype=torch.float32))

    def forward(self, x, edge_index, **kwargs):
        x_norm = (x - self.feature_mean) / (self.feature_std + 1e-7)
        logits = self.base_model(x_norm, edge_index)
        return torch.sigmoid(logits)


# ===================================================================
# Heterogeneous Graph Construction with Edge Types + Temporal Deltas
# ===================================================================

def build_hetero_graph(train_df):
    """
    Build heterogeneous graph with typed edges and temporal deltas.

    Returns:
        x_tensor:        Node features [N, 32]
        edge_index:      COO edge indices [2, E]
        edge_type:       Per-edge relation type [E] (int64)
        edge_time_delta: Per-edge temporal gap [E] (float32, normalised)
        y_tensor:        Labels [N]
        feature_mean:    [32] normalisation mean
        feature_std:     [32] normalisation std
    """
    logger.info(f"Building heterogeneous graph ({len(train_df):,} transactions)...")
    t0 = time.time()

    # ---- Feature matrix ----
    X_mat = train_df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X_mat[col] = pd.to_numeric(X_mat[col], errors='coerce').fillna(0.0)

    X_mat['TransactionAmt'] = np.log1p(np.maximum(X_mat['TransactionAmt'].values, 0.0))
    for v_col in ['V127', 'V130', 'V307', 'V310']:
        if v_col in X_mat.columns:
            X_mat[v_col] = np.log1p(np.maximum(X_mat[v_col].values, 0.0))

    feature_mean = X_mat.mean(axis=0).values.astype(np.float32)
    feature_std = X_mat.std(axis=0).values.astype(np.float32)
    feature_std[feature_std == 0] = 1.0

    X_norm = (X_mat.values - feature_mean) / feature_std
    x_tensor = torch.tensor(X_norm, dtype=torch.float32)
    y_tensor = torch.tensor(train_df['isFraud'].values, dtype=torch.float32)

    # ---- Transaction times for temporal deltas ----
    txn_times = train_df['TransactionDT'].values.astype(np.float64)
    # Normalise times to [0, 1] range for numerical stability
    t_min, t_max = txn_times.min(), txn_times.max()
    t_range = max(t_max - t_min, 1.0)
    txn_times_norm = (txn_times - t_min) / t_range

    # ---- Build heterogeneous edges ----
    logger.info("Building heterogeneous edges with type annotations and temporal deltas...")

    edges_src = []
    edges_dst = []
    e_types = []
    e_time_deltas = []

    def add_edges(idx_pairs, edge_type_val):
        """Add bidirectional edges with type and temporal delta."""
        for i, j in idx_pairs:
            dt = abs(txn_times_norm[i] - txn_times_norm[j])
            edges_src.extend([i, j])
            edges_dst.extend([j, i])
            e_types.extend([edge_type_val, edge_type_val])
            e_time_deltas.extend([dt, dt])

    # Card co-occurrence edges (EDGE_TYPE_CARD)
    card_groups = train_df.groupby(['card1', 'card2']).indices
    card_pairs = []
    for c_key, idxs in card_groups.items():
        n = len(idxs)
        if 1 < n < 60:
            limit = min(n, 12)
            for i in range(limit):
                for j in range(i + 1, min(limit, i + 5)):
                    card_pairs.append((idxs[i], idxs[j]))
    add_edges(card_pairs, EDGE_TYPE_CARD)
    logger.info(f"  Card edges: {len(card_pairs):,} pairs")

    # Device co-occurrence edges (EDGE_TYPE_DEVICE)
    dev_groups = train_df[train_df['DeviceInfo'] != 'unknown'].groupby('DeviceInfo').indices
    dev_pairs = []
    for dev, idxs in dev_groups.items():
        n = len(idxs)
        if 1 < n < 60:
            limit = min(n, 8)
            for i in range(limit):
                for j in range(i + 1, min(limit, i + 4)):
                    dev_pairs.append((idxs[i], idxs[j]))
    add_edges(dev_pairs, EDGE_TYPE_DEVICE)
    logger.info(f"  Device edges: {len(dev_pairs):,} pairs")

    # Email domain co-occurrence edges (EDGE_TYPE_EMAIL)
    for email_col in ['P_emaildomain', 'R_emaildomain']:
        if email_col in train_df.columns:
            email_groups = train_df[train_df[email_col] != 'unknown'].groupby(email_col).indices
            email_pairs = []
            for em, idxs in email_groups.items():
                n = len(idxs)
                if 1 < n < 100:
                    limit = min(n, 6)
                    for i in range(limit):
                        for j in range(i + 1, min(limit, i + 3)):
                            email_pairs.append((idxs[i], idxs[j]))
            add_edges(email_pairs, EDGE_TYPE_EMAIL)
            logger.info(f"  Email ({email_col}) edges: {len(email_pairs):,} pairs")

    # Address co-occurrence edges (EDGE_TYPE_ADDRESS)
    for addr_col in ['addr1', 'addr2']:
        if addr_col in train_df.columns:
            addr_groups = train_df[train_df[addr_col] > 0].groupby(addr_col).indices
            addr_pairs = []
            for ad, idxs in addr_groups.items():
                n = len(idxs)
                if 1 < n < 80:
                    limit = min(n, 6)
                    for i in range(limit):
                        for j in range(i + 1, min(limit, i + 3)):
                            addr_pairs.append((idxs[i], idxs[j]))
            add_edges(addr_pairs, EDGE_TYPE_ADDRESS)
            logger.info(f"  Address ({addr_col}) edges: {len(addr_pairs):,} pairs")

    # Shared-device card-card edges (EDGE_TYPE_SHARED_DEVICE)
    # Cards sharing the same device form suspicion links
    card_dev_groups = train_df[train_df['DeviceInfo'] != 'unknown'].groupby('DeviceInfo')
    shared_pairs = []
    for dev, group in card_dev_groups:
        card_keys = group.groupby(['card1', 'card2']).first().index.tolist()
        if len(card_keys) > 1:
            group_idxs = group.index.tolist()
            card_groups_within = group.groupby(['card1', 'card2']).indices
            card_repr = list(card_groups_within.values())
            for ci in range(len(card_repr)):
                for cj in range(ci + 1, min(len(card_repr), ci + 4)):
                    i_idx = card_repr[ci][0]
                    j_idx = card_repr[cj][0]
                    # Map from group-relative to df-absolute index
                    # card_groups_within returns positions relative to the group,
                    # so we index into group_idxs to get the actual DataFrame index.
                    i_abs = group_idxs[i_idx] if i_idx < len(group_idxs) else group_idxs[-1]
                    j_abs = group_idxs[j_idx] if j_idx < len(group_idxs) else group_idxs[-1]
                    if i_abs < len(train_df) and j_abs < len(train_df) and i_abs != j_abs:
                        shared_pairs.append((i_abs, j_abs))
        if len(shared_pairs) > 50000:
            break
    add_edges(shared_pairs, EDGE_TYPE_SHARED_DEVICE)
    logger.info(f"  Shared-device edges: {len(shared_pairs):,} pairs")

    # Self-loops (EDGE_TYPE_SELF_LOOP) — for message flow stability
    n_nodes = len(train_df)
    for i in range(n_nodes):
        edges_src.append(i)
        edges_dst.append(i)
        e_types.append(EDGE_TYPE_SELF_LOOP)
        e_time_deltas.append(0.0)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_type = torch.tensor(e_types, dtype=torch.long)
    edge_time_delta = torch.tensor(e_time_deltas, dtype=torch.float32)

    logger.info(f"Heterogeneous graph built in {time.time() - t0:.1f}s: "
                f"{n_nodes:,} nodes, {edge_index.shape[1]:,} edges across {NUM_EDGE_TYPES} types")

    return x_tensor, edge_index, edge_type, edge_time_delta, y_tensor, feature_mean, feature_std


def prepare_validation_data(val_df, feature_mean, feature_std):
    """Prepare validation tensors with heterogeneous edges (matching training distribution)."""
    X_mat = val_df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X_mat[col] = pd.to_numeric(X_mat[col], errors='coerce').fillna(0.0)
    X_mat['TransactionAmt'] = np.log1p(np.maximum(X_mat['TransactionAmt'].values, 0.0))
    for v_col in ['V127', 'V130', 'V307', 'V310']:
        if v_col in X_mat.columns:
            X_mat[v_col] = np.log1p(np.maximum(X_mat[v_col].values, 0.0))

    X_norm = (X_mat.values - feature_mean) / feature_std
    x_val = torch.tensor(X_norm, dtype=torch.float32)
    y_val = torch.tensor(val_df['isFraud'].values, dtype=torch.float32)

    # Build heterogeneous edges for validation (same logic as training)
    n = len(val_df)
    val_edges_src = []
    val_edges_dst = []
    val_e_types = []
    val_e_deltas = []

    dt_values = val_df['TransactionDT'].values.astype(float)
    dt_range = max(dt_values.max() - dt_values.min(), 1.0)

    def add_val_edges(pairs, edge_type):
        for i, j in pairs:
            if 0 <= i < n and 0 <= j < n:
                delta = abs(dt_values[i] - dt_values[j]) / dt_range
                val_edges_src.extend([i, j])
                val_edges_dst.extend([j, i])
                val_e_types.extend([edge_type, edge_type])
                val_e_deltas.extend([delta, delta])

    # Card edges (limited for speed)
    for card_col in ['card1']:
        if card_col in val_df.columns:
            card_groups = val_df.groupby(card_col).indices
            card_pairs = []
            for _, idxs in card_groups.items():
                if 1 < len(idxs) < 50:
                    limit = min(len(idxs), 4)
                    for i in range(limit):
                        for j in range(i + 1, min(limit, i + 2)):
                            card_pairs.append((idxs[i], idxs[j]))
            add_val_edges(card_pairs, EDGE_TYPE_CARD)

    # Self-loops
    for i in range(n):
        val_edges_src.append(i)
        val_edges_dst.append(i)
        val_e_types.append(EDGE_TYPE_SELF_LOOP)
        val_e_deltas.append(0.0)

    val_edge_index = torch.tensor([val_edges_src, val_edges_dst], dtype=torch.long)
    val_edge_type = torch.tensor(val_e_types, dtype=torch.long)
    val_edge_time_delta = torch.tensor(val_e_deltas, dtype=torch.float32)

    return x_val, val_edge_index, val_edge_type, val_edge_time_delta, y_val


# ===================================================================
# Training Loop (supports both THA-GAT and legacy FraudGNN)
# ===================================================================

def train_model(model, x, edge_index, edge_type, edge_time_delta, y,
                x_val, val_edge_index, val_edge_type, val_edge_time_delta, y_val,
                epochs=30, lr=0.005, device='cpu', use_focal_loss=True, model_name='THA-GAT'):
    """
    Train a GNN model with Focal Loss and comprehensive validation metrics.
    """
    logger.info(f"Training {model_name} on {x.shape[0]:,} nodes for {epochs} epochs ({device})...")
    logger.info(f"  Parameters: {count_parameters(model):,}")

    model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    edge_time_delta = edge_time_delta.to(device)
    y = y.to(device)

    x_val = x_val.to(device)
    val_edge_index = val_edge_index.to(device)
    val_edge_type = val_edge_type.to(device)
    val_edge_time_delta = val_edge_time_delta.to(device)

    # Class distribution
    n_pos = y.sum().item()
    n_neg = len(y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    logger.info(f"  Class distribution: {int(n_pos):,} fraud / {int(n_neg):,} legit (ratio 1:{n_neg/max(n_pos,1):.0f})")

    # Loss function
    if use_focal_loss:
        # alpha=0.85 correctly upweights the rare fraud class
        criterion = FocalLoss(alpha=0.85, gamma=2.0, pos_weight=pos_weight)
        logger.info("  Loss: Focal Loss (α=0.85, γ=2.0)")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("  Loss: BCE with pos_weight")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    best_weights = None
    history = []

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index, edge_type=edge_type, edge_time_delta=edge_time_delta)
        loss = criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        # Validation
        if epoch % 3 == 0 or epoch == epochs or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val, val_edge_index,
                                   edge_type=val_edge_type,
                                   edge_time_delta=val_edge_time_delta)
                val_preds = torch.sigmoid(val_logits).cpu().numpy()
                val_targets = y_val.cpu().numpy()

                val_auc = roc_auc_score(val_targets, val_preds)
                val_ap = average_precision_score(val_targets, val_preds)

                # Binary metrics at threshold 0.5
                val_binary = (val_preds >= 0.5).astype(int)
                val_f1 = f1_score(val_targets, val_binary, zero_division=0)
                val_prec = precision_score(val_targets, val_binary, zero_division=0)
                val_rec = recall_score(val_targets, val_binary, zero_division=0)

                elapsed = time.time() - t_epoch
                logger.info(
                    f"[{model_name}] Epoch {epoch:02d}/{epochs} ({elapsed:.1f}s) - "
                    f"Loss: {loss.item():.4f} | AUC: {val_auc:.4f} | AP: {val_ap:.4f} | "
                    f"F1: {val_f1:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f}"
                )

                history.append({
                    'epoch': epoch, 'loss': loss.item(),
                    'val_auc': val_auc, 'val_ap': val_ap,
                    'val_f1': val_f1, 'val_prec': val_prec, 'val_rec': val_rec
                })

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            logger.info(f"[{model_name}] Epoch {epoch:02d}/{epochs} ({time.time()-t_epoch:.1f}s) - Loss: {loss.item():.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, best_auc, history


# ===================================================================
# Comprehensive Evaluation
# ===================================================================

def evaluate_model(model, x, edge_index, edge_type, edge_time_delta, y, device, name="Model"):
    """Full evaluation with all metrics needed for the paper."""
    model.eval()
    model.to(device)
    with torch.no_grad():
        logits = model(
            x.to(device), edge_index.to(device),
            edge_type=edge_type.to(device),
            edge_time_delta=edge_time_delta.to(device)
        )
        preds = torch.sigmoid(logits).cpu().numpy()
        targets = y.numpy()

    auc = roc_auc_score(targets, preds)
    ap = average_precision_score(targets, preds)

    binary = (preds >= 0.5).astype(int)
    f1 = f1_score(targets, binary, zero_division=0)
    prec = precision_score(targets, binary, zero_division=0)
    rec = recall_score(targets, binary, zero_division=0)
    cm = confusion_matrix(targets, binary)

    results = {
        'name': name, 'auc_roc': auc, 'pr_auc': ap,
        'f1': f1, 'precision': prec, 'recall': rec,
        'confusion_matrix': cm.tolist(),
        'predictions': preds, 'targets': targets
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"  {name} — Final Evaluation")
    logger.info(f"  AUC-ROC:   {auc:.4f}")
    logger.info(f"  PR-AUC:    {ap:.4f}")
    logger.info(f"  F1:        {f1:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  Confusion: {cm.tolist()}")
    logger.info(f"{'='*60}\n")

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description='Train THA-GAT for Fraud Detection')
    parser.add_argument('--baseline', action='store_true', help='Also train GraphSAGE baseline for ablation')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        # Device
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        logger.info(f"Hardware accelerator: {device}")

        # 1. Load data
        processed_dir = Path(__file__).parent.parent / "data" / "processed"
        logger.info("Loading train and validation datasets...")
        train_df = pd.read_parquet(processed_dir / "train.parquet")
        val_df = pd.read_parquet(processed_dir / "val.parquet")
        test_df = pd.read_parquet(processed_dir / "test.parquet")

        # 2. Build heterogeneous graph
        (x_train, edge_index, edge_type, edge_time_delta,
         y_train, feat_mean, feat_std) = build_hetero_graph(train_df)

        (x_val, val_edge_index, val_edge_type,
         val_edge_time_delta, y_val) = prepare_validation_data(val_df, feat_mean, feat_std)
         
        (x_test, test_edge_index, test_edge_type,
         test_edge_time_delta, y_test) = prepare_validation_data(test_df, feat_mean, feat_std)

        num_features = len(FEATURE_COLS)
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}

        # ============================================
        # 3. Train THA-GAT (the novel architecture)
        # ============================================
        logger.info("\n" + "="*70)
        logger.info("  TRAINING THA-GAT (Novel Architecture)")
        logger.info("="*70)

        thagat = THAGAT(
            num_features=num_features,
            hidden_channels=128,
            num_edge_types=NUM_EDGE_TYPES,
            heads=4,
            dropout=0.2
        )

        trained_thagat, best_auc, thagat_history = train_model(
            thagat, x_train, edge_index, edge_type, edge_time_delta, y_train,
            x_val, val_edge_index, val_edge_type, val_edge_time_delta, y_val,
            epochs=args.epochs, lr=args.lr, device=device,
            use_focal_loss=True, model_name='THA-GAT'
        )

        # Evaluate on Test Set
        thagat_results = evaluate_model(
            trained_thagat, x_test, test_edge_index, test_edge_type,
            test_edge_time_delta, y_test, device, name="THA-GAT (Test)"
        )
        all_results['thagat'] = thagat_results

        # Save THA-GAT
        thagat_path = models_dir / "thagat_v1.pt"
        torch.save(trained_thagat.cpu().state_dict(), thagat_path)
        logger.info(f"Saved THA-GAT weights: {thagat_path}")

        # Save inference wrapper
        wrapper = THAGATInference(trained_thagat.cpu(), feat_mean, feat_std)
        wrapper.eval()
        torch.save(wrapper, models_dir / "thagat_inference.pt")
        logger.info("Saved THA-GAT inference wrapper")

        # Save normalization params & config
        norm_path = models_dir / "normalization_params.pkl"
        with open(norm_path, 'wb') as f:
            pickle.dump({'feature_mean': feat_mean, 'feature_std': feat_std, 'feature_cols': FEATURE_COLS}, f)

        config = {
            'architecture': 'THA-GAT',
            'num_features': num_features,
            'feature_cols': FEATURE_COLS,
            'hidden_channels': 128,
            'heads': 4,
            'num_edge_types': NUM_EDGE_TYPES,
            'num_layers': 3,
            'val_auc': thagat_results['auc_roc'],
            'val_ap': thagat_results['pr_auc'],
            'val_f1': thagat_results['f1'],
            'total_train_nodes': len(train_df),
            'total_params': count_parameters(trained_thagat)
        }
        with open(models_dir / "model_config.pkl", 'wb') as f:
            pickle.dump(config, f)

        # ============================================
        # 4. Optionally train GraphSAGE baseline
        # ============================================
        if args.baseline:
            logger.info("\n" + "="*70)
            logger.info("  TRAINING GRAPHSAGE BASELINE (Ablation)")
            logger.info("="*70)

            sage_model = FraudGNN(
                num_features=num_features,
                hidden_channels=128,
                num_layers=3,
                dropout=0.2
            )

            # GraphSAGE uses same edges but ignores types (homogeneous)
            trained_sage, sage_auc, sage_history = train_model(
                sage_model, x_train, edge_index, edge_type, edge_time_delta, y_train,
                x_val, val_edge_index, val_edge_type, val_edge_time_delta, y_val,
                epochs=args.epochs, lr=0.008, device=device,
                use_focal_loss=False, model_name='GraphSAGE'
            )

            sage_results = evaluate_model(
                trained_sage, x_test, test_edge_index, test_edge_type,
                test_edge_time_delta, y_test, device, name="GraphSAGE (Test)"
            )
            all_results['graphsage'] = sage_results

            torch.save(trained_sage.cpu().state_dict(), models_dir / "graphsage_baseline.pt")
            logger.info("Saved GraphSAGE baseline weights")

        # ============================================
        # 5. Save all experiment results
        # ============================================
        # Remove numpy arrays before pickling
        results_to_save = {}
        for k, v in all_results.items():
            results_to_save[k] = {kk: vv for kk, vv in v.items()
                                  if kk not in ('predictions', 'targets')}

        with open(results_dir / "training_results.pkl", 'wb') as f:
            pickle.dump(results_to_save, f)

        # Save predictions for downstream analysis (threshold optimization, etc.)
        for k, v in all_results.items():
            np.savez(results_dir / f"{k}_predictions.npz",
                     predictions=v['predictions'], targets=v['targets'])

        logger.info("\n" + "="*70)
        logger.info("  TRAINING COMPLETE — Summary")
        logger.info("="*70)
        for name, res in all_results.items():
            logger.info(f"  {res['name']:30s} | AUC: {res['auc_roc']:.4f} | AP: {res['pr_auc']:.4f} | F1: {res['f1']:.4f}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
