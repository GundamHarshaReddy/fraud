"""
Ablation Study for THA-GAT
==========================

Evaluates variants of THA-GAT:
1. No Temporal Decay (lambda = 0)
2. No Fraud-Ring Readout (alpha = 1.0)
3. Homogeneous Edges Only (Collapse all edge types to 0)

Usage:
    python scripts/ablation_study.py
"""

import torch
import torch.nn as nn
import time
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
import json

from train_thagat import build_hetero_graph, prepare_validation_data
from thagat_model import THAGAT, FocalLoss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_ablation_variant(variant_name, train_data, val_data, epochs=10):
    x_tr, ei_tr, et_tr, ed_tr, y_tr, f_mean, f_std = train_data
    x_vl, ei_vl, et_vl, ed_vl, y_vl = val_data
    
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Apply variant specific modifications
    ablate_temporal = (variant_name == 'no_temporal')
    ablate_readout = (variant_name == 'no_readout')
    ablate_hetero = (variant_name == 'homogeneous')
    
    if ablate_hetero:
        et_tr = torch.zeros_like(et_tr)
        et_vl = torch.zeros_like(et_vl)
        
    if ablate_temporal:
        ed_tr = torch.zeros_like(ed_tr)
        ed_vl = torch.zeros_like(ed_vl)
        
    model = THAGAT(
        num_features=32, 
        hidden_channels=128, 
        heads=4, 
        num_edge_types=6,
        ablate_readout=ablate_readout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)], device=device)
    criterion = FocalLoss(alpha=0.85, gamma=2.0, pos_weight=pos_weight)
    
    # Move to device
    x_tr, ei_tr, et_tr, ed_tr, y_tr = [t.to(device) for t in (x_tr, ei_tr, et_tr, ed_tr, y_tr)]
    x_vl, ei_vl, et_vl, ed_vl = [t.to(device) for t in (x_vl, ei_vl, et_vl, ed_vl)]
    
    best_auc = 0.0
    best_recall = 0.0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tr, ei_tr, et_tr, ed_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        
        if epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(x_vl, ei_vl, et_vl, ed_vl)
                val_preds = torch.sigmoid(val_logits).cpu().numpy()
                auc = roc_auc_score(y_vl.numpy(), val_preds)
                binary = (val_preds > 0.5).astype(int)
                rec = recall_score(y_vl.numpy(), binary)
                best_auc = auc
                best_recall = rec
                
    return {'variant': variant_name, 'auc': float(best_auc), 'recall': float(best_recall)}


def main():
    logger.info("Starting Ablation Study...")
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    
    # Load full data for ablation study
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    
    train_data = build_hetero_graph(train_df)
    f_mean, f_std = train_data[-2:]
    val_data = prepare_validation_data(val_df, f_mean, f_std)
    
    results = []
    for variant in ['full', 'no_temporal', 'no_readout', 'homogeneous']:
        logger.info(f"Training variant: {variant}")
        res = train_ablation_variant(variant, train_data, val_data, epochs=10)
        logger.info(f"Result for {variant}: AUC={res['auc']:.4f}, Recall={res['recall']:.4f}")
        results.append(res)
        
    with open('results/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
