def train_rgcn(X_train, y_train, X_val, y_val, train_df, val_df, feat_mean, feat_std):
    import torch
    import torch.nn as nn
    from torch_geometric.nn import RGCNConv
    from train_thagat import build_hetero_graph, prepare_validation_data
    import time
    from baselines import evaluate_predictions
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Training RGCN baseline...")
    t0 = time.time()

    class RGCNBaseline(nn.Module):
        def __init__(self, in_dim, hidden=128, num_relations=6, dropout=0.2):
            super().__init__()
            self.conv1 = RGCNConv(in_dim, hidden, num_relations=num_relations)
            self.conv2 = RGCNConv(hidden, hidden, num_relations=num_relations)
            self.conv3 = RGCNConv(hidden, hidden // 2, num_relations=num_relations)
            self.pred = nn.Sequential(
                nn.Linear(hidden // 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
            )

        def forward(self, x, edge_index, edge_type, **kwargs):
            h = torch.relu(self.conv1(x, edge_index, edge_type))
            h = torch.relu(self.conv2(h, edge_index, edge_type))
            h = torch.relu(self.conv3(h, edge_index, edge_type))
            return self.pred(h).squeeze(-1)

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RGCNBaseline(X_train.shape[1]).to(device)

    # Build hetero graph for RGCN (it needs edge_type)
    _, train_ei, train_et, _, _, _, _ = build_hetero_graph(train_df)
    _, val_ei, val_et, _, _ = prepare_validation_data(val_df, feat_mean, feat_std)

    x_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_vl = torch.tensor(X_val, dtype=torch.float32).to(device)
    ei_tr, et_tr = train_ei.to(device), train_et.to(device)
    ei_vl, et_vl = val_ei.to(device), val_et.to(device)

    n_pos = y_tr.sum().item()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    best_auc = 0
    best_preds = None
    from sklearn.metrics import roc_auc_score

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tr, ei_tr, et_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 29:
            model.eval()
            with torch.no_grad():
                val_preds = torch.sigmoid(model(x_vl, ei_vl, et_vl)).cpu().numpy()
                auc = roc_auc_score(y_val, val_preds)
                if auc > best_auc:
                    best_auc = auc
                    best_preds = val_preds.copy()
                logger.info(f"  RGCN Epoch {epoch+1}/30 - Loss: {loss.item():.4f} - Val AUC: {auc:.4f}")

    elapsed = time.time() - t0
    logger.info(f"RGCN trained in {elapsed:.1f}s")
    
    if best_preds is None:
        model.eval()
        with torch.no_grad():
            best_preds = torch.sigmoid(model(x_vl, ei_vl, et_vl)).cpu().numpy()

    results = evaluate_predictions("RGCN (heterogeneous)", y_val, best_preds)
    return results, best_preds
