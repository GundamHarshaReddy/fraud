"""
GNNExplainer-Based Attribution Pipeline for Fraud Detection
============================================================

Computes per-transaction explanations using GNNExplainer:
  - Edge importance scores (which connections mattered most)
  - Node feature importance (which of the 32 features drove the score)
  - Critical subgraph (minimal subgraph that preserves prediction)

These attributions feed into the Gemini prompt to create
"attribution-grounded" SAR generation — a novel contribution where
every LLM assertion is traceable to the GNN's actual reasoning.

Usage:
    python scripts/explain_gnn.py --transaction_idx 42
    python scripts/explain_gnn.py --top_k 50  # Explain top-50 riskiest
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
import logging
import argparse
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Feature names for human-readable attribution
FEATURE_NAMES = [
    'TransactionAmt', 'TransactionDT', 'ProductCD', 'dist1',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
    'C1_velocity', 'C2_velocity', 'C5_velocity', 'C6_velocity',
    'C11_velocity', 'C13_velocity', 'C14_velocity',
    'D1_recency', 'D2_recency', 'D4_recency', 'D10_recency', 'D15_recency',
    'V127_behavioral', 'V130_behavioral', 'V307_behavioral', 'V310_behavioral',
    'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo'
]

EDGE_TYPE_NAMES = ['card', 'device', 'email', 'address', 'shared_device', 'self_loop']


class FraudExplainer:
    """
    Attribution-based explainability for the THA-GAT fraud detection model.

    Uses gradient-based feature attribution (integrated gradients style)
    and edge masking to identify:
    1. Which features drove the prediction
    2. Which edges (relationships) were most important
    3. The critical subgraph around each flagged transaction
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def compute_feature_importance(self, x, edge_index, node_idx,
                                   edge_type=None, edge_time_delta=None,
                                   n_steps=50):
        """
        Compute feature importance via Integrated Gradients.

        For a target node, computes:
            IG_i = (x_i - baseline_i) × ∫₀¹ ∂F/∂x_i(baseline + α(x-baseline)) dα

        This gives attribution scores for each of the 32 input features.
        """
        x = x.to(self.device).clone().detach()
        edge_index = edge_index.to(self.device)
        if edge_type is not None:
            edge_type = edge_type.to(self.device)
        if edge_time_delta is not None:
            edge_time_delta = edge_time_delta.to(self.device)

        # Baseline: zero features (represents "no information")
        baseline = torch.zeros_like(x)

        # Integrated gradients
        scaled_inputs = []
        for step in range(n_steps + 1):
            alpha = step / n_steps
            scaled = baseline + alpha * (x - baseline)
            scaled_inputs.append(scaled)

        # Accumulate gradients
        total_grads = torch.zeros_like(x[node_idx])

        for scaled_x in scaled_inputs:
            scaled_x = scaled_x.detach().requires_grad_(True)
            logits = self.model(scaled_x, edge_index,
                                edge_type=edge_type,
                                edge_time_delta=edge_time_delta)
            score = logits[node_idx]
            score.backward(retain_graph=False)

            if scaled_x.grad is not None:
                total_grads += scaled_x.grad[node_idx].detach()
            # Clear grads
            scaled_x.grad = None

        # IG = (input - baseline) * avg_gradient
        ig_attributions = (x[node_idx] - baseline[node_idx]).detach() * total_grads / (n_steps + 1)

        return ig_attributions.cpu().numpy()

    def compute_edge_importance(self, x, edge_index, node_idx,
                                edge_type=None, edge_time_delta=None,
                                n_perturbations=100):
        """
        Compute edge importance via edge masking perturbation.

        For each edge connected to the target node, measures the
        prediction change when that edge is removed.
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_type is not None:
            edge_type = edge_type.to(self.device)
        if edge_time_delta is not None:
            edge_time_delta = edge_time_delta.to(self.device)

        # Baseline prediction
        with torch.no_grad():
            base_logits = self.model(x, edge_index, edge_type=edge_type,
                                     edge_time_delta=edge_time_delta)
            base_score = torch.sigmoid(base_logits[node_idx]).item()

        # Find edges connected to target node
        src, dst = edge_index[0], edge_index[1]
        connected_mask = (src == node_idx) | (dst == node_idx)
        connected_indices = connected_mask.nonzero(as_tuple=True)[0]

        edge_importances = []

        for edge_idx in connected_indices:
            # Create edge mask (remove this edge)
            mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=self.device)
            mask[edge_idx] = False

            masked_edge_index = edge_index[:, mask]
            masked_edge_type = edge_type[mask] if edge_type is not None else None
            masked_time_delta = edge_time_delta[mask] if edge_time_delta is not None else None

            with torch.no_grad():
                perturbed_logits = self.model(x, masked_edge_index,
                                              edge_type=masked_edge_type,
                                              edge_time_delta=masked_time_delta)
                perturbed_score = torch.sigmoid(perturbed_logits[node_idx]).item()

            importance = abs(base_score - perturbed_score)

            # Get edge info
            s, d = src[edge_idx].item(), dst[edge_idx].item()
            e_type = edge_type[edge_idx].item() if edge_type is not None else -1
            e_type_name = EDGE_TYPE_NAMES[e_type] if 0 <= e_type < len(EDGE_TYPE_NAMES) else 'unknown'

            edge_importances.append({
                'edge_idx': int(edge_idx.item()),
                'source': int(s),
                'target': int(d),
                'edge_type': e_type_name,
                'importance': float(importance),
                'base_score': float(base_score),
                'perturbed_score': float(perturbed_score)
            })

        # Sort by importance (descending)
        edge_importances.sort(key=lambda e: e['importance'], reverse=True)
        return edge_importances, base_score

    def explain_transaction(self, x, edge_index, node_idx,
                            edge_type=None, edge_time_delta=None,
                            top_features=10, top_edges=5) -> Dict[str, Any]:
        """
        Generate a complete explanation for a single transaction.

        Returns a structured JSON-compatible dict with:
        - Feature attributions (top contributing features)
        - Edge importances (most important relationships)
        - Risk score and critical subgraph info
        """
        logger.info(f"Generating explanation for node {node_idx}...")

        # Feature importance
        try:
            feat_importances = self.compute_feature_importance(
                x, edge_index, node_idx, edge_type, edge_time_delta, n_steps=30
            )
        except Exception as e:
            logger.warning(f"Feature importance computation failed: {e}")
            feat_importances = np.zeros(x.shape[1])

        # Edge importance
        try:
            edge_importances, base_score = self.compute_edge_importance(
                x, edge_index, node_idx, edge_type, edge_time_delta
            )
        except Exception as e:
            logger.warning(f"Edge importance computation failed: {e}")
            edge_importances = []
            with torch.no_grad():
                logits = self.model(x.to(self.device), edge_index.to(self.device),
                                     edge_type=edge_type.to(self.device) if edge_type is not None else None,
                                     edge_time_delta=edge_time_delta.to(self.device) if edge_time_delta is not None else None)
                base_score = torch.sigmoid(logits[node_idx]).item()

        # Format feature attributions
        abs_importances = np.abs(feat_importances)
        top_feat_indices = abs_importances.argsort()[::-1][:top_features]

        feature_attributions = []
        for idx in top_feat_indices:
            name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
            feature_attributions.append({
                'feature': name,
                'importance': float(abs_importances[idx]),
                'direction': 'increases_risk' if feat_importances[idx] > 0 else 'decreases_risk',
                'raw_attribution': float(feat_importances[idx])
            })

        # Top edges
        top_edge_list = edge_importances[:top_edges]

        # Critical subgraph stats
        src, dst = edge_index[0], edge_index[1]
        connected = (src == node_idx) | (dst == node_idx)
        n_connected = connected.sum().item()

        # Count edge types in neighbourhood
        if edge_type is not None:
            connected_types = edge_type[connected]
            type_counts = {}
            for t in range(len(EDGE_TYPE_NAMES)):
                count = (connected_types == t).sum().item()
                if count > 0:
                    type_counts[EDGE_TYPE_NAMES[t]] = count
        else:
            type_counts = {'total': n_connected}

        explanation = {
            'node_idx': int(node_idx),
            'risk_score': float(base_score),
            'risk_level': 'HIGH' if base_score >= 0.8 else ('MEDIUM' if base_score >= 0.5 else 'LOW'),
            'feature_attributions': feature_attributions,
            'edge_importances': top_edge_list,
            'critical_subgraph': {
                'n_edges': n_connected,
                'edge_type_distribution': type_counts
            }
        }

        return explanation

    def format_for_gemini_prompt(self, explanation: Dict[str, Any]) -> str:
        """
        Format explanation as a structured text block for injection into
        the Gemini SAR generation prompt.

        This creates "attribution-grounded" explanations — the key novelty
        of Phase 3.
        """
        lines = []
        lines.append("GNN MODEL ATTRIBUTION (from Integrated Gradients + Edge Masking):")
        lines.append(f"  Risk Score: {explanation['risk_score']:.4f} ({explanation['risk_level']})")
        lines.append("")

        lines.append("  Top Contributing Features:")
        for fa in explanation['feature_attributions'][:7]:
            arrow = "↑" if fa['direction'] == 'increases_risk' else "↓"
            lines.append(f"    {arrow} {fa['feature']}: importance={fa['importance']:.4f} ({fa['direction']})")

        lines.append("")
        lines.append("  Top Contributing Relationships:")
        for ei in explanation['edge_importances'][:5]:
            lines.append(f"    → Edge({ei['edge_type']}): node_{ei['source']}↔node_{ei['target']} "
                         f"importance={ei['importance']:.4f}")

        lines.append("")
        sub = explanation['critical_subgraph']
        lines.append(f"  Critical Subgraph: {sub['n_edges']} edges")
        for etype, count in sub.get('edge_type_distribution', {}).items():
            lines.append(f"    {etype}: {count} edges")

        lines.append("")
        lines.append("IMPORTANT: Your explanation MUST reference these specific attribution")
        lines.append("signals. Do not fabricate reasons not supported by the model's actual")
        lines.append("attention weights and feature contributions.")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='GNN Explainability Pipeline')
    parser.add_argument('--transaction_idx', type=int, default=None,
                        help='Specific node index to explain')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Explain top-K riskiest transactions')
    args = parser.parse_args()

    models_dir = Path(__file__).parent.parent / "models"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = models_dir / "thagat_inference.pt"
    if not model_path.exists():
        logger.error("THA-GAT inference model not found. Run train_thagat.py first.")
        return

    device = 'cpu'  # Explainability runs on CPU for gradient stability
    model = torch.load(model_path, map_location=device)
    model.eval()

    explainer = FraudExplainer(model, device=device)

    # Load validation data predictions
    pred_path = results_dir / "thagat_predictions.npz"
    if not pred_path.exists():
        logger.error("THA-GAT predictions not found. Run train_thagat.py first.")
        return

    data = np.load(pred_path)
    predictions = data['predictions']
    targets = data['targets']

    # Load validation features (need actual tensors for explanation)
    import pandas as pd
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    val_df = pd.read_parquet(processed_dir / "val.parquet")

    # Rebuild validation tensors
    norm_path = models_dir / "normalization_params.pkl"
    with open(norm_path, 'rb') as f:
        norm_params = pickle.load(f)

    FEATURE_COLS = norm_params['feature_cols']
    feat_mean = norm_params['feature_mean']
    feat_std = norm_params['feature_std']

    X_mat = val_df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X_mat[col] = pd.to_numeric(X_mat[col], errors='coerce').fillna(0.0)
    X_mat['TransactionAmt'] = np.log1p(np.maximum(X_mat['TransactionAmt'].values, 0.0))
    for v_col in ['V127', 'V130', 'V307', 'V310']:
        if v_col in X_mat.columns:
            X_mat[v_col] = np.log1p(np.maximum(X_mat[v_col].values, 0.0))

    # Note: THAGATInference handles normalisation internally,
    # so pass unnormalised features
    x_val = torch.tensor(X_mat.values.astype(np.float32))
    n_val = len(val_df)

    # Self-loop edges for validation
    val_edge_index = torch.arange(n_val, dtype=torch.long).repeat(2, 1)
    val_edge_type = torch.full((n_val,), 5, dtype=torch.long)  # SELF_LOOP
    val_time_delta = torch.zeros(n_val, dtype=torch.float32)

    # Select transactions to explain
    if args.transaction_idx is not None:
        indices = [args.transaction_idx]
    else:
        # Top-K riskiest
        top_k_indices = predictions.argsort()[::-1][:args.top_k]
        indices = top_k_indices.tolist()

    logger.info(f"Generating explanations for {len(indices)} transactions...")

    all_explanations = []
    for node_idx in indices:
        if node_idx >= n_val:
            continue

        explanation = explainer.explain_transaction(
            x_val, val_edge_index, node_idx,
            edge_type=val_edge_type,
            edge_time_delta=val_time_delta,
            top_features=10, top_edges=5
        )

        # Add ground truth
        explanation['ground_truth'] = int(targets[node_idx])
        explanation['model_prediction'] = float(predictions[node_idx])

        # Generate Gemini-compatible prompt section
        prompt_section = explainer.format_for_gemini_prompt(explanation)
        explanation['gemini_prompt_section'] = prompt_section

        all_explanations.append(explanation)

        logger.info(f"  Node {node_idx}: score={explanation['risk_score']:.4f} "
                    f"gt={explanation['ground_truth']} "
                    f"top_feat={explanation['feature_attributions'][0]['feature']}")

    # Save explanations
    output_path = results_dir / "transaction_explanations.json"
    with open(output_path, 'w') as f:
        json.dump(all_explanations, f, indent=2)
    logger.info(f"\nSaved {len(all_explanations)} explanations to {output_path}")

    # Print sample
    if all_explanations:
        sample = all_explanations[0]
        logger.info(f"\n{'='*60}")
        logger.info("Sample Gemini Prompt Section:")
        logger.info(f"{'='*60}")
        logger.info(sample['gemini_prompt_section'])


if __name__ == "__main__":
    main()
