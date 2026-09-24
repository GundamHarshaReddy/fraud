"""
Cost-Optimized Escalation Threshold Optimization
=================================================

Formalizes the GNN→GenAI escalation as a constrained optimization:

    Minimize:  C_total = C_gnn · N + C_genai · |{i : s_i ≥ θ}|
    Subject to: Recall(θ) ≥ R_min

Sweeps the escalation threshold θ and finds the Pareto-optimal operating
point on the cost-recall frontier.

This is a distinct publishable contribution: no existing paper quantifies
this cost-accuracy tradeoff for hybrid GNN+LLM fraud detection.

Usage:
    python scripts/optimize_threshold.py
    python scripts/optimize_threshold.py --r_min 0.90  # 90% recall constraint
"""

import numpy as np
import pickle
from pathlib import Path
import logging
import argparse
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cost assumptions (USD per operation)
COST_GNN_PER_TXN = 0.0001    # ~$0.10 per 1000 transactions (compute)
COST_GENAI_PER_TXN = 0.012   # ~$0.012 per Gemini Flash API call
DAILY_TRANSACTION_VOLUME = 1_000_000  # Assumed daily volume for cost projection


def load_predictions(results_dir):
    """Load saved predictions from training."""
    pred_files = list(results_dir.glob("*_predictions.npz"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {results_dir}")

    results = {}
    for f in pred_files:
        name = f.stem.replace("_predictions", "")
        data = np.load(f)
        results[name] = {
            'predictions': data['predictions'],
            'targets': data['targets']
        }
    return results


def compute_metrics_at_threshold(predictions, targets, theta):
    """Compute fraud detection metrics at a given escalation threshold."""
    flagged = predictions >= theta
    n_flagged = flagged.sum()
    n_total = len(predictions)
    escalation_rate = n_flagged / n_total

    # Among actual frauds, how many are escalated?
    fraud_mask = targets == 1
    n_fraud = fraud_mask.sum()

    if n_fraud == 0:
        return {'theta': theta, 'recall': 0.0, 'escalation_rate': escalation_rate,
                'n_escalated': int(n_flagged), 'precision_at_theta': 0.0}

    true_pos = (flagged & fraud_mask).sum()
    recall = true_pos / n_fraud
    precision = true_pos / max(n_flagged, 1)

    return {
        'theta': float(theta),
        'recall': float(recall),
        'precision_at_theta': float(precision),
        'escalation_rate': float(escalation_rate),
        'n_escalated': int(n_flagged),
        'n_fraud_caught': int(true_pos),
        'n_fraud_total': int(n_fraud),
    }


def compute_cost(n_total, n_escalated):
    """Compute total cost for a given number of escalations."""
    c_gnn = COST_GNN_PER_TXN * n_total
    c_genai = COST_GENAI_PER_TXN * n_escalated
    return c_gnn, c_genai, c_gnn + c_genai


def sweep_thresholds(predictions, targets, thresholds=None):
    """Sweep across thresholds and compute metrics + costs."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.99, 0.01)

    results = []
    n_total = len(predictions)

    for theta in thresholds:
        metrics = compute_metrics_at_threshold(predictions, targets, theta)
        c_gnn, c_genai, c_total = compute_cost(n_total, metrics['n_escalated'])

        # Project to daily cost
        scale = DAILY_TRANSACTION_VOLUME / max(n_total, 1)
        daily_cost_genai = c_genai * scale
        daily_cost_total = c_total * scale
        daily_cost_all_genai = COST_GENAI_PER_TXN * DAILY_TRANSACTION_VOLUME  # Baseline: send everything

        metrics.update({
            'cost_gnn': float(c_gnn),
            'cost_genai': float(c_genai),
            'cost_total': float(c_total),
            'daily_cost_genai_usd': float(daily_cost_genai),
            'daily_cost_total_usd': float(daily_cost_total),
            'daily_cost_all_genai_usd': float(daily_cost_all_genai),
            'cost_reduction_pct': float(max(0, 1.0 - daily_cost_genai / daily_cost_all_genai) * 100),
        })
        results.append(metrics)

    return results


def find_optimal_threshold(sweep_results, r_min=0.95):
    """
    Find the Pareto-optimal threshold θ* that minimizes cost
    while satisfying Recall(θ*) ≥ R_min.
    """
    # Filter to only thresholds meeting the recall constraint
    feasible = [r for r in sweep_results if r['recall'] >= r_min]

    if not feasible:
        logger.warning(f"No threshold achieves recall ≥ {r_min:.2f}. "
                       f"Best recall: {max(r['recall'] for r in sweep_results):.4f}")
        # Fall back to the threshold with highest recall
        feasible = sorted(sweep_results, key=lambda r: r['recall'], reverse=True)[:1]

    # Among feasible, pick the one with minimum cost
    optimal = min(feasible, key=lambda r: r['cost_total'])
    return optimal


def main():
    parser = argparse.ArgumentParser(description='Optimize escalation threshold')
    parser.add_argument('--r_min', type=float, default=0.95, help='Minimum required fraud recall')
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "results"

    try:
        all_predictions = load_predictions(results_dir)
    except FileNotFoundError:
        logger.error("No prediction files found. Run train_thagat.py first.")
        return

    for model_name, data in all_predictions.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Threshold Optimization for: {model_name}")
        logger.info(f"{'='*60}")

        preds = data['predictions']
        targets = data['targets']

        # Sweep
        sweep = sweep_thresholds(preds, targets)

        # Find optimal
        optimal = find_optimal_threshold(sweep, r_min=args.r_min)

        logger.info(f"\n  Optimal threshold θ*: {optimal['theta']:.2f}")
        logger.info(f"  Recall at θ*:         {optimal['recall']:.4f}")
        logger.info(f"  Precision at θ*:      {optimal['precision_at_theta']:.4f}")
        logger.info(f"  Escalation rate:      {optimal['escalation_rate']:.4f} ({optimal['n_escalated']:,} / {len(preds):,})")
        logger.info(f"  Daily GenAI cost:     ${optimal['daily_cost_genai_usd']:.2f}")
        logger.info(f"  Cost reduction:       {optimal['cost_reduction_pct']:.1f}% vs all-GenAI baseline")

        # Compare with baseline scenarios
        all_genai_cost = COST_GENAI_PER_TXN * DAILY_TRANSACTION_VOLUME
        gnn_only_cost = COST_GNN_PER_TXN * DAILY_TRANSACTION_VOLUME

        logger.info(f"\n  Cost Comparison (daily, {DAILY_TRANSACTION_VOLUME:,} txns):")
        logger.info(f"  {'Strategy':<30s} {'Cost':>12s} {'Recall':>8s}")
        logger.info(f"  {'-'*50}")
        logger.info(f"  {'All-GenAI (no GNN)':<30s} ${all_genai_cost:>10,.2f} {'1.0000':>8s}")
        logger.info(f"  {'GNN-only (no GenAI)':<30s} ${gnn_only_cost:>10,.2f} {'N/A':>8s}")
        logger.info(f"  {'Hybrid at θ*':<30s} ${optimal['daily_cost_total_usd']:>10,.2f} {optimal['recall']:>8.4f}")

        # Save results
        output = {
            'model': model_name,
            'optimal_threshold': optimal['theta'],
            'r_min_constraint': args.r_min,
            'sweep_results': sweep,
            'optimal_result': optimal,
            'cost_assumptions': {
                'cost_gnn_per_txn': COST_GNN_PER_TXN,
                'cost_genai_per_txn': COST_GENAI_PER_TXN,
                'daily_volume': DAILY_TRANSACTION_VOLUME
            }
        }

        output_path = results_dir / f"threshold_optimization_{model_name}.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"\n  Saved: {output_path}")

        # Save optimal threshold as a loadable config for consumer.py
        config_path = results_dir / "optimal_threshold.json"
        with open(config_path, 'w') as f:
            json.dump({
                'high_risk_threshold': optimal['theta'],
                'medium_risk_threshold': max(0.3, optimal['theta'] - 0.3),
                'model': model_name,
                'recall_at_threshold': optimal['recall'],
                'cost_reduction_pct': optimal['cost_reduction_pct']
            }, f, indent=2)
        logger.info(f"  Saved operational config: {config_path}")


if __name__ == "__main__":
    main()
