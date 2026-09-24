"""
Full Experiment Suite for IEEE Paper
=====================================

Master script that orchestrates all experiments needed for the paper:

    Experiment 1: Model Comparison Table (THA-GAT vs all baselines)
    Experiment 2: Ablation Study (remove each THA-GAT component)
    Experiment 3: Cost-Accuracy Tradeoff (escalation threshold sweep)
    Experiment 4: Latency Benchmarks
    Experiment 5: Summary statistics and LaTeX table generation

Usage:
    python scripts/run_experiments.py              # Run all experiments
    python scripts/run_experiments.py --exp 1 3     # Run specific experiments
"""

import numpy as np
import pickle
import json
import time
import logging
import argparse
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_all_predictions(results_dir):
    """Load predictions from all trained models."""
    predictions = {}
    for npz_file in results_dir.glob("*_predictions.npz"):
        name = npz_file.stem.replace("_predictions", "")
        # Handle seeded runs (e.g., "42_thagat")
        parts = name.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            seed = parts[0]
            model_name = parts[1]
        else:
            seed = "default"
            model_name = name
            
        data = np.load(npz_file)
        if model_name not in predictions:
            predictions[model_name] = {}
        predictions[model_name][seed] = {
            'predictions': data['predictions'],
            'targets': data['targets']
        }
    return predictions


def compute_all_metrics(targets, predictions, name="Model"):
    """Compute comprehensive metrics for paper tables."""
    auc = roc_auc_score(targets, predictions)
    ap = average_precision_score(targets, predictions)

    # Metrics at threshold 0.5
    binary_05 = (predictions >= 0.5).astype(int)
    f1_05 = f1_score(targets, binary_05, zero_division=0)
    prec_05 = precision_score(targets, binary_05, zero_division=0)
    rec_05 = recall_score(targets, binary_05, zero_division=0)

    # Precision@K (top-K precision)
    k_values = [100, 500, 1000]
    prec_at_k = {}
    sorted_indices = predictions.argsort()[::-1]
    for k in k_values:
        top_k = sorted_indices[:k]
        prec_at_k[k] = targets[top_k].mean()

    # Recall at 95% precision (find threshold)
    precisions, recalls, thresholds = precision_recall_curve(targets, predictions)
    recall_at_95_prec = 0.0
    for p, r in zip(precisions, recalls):
        if p >= 0.95:
            recall_at_95_prec = max(recall_at_95_prec, r)

    cm = confusion_matrix(targets, binary_05)

    return {
        'name': name,
        'auc_roc': float(auc),
        'pr_auc': float(ap),
        'f1': float(f1_05),
        'precision': float(prec_05),
        'recall': float(rec_05),
        'precision_at_100': float(prec_at_k.get(100, 0)),
        'precision_at_500': float(prec_at_k.get(500, 0)),
        'precision_at_1000': float(prec_at_k.get(1000, 0)),
        'recall_at_95_prec': float(recall_at_95_prec),
        'confusion_matrix': cm.tolist()
    }


# ===================================================================
# Experiment 1: Model Comparison
# ===================================================================

def experiment_1_model_comparison(all_predictions, results_dir):
    """Generate the main model comparison table with mean ± std."""
    logger.info("\n" + "="*70)
    logger.info("  EXPERIMENT 1: Model Comparison (Multi-Seed Aggregation)")
    logger.info("="*70)

    display_names = {
        'xgboost': 'XGBoost (no graph)',
        'mlp': 'MLP (no graph)',
        'gat': 'GAT (homogeneous)',
        'graphsage': 'GraphSAGE (baseline)',
        'rgcn': 'RGCN (heterogeneous)',
        'thagat': 'THA-GAT (proposed)',
        'ensemble': 'XGBoost + THA-GAT Ensemble'
    }

    results = []
    for model_name, seeds_data in all_predictions.items():
        if model_name not in display_names:
            continue
            
        display = display_names.get(model_name, model_name)
        
        all_metrics = []
        for seed, data in seeds_data.items():
            metrics = compute_all_metrics(data['targets'], data['predictions'], display)
            all_metrics.append(metrics)
            
        if not all_metrics:
            continue

        # Aggregate across seeds
        agg_metrics = {'name': display}
        for key in ['auc_roc', 'pr_auc', 'f1', 'precision_at_100', 'recall_at_95_prec']:
            vals = [m[key] for m in all_metrics]
            agg_metrics[key] = np.mean(vals)
            agg_metrics[f'{key}_std'] = np.std(vals)
            
        results.append(agg_metrics)

    # Sort: proposed model last
    results.sort(key=lambda r: 0 if 'proposed' in r['name'] else 1, reverse=True)

    # Print table
    header = f"{'Model':<30s} {'AUC-ROC':>15s} {'PR-AUC':>15s} {'F1':>13s} {'P@100':>13s} {'R@95P':>13s}"
    logger.info(f"\n  {header}")
    logger.info(f"  {'-' * len(header)}")
    for r in results:
        marker = " ★" if 'proposed' in r['name'] else ""
        logger.info(f"  {r['name']:<30s} {r['auc_roc']:.4f}±{r['auc_roc_std']:.4f} {r['pr_auc']:.4f}±{r['pr_auc_std']:.4f} "
                    f"{r['f1']:.4f}±{r['f1_std']:.4f} {r['precision_at_100']:.4f}±{r['precision_at_100_std']:.4f} {r['recall_at_95_prec']:.4f}±{r['recall_at_95_prec_std']:.4f}{marker}")

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[t]",
        r"\caption{Performance comparison of fraud detection models on IEEE-CIS dataset (out-of-time test set). Reported as Mean $\pm$ Std across 3 random seeds.}",
        r"\label{tab:comparison}",
        r"\centering",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{AUC-ROC} & \textbf{PR-AUC} & \textbf{F1} & \textbf{P@100} & \textbf{Recall@95\%P} \\",
        r"\midrule",
    ]
    for r in results:
        bold = r"\textbf{" if 'proposed' in r['name'] else ""
        close = "}" if bold else ""
        latex_lines.append(
            f"{bold}{r['name']}{close} & "
            f"${r['auc_roc']:.4f} \\pm {r['auc_roc_std']:.4f}$ & "
            f"${r['pr_auc']:.4f} \\pm {r['pr_auc_std']:.4f}$ & "
            f"${r['f1']:.4f} \\pm {r['f1_std']:.4f}$ & "
            f"${r['precision_at_100']:.4f} \\pm {r['precision_at_100_std']:.4f}$ & "
            f"${r['recall_at_95_prec']:.4f} \\pm {r['recall_at_95_prec_std']:.4f}$ \\\\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    latex_path = results_dir / "table_comparison.tex"
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))
    logger.info(f"  LaTeX table saved: {latex_path}")

    # Save JSON
    with open(results_dir / "experiment_1_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ===================================================================
# Experiment 2: Ablation Study
# ===================================================================

def experiment_2_ablation(all_predictions, results_dir):
    """
    Generate ablation table.
    Compare: THA-GAT vs GraphSAGE (removes all novelty)
             THA-GAT vs GAT (removes temporal + hetero)
    Note: Full ablation (removing individual components) requires
    retraining variants — this generates the table structure.
    """
    logger.info("\n" + "="*70)
    logger.info("  EXPERIMENT 2: Ablation Study")
    logger.info("="*70)

    ablation_models = {
        'thagat': 'THA-GAT (full)',
        'graphsage': '− temporal attn − hetero attn − ring readout (GraphSAGE)',
        'gat': '− temporal attn − hetero attn (standard GAT)',
        'mlp': '− graph structure entirely (MLP)',
    }

    results = []
    thagat_auc = None

    for model_name, ablation_label in ablation_models.items():
        if model_name in all_predictions:
            seeds_data = all_predictions[model_name]
            all_metrics = []
            for seed, data in seeds_data.items():
                metrics = compute_all_metrics(data['targets'], data['predictions'], ablation_label)
                all_metrics.append(metrics)
                
            if not all_metrics:
                continue

            agg_metrics = {'name': ablation_label}
            for key in ['auc_roc', 'pr_auc']:
                vals = [m[key] for m in all_metrics]
                agg_metrics[key] = np.mean(vals)
                agg_metrics[f'{key}_std'] = np.std(vals)

            if model_name == 'thagat':
                thagat_auc = agg_metrics['auc_roc']
                
            if thagat_auc and model_name != 'thagat':
                agg_metrics['delta_auc'] = agg_metrics['auc_roc'] - thagat_auc
            else:
                agg_metrics['delta_auc'] = 0.0
                
            results.append(agg_metrics)

    logger.info(f"\n  {'Variant':<55s} {'AUC-ROC':>15s} {'Δ AUC':>8s} {'PR-AUC':>15s}")
    logger.info(f"  {'-'*95}")
    for r in results:
        logger.info(f"  {r['name']:<55s} {r['auc_roc']:.4f}±{r['auc_roc_std']:.4f} {r['delta_auc']:>+8.4f} {r['pr_auc']:.4f}±{r['pr_auc_std']:.4f}")

    with open(results_dir / "experiment_2_ablation.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ===================================================================
# Experiment 3: Cost-Accuracy Tradeoff
# ===================================================================

def experiment_3_cost_tradeoff(all_predictions, results_dir):
    """Load and summarise the threshold optimization results."""
    logger.info("\n" + "="*70)
    logger.info("  EXPERIMENT 3: Cost-Accuracy Tradeoff")
    logger.info("="*70)

    for model_name, seeds_data in all_predictions.items():
        # Get the first seed's optimization results for simplicity
        seed = list(seeds_data.keys())[0]
        opt_path = results_dir / f"threshold_optimization_{model_name}.json"
        if opt_path.exists():
            with open(opt_path) as f:
                opt_data = json.load(f)

            opt = opt_data['optimal_result']
            logger.info(f"\n  {model_name}:")
            logger.info(f"    Optimal θ*:      {opt['theta']:.2f}")
            logger.info(f"    Recall:          {opt['recall']:.4f}")
            logger.info(f"    Escalation rate: {opt['escalation_rate']:.4f}")
            logger.info(f"    Cost reduction:  {opt['cost_reduction_pct']:.1f}%")
        else:
            logger.info(f"  {model_name}: Run optimize_threshold.py first")


# ===================================================================
# Experiment 4: Latency Benchmarks
# ===================================================================

def experiment_4_latency(results_dir):
    """Benchmark inference latency for each model."""
    logger.info("\n" + "="*70)
    logger.info("  EXPERIMENT 4: Latency Benchmarks")
    logger.info("="*70)

    import torch
    models_dir = Path(__file__).parent.parent / "models"

    benchmarks = {}

    # THA-GAT inference
    thagat_path = models_dir / "thagat_inference.pt"
    if thagat_path.exists():
        model = torch.load(thagat_path, map_location='cpu')
        model.eval()

        # Single transaction
        x_single = torch.randn(1, 32)
        ei_single = torch.zeros(2, 1, dtype=torch.long)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(x_single, ei_single)

        # Benchmark single
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x_single, ei_single)
            times.append((time.perf_counter() - t0) * 1000)

        benchmarks['thagat_single_ms'] = float(np.median(times))

        # Batch inference
        for batch_size in [32, 64, 256]:
            x_batch = torch.randn(batch_size, 32)
            ei_batch = torch.arange(batch_size, dtype=torch.long).repeat(2, 1)

            times = []
            for _ in range(50):
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(x_batch, ei_batch)
                times.append((time.perf_counter() - t0) * 1000)

            per_txn = np.median(times) / batch_size
            benchmarks[f'thagat_batch{batch_size}_ms'] = float(np.median(times))
            benchmarks[f'thagat_batch{batch_size}_per_txn_ms'] = float(per_txn)

    # Print results
    logger.info(f"\n  {'Component':<40s} {'Latency':>12s} {'Throughput':>15s}")
    logger.info(f"  {'-'*67}")

    if 'thagat_single_ms' in benchmarks:
        lat = benchmarks['thagat_single_ms']
        logger.info(f"  {'THA-GAT inference (single)':<40s} {lat:>10.2f}ms {1000/lat:>12.0f} txn/s")

    for bs in [32, 64, 256]:
        key = f'thagat_batch{bs}_ms'
        if key in benchmarks:
            lat = benchmarks[key]
            per = benchmarks[f'thagat_batch{bs}_per_txn_ms']
            logger.info(f"  {f'THA-GAT inference (batch {bs})':<40s} {lat:>10.2f}ms {bs*1000/lat:>12.0f} txn/s")

    # Estimated Gemini latency (from known API characteristics)
    benchmarks['gemini_api_estimated_ms'] = 1500.0
    logger.info(f"  {'Gemini API call (estimated)':<40s} {'~1500.00ms':>12s} {'~0.67 txn/s':>15s}")

    # Save
    with open(results_dir / "experiment_4_latency.json", 'w') as f:
        json.dump(benchmarks, f, indent=2)

    return benchmarks


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', type=int, default=[1, 2, 3, 4],
                        help='Experiments to run (1-4)')
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = load_all_predictions(results_dir)

    if not all_predictions:
        logger.error("No prediction files found. Run train_thagat.py and baselines.py first.")
        return

    logger.info(f"Found predictions for: {list(all_predictions.keys())}")

    if 1 in args.exp:
        experiment_1_model_comparison(all_predictions, results_dir)

    if 2 in args.exp:
        experiment_2_ablation(all_predictions, results_dir)

    if 3 in args.exp:
        experiment_3_cost_tradeoff(all_predictions, results_dir)

    if 4 in args.exp:
        experiment_4_latency(results_dir)

    logger.info("\n" + "="*70)
    logger.info("  ALL EXPERIMENTS COMPLETE")
    logger.info(f"  Results saved to: {results_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
