"""
Publication-Quality Figure Generator
======================================

Generates all figures needed for the IEEE paper:
    1. ROC curves (all models overlaid)
    2. Precision-Recall curves
    3. Cost-Recall Pareto frontier
    4. Threshold sensitivity plot
    5. Latency comparison bar chart
    6. Training convergence curves

Usage:
    python scripts/plot_results.py
"""

import numpy as np
import json
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server/CI
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed. pip install matplotlib")


# Publication style
STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
}

COLORS = {
    'thagat': '#e63946',      # Bold red (proposed)
    'graphsage': '#457b9d',   # Steel blue
    'gat': '#2a9d8f',         # Teal
    'xgboost': '#e9c46a',     # Gold
    'mlp': '#8d99ae',         # Grey
}

DISPLAY_NAMES = {
    'thagat': 'THA-GAT (proposed)',
    'graphsage': 'GraphSAGE',
    'gat': 'GAT (homogeneous)',
    'xgboost': 'XGBoost (no graph)',
    'mlp': 'MLP (no graph)',
}


def load_all_predictions(results_dir):
    predictions = {}
    for f in results_dir.glob("*_predictions.npz"):
        name = f.stem.replace("_predictions", "")
        data = np.load(f)
        predictions[name] = {
            'predictions': data['predictions'],
            'targets': data['targets']
        }
    return predictions


# ===================================================================
# Figure 1: ROC Curves
# ===================================================================

def plot_roc_curves(all_predictions, output_dir):
    """Plot ROC curves for all models on same figure."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Plot order: proposed model last (on top)
    order = ['mlp', 'xgboost', 'gat', 'graphsage', 'thagat']
    plotted = []

    for model_name in order:
        if model_name not in all_predictions:
            continue
        data = all_predictions[model_name]
        fpr, tpr, _ = roc_curve(data['targets'], data['predictions'])
        auc_val = auc(fpr, tpr)

        color = COLORS.get(model_name, '#333')
        display = DISPLAY_NAMES.get(model_name, model_name)
        lw = 2.5 if model_name == 'thagat' else 1.5
        ls = '-' if model_name == 'thagat' else '--'

        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                label=f'{display} (AUC={auc_val:.4f})')
        plotted.append(model_name)

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # ax.set_title('ROC Curves — Fraud Detection Models')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.2)

    path = output_dir / "fig_roc_curves.pdf"
    fig.savefig(path)
    fig.savefig(output_dir / "fig_roc_curves.png")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


# ===================================================================
# Figure 2: Precision-Recall Curves
# ===================================================================

def plot_pr_curves(all_predictions, output_dir):
    """Plot Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    order = ['mlp', 'xgboost', 'gat', 'graphsage', 'thagat']

    for model_name in order:
        if model_name not in all_predictions:
            continue
        data = all_predictions[model_name]
        precision, recall, _ = precision_recall_curve(data['targets'], data['predictions'])
        ap = auc(recall, precision)

        color = COLORS.get(model_name, '#333')
        display = DISPLAY_NAMES.get(model_name, model_name)
        lw = 2.5 if model_name == 'thagat' else 1.5
        ls = '-' if model_name == 'thagat' else '--'

        ax.plot(recall, precision, color=color, lw=lw, ls=ls,
                label=f'{display} (AP={ap:.4f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    # ax.set_title('Precision-Recall Curves — Fraud Detection')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.2)

    path = output_dir / "fig_pr_curves.pdf"
    fig.savefig(path)
    fig.savefig(output_dir / "fig_pr_curves.png")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


# ===================================================================
# Figure 3: Cost-Recall Pareto Frontier
# ===================================================================

def plot_cost_recall(results_dir, output_dir):
    """Plot the cost-recall Pareto frontier from threshold optimization."""
    opt_file = results_dir / "threshold_optimization_ensemble.json"
    if not opt_file.exists():
        logger.info("  Skipping cost-recall plot (no ensemble optimization results)")
        return

    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax2 = ax1.twinx()

    with open(opt_file) as f:
        opt_data = json.load(f)

    sweep = opt_data['sweep_results']
    thetas = [r['theta'] for r in sweep]
    recalls = [r['recall'] for r in sweep]
    costs = [r['cost_reduction_pct'] for r in sweep]
    escalation_rates = [r['escalation_rate'] * 100 for r in sweep]

    # Recall on left axis
    ax1.plot(thetas, recalls, color='#e63946', lw=2, label='Fraud Recall')
    ax1.fill_between(thetas, recalls, alpha=0.1, color='#e63946')

    # Cost reduction on right axis
    ax2.plot(thetas, costs, color='#457b9d', lw=2, ls='--', label='Cost Reduction %')

    # Mark optimal point
    opt = opt_data['optimal_result']
    ax1.axvline(x=opt['theta'], color='#2a9d8f', ls=':', lw=1.5, alpha=0.7)
    ax1.scatter([opt['theta']], [opt['recall']], color='#2a9d8f', s=80, zorder=5,
                label=f'θ*={opt["theta"]:.2f}')

    ax1.set_xlabel('Escalation Threshold (θ)')
    ax1.set_ylabel('Fraud Recall', color='#e63946')
    ax2.set_ylabel('GenAI Cost Reduction (%)', color='#457b9d')
    # ax1.set_title('Cost-Recall Tradeoff — Ensemble Escalation')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.98, 0.65), framealpha=0.9)

    ax1.grid(True, alpha=0.2)
    ax1.set_xlim([0, 1])

    path = output_dir / "fig_cost_recall.pdf"
    fig.savefig(path)
    fig.savefig(output_dir / "fig_cost_recall.png")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


# ===================================================================
# Figure 4: Latency Comparison
# ===================================================================

def plot_latency(results_dir, output_dir):
    """Bar chart comparing inference latency."""
    latency_path = results_dir / "experiment_4_latency.json"
    if not latency_path.exists():
        logger.info("  Skipping latency plot (run experiment 4 first)")
        return

    with open(latency_path) as f:
        benchmarks = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    components = []
    latencies = []
    colors = []

    if 'thagat_single_ms' in benchmarks:
        components.append('THA-GAT\n(single)')
        latencies.append(benchmarks['thagat_single_ms'])
        colors.append('#e63946')

    if 'thagat_batch64_ms' in benchmarks:
        components.append('THA-GAT\n(batch 64)')
        latencies.append(benchmarks['thagat_batch64_per_txn_ms'])
        colors.append('#457b9d')

    if 'gemini_api_estimated_ms' in benchmarks:
        components.append('Gemini API\n(estimated)')
        latencies.append(benchmarks['gemini_api_estimated_ms'])
        colors.append('#e9c46a')

    if not components:
        return

    bars = ax.barh(components, latencies, color=colors, edgecolor='white', height=0.5)
    ax.set_xlabel('Latency (ms)')
    # ax.set_title('Inference Latency Comparison')
    ax.set_xscale('log')

    # Add value labels
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}ms', va='center', fontsize=9)

    ax.grid(True, axis='x', alpha=0.2)

    path = output_dir / "fig_latency.pdf"
    fig.savefig(path)
    fig.savefig(output_dir / "fig_latency.png")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def main():
    if not HAS_MPL:
        logger.error("matplotlib required. pip install matplotlib")
        return

    plt.rcParams.update(STYLE)

    results_dir = Path(__file__).parent.parent / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = load_all_predictions(results_dir)
    if not all_predictions:
        logger.error("No prediction files found. Run training scripts first.")
        return

    logger.info(f"Generating figures for models: {list(all_predictions.keys())}")

    plot_roc_curves(all_predictions, figures_dir)
    plot_pr_curves(all_predictions, figures_dir)
    plot_cost_recall(results_dir, figures_dir)
    plot_latency(results_dir, figures_dir)

    logger.info(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
