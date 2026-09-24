import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_ensemble_predictions(xgboost_preds, thagat_preds, weights=[0.5, 0.5]):
    """Compute ensemble predictions from existing model outputs"""
    ensemble = weights[0] * xgboost_preds + weights[1] * thagat_preds
    return ensemble

def main():
    results_dir = Path(__file__).parent.parent / "results"
    
    seeds = [42, 43, 44]
    
    for seed in seeds:
        logger.info(f"Loading predictions for seed {seed}...")
        try:
            xgb_data = np.load(results_dir / f"{seed}_xgboost_predictions.npz")
            thagat_data = np.load(results_dir / f"{seed}_thagat_predictions.npz")
        except FileNotFoundError:
            logger.warning(f"Missing predictions for seed {seed}, skipping.")
            continue
            
        xgb_preds = xgb_data['predictions']
        thagat_preds = thagat_data['predictions']
        targets = xgb_data['targets']
        
        logger.info(f"Computing ensemble predictions (50/50 split) for seed {seed}...")
        ensemble_preds = compute_ensemble_predictions(xgb_preds, thagat_preds)
        
        # Calculate metrics
        auc = roc_auc_score(targets, ensemble_preds)
        ap = average_precision_score(targets, ensemble_preds)
        
        binary = (ensemble_preds >= 0.5).astype(int)
        f1 = f1_score(targets, binary, zero_division=0)
        prec = precision_score(targets, binary, zero_division=0)
        rec = recall_score(targets, binary, zero_division=0)
        
        logger.info(f"--- Ensemble Results (Seed {seed}) ---")
        logger.info(f"AUC: {auc:.4f} | PR-AUC: {ap:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        
        # Save ensemble predictions
        save_path = results_dir / f"{seed}_ensemble_predictions.npz"
        np.savez(save_path, predictions=ensemble_preds, targets=targets)
        logger.info(f"Saved ensemble predictions to {save_path}")

if __name__ == "__main__":
    main()
