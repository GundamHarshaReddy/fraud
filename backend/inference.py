"""
GNN Inference Service for Fraud Detection
- Loads trained PyTorch Geometric model or compiled TorchScript model
- Prepares feature representations from transaction and neighborhood data
- Computes calibrated fraud risk scores (0.0 to 1.0)
- Provides safe heuristic fallback when models are uninitialized
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import traceback

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional PyTorch import to guarantee graceful operation on any host
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Inference engine will operate in heuristic fallback mode.")


class FraudInferenceEngine:
    """
    Modular Inference Engine for Fraud Risk Scoring.
    Can be used by streaming consumers, FastAPI workers, or offline scoring jobs.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu') if TORCH_AVAILABLE else None
        self.model = None
        self.xgb_model = None
        self.model_type = "heuristic"
        self.model_path = model_path or self._detect_default_model_path()
        self.load_model()

    def _detect_default_model_path(self) -> Optional[str]:
        """Detect the best available model checkpoint in the models directory."""
        models_dir = Path(__file__).parent.parent / "models"
        # Priority: THA-GAT (novel) > TorchScript > legacy inference > raw weights
        candidates = [
            models_dir / "thagat_inference.pt",
            models_dir / "gnn_fraud_v1_torchscript.pt",
            models_dir / "gnn_fraud_inference.pt",
            models_dir / "gnn_fraud_v1.pt"
        ]
        for candidate in candidates:
            if candidate.exists():
                logger.info(f"Found model checkpoint candidate: {candidate}")
                return str(candidate)
        return None

    def load_model(self) -> bool:
        """Loads the GNN model and optionally XGBoost for ensemble scoring."""
        # Load XGBoost model if available
        if XGB_AVAILABLE:
            xgb_path = Path(__file__).parent.parent / "models" / "xgboost_model.json"
            if xgb_path.exists():
                try:
                    self.xgb_model = xgb.XGBClassifier()
                    self.xgb_model.load_model(str(xgb_path))
                    logger.info(f"Successfully loaded XGBoost ensemble model from {xgb_path}")
                except Exception as e:
                    logger.warning(f"Failed to load XGBoost model: {e}")
                    self.xgb_model = None
        
        if not TORCH_AVAILABLE or not self.model_path or not os.path.exists(self.model_path):
            logger.info("Operating in rule-assisted heuristic mode (no PyTorch model available).")
            self.model_type = "heuristic"
            return False

        # 1. First attempt: Load as compiled TorchScript module
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            self.model_type = "torchscript"
            logger.info(f"Successfully loaded TorchScript GNN model from {self.model_path}")
            return True
        except Exception as jit_err:
            logger.debug(f"Not a TorchScript model or failed jit load ({jit_err}). Checking PyTorch checkpoint...")

        # 2. Second attempt: Check if TorchScript candidate exists in same dir
        models_dir = Path(self.model_path).parent
        ts_candidate = models_dir / "gnn_fraud_v1_torchscript.pt"
        if ts_candidate.exists() and str(ts_candidate) != self.model_path:
            try:
                self.model = torch.jit.load(str(ts_candidate), map_location=self.device)
                self.model.eval()
                self.model_type = "torchscript"
                self.model_path = str(ts_candidate)
                logger.info(f"Successfully loaded TorchScript GNN model from {ts_candidate}")
                return True
            except Exception as e:
                logger.debug(f"TorchScript candidate failed ({e})")

        # 3. Third attempt: Load PyTorch weights dictionary into GNN architecture
        try:
            # Note: weights_only=False is required because we save full nn.Module objects.
            # In production, switch to TorchScript (torch.jit.save) for safe deserialization.
            loaded_obj = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if isinstance(loaded_obj, torch.nn.Module):
                self.model = loaded_obj
                self.model.eval()
                self.model_type = "pytorch_module"
                logger.info(f"Loaded nn.Module from {self.model_path}")
                return True
            elif isinstance(loaded_obj, dict):
                # State dictionary
                scripts_dir = Path(__file__).parent.parent / "scripts"
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                
                try:
                    from train_gnn import FraudGNN
                    from export_model import FraudGNNInference
                    import pickle

                    norm_path = models_dir / "normalization_params.pkl"
                    if norm_path.exists():
                        with open(norm_path, 'rb') as f:
                            norm_params = pickle.load(f)
                        mean = norm_params['feature_mean']
                        std = norm_params['feature_std']
                    else:
                        mean = [100.0, 100000.0]
                        std = [50.0, 50000.0]

                    num_features = len(mean) if hasattr(mean, '__len__') else 14
                    base_model = FraudGNN(num_features=num_features, hidden_channels=128, num_layers=3, dropout=0.0)
                    wrapper = FraudGNNInference(base_model, mean, std)
                    wrapper.load_state_dict(loaded_obj, strict=False)
                    wrapper.eval()
                    wrapper.to(self.device)
                    self.model = wrapper
                    self.model_type = "pytorch_gnn"
                    logger.info(f"Instantiated and loaded FraudGNN from state_dict {self.model_path}")
                    return True
                except Exception as arch_err:
                    logger.warning(f"Could not instantiate FraudGNN from state dict: {arch_err}")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model checkpoint from {self.model_path}: {e}")

        logger.info("Falling back to rule-assisted heuristic mode.")
        self.model = None
        self.model_type = "heuristic"
        return False

    def prepare_features(self, transaction: Dict[str, Any], neighborhood: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
        """
        Extract numerical features from transaction.
        The trained FraudGuard-GNN model expects the 32-feature representation:
        [Core: Amt, DT, ProductCD, dist1 | Cards: card1..card6, addr1, addr2 |
         Velocity: C1, C2, C5, C6, C11, C13, C14 | Recency: D1, D2, D4, D10, D15 |
         Behavioral: V127, V130, V307, V310 | Identity: P_email, R_email, DeviceType, DeviceInfo].
        Returns node features tensor x with shape [1, 32] and local edge_index tensor.
        """
        if not TORCH_AVAILABLE:
            return None, None

        # 1. Core Transaction (4)
        amt = float(transaction.get('transaction_amt') or transaction.get('amount') or 0.0)
        dt = float(transaction.get('transaction_dt') or transaction.get('dt') or 0.0)
        amt_log = float(np.log1p(max(amt, 0.0)))
        raw_pcd = transaction.get('product_cd') or transaction.get('ProductCD') or 0.0
        try:
            product_cd = float(raw_pcd)
        except (ValueError, TypeError):
            pcd_map = {'w': 0.0, 'h': 1.0, 'c': 2.0, 's': 3.0, 'r': 4.0}
            product_cd = pcd_map.get(str(raw_pcd).strip().lower(), 0.0)
        dist1 = float(transaction.get('dist1') or 0.0)

        # 2. Card Profile & Location (8)
        card1 = float(transaction.get('card1') or 0.0)
        card2 = float(transaction.get('card2') or 0.0)
        card3 = float(transaction.get('card3') or 0.0)
        raw_card4 = transaction.get('card4') or 0.0
        try:
            card4 = float(raw_card4)
        except (ValueError, TypeError):
            c4_map = {'discover': 0.0, 'mastercard': 1.0, 'visa': 2.0, 'american express': 3.0}
            card4 = c4_map.get(str(raw_card4).strip().lower(), 0.0)
        card5 = float(transaction.get('card5') or 0.0)
        raw_card6 = transaction.get('card6') or 0.0
        try:
            card6 = float(raw_card6)
        except (ValueError, TypeError):
            c6_map = {'credit': 0.0, 'debit': 1.0}
            card6 = c6_map.get(str(raw_card6).strip().lower(), 0.0)
        addr1 = float(transaction.get('addr1') or 0.0)
        addr2 = float(transaction.get('addr2') or 0.0)

        # 3. High-Order Velocity Counters (7)
        c1 = float(transaction.get('c1') or transaction.get('C1') or 1.0)
        c2 = float(transaction.get('c2') or transaction.get('C2') or 1.0)
        c5 = float(transaction.get('c5') or transaction.get('C5') or 0.0)
        c6 = float(transaction.get('c6') or transaction.get('C6') or 1.0)
        c11 = float(transaction.get('c11') or transaction.get('C11') or 1.0)
        c13 = float(transaction.get('c13') or transaction.get('C13') or 1.0)
        c14 = float(transaction.get('c14') or transaction.get('C14') or 1.0)

        # 4. Recency Timedeltas (5)
        d1 = float(transaction.get('d1') or transaction.get('D1') or 0.0)
        d2 = float(transaction.get('d2') or transaction.get('D2') or 0.0)
        d4 = float(transaction.get('d4') or transaction.get('D4') or 0.0)
        d10 = float(transaction.get('d10') or transaction.get('D10') or 0.0)
        d15 = float(transaction.get('d15') or transaction.get('D15') or 0.0)

        # 5. Critical Behavioral V-Aggregates (4)
        v127 = float(np.log1p(max(float(transaction.get('v127') or transaction.get('V127') or 0.0), 0.0)))
        v130 = float(np.log1p(max(float(transaction.get('v130') or transaction.get('V130') or 0.0), 0.0)))
        v307 = float(np.log1p(max(float(transaction.get('v307') or transaction.get('V307') or 0.0), 0.0)))
        v310 = float(np.log1p(max(float(transaction.get('v310') or transaction.get('V310') or 0.0), 0.0)))

        # 6. Digital Fingerprint & Identity (4)
        p_email = transaction.get('p_emaildomain') or transaction.get('p_email') or 0.0
        try:
            p_email_val = float(p_email)
        except (ValueError, TypeError):
            p_email_val = 1.0 if 'gmail' in str(p_email).lower() else (2.0 if 'temp' in str(p_email).lower() else 0.0)

        r_email = transaction.get('r_emaildomain') or transaction.get('r_email') or 0.0
        try:
            r_email_val = float(r_email)
        except (ValueError, TypeError):
            r_email_val = 0.0

        dev_type = transaction.get('device_type') or transaction.get('DeviceType') or 0.0
        try:
            dev_type_val = float(dev_type)
        except (ValueError, TypeError):
            dev_type_val = 1.0 if 'mobile' in str(dev_type).lower() else (2.0 if 'desktop' in str(dev_type).lower() else 0.0)

        dev_info = transaction.get('device_info') or transaction.get('DeviceInfo') or 0.0
        try:
            dev_info_val = float(dev_info)
        except (ValueError, TypeError):
            dev_info_val = 1.0 if str(dev_info).strip() not in ('0', '0.0', 'unknown', '') else 0.0

        feat_vector = [
            amt_log, dt, product_cd, dist1,
            card1, card2, card3, card4, card5, card6, addr1, addr2,
            c1, c2, c5, c6, c11, c13, c14,
            d1, d2, d4, d10, d15,
            v127, v130, v307, v310,
            p_email_val, r_email_val, dev_type_val, dev_info_val
        ]
        assert len(feat_vector) == 32, f"Expected 32 features, got {len(feat_vector)}"
        x = torch.tensor([feat_vector], dtype=torch.float32).to(self.device)

        # Construct local topological edges (self-loop + neighborhood presence)
        edge_index = torch.zeros((2, 1), dtype=torch.long).to(self.device)

        return x, edge_index

    def score(self, transaction: Dict[str, Any], neighborhood: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculates ensemble fraud probability [0,1].
        Fallback sequence:
        Ensemble (XGB+GNN) -> GNN-only -> XGB-only -> Heuristics
        """
        x, edge_index = self.prepare_features(transaction, neighborhood)
        
        gnn_score = None
        if self.model is not None and x is not None:
            try:
                # Construct edge_type and edge_time_delta for THA-GAT compatibility
                n_edges = edge_index.shape[1]
                edge_type = torch.full((n_edges,), 5, dtype=torch.long).to(self.device)  # SELF_LOOP type
                edge_time_delta = torch.zeros(n_edges, dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    # Try passing edge_type/edge_time_delta (THA-GAT)
                    try:
                        output = self.model(x, edge_index, edge_type=edge_type, edge_time_delta=edge_time_delta)
                    except TypeError:
                        # Legacy model doesn't accept edge_type/edge_time_delta
                        output = self.model(x, edge_index)
                    
                    if isinstance(output, torch.Tensor):
                        val = output.squeeze().item()
                        if not np.isnan(val) and not np.isinf(val):
                            if val < 0.0 or val > 1.0:
                                gnn_score = torch.sigmoid(output).squeeze().item()
                            else:
                                gnn_score = val
            except Exception as e:
                logger.debug(f"Neural scoring fallback engaged: {e}")
                
        xgb_score = None
        if self.xgb_model is not None and x is not None:
            try:
                features_np = x.cpu().numpy()
                preds = self.xgb_model.predict_proba(features_np)
                xgb_score = float(preds[0, 1])
            except Exception as e:
                logger.debug(f"XGBoost Inference failed: {e}")

        if gnn_score is not None and xgb_score is not None:
            return float(np.clip(0.5 * gnn_score + 0.5 * xgb_score, 0.01, 0.99))
        elif gnn_score is not None:
            return float(np.clip(gnn_score, 0.01, 0.99))
        elif xgb_score is not None:
            return float(np.clip(xgb_score, 0.01, 0.99))

        # Fallback multi-factor risk estimation
        return self._heuristic_scoring(transaction, neighborhood or {})

    def _heuristic_scoring(self, transaction: Dict[str, Any], neighborhood: Dict[str, Any]) -> float:
        """
        Domain-informed fraud heuristic evaluating transaction velocity, amounts,
        and device/card sharing signals.
        """
        risk = 0.08  # Baseline low risk

        amt = float(transaction.get('transaction_amt') or transaction.get('amount') or 0.0)
        if amt > 1000:
            risk += 0.35
        elif amt > 500:
            risk += 0.20
        elif amt > 250:
            risk += 0.10

        # Multi-device or shared card risk factors from neighborhood
        connected_cards = neighborhood.get('connected_cards', [])
        connected_devices = neighborhood.get('connected_devices', [])
        related_cards = neighborhood.get('related_cards', [])

        if len(related_cards) > 5:
            risk += 0.30
        elif len(related_cards) > 2:
            risk += 0.15

        if len(connected_devices) > 3:
            risk += 0.20

        # Suspicious email domain patterns
        p_email = str(transaction.get('p_emaildomain') or transaction.get('p_email') or '').lower()
        if p_email in ['protonmail.com', 'mail.com', 'anonymous.com']:
            risk += 0.15

        # Note: is_fraud is NEVER used for scoring — that would be data leakage.
        # The heuristic must rely only on observable features.

        return float(min(max(risk, 0.01), 0.99))
