"""
Model Export Script for Inference
- Exports trained GNN model for fast loading in serving
- Supports TorchScript and ONNX export formats
- Creates inference wrapper with preprocessing
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import sys
import pickle

# Add parent directory to path to import training module
sys.path.append(str(Path(__file__).parent))

from train_gnn import FraudGNN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudGNNInference(nn.Module):
    """Inference wrapper for FraudGNN with preprocessing"""
    
    def __init__(self, base_model, feature_mean, feature_std):
        super(FraudGNNInference, self).__init__()
        self.base_model = base_model
        self.register_buffer('feature_mean', torch.tensor(feature_mean, dtype=torch.float32))
        self.register_buffer('feature_std', torch.tensor(feature_std, dtype=torch.float32))
    
    def forward(self, x, edge_index):
        """Forward pass with normalization"""
        # Normalize features
        x_normalized = (x - self.feature_mean) / (self.feature_std + 1e-8)
        
        # Pass through base model
        logits = self.base_model(x_normalized, edge_index)
        
        # Apply sigmoid for probability
        probabilities = torch.sigmoid(logits)
        
        return probabilities

def load_trained_model(model_path, num_features, device='cpu'):
    """Load the trained model"""
    logger.info(f"Loading trained model from {model_path}...")
    
    # Initialize model architecture
    model = FraudGNN(
        num_features=num_features,
        hidden_channels=128,
        num_layers=3,
        dropout=0.0,  # No dropout during inference
        use_attention=True,
        heads=4
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    logger.info("Model loaded successfully")
    return model

def calculate_normalization_params(data):
    """Calculate feature normalization parameters"""
    logger.info("Calculating normalization parameters...")
    
    feature_mean = data.x.mean(dim=0).cpu().numpy()
    feature_std = data.x.std(dim=0).cpu().numpy()
    
    logger.info(f"Feature mean: {feature_mean}")
    logger.info(f"Feature std: {feature_std}")
    
    return feature_mean, feature_std

def export_torchscript(model, output_path, sample_input, sample_edge_index):
    """Export model to TorchScript for faster inference"""
    logger.info(f"Exporting model to TorchScript: {output_path}")
    
    model.eval()
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(model, (sample_input, sample_edge_index))
        traced_model.save(output_path)
        logger.info("TorchScript export successful")
        return True
    except Exception as e:
        logger.warning(f"TorchScript export failed: {e}")
        return False

def export_onnx(model, output_path, sample_input, sample_edge_index):
    """Export model to ONNX format"""
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    model.eval()
    
    try:
        torch.onnx.export(
            model,
            (sample_input, sample_edge_index),
            output_path,
            export_params=True,
            opset_version=12,
            input_names=['features', 'edge_index'],
            output_names=['predictions'],
            dynamic_axes={
                'features': {0: 'num_nodes'},
                'edge_index': {1: 'num_edges'},
                'predictions': {0: 'num_nodes'}
            }
        )
        logger.info("ONNX export successful")
        return True
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
        return False

def save_inference_artifacts(model, feature_mean, feature_std, output_dir):
    """Save inference artifacts including normalization params"""
    logger.info(f"Saving inference artifacts to {output_dir}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save normalization parameters
    norm_params = {
        'feature_mean': feature_mean,
        'feature_std': feature_std
    }
    
    with open(output_dir / 'normalization_params.pkl', 'wb') as f:
        pickle.dump(norm_params, f)
    
    # Save model config
    model_config = {
        'num_features': model.base_model.convs[0].in_channels,
        'hidden_channels': model.base_model.convs[0].out_channels,
        'num_layers': model.base_model.num_layers,
        'use_attention': True,  # Simplified for export
        'dropout': model.base_model.dropout
    }
    
    # Only save if not already saved
    config_path = output_dir / 'model_config.pkl'
    if not config_path.exists():
        with open(config_path, 'wb') as f:
            pickle.dump(model_config, f)
    
    logger.info("Inference artifacts saved successfully")

def main():
    """Main export pipeline"""
    try:
        # Set device
        device = torch.device('cpu')  # Use CPU for export compatibility
        logger.info(f"Using device: {device}")
        
        # Paths
        model_path = Path(__file__).parent.parent / "models" / "gnn_fraud_v1.pt"
        output_dir = Path(__file__).parent.parent / "models"
        
        # For simplicity, we'll use a fixed number of features
        # In production, this should be loaded from the training data
        num_features = 2  # TransactionAmt, TransactionDT
        
        # Update the model configuration to match actual features
        model_config = {
            'num_features': num_features,
            'hidden_channels': 128,
            'num_layers': 3,
            'use_attention': True,
            'dropout': 0.0
        }
        
        # Save model config first
        with open(output_dir / 'model_config.pkl', 'wb') as f:
            pickle.dump(model_config, f)
        
        # Load trained model
        base_model = load_trained_model(model_path, num_features, device=device)
        
        # Create sample input for tracing (adjust based on your actual data)
        sample_input = torch.randn(100, num_features).to(device)
        sample_edge_index = torch.randint(0, 100, (2, 200)).to(device)
        
        # Calculate normalization parameters (using dummy values for now)
        feature_mean = [100.0, 100000.0]  # Example: mean amount, mean time
        feature_std = [50.0, 50000.0]    # Example: std amount, std time
        
        # Create inference wrapper
        inference_model = FraudGNNInference(base_model, feature_mean, feature_std)
        inference_model.eval()
        
        # Export to TorchScript
        torchscript_path = output_dir / "gnn_fraud_v1_torchscript.pt"
        torchscript_success = export_torchscript(inference_model, torchscript_path, 
                                                  sample_input, sample_edge_index)
        
        # Export to ONNX
        onnx_path = output_dir / "gnn_fraud_v1.onnx"
        onnx_success = export_onnx(inference_model, onnx_path, sample_input, sample_edge_index)
        
        # Save inference artifacts
        save_inference_artifacts(inference_model, feature_mean, feature_std, output_dir)
        
        # Also save the PyTorch model as fallback
        torch.save(inference_model.state_dict(), output_dir / "gnn_fraud_inference.pt")
        
        logger.info("Model export completed successfully")
        logger.info(f"TorchScript: {'✓' if torchscript_success else '✗'}")
        logger.info(f"ONNX: {'✓' if onnx_success else '✗'}")
        
    except Exception as e:
        logger.error(f"Error in model export: {e}")
        raise

if __name__ == "__main__":
    main()
