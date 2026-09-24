"""
Data Cleaning and Preprocessing Script (FraudGuard-GNN 32-Feature Schema)
- Extracts curated top-32 dense features from IEEE-CIS Fraud Detection dataset
- Handles missing values and categorical factorizations
- Generates train/validation/test chronological splits
- Outputs cleaned parquet files for GNN training and streaming pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CURATED_32_FEATURES = [
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

def load_data():
    """Load curated transaction and identity columns efficiently"""
    logger.info("Loading curated 32-feature datasets from raw files...")
    
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    
    txn_cols = [
        'TransactionID', 'isFraud', 'TransactionAmt', 'TransactionDT', 'ProductCD', 'dist1',
        'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
        'C1', 'C2', 'C5', 'C6', 'C11', 'C13', 'C14',
        'D1', 'D2', 'D4', 'D10', 'D15',
        'V127', 'V130', 'V307', 'V310',
        'P_emaildomain', 'R_emaildomain'
    ]
    
    id_cols = ['TransactionID', 'DeviceType', 'DeviceInfo']
    
    transactions = pd.read_csv(data_dir / "train_transaction.csv", usecols=txn_cols)
    identity = pd.read_csv(data_dir / "train_identity.csv", usecols=id_cols)
    
    logger.info(f"Transactions shape: {transactions.shape}")
    logger.info(f"Identity shape: {identity.shape}")
    
    return transactions, identity

def merge_data(transactions, identity):
    """Merge transaction and identity data on TransactionID"""
    logger.info("Merging datasets on TransactionID...")
    merged = transactions.merge(identity, on='TransactionID', how='left')
    logger.info(f"Merged shape: {merged.shape}")
    return merged

def handle_missing_and_encode(df):
    """Handle missing values for 32 curated features.
    
    NOTE: Categorical columns are kept as strings here and NOT factorized.
    Factorization happens downstream in the training scripts (train_thagat.py,
    baselines.py) via pd.to_numeric(errors='coerce'), which correctly operates
    within each temporal split to prevent encoding leakage (M1).
    """
    logger.info("Handling missing values and encoding categoricals...")
    
    categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo']
    
    # Fill categoricals with 'unknown' — keep as strings for downstream per-split encoding
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str)
            
    # Fill numeric columns with 0.0
    numeric_cols = [c for c in df.columns if c not in categorical_cols and c not in ['TransactionID', 'isFraud']]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    return df

def main():
    """Main preprocessing pipeline"""
    try:
        transactions, identity = load_data()
        merged = merge_data(transactions, identity)
        cleaned = handle_missing_and_encode(merged)
        
        # Chronological sort strictly by TransactionDT to prevent lookahead leakage
        cleaned = cleaned.sort_values('TransactionDT').reset_index(drop=True)
        
        output_dir = Path(__file__).parent.parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full cleaned dataset (splitting is handled by 03_split.py)
        full_path = output_dir / "transactions_clean.parquet"
        cleaned.to_parquet(full_path, index=False)
        logger.info(f"Saved full cleaned data to {full_path} (Shape: {cleaned.shape})")
        logger.info(f"Overall Fraud rate: {cleaned['isFraud'].mean():.4f}")
        logger.info("32-feature dataset preprocessing completed successfully!")
        logger.info("Next step: run 03_split.py to create train/val/test splits")
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

