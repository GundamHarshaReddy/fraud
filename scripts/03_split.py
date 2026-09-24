"""
Train/Validation/Test Split Script
- Performs time-based split (critical for fraud detection)
- Splits data by TransactionDT to prevent temporal leakage
- Outputs split datasets for GNN training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_cleaned_data():
    """Load the cleaned transaction data"""
    logger.info("Loading cleaned data...")
    
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    df = pd.read_parquet(data_dir / "transactions_clean.parquet")
    
    logger.info(f"Loaded data shape: {df.shape}")
    return df

def time_based_split(df, train_ratio=0.7, val_ratio=0.15):
    """
    Perform time-based split using TransactionDT
    This prevents temporal leakage - important for fraud detection
    """
    logger.info("Performing time-based split...")
    
    # Sort by TransactionDT
    df_sorted = df.sort_values('TransactionDT').reset_index(drop=True)
    
    # Calculate split points
    n_total = len(df_sorted)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val:].copy()
    
    logger.info(f"Train set: {len(train_df)} samples ({len(train_df)/n_total:.1%})")
    logger.info(f"Validation set: {len(val_df)} samples ({len(val_df)/n_total:.1%})")
    logger.info(f"Test set: {len(test_df)} samples ({len(test_df)/n_total:.1%})")
    
    # Log fraud rates in each split
    logger.info(f"Train fraud rate: {train_df['isFraud'].mean():.4f}")
    logger.info(f"Validation fraud rate: {val_df['isFraud'].mean():.4f}")
    logger.info(f"Test fraud rate: {test_df['isFraud'].mean():.4f}")
    
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df):
    """Save the split datasets"""
    logger.info("Saving split datasets...")
    
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    
    logger.info(f"Saved splits to {output_dir}")

def analyze_temporal_drift(train_df, val_df, test_df):
    """Analyze temporal distribution across splits"""
    logger.info("Analyzing temporal distribution...")
    
    # Check TransactionDT ranges
    logger.info(f"Train TransactionDT range: {train_df['TransactionDT'].min()} - {train_df['TransactionDT'].max()}")
    logger.info(f"Val TransactionDT range: {val_df['TransactionDT'].min()} - {val_df['TransactionDT'].max()}")
    logger.info(f"Test TransactionDT range: {test_df['TransactionDT'].min()} - {test_df['TransactionDT'].max()}")
    
    # Check amount distributions
    logger.info(f"Train TransactionAmt mean: ${train_df['TransactionAmt'].mean():.2f}")
    logger.info(f"Val TransactionAmt mean: ${val_df['TransactionAmt'].mean():.2f}")
    logger.info(f"Test TransactionAmt mean: ${test_df['TransactionAmt'].mean():.2f}")

def main():
    """Main split pipeline"""
    try:
        # Load cleaned data
        df = load_cleaned_data()
        
        # Perform time-based split
        train_df, val_df, test_df = time_based_split(df)
        
        # Analyze temporal drift
        analyze_temporal_drift(train_df, val_df, test_df)
        
        # Save splits
        save_splits(train_df, val_df, test_df)
        
        logger.info("Data split completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data split: {e}")
        raise

if __name__ == "__main__":
    main()
