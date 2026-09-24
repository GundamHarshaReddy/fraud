"""
Redpanda Producer Script
- Replays transaction data through Redpanda to simulate live streaming
- Produces messages to transactions.raw topic
- Simulates real-time transaction processing
"""

import os
import pandas as pd
import json
import time
from pathlib import Path
import logging
from kafka import KafkaProducer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransactionProducer:
    """Kafka/Redpanda producer for transaction streaming"""
    
    def __init__(self, bootstrap_servers=None, topic='transactions.raw'):
        """Initialize Kafka producer"""
        bootstrap_servers = bootstrap_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda v: str(v).encode('utf-8') if v else None,
            acks='all',
            retries=3
        )
        logger.info(f"Kafka producer initialized for topic: {topic}")
    
    def load_transaction_data(self):
        """Load unseen test transaction data for live streaming"""
        logger.info("Loading transaction data...")
        
        processed_dir = Path(__file__).parent.parent / "data" / "processed"
        test_file = processed_dir / "test.parquet"
        if test_file.exists():
            df = pd.read_parquet(test_file)
            logger.info(f"Loaded {len(df):,} transactions from test.parquet (unseen evaluation stream)")
        else:
            df = pd.read_parquet(processed_dir / "train.parquet")
            logger.info(f"Loaded {len(df):,} transactions from train.parquet")
        
        # Sort by TransactionDT to simulate chronological order
        df = df.sort_values('TransactionDT').reset_index(drop=True)
        return df
    
    def prepare_transaction_message(self, row):
        """Prepare a transaction message for Kafka with all 32 trained features"""
        message = {
            'transaction_id': int(row['TransactionID']),
            'transaction_dt': int(row['TransactionDT']),
            'transaction_amt': float(row['TransactionAmt']),
            # NOTE: is_fraud is intentionally NOT included — sending ground-truth
            # labels on the wire causes data leakage in downstream scoring.
            # Card Profile & Location (8)
            'card1': int(row.get('card1', 0)) if pd.notna(row.get('card1')) else None,
            'card2': int(row.get('card2', 0)) if pd.notna(row.get('card2')) else None,
            'card3': int(row.get('card3', 0)) if pd.notna(row.get('card3')) else None,
            'card4': str(row.get('card4', '')) if pd.notna(row.get('card4')) else None,
            'card5': int(row.get('card5', 0)) if pd.notna(row.get('card5')) else None,
            'card6': str(row.get('card6', '')) if pd.notna(row.get('card6')) else None,
            'addr1': int(row.get('addr1', 0)) if pd.notna(row.get('addr1')) else None,
            'addr2': int(row.get('addr2', 0)) if pd.notna(row.get('addr2')) else None,
            # Core features previously missing
            'ProductCD': str(row.get('ProductCD', '')) if pd.notna(row.get('ProductCD')) else None,
            'dist1': float(row.get('dist1', 0)) if pd.notna(row.get('dist1')) else None,
            # Velocity Counters (7)
            'C1': float(row.get('C1', 0)) if pd.notna(row.get('C1')) else None,
            'C2': float(row.get('C2', 0)) if pd.notna(row.get('C2')) else None,
            'C5': float(row.get('C5', 0)) if pd.notna(row.get('C5')) else None,
            'C6': float(row.get('C6', 0)) if pd.notna(row.get('C6')) else None,
            'C11': float(row.get('C11', 0)) if pd.notna(row.get('C11')) else None,
            'C13': float(row.get('C13', 0)) if pd.notna(row.get('C13')) else None,
            'C14': float(row.get('C14', 0)) if pd.notna(row.get('C14')) else None,
            # Recency Timedeltas (5)
            'D1': float(row.get('D1', 0)) if pd.notna(row.get('D1')) else None,
            'D2': float(row.get('D2', 0)) if pd.notna(row.get('D2')) else None,
            'D4': float(row.get('D4', 0)) if pd.notna(row.get('D4')) else None,
            'D10': float(row.get('D10', 0)) if pd.notna(row.get('D10')) else None,
            'D15': float(row.get('D15', 0)) if pd.notna(row.get('D15')) else None,
            # Behavioral V-Aggregates (4)
            'V127': float(row.get('V127', 0)) if pd.notna(row.get('V127')) else None,
            'V130': float(row.get('V130', 0)) if pd.notna(row.get('V130')) else None,
            'V307': float(row.get('V307', 0)) if pd.notna(row.get('V307')) else None,
            'V310': float(row.get('V310', 0)) if pd.notna(row.get('V310')) else None,
            # Identity (4)
            'device_info': str(row.get('DeviceInfo', '')) if pd.notna(row.get('DeviceInfo')) else None,
            'p_emaildomain': str(row.get('P_emaildomain', '')) if pd.notna(row.get('P_emaildomain')) else None,
            'r_emaildomain': str(row.get('R_emaildomain', '')) if pd.notna(row.get('R_emaildomain')) else None,
            'device_type': str(row.get('DeviceType', '')) if pd.notna(row.get('DeviceType')) else None
        }
        
        return message
    
    def stream_transactions(self, df, speed_factor=1.0, max_transactions=None):
        """Stream transactions through Kafka"""
        logger.info(f"Starting transaction stream with speed factor: {speed_factor}")
        
        total_transactions = len(df)
        if max_transactions:
            total_transactions = min(total_transactions, max_transactions)
            df = df.head(max_transactions)
        
        # Calculate time gaps between transactions for realistic streaming
        time_gaps = df['TransactionDT'].diff().fillna(0)
        
        successful_sends = 0
        failed_sends = 0
        
        for pos, (idx, row) in enumerate(df.iterrows()):
            try:
                # Prepare message
                message = self.prepare_transaction_message(row)
                
                # Send to Kafka
                future = self.producer.send(
                    self.topic,
                    key=message['transaction_id'],
                    value=message
                )
                
                # Wait for send to complete (with timeout)
                record_metadata = future.get(timeout=10)
                
                successful_sends += 1
                
                if successful_sends % 1000 == 0:
                    logger.info(f"Sent {successful_sends}/{total_transactions} transactions")
                
                # Simulate realistic timing based on TransactionDT
                time_gap = time_gaps.iloc[pos] if pos < len(time_gaps) else 0
                if time_gap > 0 and speed_factor > 0:
                    sleep_time = (time_gap / speed_factor) / 1000  # Convert to seconds
                    time.sleep(max(0, min(sleep_time, 1.0)))  # Cap at 1 second
                
            except Exception as e:
                logger.error(f"Failed to send transaction {row['TransactionID']}: {e}")
                failed_sends += 1
        
        logger.info(f"Streaming completed. Success: {successful_sends}, Failed: {failed_sends}")
        return successful_sends, failed_sends
    
    def close(self):
        """Close the producer"""
        logger.info("Closing Kafka producer...")
        self.producer.flush()
        self.producer.close()
        logger.info("Producer closed")

def main():
    """Main producer pipeline"""
    try:
        bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        topic = os.getenv('KAFKA_TOPIC_RAW', 'transactions.raw')
        # Initialize producer
        producer = TransactionProducer(
            bootstrap_servers=bootstrap,
            topic=topic
        )
        
        # Load transaction data
        df = producer.load_transaction_data()
        
        # Stream transactions (adjust speed_factor as needed)
        # speed_factor > 1 means faster than real-time, < 1 means slower
        successful, failed = producer.stream_transactions(
            df, 
            speed_factor=100.0,  # Stream 100x faster than real-time for demo
            max_transactions=10000  # Limit to 10k for initial testing
        )
        
        # Close producer
        producer.close()
        
        logger.info("Redpanda producer completed successfully")
        
    except Exception as e:
        logger.error(f"Error in Redpanda producer: {e}")
        raise

if __name__ == "__main__":
    main()
