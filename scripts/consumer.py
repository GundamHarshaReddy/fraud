"""
Redpanda Consumer Script with GNN Inference
- Consumes transactions from Redpanda
- Fetches local neighborhood from Redis
- Runs GNN inference for risk scoring
- Implements decision logic for GenAI escalation
- Publishes scored transactions
"""

import json
import torch
import numpy as np
import redis
from kafka import KafkaConsumer
from pathlib import Path
import logging
import pickle
import time
import sys
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root, backend, and scripts directories to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
SCRIPTS_DIR = Path(__file__).resolve().parent
for directory in (PROJECT_ROOT, BACKEND_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

# Import backend services
try:
    from backend.gemini_service import GeminiEscalationService
except ImportError:
    try:
        from gemini_service import GeminiEscalationService
    except ImportError as e:
        logging.warning(f"Could not import GeminiEscalationService: {e}")
        GeminiEscalationService = None

try:
    from backend.inference import FraudInferenceEngine
except ImportError:
    try:
        from inference import FraudInferenceEngine
    except ImportError as e:
        logging.warning(f"Could not import FraudInferenceEngine: {e}")
        FraudInferenceEngine = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetectionConsumer:
    """Consumer that processes transactions with GNN inference"""
    
    def __init__(self, 
                 bootstrap_servers=None,
                 input_topic='transactions.raw',
                 output_topic='transactions.scored',
                 redis_host=None,
                 redis_port=None,
                 model_path=None):
        """Initialize consumer with dependencies"""
        
        # Kafka setup
        bootstrap_servers = bootstrap_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        redis_host = redis_host or os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(redis_port or os.getenv('REDIS_PORT', 6379))
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda m: m.decode('utf-8') if m else None,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            consumer_timeout_ms=1000,
            group_id='fraud-detection-group'
        )
        
        self.producer = None  # Will be initialized when needed
        self.is_running = True
        self.last_gemini_escalation = 0.0
        
        # Redis setup
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        # Model setup
        self.model = None
        self.inference_engine = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Decision thresholds — priority: .env > optimal_threshold.json > hardcoded defaults
        env_high = os.getenv('HIGH_RISK_THRESHOLD')
        env_med = os.getenv('MEDIUM_RISK_THRESHOLD')
        self.high_risk_threshold = float(env_high) if env_high else 0.8
        self.medium_risk_threshold = float(env_med) if env_med else 0.5
        # Only load JSON optimized thresholds if env vars were NOT explicitly set
        if not env_high and not env_med:
            try:
                import json
                threshold_config = PROJECT_ROOT / "results" / "optimal_threshold.json"
                if threshold_config.exists():
                    with open(threshold_config) as f:
                        opt = json.load(f)
                    self.high_risk_threshold = opt.get('high_risk_threshold', 0.8)
                    self.medium_risk_threshold = opt.get('medium_risk_threshold', 0.5)
                    logger.info(f"Loaded optimized thresholds: high={self.high_risk_threshold:.2f}, "
                               f"medium={self.medium_risk_threshold:.2f} "
                               f"(recall={opt.get('recall_at_threshold', 'N/A')}, "
                               f"cost_reduction={opt.get('cost_reduction_pct', 'N/A')}%)")
            except Exception as e:
                logger.debug(f"Using default thresholds (config not loaded: {e})")
        
        # Gemini integration
        self.gemini_service = None
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            try:
                if GeminiEscalationService is not None:
                    self.gemini_service = GeminiEscalationService(gemini_api_key)
                    logger.info("Gemini escalation service initialized")
                else:
                    logger.warning("GeminiEscalationService module could not be imported")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini service: {e}")
        
        logger.info("Fraud detection consumer initialized")
    
    def load_model(self):
        """Load the trained model with clean priority chain:
        1. THA-GAT inference wrapper (novel architecture)
        2. Legacy GNN inference wrapper
        3. FraudInferenceEngine (handles TorchScript/state_dict)
        4. Rule-based fallback
        """
        models_dir = PROJECT_ROOT / "models"
        
        # Priority 1: THA-GAT inference wrapper (our novel architecture)
        thagat_path = models_dir / "thagat_inference.pt"
        if thagat_path.exists():
            try:
                # weights_only=False: we save full nn.Module objects (THAGATInference).
                # For safer loading, switch to TorchScript checkpoints.
                self.model = torch.load(thagat_path, map_location=self.device, weights_only=False)
                self.model.eval()
                self.model.to(self.device)
                logger.info(f"Loaded THA-GAT inference model from {thagat_path}")
                return
            except Exception as e:
                logger.warning(f"THA-GAT load failed: {e}")

        # Priority 2: Legacy GNN inference wrapper
        legacy_path = models_dir / "gnn_fraud_inference.pt"
        if legacy_path.exists():
            try:
                self.model = torch.load(legacy_path, map_location=self.device, weights_only=False)
                self.model.eval()
                self.model.to(self.device)
                logger.info(f"Loaded legacy GNN inference model from {legacy_path}")
                return
            except Exception as e:
                logger.warning(f"Legacy GNN load failed: {e}")

        # Priority 3: FraudInferenceEngine (handles TorchScript and state_dict)
        if FraudInferenceEngine is not None:
            try:
                self.inference_engine = FraudInferenceEngine(model_path=self.model_path)
                logger.info(f"Inference engine initialized (type: {self.inference_engine.model_type})")
                return
            except Exception as e:
                logger.warning(f"Inference engine init failed: {e}")

        # Priority 4: Rule-based fallback
        logger.warning("No model loaded — using rule-based fallback scoring")
    
    def get_transaction_neighborhood(self, transaction_id: str) -> Dict[str, Any]:
        """Fetch local neighborhood from Redis"""
        try:
            # Get transaction data
            txn_data = self.redis_client.hgetall(f"txn:{transaction_id}")
            
            if not txn_data:
                logger.warning(f"Transaction {transaction_id} not found in Redis")
                return self._create_empty_neighborhood(transaction_id)
            
            neighborhood = {
                'transaction_id': transaction_id,
                'transaction_data': txn_data,
                'connected_cards': [],
                'connected_devices': [],
                'related_cards': []
            }
            
            # Get card information
            parts = []
            for k in ('card1', 'card2', 'card3'):
                v = txn_data.get(k)
                if v and str(v).strip() and str(v).lower() not in ('nan', 'none', '0', '0.0', ''):
                    try:
                        parts.append(str(int(float(v))))
                    except Exception:
                        parts.append(str(v).strip())
            card_id = "_".join(parts) if parts else ""
            if card_id:
                neighborhood['connected_cards'].append(card_id)
                
                # Get devices for this card
                devices = self.redis_client.smembers(f"card:{card_id}:devices")
                neighborhood['connected_devices'].extend(list(devices))
                
                # Get related cards
                related_cards = self.redis_client.smembers(f"card:{card_id}:related_cards")
                neighborhood['related_cards'].extend(list(related_cards))
            
            return neighborhood
            
        except Exception as e:
            logger.error(f"Error fetching neighborhood: {e}")
            return self._create_empty_neighborhood(transaction_id)
    
    def _create_empty_neighborhood(self, transaction_id: str) -> Dict[str, Any]:
        """Create empty neighborhood when Redis lookup fails"""
        return {
            'transaction_id': transaction_id,
            'transaction_data': {},
            'connected_cards': [],
            'connected_devices': [],
            'related_cards': []
        }
    
    def prepare_model_input(self, transaction: Dict[str, Any], neighborhood: Dict[str, Any]):
        """Prepare full 32-feature input tensor with heterogeneous edges for THA-GAT.
        
        Extracts the same 32 features used in training (train_thagat.py):
        [Core: Amt, DT, ProductCD, dist1 | Cards: card1..card6, addr1, addr2 |
         Velocity: C1..C14 | Recency: D1..D15 | Behavioral: V127..V310 |
         Identity: P_email, R_email, DeviceType, DeviceInfo]
        
        Also constructs edge_type and edge_time_delta tensors for THA-GAT.
        """
        try:
            # 1. Core Transaction (4)
            amt = float(transaction.get('transaction_amt') or transaction.get('amount') or 0.0)
            amt_log = float(np.log1p(max(amt, 0.0)))
            dt = float(transaction.get('transaction_dt') or transaction.get('dt') or 0.0)
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

            # 5. Critical Behavioral V-Aggregates (4) — with log1p matching training
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

            # Assemble 32-feature vector
            feat_vector = [
                amt_log, dt, product_cd, dist1,
                card1, card2, card3, card4, card5, card6, addr1, addr2,
                c1, c2, c5, c6, c11, c13, c14,
                d1, d2, d4, d10, d15,
                v127, v130, v307, v310,
                p_email_val, r_email_val, dev_type_val, dev_info_val
            ]
            assert len(feat_vector) == 32, f"Expected 32 features, got {len(feat_vector)}"
            x_list = [feat_vector]

            # Add neighborhood edges if available from Redis
            connected_cards = neighborhood.get('connected_cards', [])
            connected_devices = neighborhood.get('connected_devices', [])
            related_cards = neighborhood.get('related_cards', [])

            # Construct additional nodes and edges based on neighborhood (H1)
            # Node 0 is the current transaction. We append dummy features for neighbors.
            n_card_edges = min(len(connected_cards), 5)
            n_device_edges = min(len(connected_devices), 3)
            n_shared_edges = min(len(related_cards), 4)

            extra_src = []
            extra_dst = []
            extra_types = []
            extra_deltas = []
            node_idx = 1
            
            for _ in range(n_card_edges):
                x_list.append([0.0] * 32)
                extra_src.extend([0, node_idx]); extra_dst.extend([node_idx, 0])
                extra_types.extend([0, 0]); extra_deltas.extend([0.0, 0.0])  # EDGE_TYPE_CARD
                node_idx += 1
                
            for _ in range(n_device_edges):
                x_list.append([0.0] * 32)
                extra_src.extend([0, node_idx]); extra_dst.extend([node_idx, 0])
                extra_types.extend([1, 1]); extra_deltas.extend([0.0, 0.0])  # EDGE_TYPE_DEVICE
                node_idx += 1
                
            for _ in range(n_shared_edges):
                x_list.append([0.0] * 32)
                extra_src.extend([0, node_idx]); extra_dst.extend([node_idx, 0])
                extra_types.extend([4, 4]); extra_deltas.extend([0.0, 0.0])  # EDGE_TYPE_SHARED_DEVICE
                node_idx += 1

            x = torch.tensor(x_list, dtype=torch.float32).to(self.device)

            # Self-loop edge (EDGE_TYPE_SELF_LOOP = 5)
            edge_index = torch.zeros((2, 1), dtype=torch.long).to(self.device)
            edge_type = torch.tensor([5], dtype=torch.long).to(self.device)
            edge_time_delta = torch.zeros(1, dtype=torch.float32).to(self.device)

            if len(extra_src) > 0:
                extra_ei = torch.tensor([extra_src, extra_dst], dtype=torch.long).to(self.device)
                extra_et = torch.tensor(extra_types, dtype=torch.long).to(self.device)
                extra_td = torch.tensor(extra_deltas, dtype=torch.float32).to(self.device)

                edge_index = torch.cat([edge_index, extra_ei], dim=1)
                edge_type = torch.cat([edge_type, extra_et])
                edge_time_delta = torch.cat([edge_time_delta, extra_td])

            return x, edge_index, edge_type, edge_time_delta

        except Exception as e:
            logger.error(f"Error preparing model input: {e}")
            x = torch.zeros(1, 32, dtype=torch.float32).to(self.device)
            edge_index = torch.zeros((2, 1), dtype=torch.long).to(self.device)
            edge_type = torch.tensor([5], dtype=torch.long).to(self.device)
            edge_time_delta = torch.zeros(1, dtype=torch.float32).to(self.device)
            return x, edge_index, edge_type, edge_time_delta
    
    def score_transaction(self, transaction: Dict[str, Any], neighborhood: Dict[str, Any]) -> float:
        """Score transaction using THA-GAT model, inference engine, or rule-based fallback."""
        # Route 1: Direct model (THA-GAT or legacy wrapper)
        if self.model is not None:
            try:
                x, edge_index, edge_type, edge_time_delta = self.prepare_model_input(transaction, neighborhood)
                with torch.no_grad():
                    # THA-GAT and THAGATInference support edge_type and edge_time_delta
                    try:
                        output = self.model(x, edge_index, edge_type=edge_type, edge_time_delta=edge_time_delta)
                    except TypeError:
                        # Legacy model doesn't accept edge_type/edge_time_delta
                        output = self.model(x, edge_index)
                    
                    if isinstance(output, torch.Tensor):
                        val = output.squeeze().item()
                        if np.isnan(val) or np.isinf(val):
                            return self._rule_based_scoring(transaction, neighborhood)
                        # Apply sigmoid if raw logits
                        if val < 0.0 or val > 1.0:
                            score = torch.sigmoid(output).squeeze().item()
                        else:
                            score = val
                        return float(np.clip(score, 0.01, 0.99))
            except Exception as e:
                logger.debug(f"Model scoring fallback: {e}")

        # Route 2: Inference engine
        if self.inference_engine is not None:
            try:
                return self.inference_engine.score(transaction, neighborhood)
            except Exception as e:
                logger.debug(f"Inference engine scoring fallback: {e}")

        # Route 3: Rule-based fallback
        return self._rule_based_scoring(transaction, neighborhood)
    
    def _rule_based_scoring(self, transaction: Dict[str, Any], neighborhood: Dict[str, Any]) -> float:
        """Fallback rule-based scoring when model is unavailable"""
        risk_score = 0.08  # Base risk
        amount = float(transaction.get('transaction_amt', 0))
        if amount > 1000:
            risk_score += 0.35
        elif amount > 500:
            risk_score += 0.20
        elif amount > 250:
            risk_score += 0.10
        
        if len(neighborhood.get('related_cards', [])) > 5:
            risk_score += 0.25
        elif len(neighborhood.get('related_cards', [])) > 2:
            risk_score += 0.15
        
        return min(risk_score, 0.99)
    
    def make_decision(self, risk_score: float, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision based on risk score"""
        decision = {
            'transaction_id': transaction['transaction_id'],
            'risk_score': risk_score,
            'timestamp': int(time.time()),
            'decision': None,
            'requires_genai': False,
            'status': None
        }
        
        if risk_score >= self.high_risk_threshold:
            decision['decision'] = 'escalate'
            decision['requires_genai'] = True
            decision['status'] = 'pending_investigation'
        elif risk_score >= self.medium_risk_threshold:
            decision['decision'] = 'flag'
            decision['status'] = 'needs_review'
        else:
            decision['decision'] = 'approve'
            decision['status'] = 'approved'
        
        return decision
    
    def publish_scored_transaction(self, transaction: Dict[str, Any], decision: Dict[str, Any]):
        """Publish scored transaction to output topic"""
        try:
            if self.producer is None:
                from kafka import KafkaProducer
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda v: str(v).encode('utf-8')
                )
            
            # Combine transaction and decision
            scored_message = {
                'transaction': transaction,
                'decision': decision
            }
            
            self.producer.send(
                self.output_topic,
                key=transaction['transaction_id'],
                value=scored_message
            )
            # Ensure message is flushed to Kafka to prevent loss on crash
            self.producer.flush()
            
        except Exception as e:
            logger.error(f"Error publishing scored transaction: {e}")
    
    def process_message(self, message):
        """Process a single transaction message"""
        try:
            transaction = message.value
            transaction_id = transaction['transaction_id']
            
            # Fetch neighborhood from Redis
            neighborhood = self.get_transaction_neighborhood(str(transaction_id))
            
            # Score transaction
            risk_score = self.score_transaction(transaction, neighborhood)
            
            # Make decision
            decision = self.make_decision(risk_score, transaction)
            
            logger.info(f"Transaction {transaction_id}: Risk={risk_score:.3f}, Decision={decision['decision']}")
            
            # Publish scored transaction
            self.publish_scored_transaction(transaction, decision)
            
            # Escalate to Gemini if high risk (throttled to 1 per 15s to keep consumer streaming smoothly)
            now = time.time()
            if decision['requires_genai'] and self.gemini_service and (now - self.last_gemini_escalation > 15.0):
                try:
                    self.last_gemini_escalation = now
                    logger.info(f"Escalating high-risk transaction {transaction_id} to Gemini")
                    self.gemini_service.process_high_risk_transaction(transaction, decision)
                except Exception as e:
                    logger.warning(f"Gemini escalation skipped/failed: {e}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    def run(self, max_messages=None):
        """Main consumer loop"""
        logger.info("Starting fraud detection consumer...")
        
        # Load model
        self.load_model()
        
        processed_count = 0
        
        try:
            while self.is_running:
                batch_count = 0
                for message in self.consumer:
                    if not self.is_running:
                        break
                    self.process_message(message)
                    processed_count += 1
                    batch_count += 1
                    
                    if max_messages and processed_count >= max_messages:
                        logger.info(f"Processed {processed_count} messages, stopping")
                        self.is_running = False
                        break
                    
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} transactions")
                
                if batch_count == 0:
                    time.sleep(0.1)
                        
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}")
        finally:
            self.close()
    
    def close(self):
        """Clean up resources"""
        logger.info("Closing consumer...")
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        logger.info("Consumer closed")

def main():
    """Main consumer pipeline"""
    try:
        consumer = FraudDetectionConsumer(
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
            input_topic=os.getenv('KAFKA_TOPIC_RAW', 'transactions.raw'),
            output_topic=os.getenv('KAFKA_TOPIC_SCORED', 'transactions.scored'),
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', 6379)),
            model_path=os.getenv('MODEL_PATH', None)
        )
        
        # Run consumer continuously
        consumer.run()
        
        logger.info("Redpanda consumer completed successfully")
        
    except Exception as e:
        logger.error(f"Error in Redpanda consumer: {e}")
        raise

if __name__ == "__main__":
    main()
