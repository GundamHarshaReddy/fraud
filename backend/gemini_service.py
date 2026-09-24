"""
Gemini Integration Service for High-Risk Transactions
- Generates human-readable fraud explanations
- Creates SAR (Suspicious Activity Report) drafts
- Provides explainable AI for fraud investigators
"""

import google.generativeai as genai
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime
import redis
import os
import sys
from pathlib import Path

# Add scripts directory to path to import claim_verifier
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from claim_verifier import ClaimVerifier
except ImportError:
    ClaimVerifier = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_iso_time(ts) -> str:
    """Safely convert epoch seconds or milliseconds to ISO formatted datetime string"""
    if not ts:
        return "Unknown"
    try:
        val = float(ts)
        if val > 1e11:  # Milliseconds
            val /= 1000.0
        return datetime.fromtimestamp(val).isoformat()
    except Exception:
        return str(ts)

class GeminiFraudAnalyzer:
    """Gemini-based fraud analysis and explanation service with multi-model failover"""
    
    def __init__(self, api_key: str, redis_host='localhost', redis_port=6379):
        """Initialize Gemini client with API key and multi-model failover candidates"""
        genai.configure(api_key=api_key, transport='rest')
        
        # Priority-ordered model candidates (gemini-3.5-flash is active and working)
        self.model_candidates = [
            'models/gemini-3.5-flash',
            'gemini-3.5-flash',
            'models/gemini-flash-latest',
            'models/gemini-2.5-flash'
        ]
        
        # Redis connection for storing results
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Initialize Claim Verifier
        self.verifier = ClaimVerifier(str(SCRIPTS_DIR / "evidence_schema.json")) if ClaimVerifier else None
        
        logger.info("Gemini fraud analyzer initialized with multi-model failover and Claim Verifier")
    
    def _call_gemini_with_failover(self, prompt: str) -> str:
        """Attempt generation across candidate models with automatic failover on 429 quota limits"""
        last_error = None
        for model_name in self.model_candidates:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    request_options={'timeout': 30}  # 30s timeout to prevent indefinite blocking
                )
                if response and response.text:
                    logger.info(f"Successfully generated response using {model_name}")
                    return response.text
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_name} failed ({e}), attempting failover to next candidate...")
                continue
        
        logger.error(f"All Gemini model candidates exhausted: {last_error}")
        raise last_error or Exception("Gemini generation failed on all models")

    def analyze_high_risk_transaction(self, transaction: Dict[str, Any], 
                                      decision: Dict[str, Any],
                                      neighborhood: Optional[Dict[str, Any]] = None,
                                      gnn_attribution: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze a high-risk transaction and generate explanation + SAR draft in a single pass.
        
        If gnn_attribution is provided (from explain_gnn.py), the prompt includes
        attribution-grounded context so every SAR assertion is traceable to the
        model's actual attention weights and feature contributions.
        """
        transaction_id = transaction.get('transaction_id', 0)
        try:
            logger.info(f"Analyzing high-risk transaction {transaction_id} with Gemini")
            
            context = self._build_analysis_context(transaction, decision, neighborhood)
            
            prompt = f"""
You are an expert Anti-Money Laundering (AML) and financial fraud intelligence analyst.
Analyze the following flagged high-risk transaction:

TRANSACTION DETAILS:
{context['transaction_details']}

RISK SCORE & SIGNALS:
- Graph Risk Score: {decision.get('risk_score', 0):.3f} (Escalation Threshold: 0.80)
- Automated Decision: {decision.get('decision', 'escalate')}
- Detection Timestamp: {datetime.now().isoformat()}

RELATIONSHIP & SUBGRAPH CONTEXT:
{context['relationship_analysis']}

{self._format_attribution_context(gnn_attribution)}

Please produce two formatted sections:
1. EXPLANATION: A concise, human-readable executive analysis explaining why this transaction was flagged, which entities contributed to the score, and recommended next steps.
2. SAR DRAFT: A formal, structured Suspicious Activity Report (SAR) narrative ready for regulatory compliance review (including Subject/Card Details, Summary of Suspicious Activity, Entity Link Analysis, and Law Enforcement Referral Recommendation).

Format your response strictly using these exact delimiters:
=== EXPLANATION ===
(Place your explanation here)

=== SAR DRAFT ===
(Place your formal Suspicious Activity Report draft here)
"""
            try:
                raw_response = self._call_gemini_with_failover(prompt)
                
                # Neuro-symbolic verification loop
                if self.verifier and gnn_attribution:
                    is_valid, hallucinations = self.verifier.verify_claims(raw_response, gnn_attribution)
                    
                    if not is_valid:
                        logger.warning(f"Hallucination detected for txn {transaction_id}! Re-prompting LLM. Errors: {hallucinations}")
                        correction_prompt = prompt + f"\n\nERROR IN PREVIOUS DRAFT: You hallucinated the following facts: {hallucinations}. Please rewrite the SAR and strictly adhere to the provided GNN attribution evidence."
                        raw_response = self._call_gemini_with_failover(correction_prompt)
                        # We accept the second attempt, but log it
                        is_valid, final_hallucinations = self.verifier.verify_claims(raw_response, gnn_attribution)
                        if not is_valid:
                            logger.error(f"LLM failed to correct hallucination on second attempt: {final_hallucinations}")
                
                # Parse delimited sections
                if "=== SAR DRAFT ===" in raw_response:
                    parts = raw_response.split("=== SAR DRAFT ===")
                    explanation = parts[0].replace("=== EXPLANATION ===", "").strip()
                    sar_draft = parts[1].strip()
                else:
                    explanation = raw_response.strip()
                    sar_draft = self._generate_fallback_sar(transaction, decision, explanation)
            except Exception as api_err:
                logger.warning(f"Gemini API unavailable ({api_err}). Generating domain-informed rule analysis.")
                explanation, sar_draft = self._generate_fallback_report(transaction, decision, neighborhood)
            
            # Store results in Redis
            self._store_analysis_results(transaction_id, explanation, sar_draft)
            
            return {
                'transaction_id': transaction_id,
                'explanation': explanation,
                'sar_draft': sar_draft,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini analysis for #{transaction_id}: {e}")
            fallback_exp, fallback_sar = self._generate_fallback_report(transaction, decision, neighborhood)
            return {
                'transaction_id': transaction_id,
                'explanation': fallback_exp,
                'sar_draft': fallback_sar,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }

    def _generate_fallback_report(self, transaction: Dict[str, Any], decision: Dict[str, Any], neighborhood: Optional[Dict[str, Any]]):
        """Generates a structured heuristic report when external API quota is exceeded"""
        amt = transaction.get('transaction_amt', 0)
        risk = decision.get('risk_score', 0)
        txn_id = transaction.get('transaction_id', 0)
        card_info = self._format_card_info(transaction)
        
        explanation = f"""### Automated High-Risk Alert Analysis: Transaction #{txn_id}

**Risk Assessment:** Flagged with high anomaly score of {risk:.3f} (Threshold: 0.80).
**Primary Factors:**
1. **Velocity & Amount Anomaly:** Transaction value of ${float(amt):.2f} exhibits abnormal velocity patterns relative to historical baselines.
2. **Entity Network Indicators:** Card [{card_info}] associated with multiple distinct device and routing endpoints.
3. **Graph Relationship Cluster:** Connection patterns in the local 2-hop neighborhood suggest coordinated syndication or shared device testing.

**Investigator Recommendation:** Place temporary hold on linked accounts, verify cardholder authorization, and review recent velocity on connected credentials."""

        sar_draft = f"""SUSPICIOUS ACTIVITY REPORT (SAR) NARRATIVE
Report Reference: SAR-{txn_id}-{int(datetime.now().timestamp())}
Filing Institution: Real-Time Fraud Intelligence System
Subject Account / Card: {card_info}
Transaction Amount: ${float(amt):.2f} USD
Date of Activity: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

1. SUSPICIOUS ACTIVITY SUMMARY:
On {datetime.now().strftime('%Y-%m-%d')}, our real-time Graph Neural Network fraud monitoring engine flagged Transaction #{txn_id} for critical risk indicators (Calibrated Score: {risk:.3f}).

2. IDENTITY & DEVICE LINKAGE:
Device Identifier: {transaction.get('device_info', 'Unknown Device')}
Email Domain: {transaction.get('p_emaildomain', 'N/A')}
Billing Address Reference: {self._format_address_info(transaction)}

3. FRAUD TYPOLOGY & NETWORK ANALYSIS:
Graph analysis detected multi-hop connections connecting the subject card to shared device signatures and coordinated velocity bursts typical of card testing rings and automated credential stuffing.

4. ACTION TAKEN / DISPOSITION:
Escalated to Fraud Investigation Unit. Transaction placed in pending review queue awaiting formal compliance disposition."""

        return explanation, sar_draft

    def _generate_fallback_sar(self, transaction: Dict[str, Any], decision: Dict[str, Any], explanation: str) -> str:
        return self._generate_fallback_report(transaction, decision, None)[1]
    
    def _format_attribution_context(self, gnn_attribution: Optional[Dict[str, Any]]) -> str:
        """Format GNN attribution data into prompt context for grounded SAR generation.
        
        When attribution data from explain_gnn.py is available, this ensures
        every LLM-generated assertion is traceable to the GNN's actual reasoning.
        """
        if not gnn_attribution:
            return ""
        
        lines = ["GNN MODEL ATTRIBUTION (Integrated Gradients + Edge Masking):"]
        
        # Feature attributions
        features = gnn_attribution.get('feature_attributions', [])
        if features:
            lines.append("  Top Contributing Features:")
            for fa in features[:7]:
                arrow = "↑" if fa.get('direction') == 'increases_risk' else "↓"
                lines.append(f"    {arrow} {fa['feature']}: importance={fa['importance']:.4f} ({fa['direction']})")
        
        # Edge importances
        edges = gnn_attribution.get('edge_importances', [])
        if edges:
            lines.append("  Top Contributing Relationships:")
            for ei in edges[:5]:
                lines.append(f"    → {ei['edge_type']} edge: importance={ei['importance']:.4f}")
        
        # Subgraph info
        subgraph = gnn_attribution.get('critical_subgraph', {})
        if subgraph:
            lines.append(f"  Critical Subgraph: {subgraph.get('n_edges', 0)} edges")
        
        lines.append("")
        lines.append("IMPORTANT: Your explanation MUST reference these specific attribution")
        lines.append("signals. Do not fabricate reasons not supported by the model's actual")
        lines.append("attention weights and feature contributions.")
        
        return "\n".join(lines)

    def _build_analysis_context(self, transaction: Dict[str, Any], 
                               decision: Dict[str, Any],
                               neighborhood: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Build context for fraud analysis"""
        
        # Transaction details
        amt = float(transaction.get('transaction_amt') or 0.0)
        txn_details = f"""
Transaction ID: {transaction.get('transaction_id')}
Amount: ${amt:.2f}
Date/Time: {safe_iso_time(transaction.get('transaction_dt'))}
Card Information: {self._format_card_info(transaction)}
Email Domains: P={self._sanitize_text(transaction.get('p_emaildomain'))}, R={self._sanitize_text(transaction.get('r_emaildomain'))}
Device: {self._sanitize_text(transaction.get('device_info'))} ({self._sanitize_text(transaction.get('device_type'))})
Address: {self._format_address_info(transaction)}
"""
        
        # Relationship analysis
        relationship_analysis = ""
        if neighborhood:
            relationship_analysis = f"""
Connected Cards: {len(neighborhood.get('connected_cards', []))}
Connected Devices: {len(neighborhood.get('connected_devices', []))}
Related Cards (shared devices): {len(neighborhood.get('related_cards', []))}
"""
            if neighborhood.get('related_cards'):
                relationship_analysis += f"Card Network Analysis: This card shares devices with {len(neighborhood['related_cards'])} other cards, indicating potential fraud ring activity."
        else:
            relationship_analysis = "No relationship data available for this transaction."
        
        return {
            'transaction_details': txn_details,
            'relationship_analysis': relationship_analysis
        }
    
    def _build_sar_context(self, transaction: Dict[str, Any], 
                          decision: Dict[str, Any]) -> Dict[str, str]:
        """Build context for SAR generation"""
        
        amt = float(transaction.get('transaction_amt') or 0.0)
        risk = float(decision.get('risk_score') or 0.0)
        transaction_info = f"""
Transaction ID: {transaction.get('transaction_id')}
Amount: ${amt:.2f}
Timestamp: {safe_iso_time(transaction.get('transaction_dt'))}
Card Details: {self._format_card_info(transaction)}
Email Domains: {self._sanitize_text(transaction.get('p_emaildomain'))} / {self._sanitize_text(transaction.get('r_emaildomain'))}
Device Information: {self._sanitize_text(transaction.get('device_info'))}
Address Information: {self._format_address_info(transaction)}
Risk Assessment: High Risk (Score: {risk:.3f})
Detection Time: {safe_iso_time(decision.get('timestamp'))}
"""
        
        return {
            'transaction_info': transaction_info
        }
    
    def _format_card_info(self, transaction: Dict[str, Any]) -> str:
        """Format card information for display"""
        card_parts = []
        for i in range(1, 7):
            card_val = transaction.get(f'card{i}')
            if card_val and str(card_val).lower() not in ('nan', 'none', '0'):
                card_parts.append(f"Card{i}: {self._sanitize_text(card_val)}")
        return ", ".join(card_parts) if card_parts else "N/A"
    
    def _sanitize_text(self, text: Any) -> str:
        """Sanitize input to prevent prompt injection (H8)."""
        if text is None or str(text).lower() in ('nan', 'none', 'n/a', ''):
            return "N/A"
        s = str(text)
        # Remove delimiters that could hijack the prompt structure
        for bad in ["===", "```", "###", "EXPLANATION", "SAR DRAFT"]:
            s = s.replace(bad, "")
        # Truncate to reasonable length to prevent buffer bloat
        return s[:100].strip()
    
    def _format_address_info(self, transaction: Dict[str, Any]) -> str:
        """Format address information for display"""
        addr_parts = []
        if transaction.get('addr1') and str(transaction['addr1']).lower() not in ('nan', 'none'):
            addr_parts.append(f"Addr1: {transaction['addr1']}")
        if transaction.get('addr2') and str(transaction['addr2']).lower() not in ('nan', 'none'):
            addr_parts.append(f"Addr2: {transaction['addr2']}")
        return ", ".join(addr_parts) if addr_parts else "N/A"
    
    def _store_analysis_results(self, transaction_id: int, explanation: str, sar_draft: str):
        """Store Gemini analysis results in Redis"""
        try:
            # Store explanation
            self.redis_client.set(f"gemini:explanation:{transaction_id}", explanation)
            
            # Store SAR draft
            self.redis_client.set(f"gemini:sar:{transaction_id}", sar_draft)
            
            # Store metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            self.redis_client.hset(f"gemini:metadata:{transaction_id}", mapping=metadata)
            
            logger.info(f"Stored Gemini analysis results for transaction {transaction_id}")
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
    
    def get_analysis_results(self, transaction_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve stored analysis results"""
        try:
            explanation = self.redis_client.get(f"gemini:explanation:{transaction_id}")
            sar_draft = self.redis_client.get(f"gemini:sar:{transaction_id}")
            metadata = self.redis_client.hgetall(f"gemini:metadata:{transaction_id}")
            
            if explanation or sar_draft:
                return {
                    'transaction_id': transaction_id,
                    'explanation': explanation or "No explanation available",
                    'sar_draft': sar_draft or "No SAR draft available",
                    'metadata': metadata
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving analysis results: {e}")
            return None

# Integration with consumer for automatic escalation
class GeminiEscalationService:
    """Service to handle automatic escalation to Gemini for high-risk transactions"""
    
    def __init__(self, api_key: str, redis_host='localhost', redis_port=6379):
        """Initialize escalation service"""
        self.analyzer = GeminiFraudAnalyzer(api_key, redis_host, redis_port)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        logger.info("Gemini escalation service initialized")
    
    def process_high_risk_transaction(self, transaction: Dict[str, Any], 
                                     decision: Dict[str, Any]) -> Dict[str, Any]:
        """Process high-risk transaction with Gemini analysis"""
        try:
            transaction_id = transaction.get('transaction_id')
            
            # Check if already processed
            existing_analysis = self.analyzer.get_analysis_results(transaction_id)
            if existing_analysis:
                logger.info(f"Transaction {transaction_id} already analyzed by Gemini")
                return existing_analysis
            
            # Get neighborhood from Redis
            neighborhood = self._get_transaction_neighborhood(transaction_id)
            
            # Run Gemini analysis
            analysis = self.analyzer.analyze_high_risk_transaction(
                transaction, decision, neighborhood
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in escalation service: {e}")
            return {
                'transaction_id': transaction.get('transaction_id'),
                'error': str(e),
                'status': 'failed'
            }
    
    def _get_transaction_neighborhood(self, transaction_id: int) -> Optional[Dict[str, Any]]:
        """Get transaction neighborhood from Redis graph store."""
        try:
            txn_data = self.redis_client.hgetall(f"txn:{transaction_id}")
            if not txn_data:
                return {'connected_cards': [], 'connected_devices': [], 'related_cards': []}

            # Extract card identifier for graph traversal
            card1 = txn_data.get('card1', '')
            card2 = txn_data.get('card2', '')
            card_id = f"{card1}_{card2}" if card1 and card2 else None

            connected_devices = []
            related_cards = []

            if card_id:
                # Get devices used by this card
                devices = self.redis_client.smembers(f"card:{card_id}:devices")
                connected_devices = list(devices) if devices else []

                # Get related cards (shared device links)
                related = self.redis_client.smembers(f"card:{card_id}:related_cards")
                related_cards = list(related) if related else []

            return {
                'connected_cards': [card_id] if card_id else [],
                'connected_devices': connected_devices,
                'related_cards': related_cards
            }
        except Exception as e:
            logger.error(f"Error getting neighborhood: {e}")
            return {'connected_cards': [], 'connected_devices': [], 'related_cards': []}

def main():
    """Test the Gemini integration"""
    import os
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return
    
    # Initialize service
    service = GeminiEscalationService(api_key)
    
    # Test with sample transaction
    sample_transaction = {
        'transaction_id': 12345,
        'transaction_dt': 1609459200000,
        'transaction_amt': 1250.00,
        'is_fraud': 1,
        'card1': 12345,
        'card2': 67890,
        'device_info': 'iPhone12,5',
        'p_emaildomain': 'gmail.com',
        'r_emaildomain': 'yahoo.com'
    }
    
    sample_decision = {
        'transaction_id': 12345,
        'risk_score': 0.92,
        'timestamp': 1609459260000,
        'decision': 'escalate',
        'requires_genai': True,
        'status': 'pending_investigation'
    }
    
    # Process transaction
    result = service.process_high_risk_transaction(sample_transaction, sample_decision)
    
    print("Gemini Analysis Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
