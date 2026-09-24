import json
import re
import logging
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClaimVerifier:
    """
    Neuro-symbolic Claim Verifier.
    Parses LLM-generated text for relational claims (e.g., shared devices, emails)
    and mathematically verifies they exist in the GNNExplainer Evidence Object.
    """
    def __init__(self, schema_path: str = "scripts/evidence_schema.json"):
        with open(schema_path, "r") as f:
            self.schema = json.load(f)
            
        # Keywords that indicate the LLM is claiming a specific edge type exists
        self.claim_keywords = {
            "device": ["shared device", "device linkage", "same device", "device identifier"],
            "email": ["shared email", "email domain", "same email"],
            "card": ["shared card", "same card", "card usage"],
            "address": ["shared address", "billing address", "same address"],
            "shared_device": ["cross-card", "different cards same device", "account takeover"]
        }

    def verify_claims(self, llm_text: str, evidence_object: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Verifies that any structural claims in the text are backed by the evidence object.
        Returns (is_valid, list_of_hallucinations).
        """
        llm_text_lower = llm_text.lower()
        hallucinations = []
        
        # Extract the edges the GNN *actually* found and passed to the LLM
        actual_edges = set()
        for edge in evidence_object.get("edge_importances", []):
            actual_edges.add(edge["edge_type"])
            
        # Check if the LLM claimed an edge exists that the GNN didn't actually provide
        for edge_type, keywords in self.claim_keywords.items():
            claimed = any(kw in llm_text_lower for kw in keywords)
            if claimed and edge_type not in actual_edges:
                hallucination_msg = f"Hallucination detected: LLM claimed a '{edge_type}' relationship, but no such edge exists in the GNN Evidence Object."
                hallucinations.append(hallucination_msg)
                
        is_valid = len(hallucinations) == 0
        return is_valid, hallucinations

if __name__ == "__main__":
    # Test the verifier
    verifier = ClaimVerifier("scripts/evidence_schema.json")
    
    fake_evidence = {
        "transaction_id": "123",
        "feature_attributions": [],
        "edge_importances": [
            {"edge_type": "email", "source_node": "A", "target_node": "B", "importance": 0.9}
        ],
        "critical_subgraph": {"n_nodes": 2, "n_edges": 1}
    }
    
    # Text that accurately reflects the evidence
    good_text = "The user exhibits suspicious behavior due to a shared email domain."
    is_valid, errors = verifier.verify_claims(good_text, fake_evidence)
    print(f"Good text valid? {is_valid} Errors: {errors}")
    
    # Text that hallucinates a device edge
    bad_text = "The user exhibits suspicious behavior due to a shared email domain and a shared device."
    is_valid, errors = verifier.verify_claims(bad_text, fake_evidence)
    print(f"Bad text valid? {is_valid} Errors: {errors}")
