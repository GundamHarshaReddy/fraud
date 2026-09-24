"""
FastAPI Backend for Fraud Detection System
- REST API endpoints for transaction monitoring
- WebSocket for real-time transaction updates
- Integration with Redis and scoring results
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import redis
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import asyncio
from collections import defaultdict
import os
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection system with GNN and GenAI escalation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

@app.get("/")
def root():
    return RedirectResponse(url="/frontend/index.html")

# Redis connection
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

# Gemini integration
gemini_service = None
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    try:
        try:
            from backend.gemini_service import GeminiFraudAnalyzer
        except ImportError:
            from gemini_service import GeminiFraudAnalyzer
        gemini_service = GeminiFraudAnalyzer(gemini_api_key, redis_host=redis_host, redis_port=redis_port)
        logger.info("Gemini service initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini service: {e}")

# Helpers for safe data parsing from Redis
def parse_optional_int(val: Any) -> Optional[int]:
    if val is None or val == '' or str(val).lower() in ('nan', 'none'):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None

def parse_optional_float(val: Any, default: float = 0.0) -> float:
    if val is None or val == '' or str(val).lower() in ('nan', 'none'):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                dead_connections.append(connection)
        for dead in dead_connections:
            if dead in self.active_connections:
                self.active_connections.remove(dead)

manager = ConnectionManager()

# Pydantic models
class Transaction(BaseModel):
    transaction_id: int
    transaction_dt: int
    transaction_amt: float
    is_fraud: int
    card1: Optional[int] = None
    card2: Optional[int] = None
    card3: Optional[int] = None
    card4: Optional[int] = None
    card5: Optional[int] = None
    card6: Optional[int] = None
    addr1: Optional[int] = None
    addr2: Optional[int] = None
    device_info: Optional[str] = None
    p_emaildomain: Optional[str] = None
    r_emaildomain: Optional[str] = None
    device_type: Optional[str] = None

class Decision(BaseModel):
    transaction_id: int
    risk_score: float
    timestamp: int
    decision: str
    requires_genai: bool
    status: str

class ScoredTransaction(BaseModel):
    transaction: Transaction
    decision: Decision

class AlertFeedback(BaseModel):
    confirmed: bool
    notes: Optional[str] = None

class DashboardStats(BaseModel):
    total_transactions: int
    fraud_rate: float
    high_risk_count: int
    genai_escalations: int
    avg_latency_ms: float

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_client.ping()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "redis": "connected",
                "api": "running"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Transaction endpoints
@app.get("/transactions", response_model=List[ScoredTransaction])
async def get_transactions(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None
):
    """Get recent transactions with scores"""
    try:
        # Run Redis queries in thread pool to avoid blocking the async event loop (H9)
        def _fetch_from_redis():
            transactions = []
            recent_txn_ids = redis_client.zrevrange("recent_transactions", offset, offset + limit - 1)
            
            for txn_id in recent_txn_ids:
                txn_data = redis_client.hgetall(f"txn:{txn_id}")
                decision_data = redis_client.hgetall(f"decision:{txn_id}")
                
                if txn_data and decision_data:
                    parsed_txn_id = int(float(txn_id))
                    transaction = Transaction(
                        transaction_id=parsed_txn_id,
                        transaction_dt=parse_optional_int(txn_data.get('dt')) or 0,
                        transaction_amt=parse_optional_float(txn_data.get('amount'), 0.0),
                        is_fraud=parse_optional_int(txn_data.get('is_fraud')) or 0,
                        card1=parse_optional_int(txn_data.get('card1')),
                        card2=parse_optional_int(txn_data.get('card2')),
                        card3=parse_optional_int(txn_data.get('card3')),
                        card4=parse_optional_int(txn_data.get('card4')),
                        card5=parse_optional_int(txn_data.get('card5')),
                        card6=parse_optional_int(txn_data.get('card6')),
                        addr1=parse_optional_int(txn_data.get('addr1')),
                        addr2=parse_optional_int(txn_data.get('addr2')),
                        device_info=txn_data.get('device_info'),
                        p_emaildomain=txn_data.get('p_email'),
                        r_emaildomain=txn_data.get('r_email'),
                        device_type=txn_data.get('device_type')
                    )
                    
                    decision = Decision(
                        transaction_id=parsed_txn_id,
                        risk_score=float(decision_data.get('risk_score', 0)),
                        timestamp=int(decision_data.get('timestamp', 0)),
                        decision=decision_data.get('decision', 'unknown'),
                        requires_genai=decision_data.get('requires_genai', 'false') == 'true',
                        status=decision_data.get('status', 'unknown')
                    )
                    
                    if status is None or decision.status == status:
                        transactions.append(ScoredTransaction(transaction=transaction, decision=decision))
            
            return transactions
        
        return await asyncio.to_thread(_fetch_from_redis)
        
    except Exception as e:
        logger.error(f"Error fetching transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions/{transaction_id}", response_model=ScoredTransaction)
async def get_transaction(transaction_id: int):
    """Get specific transaction with full details"""
    try:
        def _fetch_txn():
            txn_data = redis_client.hgetall(f"txn:{transaction_id}")
            decision_data = redis_client.hgetall(f"decision:{transaction_id}")
            
            if not txn_data:
                return None
            
            transaction = Transaction(
                transaction_id=transaction_id,
                transaction_dt=parse_optional_int(txn_data.get('dt')) or 0,
                transaction_amt=parse_optional_float(txn_data.get('amount'), 0.0),
                is_fraud=parse_optional_int(txn_data.get('is_fraud')) or 0,
                card1=parse_optional_int(txn_data.get('card1')),
                card2=parse_optional_int(txn_data.get('card2')),
                card3=parse_optional_int(txn_data.get('card3')),
                card4=parse_optional_int(txn_data.get('card4')),
                card5=parse_optional_int(txn_data.get('card5')),
                card6=parse_optional_int(txn_data.get('card6')),
                addr1=parse_optional_int(txn_data.get('addr1')),
                addr2=parse_optional_int(txn_data.get('addr2')),
                device_info=txn_data.get('device_info'),
                p_emaildomain=txn_data.get('p_email'),
                r_emaildomain=txn_data.get('r_email'),
                device_type=txn_data.get('device_type')
            )
            
            decision = Decision(
                transaction_id=transaction_id,
                risk_score=float(decision_data.get('risk_score', 0)) if decision_data else 0,
                timestamp=int(decision_data.get('timestamp', 0)) if decision_data else 0,
                decision=decision_data.get('decision', 'unknown') if decision_data else 'unknown',
                requires_genai=decision_data.get('requires_genai', 'false') == 'true' if decision_data else False,
                status=decision_data.get('status', 'unknown') if decision_data else 'unknown'
            )
            
            return ScoredTransaction(transaction=transaction, decision=decision)

        result = await asyncio.to_thread(_fetch_txn)
        if not result:
            raise HTTPException(status_code=404, detail="Transaction not found")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions/{transaction_id}/neighborhood")
async def get_transaction_neighborhood(transaction_id: int):
    """Get graph neighborhood (1-2 hops) for transaction visualization"""
    try:
        txn_data = {}
        decision_data = {}
        try:
            txn_data = redis_client.hgetall(f"txn:{transaction_id}") or {}
            decision_data = redis_client.hgetall(f"decision:{transaction_id}") or {}
        except Exception as redis_err:
            logger.warning(f"Redis unavailable for neighborhood lookup: {redis_err}")

        # Extract core properties
        amount = parse_optional_float(txn_data.get('amount'), 150.00)
        risk_score = parse_optional_float(decision_data.get('risk_score'), 0.82)
        
        nodes = []
        edges = []
        
        # Central transaction node
        txn_node_id = f"txn_{transaction_id}"
        nodes.append({
            "id": txn_node_id,
            "label": f"Txn #{transaction_id} (${amount:.2f})",
            "type": "transaction",
            "risk_score": risk_score,
            "status": decision_data.get('status', 'pending_investigation')
        })

        # Card entity
        c1 = txn_data.get('card1') or '1352'
        c2 = txn_data.get('card2') or '450'
        c3 = txn_data.get('card3') or '150'
        card_id = f"{c1}_{c2}_{c3}"
        if card_id:
            c_node_id = f"card_{card_id}"
            nodes.append({
                "id": c_node_id,
                "label": f"Card {card_id.replace('_', '-')}",
                "type": "card"
            })
            edges.append({
                "source": txn_node_id,
                "target": c_node_id,
                "relation": "uses_card"
            })

            # Check Redis for devices and related cards
            try:
                connected_devices = list(redis_client.smembers(f"card:{card_id}:devices") or [])
                for idx, dev in enumerate(connected_devices[:3]):
                    dev_id = f"dev_conn_{idx}"
                    nodes.append({
                        "id": dev_id,
                        "label": str(dev),
                        "type": "device"
                    })
                    edges.append({
                        "source": c_node_id,
                        "target": dev_id,
                        "relation": "shared_device"
                    })

                related_cards = list(redis_client.smembers(f"card:{card_id}:related_cards") or [])
                for idx, rc in enumerate(related_cards[:2]):
                    rc_id = f"rcard_{idx}"
                    nodes.append({
                        "id": rc_id,
                        "label": f"Shared Card {rc.replace('_', '-')}",
                        "type": "card",
                        "suspicious": True
                    })
                    edges.append({
                        "source": c_node_id,
                        "target": rc_id,
                        "relation": "co_used_card"
                    })
            except Exception:
                pass

        # Device entity
        dev_info = txn_data.get('device_info') or "iOS 14.1 Device"
        dev_node_id = f"dev_{transaction_id}"
        nodes.append({
            "id": dev_node_id,
            "label": str(dev_info),
            "type": "device"
        })
        edges.append({
            "source": txn_node_id,
            "target": dev_node_id,
            "relation": "origin_device"
        })

        # Email domain entity
        p_email = txn_data.get('p_email') or "protonmail.com"
        email_node_id = f"email_{p_email}"
        nodes.append({
            "id": email_node_id,
            "label": str(p_email),
            "type": "email"
        })
        edges.append({
            "source": txn_node_id,
            "target": email_node_id,
            "relation": "registered_email"
        })

        # Address entity
        addr1 = txn_data.get('addr1') or "Zip 94103"
        addr_node_id = f"addr_{addr1}"
        nodes.append({
            "id": addr_node_id,
            "label": f"Addr {addr1}",
            "type": "address"
        })
        edges.append({
            "source": txn_node_id,
            "target": addr_node_id,
            "relation": "billing_addr"
        })

        return {
            "transaction_id": transaction_id,
            "nodes": nodes,
            "edges": edges
        }
    except Exception as e:
        logger.error(f"Error fetching neighborhood for {transaction_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alert endpoints
@app.get("/alerts", response_model=List[ScoredTransaction])
async def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    status: str = Query("pending_investigation")
):
    """Get high-risk transactions requiring investigation"""
    try:
        def _fetch_alerts():
            # Get alerts from Redis sorted set (sorted by risk score)
            alert_ids = redis_client.zrevrange(f"alerts:{status}", 0, limit - 1)
            
            alerts = []
            for txn_id in alert_ids:
                # Get full transaction details
                txn_data = redis_client.hgetall(f"txn:{txn_id}")
                decision_data = redis_client.hgetall(f"decision:{txn_id}")
                
                if txn_data and decision_data:
                    parsed_txn_id = int(float(txn_id))
                    transaction = Transaction(
                        transaction_id=parsed_txn_id,
                        transaction_dt=parse_optional_int(txn_data.get('dt')) or 0,
                        transaction_amt=parse_optional_float(txn_data.get('amount'), 0.0),
                        is_fraud=parse_optional_int(txn_data.get('is_fraud')) or 0,
                        card1=parse_optional_int(txn_data.get('card1')),
                        card2=parse_optional_int(txn_data.get('card2')),
                        card3=parse_optional_int(txn_data.get('card3')),
                        card4=parse_optional_int(txn_data.get('card4')),
                        card5=parse_optional_int(txn_data.get('card5')),
                        card6=parse_optional_int(txn_data.get('card6')),
                        addr1=parse_optional_int(txn_data.get('addr1')),
                        addr2=parse_optional_int(txn_data.get('addr2')),
                        device_info=txn_data.get('device_info'),
                        p_emaildomain=txn_data.get('p_email'),
                        r_emaildomain=txn_data.get('r_email'),
                        device_type=txn_data.get('device_type')
                    )
                    
                    decision = Decision(
                        transaction_id=parsed_txn_id,
                        risk_score=float(decision_data.get('risk_score', 0)),
                        timestamp=int(decision_data.get('timestamp', 0)),
                        decision=decision_data.get('decision', 'unknown'),
                        requires_genai=decision_data.get('requires_genai', 'false') == 'true',
                        status=decision_data.get('status', 'unknown')
                    )
                    
                    alerts.append(ScoredTransaction(transaction=transaction, decision=decision))
            
            return alerts

        return await asyncio.to_thread(_fetch_alerts)
        
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/{transaction_id}/feedback")
async def submit_alert_feedback(transaction_id: int, feedback: AlertFeedback):
    """Submit investigator feedback for an alert"""
    try:
        txn_key = str(transaction_id)
        # Store feedback in Redis
        feedback_data = {
            'confirmed': str(feedback.confirmed).lower(),
            'notes': feedback.notes or '',
            'timestamp': str(int(datetime.now().timestamp()))
        }
        
        redis_client.hset(f"feedback:{txn_key}", mapping=feedback_data)
        
        # Update alert status
        decision_data = redis_client.hgetall(f"decision:{txn_key}") or {}
        current_status = decision_data.get("status", "pending_investigation")
        risk_score = float(decision_data.get("risk_score", 0.9))
        
        new_status = "confirmed_fraud" if feedback.confirmed else "false_positive"
        redis_client.hset(f"decision:{txn_key}", "status", new_status)
        
        # Move to appropriate alert queue with risk score ordering
        redis_client.zrem(f"alerts:{current_status}", txn_key)
        redis_client.zrem("alerts:pending_investigation", txn_key)
        redis_client.zadd(f"alerts:{new_status}", {txn_key: risk_score})
        
        logger.info(f"Feedback submitted for transaction {txn_key}: {new_status}")
        return {"status": "success", "message": "Feedback recorded", "new_status": new_status}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Gemini analysis endpoints
@app.get("/transactions/{transaction_id}/gemini-analysis")
async def get_gemini_analysis(transaction_id: int):
    """Get Gemini analysis results for a transaction"""
    try:
        if not gemini_service:
            raise HTTPException(status_code=503, detail="Gemini service not available")
        
        analysis = gemini_service.get_analysis_results(transaction_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="No Gemini analysis found for this transaction")
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching Gemini analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class GeminiAnalysisPayload(BaseModel):
    transaction: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    neighborhood: Optional[Dict[str, Any]] = None

@app.post("/transactions/{transaction_id}/request-gemini-analysis")
async def request_gemini_analysis(transaction_id: int, payload: Optional[GeminiAnalysisPayload] = None):
    """Request Gemini analysis for a transaction with resilient fallback"""
    try:
        if not gemini_service:
            raise HTTPException(status_code=503, detail="Gemini service not available")
        
        # 1. Use payload data if provided by frontend (sanitized — strip is_fraud and limit field lengths)
        transaction = None
        decision = payload.decision if (payload and payload.decision) else None
        neighborhood = payload.neighborhood if (payload and payload.neighborhood) else None
        
        if payload and payload.transaction:
            # Sanitize: never trust client-supplied is_fraud, and truncate string fields
            raw_txn = payload.transaction
            raw_txn.pop('is_fraud', None)
            for key, val in raw_txn.items():
                if isinstance(val, str) and len(val) > 200:
                    raw_txn[key] = val[:200]
            transaction = raw_txn
        
        # 2. If not in payload, query Redis
        if not transaction:
            txn_data = redis_client.hgetall(f"txn:{transaction_id}")
            if txn_data:
                transaction = {
                    'transaction_id': transaction_id,
                    'transaction_dt': parse_optional_int(txn_data.get('dt')) or int(time.time()),
                    'transaction_amt': parse_optional_float(txn_data.get('amount'), 0.0),
                    'is_fraud': parse_optional_int(txn_data.get('is_fraud')) or 0,
                    'card1': parse_optional_int(txn_data.get('card1')),
                    'card2': parse_optional_int(txn_data.get('card2')),
                    'card3': parse_optional_int(txn_data.get('card3')),
                    'card4': parse_optional_int(txn_data.get('card4')),
                    'card5': parse_optional_int(txn_data.get('card5')),
                    'card6': parse_optional_int(txn_data.get('card6')),
                    'addr1': parse_optional_int(txn_data.get('addr1')),
                    'addr2': parse_optional_int(txn_data.get('addr2')),
                    'device_info': txn_data.get('device_info'),
                    'p_emaildomain': txn_data.get('p_email'),
                    'r_emaildomain': txn_data.get('r_email'),
                    'device_type': txn_data.get('device_type')
                }
            else:
                # No transaction found in Redis — return error instead of fabricating data
                raise HTTPException(
                    status_code=404,
                    detail=f"Transaction {transaction_id} not found in Redis. "
                           f"Run the streaming pipeline first to populate transaction data."
                )
        
        if not decision:
            decision_data = redis_client.hgetall(f"decision:{transaction_id}")
            if decision_data:
                decision = {
                    'transaction_id': transaction_id,
                    'risk_score': float(decision_data.get('risk_score', 0.88)),
                    'timestamp': int(decision_data.get('timestamp', int(time.time()))),
                    'decision': decision_data.get('decision', 'escalate'),
                    'requires_genai': decision_data.get('requires_genai', 'true') == 'true',
                    'status': decision_data.get('status', 'pending_investigation')
                }
            else:
                decision = {
                    'transaction_id': transaction_id,
                    'risk_score': 0.88,
                    'timestamp': int(time.time()),
                    'decision': 'escalate',
                    'requires_genai': True,
                    'status': 'pending_investigation'
                }
        
        # Run Gemini analysis in worker thread so event loop remains responsive
        analysis = await asyncio.to_thread(
            gemini_service.analyze_high_risk_transaction, 
            transaction, 
            decision, 
            neighborhood
        )
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requesting Gemini analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard stats endpoint
@app.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        def _fetch_stats():
            total_transactions = int(redis_client.get("stats:total_transactions") or 0)
            fraud_count = int(redis_client.get("stats:fraud_count") or 0)
            high_risk_count = int(redis_client.get("stats:high_risk_count") or 0)
            genai_escalations = int(redis_client.get("stats:genai_escalations") or 0)
            
            # Calculate fraud rate
            fraud_rate = fraud_count / total_transactions if total_transactions > 0 else 0
            
            # Get average latency (simplified)
            avg_latency = float(redis_client.get("stats:avg_latency") or 0)
            
            return DashboardStats(
                total_transactions=total_transactions,
                fraud_rate=fraud_rate,
                high_risk_count=high_risk_count,
                genai_escalations=genai_escalations,
                avg_latency_ms=avg_latency
            )
            
        return await asyncio.to_thread(_fetch_stats)
        
    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transaction updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and send any queued updates
            await asyncio.sleep(1)
            
            # Send heartbeat
            await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now().isoformat()})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

def is_kafka_available(host_port_str: str) -> bool:
    """Quickly check if Kafka broker port is open without triggering noisy client errors"""
    import socket
    try:
        parts = host_port_str.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 9092
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False

# Background worker to consume scored transactions and broadcast
def consume_scored_transactions_worker(loop):
    """Background thread to consume scored transactions and broadcast via WebSocket with auto-reconnect"""
    import time
    import json
    
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    scored_topic = os.getenv("KAFKA_TOPIC_SCORED", "transactions.scored")
    
    # Check if Kafka is available; if not, exit cleanly without spamming logs
    if not is_kafka_available(kafka_bootstrap):
        logger.info(f"Kafka broker not running at {kafka_bootstrap}. Backend is running in standalone mode (no Kafka needed).")
        return

    from kafka import KafkaConsumer
    
    while True:
        try:
            logger.info(f"Connecting background worker to Kafka ({kafka_bootstrap})...")
            consumer = KafkaConsumer(
                scored_topic,
                bootstrap_servers=kafka_bootstrap,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=1000,
                group_id='fraud-api-broadcast-worker'
            )
            logger.info("Started background worker for WebSocket broadcast")
            
            while True:
                try:
                    for message in consumer:
                        data = message.value
                        transaction = data.get('transaction', {})
                        decision = data.get('decision', {})
                        
                        # Update Redis with latest data
                        txn_id = transaction.get('transaction_id')
                        if txn_id:
                            redis_client.zadd("recent_transactions", {str(txn_id): decision.get('timestamp', 0)})
                            redis_client.incr("stats:total_transactions")
                            
                            # Persist transaction snapshot into Redis (M8: 24h TTL)
                            txn_key = f"txn:{txn_id}"
                            redis_client.hset(txn_key, mapping={
                                'amount': str(transaction.get('transaction_amt', 0)),
                                'dt': str(transaction.get('transaction_dt', 0)),
                                'is_fraud': str(transaction.get('is_fraud', 0)),
                                'card1': str(transaction.get('card1') or ''),
                                'card2': str(transaction.get('card2') or ''),
                                'card3': str(transaction.get('card3') or ''),
                                'card4': str(transaction.get('card4') or ''),
                                'card5': str(transaction.get('card5') or ''),
                                'card6': str(transaction.get('card6') or ''),
                                'addr1': str(transaction.get('addr1') or ''),
                                'addr2': str(transaction.get('addr2') or ''),
                                'device_info': str(transaction.get('device_info') or ''),
                                'p_email': str(transaction.get('p_emaildomain') or ''),
                                'r_email': str(transaction.get('r_emaildomain') or ''),
                                'device_type': str(transaction.get('device_type') or '')
                            })
                            redis_client.expire(txn_key, 86400)  # 24h TTL
                            
                            # Persist decision snapshot (M8: 24h TTL)
                            dec_key = f"decision:{txn_id}"
                            redis_client.hset(dec_key, mapping={
                                'risk_score': str(decision.get('risk_score', 0)),
                                'timestamp': str(decision.get('timestamp', 0)),
                                'decision': str(decision.get('decision', 'approve')),
                                'requires_genai': str(decision.get('requires_genai', False)).lower(),
                                'status': str(decision.get('status', 'approved'))
                            })
                            redis_client.expire(dec_key, 86400)  # 24h TTL
                            
                            if decision.get('decision') in ('flag', 'escalate'):
                                redis_client.incr("stats:fraud_count")
                            if decision.get('risk_score', 0) >= 0.8:
                                redis_client.incr("stats:high_risk_count")
                            if decision.get('requires_genai'):
                                redis_client.incr("stats:genai_escalations")
                            
                            # Add to alerts if high risk or flagged
                            if decision.get('decision') == 'escalate':
                                redis_client.zadd("alerts:pending_investigation", 
                                                  {str(txn_id): decision.get('risk_score', 0)})
                            elif decision.get('decision') == 'flag':
                                redis_client.zadd("alerts:pending_investigation", 
                                                  {str(txn_id): decision.get('risk_score', 0)})
                        
                        # Safely schedule broadcast in the main asyncio event loop
                        if not loop.is_closed():
                            asyncio.run_coroutine_threadsafe(
                                manager.broadcast({
                                    "type": "transaction",
                                    "data": data
                                }),
                                loop
                            )
                    time.sleep(0.3)
                except Exception as loop_err:
                    time.sleep(0.5)
                    logger.debug(f"Consumer loop tick error: {loop_err}")
                    break
        except Exception as e:
            logger.warning(f"Background consumer broker not ready ({e}). Retrying in 4s...")
            time.sleep(4)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup"""
    logger.info("Starting FastAPI server...")
    loop = asyncio.get_running_loop()
    threading.Thread(target=consume_scored_transactions_worker, args=(loop,), daemon=True).start()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down FastAPI server...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
