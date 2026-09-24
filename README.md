# Real-Time Financial Fraud Detection System

A comprehensive fraud detection system that uses Graph Neural Networks (GNNs) to analyze transaction relationships in real-time, with intelligent escalation to Generative AI for high-risk cases.

## Architecture Overview

```
Transaction → Redpanda → Redis (Graph Store) → GNN (Risk Score) → Decision Logic
                                                                ↓
                                                        Low Risk: Approve
                                                        High Risk: Gemini (GenAI) → SAR Draft → Frontend
```

## Key Components

1. **Data Pipeline**: IEEE-CIS Fraud Detection dataset processing
2. **Graph Construction**: Building transaction-entity relationships
3. **GNN Model**: PyTorch Geometric GraphSAGE/GAT for fraud detection
4. **Real-Time Processing**: Redpanda streaming + Redis graph store
5. **Intelligent Escalation**: Gemini for explainable AI on high-risk transactions
6. **FastAPI Backend**: REST APIs and WebSocket for real-time updates
7. **Frontend Dashboard**: Live transaction feed, alerts, investigation interface

## Project Structure

```
fraud-detection-system/
├── data/
│   ├── raw/              # Original dataset files
│   ├── processed/        # Cleaned and preprocessed data
│   └── graph/           # Graph nodes and edges
├── models/              # Trained model checkpoints
├── scripts/             # Data processing and training scripts
├── backend/            # FastAPI application
├── frontend/           # React/Vue frontend
└── notebooks/          # Jupyter notebooks for exploration
```

## Setup Instructions

1. Install dependencies: `pip install -r requirements.txt`
2. Download IEEE-CIS dataset from Kaggle to `data/raw/`
3. Run preprocessing: `python scripts/01_clean.py`
4. Build graph: `python scripts/02_build_graph.py`
5. Train model: `python scripts/train_gnn.py`
6. Start services: See individual component READMEs

## Technology Stack

- **Data Processing**: Pandas, NumPy
- **Graph ML**: PyTorch Geometric, NetworkX
- **Streaming**: Redpanda (Kafka-compatible)
- **Cache/Graph Store**: Redis
- **Backend**: FastAPI, Uvicorn
- **GenAI**: Google Gemini API
- **Frontend**: React (planned)

## Key Features

- Real-time transaction scoring with sub-millisecond latency
- Graph-based fraud pattern detection
- Cost-effective GenAI escalation only for high-risk cases
- Explainable AI with human-readable fraud explanations
- Investigator workflow with SAR drafting
