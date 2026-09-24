# Setup Guide for Real-Time Fraud Detection System

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (for easy deployment)
- Redis (or use Docker)
- Redpanda/Kafka (or use Docker)
- Google Gemini API key (for GenAI features)

## Quick Start with Docker

1. **Clone and navigate to the project**
```bash
cd ~/Downloads/fraud-detection-system
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

3. **Start infrastructure services**
```bash
docker-compose up redis redpanda -d
```

4. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

5. **Download and prepare dataset**
```bash
# Dataset should already be in data/raw/ from our setup
# If not, download from Kaggle and place train_transaction.csv and train_identity.csv in data/raw/
```

6. **Run data processing pipeline**
```bash
# Step 1: Clean data
python scripts/01_clean.py

# Step 2: Build graph
python scripts/02_build_graph.py

# Step 3: Split data
python scripts/03_split.py
```

7. **Train GNN model**
```bash
python scripts/train_gnn.py
```

8. **Export model for inference**
```bash
python scripts/export_model.py
```

9. **Set up Redis graph store**
```bash
python scripts/setup_redis.py
```

10. **Start Redpanda producer**
```bash
# In one terminal
python scripts/producer.py
```

11. **Start consumer with GNN inference**
```bash
# In another terminal
python scripts/consumer.py
```

12. **Start FastAPI backend**
```bash
# In another terminal
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

13. **Open frontend**
```bash
# Open frontend/index.html in your browser
# Or serve with a simple HTTP server
cd frontend
python -m http.server 3000
```

## Manual Setup (Without Docker)

### Install Redis
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu
sudo apt-get install redis-server
sudo systemctl start redis
```

### Install Redpanda
```bash
# Download and install Redpanda
# Visit https://docs.redpanda.com/docs/get-started/
```

### Create Kafka Topics
```bash
# Using Redpanda's rpk CLI
rpk topic create transactions.raw
rpk topic create transactions.scored
```

## Running the Pipeline

### Phase 1: Data Processing (One-time)
```bash
python scripts/01_clean.py    # Clean and preprocess data
python scripts/02_build_graph.py  # Build transaction graph
python scripts/03_split.py    # Split into train/val/test
```

### Phase 2: Model Training (One-time)
```bash
python scripts/train_gnn.py   # Train GNN model
python scripts/export_model.py  # Export for inference
```

### Phase 3: Infrastructure Setup (One-time)
```bash
python scripts/setup_redis.py  # Populate Redis with graph data
```

### Phase 4: Real-Time Processing (Always Running)
```bash
# Terminal 1: Stream transactions
python scripts/producer.py

# Terminal 2: Process with GNN
python scripts/consumer.py

# Terminal 3: Serve API
cd backend && uvicorn main:app --reload

# Terminal 4: Serve frontend
cd frontend && python -m http.server 3000
```

## API Endpoints

- `GET /health` - Health check
- `GET /transactions` - Get recent transactions
- `GET /transactions/{id}` - Get specific transaction
- `GET /alerts` - Get high-risk alerts
- `POST /alerts/{id}/feedback` - Submit investigator feedback
- `GET /transactions/{id}/gemini-analysis` - Get AI analysis
- `POST /transactions/{id}/request-gemini-analysis` - Request AI analysis
- `GET /dashboard/stats` - Get dashboard statistics
- `WS /ws/live` - WebSocket for real-time updates

## Frontend Features

- **Live Transaction Feed**: Real-time streaming of scored transactions
- **Alert Queue**: High-risk transactions requiring investigation
- **Transaction Details**: Detailed view with graph visualization
- **AI Analysis**: Gemini-generated explanations and SAR drafts
- **Dashboard Stats**: Aggregate metrics and performance indicators

## Configuration

Edit `.env` file to configure:
- `GEMINI_API_KEY`: Required for GenAI features
- `REDIS_HOST/PORT`: Redis connection settings
- `KAFKA_BOOTSTRAP_SERVERS`: Redpanda/Kafka connection
- `HIGH_RISK_THRESHOLD`: Threshold for GenAI escalation (default: 0.8)
- `MEDIUM_RISK_THRESHOLD`: Threshold for flagging (default: 0.5)

## Troubleshooting

### Redis Connection Issues
```bash
# Check if Redis is running
redis-cli ping

# Should return PONG
```

### Redpanda Connection Issues
```bash
# Check Redpanda status
rpk cluster info

# List topics
rpk topic list
```

### Model Loading Issues
```bash
# Check if model files exist
ls -la models/

# Re-export model if needed
python scripts/export_model.py
```

### GPU Memory Issues
If you encounter GPU memory issues during training:
```bash
# Use CPU instead
# The training script automatically detects CUDA availability
# Force CPU by setting CUDA_VISIBLE_DEVICES=""
CUDA_VISIBLE_DEVICES="" python scripts/train_gnn.py
```

## Performance Optimization

### For Production:
1. Use GPU for GNN training
2. Increase Redpanda partitions for better throughput
3. Use Redis clustering for scalability
4. Implement proper error handling and retry logic
5. Add monitoring and alerting

### For Development:
1. Reduce dataset size for faster iteration
2. Use smaller GNN models
3. Limit consumer processing rate
4. Disable Gemini integration if not needed

## Next Steps

1. **Add more sophisticated features**: Implement temporal features, behavioral patterns
2. **Improve graph construction**: Add more entity types and relationship types
3. **Enhance explainability**: Implement GNNExplainer for better model interpretability
4. **Add feedback loop**: Use investigator feedback to retrain models
5. **Implement monitoring**: Add comprehensive logging and metrics
6. **Scale horizontally**: Add load balancing and multiple consumer instances

## Support

For issues or questions:
1. Check logs in the `logs/` directory
2. Review this setup guide
3. Check component-specific documentation
4. Verify all services are running correctly
