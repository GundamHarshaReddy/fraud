// Fraud Detection Dashboard JavaScript
// Real-time GNN Graph Visualization & Investigator Workspace

// API Configuration
const isDirectPort = window.location.port === '8000';
const API_BASE_URL = isDirectPort ? 'http://localhost:8000' : (window.location.origin.includes('localhost') ? 'http://localhost:8000' : window.location.origin);
const WS_URL = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + 
               (window.location.hostname || 'localhost') + 
               ':8000/ws/live';

// State Management
let websocket = null;
let isPaused = false;
let transactions = [];
let alerts = [];
let isSimulating = false;
let simTimer = null;
let simTxnCounter = 300100;

// Mock pools for streaming simulation
const MOCK_CARDS = ["1004-550-150", "4412-320-102", "8821-190-220", "9912-880-180", "2250-410-110", "6701-900-330"];
const MOCK_DEVICES = ["iOS Device / Safari 16.1", "Windows 11 / Chrome 122", "MacOS / Chrome 121", "Android 14 / Samsung SM-S918B", "Tor Browser / Linux"];
const MOCK_EMAILS = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "temp-mail.org", "proton.me"];

// DOM Elements
const connectionStatus = document.getElementById('connectionStatus');
const connectionText = document.getElementById('connectionText');
const transactionFeed = document.getElementById('transactionFeed');
const alertQueue = document.getElementById('alertQueue');
const pauseFeedBtn = document.getElementById('pauseFeed');
const clearFeedBtn = document.getElementById('clearFeed');
const refreshAlertsBtn = document.getElementById('refreshAlerts');
const alertFilter = document.getElementById('alertFilter');
const btnDemoFeed = document.getElementById('btnDemoFeed');
const btnInjectBatch = document.getElementById('btnInjectBatch');

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeWebSocket();
    loadDashboardStats();
    loadRecentTransactions();
    loadAlerts();
    setupEventListeners();
});

function generateMockTransaction(forceAlert = null) {
    simTxnCounter++;
    const isAlert = forceAlert !== null ? forceAlert : Math.random() < 0.22;
    const card = isAlert ? "9912-880-180" : MOCK_CARDS[Math.floor(Math.random() * MOCK_CARDS.length)];
    const cardParts = card.split('-');
    const dev = isAlert ? "Tor Browser / Linux" : MOCK_DEVICES[Math.floor(Math.random() * MOCK_DEVICES.length)];
    const email = isAlert ? "temp-mail.org" : MOCK_EMAILS[Math.floor(Math.random() * MOCK_EMAILS.length)];
    const amt = isAlert ? (Math.random() * 2200 + 450).toFixed(2) : (Math.random() * 180 + 12).toFixed(2);
    const risk = isAlert ? (Math.random() * 0.16 + 0.82).toFixed(4) : (Math.random() * 0.35 + 0.04).toFixed(4);

    return {
        transaction: {
            transaction_id: simTxnCounter,
            transaction_dt: Date.now(),
            transaction_amt: parseFloat(amt),
            is_fraud: isAlert ? 1 : 0,
            card1: parseInt(cardParts[0]),
            card2: parseInt(cardParts[1]),
            card3: parseInt(cardParts[2]),
            device_info: dev,
            p_emaildomain: email
        },
        decision: {
            transaction_id: simTxnCounter,
            risk_score: parseFloat(risk),
            timestamp: Date.now(),
            decision: isAlert ? 'escalate' : 'approve',
            requires_genai: isAlert,
            status: isAlert ? 'pending_investigation' : 'approved'
        }
    };
}

function seedInitialTransactions() {
    for (let i = 0; i < 8; i++) {
        const mock = generateMockTransaction(i === 2 || i === 6);
        transactions.push(mock);
        if (mock.decision.risk_score >= 0.8) {
            alerts.push(mock);
        }
    }
    renderTransactionFeed();
    if (alerts.length > 0) {
        renderAlerts(alerts);
    }
    document.getElementById('totalTransactions').textContent = transactions.length.toLocaleString();
    document.getElementById('highRiskCount').textContent = alerts.length.toLocaleString();
    document.getElementById('avgLatency').textContent = '8.2ms';
}

async function loadRecentTransactions() {
    try {
        const response = await fetch(`${API_BASE_URL}/transactions?limit=30`);
        if (response.ok) {
            const list = await response.json();
            if (list && list.length > 0) {
                transactions = list;
                renderTransactionFeed();
                return;
            }
        }
    } catch (err) {
        console.warn('Failed to preload transactions:', err);
    }
    // Seed initial demo data so the table is never blank
    seedInitialTransactions();
}

// WebSocket Connection
function initializeWebSocket() {
    try {
        websocket = new WebSocket(WS_URL);
        
        websocket.onopen = () => {
            updateConnectionStatus(true);
            console.log('WebSocket connected');
        };
        
        websocket.onmessage = (event) => {
            if (!isPaused) {
                try {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                } catch (err) {
                    console.error('Error parsing WebSocket message:', err);
                }
            }
        };
        
        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateConnectionStatus(false);
        };
        
        websocket.onclose = () => {
            updateConnectionStatus(false);
            console.log('WebSocket disconnected');
            setTimeout(initializeWebSocket, 5000);
        };
        
    } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        updateConnectionStatus(false);
    }
}

function updateConnectionStatus(connected) {
    if (connected) {
        connectionStatus.classList.add('connected');
        connectionStatus.classList.remove('disconnected');
        connectionText.textContent = 'Live Feed Connected';
    } else {
        connectionStatus.classList.remove('connected');
        connectionStatus.classList.add('disconnected');
        connectionText.textContent = 'Disconnected (Reconnecting...)';
    }
}

// Handle WebSocket Messages with Throttled Stats Updates
let statsUpdateTimer = null;
function scheduleStatsUpdate() {
    if (statsUpdateTimer) return;
    statsUpdateTimer = setTimeout(() => {
        statsUpdateTimer = null;
        loadDashboardStats();
    }, 2000);
}

function handleWebSocketMessage(data) {
    if (data.type === 'transaction') {
        addTransactionToFeed(data.data);
        scheduleStatsUpdate();
    } else if (data.type === 'heartbeat') {
        // Keep alive
    }
}

// API Calls
async function loadDashboardStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
        if (response.ok) {
            const stats = await response.json();
            updateStatsDisplay(stats);
        }
    } catch (error) {
        console.error('Failed to load dashboard stats:', error);
    }
}

async function loadAlerts() {
    try {
        const status = alertFilter.value;
        const response = await fetch(`${API_BASE_URL}/alerts?status=${status}&limit=50`);
        const alertsData = await response.json();
        alerts = alertsData;
        renderAlerts(alertsData);
    } catch (error) {
        console.error('Failed to load alerts:', error);
    }
}

async function loadTransactionDetails(transactionId) {
    try {
        const response = await fetch(`${API_BASE_URL}/transactions/${transactionId}`);
        return await response.json();
    } catch (error) {
        console.error('Failed to load transaction details:', error);
        return null;
    }
}

async function loadTransactionNeighborhood(transactionId) {
    try {
        const response = await fetch(`${API_BASE_URL}/transactions/${transactionId}/neighborhood`);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.warn('Could not load neighborhood API, will fallback to transaction nodes:', error);
    }
    return null;
}

async function loadGeminiAnalysis(transactionId) {
    try {
        const response = await fetch(`${API_BASE_URL}/transactions/${transactionId}/gemini-analysis`);
        if (response.ok) {
            return await response.json();
        }
        return null;
    } catch (error) {
        console.error('Failed to load Gemini analysis:', error);
        return null;
    }
}

async function requestGeminiAnalysis(transactionId) {
    try {
        const response = await fetch(`${API_BASE_URL}/transactions/${transactionId}/request-gemini-analysis`, {
            method: 'POST'
        });
        return await response.json();
    } catch (error) {
        console.error('Failed to request Gemini analysis:', error);
        return null;
    }
}

async function submitAlertFeedback(transactionId, confirmed, notes) {
    try {
        const response = await fetch(`${API_BASE_URL}/alerts/${transactionId}/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ confirmed, notes })
        });
        return await response.json();
    } catch (error) {
        console.error('Failed to submit feedback:', error);
        return null;
    }
}

// UI Updates
function updateStatsDisplay(stats) {
    if (!stats) return;
    document.getElementById('totalTransactions').textContent = (stats.total_transactions || 0).toLocaleString();
    document.getElementById('fraudRate').textContent = ((stats.fraud_rate || 0) * 100).toFixed(2) + '%';
    document.getElementById('highRiskCount').textContent = (stats.high_risk_count || 0).toLocaleString();
    document.getElementById('genaiEscalations').textContent = (stats.genai_escalations || 0).toLocaleString();
    document.getElementById('avgLatency').textContent = (stats.avg_latency_ms || 0).toFixed(1) + 'ms';
}

function addTransactionToFeed(data) {
    const transaction = data.transaction;
    const decision = data.decision;
    if (!transaction || !decision) return;
    
    transactions.unshift({ transaction, decision });
    if (transactions.length > 60) {
        transactions.pop();
    }
    renderTransactionFeed();

    // If alert/high-risk, add to alerts queue
    if (decision.risk_score >= 0.8 || decision.decision === 'escalate' || decision.status === 'pending_investigation') {
        const existingIdx = alerts.findIndex(a => a.transaction.transaction_id === transaction.transaction_id);
        if (existingIdx === -1) {
            alerts.unshift({ transaction, decision });
            if (alerts.length > 40) alerts.pop();
            renderAlerts(alerts);
        }
    }

    // Increment live counter cards
    incrementLocalStats(decision);
}

function incrementLocalStats(decision) {
    const totalEl = document.getElementById('totalTransactions');
    const highRiskEl = document.getElementById('highRiskCount');
    const genaiEl = document.getElementById('genaiEscalations');
    const latencyEl = document.getElementById('avgLatency');

    if (totalEl) {
        let total = parseInt(totalEl.textContent.replace(/,/g, '')) || 0;
        totalEl.textContent = (total + 1).toLocaleString();
    }
    if (decision && decision.risk_score >= 0.8 && highRiskEl) {
        let hr = parseInt(highRiskEl.textContent.replace(/,/g, '')) || 0;
        highRiskEl.textContent = (hr + 1).toLocaleString();
    }
    if (decision && decision.requires_genai && genaiEl) {
        let ge = parseInt(genaiEl.textContent.replace(/,/g, '')) || 0;
        genaiEl.textContent = (ge + 1).toLocaleString();
    }
    if (latencyEl) {
        latencyEl.textContent = (Math.random() * 4 + 6).toFixed(1) + 'ms';
    }
}

function renderTransactionFeed() {
    transactionFeed.innerHTML = '';
    
    transactions.forEach(({ transaction, decision }) => {
        const isGeminiTagged = decision && (decision.requires_genai || decision.risk_score >= 0.75 || decision.decision === 'escalate');
        const geminiBadge = isGeminiTagged
            ? `<span style="background:rgba(168,85,247,0.25); border:1px solid #c084fc; color:#e9d5ff; border-radius:4px; padding:2px 6px; font-size:10px; font-weight:700; margin-left:6px; display:inline-block;">🤖 GEMINI AI</span>`
            : '';

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>#${transaction.transaction_id}</td>
            <td>$${(transaction.transaction_amt || 0).toFixed(2)}</td>
            <td>${formatCardInfo(transaction)}</td>
            <td>${transaction.device_info || 'N/A'}</td>
            <td class="${getRiskClass(decision.risk_score)}">${(decision.risk_score || 0).toFixed(3)}</td>
            <td><span class="status-badge ${getStatusClass(decision.status)}">${formatStatus(decision.status || '')}</span>${geminiBadge}</td>
            <td>${formatTime(decision.timestamp || transaction.transaction_dt)}</td>
        `;
        
        row.addEventListener('click', () => showTransactionDetails(transaction.transaction_id));
        row.style.cursor = 'pointer';
        transactionFeed.appendChild(row);
    });
}

function renderAlerts(alertsData) {
    alertQueue.innerHTML = '';
    
    if (!alertsData || alertsData.length === 0) {
        alertQueue.innerHTML = '<tr><td colspan="5" class="loading">No alerts in this queue</td></tr>';
        return;
    }
    
    alertsData.forEach(({ transaction, decision }) => {
        const isGeminiTagged = decision && (decision.requires_genai || decision.risk_score >= 0.75 || decision.decision === 'escalate');
        const geminiBadge = isGeminiTagged
            ? `<span style="background:rgba(168,85,247,0.25); border:1px solid #c084fc; color:#e9d5ff; border-radius:4px; padding:2px 6px; font-size:10px; font-weight:700; margin-left:6px; display:inline-block;">🤖 GEMINI ESCALATED</span>`
            : '';

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>#${transaction.transaction_id}</td>
            <td>$${(transaction.transaction_amt || 0).toFixed(2)}</td>
            <td class="${getRiskClass(decision.risk_score)}">${(decision.risk_score || 0).toFixed(3)}</td>
            <td><span class="status-badge ${getStatusClass(decision.status)}">${formatStatus(decision.status || '')}</span>${geminiBadge}</td>
            <td>
                <button class="btn btn-primary" onclick="showTransactionDetails(${transaction.transaction_id})">Investigate</button>
            </td>
        `;
        alertQueue.appendChild(row);
    });
}

// Modal & Investigation Functions
async function showTransactionDetails(transactionId) {
    const modal = document.getElementById('transactionModal');
    const modalBody = document.getElementById('modalBody');
    
    modalBody.innerHTML = '<div class="loading"><div class="loading-spinner"></div><p>Loading transaction graph & details...</p></div>';
    modal.classList.add('active');
    
    const [txnData, neighborhoodData] = await Promise.all([
        loadTransactionDetails(transactionId),
        loadTransactionNeighborhood(transactionId)
    ]);
    
    if (txnData) {
        renderTransactionDetails(txnData, transactionId, neighborhoodData);
    } else {
        modalBody.innerHTML = '<p class="loading">Failed to load transaction details.</p>';
    }
}

function closeModal() {
    const modal = document.getElementById('transactionModal');
    modal.classList.remove('active');
}

function renderTransactionDetails(data, transactionId, neighborhoodData) {
    const modalBody = document.getElementById('modalBody');
    const { transaction, decision } = data;
    
    modalBody.innerHTML = `
        <div class="detail-section">
            <h3>💳 Core Transaction Attributes</h3>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-label">Transaction ID</div>
                    <div class="detail-value">#${transaction.transaction_id}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Amount</div>
                    <div class="detail-value">$${(transaction.transaction_amt || 0).toFixed(2)}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Timestamp / Date</div>
                    <div class="detail-value">${formatTime(transaction.transaction_dt)}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">GNN Risk Score</div>
                    <div class="detail-value ${getRiskClass(decision.risk_score)}">${(decision.risk_score || 0).toFixed(3)}</div>
                </div>
            </div>
        </div>
        
        <div class="detail-section">
            <h3>🔗 Entity Identifiers</h3>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-label">Card Multi-Field</div>
                    <div class="detail-value">${formatCardInfo(transaction)}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Device Info</div>
                    <div class="detail-value">${transaction.device_info || 'Unknown Device'}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Purchaser / Recipient Email</div>
                    <div class="detail-value">${transaction.p_emaildomain || 'N/A'} / ${transaction.r_emaildomain || 'N/A'}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Billing / Region Code</div>
                    <div class="detail-value">${transaction.addr1 || 'N/A'} (Zone ${transaction.addr2 || 'N/A'})</div>
                </div>
            </div>
        </div>
        
        <div class="detail-section">
            <h3>🕸️ Local Subgraph & Relationship Neighborhood (1-2 Hops)</h3>
            <div class="graph-container">
                <canvas id="neighborhoodCanvas"></canvas>
                <div class="graph-legend">
                    <div class="legend-item"><span class="legend-dot" style="background:#ef4444;"></span> Transaction</div>
                    <div class="legend-item"><span class="legend-dot" style="background:#38bdf8;"></span> Card Entity</div>
                    <div class="legend-item"><span class="legend-dot" style="background:#a855f7;"></span> Device</div>
                    <div class="legend-item"><span class="legend-dot" style="background:#10b981;"></span> Email Domain</div>
                    <div class="legend-item"><span class="legend-dot" style="background:#f59e0b;"></span> Billing Address</div>
                    <div class="legend-item"><span class="legend-dot" style="background:#ec4899;"></span> Shared Card Ring</div>
                </div>
            </div>
        </div>
        
        <div class="detail-section">
            <h3>⚖️ Decision Engine</h3>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-label">Automated Action</div>
                    <div class="detail-value">${(decision.decision || 'N/A').toUpperCase()}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Current Status</div>
                    <div class="detail-value"><span class="status-badge ${getStatusClass(decision.status)}">${formatStatus(decision.status || '')}</span></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Tiered GenAI Triggered</div>
                    <div class="detail-value">${decision.requires_genai ? 'Yes (≥0.80)' : 'No (<0.80)'}</div>
                </div>
            </div>
        </div>
        
        <div class="detail-section" id="geminiSection">
            <h3>🤖 Generative AI Explanation & SAR Narrative</h3>
            <div id="geminiContent">
                <button class="btn btn-primary" onclick="requestGeminiAnalysisUI(${transactionId})">✨ Generate AI Explanation & SAR Draft</button>
            </div>
        </div>
        
        <div class="detail-section">
            <h3>👮 Investigator Verdict</h3>
            <div class="controls">
                <button class="btn btn-success" onclick="submitFeedback(${transactionId}, true)">✓ Confirm Fraud & File SAR</button>
                <button class="btn btn-danger" onclick="submitFeedback(${transactionId}, false)">✗ Mark False Positive</button>
            </div>
        </div>
    `;
    
    // Draw Graph
    setTimeout(() => {
        const canvas = document.getElementById('neighborhoodCanvas');
        if (canvas) {
            const graphData = neighborhoodData || buildFallbackGraphData(transaction, decision);
            renderNeighborhoodGraph(canvas, graphData);
        }
    }, 50);
    
    // Check if Gemini analysis already exists or auto-trigger if high-risk alert
    loadGeminiAnalysis(transactionId).then(geminiData => {
        if (geminiData && geminiData.explanation) {
            renderGeminiAnalysis(geminiData, transactionId);
        } else if (decision && (decision.requires_genai || decision.risk_score >= 0.75)) {
            // Auto-trigger Gemini analysis for escalated transactions
            requestGeminiAnalysisUI(transactionId);
        }
    });
}

function buildFallbackGraphData(transaction, decision) {
    const txnId = transaction.transaction_id;
    return {
        transaction_id: txnId,
        nodes: [
            { id: `txn_${txnId}`, label: `Txn #${txnId}`, type: 'transaction', risk_score: decision.risk_score },
            { id: `card_${txnId}`, label: `Card ${formatCardInfo(transaction)}`, type: 'card' },
            { id: `dev_${txnId}`, label: transaction.device_info || 'Device Info', type: 'device' },
            { id: `email_${txnId}`, label: transaction.p_emaildomain || 'email domain', type: 'email' },
            { id: `addr_${txnId}`, label: `Addr ${transaction.addr1 || '94103'}`, type: 'address' }
        ],
        edges: [
            { source: `txn_${txnId}`, target: `card_${txnId}`, relation: 'uses_card' },
            { source: `txn_${txnId}`, target: `dev_${txnId}`, relation: 'origin_device' },
            { source: `txn_${txnId}`, target: `email_${txnId}`, relation: 'registered_email' },
            { source: `txn_${txnId}`, target: `addr_${txnId}`, relation: 'billing_addr' }
        ]
    };
}

// Interactive HTML5 Canvas Graph Renderer
function renderNeighborhoodGraph(canvas, graph) {
    if (!canvas || !graph || !graph.nodes) return;
    
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = (rect.height || 320) * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height || 320;
    const centerX = width / 2;
    const centerY = height / 2;
    
    ctx.clearRect(0, 0, width, height);
    
    // Node Color Mapping
    const typeColors = {
        transaction: '#ef4444',
        card: '#38bdf8',
        device: '#a855f7',
        email: '#10b981',
        address: '#f59e0b',
        shared_card: '#ec4899'
    };

    // Calculate positions
    const nodePositions = {};
    const centerNode = graph.nodes.find(n => n.type === 'transaction') || graph.nodes[0];
    nodePositions[centerNode.id] = { x: centerX, y: centerY, node: centerNode };
    
    const peripheralNodes = graph.nodes.filter(n => n.id !== centerNode.id);
    const innerRadius = Math.min(width, height) * 0.32;
    const outerRadius = Math.min(width, height) * 0.42;
    
    peripheralNodes.forEach((node, i) => {
        const isOuter = node.suspicious || node.id.startsWith('rcard');
        const r = isOuter ? outerRadius : innerRadius;
        const angle = (i / peripheralNodes.length) * 2 * Math.PI - (Math.PI / 2);
        nodePositions[node.id] = {
            x: centerX + r * Math.cos(angle),
            y: centerY + r * Math.sin(angle),
            node: node
        };
    });

    // Draw Edges
    (graph.edges || []).forEach(edge => {
        const p1 = nodePositions[edge.source];
        const p2 = nodePositions[edge.target];
        if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = edge.relation === 'co_used_card' ? 'rgba(236, 72, 153, 0.6)' : 'rgba(56, 189, 248, 0.25)';
            ctx.lineWidth = edge.relation === 'co_used_card' ? 2 : 1.5;
            if (edge.relation === 'co_used_card') {
                ctx.setLineDash([4, 4]);
            } else {
                ctx.setLineDash([]);
            }
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Draw edge relation label
            if (edge.relation) {
                const midX = (p1.x + p2.x) / 2;
                const midY = (p1.y + p2.y) / 2;
                ctx.fillStyle = '#64748b';
                ctx.font = '9px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(edge.relation.replace(/_/g, ' '), midX, midY - 3);
            }
        }
    });

    // Draw Nodes
    Object.values(nodePositions).forEach(({ x, y, node }) => {
        const isCenter = node.id === centerNode.id;
        const radius = isCenter ? 24 : 16;
        const color = typeColors[node.type] || '#94a3b8';
        
        // Outer Glow
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, 2 * Math.PI);
        ctx.fillStyle = isCenter ? 'rgba(239, 68, 68, 0.2)' : 'rgba(56, 189, 248, 0.1)';
        ctx.fill();

        // Node Circle
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = '#111726';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Icon/Text inside node
        ctx.fillStyle = color;
        ctx.font = isCenter ? 'bold 11px Inter, sans-serif' : '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const symbol = isCenter ? 'TX' : (node.type ? node.type[0].toUpperCase() : '•');
        ctx.fillText(symbol, x, y);

        // Node Label Pill Below
        ctx.fillStyle = '#f1f5f9';
        ctx.font = '10px Inter, sans-serif';
        const label = node.label || node.id;
        const labelWidth = ctx.measureText(label).width;
        
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.fillRect(x - (labelWidth / 2) - 4, y + radius + 4, labelWidth + 8, 16);
        
        ctx.fillStyle = '#cbd5e1';
        ctx.fillText(label, x, y + radius + 14);
    });
}

function renderGeminiAnalysis(geminiData, transactionId) {
    const geminiContent = document.getElementById('geminiContent');
    if (!geminiContent) return;
    
    geminiContent.innerHTML = `
        <div class="gemini-analysis">
            <h4>🤖 AI Risk Reasoning</h4>
            <p>${geminiData.explanation || 'No detailed reasoning provided.'}</p>
        </div>
        <div class="sar-container">
            <div class="sar-header">
                <h4>📋 Suspicious Activity Report (SAR) Draft Workspace</h4>
                <div class="sar-actions">
                    <button class="btn btn-secondary" onclick="copySarDraft()">📋 Copy</button>
                    <button class="btn btn-secondary" onclick="downloadSarDraft(${transactionId})">💾 Export</button>
                </div>
            </div>
            <textarea id="sarDraftText" class="sar-editor" rows="10">${geminiData.sar_draft || ''}</textarea>
        </div>
    `;
}

function requestGeminiAnalysisUI(transactionId) {
    const geminiContent = document.getElementById('geminiContent');
    geminiContent.innerHTML = '<div class="loading"><div class="loading-spinner"></div><p>Querying Gemini for plain-language reasoning and SAR draft...</p></div>';
    
    requestGeminiAnalysis(transactionId).then(result => {
        if (result && result.explanation) {
            renderGeminiAnalysis(result, transactionId);
            showToast('SAR Draft successfully generated by Gemini.');
        } else {
            geminiContent.innerHTML = '<p class="loading">Failed to generate AI analysis. Check GEMINI_API_KEY.</p>';
        }
    });
}

function copySarDraft() {
    const textarea = document.getElementById('sarDraftText');
    if (textarea) {
        navigator.clipboard.writeText(textarea.value).then(() => {
            showToast('✓ SAR Draft copied to clipboard!');
        }).catch(() => {
            textarea.select();
            document.execCommand('copy');
            showToast('✓ SAR Draft copied to clipboard!');
        });
    }
}

function downloadSarDraft(transactionId) {
    const textarea = document.getElementById('sarDraftText');
    if (!textarea) return;
    
    const blob = new Blob([textarea.value], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SAR_Report_Txn_${transactionId}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`✓ Exported SAR_Report_Txn_${transactionId}.txt`);
}

function submitFeedback(transactionId, confirmed) {
    const verdict = confirmed ? 'Confirmed Fraud' : 'False Positive';
    const notes = prompt(`Enter investigator resolution notes for Transaction #${transactionId} (${verdict}):`);
    if (notes === null) return; // User cancelled
    
    submitAlertFeedback(transactionId, confirmed, notes).then(result => {
        if (result && result.status === 'success') {
            showToast(`✓ Transaction #${transactionId} marked as ${verdict}`);
            closeModal();
            loadAlerts();
            loadDashboardStats();
        } else {
            alert('Failed to submit feedback.');
        }
    });
}

function showToast(message) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3200);
}

// Helpers
function formatCardInfo(transaction) {
    const cardParts = [];
    for (let i = 1; i <= 6; i++) {
        const cardVal = transaction[`card${i}`];
        if (cardVal && cardVal !== 'nan' && cardVal !== 0) {
            cardParts.push(cardVal);
        }
    }
    return cardParts.length > 0 ? cardParts.join('-') : 'N/A';
}

function formatTime(timestamp) {
    if (!timestamp) return 'N/A';
    // If timestamp is integer seconds/milliseconds
    const num = Number(timestamp);
    if (!isNaN(num)) {
        const date = num > 1e11 ? new Date(num) : new Date(num * 1000);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
    return String(timestamp);
}

function formatStatus(status) {
    if (!status) return 'Unknown';
    return status.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function getRiskClass(riskScore) {
    if (riskScore >= 0.8) return 'risk-high';
    if (riskScore >= 0.5) return 'risk-medium';
    return 'risk-low';
}

function getStatusClass(status) {
    switch (status) {
        case 'approved': return 'status-approved';
        case 'flagged':
        case 'needs_review': return 'status-flagged';
        case 'pending_investigation': return 'status-pending';
        case 'confirmed_fraud': return 'status-confirmed';
        default: return '';
    }
}

function toggleSimulator() {
    if (isSimulating) {
        clearInterval(simTimer);
        isSimulating = false;
        if (btnDemoFeed) {
            btnDemoFeed.textContent = '⚡ Stream Demo: OFF';
            btnDemoFeed.classList.remove('active');
        }
    } else {
        isSimulating = true;
        if (btnDemoFeed) {
            btnDemoFeed.textContent = '⏸ Stream Demo: ON';
            btnDemoFeed.classList.add('active');
        }
        simTimer = setInterval(() => {
            if (!isPaused) {
                const mock = generateMockTransaction();
                addTransactionToFeed(mock);
            }
        }, 1200);
    }
}

function injectBatch(count = 10) {
    for (let i = 0; i < count; i++) {
        const mock = generateMockTransaction(i % 3 === 0);
        addTransactionToFeed(mock);
    }
}

// Event Listeners
function setupEventListeners() {
    if (btnDemoFeed) {
        btnDemoFeed.addEventListener('click', toggleSimulator);
    }

    if (btnInjectBatch) {
        btnInjectBatch.addEventListener('click', () => injectBatch(10));
    }

    pauseFeedBtn.addEventListener('click', () => {
        isPaused = !isPaused;
        pauseFeedBtn.textContent = isPaused ? '▶ Resume Feed' : '⏸ Pause Feed';
        pauseFeedBtn.classList.toggle('btn-primary');
        pauseFeedBtn.classList.toggle('btn-secondary');
    });
    
    clearFeedBtn.addEventListener('click', () => {
        transactions = [];
        renderTransactionFeed();
    });
    
    refreshAlertsBtn.addEventListener('click', loadAlerts);
    alertFilter.addEventListener('change', loadAlerts);
    
    document.getElementById('transactionModal').addEventListener('click', (e) => {
        if (e.target.id === 'transactionModal') {
            closeModal();
        }
    });
    
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
}
