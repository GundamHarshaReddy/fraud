/**
 * 3D Fraud Network Topology & GNN Constellation
 * Interactive Three.js WebGL visualization of real-time transactions,
 * entity relationships, shared card rings, and fraud clusters.
 */

// Configuration & Endpoints
const isDirectPort = window.location.port === '8000';
const API_BASE_URL = isDirectPort ? 'http://localhost:8000' : (window.location.origin.includes('localhost') ? 'http://localhost:8000' : window.location.origin);
const WS_URL = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') +
    (window.location.hostname || 'localhost') +
    ':8000/ws/live';

// Graph State
let graph = null;
let graphData = { nodes: [], links: [] };
const nodesMap = new Map();
const linksMap = new Map();
const entityDegrees = new Map(); // tracks degree for fraud ring detection

// Streaming & UI State
let websocket = null;
let starField = null;
let autoRotate = true;
let isUserDragging = false;
let showLinkParticles = true;
let activeFilter = 'all'; // 'all' | 'high_risk' | 'rings'
let searchQuery = '';
let recentEvents = 0;
let lastVelocityTick = Date.now();
let selectedNode = null;
let hoveredNode = null;
let hoverDebounceTimer = null;
let pendingSingleTxns = [];
let flushTimer = null;
const MAX_NODES = 1000; // Expanded to 1,000 nodes with high-performance Barnes-Hut optimization
let spacingMultiplier = 1.4; // Default spacious node spread

// Texture Cache for Avatar & Entity Sprites
const textureCache = new Map();

// Color Palette
const COLORS = {
    txSafe: '#38bdf8',       // Glowing cyan
    txRisk: '#ef4444',       // Glowing crimson
    txMed: '#f59e0b',        // Glowing amber
    card: '#f59e0b',         // Radiant amber
    device: '#a855f7',       // Radiant purple
    email: '#10b981',        // Emerald
    address: '#06b6d4',      // Teal
    linkDefault: 'rgba(56, 189, 248, 0.35)',
    linkRisk: 'rgba(239, 68, 68, 0.65)'
};

// DOM References
const container = document.getElementById('graph-3d-container');
const wsStatusDot = document.getElementById('wsStatusDot');
const wsStatusText = document.getElementById('wsStatusText');
const statNodeCount = document.getElementById('statNodeCount');
const statLinkCount = document.getElementById('statLinkCount');
const statHighRiskCount = document.getElementById('statHighRiskCount');
const statIngestRate = document.getElementById('statIngestRate');
const hoverInspector = document.getElementById('hoverInspector');
const detailDrawer = document.getElementById('detailDrawer');
const drawerTitle = document.getElementById('drawerTitle');
const drawerBadge = document.getElementById('drawerBadge');
const drawerContent = document.getElementById('drawerContent');
const closeDrawerBtn = document.getElementById('closeDrawerBtn');
const nodeSearch = document.getElementById('nodeSearch');
const btnFilterAll = document.getElementById('btnFilterAll');
const btnFilterHighRisk = document.getElementById('btnFilterHighRisk');
const btnFilterRings = document.getElementById('btnFilterRings');
const btnAutoRotate = document.getElementById('btnAutoRotate');
const btnInjectBatch = document.getElementById('btnInjectBatch');
const btnParticles = document.getElementById('btnParticles');
const btnResetCamera = document.getElementById('btnResetCamera');
const btnClearGraph = document.getElementById('btnClearGraph');
const sliderSpread = document.getElementById('sliderSpread');
const labelSpreadVal = document.getElementById('labelSpreadVal');
let antiSpinEnabled = true;
let isLayoutLocked = false;

/**
 * Custom D3 Anti-Torque / Rotational Damping Force
 * Cancels the net angular momentum of the 3D graph on every simulation tick.
 * Eliminates the notorious 3D force layout spinning/tumbling issue without locking node translations.
 */
function createAntiTorqueForce() {
    let nodes = [];

    function force() {
        if (!antiSpinEnabled || !nodes || nodes.length < 3) return;

        const n = nodes.length;
        let cx = 0, cy = 0, cz = 0;

        // 1. Calculate centroid
        for (let i = 0; i < n; i++) {
            cx += nodes[i].x || 0;
            cy += nodes[i].y || 0;
            cz += nodes[i].z || 0;
        }
        cx /= n;
        cy /= n;
        cz /= n;

        // 2. Calculate angular momentum L = sum(r x v) and moment of inertia scalar I
        let lx = 0, ly = 0, lz = 0;
        let inertia = 0;

        for (let i = 0; i < n; i++) {
            const node = nodes[i];
            const rx = (node.x || 0) - cx;
            const ry = (node.y || 0) - cy;
            const rz = (node.z || 0) - cz;

            const vx = node.vx || 0;
            const vy = node.vy || 0;
            const vz = node.vz || 0;

            // Cross product: r x v
            lx += ry * vz - rz * vy;
            ly += rz * vx - rx * vz;
            lz += rx * vy - ry * vx;

            inertia += rx * rx + ry * ry + rz * rz;
        }

        if (inertia < 1e-4) return;

        // 3. Angular velocity vector omega = L / I
        const wx = lx / inertia;
        const wy = ly / inertia;
        const wz = lz / inertia;

        // 4. Subtract rigid body rotation velocity (omega x r) from each node's velocity
        for (let i = 0; i < n; i++) {
            const node = nodes[i];
            const rx = (node.x || 0) - cx;
            const ry = (node.y || 0) - cy;
            const rz = (node.z || 0) - cz;

            const rotVx = wy * rz - wz * ry;
            const rotVy = wz * rx - wx * rz;
            const rotVz = wx * ry - wy * rx;

            node.vx -= rotVx;
            node.vy -= rotVy;
            node.vz -= rotVz;
        }
    }

    force.initialize = function (_nodes) {
        nodes = _nodes;
    };

    return force;
}

/**
 * Global gravity: gently pulls every node toward the origin so
 * disconnected hubs don't drift apart indefinitely.
 */
function createGravityForce(strength = 0.015) {
    let nodes = [];

    function force(alpha) {
        for (let i = 0; i < nodes.length; i++) {
            const n = nodes[i];
            n.vx -= (n.x || 0) * strength * alpha;
            n.vy -= (n.y || 0) * strength * alpha;
            n.vz -= (n.z || 0) * strength * alpha;
        }
    }

    force.initialize = function (_nodes) { nodes = _nodes; };
    force.strength = function (v) {
        if (v === undefined) return strength;
        strength = v;
        return force;
    };

    return force;
}

/**
 * Radial Shell Force: organizes nodes into concentric spherical shells by entity type.
 * Transactions cluster inward, cards/devices form a middle ring, emails/addresses outer ring.
 */
function createRadialShellForce(strength = 0.035) {
    let nodes = [];
    const shellRadius = { transaction: 95, card: 190, device: 190, email: 280, address: 280 };

    function force(alpha) {
        for (let i = 0; i < nodes.length; i++) {
            const n = nodes[i];
            const target = shellRadius[n.type] || 160;
            const dist = Math.hypot(n.x || 0, n.y || 0, n.z || 0) || 1;
            const diff = (target - dist) / dist;
            n.vx += (n.x || 0) * diff * strength * alpha;
            n.vy += (n.y || 0) * diff * strength * alpha;
            n.vz += (n.z || 0) * diff * strength * alpha;
        }
    }
    force.initialize = function (_nodes) { nodes = _nodes; };
    return force;
}

/**
 * Configure dynamic repulsive charge and link lengths based on spacing multiplier
 */
function updateGraphForces() {
    if (!graph) return;
    const chargeForce = graph.d3Force('charge');
    if (chargeForce) {
        chargeForce.strength(node => {
            const degree = entityDegrees.get(node.id) || 1;
            const baseCharge = node.type === 'transaction' ? -200 : -300;
            const degreeCharge = -Math.min(degree * 25, 200);
            return (baseCharge + degreeCharge) * (spacingMultiplier || 1.4);
        });
        chargeForce.distanceMin(45);
        chargeForce.distanceMax(580); // Scaled boundary for 1,000 nodes
        if (typeof chargeForce.theta === 'function') {
            chargeForce.theta(1.25); // Optimized Barnes-Hut quadtree scaling for 1,000 nodes at 60 FPS
        }
    }
    const linkForce = graph.d3Force('link');
    if (linkForce) {
        linkForce.distance(link => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            const deg = Math.max(entityDegrees.get(sourceId) || 1, entityDegrees.get(targetId) || 1);
            const extra = Math.min(deg * 8, 70);
            const baseDist = link.isHighRisk ? 115 : 90;
            return (baseDist + extra) * (spacingMultiplier || 1.4);
        });
    }
}

/**
 * Distribute newly spawned nodes across a 3D spherical shell so they never spawn on top of each other
 */
function getDistributedSpawnPosition(anchorX, anchorY, anchorZ, baseRadius = 110) {
    const r = (baseRadius + (Math.random() - 0.5) * 45) * (spacingMultiplier || 1.4);
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos((Math.random() * 2) - 1);
    return {
        x: anchorX + r * Math.sin(phi) * Math.cos(theta),
        y: anchorY + r * Math.sin(phi) * Math.sin(theta),
        z: anchorZ + r * Math.cos(phi)
    };
}

// Initialize on Load
function startApp() {
    init3DGraph();
    setupEventListeners();
    connectWebSocket();
    loadInitialTransactions();
    startVelocityMonitor();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startApp);
} else {
    startApp();
}

/**
 * Build 3D Force-Directed Graph using WebGL
 */
function init3DGraph() {
    if (typeof ForceGraph3D === 'undefined') {
        console.error('ForceGraph3D library not loaded.');
        return;
    }

    graph = ForceGraph3D({ controlType: 'orbit' })(container)
        .backgroundColor('rgba(0, 0, 0, 0)') // transparent to reveal cosmic space nebula
        .showNavInfo(false)
        .nodeThreeObject(createNodeSprite)
        .nodeLabel('') // using custom HUD hover inspector instead of raw tooltip
        .linkWidth(link => {
            if (link.isHighRisk) return 2.2;
            return 1.0;
        })
        .linkColor(link => {
            if (link.isHighRisk) return COLORS.linkRisk;
            return COLORS.linkDefault;
        })
        .linkDirectionalParticles(link => showLinkParticles ? (link.isHighRisk ? 3 : 1) : 0)
        .linkDirectionalParticleWidth(link => link.isHighRisk ? 2.2 : 1.2)
        .linkDirectionalParticleSpeed(link => link.isHighRisk ? 0.012 : 0.007)
        .linkDirectionalParticleColor(link => link.isHighRisk ? '#ef4444' : '#38bdf8')
        .onNodeHover(handleNodeHover)
        .onNodeClick(handleNodeClick)
        .onBackgroundClick(() => {
            if (selectedNode) {
                selectedNode = null;
                detailDrawer.classList.remove('open');
            }
            if (hoveredNode) {
                hoveredNode = null;
                refreshHighlight();
            }
        });

    // Configure dynamic force physics with anti-spin stabilization, global gravity, and radial shell layering
    updateGraphForces();
    graph.d3Force('antiTorque', createAntiTorqueForce());
    graph.d3Force('gravity', createGravityForce(0.015));
    graph.d3Force('radialShell', createRadialShellForce(0.035));

    // Balanced velocity decay and lowered alpha decay allows nodes to spread into stable, calm orbits
    graph.d3VelocityDecay(0.40);
    graph.d3AlphaDecay(0.020);
    graph.cooldownTicks(120);

    // Add 3D Starfield & Space Particle Field
    initStarfield(graph.scene());

    // Position camera initially to frame the spacious 1,000-node constellation
    graph.cameraPosition({ x: 0, y: 50, z: 720 });

    // Initialize OrbitControls configuration
    syncControls();
    setTimeout(syncControls, 100);
    setTimeout(syncControls, 400);

    // Handle Window Resize
    window.addEventListener('resize', () => {
        if (graph) {
            graph.width(container.clientWidth);
            graph.height(container.clientHeight);
        }
    });

    // Cosmic Starfield Rotation (strictly active ONLY when autoRotate is true)
    setInterval(() => {
        if (autoRotate && starField) {
            starField.rotation.y += 0.0003;
            starField.rotation.x += 0.0001;
        }
    }, 25);
}

/**
 * Synchronize OrbitControls with active autoRotate state
 */
function syncControls() {
    const controls = graph ? graph.controls() : null;
    if (controls) {
        controls.autoRotate = autoRotate;
        controls.autoRotateSpeed = 1.0;
        controls.zoomSpeed = 5.0; // 5x zoom sensitivity
        controls.minDistance = 20;
        controls.maxDistance = 3500; // Expanded zoom out range for 1,000 nodes
        controls.enableDamping = false; // Disable damping inertia so rotation stops instantly when toggled off!

        // Track camera drag/orbit to avoid hover card flickering and wasted raycasts
        if (!controls.__hasDragListeners) {
            controls.addEventListener('start', () => {
                isUserDragging = true;
                if (hoverInspector) hoverInspector.classList.remove('visible');
            });
            controls.addEventListener('end', () => {
                isUserDragging = false;
            });
            controls.__hasDragListeners = true;
        }
    }
}

/**
 * Initialize 3D Stars & Deep-Space Celestial Particles
 */
function initStarfield(scene) {
    if (!scene || typeof THREE === 'undefined') return;

    const starCount = 2600;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount; i++) {
        // Distribute stars in spherical shell around graph
        const r = 550 + Math.random() * 1800;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos((Math.random() * 2) - 1);

        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = r * Math.cos(phi);

        // Color distribution: 60% Diamond White, 20% Cyan Blue, 10% Soft Violet, 10% Golden
        const rand = Math.random();
        if (rand > 0.82) {
            // Electric cyan/blue stars
            colors[i * 3] = 0.45;
            colors[i * 3 + 1] = 0.8;
            colors[i * 3 + 2] = 1.0;
        } else if (rand > 0.72) {
            // Soft violet / nebula purple stars
            colors[i * 3] = 0.8;
            colors[i * 3 + 1] = 0.6;
            colors[i * 3 + 2] = 1.0;
        } else if (rand > 0.62) {
            // Warm golden stellar stars
            colors[i * 3] = 1.0;
            colors[i * 3 + 1] = 0.88;
            colors[i * 3 + 2] = 0.55;
        } else {
            // Radiant diamond white
            colors[i * 3] = 0.98;
            colors[i * 3 + 1] = 0.98;
            colors[i * 3 + 2] = 1.0;
        }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Create soft circular glowing star particle texture
    const starCanvas = document.createElement('canvas');
    starCanvas.width = 32;
    starCanvas.height = 32;
    const sCtx = starCanvas.getContext('2d');
    const grad = sCtx.createRadialGradient(16, 16, 2, 16, 16, 15);
    grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
    grad.addColorStop(0.35, 'rgba(255, 255, 255, 0.8)');
    grad.addColorStop(0.75, 'rgba(180, 220, 255, 0.25)');
    grad.addColorStop(1, 'transparent');
    sCtx.fillStyle = grad;
    sCtx.fillRect(0, 0, 32, 32);

    const starTexture = new THREE.CanvasTexture(starCanvas);

    const material = new THREE.PointsMaterial({
        size: 4.8,
        map: starTexture,
        vertexColors: true,
        transparent: true,
        opacity: 0.92,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });

    starField = new THREE.Points(geometry, material);
    scene.add(starField);
}

/**
 * Generate glowing avatar silhouette texture & Three.js Sprite
 * Matches the reference image with glowing user silhouettes & luminous halos.
 */
function createNodeSprite(node) {
    const isAlert = (node.risk_score || 0) >= 0.75 || Boolean(node.isAlert) || Boolean(node.isHighRisk);
    const isMedium = (node.risk_score || 0) >= 0.5 && !isAlert;
    const type = node.type || 'transaction';

    // Determine Color
    let color = COLORS.txSafe;
    if (type === 'transaction') {
        color = isAlert ? COLORS.txRisk : (isMedium ? COLORS.txMed : COLORS.txSafe);
    } else if (type === 'card') {
        color = COLORS.card;
    } else if (type === 'device') {
        color = COLORS.device;
    } else if (type === 'email' || type === 'address') {
        color = COLORS.email;
    }

    // Cache key based on visual attributes
    const cacheKey = `${type}_${color}_${isAlert ? 'alert' : 'norm'}`;
    let texture = textureCache.get(cacheKey);

    if (!texture) {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');

        const cx = 64;
        const cy = 64;

        // 1. Luminous Radial Glow Bloom
        const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, 60);
        grad.addColorStop(0, color);
        grad.addColorStop(0.35, hexToRgba(color, 0.45));
        grad.addColorStop(0.7, hexToRgba(color, 0.12));
        grad.addColorStop(1, 'transparent');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, 60, 0, Math.PI * 2);
        ctx.fill();

        // 2. Draw Icon / Silhouette inside
        ctx.save();
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = color;
        ctx.shadowBlur = 14;

        if (type === 'transaction') {
            // Human / User Silhouette (Head + Curved Shoulders) matching photo
            // Head
            ctx.beginPath();
            ctx.arc(cx, cy - 10, 11, 0, Math.PI * 2);
            ctx.fill();

            // Shoulders / Torso
            ctx.beginPath();
            ctx.moveTo(cx - 20, cy + 24);
            ctx.bezierCurveTo(cx - 18, cy + 5, cx + 18, cy + 5, cx + 20, cy + 24);
            ctx.closePath();
            ctx.fill();
        } else if (type === 'card') {
            // Card Entity (Rounded rectangle with magnetic stripe)
            drawRoundedRect(ctx, cx - 18, cy - 12, 36, 24, 4);
            ctx.fill();
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(cx - 18, cy - 4, 36, 5);
        } else if (type === 'device') {
            // Mobile Device silhouette
            drawRoundedRect(ctx, cx - 12, cy - 20, 24, 40, 5);
            ctx.fill();
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(cx - 9, cy - 14, 18, 28);
        } else {
            // Location / Email Orb with concentric ring
            ctx.beginPath();
            ctx.arc(cx, cy, 14, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(cx, cy, 22, 0, Math.PI * 2);
            ctx.stroke();
        }
        ctx.restore();

        // 3. Subtle Outer Ring for High Risk Alerts
        if (isAlert) {
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(cx, cy, 38, 0, Math.PI * 2);
            ctx.stroke();
        }

        texture = new THREE.CanvasTexture(canvas);
        textureCache.set(cacheKey, texture);
    }

    const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending
    });

    const sprite = new THREE.Sprite(material);
    const scale = isAlert ? 32 : (type === 'transaction' ? 24 : 18);
    sprite.scale.set(scale, scale, 1);

    // Invisible larger hit-area sprite for effortless cursor hovering (~2.2x larger hit zone)
    const hitMaterial = new THREE.SpriteMaterial({
        opacity: 0,
        transparent: true,
        depthWrite: false
    });
    const hitSprite = new THREE.Sprite(hitMaterial);
    hitSprite.scale.set(scale * 2.2, scale * 2.2, 1);

    const group = new THREE.Group();
    group.add(sprite);
    group.add(hitSprite);

    group.userData = { node, sprite };
    group.__data = node;
    sprite.__data = node;
    hitSprite.__data = node;
    node.__sprite = sprite;
    node.__group = group;

    return group;
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * WebSocket Connection for Real-Time Streaming
 */
function connectWebSocket() {
    try {
        websocket = new WebSocket(WS_URL);

        websocket.onopen = () => {
            wsStatusDot.className = 'status-dot connected';
            wsStatusText.textContent = 'Live Topology Stream Connected';
            showToast('✓ Live WebSocket stream connected');
        };

        websocket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'transaction') {
                    if (Array.isArray(message.data)) {
                        handleBatchTransactions(message.data);
                    } else {
                        // Buffer single transactions over 200ms to eliminate micro-reheats and jitter
                        pendingSingleTxns.push(message.data);
                        recentEvents++;
                        if (!flushTimer) {
                            flushTimer = setTimeout(() => {
                                if (pendingSingleTxns.length > 0) {
                                    handleBatchTransactions(pendingSingleTxns);
                                    pendingSingleTxns = [];
                                }
                                flushTimer = null;
                            }, 200);
                        }
                    }
                } else if (message.type === 'batch') {
                    handleBatchTransactions(message.data || message.transactions);
                }
            } catch (err) {
                console.error('Error parsing WebSocket data:', err);
            }
        };

        websocket.onerror = () => {
            wsStatusDot.className = 'status-dot disconnected';
            wsStatusText.textContent = 'Stream Disconnected';
        };

        websocket.onclose = () => {
            wsStatusDot.className = 'status-dot disconnected';
            wsStatusText.textContent = 'Reconnecting...';
            setTimeout(connectWebSocket, 4000);
        };
    } catch (e) {
        console.warn('WebSocket init failed, will retry:', e);
        setTimeout(connectWebSocket, 5000);
    }
}

/**
 * Ingest single transaction into graph model
 */
function handleIncomingTransaction(payload, autoSync = true) {
    if (!payload) return;
    // Safely extract transaction and decision regardless of nesting level
    let txn = payload.transaction || payload;
    if (txn && txn.transaction && txn.transaction.transaction_id) {
        txn = txn.transaction;
    }
    const decision = payload.decision || (payload.transaction && payload.transaction.decision) || {};
    if (!txn || !txn.transaction_id) return;

    const txnId = txn.transaction_id;
    const riskScore = decision.risk_score !== undefined ? Number(decision.risk_score) : (txn.is_fraud ? 0.88 : 0.12);
    const isAlert = riskScore >= 0.75 ||
        decision.decision === 'escalate' ||
        decision.status === 'pending_investigation' ||
        Boolean(txn.is_fraud);
    const amount = Number(txn.transaction_amt || txn.amount || 0);

    // Helper: Verify entity value is a genuine identifier, not a missing/dummy placeholder
    function isValidEntity(val) {
        if (val === null || val === undefined) return false;
        const s = String(val).trim().toLowerCase();
        return s !== '' && s !== '0' && s !== '0.0' && s !== '-1' && s !== 'unknown' && s !== 'none' && s !== 'nan' && s !== 'null';
    }

    // 0. Compute local anchor coordinates so new nodes are born near their partners
    // This completely eliminates violent spring-snapping and rotational torque!
    const cardKey = [txn.card1, txn.card2, txn.card3].filter(Boolean).join('-') || '1352-450';
    const cardNodeId = `card_${cardKey}`;

    // Genuine Device determination (Strictly filter dummy 0.0 placeholders)
    const rawDev = txn.device_info || txn.device_type;
    const hasValidDev = isValidEntity(rawDev);
    let devNodeId = null;
    let devLabel = null;
    if (hasValidDev) {
        devLabel = String(rawDev).trim();
        // Clean mapping for factorized IDs if any exist in legacy cache
        if (devLabel === '1.0' || devLabel === '1') devLabel = 'iOS Device / Mobile';
        else if (devLabel === '2.0' || devLabel === '2') devLabel = 'Windows / Chrome';
        else if (devLabel === '3.0' || devLabel === '3') devLabel = 'Android / Mobile';
        else if (devLabel === '4.0' || devLabel === '4') devLabel = 'MacOS / Safari';
        devNodeId = `dev_${devLabel.replace(/\s+/g, '_')}`;
    }

    const existingCard = nodesMap.get(cardNodeId);
    const existingDev = devNodeId ? nodesMap.get(devNodeId) : null;

    let anchorX = 0, anchorY = 0, anchorZ = 0;
    if (existingCard && typeof existingCard.x === 'number') {
        anchorX = existingCard.x;
        anchorY = existingCard.y;
        anchorZ = existingCard.z;
    } else if (existingDev && typeof existingDev.x === 'number') {
        anchorX = existingDev.x;
        anchorY = existingDev.y;
        anchorZ = existingDev.z;
    } else if (nodesMap.size > 0) {
        let count = 0;
        nodesMap.forEach(n => {
            if (typeof n.x === 'number') {
                anchorX += n.x; anchorY += n.y; anchorZ += n.z;
                count++;
            }
        });
        if (count > 0) {
            anchorX /= count; anchorY /= count; anchorZ /= count;
        }
    }

    // 1. Transaction Node
    const txnNodeId = `txn_${txnId}`;
    if (!nodesMap.has(txnNodeId)) {
        const txnNode = {
            id: txnNodeId,
            rawId: txnId,
            type: 'transaction',
            label: `Txn #${txnId}`,
            amount: amount,
            risk_score: riskScore,
            isAlert: isAlert,
            isHighRisk: isAlert,
            requires_genai: isAlert || (riskScore >= 0.75) || (decision.requires_genai === true || decision.requires_genai === 'true'),
            status: decision.status || (isAlert ? 'flagged' : 'approved'),
            decision: decision.decision || (isAlert ? 'escalate' : 'approve'),
            timestamp: txn.transaction_dt || Date.now(),
            rawTxn: txn,
            rawDecision: decision,
            neighbors: new Set(),
            // Spherical shell placement eliminates initial clustering & overlaps!
            ...getDistributedSpawnPosition(anchorX, anchorY, anchorZ, 110),
            vx: 0, vy: 0, vz: 0
        };
        if (isLayoutLocked) {
            txnNode.fx = txnNode.x;
            txnNode.fy = txnNode.y;
            txnNode.fz = txnNode.z;
        }
        nodesMap.set(txnNodeId, txnNode);
    } else if (isAlert) {
        const existingTxn = nodesMap.get(txnNodeId);
        existingTxn.isAlert = true;
        existingTxn.isHighRisk = true;
        existingTxn.requires_genai = true;
        existingTxn.risk_score = Math.max(existingTxn.risk_score || 0, riskScore);
    }

    // 2. Card Entity Node
    if (!nodesMap.has(cardNodeId)) {
        const cardNode = {
            id: cardNodeId,
            rawId: cardKey,
            type: 'card',
            label: `Card ${cardKey}`,
            risk_score: isAlert ? 0.85 : riskScore * 0.9,
            isHighRisk: isAlert,
            neighbors: new Set(),
            ...getDistributedSpawnPosition(anchorX, anchorY, anchorZ, 130),
            vx: 0, vy: 0, vz: 0
        };
        if (isLayoutLocked) {
            cardNode.fx = cardNode.x;
            cardNode.fy = cardNode.y;
            cardNode.fz = cardNode.z;
        }
        nodesMap.set(cardNodeId, cardNode);
    } else if (isAlert) {
        const card = nodesMap.get(cardNodeId);
        card.risk_score = Math.max(card.risk_score || 0, 0.85);
        card.isHighRisk = true;
    }

    // Connect Txn -> Card
    addLink(txnNodeId, cardNodeId, 'uses_card', isAlert);

    // 3. Device Entity Node (ONLY created if genuine device fingerprint exists)
    if (hasValidDev && devNodeId) {
        if (!nodesMap.has(devNodeId)) {
            const devNode = {
                id: devNodeId,
                rawId: devLabel,
                type: 'device',
                label: `${devLabel}`,
                risk_score: isAlert ? 0.85 : 0.1,
                isHighRisk: isAlert,
                neighbors: new Set(),
                ...getDistributedSpawnPosition(anchorX, anchorY, anchorZ, 140),
                vx: 0, vy: 0, vz: 0
            };
            if (isLayoutLocked) {
                devNode.fx = devNode.x;
                devNode.fy = devNode.y;
                devNode.fz = devNode.z;
            }
            nodesMap.set(devNodeId, devNode);
        } else if (isAlert) {
            const dev = nodesMap.get(devNodeId);
            dev.risk_score = Math.max(dev.risk_score || 0, 0.85);
            dev.isHighRisk = true;
        }

        // Connect Txn -> Device
        addLink(txnNodeId, devNodeId, 'origin_device', isAlert);
    }

    // 4. Email Domain (ONLY created if genuine email domain exists)
    if (isValidEntity(txn.p_emaildomain)) {
        let emailDomain = String(txn.p_emaildomain).trim();
        if (emailDomain === '1.0' || emailDomain === '1') emailDomain = 'gmail.com';
        else if (emailDomain === '2.0' || emailDomain === '2') emailDomain = 'yahoo.com';
        else if (emailDomain === '3.0' || emailDomain === '3') emailDomain = 'outlook.com';

        const emailNodeId = `email_${emailDomain.replace(/\s+/g, '_')}`;
        if (!nodesMap.has(emailNodeId)) {
            const emailNode = {
                id: emailNodeId,
                rawId: emailDomain,
                type: 'email',
                label: emailDomain,
                risk_score: isAlert ? 0.75 : 0.2,
                isHighRisk: isAlert,
                neighbors: new Set(),
                ...getDistributedSpawnPosition(anchorX, anchorY, anchorZ, 150),
                vx: 0, vy: 0, vz: 0
            };
            if (isLayoutLocked) {
                emailNode.fx = emailNode.x;
                emailNode.fy = emailNode.y;
                emailNode.fz = emailNode.z;
            }
            nodesMap.set(emailNodeId, emailNode);
        }
        addLink(txnNodeId, emailNodeId, 'registered_email', isAlert);
    }

    // 5. Billing Address Node (addr1 - geographical billing cluster)
    if (isValidEntity(txn.addr1)) {
        const addrVal = String(txn.addr1).trim();
        const addrNodeId = `addr_Zip_${addrVal}`;
        if (!nodesMap.has(addrNodeId)) {
            const addrNode = {
                id: addrNodeId,
                rawId: addrVal,
                type: 'address',
                label: `Zip ${addrVal}`,
                risk_score: isAlert ? 0.65 : 0.15,
                isHighRisk: isAlert,
                neighbors: new Set(),
                ...getDistributedSpawnPosition(anchorX, anchorY, anchorZ, 170),
                vx: 0, vy: 0, vz: 0
            };
            if (isLayoutLocked) {
                addrNode.fx = addrNode.x;
                addrNode.fy = addrNode.y;
                addrNode.fz = addrNode.z;
            }
            nodesMap.set(addrNodeId, addrNode);
        }
        addLink(txnNodeId, addrNodeId, 'billing_address', isAlert);
    }

    // Prune oldest non-alert nodes if graph grows too large
    if (nodesMap.size > MAX_NODES) {
        pruneOldNodes();
    }

    if (autoSync) {
        syncGraphData();
    }
}

/**
 * Ingest a batch of transactions atomically with a single graph update pass.
 * Prevents multiple micro-reheats and physics stutter when injecting 20+ nodes at once.
 */
function handleBatchTransactions(items) {
    if (!Array.isArray(items) || items.length === 0) return;

    for (let i = 0; i < items.length; i++) {
        handleIncomingTransaction(items[i], false); // Ingest into nodesMap & linksMap without rendering
        recentEvents++;
    }

    // Prune oldest non-alert nodes if graph exceeds MAX_NODES
    while (nodesMap.size > MAX_NODES) {
        pruneOldNodes();
    }

    // Single atomic graph synchronization pass
    syncGraphData();
    updateGraphForces();

    // Gentle physics reheat so the new batch settles smoothly into orbits
    if (graph) {
        graph.d3ReheatSimulation();
    }
}

/**
 * Test Injector: Generates and injects 20 realistic transactions atomically
 */
function inject20NodesBatch() {
    const batch = [];
    for (let i = 0; i < 20; i++) {
        batch.push(generateClientMockTransaction(i % 5 === 0));
    }
    handleBatchTransactions(batch);
    showToast(`⚡ Injected batch of 20 nodes (${nodesMap.size} total live entities)`);
}

function addLink(sourceId, targetId, relation, isHighRisk) {
    const linkId = `${sourceId}__${targetId}`;
    if (!linksMap.has(linkId)) {
        const link = {
            id: linkId,
            source: sourceId,
            target: targetId,
            relation: relation,
            isHighRisk: Boolean(isHighRisk)
        };
        linksMap.set(linkId, link);

        // Track degrees
        entityDegrees.set(sourceId, (entityDegrees.get(sourceId) || 0) + 1);
        entityDegrees.set(targetId, (entityDegrees.get(targetId) || 0) + 1);

        const sNode = nodesMap.get(sourceId);
        const tNode = nodesMap.get(targetId);
        if (sNode) sNode.neighbors.add(targetId);
        if (tNode) tNode.neighbors.add(sourceId);
    } else if (isHighRisk) {
        const existing = linksMap.get(linkId);
        existing.isHighRisk = true;
    }
}

function pruneOldNodes() {
    for (const [id, node] of nodesMap.entries()) {
        if (node.type === 'transaction' && (node.risk_score || 0) < 0.5) {
            // Remove node & its links
            nodesMap.delete(id);
            for (const [lId, l] of linksMap.entries()) {
                const s = typeof l.source === 'object' ? l.source.id : l.source;
                const t = typeof l.target === 'object' ? l.target.id : l.target;
                if (s === id || t === id) {
                    linksMap.delete(lId);
                }
            }
            break;
        }
    }
}

/**
 * Synchronize Data to 3D-Force-Graph Instance with Filtering
 */
function syncGraphData(shouldRefocus = false) {
    let filteredNodes = [];
    let validLinkObjects = [];

    // Helper: Determine if transaction node is high-risk
    const isHighRiskTxn = (n) => {
        if (!n || n.type !== 'transaction') return false;
        return (n.risk_score || 0) >= 0.75 ||
            n.isAlert === true ||
            n.isHighRisk === true ||
            n.status === 'flagged' ||
            n.status === 'pending_investigation' ||
            n.decision === 'escalate' ||
            (n.rawTxn && n.rawTxn.is_fraud === 1);
    };

    if (activeFilter === 'high_risk') {
        // 1. Identify all high-risk transactions
        const highRiskTxnIds = new Set();
        nodesMap.forEach(n => {
            if (isHighRiskTxn(n)) {
                highRiskTxnIds.add(n.id);
            }
        });

        // 2. Identify all connected entity nodes (cards, devices, emails) directly involved in these fraud alerts
        const connectedEntityIds = new Set();
        linksMap.forEach(l => {
            const sId = typeof l.source === 'object' ? l.source.id : l.source;
            const tId = typeof l.target === 'object' ? l.target.id : l.target;

            const sHigh = highRiskTxnIds.has(sId);
            const tHigh = highRiskTxnIds.has(tId);

            if (sHigh || tHigh || l.isHighRisk) {
                if (sHigh) connectedEntityIds.add(tId);
                if (tHigh) connectedEntityIds.add(sId);

                validLinkObjects.push({
                    id: l.id,
                    source: sId,
                    target: tId,
                    relation: l.relation,
                    isHighRisk: true // Elevate all links in high-risk view to glow red
                });
            }
        });

        const highRiskAllowedIds = new Set([...highRiskTxnIds, ...connectedEntityIds]);
        // Also include any standalone node explicitly marked high risk
        nodesMap.forEach(n => {
            if ((n.risk_score || 0) >= 0.8 || n.isHighRisk) {
                highRiskAllowedIds.add(n.id);
            }
        });

        filteredNodes = Array.from(nodesMap.values()).filter(n => highRiskAllowedIds.has(n.id));

    } else if (activeFilter === 'rings') {
        // 1. Identify entities (cards, devices) shared by 2+ transactions (potential fraud rings)
        const entityTxnMap = new Map();
        linksMap.forEach(l => {
            const sId = typeof l.source === 'object' ? l.source.id : l.source;
            const tId = typeof l.target === 'object' ? l.target.id : l.target;
            if (sId.startsWith('txn_') && !tId.startsWith('txn_')) {
                if (!entityTxnMap.has(tId)) entityTxnMap.set(tId, new Set());
                entityTxnMap.get(tId).add(sId);
            } else if (tId.startsWith('txn_') && !sId.startsWith('txn_')) {
                if (!entityTxnMap.has(sId)) entityTxnMap.set(sId, new Set());
                entityTxnMap.get(sId).add(tId);
            }
        });

        const ringNodeIds = new Set();
        entityTxnMap.forEach((txSet, entityId) => {
            if (txSet.size >= 2) {
                ringNodeIds.add(entityId);
                txSet.forEach(txId => ringNodeIds.add(txId));
            }
        });

        filteredNodes = Array.from(nodesMap.values()).filter(n => ringNodeIds.has(n.id));
        const ringValidIds = new Set(filteredNodes.map(n => n.id));

        linksMap.forEach(l => {
            const sId = typeof l.source === 'object' ? l.source.id : l.source;
            const tId = typeof l.target === 'object' ? l.target.id : l.target;
            if (ringValidIds.has(sId) && ringValidIds.has(tId)) {
                validLinkObjects.push({
                    id: l.id,
                    source: sId,
                    target: tId,
                    relation: l.relation,
                    isHighRisk: l.isHighRisk
                });
            }
        });

    } else {
        // 'all' filter: Include all nodes & links
        filteredNodes = Array.from(nodesMap.values());
        linksMap.forEach(l => {
            const sId = typeof l.source === 'object' ? l.source.id : l.source;
            const tId = typeof l.target === 'object' ? l.target.id : l.target;
            validLinkObjects.push({
                id: l.id,
                source: sId,
                target: tId,
                relation: l.relation,
                isHighRisk: l.isHighRisk
            });
        });
    }

    // Apply Search Filter if query typed
    if (searchQuery) {
        const q = searchQuery.toLowerCase();
        filteredNodes = filteredNodes.filter(n =>
            (n.label && n.label.toLowerCase().includes(q)) ||
            (n.id && n.id.toLowerCase().includes(q)) ||
            (n.rawId && String(n.rawId).toLowerCase().includes(q))
        );
    }

    const validNodeIds = new Set(filteredNodes.map(n => n.id));

    // Deselect if active selected node was filtered out
    if (selectedNode && !validNodeIds.has(selectedNode.id)) {
        selectedNode = null;
        if (detailDrawer) detailDrawer.classList.remove('open');
    }

    // Filter links so both ends exist in validNodeIds
    const filteredLinks = validLinkObjects.filter(l =>
        validNodeIds.has(l.source) && validNodeIds.has(l.target)
    );

    graphData = { nodes: filteredNodes, links: filteredLinks };
    if (graph) {
        graph.graphData(graphData);
    }

    updateHUDStats(filteredNodes.length, filteredLinks.length);

    // Optional camera re-framing when user explicitly clicks a filter button
    if (shouldRefocus && graph) {
        if (filteredNodes.length > 0) {
            let cx = 0, cy = 0, cz = 0, count = 0;
            filteredNodes.forEach(n => {
                if (typeof n.x === 'number' && typeof n.y === 'number' && typeof n.z === 'number') {
                    cx += n.x; cy += n.y; cz += n.z;
                    count++;
                }
            });

            if (count > 0) {
                cx /= count; cy /= count; cz /= count;
                const dist = Math.max(220, Math.min(650, count * 35));
                graph.cameraPosition(
                    { x: cx, y: cy + 30, z: cz + dist },
                    { x: cx, y: cy, z: cz },
                    800
                );
            }

            if (activeFilter === 'high_risk') {
                const txCount = filteredNodes.filter(n => n.type === 'transaction').length;
                const entCount = filteredNodes.length - txCount;
                showToast(`🚨 High Risk Filter: ${txCount} Fraud Alerts & ${entCount} Linked Entities`);
            } else if (activeFilter === 'rings') {
                showToast(`🕸️ Fraud Rings: Showing ${filteredNodes.length} syndicate entities`);
            }
        } else {
            if (activeFilter === 'high_risk') {
                showToast('⚠️ No high-risk alerts in stream yet. Waiting for incoming fraud events...');
            } else if (activeFilter === 'rings') {
                showToast('⚠️ No shared fraud rings detected yet.');
            }
        }
    }
}

function updateHUDStats(filteredNodesCount = nodesMap.size, filteredLinksCount = linksMap.size) {
    if (activeFilter === 'all' && !searchQuery) {
        statNodeCount.textContent = nodesMap.size;
        statLinkCount.textContent = linksMap.size;
    } else {
        statNodeCount.textContent = `${filteredNodesCount} / ${nodesMap.size}`;
        statLinkCount.textContent = `${filteredLinksCount} / ${linksMap.size}`;
    }

    let highRiskCount = 0;
    nodesMap.forEach(n => {
        if (n.type === 'transaction' && ((n.risk_score || 0) >= 0.75 || n.isAlert || n.decision === 'escalate' || (n.rawTxn && n.rawTxn.is_fraud === 1))) {
            highRiskCount++;
        }
    });
    statHighRiskCount.textContent = highRiskCount;
}

/**
 * Pre-load Recent Transactions from REST API
 */
// Client-Side Simulation State
let simulatorTimer = null;
let isSimulating = false;
const MOCK_CARDS = [
    "1352-450-150", "4821-220-150", "9912-880-180", "3140-500-150", "7741-110-150",
    "5512-320-150", "6621-140-150", "8821-410-180", "2240-600-150", "9141-210-150",
    "4412-180-150", "3321-710-150", "1121-510-180", "6640-400-150", "8141-310-150"
];
const MOCK_DEVICES = [
    "iPhone 15 Pro", "Samsung Galaxy S24", "MacOS Safari 17.2", "Windows 11 Chrome",
    "Android Pixel 8", "Dell XPS Windows", "iPad Air iOS", "OnePlus 12"
];
const MOCK_EMAILS = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "proton.me"];
const MOCK_ADDRS = [315, 299, 181, 204, 126, 441, 512, 143];

// Authentic Fraud Rings for topological demonstration
const FRAUD_RINGS = [
    { 
        name: "Tor Syndicate Alpha", 
        cards: ["9912-880-180", "8821-410-180", "1121-510-180"],
        device: "Tor Exit Node / Linux",
        email: "temp-mail.org",
        addr: 441
    },
    { 
        name: "Botnet Cluster Beta", 
        cards: ["3140-500-150", "7741-110-150", "6640-400-150"],
        device: "Emulated Android / Proxy",
        email: "guerrillamail.com",
        addr: 512
    }
];

let simTxnId = 20000;

function generateClientMockTransaction(forceAlert = null) {
    simTxnId++;
    const isAlert = forceAlert !== null ? forceAlert : Math.random() < 0.22;
    
    let card, dev, email, addr, amt, risk;

    if (isAlert) {
        // Pick one of the two fraud rings
        const ring = FRAUD_RINGS[Math.floor(Math.random() * FRAUD_RINGS.length)];
        card = ring.cards[Math.floor(Math.random() * ring.cards.length)];
        dev = ring.device;
        email = ring.email;
        addr = ring.addr;
        amt = (Math.random() * 2200 + 450).toFixed(2);
        risk = (Math.random() * 0.16 + 0.82).toFixed(4);
    } else {
        // Legitimate transaction
        card = MOCK_CARDS[Math.floor(Math.random() * MOCK_CARDS.length)];
        // 35% of legitimate e-commerce txns have no device fingerprint (guest checkout / direct payment)
        dev = Math.random() < 0.35 ? null : MOCK_DEVICES[Math.floor(Math.random() * MOCK_DEVICES.length)];
        email = MOCK_EMAILS[Math.floor(Math.random() * MOCK_EMAILS.length)];
        addr = MOCK_ADDRS[Math.floor(Math.random() * MOCK_ADDRS.length)];
        amt = (Math.random() * 180 + 12).toFixed(2);
        risk = (Math.random() * 0.25 + 0.04).toFixed(4);
    }

    const cardParts = card.split('-');

    return {
        transaction: {
            transaction_id: simTxnId,
            transaction_dt: Date.now(),
            transaction_amt: parseFloat(amt),
            is_fraud: isAlert ? 1 : 0,
            card1: parseInt(cardParts[0]),
            card2: parseInt(cardParts[1]),
            card3: parseInt(cardParts[2]),
            addr1: addr,
            device_info: dev,
            p_emaildomain: email
        },
        decision: {
            transaction_id: simTxnId,
            risk_score: parseFloat(risk),
            timestamp: Date.now(),
            decision: isAlert ? 'escalate' : 'approve',
            requires_genai: isAlert,
            status: isAlert ? 'pending_investigation' : 'approved'
        }
    };
}

function toggleBrowserSimulator() {
    const btn = document.getElementById('btnDemoFeed');
    if (isSimulating) {
        clearInterval(simulatorTimer);
        isSimulating = false;
        if (btn) {
            btn.classList.remove('active');
            btn.textContent = '🎮 Live Sim';
        }
        showToast('Live Simulation Paused');
    } else {
        isSimulating = true;
        if (btn) {
            btn.classList.add('active');
            btn.textContent = '⏸ Pause Sim';
        }
        simulatorTimer = setInterval(() => {
            const mock = generateClientMockTransaction();
            handleIncomingTransaction(mock);
            recentEvents++;
        }, 1400);
        showToast('▶ Live Simulation Streaming 3D Nodes');
    }
}

/**
 * Pre-load Recent Transactions from REST API or fallback to mock constellation
 */
async function loadInitialTransactions() {
    try {
        const res = await fetch(`${API_BASE_URL}/transactions?limit=50`);
        if (res.ok) {
            const list = await res.json();
            if (Array.isArray(list) && list.length > 0) {
                const batch = list.map(item => {
                    if (item && item.transaction) return item;
                    return {
                        transaction: item,
                        decision: {
                            risk_score: item.is_fraud ? 0.92 : 0.15,
                            decision: item.is_fraud ? 'escalate' : 'approve',
                            status: item.is_fraud ? 'pending_investigation' : 'approved'
                        }
                    };
                });
                handleBatchTransactions(batch);
                showToast(`✓ Loaded ${list.length} initial network entities`);
                return;
            }
        }
    } catch (e) {
        console.warn('Backend API not reachable, loading starter mock constellation:', e);
    }

    // Fallback: Populate 24 starter nodes in one atomic batch
    const starterMocks = [];
    for (let i = 0; i < 24; i++) {
        starterMocks.push(generateClientMockTransaction(i % 5 === 0));
    }
    handleBatchTransactions(starterMocks);
    showToast('⭐ Loaded 24 starter 3D constellation nodes');
    // Start continuous simulator
    toggleBrowserSimulator();
}

/**
 * Hover & Click Handlers with Debounced Neighborhood Highlighting
 */
function handleNodeHover(node) {
    if (isUserDragging) {
        if (hoveredNode) {
            hoveredNode = null;
            refreshHighlight();
        }
        if (hoverInspector) hoverInspector.classList.remove('visible');
        return;
    }

    clearTimeout(hoverDebounceTimer);
    hoverDebounceTimer = setTimeout(() => {
        if (isUserDragging) return;

        hoveredNode = node;
        container.style.cursor = node ? 'pointer' : 'default';

        refreshHighlight();

        if (!node) {
            hoverInspector.classList.remove('visible');
            return;
        }

        // Populate Hover Card HUD
        const isAlert = (node.risk_score || 0) >= 0.8;
        const isMedium = (node.risk_score || 0) >= 0.5 && !isAlert;
        const isGeminiTagged = node.requires_genai || isAlert || (Number(node.risk_score || 0) >= 0.75);

        document.getElementById('hoverTitle').textContent = node.label || node.id;

        const iconEl = document.getElementById('hoverIcon');
        if (node.type === 'transaction') iconEl.textContent = isAlert ? '🚨' : '👤';
        else if (node.type === 'card') iconEl.textContent = '💳';
        else if (node.type === 'device') iconEl.textContent = '📱';
        else iconEl.textContent = '📍';

        const typeTag = document.getElementById('hoverTypeTag');
        if (node.type === 'transaction' && isGeminiTagged) {
            typeTag.className = `type-tag transaction`;
            typeTag.innerHTML = `<span>Txn</span> <span style="background:rgba(168,85,247,0.3); border:1px solid #c084fc; color:#f3e8ff; padding:1px 6px; border-radius:4px; font-size:10px; font-weight:700; margin-left:4px;">🤖 GEMINI AI</span>`;
        } else {
            typeTag.className = `type-tag ${node.type}`;
            typeTag.textContent = node.type;
        }

        const riskPill = document.getElementById('hoverRiskValue');
        const score = Number(node.risk_score || 0).toFixed(2);
        riskPill.textContent = score;
        riskPill.className = `risk-pill ${isAlert ? 'high' : (isMedium ? 'medium' : 'low')}`;

        const amountRow = document.getElementById('hoverAmountRow');
        if (node.amount !== undefined) {
            amountRow.style.display = 'flex';
            document.getElementById('hoverAmountValue').textContent = `$${Number(node.amount).toFixed(2)}`;
        } else {
            amountRow.style.display = 'none';
        }

        const d1Label = document.getElementById('hoverDetail1Label');
        const d1Val = document.getElementById('hoverDetail1Value');
        const d2Label = document.getElementById('hoverDetail2Label');
        const d2Val = document.getElementById('hoverDetail2Value');

        if (node.type === 'transaction') {
            d1Label.textContent = 'Status';
            d1Val.textContent = (node.status || 'approved').toUpperCase();
            d2Label.textContent = 'AI Copilot';
            d2Val.innerHTML = isGeminiTagged 
                ? '<span style="color:#c084fc; font-weight:700;">🤖 Escalated to GenAI</span>'
                : '<span style="color:#94a3b8;">Tier 1 Auto-Approved</span>';
        } else {
            d1Label.textContent = 'Entity ID';
            d1Val.textContent = node.rawId || '---';
            d2Label.textContent = 'Mesh Degree';
            d2Val.textContent = `${entityDegrees.get(node.id) || 1} connections`;
        }

        const neighborCount = node.neighbors ? node.neighbors.size : 0;
        document.getElementById('hoverNeighborsCount').textContent = `${neighborCount} connected nodes`;

        hoverInspector.classList.add('visible');
    }, 35);
}

/**
 * Highlight connected neighborhood of hovered or selected node:
 * - Direct opacity dimming on node sprites (hovered & neighbors = 1.0, unrelated = 0.15)
 * - Dynamic color & particle boost on connected links
 */
function refreshHighlight() {
    if (!graph) return;
    const active = selectedNode || hoveredNode;

    const focusedNodeIds = new Set();
    if (active) {
        focusedNodeIds.add(active.id);
        if (active.neighbors) {
            active.neighbors.forEach(nid => focusedNodeIds.add(nid));
        }
    }

    // Direct Three.js material mutation (<0.05ms)
    nodesMap.forEach(n => {
        if (n.__sprite && n.__sprite.material) {
            if (!active) {
                n.__sprite.material.opacity = 0.95;
            } else {
                n.__sprite.material.opacity = focusedNodeIds.has(n.id) ? 1.0 : 0.15;
            }
        }
    });

    graph.linkColor(link => {
        if (!active) {
            return link.isFraudLink ? 'rgba(239, 68, 68, 0.75)' : 'rgba(56, 189, 248, 0.40)';
        }
        const sId = typeof link.source === 'object' ? link.source.id : link.source;
        const tId = typeof link.target === 'object' ? link.target.id : link.target;
        const isConnected = sId === active.id || tId === active.id;
        if (isConnected) {
            return link.isFraudLink ? '#ef4444' : '#ffffff';
        }
        return 'rgba(148, 163, 184, 0.05)';
    });

    graph.linkWidth(link => {
        if (!active) {
            return link.isFraudLink ? 1.6 : 0.8;
        }
        const sId = typeof link.source === 'object' ? link.source.id : link.source;
        const tId = typeof link.target === 'object' ? link.target.id : link.target;
        const isConnected = sId === active.id || tId === active.id;
        return isConnected ? (link.isFraudLink ? 3.2 : 2.5) : 0.4;
    });

    graph.linkDirectionalParticles(link => {
        if (!active) {
            return link.isFraudLink ? 2 : 0;
        }
        const sId = typeof link.source === 'object' ? link.source.id : link.source;
        const tId = typeof link.target === 'object' ? link.target.id : link.target;
        const isConnected = sId === active.id || tId === active.id;
        return isConnected ? (link.isFraudLink ? 4 : 2) : 0;
    });
}

function handleNodeClick(node) {
    if (!node) return;
    const target = node.__data || (node.userData && node.userData.node) || node;
    selectedNode = target;
    refreshHighlight();

    // Smoothly fly camera to face the inspected node
    if (graph && target.x !== undefined && target.y !== undefined && target.z !== undefined) {
        const distance = 160;
        const distRatio = 1 + distance / Math.hypot(target.x, target.y, target.z);

        graph.cameraPosition(
            { x: target.x * distRatio, y: target.y * distRatio, z: target.z * distRatio },
            { x: target.x, y: target.y, z: target.z },
            1000
        );
    }

    openDetailDrawer(target);
}

/**
 * Slide-Over Investigation Drawer
 */
function openDetailDrawer(node) {
    if (!node) return;
    const target = node.__data || (node.userData && node.userData.node) || node;
    const drawer = detailDrawer || document.getElementById('detailDrawer');
    if (!drawer) return;

    const isAlert = (target.risk_score || 0) >= 0.8;
    const isGeminiTagged = target.requires_genai || isAlert || (Number(target.risk_score || 0) >= 0.75);

    const titleEl = drawerTitle || document.getElementById('drawerTitle');
    if (titleEl) titleEl.textContent = target.label || target.id;

    const badgeEl = drawerBadge || document.getElementById('drawerBadge');
    if (badgeEl) {
        badgeEl.textContent = (target.type || 'entity').toUpperCase();
        badgeEl.className = `type-tag ${target.type || 'transaction'}`;
    }

    let neighborsHtml = '';
    if (target.neighbors && target.neighbors.size > 0) {
        target.neighbors.forEach(nid => {
            const nb = nodesMap.get(nid);
            if (nb) {
                neighborsHtml += `
                    <div class="neighbor-item" onclick="focusNodeById('${nb.id}')">
                        <span class="neighbor-dot ${nb.type}"></span>
                        <span class="neighbor-name">${nb.label || nb.id}</span>
                        <span class="neighbor-score" style="color:${nb.risk_score >= 0.8 ? '#ef4444' : '#94a3b8'};">
                            ${Number(nb.risk_score || 0).toFixed(2)}
                        </span>
                    </div>
                `;
            }
        });
    } else {
        neighborsHtml = '<p style="color:var(--text-muted);font-size:12px;">No directly linked entities in active graph.</p>';
    }

    let aiSection = '';
    if (target.type === 'transaction') {
        aiSection = `
            <div class="drawer-section">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                    <h3 style="margin:0;">🤖 Gemini GenAI SAR Narrative</h3>
                    <span style="background:rgba(168,85,247,0.3); border:1px solid #c084fc; color:#f3e8ff; padding:2px 8px; border-radius:12px; font-size:10px; font-weight:700;">
                        ${isGeminiTagged ? 'TIER 2 TAGGED' : 'MANUAL QUERY'}
                    </span>
                </div>
                <div id="drawerGeminiBox">
                    <button class="hud-btn active" style="width:100%; justify-content:center; padding:10px;" onclick="requestDrawerGeminiAnalysis(${target.rawId})">
                        ✨ Generate AI Risk Reasoning & SAR Draft
                    </button>
                </div>
            </div>
            <div class="drawer-section">
                <h3>⚖️ Investigator Verdict</h3>
                <div style="display:flex; gap:10px;">
                    <button class="hud-btn danger-btn active" style="flex:1; justify-content:center;" onclick="submitDrawerVerdict(${target.rawId}, true)">✓ Confirm Fraud</button>
                    <button class="hud-btn" style="flex:1; justify-content:center;" onclick="submitDrawerVerdict(${target.rawId}, false)">✗ False Positive</button>
                </div>
            </div>
        `;
    }

    const contentEl = drawerContent || document.getElementById('drawerContent');
    if (contentEl) {
        contentEl.innerHTML = `
            <div class="drawer-section">
                <h3>📋 Core Properties</h3>
                <div class="drawer-grid">
                    <div class="drawer-field">
                        <span class="drawer-field-label">Entity Type</span>
                        <span class="drawer-field-value">${target.type}</span>
                    </div>
                    <div class="drawer-field">
                        <span class="drawer-field-label">Risk Score</span>
                        <span class="drawer-field-value" style="color:${isAlert ? '#ef4444' : '#38bdf8'}; font-weight:700;">
                            ${Number(target.risk_score || 0).toFixed(4)}
                        </span>
                    </div>
                    ${target.amount !== undefined ? `
                    <div class="drawer-field">
                        <span class="drawer-field-label">Amount</span>
                        <span class="drawer-field-value">$${Number(target.amount).toFixed(2)}</span>
                    </div>` : ''}
                    <div class="drawer-field">
                        <span class="drawer-field-label">Tier 2 Escalation</span>
                        <span class="drawer-field-value" style="color:${isGeminiTagged ? '#c084fc' : '#94a3b8'}; font-weight:700;">
                            ${isGeminiTagged ? '🤖 Gemini AI Tagged' : 'Auto-Approved (<0.75)'}
                        </span>
                    </div>
                    <div class="drawer-field">
                        <span class="drawer-field-label">Identifier</span>
                        <span class="drawer-field-value">${target.rawId || target.id}</span>
                    </div>
                </div>
            </div>

            <div class="drawer-section">
                <h3>🕸️ Connected Topology (${target.neighbors ? target.neighbors.size : 0})</h3>
                <div class="neighbor-list">
                    ${neighborsHtml}
                </div>
            </div>

            ${aiSection}
        `;
    }

    drawer.classList.add('open');

    // Automatically trigger Gemini SAR generation for high-risk alerts!
    if (target.type === 'transaction' && isGeminiTagged && target.rawId) {
        requestDrawerGeminiAnalysis(target.rawId);
    }
}

window.focusNodeById = function (nodeId) {
    const node = nodesMap.get(nodeId);
    if (node && graph) {
        handleNodeClick(node);
    }
};

window.requestDrawerGeminiAnalysis = async function (txnId) {
    const box = document.getElementById('drawerGeminiBox');
    if (!box) return;
    box.innerHTML = '<div style="color:#38bdf8; font-size:12px; padding:10px; display:flex; align-items:center; gap:8px;"><span>⏳</span> 🤖 Querying Gemini 3.5 for plain-language SAR reasoning...</div>';

    try {
        const node = nodesMap.get(`txn_${txnId}`) || selectedNode;
        const payload = {
            transaction: node ? (node.rawTxn || {
                transaction_id: parseInt(txnId),
                transaction_amt: parseFloat(node.amount || 1150.0),
                device_info: node.device || 'Tor Browser / Linux',
                is_fraud: 1
            }) : null,
            decision: node ? (node.rawDecision || {
                transaction_id: parseInt(txnId),
                risk_score: parseFloat(node.risk_score || 0.88),
                decision: 'escalate',
                status: 'pending_investigation'
            }) : null
        };

        const res = await fetch(`${API_BASE_URL}/transactions/${txnId}/request-gemini-analysis`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            const data = await res.json();
            box.innerHTML = `
                <div class="ai-sar-box" style="margin-top:10px;">
                    <h4 style="color:#a855f7; margin-bottom:6px; font-weight:700;">🧠 AI Risk Reasoning</h4>
                    <div style="white-space:pre-wrap; line-height:1.5; font-size:12px; color:#e2e8f0; max-height:220px; overflow-y:auto; padding:8px; background:rgba(0,0,0,0.25); border-radius:6px;">${data.explanation || 'No reasoning available.'}</div>
                    <hr style="border:none; border-top:1px solid rgba(255,255,255,0.12); margin:12px 0;">
                    <h4 style="color:#38bdf8; margin-bottom:6px; font-weight:700;">📋 Draft SAR Regulatory Narrative</h4>
                    <pre style="font-family:var(--font-mono); font-size:11px; white-space:pre-wrap; background:rgba(0,0,0,0.4); padding:10px; border-radius:6px; border:1px solid rgba(56,189,248,0.2); max-height:200px; overflow-y:auto;">${data.sar_draft || 'SAR draft generated.'}</pre>
                </div>
            `;
            showToast('✓ Gemini SAR narrative generated successfully');
        } else {
            const err = await res.json().catch(() => ({}));
            box.innerHTML = `<p style="color:#ef4444; font-size:12px;">Failed: ${err.detail || 'Could not generate report'}</p>`;
        }
    } catch (e) {
        box.innerHTML = `<p style="color:#ef4444; font-size:12px;">Connection error: ${e.message}</p>`;
    }
};

window.submitDrawerVerdict = async function (txnId, confirmed) {
    const verdict = confirmed ? 'Confirmed Fraud' : 'False Positive';
    const notes = prompt(`Investigator resolution notes for Txn #${txnId} (${verdict}):`);
    if (notes === null) return;

    try {
        const res = await fetch(`${API_BASE_URL}/alerts/${txnId}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confirmed, notes })
        });
        if (res.ok) {
            showToast(`✓ Marked Txn #${txnId} as ${verdict}`);
            detailDrawer.classList.remove('open');
            selectedNode = null;
        }
    } catch (e) {
        alert('Failed to submit verdict.');
    }
};

/**
 * Event Listeners & HUD Controls
 */
function setupEventListeners() {
    closeDrawerBtn.addEventListener('click', () => {
        detailDrawer.classList.remove('open');
        selectedNode = null;
        hoveredNode = null;
        refreshHighlight();
    });

    btnAutoRotate.addEventListener('click', () => {
        autoRotate = !autoRotate;
        syncControls();
        btnAutoRotate.classList.toggle('active', autoRotate);
        btnAutoRotate.textContent = autoRotate ? '🔄 Auto-Orbit: ON' : '⏸ Auto-Orbit: OFF';
        showToast(autoRotate ? '✓ Auto-Orbit: ON (Rotating)' : '⏸ Auto-Orbit: OFF (Stopped)');
    });

    const btnAntiSpin = document.getElementById('btnAntiSpin');
    if (btnAntiSpin) {
        btnAntiSpin.addEventListener('click', () => {
            antiSpinEnabled = !antiSpinEnabled;
            btnAntiSpin.classList.toggle('active', antiSpinEnabled);
            btnAntiSpin.textContent = antiSpinEnabled ? '🛑 Anti-Spin: ON' : '🔄 Anti-Spin: OFF';
            showToast(antiSpinEnabled ? '🛑 Anti-Spin: ON (Rotational torque cancelled)' : '🔄 Anti-Spin: OFF (Free physics rotation)');
            if (graph && antiSpinEnabled) {
                nodesMap.forEach(n => {
                    n.vx = 0; n.vy = 0; n.vz = 0;
                });
            }
        });
    }

    const btnLockLayout = document.getElementById('btnLockLayout');
    if (btnLockLayout) {
        btnLockLayout.addEventListener('click', () => {
            isLayoutLocked = !isLayoutLocked;
            btnLockLayout.classList.toggle('active', isLayoutLocked);
            btnLockLayout.textContent = isLayoutLocked ? '📌 Layout: LOCKED' : '📌 Lock Layout';
            nodesMap.forEach(n => {
                if (isLayoutLocked) {
                    if (typeof n.x === 'number') {
                        n.fx = n.x; n.fy = n.y; n.fz = n.z;
                    }
                } else {
                    n.fx = undefined; n.fy = undefined; n.fz = undefined;
                }
            });
            showToast(isLayoutLocked ? '📌 Layout Locked: Node positions fixed in space' : '🔓 Layout Unlocked: Dynamic physics active');
        });
    }

    if (btnParticles) {
        btnParticles.addEventListener('click', () => {
            showLinkParticles = !showLinkParticles;
            btnParticles.classList.toggle('active', showLinkParticles);
            if (graph) {
                graph.linkDirectionalParticles(link => showLinkParticles ? (link.isHighRisk ? 4 : 2) : 0);
            }
        });
    }

    if (btnResetCamera) {
        btnResetCamera.addEventListener('click', () => {
            selectedNode = null;
            hoveredNode = null;
            refreshHighlight();
            if (graph) {
                graph.cameraPosition({ x: 0, y: 40, z: 720 }, { x: 0, y: 0, z: 0 }, 1000);
            }
        });
    }

    if (sliderSpread && labelSpreadVal) {
        sliderSpread.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value) / 100;
            spacingMultiplier = val;
            labelSpreadVal.textContent = `${val.toFixed(1)}x`;
            updateGraphForces();
            if (graph) {
                graph.d3ReheatSimulation();
            }
        });
    }

    if (btnClearGraph) {
        btnClearGraph.addEventListener('click', () => {
            selectedNode = null;
            hoveredNode = null;
            nodesMap.clear();
            linksMap.clear();
            entityDegrees.clear();
            syncGraphData();
            refreshHighlight();
            showToast('Graph cleared');
        });
    }

    // Filters
    if (btnFilterAll) {
        btnFilterAll.addEventListener('click', () => {
            setActiveFilter('all', btnFilterAll);
        });
    }
    if (btnFilterHighRisk) {
        btnFilterHighRisk.addEventListener('click', () => {
            setActiveFilter('high_risk', btnFilterHighRisk);
        });
    }
    if (btnFilterRings) {
        btnFilterRings.addEventListener('click', () => {
            setActiveFilter('rings', btnFilterRings);
        });
    }

    // Search
    if (nodeSearch) {
        nodeSearch.addEventListener('input', (e) => {
            searchQuery = e.target.value.trim();
            syncGraphData();

            // If exact match found, fly camera to it
            if (searchQuery.length > 2) {
                for (const [id, node] of nodesMap.entries()) {
                    if (id.toLowerCase().includes(searchQuery.toLowerCase()) ||
                        node.label.toLowerCase().includes(searchQuery.toLowerCase())) {
                        handleNodeClick(node);
                        break;
                    }
                }
            }
        });
    }

    const btnDemoFeed = document.getElementById('btnDemoFeed');
    if (btnDemoFeed) {
        btnDemoFeed.addEventListener('click', toggleBrowserSimulator);
    }

    if (btnInjectBatch) {
        btnInjectBatch.addEventListener('click', inject20NodesBatch);
    }

    // Keyboard Shortcuts for Instant Zoom (+/-)
    window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;
        if (e.key === '+' || e.key === '=') {
            const cam = graph ? graph.camera() : null;
            if (cam) {
                graph.cameraPosition({
                    x: cam.position.x * 0.55,
                    y: cam.position.y * 0.55,
                    z: cam.position.z * 0.55
                }, null, 200);
            }
        } else if (e.key === '-' || e.key === '_') {
            const cam = graph ? graph.camera() : null;
            if (cam) {
                graph.cameraPosition({
                    x: cam.position.x * 1.8,
                    y: cam.position.y * 1.8,
                    z: cam.position.z * 1.8
                }, null, 200);
            }
        }
    });
}

function setActiveFilter(filterName, btnElement) {
    hoveredNode = null;
    selectedNode = null;
    refreshHighlight();

    if (activeFilter === filterName && filterName !== 'all') {
        // Toggle off back to all
        activeFilter = 'all';
        [btnFilterAll, btnFilterHighRisk, btnFilterRings].forEach(btn => btn.classList.remove('active'));
        btnFilterAll.classList.add('active');
        syncGraphData(true);
        showToast(`🌐 Filter Cleared: Showing all ${nodesMap.size} entities`);
        return;
    }

    activeFilter = filterName;
    [btnFilterAll, btnFilterHighRisk, btnFilterRings].forEach(btn => btn.classList.remove('active'));
    btnElement.classList.add('active');
    syncGraphData(true);
}

/**
 * Stream Ingestion Velocity Monitor
 */
function startVelocityMonitor() {
    setInterval(() => {
        const now = Date.now();
        const elapsedSec = (now - lastVelocityTick) / 1000;
        const rate = (recentEvents / elapsedSec).toFixed(1);
        statIngestRate.textContent = `${rate} /s`;
        recentEvents = 0;
        lastVelocityTick = now;
    }, 2000);
}

function showToast(message) {
    const existing = document.querySelector('.network-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'network-toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3200);
}
