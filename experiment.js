// ============================================================================
// EXPERIMENT ENTRY POINT
// ============================================================================
// Standalone page for experimental widgets. Imports only what it needs from
// the shared modules — no dependency on app.js or tutorial.js.

import { Simulation } from './simulation.js';
import { LossChart, RightChart, GradProjectionChart, LineChart } from './visualization.js';
import { CHEBYSHEV_DEFAULTS, describeDefaults } from './defaults.js';

// ============================================================================
// EIGENVECTOR WIDGET
// ============================================================================

function initEigenvectorWidget() {
  const startBtn = document.getElementById('eigenvector-start');
  const resetBtn = document.getElementById('eigenvector-reset');
  if (!startBtn) return null;

  const infoEl = document.getElementById('eigenvector-info');
  if (infoEl) infoEl.textContent = `${describeDefaults(CHEBYSHEV_DEFAULTS)}, Chebyshev degree ${CHEBYSHEV_DEFAULTS.taskParams.degree}, ${CHEBYSHEV_DEFAULTS.taskParams.nTrain} points — eigenvector locked at 5% of 2/η`;

  const sim = new Simulation({
    kEigs: 1,
    stepsPerSecId: null,
    sharpnessProximityThreshold: 0.05
  });

  const lossChart      = new LossChart('eigenvector-loss', { showTest: false, showEma: false });
  const sharpnessChart = new RightChart('eigenvector-sharpness', { kEigs: 1 });
  const projChart      = new GradProjectionChart('eigenvector-projection');

  sim.onFrameUpdate = () => {
    const state = sim.getState();
    lossChart.update(state.lossHistory, state.testLossHistory, state.eta);
    sharpnessChart.update(state.eigenvalueHistory, state.eta);
    projChart.update(state.gradProjectionHistory, state.eta);

    const captureEl = document.getElementById('eigenvector-capture-info');
    if (captureEl && state.eigenvectorCaptureStep !== null && captureEl.dataset.set !== '1') {
      captureEl.textContent = `Eigenvector captured at step ${state.eigenvectorCaptureStep} (λ_max = ${state.eigenvectorCaptureValue.toFixed(3)})`;
      captureEl.style.display = 'block';
      captureEl.dataset.set = '1';
    }
  };

  sim.onDiverge = () => {
    startBtn.textContent = 'diverged';
    startBtn.disabled = true;
  };

  startBtn.addEventListener('click', () => {
    if (!sim.isRunning) {
      if (!sim.model) {
        const d = CHEBYSHEV_DEFAULTS;
        sim.captureParams(d.taskKey, d.taskParams, d.activation, d.hiddenDims, d.eta, d.batchSize, d.modelSeed);
      }
      sim.start();
      startBtn.textContent = 'pause';
    } else {
      sim.pause();
      startBtn.textContent = 'resume';
    }
  });

  resetBtn.addEventListener('click', () => {
    sim.reset();
    lossChart.clear();
    sharpnessChart.clear();
    projChart.clear();
    startBtn.textContent = 'start';
    startBtn.disabled = false;
    const captureEl = document.getElementById('eigenvector-capture-info');
    if (captureEl) {
      captureEl.textContent = '';
      captureEl.style.display = 'none';
      captureEl.dataset.set = '';
    }
  });

  return sim;
}

// ============================================================================
// DEEP SCALAR NETWORK — loss landscape + gradient descent widget
// ============================================================================
// Models the scalar network L(x, y) = (1 - x*y)^2 — the simplest two-layer
// deep network with scalar weights. The minimum manifold is the hyperbola xy=1,
// making it a clean toy model for studying loss landscape geometry and EoS.
//
// All gradient and Hessian calculations use closed-form expressions.
// No MLP / Trainer / Lanczos machinery is used here.
//
// Closed-form expressions:
//   L(x, y)   = (1 - xy)²
//   ∇L        = [-2y(1-xy),  -2x(1-xy)]
//   H         = [[ 2y²,        2(1-2xy) ],
//                [ 2(1-2xy),   2x²      ]]
//   λ_max(H)  = (y²+x²) + sqrt((y²-x²)² + 4(1-2xy)²)   [closed form, 2×2 sym]
// ============================================================================

// Viewport bounds (matching the Python xlim/ylim)
const DSN_X_MIN = -2, DSN_X_MAX = 6;
const DSN_Y_MIN = -2, DSN_Y_MAX = 6;
const DSN_MAX_ITER = 200;
const DSN_DIVERGE_THRESHOLD = 1e6;

// --- Closed-form math ---

function dsnLoss(x, y) {
  const p = 1 - x * y;
  return p * p;
}

function dsnGradient(x, y) {
  const p = 1 - x * y;
  return [-2 * y * p, -2 * x * p];
}

function dsnMaxEigenvalue(x, y) {
  const a    = 2 * y * y;
  const d    = 2 * x * x;
  const b    = 2 * (1 - 2 * x * y);
  const mid  = (a + d) / 2;
  const half = (a - d) / 2;
  return mid + Math.sqrt(half * half + b * b);
}

// Run gradient descent to completion. Returns trajectory + companion histories.
function dsnRunGD(x0, y0, eta) {
  const trajectory = [{ x: x0, y: y0 }];
  const lossHist   = [{ iteration: 0, value: dsnLoss(x0, y0) }];
  const eigHist    = [{ iteration: 0, value: dsnMaxEigenvalue(x0, y0) }];
  const prodHist   = [{ iteration: 0, value: x0 * y0 }];

  let x = x0, y = y0;

  for (let t = 1; t <= DSN_MAX_ITER; t++) {
    const [gx, gy] = dsnGradient(x, y);
    x -= eta * gx;
    y -= eta * gy;

    const loss = dsnLoss(x, y);
    trajectory.push({ x, y });
    lossHist.push({ iteration: t, value: loss });
    eigHist.push({ iteration: t, value: dsnMaxEigenvalue(x, y) });
    prodHist.push({ iteration: t, value: x * y });

    if (!isFinite(loss) || loss > DSN_DIVERGE_THRESHOLD) break;
    if (loss < 0.001) break; // converged
  }

  return { trajectory, lossHist, eigHist, prodHist };
}

// --- Landscape canvas (precomputed pixel grid, drawn once) ---

// Red-blue diverging colormap centered at log10(L) = 0 (i.e. L = 1).
// Blue = low loss (near the xy=1 manifold), Red = high loss.
// Matches the Python matplotlib colormap.
function logLossToColor(logL) {
  // Compress the range: /4 maps ±4 log units to the full ±1 color range.
  const t = Math.max(-1, Math.min(1, logL / 4));
  if (t < 0) {
    // White -> blue  (low loss)
    const s = -t;
    return [Math.round(255 * (1 - s)), Math.round(255 * (1 - s)), 255];
  } else {
    // White -> red  (high loss)
    return [255, Math.round(255 * (1 - t)), Math.round(255 * (1 - t))];
  }
}

function dsnPrecomputeLandscape(canvas) {
  const W   = canvas.width;
  const H   = canvas.height;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(W, H);
  const d   = img.data;

  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      // py=0 is top of canvas = y=DSN_Y_MAX
      const wx = DSN_X_MIN + (px / (W - 1)) * (DSN_X_MAX - DSN_X_MIN);
      const wy = DSN_Y_MAX - (py / (H - 1)) * (DSN_Y_MAX - DSN_Y_MIN);
      const [r, g, b] = logLossToColor(Math.log10(dsnLoss(wx, wy) + 1e-10));
      const i = (py * W + px) * 4;
      d[i] = r; d[i+1] = g; d[i+2] = b; d[i+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return img; // caller can reuse to restore background before redrawing overlay
}

// --- Coordinate helpers ---

function dsnWorldToPixel(wx, wy, W, H) {
  return [
    ((wx - DSN_X_MIN) / (DSN_X_MAX - DSN_X_MIN)) * (W - 1),
    ((DSN_Y_MAX - wy) / (DSN_Y_MAX - DSN_Y_MIN)) * (H - 1)
  ];
}

function dsnPixelToWorld(px, py, W, H) {
  return [
    DSN_X_MIN + (px / (W - 1)) * (DSN_X_MAX - DSN_X_MIN),
    DSN_Y_MAX - (py / (H - 1)) * (DSN_Y_MAX - DSN_Y_MIN)
  ];
}

// --- Overlay drawing ---

function dsnDrawOverlay(overlayCanvas, trajectory, startX, startY) {
  const W   = overlayCanvas.width;
  const H   = overlayCanvas.height;
  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  // Draw trajectory path and end point
  if (trajectory && trajectory.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(20, 20, 20, 0.85)';
    ctx.lineWidth = 1.5;
    const [x0, y0] = dsnWorldToPixel(trajectory[0].x, trajectory[0].y, W, H);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < trajectory.length; i++) {
      const [px, py] = dsnWorldToPixel(trajectory[i].x, trajectory[i].y, W, H);
      ctx.lineTo(px, py);
    }
    ctx.stroke();

    // End point marker
    const [ex, ey] = dsnWorldToPixel(
      trajectory[trajectory.length - 1].x,
      trajectory[trajectory.length - 1].y, W, H
    );
    ctx.beginPath();
    ctx.arc(ex, ey, 4, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(20, 20, 20, 0.9)';
    ctx.fill();
  }

  // Start point (draggable) — white circle with dark border
  const [sx, sy] = dsnWorldToPixel(startX, startY, W, H);
  ctx.beginPath();
  ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
  ctx.fillStyle = 'white';
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 2;
  ctx.fill();
  ctx.stroke();
}

// --- Widget initializer ---

function initDeepScalarWidget() {
  const landscapeCanvas = document.getElementById('dsn-landscape');
  const overlayCanvas   = document.getElementById('dsn-overlay');
  const etaSlider       = document.getElementById('dsn-eta-slider');
  const etaDisplay      = document.getElementById('dsn-eta-value');
  if (!landscapeCanvas || !overlayCanvas || !etaSlider) return;

  // Companion charts
  const lossChart = new LineChart('dsn-loss', {
    label: 'loss',
    color: 'rgb(40, 130, 130)'
  });
  const eigChart = new LineChart('dsn-eig', {
    label: 'λ_max',
    color: 'rgb(220, 50, 50)',
    refLabel: '2/η'
  });
  const prodChart = new LineChart('dsn-prod', {
    label: 'xy',
    color: 'rgb(130, 60, 200)'
  });

  // Precompute landscape (static)
  dsnPrecomputeLandscape(landscapeCanvas);

  // State
  let startX = 3.0, startY = 1.0;
  let currentResult = null;

  function runAndDraw() {
    const eta = parseFloat(etaSlider.value);
    etaDisplay.textContent = eta.toFixed(3);

    currentResult = dsnRunGD(startX, startY, eta);
    const { trajectory, lossHist, eigHist, prodHist } = currentResult;

    dsnDrawOverlay(overlayCanvas, trajectory, startX, startY);
    lossChart.update(lossHist);
    eigChart.update(eigHist, 2 / eta);  // 2/η ref line updates with slider
    prodChart.update(prodHist);
  }

  runAndDraw();

  // Eta slider — recompute and redraw immediately
  etaSlider.addEventListener('input', runAndDraw);

  // --- Drag handling ---

  let isDragging = false;

  function getCanvasPos(e) {
    const rect   = overlayCanvas.getBoundingClientRect();
    const scaleX = overlayCanvas.width  / rect.width;
    const scaleY = overlayCanvas.height / rect.height;
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return [(cx - rect.left) * scaleX, (cy - rect.top) * scaleY];
  }

  function isNearStart(px, py) {
    const [sx, sy] = dsnWorldToPixel(startX, startY, overlayCanvas.width, overlayCanvas.height);
    const dx = px - sx, dy = py - sy;
    return Math.sqrt(dx * dx + dy * dy) < 14;
  }

  function applyDragPos(px, py) {
    const cpx = Math.max(0, Math.min(overlayCanvas.width  - 1, px));
    const cpy = Math.max(0, Math.min(overlayCanvas.height - 1, py));
    [startX, startY] = dsnPixelToWorld(cpx, cpy, overlayCanvas.width, overlayCanvas.height);
    runAndDraw(); // rerun GD immediately on every move
  }

  overlayCanvas.addEventListener('mousedown', (e) => {
    const [px, py] = getCanvasPos(e);
    if (isNearStart(px, py)) { isDragging = true; e.preventDefault(); }
  });

  overlayCanvas.addEventListener('mousemove', (e) => {
    const [px, py] = getCanvasPos(e);
    if (isDragging) {
      applyDragPos(px, py);
      e.preventDefault();
    } else {
      overlayCanvas.style.cursor = isNearStart(px, py) ? 'grab' : 'default';
    }
  });

  overlayCanvas.addEventListener('mouseup',    () => { isDragging = false; });
  overlayCanvas.addEventListener('mouseleave', () => { isDragging = false; });

  // Touch events
  overlayCanvas.addEventListener('touchstart', (e) => {
    const [px, py] = getCanvasPos(e);
    if (isNearStart(px, py)) { isDragging = true; e.preventDefault(); }
  }, { passive: false });

  overlayCanvas.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    const [px, py] = getCanvasPos(e);
    applyDragPos(px, py);
    e.preventDefault();
  }, { passive: false });

  overlayCanvas.addEventListener('touchend', () => { isDragging = false; });
}

// ============================================================================
// INIT — wait for MathJax then start
// ============================================================================

function init() {
  initEigenvectorWidget();
  initDeepScalarWidget();
}

function waitForMathJax(attempts = 0) {
  if (window.MathJax && window.MathJax.typesetPromise && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(init).catch(err => console.error('init error:', err));
  } else if (attempts < 50) {
    setTimeout(() => waitForMathJax(attempts + 1), 50);
  } else {
    init();
  }
}

waitForMathJax();
