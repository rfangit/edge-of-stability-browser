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
// DEEP SCALAR NETWORK — shared infrastructure
// ============================================================================
// Both DSN widgets share:
//   - logLossToColor()    — red-blue diverging colormap (viewport-independent)
//   - DeepScalarWidget    — generic class handling landscape, overlay, GD,
//                           drag, charts. Parameterised entirely by a config
//                           object so adding a new variant requires no new
//                           plumbing code.
//
// Each variant supplies only:
//   - Its own math functions (loss, gradient, maxEigenvalue, productFn)
//   - Its viewport bounds (xMin, xMax, yMin, yMax)
//   - Its DOM IDs and initial start point
// ============================================================================

// Red-blue diverging colormap centred at log10(L) = 0 (L = 1).
// Blue = low loss, Red = high loss. Shared by all DSN variants.
function logLossToColor(logL) {
  const t = Math.max(-1, Math.min(1, logL / 4));
  if (t < 0) {
    const s = -t;
    return [Math.round(255 * (1 - s)), Math.round(255 * (1 - s)), 255];
  } else {
    return [255, Math.round(255 * (1 - t)), Math.round(255 * (1 - t))];
  }
}

// ============================================================================
// DeepScalarWidget
// ============================================================================
// Config object fields:
//
//   Viewport (required)
//     xMin, xMax, yMin, yMax   — world-space bounds of the landscape canvas
//
//   Math (required — pure functions, no shared state)
//     loss(x, y)               — scalar loss value
//     gradient(x, y)           — [gx, gy] gradient vector
//     maxEigenvalue(x, y)      — largest Hessian eigenvalue (closed form)
//     productFn(x, y)          — scalar quantity tracked in third chart
//     productLabel             — y-axis label for that chart (e.g. 'xy')
//
//   DOM IDs (required)
//     landscapeId, overlayId   — two stacked canvas elements
//     etaSliderId, etaValueId  — learning rate slider + display span
//     lossChartId              — canvas for loss chart
//     eigChartId               — canvas for eigenvalue chart
//     prodChartId              — canvas for product chart
//
//   Initial state (optional)
//     startX, startY           — initial draggable point (default 1, 1)
//     maxIter                  — GD iteration cap (default 200)
//     divergeThreshold         — stop if loss exceeds this (default 1e6)
//     convergeThreshold        — stop if loss falls below this (default 0.001)

class DeepScalarWidget {
  constructor(config) {
    this.cfg = {
      startX: 1, startY: 1,
      maxIter: 200,
      divergeThreshold: 1e6,
      convergeThreshold: 0.001,
      ...config
    };

    // Grab DOM elements
    this.landscapeCanvas = document.getElementById(this.cfg.landscapeId);
    this.overlayCanvas   = document.getElementById(this.cfg.overlayId);
    this.etaSlider       = document.getElementById(this.cfg.etaSliderId);
    this.etaDisplay      = document.getElementById(this.cfg.etaValueId);
    if (!this.landscapeCanvas || !this.overlayCanvas || !this.etaSlider) return;

    // Charts
    this.lossChart = new LineChart(this.cfg.lossChartId, {
      label: 'loss', color: 'rgb(40, 130, 130)'
    });
    this.eigChart = new LineChart(this.cfg.eigChartId, {
      label: 'λ_max', color: 'rgb(220, 50, 50)', refLabel: '2/η'
    });
    this.prodChart = new LineChart(this.cfg.prodChartId, {
      label: this.cfg.productLabel, color: 'rgb(130, 60, 200)'
    });

    // State
    this.startX = this.cfg.startX;
    this.startY = this.cfg.startY;
    this.currentResult = null;
    this.isDragging = false;

    // Precompute static landscape
    this._precomputeLandscape();

    // Initial run
    this._runAndDraw();

    // Wire up slider and drag events
    this.etaSlider.addEventListener('input', () => this._runAndDraw());
    this._bindDrag();
  }

  // --- Coordinate helpers (use this.cfg viewport bounds) ---

  _worldToPixel(wx, wy) {
    const { xMin, xMax, yMin, yMax } = this.cfg;
    const W = this.overlayCanvas.width;
    const H = this.overlayCanvas.height;
    return [
      ((wx - xMin) / (xMax - xMin)) * (W - 1),
      ((yMax - wy) / (yMax - yMin)) * (H - 1)
    ];
  }

  _pixelToWorld(px, py) {
    const { xMin, xMax, yMin, yMax } = this.cfg;
    const W = this.overlayCanvas.width;
    const H = this.overlayCanvas.height;
    return [
      xMin + (px / (W - 1)) * (xMax - xMin),
      yMax - (py / (H - 1)) * (yMax - yMin)
    ];
  }

  // --- Landscape precomputation ---

  _precomputeLandscape() {
    const canvas = this.landscapeCanvas;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(W, H);
    const d   = img.data;
    const { xMin, xMax, yMin, yMax, loss } = this.cfg;

    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const wx = xMin + (px / (W - 1)) * (xMax - xMin);
        const wy = yMax - (py / (H - 1)) * (yMax - yMin);
        const [r, g, b] = logLossToColor(Math.log10(loss(wx, wy) + 1e-10));
        const i = (py * W + px) * 4;
        d[i] = r; d[i+1] = g; d[i+2] = b; d[i+3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  // --- Overlay drawing ---

  _drawOverlay() {
    const canvas = this.overlayCanvas;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    const traj = this.currentResult ? this.currentResult.trajectory : null;

    if (traj && traj.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(20, 20, 20, 0.85)';
      ctx.lineWidth = 1.5;
      const [x0, y0] = this._worldToPixel(traj[0].x, traj[0].y);
      ctx.moveTo(x0, y0);
      for (let i = 1; i < traj.length; i++) {
        const [px, py] = this._worldToPixel(traj[i].x, traj[i].y);
        ctx.lineTo(px, py);
      }
      ctx.stroke();

      const [ex, ey] = this._worldToPixel(
        traj[traj.length - 1].x, traj[traj.length - 1].y
      );
      ctx.beginPath();
      ctx.arc(ex, ey, 4, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(20, 20, 20, 0.9)';
      ctx.fill();
    }

    // Draggable start point
    const [sx, sy] = this._worldToPixel(this.startX, this.startY);
    ctx.beginPath();
    ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
    ctx.fillStyle = 'white';
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
  }

  // --- GD runner ---

  _runGD(x0, y0, eta) {
    const { loss, gradient, maxEigenvalue, productFn,
            maxIter, divergeThreshold, convergeThreshold } = this.cfg;

    const trajectory = [{ x: x0, y: y0 }];
    const lossHist   = [{ iteration: 0, value: loss(x0, y0) }];
    const eigHist    = [{ iteration: 0, value: maxEigenvalue(x0, y0) }];
    const prodHist   = [{ iteration: 0, value: productFn(x0, y0) }];

    let x = x0, y = y0;

    for (let t = 1; t <= maxIter; t++) {
      const [gx, gy] = gradient(x, y);
      x -= eta * gx;
      y -= eta * gy;

      const l = loss(x, y);
      trajectory.push({ x, y });
      lossHist.push({ iteration: t, value: l });
      eigHist.push({ iteration: t, value: maxEigenvalue(x, y) });
      prodHist.push({ iteration: t, value: productFn(x, y) });

      if (!isFinite(l) || l > divergeThreshold) break;
      if (l < convergeThreshold) break;
    }

    return { trajectory, lossHist, eigHist, prodHist };
  }

  // --- Run GD and update all charts and overlay ---

  _runAndDraw() {
    const eta = parseFloat(this.etaSlider.value);
    if (this.etaDisplay) this.etaDisplay.textContent = eta.toFixed(3);

    this.currentResult = this._runGD(this.startX, this.startY, eta);
    const { lossHist, eigHist, prodHist } = this.currentResult;

    this._drawOverlay();
    this.lossChart.update(lossHist);
    this.eigChart.update(eigHist, 2 / eta);
    this.prodChart.update(prodHist);
  }

  // --- Drag handling ---

  _getCanvasPos(e) {
    const rect   = this.overlayCanvas.getBoundingClientRect();
    const scaleX = this.overlayCanvas.width  / rect.width;
    const scaleY = this.overlayCanvas.height / rect.height;
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return [(cx - rect.left) * scaleX, (cy - rect.top) * scaleY];
  }

  _isNearStart(px, py) {
    const [sx, sy] = this._worldToPixel(this.startX, this.startY);
    const dx = px - sx, dy = py - sy;
    return Math.sqrt(dx * dx + dy * dy) < 14;
  }

  _applyDragPos(px, py) {
    const W = this.overlayCanvas.width, H = this.overlayCanvas.height;
    const cpx = Math.max(0, Math.min(W - 1, px));
    const cpy = Math.max(0, Math.min(H - 1, py));
    [this.startX, this.startY] = this._pixelToWorld(cpx, cpy);
    this._runAndDraw();
  }

  _bindDrag() {
    const oc = this.overlayCanvas;

    oc.addEventListener('mousedown', (e) => {
      const [px, py] = this._getCanvasPos(e);
      if (this._isNearStart(px, py)) { this.isDragging = true; e.preventDefault(); }
    });

    oc.addEventListener('mousemove', (e) => {
      const [px, py] = this._getCanvasPos(e);
      if (this.isDragging) {
        this._applyDragPos(px, py);
        e.preventDefault();
      } else {
        oc.style.cursor = this._isNearStart(px, py) ? 'grab' : 'default';
      }
    });

    oc.addEventListener('mouseup',    () => { this.isDragging = false; });
    oc.addEventListener('mouseleave', () => { this.isDragging = false; });

    oc.addEventListener('touchstart', (e) => {
      const [px, py] = this._getCanvasPos(e);
      if (this._isNearStart(px, py)) { this.isDragging = true; e.preventDefault(); }
    }, { passive: false });

    oc.addEventListener('touchmove', (e) => {
      if (!this.isDragging) return;
      const [px, py] = this._getCanvasPos(e);
      this._applyDragPos(px, py);
      e.preventDefault();
    }, { passive: false });

    oc.addEventListener('touchend', () => { this.isDragging = false; });
  }
}

// ============================================================================
// DSN1 — L(x, y) = (1 - xy)²
// ============================================================================
// Minimum manifold: hyperbola xy = 1
//
// ∇L        = [-2y(1-xy),  -2x(1-xy)]
// H         = [[ 2y²,        2(1-2xy) ],
//              [ 2(1-2xy),   2x²      ]]
// λ_max(H)  = (x²+y²) + sqrt((y²-x²)² + 4(1-2xy)²)

function dsn1Loss(x, y) {
  const p = 1 - x * y;
  return p * p;
}

function dsn1Gradient(x, y) {
  const p = 1 - x * y;
  return [-2 * y * p, -2 * x * p];
}

function dsn1MaxEigenvalue(x, y) {
  const a    = 2 * y * y;
  const d    = 2 * x * x;
  const b    = 2 * (1 - 2 * x * y);
  const mid  = (a + d) / 2;
  const half = (a - d) / 2;
  return mid + Math.sqrt(half * half + b * b);
}

// ============================================================================
// DSN2 — L(x, y) = (1 - x²y²)²
// ============================================================================
// Minimum manifold: hyperbolas xy = +1 and xy = -1 (four branches)
//
// ∂L/∂x     = -4xy²(1-x²y²)
// ∂L/∂y     = -4x²y(1-x²y²)
// ∂²L/∂x²   = 4y²(3x²y² - 1)
// ∂²L/∂y²   = 4x²(3x²y² - 1)
// ∂²L/∂x∂y  = -8xy(1 - 2x²y²)

function dsn2Loss(x, y) {
  const p = 1 - x * x * y * y;
  return p * p;
}

function dsn2Gradient(x, y) {
  const p = 1 - x * x * y * y;
  return [-4 * x * y * y * p, -4 * x * x * y * p];
}

function dsn2MaxEigenvalue(x, y) {
  const q    = x * x * y * y;
  const a    =  4 * y * y * (3 * q - 1);
  const d    =  4 * x * x * (3 * q - 1);
  const b    = -8 * x * y * (1 - 2 * q);
  const mid  = (a + d) / 2;
  const half = (a - d) / 2;
  return mid + Math.sqrt(half * half + b * b);
}

// ============================================================================
// INIT — wait for MathJax then start
// ============================================================================

function init() {
  initEigenvectorWidget();

  new DeepScalarWidget({
    xMin: -2, xMax: 6, yMin: -2, yMax: 6,
    loss: dsn1Loss,
    gradient: dsn1Gradient,
    maxEigenvalue: dsn1MaxEigenvalue,
    productFn: (x, y) => x * y,
    productLabel: 'xy',
    landscapeId: 'dsn-landscape',
    overlayId:   'dsn-overlay',
    etaSliderId: 'dsn-eta-slider',
    etaValueId:  'dsn-eta-value',
    lossChartId: 'dsn-loss',
    eigChartId:  'dsn-eig',
    prodChartId: 'dsn-prod',
    startX: 3.0, startY: 1.0
  });

  new DeepScalarWidget({
    xMin: -3, xMax: 3, yMin: -3, yMax: 3,
    loss: dsn2Loss,
    gradient: dsn2Gradient,
    maxEigenvalue: dsn2MaxEigenvalue,
    productFn: (x, y) => x * x * y * y,
    productLabel: 'x²y²',
    landscapeId: 'dsn2-landscape',
    overlayId:   'dsn2-overlay',
    etaSliderId: 'dsn2-eta-slider',
    etaValueId:  'dsn2-eta-value',
    lossChartId: 'dsn2-loss',
    eigChartId:  'dsn2-eig',
    prodChartId: 'dsn2-prod',
    startX: 1.5, startY: 0.5
  });
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
