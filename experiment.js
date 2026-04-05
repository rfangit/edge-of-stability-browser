// ============================================================================
// EXPERIMENT ENTRY POINT
// ============================================================================
// Standalone page for experimental widgets. No dependency on app.js or
// tutorial.js — imports only what it needs from the shared modules.

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

  const sim = new Simulation({ kEigs: 1, stepsPerSecId: null, sharpnessProximityThreshold: 0.05 });

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

  sim.onDiverge = () => { startBtn.textContent = 'diverged'; startBtn.disabled = true; };

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
    lossChart.clear(); sharpnessChart.clear(); projChart.clear();
    startBtn.textContent = 'start';
    startBtn.disabled = false;
    const captureEl = document.getElementById('eigenvector-capture-info');
    if (captureEl) { captureEl.textContent = ''; captureEl.style.display = 'none'; captureEl.dataset.set = ''; }
  });

  return sim;
}

// ============================================================================
// DEEP SCALAR NETWORK — shared infrastructure
// ============================================================================
//
// Architecture:
//
//   LandscapeCanvas   — renders one canvas (background colormap + overlay).
//                       Two modes:
//                         primary:true  — owns drag, runs GD, fires onResult(result)
//                         primary:false — exposes setTrajectory(traj) to mirror
//                       Colormaps and contours are passed as functions so the
//                       same class renders both loss and sharpness canvases.
//
//   DeepScalarExperiment — orchestrator. Owns one or more LandscapeCanvas
//                          instances, one or more sliders (η, δ, ...), and
//                          zero or more LineChart instances below the canvases.
//                          The primary canvas's onResult feeds all mirrors and
//                          charts. Adding a new variant = new math functions +
//                          new experiment config. No new plumbing.
//
// Colormaps:
//   colormapLoss(x,y,params)      — log-scale red-blue (blue=low, red=high)
//   colormapSharpness(x,y,params) — log-scale red-blue applied to λ_max
//
// Contours:
//   A contour line is drawn on the overlay of any canvas whose config includes
//   contourFn(params) → scalar. Pixels where the canvas's scalar field crosses
//   that value are coloured black using a lightweight per-pixel scan.

// ---- Shared colourmap ----

// Red-blue diverging colourmap. t in [-1,1]: -1=blue, 0=white, 1=red.
function valueToColorAutoscale(t) {
  if (t < 0) {
    const s = -t;
    return [Math.round(255 * (1 - s)), Math.round(255 * (1 - s)), 255];
  } else {
    return [255, Math.round(255 * (1 - t)), Math.round(255 * (1 - t))];
  }
}

// ============================================================================
// LandscapeCanvas
// ============================================================================
// Config:
//   canvasId      string   — background canvas (static colormap)
//   overlayId     string   — overlay canvas (trajectory + contour + drag point)
//   viewport      {xMin,xMax,yMin,yMax}
//   colormap      (x,y,params) => [r,g,b]   — called once per pixel at init
//   contourFn     (params) => number|null   — if set, draws isoline at this value
//   scalarField   (x,y,params) => number    — field used for contour detection
//   primary       bool     — true: owns drag + GD; false: mirror only
//   startX,startY number   — initial position (primary only)
//   onResult      fn       — called with GD result after each run (primary only)
//   params        object   — live params object (shared with experiment)

class LandscapeCanvas {
  constructor(cfg) {
    this.cfg = cfg;
    this.bgCanvas = document.getElementById(cfg.canvasId);
    this.ovCanvas = document.getElementById(cfg.overlayId);
    if (!this.bgCanvas || !this.ovCanvas) return;

    this.startX    = cfg.startX || 0;
    this.startY    = cfg.startY || 0;
    this.trajectory = null;
    this.isDragging = false;

    this._renderBackground();

    if (cfg.primary) this._bindDrag();
  }

  // ---- Coordinate helpers ----

  _w2p(wx, wy) {
    const { xMin, xMax, yMin, yMax } = this.cfg.viewport;
    const W = this.bgCanvas.width, H = this.bgCanvas.height;
    return [
      ((wx - xMin) / (xMax - xMin)) * (W - 1),
      ((yMax - wy) / (yMax - yMin)) * (H - 1)
    ];
  }

  _p2w(px, py) {
    const { xMin, xMax, yMin, yMax } = this.cfg.viewport;
    const W = this.bgCanvas.width, H = this.bgCanvas.height;
    return [
      xMin + (px / (W - 1)) * (xMax - xMin),
      yMax - (py / (H - 1)) * (yMax - yMin)
    ];
  }

  // ---- Background (autoscaled to actual range of scalarField in the viewport) ----
  // logScale: true  — log10 transform before mapping (default)
  //           false — linear
  // Both use red-blue-white diverging colourmap, white centred at the midpoint
  // of the value range in the viewport.

  _renderBackground() {
    const canvas = this.bgCanvas;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    const { xMin, xMax, yMin, yMax } = this.cfg.viewport;
    const sf     = this.cfg.scalarField;
    const useLog = this.cfg.logScale !== false;

    // Pass 1: evaluate and transform scalar field
    const vals = new Float32Array(W * H);
    let vMin =  Infinity;
    let vMax = -Infinity;

    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const wx = xMin + (px / (W - 1)) * (xMax - xMin);
        const wy = yMax - (py / (H - 1)) * (yMax - yMin);
        const raw = sf(wx, wy, this.cfg.params);
        const v   = useLog ? Math.log10(Math.max(raw, 1e-10)) : raw;
        vals[py * W + px] = v;
        if (v > vMax) vMax = v;
        if (v < vMin) vMin = v;
      }
    }

    // If a fixed whitePoint is specified, use it as the centre directly.
    // Otherwise fall back to the midpoint of the range.
    let vMid;
    if (this.cfg.whitePoint !== undefined) {
      const wp = typeof this.cfg.whitePoint === 'function'
        ? this.cfg.whitePoint(this.cfg.params)
        : this.cfg.whitePoint;
      vMid = useLog
        ? Math.log10(Math.max(wp, 1e-10))
        : wp;
    } else {
      vMid = (vMin + vMax) / 2;
    }
    const vHalf = Math.max(Math.max(vMax - vMid, vMid - vMin), 1e-6);

    // Pass 2: map to colour
    const img = ctx.createImageData(W, H);
    const d   = img.data;

    for (let i = 0; i < W * H; i++) {
      const t = Math.max(-1, Math.min(1, (vals[i] - vMid) / vHalf));
      const [r, g, b] = valueToColorAutoscale(t);
      d[i*4] = r; d[i*4+1] = g; d[i*4+2] = b; d[i*4+3] = 255;
    }

    ctx.putImageData(img, 0, 0);
  }

  // ---- Overlay (trajectory + contour + drag point) ----

  drawOverlay() {
    const canvas = this.ovCanvas;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    // Contour line: scan pixels for sign-change of (scalarField - contourValue)
    if (this.cfg.contourFn && this.cfg.scalarField) {
      const contourVal = this.cfg.contourFn(this.cfg.params);
      if (contourVal !== null && isFinite(contourVal)) {
        const { xMin, xMax, yMin, yMax } = this.cfg.viewport;
        const imgData = ctx.createImageData(W, H);
        const cd = imgData.data;

        // Build scalar field grid (reuse across neighbours)
        const field = new Float32Array(W * H);
        for (let py = 0; py < H; py++) {
          for (let px = 0; px < W; px++) {
            const wx = xMin + (px / (W - 1)) * (xMax - xMin);
            const wy = yMax - (py / (H - 1)) * (yMax - yMin);
            field[py * W + px] = this.cfg.scalarField(wx, wy, this.cfg.params);
          }
        }

        // Mark pixels where the field crosses contourVal (check right + down neighbours)
        for (let py = 0; py < H - 1; py++) {
          for (let px = 0; px < W - 1; px++) {
            const v  = field[py * W + px] - contourVal;
            const vr = field[py * W + px + 1] - contourVal;
            const vd = field[(py + 1) * W + px] - contourVal;
            if (v * vr < 0 || v * vd < 0) {
              const i = (py * W + px) * 4;
              cd[i] = 0; cd[i+1] = 0; cd[i+2] = 0; cd[i+3] = 220;
            }
          }
        }
        ctx.putImageData(imgData, 0, 0);
      }
    }

    // Trajectory
    if (this.trajectory && this.trajectory.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(20, 20, 20, 0.85)';
      ctx.lineWidth = 1.5;
      const [x0, y0] = this._w2p(this.trajectory[0].x, this.trajectory[0].y);
      ctx.moveTo(x0, y0);
      for (let i = 1; i < this.trajectory.length; i++) {
        const [px, py] = this._w2p(this.trajectory[i].x, this.trajectory[i].y);
        ctx.lineTo(px, py);
      }
      ctx.stroke();

      const last = this.trajectory[this.trajectory.length - 1];
      const [ex, ey] = this._w2p(last.x, last.y);
      ctx.beginPath();
      ctx.arc(ex, ey, 4, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(20, 20, 20, 0.9)';
      ctx.fill();
    }

    // Start point (primary only — draggable white circle)
    if (this.cfg.primary) {
      const [sx, sy] = this._w2p(this.startX, this.startY);
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
      ctx.fillStyle = 'white';
      ctx.strokeStyle = '#222';
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();
    }
  }

  // ---- Mirror API ----

  // Called by the experiment when GD produces a new result
  setTrajectory(trajectory) {
    this.trajectory = trajectory;
    this.drawOverlay();
  }

  // Called when params change (η slider) — rerender background then overlay
  refresh() {
    this._renderBackground();
    this.drawOverlay();
  }

  // ---- Drag (primary only) ----

  _getCanvasPos(e) {
    const rect   = this.ovCanvas.getBoundingClientRect();
    const scaleX = this.ovCanvas.width  / rect.width;
    const scaleY = this.ovCanvas.height / rect.height;
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return [(cx - rect.left) * scaleX, (cy - rect.top) * scaleY];
  }

  _isNearStart(px, py) {
    const [sx, sy] = this._w2p(this.startX, this.startY);
    return Math.hypot(px - sx, py - sy) < 14;
  }

  _applyDrag(px, py) {
    const W = this.ovCanvas.width, H = this.ovCanvas.height;
    const cpx = Math.max(0, Math.min(W - 1, px));
    const cpy = Math.max(0, Math.min(H - 1, py));
    [this.startX, this.startY] = this._p2w(cpx, cpy);
    if (this.cfg.onResult) this.cfg.onResult(null); // trigger experiment rerun
  }

  _bindDrag() {
    const oc = this.ovCanvas;

    oc.addEventListener('mousedown', (e) => {
      const [px, py] = this._getCanvasPos(e);
      if (this._isNearStart(px, py)) { this.isDragging = true; e.preventDefault(); }
    });

    oc.addEventListener('mousemove', (e) => {
      const [px, py] = this._getCanvasPos(e);
      if (this.isDragging) {
        this._applyDrag(px, py);
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
      this._applyDrag(px, py);
      e.preventDefault();
    }, { passive: false });

    oc.addEventListener('touchend', () => { this.isDragging = false; });
  }
}

// ============================================================================
// DeepScalarExperiment
// ============================================================================
// Config:
//   params        object   — live mutable params (eta, delta, etc.)
//                            all sliders write into this; math fns close over it
//   math          { loss, gradient, maxEigenvalue }
//   sliders       [{ id, valueId, param, min, max, step, decimals }]
//   canvases      [{ canvasId, overlayId, primary, colormap, contourFn,
//                    scalarField, startX, startY }]
//   charts        [{ id, label, color, dataKey, refFn }]
//                   dataKey: 'lossHist' | 'eigHist'
//                   refFn(params): optional reference line value
//   viewport      {xMin, xMax, yMin, yMax}
//   maxIter, divergeThreshold, convergeThreshold

class DeepScalarExperiment {
  constructor(cfg) {
    this.cfg    = cfg;
    this.params = cfg.params;

    // Build canvases
    this.canvases = cfg.canvases.map(cc => {
      const lc = new LandscapeCanvas({
        ...cc,
        viewport: cfg.viewport,
        params:   this.params,
        onResult: cc.primary ? () => this._runAndDraw() : null
      });
      return { lc, cc };
    });

    // Build charts
    this.charts = cfg.charts.map(cc => ({
      chart: new LineChart(cc.id, { label: cc.label, color: cc.color, refLabel: cc.refLabel }),
      cc
    }));

    // Wire sliders
    cfg.sliders.forEach(sc => {
      const el  = document.getElementById(sc.id);
      const val = document.getElementById(sc.valueId);
      if (!el) return;
      const dec = sc.decimals !== undefined ? sc.decimals : 3;
      el.addEventListener('input', () => {
        this.params[sc.param] = parseFloat(el.value);
        if (val) val.textContent = this.params[sc.param].toFixed(dec);
        // Refresh all canvas backgrounds — both loss and sharpness depend on
        // params (loss changes with δ, sharpness contour shifts with η)
        this._refreshAllCanvases();
        this._runAndDraw();
      });
      // Set initial display
      if (val) val.textContent = this.params[sc.param].toFixed(dec);
    });

    // Initial run
    this._runAndDraw();
  }

  _runGD() {
    const { loss, gradient, maxEigenvalue } = this.cfg.math;
    const { maxIter = 200, divergeThreshold = 1e6, convergeThreshold = 0.001 } = this.cfg;
    const primary = this.canvases.find(c => c.cc.primary);
    if (!primary) return null;

    const x0 = primary.lc.startX;
    const y0 = primary.lc.startY;

    const trajectory = [{ x: x0, y: y0 }];
    const lossHist   = [{ iteration: 0, value: loss(x0, y0) }];
    const eigHist    = [{ iteration: 0, value: maxEigenvalue(x0, y0) }];

    let x = x0, y = y0;

    for (let t = 1; t <= maxIter; t++) {
      const [gx, gy] = gradient(x, y);
      x -= this.params.eta * gx;
      y -= this.params.eta * gy;

      const l = loss(x, y);
      trajectory.push({ x, y });
      lossHist.push({ iteration: t, value: l });
      eigHist.push({ iteration: t, value: maxEigenvalue(x, y) });

      if (!isFinite(l) || l > divergeThreshold) break;
      if (l < convergeThreshold) break;
    }

    return { trajectory, lossHist, eigHist };
  }

  _runAndDraw() {
    const result = this._runGD();
    if (!result) return;

    const { trajectory, lossHist, eigHist } = result;

    // Push trajectory to all canvases
    this.canvases.forEach(({ lc }) => lc.setTrajectory(trajectory));

    // Update charts
    this.charts.forEach(({ chart, cc }) => {
      const hist = cc.dataKey === 'lossHist' ? lossHist : eigHist;
      const ref  = cc.refFn ? cc.refFn(this.params) : undefined;
      chart.update(hist, ref);
    });
  }

  // Refresh all canvas backgrounds when params change
  _refreshAllCanvases() {
    this.canvases.forEach(({ lc }) => lc.refresh());
  }
}

// ============================================================================
// DSN1 — L(x, y) = (1 - xy)²
// ============================================================================
// Minimum manifold: hyperbola xy = 1
//
// ∇L       = [-2y(1-xy),  -2x(1-xy)]
// ∂²L/∂x²  = 2y²
// ∂²L/∂y²  = 2x²
// ∂²L/∂x∂y = 2(1-2xy)
// λ_max    = (x²+y²) + sqrt((y²-x²)² + 4(1-2xy)²)

function dsn1Loss(x, y)         { const p = 1-x*y; return p*p; }
function dsn1Gradient(x, y)     { const p = 1-x*y; return [-2*y*p, -2*x*p]; }
function dsn1MaxEig(x, y) {
  const a = 2*y*y, d = 2*x*x, b = 2*(1-2*x*y);
  const mid = (a+d)/2, half = (a-d)/2;
  return mid + Math.sqrt(half*half + b*b);
}

// ============================================================================
// DSN2 — L(x, y) = (1 - x²y²)²
// ============================================================================
// Minimum manifold: xy = ±1 (four branches)
//
// ∂L/∂x    = -4xy²(1-x²y²)
// ∂L/∂y    = -4x²y(1-x²y²)
// ∂²L/∂x²  = 4y²(3x²y²-1)
// ∂²L/∂y²  = 4x²(3x²y²-1)
// ∂²L/∂x∂y = -8xy(1-2x²y²)

function dsn2Loss(x, y)         { const p = 1-x*x*y*y; return p*p; }
function dsn2Gradient(x, y)     { const p = 1-x*x*y*y; return [-4*x*y*y*p, -4*x*x*y*p]; }
function dsn2MaxEig(x, y) {
  const q = x*x*y*y;
  const a = 4*y*y*(3*q-1), d = 4*x*x*(3*q-1), b = -8*x*y*(1-2*q);
  const mid = (a+d)/2, half = (a-d)/2;
  return mid + Math.sqrt(half*half + b*b);
}

// ============================================================================
// DSN3 — L(x, y) = (1 + δx²)y²
// ============================================================================
// Minimum: y=0 for all x (the entire x-axis).
// δ controls how the curvature in y varies with x — at x=0 the bowl is
// uniform; as |x| grows the bowl steepens (δ>0) making the minimum harder
// to approach from large x.
//
// ∂L/∂x    = 2δxy²
// ∂L/∂y    = 2(1+δx²)y
// ∂²L/∂x²  = 2δy²
// ∂²L/∂y²  = 2(1+δx²)
// ∂²L/∂x∂y = 4δxy
// λ_max    = largest eigenvalue of the 2×2 Hessian (closed form)

function dsn3Loss(x, y, p)      { return (1 + p.delta*x*x)*y*y; }
function dsn3Gradient(x, y, p)  { return [2*p.delta*x*y*y, 2*(1+p.delta*x*x)*y]; }
function dsn3MaxEig(x, y, p) {
  const a = 2*p.delta*y*y;
  const d = 2*(1+p.delta*x*x);
  const b = 4*p.delta*x*y;
  const mid = (a+d)/2, half = (a-d)/2;
  return mid + Math.sqrt(half*half + b*b);
}

// ============================================================================
// DSN4 — L(x, y) = (1 + δx)y²
// ============================================================================
// Minimum: y=0 for all x with 1+δx > 0 (i.e. x > -1/δ).
// The curvature in y grows linearly with x, breaking left-right symmetry.
//
// ∂L/∂x    = δy²
// ∂L/∂y    = 2(1+δx)y
// ∂²L/∂x²  = 0
// ∂²L/∂y²  = 2(1+δx)
// ∂²L/∂x∂y = 2δy
// λ_max    = largest eigenvalue of the 2×2 Hessian (closed form)

function dsn4Loss(x, y, p)      { return (1 + p.delta*x)*y*y; }
function dsn4Gradient(x, y, p)  { return [p.delta*y*y, 2*(1+p.delta*x)*y]; }
function dsn4MaxEig(x, y, p) {
  const a = 0;
  const d = 2*(1+p.delta*x);
  const b = 2*p.delta*y;
  const mid = (a+d)/2, half = (a-d)/2;
  return mid + Math.sqrt(half*half + b*b);
}

// ============================================================================
// Function 5 — L(x, y) = (1 - (ax² + y²))²
// ============================================================================
// Minimum manifold: ellipse ax² + y² = 1.
// When a=1 this is a circle; a<1 stretches it along x, a>1 compresses it.
//
// Let p = 1 - (ax² + y²):
// ∂L/∂x    = -4ax·p
// ∂L/∂y    = -4y·p
// ∂²L/∂x²  = 4a(2ax² - p)
// ∂²L/∂y²  = 4(2y² - p)
// ∂²L/∂x∂y = 8axy

function dsn5Loss(x, y, p) {
  const q = 1 - (p.a * x*x + y*y);
  return q * q;
}

function dsn5Gradient(x, y, p) {
  const q = 1 - (p.a * x*x + y*y);
  return [-4 * p.a * x * q, -4 * y * q];
}

function dsn5MaxEig(x, y, p) {
  const q    = 1 - (p.a * x*x + y*y);
  const a11  = 4 * p.a * (2 * p.a * x*x - q);
  const a22  = 4 * (2 * y*y - q);
  const a12  = 8 * p.a * x * y;
  const mid  = (a11 + a22) / 2;
  const half = (a11 - a22) / 2;
  return mid + Math.sqrt(half*half + a12*a12);
}

function init() {
  initEigenvectorWidget();

  // Shared viewport for DSN1 and DSN2
  const vp1 = { xMin: -2, xMax: 6, yMin: -2, yMax: 6 };
  const vp2 = { xMin: -3, xMax: 3, yMin: -3, yMax: 3 };
  const vp34 = { xMin: -3, xMax: 3, yMin: -3, yMax: 3 };

  // ---- DSN1 ----
  const p1 = { eta: 0.2 };
  new DeepScalarExperiment({
    params:  p1,
    viewport: vp1,
    math: { loss: dsn1Loss, gradient: dsn1Gradient, maxEigenvalue: dsn1MaxEig },
    sliders: [{ id: 'dsn1-eta-slider', valueId: 'dsn1-eta-value', param: 'eta', decimals: 3 }],
    canvases: [
      {
        canvasId: 'dsn1-loss-bg', overlayId: 'dsn1-loss-ov', primary: true,
        scalarField: (x, y) => dsn1Loss(x, y),
        startX: 3.0, startY: 1.0
      },
      {
        canvasId: 'dsn1-sharp-bg', overlayId: 'dsn1-sharp-ov', primary: false,
        scalarField: (x, y) => dsn1MaxEig(x, y),
        logScale: false,
        whitePoint: (params) => 2 / params.eta,
        contourFn:   (params) => 2 / params.eta
      }
    ],
    charts: [
      { id: 'dsn1-loss-chart', label: 'loss',   color: 'rgb(40,130,130)', dataKey: 'lossHist' },
      { id: 'dsn1-eig-chart',  label: 'λ_max',  color: 'rgb(220,50,50)',  dataKey: 'eigHist',
        refFn: (p) => 2/p.eta, refLabel: '2/η' }
    ]
  });

  // ---- DSN2 ----
  const p2 = { eta: 0.2 };
  new DeepScalarExperiment({
    params:  p2,
    viewport: vp2,
    math: { loss: dsn2Loss, gradient: dsn2Gradient, maxEigenvalue: dsn2MaxEig },
    sliders: [{ id: 'dsn2-eta-slider', valueId: 'dsn2-eta-value', param: 'eta', decimals: 3 }],
    canvases: [
      {
        canvasId: 'dsn2-loss-bg', overlayId: 'dsn2-loss-ov', primary: true,
        scalarField: (x, y) => dsn2Loss(x, y),
        startX: 1.5, startY: 0.5
      },
      {
        canvasId: 'dsn2-sharp-bg', overlayId: 'dsn2-sharp-ov', primary: false,
        scalarField: (x, y) => dsn2MaxEig(x, y),
        logScale: false,
        whitePoint: (params) => 2 / params.eta,
        contourFn:   (params) => 2 / params.eta
      }
    ],
    charts: [
      { id: 'dsn2-loss-chart', label: 'loss',  color: 'rgb(40,130,130)', dataKey: 'lossHist' },
      { id: 'dsn2-eig-chart',  label: 'λ_max', color: 'rgb(220,50,50)', dataKey: 'eigHist',
        refFn: (p) => 2/p.eta, refLabel: '2/η' }
    ]
  });

  // ---- DSN3 ----
  const p3 = { eta: 0.2, delta: 2 };
  new DeepScalarExperiment({
    params:  p3,
    viewport: vp34,
    math: {
      loss:          (x, y) => dsn3Loss(x, y, p3),
      gradient:      (x, y) => dsn3Gradient(x, y, p3),
      maxEigenvalue: (x, y) => dsn3MaxEig(x, y, p3)
    },
    sliders: [
      { id: 'dsn3-eta-slider',   valueId: 'dsn3-eta-value',   param: 'eta',   decimals: 3 },
      { id: 'dsn3-delta-slider', valueId: 'dsn3-delta-value', param: 'delta', decimals: 1 }
    ],
    canvases: [
      {
        canvasId: 'dsn3-loss-bg', overlayId: 'dsn3-loss-ov', primary: true,
        scalarField: (x, y) => dsn3Loss(x, y, p3),
        logScale: true, whitePoint: 0.1,
        startX: 0, startY: 2
      },
      {
        canvasId: 'dsn3-sharp-bg', overlayId: 'dsn3-sharp-ov', primary: false,
        scalarField: (x, y) => dsn3MaxEig(x, y, p3),
        logScale: false,
        whitePoint: (params) => 2 / params.eta,
        contourFn: (params) => 2 / params.eta
      }
    ],
    charts: [
      { id: 'dsn3-loss-chart', label: 'loss',  color: 'rgb(40,130,130)', dataKey: 'lossHist' },
      { id: 'dsn3-eig-chart',  label: 'λ_max', color: 'rgb(220,50,50)', dataKey: 'eigHist',
        refFn: (p) => 2/p.eta, refLabel: '2/η' }
    ]
  });

  // ---- DSN4 ----
  const p4 = { eta: 0.2, delta: 0.05 };
  new DeepScalarExperiment({
    params:  p4,
    viewport: vp34,
    math: {
      loss:          (x, y) => dsn4Loss(x, y, p4),
      gradient:      (x, y) => dsn4Gradient(x, y, p4),
      maxEigenvalue: (x, y) => dsn4MaxEig(x, y, p4)
    },
    sliders: [
      { id: 'dsn4-eta-slider',   valueId: 'dsn4-eta-value',   param: 'eta',   decimals: 3 },
      { id: 'dsn4-delta-slider', valueId: 'dsn4-delta-value', param: 'delta', decimals: 3 }
    ],
    canvases: [
      {
        canvasId: 'dsn4-loss-bg', overlayId: 'dsn4-loss-ov', primary: true,
        scalarField: (x, y) => dsn4Loss(x, y, p4),
        logScale: false,
        startX: 0, startY: 2
      },
      {
        canvasId: 'dsn4-sharp-bg', overlayId: 'dsn4-sharp-ov', primary: false,
        scalarField: (x, y) => dsn4MaxEig(x, y, p4),
        logScale: false,
        whitePoint: (params) => 2 / params.eta,
        contourFn: (params) => 2 / params.eta
      }
    ],
    charts: [
      { id: 'dsn4-loss-chart', label: 'loss',  color: 'rgb(40,130,130)', dataKey: 'lossHist' },
      { id: 'dsn4-eig-chart',  label: 'λ_max', color: 'rgb(220,50,50)', dataKey: 'eigHist',
        refFn: (p) => 2/p.eta, refLabel: '2/η' }
    ]
  });

  // ---- Function 5 ----
  const p5 = { eta: 0.2, a: 0.5 };
  new DeepScalarExperiment({
    params:  p5,
    viewport: { xMin: -2, xMax: 2, yMin: -2, yMax: 2 },
    math: {
      loss:          (x, y) => dsn5Loss(x, y, p5),
      gradient:      (x, y) => dsn5Gradient(x, y, p5),
      maxEigenvalue: (x, y) => dsn5MaxEig(x, y, p5)
    },
    sliders: [
      { id: 'dsn5-eta-slider', valueId: 'dsn5-eta-value', param: 'eta', decimals: 3 },
      { id: 'dsn5-a-slider',   valueId: 'dsn5-a-value',   param: 'a',   decimals: 2 }
    ],
    canvases: [
      {
        canvasId: 'dsn5-loss-bg', overlayId: 'dsn5-loss-ov', primary: true,
        scalarField: (x, y) => dsn5Loss(x, y, p5) + 1e-4,
        whitePoint: 0.02,
        startX: 0.5, startY: 0.5
      },
      {
        canvasId: 'dsn5-sharp-bg', overlayId: 'dsn5-sharp-ov', primary: false,
        scalarField: (x, y) => dsn5MaxEig(x, y, p5),
        logScale: false,
        whitePoint: (params) => 2 / params.eta,
        contourFn: (params) => 2 / params.eta
      }
    ],
    charts: [
      { id: 'dsn5-loss-chart', label: 'loss',  color: 'rgb(40,130,130)', dataKey: 'lossHist' },
      { id: 'dsn5-eig-chart',  label: 'λ_max', color: 'rgb(220,50,50)', dataKey: 'eigHist',
        refFn: (p) => 2/p.eta, refLabel: '2/η' }
    ]
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
