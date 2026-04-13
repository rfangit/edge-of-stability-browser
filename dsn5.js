// dsn5.js — Function 5 interactive widget
// L(x, y) = (1 - (ax² + y²))²
// Minimum manifold: ellipse ax² + y² = 1.

import { LineChart } from './visualization.js';

// ============================================================================
// Shared colourmap — red-blue diverging, t in [-1, 1]: -1=blue, 0=white, 1=red
// ============================================================================

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
// Renders one canvas pair (background colormap + overlay).
//
// Config:
//   canvasId    string   — background canvas (static colormap)
//   overlayId   string   — overlay canvas (trajectory + contour + drag point)
//   viewport    { xMin, xMax, yMin, yMax }
//   scalarField (x, y, params) => number   — used for colormap and contour detection
//   logScale    bool     — log10 transform before colour mapping (default: true)
//   whitePoint  number | (params) => number | undefined
//                        — value mapped to white; defaults to midpoint of range
//   contourFn   (params) => number | null
//                        — if set, draws a solid isoline at this value
//   dashedContours  [{ fn, field, color, dash, lineWidth }]
//                        — array of additional dashed isolines
//   primary     bool     — true: owns drag + GD trigger; false: mirror only
//   startX, startY  number   — initial position (primary only)
//   onResult    fn       — called when drag repositions the start point (primary only)
//   params      object   — live params shared with the experiment

class LandscapeCanvas {
  constructor(cfg) {
    this.cfg = cfg;
    this.bgCanvas = document.getElementById(cfg.canvasId);
    this.ovCanvas = document.getElementById(cfg.overlayId);
    if (!this.bgCanvas || !this.ovCanvas) return;

    this.startX     = cfg.startX || 0;
    this.startY     = cfg.startY || 0;
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

  // ---- Background (autoscaled colormap) ----

  _renderBackground() {
    const canvas = this.bgCanvas;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    const { xMin, xMax, yMin, yMax } = this.cfg.viewport;
    const sf     = this.cfg.scalarField;
    const useLog = this.cfg.logScale !== false;

    // Pass 1: evaluate and optionally log-transform the scalar field
    const vals = new Float32Array(W * H);
    let vMin =  Infinity, vMax = -Infinity;
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

    // White point: explicit value, function of params, or midpoint of range
    let vMid;
    if (this.cfg.whitePoint !== undefined) {
      const wp = typeof this.cfg.whitePoint === 'function'
        ? this.cfg.whitePoint(this.cfg.params)
        : this.cfg.whitePoint;
      vMid = useLog ? Math.log10(Math.max(wp, 1e-10)) : wp;
    } else {
      vMid = (vMin + vMax) / 2;
    }
    const vHalf = Math.max(vMax - vMid, vMid - vMin, 1e-6);

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

  // ---- Overlay (contours + trajectory + drag handle) ----

  drawOverlay() {
    const canvas = this.ovCanvas;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    // Solid contour: mark pixels where scalarField crosses contourFn(params)
    if (this.cfg.contourFn && this.cfg.scalarField) {
      const contourVal = this.cfg.contourFn(this.cfg.params);
      if (contourVal !== null && isFinite(contourVal)) {
        const { xMin, xMax, yMin, yMax } = this.cfg.viewport;
        const imgData = ctx.createImageData(W, H);
        const cd  = imgData.data;
        const field = new Float32Array(W * H);
        for (let py = 0; py < H; py++)
          for (let px = 0; px < W; px++) {
            const wx = xMin + (px / (W - 1)) * (xMax - xMin);
            const wy = yMax - (py / (H - 1)) * (yMax - yMin);
            field[py * W + px] = this.cfg.scalarField(wx, wy, this.cfg.params);
          }
        for (let py = 0; py < H - 1; py++)
          for (let px = 0; px < W - 1; px++) {
            const v  = field[py * W + px] - contourVal;
            const vr = field[py * W + px + 1] - contourVal;
            const vd = field[(py + 1) * W + px] - contourVal;
            if (v * vr < 0 || v * vd < 0) {
              const i = (py * W + px) * 4;
              cd[i] = 0; cd[i+1] = 0; cd[i+2] = 0; cd[i+3] = 220;
            }
          }
        ctx.putImageData(imgData, 0, 0);
      }
    }

    // Dashed contours: each drawn as a canvas path along sign-change pixels
    if (this.cfg.dashedContours) {
      const { xMin, xMax, yMin, yMax } = this.cfg.viewport;
      for (const dc of this.cfg.dashedContours) {
        const val = dc.fn ? dc.fn(this.cfg.params) : null;
        if (val === null || !isFinite(val)) continue;
        ctx.save();
        ctx.strokeStyle = dc.color     || 'navy';
        ctx.lineWidth   = dc.lineWidth || 1.5;
        ctx.setLineDash(dc.dash        || [6, 4]);
        const field = new Float32Array(W * H);
        for (let py = 0; py < H; py++)
          for (let px = 0; px < W; px++) {
            const wx = xMin + (px / (W - 1)) * (xMax - xMin);
            const wy = yMax - (py / (H - 1)) * (yMax - yMin);
            field[py * W + px] = dc.field(wx, wy, this.cfg.params) - val;
          }
        ctx.beginPath();
        for (let py = 0; py < H - 1; py++)
          for (let px = 0; px < W - 1; px++) {
            const v  = field[py * W + px];
            const vr = field[py * W + px + 1];
            const vd = field[(py + 1) * W + px];
            if (v * vr < 0) { const t = v / (v - vr); ctx.moveTo(px + t, py);     ctx.lineTo(px + t, py + 1); }
            if (v * vd < 0) { const t = v / (v - vd); ctx.moveTo(px,     py + t); ctx.lineTo(px + 1, py + t); }
          }
        ctx.stroke();
        ctx.restore();
      }
    }

    // Trajectory line + endpoint dot
    if (this.trajectory && this.trajectory.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(20, 20, 20, 0.85)';
      ctx.lineWidth   = 1.5;
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

    // Start point: draggable white circle (primary canvas only)
    if (this.cfg.primary) {
      const [sx, sy] = this._w2p(this.startX, this.startY);
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
      ctx.fillStyle   = 'white';
      ctx.strokeStyle = '#222';
      ctx.lineWidth   = 2;
      ctx.fill();
      ctx.stroke();
    }
  }

  // ---- Mirror API (called by DeepScalarExperiment) ----

  setTrajectory(trajectory) {
    this.trajectory = trajectory;
    this.drawOverlay();
  }

  // Rerender background then overlay when params change
  refresh() {
    this._renderBackground();
    this.drawOverlay();
  }

  // ---- Drag (primary canvas only) ----

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
    [this.startX, this.startY] = this._p2w(
      Math.max(0, Math.min(W - 1, px)),
      Math.max(0, Math.min(H - 1, py))
    );
    if (this.cfg.onResult) this.cfg.onResult(null);
  }

  _bindDrag() {
    const oc = this.ovCanvas;
    oc.addEventListener('mousedown',  (e) => { const [px, py] = this._getCanvasPos(e); if (this._isNearStart(px, py)) { this.isDragging = true; e.preventDefault(); } });
    oc.addEventListener('mousemove',  (e) => { const [px, py] = this._getCanvasPos(e); if (this.isDragging) { this._applyDrag(px, py); e.preventDefault(); } else { oc.style.cursor = this._isNearStart(px, py) ? 'grab' : 'default'; } });
    oc.addEventListener('mouseup',    ()  => { this.isDragging = false; });
    oc.addEventListener('mouseleave', ()  => { this.isDragging = false; });
    oc.addEventListener('touchstart', (e) => { const [px, py] = this._getCanvasPos(e); if (this._isNearStart(px, py)) { this.isDragging = true; e.preventDefault(); } }, { passive: false });
    oc.addEventListener('touchmove',  (e) => { if (!this.isDragging) return; this._applyDrag(...this._getCanvasPos(e)); e.preventDefault(); }, { passive: false });
    oc.addEventListener('touchend',   ()  => { this.isDragging = false; });
  }
}

// ============================================================================
// DeepScalarExperiment
// ============================================================================
// Orchestrator: owns one or more LandscapeCanvas instances, one or more sliders,
// and one or more LineChart instances. The primary canvas's drag trigger fires
// _runAndDraw(), which pushes updated trajectory and chart data to all mirrors.
//
// Config:
//   params        object   — live mutable params (eta, a, etc.)
//   math          { loss, gradient, maxEigenvalue }
//   sliders       [{ id, valueId, param, decimals }]
//   canvases      [{ canvasId, overlayId, primary, scalarField, logScale,
//                    whitePoint, contourFn, dashedContours, startX, startY }]
//   charts        [{ id, label, color, dataKey, refFn, refLabel }]
//                   dataKey: 'lossHist' | 'eigHist'
//   viewport      { xMin, xMax, yMin, yMax }
//   maxIter, divergeThreshold, convergeThreshold

class DeepScalarExperiment {
  constructor(cfg) {
    this.cfg    = cfg;
    this.params = cfg.params;

    // Build canvas wrappers
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
        this._refreshAllCanvases();
        this._runAndDraw();
      });
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

    const x0 = primary.lc.startX, y0 = primary.lc.startY;
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
    this.canvases.forEach(({ lc })        => lc.setTrajectory(trajectory));
    this.charts.forEach(({ chart, cc }) => {
      const hist = cc.dataKey === 'lossHist' ? lossHist : eigHist;
      const ref  = cc.refFn ? cc.refFn(this.params) : undefined;
      chart.update(hist, ref);
    });
  }

  _refreshAllCanvases() {
    this.canvases.forEach(({ lc }) => lc.refresh());
  }
}

// ============================================================================
// Function 5 math — L(x, y) = (1 - (ax² + y²))²
// ============================================================================
// Minimum manifold: ellipse ax² + y² = 1.
// Let q = 1 - (ax² + y²):
//   ∂L/∂x    = -4ax·q
//   ∂L/∂y    = -4y·q
//   ∂²L/∂x²  = 4a(2ax² - q)
//   ∂²L/∂y²  = 4(2y² - q)
//   ∂²L/∂x∂y = 8axy

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

// ============================================================================
// Init
// ============================================================================

function initDSN5() {
  const p5 = { eta: 0.3, a: 0.5 };
  new DeepScalarExperiment({
    params:   p5,
    viewport: { xMin: -2, xMax: 2, yMin: -2, yMax: 2 },
    math: {
      loss:          (x, y) => dsn5Loss(x, y, p5),
      gradient:      (x, y) => dsn5Gradient(x, y, p5),
      maxEigenvalue: (x, y) => dsn5MaxEig(x, y, p5)
    },
    sliders: [
      { id: 'dsn5-eta-slider', valueId: 'dsn5-eta-value', param: 'eta', decimals: 3 },
      { id: 'dsn5-a-slider',   valueId: 'dsn5-a-value',   param: 'a',   decimals: 2  }
    ],
    canvases: [
      {
        canvasId:    'dsn5-loss-bg',
        overlayId:   'dsn5-loss-ov',
        primary:     true,
        scalarField: (x, y) => dsn5Loss(x, y, p5) + 1e-4,  // +1e-4 avoids log10(0)
        whitePoint:  0.02,
        startX:      0.2,
        startY:      0.2
      },
      {
        canvasId:    'dsn5-sharp-bg',
        overlayId:   'dsn5-sharp-ov',
        primary:     false,
        scalarField: (x, y) => dsn5MaxEig(x, y, p5),
        logScale:    false,
        whitePoint:  (params) => 2 / params.eta,
        contourFn:   (params) => 2 / params.eta,  // solid black isoline at 2/η
        dashedContours: [
          {
            // Minima ellipse: ax² + y² = 1
            fn:        () => 1,
            field:     (x, y, params) => params.a * x*x + y*y,
            color:     'rgba(0, 0, 139, 0.85)',
            dash:      [7, 5],
            lineWidth: 1.8
          }
        ]
      }
    ],
    charts: [
      {
        id:      'dsn5-loss-chart',
        label:   'loss',
        color:   'rgb(40,130,130)',
        dataKey: 'lossHist'
      },
      {
        id:       'dsn5-eig-chart',
        label:    'λ_max',
        color:    'rgb(220,50,50)',
        dataKey:  'eigHist',
        refFn:    (p) => 2 / p.eta,
        refLabel: '2/η'
      }
    ]
  });
}

// Wait for MathJax before initialising (same pattern as app.js and the existing
// inline widgets on this page — all poll the same MathJax.startup.promise).
function waitForMathJax(attempts = 0) {
  if (window.MathJax && window.MathJax.typesetPromise && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(initDSN5).catch(() => initDSN5());
  } else if (attempts < 50) {
    setTimeout(() => waitForMathJax(attempts + 1), 50);
  } else {
    initDSN5();
  }
}

waitForMathJax();
