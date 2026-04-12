// ============================================================================
// gd_landscape.js — 2-D loss-surface visualisation with interactive GD
// ============================================================================
//
// Exports:
//   LandscapeExperiment   — orchestrator (sliders, panels, charts, reset)
//   LineChart             — minimal canvas line chart (no external deps)
//
// Both classes are self-contained. The only import is gd_core.js.
//
// ---- Math contract ----
// All math functions receive (theta: number[], params: object):
//   loss(theta, params)           → number
//   gradient(theta, params)       → number[]
//   maxEigenvalue(theta, params)  → number   (optional)
//
// For 2-D visualisation theta = [x, y].
//
// ---- LandscapeExperiment config ----
//   params          object     — live mutable params (eta, a, delta, …)
//   math            { loss, gradient, maxEigenvalue? }
//   viewport        { xMin, xMax, yMin, yMax }
//   panels          LandscapePanelConfig[]  (see below)
//   sliders         SliderConfig[]
//   charts          ChartConfig[]
//   resetButtonId?  string
//   maxIter?        number   (default 200)
//   divergeThresh?  number   (default 1e6)
//   convergeThresh? number   (default 1e-6)
//
// ---- LandscapePanelConfig ----
//   bgCanvasId    string
//   ovCanvasId    string
//   primary?      bool        — exactly one panel should be primary
//   scalarField   (theta: number[], params: object) => number
//   logScale?     bool        (default true)
//   whitePoint?   number | (params) => number
//   contourFn?    (params) => number | null
//   startTheta?   [x, y]     (primary only; default [0, 0])
//
// ---- SliderConfig ----
//   id, valueId, param, decimals?
//
// ---- ChartConfig ----
//   id, label, color, dataKey ('lossHist' | 'eigHist'), refFn?, refLabel?
// ============================================================================

import { runGD } from './gd_core.js';

// ============================================================================
// Internal helpers
// ============================================================================

// Red-blue diverging colormap. t ∈ [-1, 1]: −1 → blue, 0 → white, 1 → red.
function _divergingRGB(t) {
  const s = Math.max(-1, Math.min(1, t));
  if (s < 0) {
    const k = -s;
    return [Math.round(255 * (1 - k)), Math.round(255 * (1 - k)), 255];
  }
  return [255, Math.round(255 * (1 - s)), Math.round(255 * (1 - s))];
}

function _fmt(v) {
  if (!isFinite(v)) return '';
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(1);
  return parseFloat(v.toFixed(2)).toString();
}

// ============================================================================
// LineChart
// ============================================================================
// Minimal canvas line chart — no external dependencies.
// update(data, refValue?)  where data = [{ step, value }, …]

export class LineChart {
  constructor(canvasId, cfg = {}) {
    this.canvas = document.getElementById(canvasId);
    this.cfg    = cfg;   // { label, color, refLabel }
    this._data  = [];
    this._ref   = undefined;
  }

  update(data, refValue) {
    this._data = data || [];
    this._ref  = refValue;
    this._draw();
  }

  clear() {
    this._data = [];
    this._ref  = undefined;
    if (this.canvas) {
      this.canvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  _draw() {
    const canvas = this.canvas;
    if (!canvas || this._data.length < 2) return;

    // HiDPI
    const dpr  = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W    = rect.width  || canvas.width;
    const H    = rect.height || canvas.height;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width        = W * dpr;
      canvas.height       = H * dpr;
      canvas.style.width  = W + 'px';
      canvas.style.height = H + 'px';
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const pad = { top: 8, right: 12, bottom: 28, left: 44 };
    const cw  = W - pad.left - pad.right;
    const ch  = H - pad.top  - pad.bottom;

    const vals  = this._data.map(d => d.value);
    const steps = this._data.map(d => d.step);
    let vMin    = Math.min(...vals);
    let vMax    = Math.max(...vals);
    if (this._ref !== undefined) {
      vMin = Math.min(vMin, this._ref);
      vMax = Math.max(vMax, this._ref);
    }
    const vRange = vMax - vMin || 1;
    const sRange = (steps[steps.length - 1] - steps[0]) || 1;

    const sx = s => pad.left + ((s  - steps[0]) / sRange) * cw;
    const sy = v => pad.top  + (1  - (v - vMin) / vRange) * ch;

    // Reference line
    if (this._ref !== undefined && isFinite(this._ref)) {
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#aaa';
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, sy(this._ref));
      ctx.lineTo(pad.left + cw, sy(this._ref));
      ctx.stroke();
      if (this.cfg.refLabel) {
        ctx.fillStyle    = '#999';
        ctx.font         = '10px Monaco, Consolas, monospace';
        ctx.textAlign    = 'right';
        ctx.textBaseline = 'bottom';
        ctx.fillText(this.cfg.refLabel, pad.left + cw, sy(this._ref) - 2);
      }
      ctx.restore();
    }

    // Data line
    ctx.beginPath();
    ctx.strokeStyle = this.cfg.color || '#2563eb';
    ctx.lineWidth   = 1.5;
    this._data.forEach((d, i) => {
      i === 0 ? ctx.moveTo(sx(d.step), sy(d.value))
              : ctx.lineTo(sx(d.step), sy(d.value));
    });
    ctx.stroke();

    // Axes
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + ch);
    ctx.lineTo(pad.left + cw, pad.top + ch);
    ctx.stroke();

    // Y tick labels (3 ticks)
    ctx.fillStyle    = '#777';
    ctx.font         = '10px Monaco, Consolas, monospace';
    ctx.textAlign    = 'right';
    ctx.textBaseline = 'middle';
    [vMin, (vMin + vMax) / 2, vMax].forEach(v => {
      ctx.fillText(_fmt(v), pad.left - 4, sy(v));
    });

    // X tick labels
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(steps[0], sx(steps[0]), pad.top + ch + 4);
    ctx.fillText(steps[steps.length - 1], sx(steps[steps.length - 1]), pad.top + ch + 4);
  }
}

// ============================================================================
// LandscapePanel  (internal — not exported; used only by LandscapeExperiment)
// ============================================================================

class LandscapePanel {
  constructor(cfg) {
    this.cfg      = cfg;
    this.bgCanvas = document.getElementById(cfg.bgCanvasId);
    this.ovCanvas = document.getElementById(cfg.ovCanvasId);
    if (!this.bgCanvas || !this.ovCanvas) return;

    this.startTheta = (cfg.startTheta || [0, 0]).slice();
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
      ((yMax - wy) / (yMax - yMin)) * (H - 1),
    ];
  }

  _p2w(px, py) {
    const { xMin, xMax, yMin, yMax } = this.cfg.viewport;
    const W = this.bgCanvas.width, H = this.bgCanvas.height;
    return [
      xMin + (px / (W - 1)) * (xMax - xMin),
      yMax - (py / (H - 1)) * (yMax - yMin),
    ];
  }

  // ---- Background colormap ----

  _renderBackground() {
    const canvas = this.bgCanvas;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    const { xMin, xMax, yMin, yMax } = this.cfg.viewport;
    const sf     = this.cfg.scalarField;
    const useLog = this.cfg.logScale !== false;

    // Pass 1: evaluate and (optionally) log-transform
    const vals = new Float32Array(W * H);
    let vMin = Infinity, vMax = -Infinity;

    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const wx  = xMin + (px / (W - 1)) * (xMax - xMin);
        const wy  = yMax - (py / (H - 1)) * (yMax - yMin);
        const raw = sf([wx, wy], this.cfg.params);
        const v   = useLog ? Math.log10(Math.max(raw, 1e-10)) : raw;
        vals[py * W + px] = v;
        if (v > vMax) vMax = v;
        if (v < vMin) vMin = v;
      }
    }

    // White centre
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

    // Pass 2: colormap
    const img = ctx.createImageData(W, H);
    const d   = img.data;
    for (let i = 0; i < W * H; i++) {
      const [r, g, b] = _divergingRGB((vals[i] - vMid) / vHalf);
      d[i*4] = r; d[i*4+1] = g; d[i*4+2] = b; d[i*4+3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  // ---- Overlay: contour + trajectory + drag handle ----

  drawOverlay() {
    const canvas = this.ovCanvas;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    // Contour isoline
    if (this.cfg.contourFn && this.cfg.scalarField) {
      const contourVal = this.cfg.contourFn(this.cfg.params);
      if (contourVal !== null && isFinite(contourVal)) {
        const { xMin, xMax, yMin, yMax } = this.cfg.viewport;
        const field = new Float32Array(W * H);
        for (let py = 0; py < H; py++) {
          for (let px = 0; px < W; px++) {
            const wx = xMin + (px / (W - 1)) * (xMax - xMin);
            const wy = yMax - (py / (H - 1)) * (yMax - yMin);
            field[py * W + px] = this.cfg.scalarField([wx, wy], this.cfg.params);
          }
        }
        const imgData = ctx.createImageData(W, H);
        const cd = imgData.data;
        for (let py = 0; py < H - 1; py++) {
          for (let px = 0; px < W - 1; px++) {
            const v  = field[py * W + px]     - contourVal;
            const vr = field[py * W + px + 1] - contourVal;
            const vd = field[(py+1) * W + px] - contourVal;
            if (v * vr < 0 || v * vd < 0) {
              const i = (py * W + px) * 4;
              cd[i] = 0; cd[i+1] = 0; cd[i+2] = 0; cd[i+3] = 200;
            }
          }
        }
        ctx.putImageData(imgData, 0, 0);
      }
    }

    // Trajectory line
    if (this.trajectory && this.trajectory.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(20,20,20,0.85)';
      ctx.lineWidth   = 1.5;
      const [x0, y0] = this._w2p(this.trajectory[0].theta[0], this.trajectory[0].theta[1]);
      ctx.moveTo(x0, y0);
      for (let i = 1; i < this.trajectory.length; i++) {
        const [px, py] = this._w2p(this.trajectory[i].theta[0], this.trajectory[i].theta[1]);
        ctx.lineTo(px, py);
      }
      ctx.stroke();

      // End dot
      const last = this.trajectory[this.trajectory.length - 1];
      const [ex, ey] = this._w2p(last.theta[0], last.theta[1]);
      ctx.beginPath();
      ctx.arc(ex, ey, 4, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(20,20,20,0.9)';
      ctx.fill();
    }

    // Drag handle (primary only)
    if (this.cfg.primary) {
      const [sx, sy] = this._w2p(this.startTheta[0], this.startTheta[1]);
      ctx.beginPath();
      ctx.arc(sx, sy, 7, 0, 2 * Math.PI);
      ctx.fillStyle   = 'white';
      ctx.strokeStyle = '#222';
      ctx.lineWidth   = 2;
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(sx, sy, 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = '#2563eb';
      ctx.fill();
    }
  }

  // ---- Mirror API ----

  setTrajectory(trajectory) {
    this.trajectory = trajectory;
    this.drawOverlay();
  }

  // Re-render background then redraw overlay (called on slider change)
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
    const [sx, sy] = this._w2p(this.startTheta[0], this.startTheta[1]);
    return Math.hypot(px - sx, py - sy) < 14;
  }

  _applyDrag(px, py) {
    const W = this.ovCanvas.width, H = this.ovCanvas.height;
    this.startTheta = this._p2w(
      Math.max(0, Math.min(W - 1, px)),
      Math.max(0, Math.min(H - 1, py)),
    );
    if (this.cfg.onDrag) this.cfg.onDrag();
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
        this._applyDrag(px, py); e.preventDefault();
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
      this._applyDrag(px, py); e.preventDefault();
    }, { passive: false });

    oc.addEventListener('touchend', () => { this.isDragging = false; });
  }
}

// ============================================================================
// LandscapeExperiment
// ============================================================================

export class LandscapeExperiment {
  constructor(cfg) {
    this.cfg    = cfg;
    this.params = cfg.params;

    // Snapshot defaults for reset
    this._defaultParams = { ...cfg.params };
    this._defaultTheta  = null;

    // Build panels
    this.panels = cfg.panels.map(pc => {
      const lp = new LandscapePanel({
        ...pc,
        viewport: cfg.viewport,
        params:   this.params,
        onDrag:   pc.primary ? () => this._runAndDraw() : null,
      });
      if (pc.primary) this._defaultTheta = lp.startTheta.slice();
      return { lp, pc };
    });

    // Build charts
    this.charts = (cfg.charts || []).map(cc => ({
      chart: new LineChart(cc.id, { label: cc.label, color: cc.color, refLabel: cc.refLabel }),
      cc,
    }));

    // Wire sliders
    (cfg.sliders || []).forEach(sc => {
      const el  = document.getElementById(sc.id);
      const val = document.getElementById(sc.valueId);
      if (!el) return;
      const dec = sc.decimals ?? 3;
      el.addEventListener('input', () => {
        this.params[sc.param] = parseFloat(el.value);
        if (val) val.textContent = this.params[sc.param].toFixed(dec);
        this._refreshAllPanels();
        this._runAndDraw();
      });
      if (val) val.textContent = this.params[sc.param].toFixed(dec);
    });

    // Wire reset button
    if (cfg.resetButtonId) {
      const btn = document.getElementById(cfg.resetButtonId);
      if (btn) btn.addEventListener('click', () => this._reset());
    }

    this._runAndDraw();
  }

  // ---- GD run ----

  _runAndDraw() {
    const primary = this.panels.find(p => p.pc.primary);
    if (!primary) return;

    const { loss, gradient, maxEigenvalue = null } = this.cfg.math;

    const result = runGD({
      theta0:        primary.lp.startTheta.slice(),
      eta:           this.params.eta,
      loss:          theta => loss(theta, this.params),
      gradient:      theta => gradient(theta, this.params),
      maxEigenvalue: maxEigenvalue ? theta => maxEigenvalue(theta, this.params) : null,
      nSteps:        this.cfg.maxIter       ?? 200,
      divergeThresh:  this.cfg.divergeThresh ?? 1e6,
      convergeThresh: this.cfg.convergeThresh ?? 1e-6,
    });

    this.panels.forEach(({ lp }) => lp.setTrajectory(result.trajectory));

    this.charts.forEach(({ chart, cc }) => {
      const hist = cc.dataKey === 'eigHist' ? result.eigHist : result.lossHist;
      if (!hist) return;
      chart.update(hist, cc.refFn ? cc.refFn(this.params) : undefined);
    });
  }

  _refreshAllPanels() {
    this.panels.forEach(({ lp }) => lp.refresh());
  }

  _reset() {
    Object.assign(this.params, this._defaultParams);

    (this.cfg.sliders || []).forEach(sc => {
      const el  = document.getElementById(sc.id);
      const val = document.getElementById(sc.valueId);
      const dec = sc.decimals ?? 3;
      if (el)  el.value        = this.params[sc.param];
      if (val) val.textContent = this.params[sc.param].toFixed(dec);
    });

    const primary = this.panels.find(p => p.pc.primary);
    if (primary && this._defaultTheta) {
      primary.lp.startTheta = this._defaultTheta.slice();
    }

    this._refreshAllPanels();
    this._runAndDraw();
  }
}
