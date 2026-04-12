// ============================================================================
// gd_quadratic.js — 1-D quadratic GD visualisation
// ============================================================================
//
// Uses gd_core.js for the GD computation.
// Renders its own parabola + fading-arrow style (not a heatmap).
//
// Public API: GDQuadPair(cfg)
//
// cfg:
//   leftCanvasId, rightCanvasId  string
//   etaSliderId, etaValueId      string
//   resetButtonId                string
//   S_conv   number  — curvature of left (converging) panel
//   S_div    number  — curvature of right (diverging) panel
//   eta      number  — shared learning rate
//   theta0   number  — shared starting position
//   nSteps   number
//   viewport { thetaMin, thetaMax, lossMin, lossMax }
// ============================================================================

import { runGD } from './gd_core.js';

// ============================================================================
// GDQuadPanel — one canvas, one parabola, one GD trajectory
// ============================================================================

class GDQuadPanel {
  constructor(canvas, cfg) {
    this.canvas = canvas;
    if (!this.canvas) return;

    this.S      = cfg.S;
    this.eta    = cfg.eta;
    this.nSteps = cfg.nSteps ?? 7;
    this.vp     = cfg.viewport;
    this.theta0 = cfg.theta0 ?? 0.7;

    this.curveColor = '#555555';
    this.pathColor  = '#2563eb';

    this._setupCanvas();
  }

  // ---- HiDPI ----

  _setupCanvas() {
    const dpr  = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w    = rect.width  || this.canvas.width;
    const h    = rect.height || this.canvas.height;
    this.canvas.width        = w * dpr;
    this.canvas.height       = h * dpr;
    this.canvas.style.width  = w + 'px';
    this.canvas.style.height = h + 'px';
    this.ctx = this.canvas.getContext('2d');
    this.ctx.scale(dpr, dpr);
    this.W = w;
    this.H = h;
  }

  // ---- Coordinate transforms ----

  t2px(theta) {
    const { thetaMin, thetaMax } = this.vp;
    return ((theta - thetaMin) / (thetaMax - thetaMin)) * this.W;
  }

  l2py(loss) {
    const { lossMin, lossMax } = this.vp;
    return this.H - ((loss - lossMin) / (lossMax - lossMin)) * this.H;
  }

  px2t(px) {
    const { thetaMin, thetaMax } = this.vp;
    return thetaMin + (px / this.W) * (thetaMax - thetaMin);
  }

  // ---- Math ----

  _loss(theta)     { return 0.5 * this.S * theta * theta; }
  isConverging()   { return this.eta < 2 / this.S; }

  // ---- GD via shared engine ----

  _runGD() {
    const result = runGD({
      theta0:        [this.theta0],
      eta:           this.eta,
      loss:          ([t]) => this._loss(t),
      gradient:      ([t]) => [this.S * t],
      nSteps:        this.nSteps,
      divergeThresh:  Math.abs(this.vp.thetaMax) ** 2 * this.S * 20,
      convergeThresh: 1e-8,
    });
    return result.trajectory.map(pt => pt.theta[0]);
  }

  // ---- Hit-test ----

  isNearStart(canvasX, canvasY) {
    const sx = this.t2px(this.theta0);
    const sy = this.l2py(this._loss(this.theta0));
    return Math.hypot(canvasX - sx, canvasY - sy) < 16;
  }

  // ---- Full redraw ----

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.W, this.H);
    this._drawQuadratic(ctx);
    this._drawGDPath(ctx);
    this._drawStartDot(ctx);
    this._drawLabels(ctx);
  }

  // ---- Sub-draws ----

  _drawQuadratic(ctx) {
    const { thetaMin, thetaMax } = this.vp;
    ctx.beginPath();
    ctx.strokeStyle = this.curveColor;
    ctx.lineWidth   = 2.5;
    const nPts = 400;
    for (let i = 0; i <= nPts; i++) {
      const theta = thetaMin + (thetaMax - thetaMin) * (i / nPts);
      const px = this.t2px(theta);
      const py = this.l2py(this._loss(theta));
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  _fade(i) {
    if (i === 0) return 1.0;
    return Math.max(0.15, 0.6 - (i - 1) * 0.1);
  }

  _drawGDPath(ctx) {
    const thetas = this._runGD();
    const n = thetas.length;

    for (let i = 0; i < n - 1; i++) {
      const x1 = this.t2px(thetas[i]);
      const y1 = this.l2py(this._loss(thetas[i]));
      const x2 = this.t2px(thetas[i + 1]);
      const y2 = this.l2py(this._loss(thetas[i + 1]));

      if ((x1 < -200 && x2 < -200) || (x1 > this.W + 200 && x2 > this.W + 200)) continue;

      const dx = x2 - x1, dy = y2 - y1;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len < 1) continue;

      const alpha   = this._fade(i);
      const headLen = Math.min(12, len * 0.35);
      const angle   = Math.atan2(dy, dx);

      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = this.pathColor;
      ctx.fillStyle   = this.pathColor;
      ctx.lineWidth   = 2.2;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2 - (dx / len) * headLen * 0.6, y2 - (dy / len) * headLen * 0.6);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - headLen * Math.cos(angle - 0.38), y2 - headLen * Math.sin(angle - 0.38));
      ctx.lineTo(x2 - headLen * Math.cos(angle + 0.38), y2 - headLen * Math.sin(angle + 0.38));
      ctx.closePath();
      ctx.fill();

      ctx.restore();
    }

    // Step dots (skip index 0)
    for (let i = 1; i < n; i++) {
      const px = this.t2px(thetas[i]);
      const py = this.l2py(this._loss(thetas[i]));
      if (px < -50 || px > this.W + 50 || py < -50 || py > this.H + 50) continue;

      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.globalAlpha = 0.9;
      ctx.fillStyle   = this.pathColor;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#222';
      ctx.lineWidth   = 1;
      ctx.stroke();
    }
  }

  _drawStartDot(ctx) {
    const sx = this.t2px(this.theta0);
    const sy = this.l2py(this._loss(this.theta0));

    ctx.beginPath();
    ctx.arc(sx, sy, 8, 0, 2 * Math.PI);
    ctx.fillStyle   = '#ffffff';
    ctx.fill();
    ctx.strokeStyle = '#222';
    ctx.lineWidth   = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(sx, sy, 4, 0, 2 * Math.PI);
    ctx.fillStyle = this.pathColor;
    ctx.fill();
  }

  // Title sign and converges/diverges label — both flip with actual state
  _drawLabels(ctx) {
    const converging  = this.isConverging();
    const color       = converging ? '#16a34a' : '#dc2626';
    const titleText   = converging ? '\u03b7 < 2/S' : '\u03b7 > 2/S';
    const statusText  = converging ? 'converges' : 'diverges';

    ctx.save();
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'top';

    ctx.font      = '600 18px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = color;
    ctx.fillText(titleText, this.W / 2, 10);

    ctx.font      = '500 15px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = color;
    ctx.fillText(statusText, this.W / 2, 34);

    ctx.restore();
  }
}

// ============================================================================
// GDQuadPair — orchestrator: two panels, same η, same θ₀, different S
// ============================================================================

export class GDQuadPair {
  constructor(cfg) {
    const leftCanvas  = document.getElementById(cfg.leftCanvasId);
    const rightCanvas = document.getElementById(cfg.rightCanvasId);
    if (!leftCanvas || !rightCanvas) return;

    this.S_conv        = cfg.S_conv  ?? 1.3;
    this.S_div         = cfg.S_div   ?? 2.0;
    this.eta           = cfg.eta     ?? 1.2;
    this.nSteps        = cfg.nSteps  ?? 7;
    this.defaultEta    = cfg.eta     ?? 1.2;
    this.defaultTheta0 = cfg.theta0  ?? -0.6;
    this.theta0        = this.defaultTheta0;

    const vp = cfg.viewport ?? {
      thetaMin: -1.5, thetaMax: 1.5, lossMin: -0.2, lossMax: 1.5
    };
    this.vp = vp;

    this.left = new GDQuadPanel(leftCanvas, {
      S: this.S_conv, eta: this.eta, theta0: this.theta0,
      nSteps: this.nSteps, viewport: { ...vp },
    });

    this.right = new GDQuadPanel(rightCanvas, {
      S: this.S_div, eta: this.eta, theta0: this.theta0,
      nSteps: this.nSteps, viewport: { ...vp },
    });

    this._wireEtaSlider(cfg.etaSliderId, cfg.etaValueId);
    this._wireResetButton(cfg.resetButtonId);
    this._bindDrag(leftCanvas,  this.left);
    this._bindDrag(rightCanvas, this.right);

    this._drawAll();
  }

  _wireEtaSlider(sliderId, valueId) {
    if (!sliderId) return;
    const el  = document.getElementById(sliderId);
    const val = document.getElementById(valueId);
    if (!el) return;
    this.etaSlider = el;
    this.etaValue  = val;

    el.addEventListener('input', () => {
      this.eta = parseFloat(el.value);
      this.left.eta  = this.eta;
      this.right.eta = this.eta;
      if (val) val.textContent = this.eta.toFixed(3);
      this._drawAll();
    });
    if (val) val.textContent = this.eta.toFixed(3);
  }

  _wireResetButton(buttonId) {
    if (!buttonId) return;
    const btn = document.getElementById(buttonId);
    if (!btn) return;

    btn.addEventListener('click', () => {
      this.eta           = this.defaultEta;
      this.left.eta      = this.eta;
      this.right.eta     = this.eta;
      if (this.etaSlider) this.etaSlider.value        = this.eta;
      if (this.etaValue)  this.etaValue.textContent   = this.eta.toFixed(3);
      this._setTheta0(this.defaultTheta0);
      this._drawAll();
    });
  }

  _setTheta0(theta) {
    const { thetaMin, thetaMax } = this.vp;
    this.theta0        = Math.max(thetaMin + 0.05, Math.min(thetaMax - 0.05, theta));
    this.left.theta0   = this.theta0;
    this.right.theta0  = this.theta0;
  }

  _drawAll() {
    this.left.draw();
    this.right.draw();
  }

  _bindDrag(canvas, panel) {
    let dragging = false;

    const getPos = (e) => {
      const rect = canvas.getBoundingClientRect();
      const cx = e.touches ? e.touches[0].clientX : e.clientX;
      const cy = e.touches ? e.touches[0].clientY : e.clientY;
      return [cx - rect.left, cy - rect.top];
    };

    const onDown = (e) => {
      const [px, py] = getPos(e);
      if (panel.isNearStart(px, py)) { dragging = true; canvas.style.cursor = 'grabbing'; e.preventDefault(); }
    };

    const onMove = (e) => {
      const [px, py] = getPos(e);
      if (dragging) {
        this._setTheta0(panel.px2t(px));
        this._drawAll();
        e.preventDefault();
      } else {
        canvas.style.cursor = panel.isNearStart(px, py) ? 'grab' : 'default';
      }
    };

    const onUp = () => { dragging = false; canvas.style.cursor = 'default'; };

    canvas.addEventListener('mousedown',  onDown);
    canvas.addEventListener('mousemove',  onMove);
    canvas.addEventListener('mouseup',    onUp);
    canvas.addEventListener('mouseleave', onUp);
    canvas.addEventListener('touchstart', onDown, { passive: false });
    canvas.addEventListener('touchmove',  onMove, { passive: false });
    canvas.addEventListener('touchend',   onUp);
  }
}
