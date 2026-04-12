// ============================================================================
// gd_quadratic2d.js — Interactive 2-D quadratic GD visualisation
// ============================================================================
//
// Loss:  L(w) = ½ (λ₁ w̃₁² + λ₂ w̃₂²)   where w̃ = Rᵀ w, R = rotation(angle)
//
// Rendered as:
//   • filled + stroked elliptical contour lines
//   • GD trajectory with dots (red, fading)
//   • draggable starting point (large white-ringed dot)
//   • two eigenvector arrows (q₁, q₂) in a fixed corner, rotated with R
//
// Sliders control λ₁ (x-sharpness), λ₂ (y-sharpness), rotation angle, η.
//
// Public API:
//   new GDQuadratic2D(cfg)
//
// cfg:
//   canvasId      string
//   lam1SliderId, lam1ValueId   string
//   lam2SliderId, lam2ValueId   string
//   etaSliderId,  etaValueId    string
//   resetButtonId string
//   defaults: { lam1, lam2, angleDeg, eta, w0: [x,y] }
//
// angleDeg is fixed at construction time (no slider) — set via defaults.
//
// No external dependencies — pure Canvas 2D.
// ============================================================================

export class GDQuadratic2D {
  constructor(cfg) {
    this.canvas = document.getElementById(cfg.canvasId);
    if (!this.canvas) return;

    this.cfg = cfg;

    // Live params — initialised from defaults
    const d = cfg.defaults;
    this.lam1     = d.lam1     ?? 8.0;
    this.lam2     = d.lam2     ?? 0.5;
    this.angleDeg = d.angleDeg ?? 60;
    this.eta      = d.eta      ?? 0.27;
    this.w0       = (d.w0 ?? [0.3, 1.6]).slice();

    // Saved for reset
    this._defaults = {
      lam1: this.lam1, lam2: this.lam2,
      angleDeg: this.angleDeg, eta: this.eta,
      w0: this.w0.slice(),
    };

    this._setupCanvas();
    this._wireSliders();
    this._wireReset();
    this._bindDrag();
    this.draw();
  }

  // ---- HiDPI setup ----

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

  // ---- Rotation helpers ----

  _R() {
    const t = this.angleDeg * Math.PI / 180;
    const c = Math.cos(t), s = Math.sin(t);
    return [[c, -s], [s, c]];
  }

  // Rotate vector [x,y] by matrix [[a,b],[c,d]]
  _rot([a, b], [[r00, r01], [r10, r11]]) {
    return [r00 * a + r01 * b, r10 * a + r11 * b];
  }

  // ---- Coordinate transforms ----
  // Viewport: xMin..xMax, yMin..yMax mapped to canvas pixels

  get _vp() {
    const aspect = this.W / this.H;
    const boundY = 3.5;
    const boundX = boundY * aspect;
    return { xMin: -boundX, xMax: boundX, yMin: -boundY, yMax: boundY };
  }

  _w2p(wx, wy) {
    const { xMin, xMax, yMin, yMax } = this._vp;
    return [
      ((wx - xMin) / (xMax - xMin)) * this.W,
      ((yMax - wy) / (yMax - yMin)) * this.H,
    ];
  }

  _p2w(px, py) {
    const { xMin, xMax, yMin, yMax } = this._vp;
    return [
      xMin + (px / this.W) * (xMax - xMin),
      yMax - (py / this.H) * (yMax - yMin),
    ];
  }

  // ---- Loss in eigenbasis ----

  _loss(w) {
    // w̃ = Rᵀ w
    const R = this._R();
    // Rᵀ = [[R[0][0], R[1][0]], [R[0][1], R[1][1]]]
    const w0t = R[0][0] * w[0] + R[1][0] * w[1];
    const w1t = R[0][1] * w[0] + R[1][1] * w[1];
    return 0.5 * (this.lam1 * w0t * w0t + this.lam2 * w1t * w1t);
  }

  _gradient(w) {
    // ∇L = R Λ Rᵀ w  where Λ = diag(λ₁, λ₂)
    const R = this._R();
    const w0t = R[0][0] * w[0] + R[1][0] * w[1];
    const w1t = R[0][1] * w[0] + R[1][1] * w[1];
    // Λ Rᵀ w
    const lw0 = this.lam1 * w0t;
    const lw1 = this.lam2 * w1t;
    // R (Λ Rᵀ w)
    return [
      R[0][0] * lw0 + R[0][1] * lw1,
      R[1][0] * lw0 + R[1][1] * lw1,
    ];
  }

  // ---- GD trajectory ----
  // Run until 3 consecutive steps are fully outside the viewport, or
  // the point converges to near-zero, or a hard cap of 300 steps.

  _runGD() {
    const { xMin, xMax, yMin, yMax } = this._vp;
    const isOutOfBounds = (w) =>
      w[0] < xMin || w[0] > xMax || w[1] < yMin || w[1] > yMax;

    let w = this.w0.slice();
    const traj = [w.slice()];
    let consecutiveOOB = 0;
    const maxSteps = 300;

    for (let t = 0; t < maxSteps; t++) {
      const grad = this._gradient(w);
      w = [w[0] - this.eta * grad[0], w[1] - this.eta * grad[1]];
      traj.push(w.slice());

      if (!isFinite(w[0]) || !isFinite(w[1])) break;

      // Converged close enough to origin
      if (Math.hypot(w[0], w[1]) < 1e-6) break;

      if (isOutOfBounds(w)) {
        consecutiveOOB++;
        if (consecutiveOOB >= 3) break;
      } else {
        consecutiveOOB = 0;
      }
    }

    return traj;
  }

  // ---- Full draw ----

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.W, this.H);

    this._drawContours(ctx);
    const traj = this._runGD();
    this._drawTrajectory(ctx, traj);
    this._drawEigenvectors(ctx);
  }

  // ---- Contours ----
  // Draw filled + stroked elliptical iso-loss curves.
  // Each contour is sampled by sweeping angle and finding the radius where
  // L(R · [r cosφ, r sinφ]) = level  (i.e. in eigenbasis it's just an ellipse).

  _drawContours(ctx) {
    const levels    = [0.05, 0.2, 0.6, 1.5, 3.5, 7, 12, 20];
    const fillAlphas = [0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03];

    // Build paths once, reuse for fill and stroke
    const paths = levels.map(l => this._contourPath(l));

    // Fill from outermost inward so inner ellipses paint on top
    for (let li = levels.length - 1; li >= 0; li--) {
      if (!paths[li]) continue;
      ctx.save();
      ctx.fillStyle = `rgba(59, 130, 246, ${fillAlphas[li] ?? 0.04})`;
      ctx.fill(paths[li]);
      ctx.restore();
    }

    // Stroke all contours
    ctx.save();
    ctx.strokeStyle = '#999';
    ctx.lineWidth   = 0.9;
    for (let li = 0; li < levels.length; li++) {
      if (!paths[li]) continue;
      ctx.stroke(paths[li]);
    }
    ctx.restore();
  }

  // Returns a Path2D for the iso-loss ellipse at the given level.
  // In eigenbasis the ellipse is:  λ₁/2 · u² + λ₂/2 · v² = level
  //  → u = sqrt(2·level/λ₁)·cosφ,  v = sqrt(2·level/λ₂)·sinφ
  // Then rotate by R to get world coords.

  _contourPath(level) {
    if (level <= 0) return null;
    const ra = Math.sqrt(2 * level / this.lam1);  // semi-axis along q₁
    const rb = Math.sqrt(2 * level / this.lam2);  // semi-axis along q₂
    if (!isFinite(ra) || !isFinite(rb)) return null;

    const R    = this._R();
    const nPts = 256;
    const path = new Path2D();

    for (let i = 0; i <= nPts; i++) {
      const phi = (i / nPts) * 2 * Math.PI;
      const u   = ra * Math.cos(phi);
      const v   = rb * Math.sin(phi);
      // world coords: w = R [u, v]
      const wx  = R[0][0] * u + R[0][1] * v;
      const wy  = R[1][0] * u + R[1][1] * v;
      const [px, py] = this._w2p(wx, wy);
      if (i === 0) path.moveTo(px, py); else path.lineTo(px, py);
    }
    path.closePath();
    return path;
  }

  // ---- Trajectory ----

  _drawTrajectory(ctx, traj) {
    if (traj.length < 1) return;

    const dotColor = '#cc3333';

    // Line
    if (traj.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(204,51,51,0.5)';
      ctx.lineWidth   = 2;
      const [x0, y0] = this._w2p(traj[0][0], traj[0][1]);
      ctx.moveTo(x0, y0);
      for (let i = 1; i < traj.length; i++) {
        const [px, py] = this._w2p(traj[i][0], traj[i][1]);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Step dots (skip index 0 — drawn as start handle)
    for (let i = 1; i < traj.length; i++) {
      const [px, py] = this._w2p(traj[i][0], traj[i][1]);
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.fillStyle   = dotColor;
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth   = 0.9;
      ctx.stroke();
    }

    // Start dot — large with white ring (draggable)
    const [sx, sy] = this._w2p(this.w0[0], this.w0[1]);
    ctx.beginPath();
    ctx.arc(sx, sy, 10, 0, 2 * Math.PI);
    ctx.fillStyle   = dotColor;
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth   = 1.8;
    ctx.stroke();
  }

  // ---- Eigenvector arrows ----
  // Arrow origin is fixed in eigenbasis at (-0.5, -2.8), rotated to world.
  // Arrow tips are at origin + R·[arrowLen, 0] and origin + R·[0, arrowLen].

  _drawEigenvectors(ctx) {
    const R         = this._R();
    const arrowLen  = 1.5;
    const originEig = [-0.5, -2.8];

    // Origin in world coords
    const oWorld = this._rot(originEig, R);
    const [ox, oy] = this._w2p(oWorld[0], oWorld[1]);

    // q₁ tip
    const q1Eig  = [originEig[0] + arrowLen, originEig[1]];
    const q1World = this._rot(q1Eig, R);
    const [q1x, q1y] = this._w2p(q1World[0], q1World[1]);

    // q₂ tip
    const q2Eig  = [originEig[0], originEig[1] + arrowLen];
    const q2World = this._rot(q2Eig, R);
    const [q2x, q2y] = this._w2p(q2World[0], q2World[1]);

    this._arrowLine(ctx, ox, oy, q1x, q1y);
    this._arrowLine(ctx, ox, oy, q2x, q2y);

    // Labels — offset slightly beyond arrow tip in the same direction
    const labelOffset = 0.3;

    const q1LblEig  = [originEig[0] + arrowLen + labelOffset, originEig[1]];
    const q1LblWorld = this._rot(q1LblEig, R);
    const [lq1x, lq1y] = this._w2p(q1LblWorld[0], q1LblWorld[1]);

    const q2LblEig  = [originEig[0], originEig[1] + arrowLen + labelOffset];
    const q2LblWorld = this._rot(q2LblEig, R);
    const [lq2x, lq2y] = this._w2p(q2LblWorld[0], q2LblWorld[1]);

    ctx.save();
    ctx.font      = 'bold 15px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = '#333';

    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('q₁', lq1x, lq1y);
    ctx.fillText('q₂', lq2x, lq2y);
    ctx.restore();
  }

  _arrowLine(ctx, x1, y1, x2, y2) {
    const dx  = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 2) return;

    const headLen = 10;
    const angle   = Math.atan2(dy, dx);

    ctx.save();
    ctx.strokeStyle = '#333';
    ctx.fillStyle   = '#333';
    ctx.lineWidth   = 2.5;

    // Shaft
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2 - (dx / len) * headLen * 0.6, y2 - (dy / len) * headLen * 0.6);
    ctx.stroke();

    // Head
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headLen * Math.cos(angle - 0.35), y2 - headLen * Math.sin(angle - 0.35));
    ctx.lineTo(x2 - headLen * Math.cos(angle + 0.35), y2 - headLen * Math.sin(angle + 0.35));
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }

  // ---- Sliders ----

  _wireSliders() {
    const wire = (sliderId, valueId, param, decimals) => {
      const el  = document.getElementById(sliderId);
      const val = document.getElementById(valueId);
      if (!el) return;
      // Set initial display (HTML can post-process, e.g. append '°')
      if (val) val.textContent = this[param].toFixed(decimals);
      el.addEventListener('input', () => {
        this[param] = parseFloat(el.value);
        if (val) val.textContent = this[param].toFixed(decimals);
        this.draw();
      });
    };

    const c = this.cfg;
    wire(c.lam1SliderId,  c.lam1ValueId,  'lam1',  1);
    wire(c.lam2SliderId,  c.lam2ValueId,  'lam2',  2);
    wire(c.etaSliderId,   c.etaValueId,   'eta',   3);
  }

  // ---- Reset ----

  _wireReset() {
    const btn = document.getElementById(this.cfg.resetButtonId);
    if (!btn) return;
    btn.addEventListener('click', () => {
      const d = this._defaults;
      this.lam1     = d.lam1;
      this.lam2     = d.lam2;
      this.angleDeg = d.angleDeg;
      this.eta      = d.eta;
      this.w0       = d.w0.slice();

      // Sync slider positions and value displays
      const sync = (sliderId, valueId, val, decimals) => {
        const el  = document.getElementById(sliderId);
        const vel = document.getElementById(valueId);
        if (el)  el.value        = val;
        if (vel) vel.textContent = val.toFixed(decimals);
      };
      const c = this.cfg;
      sync(c.lam1SliderId,  c.lam1ValueId,  this.lam1,  1);
      sync(c.lam2SliderId,  c.lam2ValueId,  this.lam2,  2);
      sync(c.etaSliderId,   c.etaValueId,   this.eta,   3);

      this.draw();
    });
  }

  // ---- Drag ----

  _isNearStart(px, py) {
    const [sx, sy] = this._w2p(this.w0[0], this.w0[1]);
    return Math.hypot(px - sx, py - sy) < 16;
  }

  _bindDrag() {
    const canvas = this.canvas;
    let dragging = false;

    const getPos = (e) => {
      const rect = canvas.getBoundingClientRect();
      const cx = e.touches ? e.touches[0].clientX : e.clientX;
      const cy = e.touches ? e.touches[0].clientY : e.clientY;
      return [cx - rect.left, cy - rect.top];
    };

    canvas.addEventListener('mousedown', (e) => {
      const [px, py] = getPos(e);
      if (this._isNearStart(px, py)) { dragging = true; e.preventDefault(); }
    });

    canvas.addEventListener('mousemove', (e) => {
      const [px, py] = getPos(e);
      if (dragging) {
        this.w0 = this._p2w(px, py);
        this.draw();
        e.preventDefault();
      } else {
        canvas.style.cursor = this._isNearStart(px, py) ? 'grab' : 'default';
      }
    });

    canvas.addEventListener('mouseup',    () => { dragging = false; canvas.style.cursor = 'default'; });
    canvas.addEventListener('mouseleave', () => { dragging = false; canvas.style.cursor = 'default'; });

    canvas.addEventListener('touchstart', (e) => {
      const [px, py] = getPos(e);
      if (this._isNearStart(px, py)) { dragging = true; e.preventDefault(); }
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
      if (!dragging) return;
      const [px, py] = getPos(e);
      this.w0 = this._p2w(px, py);
      this.draw();
      e.preventDefault();
    }, { passive: false });

    canvas.addEventListener('touchend', () => { dragging = false; });
  }
}
