// experiments2.js — 3D trajectory viewer, Canvas 2D

// ---- Config ----
const DATA_PATH   = 'runs/cifar10_sharp/gradient_displacement_data_old.json';
const MAP_X       = 'residual_direction';
const MAP_Y       = 'sharpness_gradient';  // negated for display
const MAP_Z       = 'top_eigenvector';     // negated for display
const AXIS_LABELS = ['residual', 'sharp∇', 'top eig'];
const AXIS_COLORS = ['#c01818', '#0a8c25', '#1830cc'];
const LINE_WIDTH  = 2.5;   // curve thickness in CSS pixels
const PLANES = [
  { axis: 'y', value: 'mid', color: 'rgba(20,160,60,0.15)', gridColor: 'rgba(20,160,60,0.35)', gridStep: 0.2 },
];

// ---- Data ----
function parseDataset(json) {
  const xs     = json.displacements[MAP_X].cumulative;
  const ys     = json.displacements[MAP_Y].cumulative;
  const zs     = json.displacements[MAP_Z].cumulative;
  const epochs = json.epochs;
  const n      = Math.min(xs.length, ys.length, zs.length, epochs.length);

  let div = 0;
  for (let i = 0; i < n; i++)
    div = Math.max(div, Math.abs(xs[i]), Math.abs(ys[i]), Math.abs(zs[i]));
  if (div === 0) div = 1;

  const points = [];
  for (let i = 0; i < n; i++)
    points.push({ x: xs[i]/div, y: -ys[i]/div, z: -zs[i]/div, epoch: epochs[i] });

  return { points, epochs };
}

// ---- Matrix math ----
function matMul(a, b) {
  const o = new Array(16).fill(0);
  for (let r=0; r<4; r++)
    for (let c=0; c<4; c++)
      for (let k=0; k<4; k++)
        o[r + c*4] += a[r + k*4] * b[k + c*4];
  return o;
}

function matPerspective(fov, aspect) {
  const f = 1 / Math.tan(fov / 2);
  const near = 0.1, far = 100, nf = 1 / (near - far);
  return [
    f/aspect, 0, 0,              0,
    0,        f, 0,              0,
    0,        0, (far+near)*nf, -1,
    0,        0, 2*far*near*nf,  0,
  ];
}

function matRotX(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1];
}

function matRotY(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1];
}

function matTranslate(x, y, z) {
  return [1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1];
}

function project(wx, wy, wz, mvp, W, H) {
  const x = mvp[0]*wx + mvp[4]*wy + mvp[8]*wz  + mvp[12];
  const y = mvp[1]*wx + mvp[5]*wy + mvp[9]*wz  + mvp[13];
  const w = mvp[3]*wx + mvp[7]*wy + mvp[11]*wz + mvp[15];
  if (w <= 0) return null;
  return {
    sx: ( x/w * 0.5 + 0.5) * W,
    sy: (-y/w * 0.5 + 0.5) * H,
  };
}

// ---- Main ----
async function init() {
  const canvas      = document.getElementById('curve3d-canvas');
  const epochSlider = document.getElementById('epoch-slider');
  const epochVal    = document.getElementById('epoch-value');
  const rotXSlider  = document.getElementById('rotx-slider');
  const rotXVal     = document.getElementById('rotx-value');
  const showAxes    = document.getElementById('check-axes');
  const showPlanes  = document.getElementById('check-planes');
  const loadStatus  = document.getElementById('load-status');
  const ctx         = canvas.getContext('2d');

  const dpr = window.devicePixelRatio || 1;
  function resize() {
    const r = canvas.getBoundingClientRect();
    canvas.width  = Math.round(r.width  * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  resize();
  window.addEventListener('resize', resize);

  // Camera state
  let ax   = -10 * Math.PI / 180;
  let zoom = 2.0;  // default closer in
  let panX = 0, panY = 0;
  let autoX = 0;
  let fixedView = false;  // when true, view doesn't follow epoch tip

  let dataset    = null;
  let currentPts = [];
  let axisExtent = null;

  // ---- Load data ----
  try {
    loadStatus.textContent = 'Loading data…';
    const resp = await fetch(DATA_PATH);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const raw = await resp.json();
    console.log(JSON.stringify(Object.keys(raw.displacements))); // ← add this
    dataset = parseDataset(raw);

    const n = dataset.points.length;
    epochSlider.min   = 1;
    epochSlider.max   = n;
    epochSlider.value = n;
    epochVal.textContent = dataset.epochs[n - 1];
    loadStatus.textContent = `Loaded ${n} epochs (${dataset.epochs[0]}–${dataset.epochs[n-1]})`;

    let x0=0, x1=0, y0=0, y1=0, z0=0, z1=0;
    for (const p of dataset.points) {
      x0=Math.min(x0,p.x); x1=Math.max(x1,p.x);
      y0=Math.min(y0,p.y); y1=Math.max(y1,p.y);
      z0=Math.min(z0,p.z); z1=Math.max(z1,p.z);
    }
    const pad = v => v >= 0 ? v*1.15 + 0.05 : v*1.15 - 0.05;
    axisExtent = { x0:pad(x0), x1:pad(x1), y0:pad(y0), y1:pad(y1), z0:pad(z0), z1:pad(z1) };

    rebuild();
  } catch (e) {
    loadStatus.textContent = `Failed to load: ${e.message}`;
    console.error(e); return;
  }

  // ---- Rebuild ----
  function rebuild() {
    if (!dataset) return;
    const n = parseInt(epochSlider.value);
    epochVal.textContent = dataset.epochs[n - 1];
    currentPts = dataset.points.slice(0, n);
    if (!fixedView) autoX = -currentPts[currentPts.length - 1].x;
  }

  // ---- Slider sync ----
  function syncSliders() {
    rotXSlider.value = Math.round(ax * 180/Math.PI);
    rotXVal.textContent = rotXSlider.value + '°';
  }
  syncSliders();

  // ---- Events ----
  let dragging = false, lx = 0;
  canvas.addEventListener('mousedown', e => { dragging = true; lx = e.clientX; });
  canvas.addEventListener('mouseup',   () => { dragging = false; });
  canvas.addEventListener('mouseleave',() => { dragging = false; });
  canvas.addEventListener('mousemove', e => {
    if (!dragging) return;
    panX += (e.clientX - lx) * zoom * 0.002;
    lx = e.clientX;
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    zoom = Math.max(0.05, Math.min(20, zoom * (1 + e.deltaY * 0.002)));
  }, { passive: false });

  rotXSlider.addEventListener('input', () => {
    ax = parseFloat(rotXSlider.value) * Math.PI / 180;
    rotXVal.textContent = rotXSlider.value + '°';
  });
  epochSlider.addEventListener('input', rebuild);

  const fixBtn = document.getElementById('fix-btn');
  fixBtn.addEventListener('click', () => {
    fixedView = !fixedView;
    fixBtn.textContent = fixedView ? 'Unfix view' : 'Fix view';
    fixBtn.style.fontWeight = fixedView ? '600' : 'normal';
  });

  // ---- Draw helpers ----
  function line3d(x0,y0,z0, x1,y1,z1, mvp, W, H) {
    const a = project(x0,y0,z0, mvp,W,H);
    const b = project(x1,y1,z1, mvp,W,H);
    if (!a || !b) return;
    ctx.beginPath();
    ctx.moveTo(a.sx, a.sy);
    ctx.lineTo(b.sx, b.sy);
    ctx.stroke();
  }

  function quad3d(corners, mvp, W, H) {
    const pts = corners.map(([x,y,z]) => project(x,y,z, mvp,W,H));
    if (pts.some(p => !p)) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].sx, pts[0].sy);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i].sx, pts[i].sy);
    ctx.closePath();
    ctx.fill();
  }

  // ---- Render loop ----
  function render() {
    requestAnimationFrame(render);
    if (!dataset) return;

    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, W, H);

    const mvp = matMul(
      matPerspective(Math.PI / 4, W / H),
      matMul(matTranslate(autoX + panX, panY, -zoom), matRotX(ax))
    );

    const { x0,x1, y0,y1, z0,z1 } = axisExtent;

    // Planes + gridlines
    if (showPlanes && showPlanes.checked) {
      for (const pl of PLANES) {
        const mid = { x:(x0+x1)/2, y:(y0+y1)/2, z:(z0+z1)/2 };
        const v   = pl.value === 'mid' ? mid[pl.axis] : pl.value;

        // Filled quad
        ctx.fillStyle = pl.color;
        if      (pl.axis === 'y') quad3d([[x0,v,z0],[x1,v,z0],[x1,v,z1],[x0,v,z1]], mvp,W,H);
        else if (pl.axis === 'x') quad3d([[v,y0,z0],[v,y1,z0],[v,y1,z1],[v,y0,z1]], mvp,W,H);
        else                      quad3d([[x0,y0,v],[x1,y0,v],[x1,y1,v],[x0,y1,v]], mvp,W,H);

        // Grid lines
        if (pl.gridColor && pl.gridStep) {
          ctx.strokeStyle = pl.gridColor;
          ctx.lineWidth   = 0.5 * dpr;
          ctx.globalAlpha = 1.0;
          const step = pl.gridStep;

          if (pl.axis === 'y') {
            // Lines along X at fixed Z intervals
            for (let gz = Math.ceil(z0/step)*step; gz <= z1; gz += step)
              line3d(x0,v,gz, x1,v,gz, mvp,W,H);
            // Lines along Z at fixed X intervals
            for (let gx = Math.ceil(x0/step)*step; gx <= x1; gx += step)
              line3d(gx,v,z0, gx,v,z1, mvp,W,H);
          } else if (pl.axis === 'x') {
            for (let gz = Math.ceil(z0/step)*step; gz <= z1; gz += step)
              line3d(v,y0,gz, v,y1,gz, mvp,W,H);
            for (let gy = Math.ceil(y0/step)*step; gy <= y1; gy += step)
              line3d(v,gy,z0, v,gy,z1, mvp,W,H);
          } else {
            for (let gx = Math.ceil(x0/step)*step; gx <= x1; gx += step)
              line3d(gx,y0,v, gx,y1,v, mvp,W,H);
            for (let gy = Math.ceil(y0/step)*step; gy <= y1; gy += step)
              line3d(x0,gy,v, x1,gy,v, mvp,W,H);
          }
        }
      }
    }

    // Axes
    if (showAxes && showAxes.checked) {
      ctx.lineWidth = dpr;

      ctx.globalAlpha = 0.25;
      ctx.strokeStyle = AXIS_COLORS[0]; line3d(x0,0,0, 0,0,0, mvp,W,H);
      ctx.strokeStyle = AXIS_COLORS[1]; line3d(0,y0,0, 0,0,0, mvp,W,H);
      ctx.strokeStyle = AXIS_COLORS[2]; line3d(0,0,z0, 0,0,0, mvp,W,H);

      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = AXIS_COLORS[0]; line3d(0,0,0, x1,0,0, mvp,W,H);
      ctx.strokeStyle = AXIS_COLORS[1]; line3d(0,0,0, 0,y1,0, mvp,W,H);
      ctx.strokeStyle = AXIS_COLORS[2]; line3d(0,0,0, 0,0,z1, mvp,W,H);

      ctx.font = `bold ${13 * dpr}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      [[x1,0,0],[0,y1,0],[0,0,z1]].forEach(([tx,ty,tz], i) => {
        const s = project(tx,ty,tz, mvp,W,H);
        if (!s) return;
        ctx.strokeStyle = 'rgba(255,255,255,0.85)';
        ctx.lineWidth   = 4 * dpr;
        ctx.strokeText(AXIS_LABELS[i], s.sx, s.sy);
        ctx.fillStyle = AXIS_COLORS[i];
        ctx.fillText(AXIS_LABELS[i], s.sx, s.sy);
      });
    }

    // Curve
    if (currentPts.length > 1) {
      const N       = currentPts.length;
      const SEG_LEN = 8;
      ctx.lineWidth   = LINE_WIDTH * dpr;
      ctx.lineCap     = 'round';
      ctx.lineJoin    = 'round';
      ctx.globalAlpha = 1.0;

      for (let i = 0; i < N - 1; i += SEG_LEN) {
        const end = Math.min(i + SEG_LEN + 1, N);
        const t   = Math.pow((i + SEG_LEN / 2) / (N - 1), 1.5);
        ctx.strokeStyle = `rgb(0, ${Math.round(t*64)}, ${Math.round(t*217+13)})`;

        ctx.beginPath();
        let started = false;
        for (let j = i; j < end; j++) {
          const s = project(currentPts[j].x, currentPts[j].y, currentPts[j].z, mvp,W,H);
          if (!s) continue;
          if (!started) { ctx.moveTo(s.sx, s.sy); started = true; }
          else            ctx.lineTo(s.sx, s.sy);
        }
        ctx.stroke();
      }
    }
  }

  render();
}

init();
