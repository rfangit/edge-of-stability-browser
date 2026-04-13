// experiments2.js — 3D trajectory viewer + 2D projections

// ---- Config ----
const DATA_PATH   = 'runs/cifar10_sharp/gradient_displacement_data.json';
const AXIS_LABELS = ['∇L ⊥ v₁ ', 'sharpness λ₁', 'eigenvector v₁ displacement'];
const AXIS_COLORS = ['#c01818', '#0a8c25', '#1830cc'];
const LINE_WIDTH  = 2.0;
const GHOST_ALPHA = 0.12;
const SHARPNESS_CRITICAL = 100;  // raw eigenvalue value of the critical threshold

// ---- Data ----
// x = displacements.residual_direction.cumulative
// y = sharpness.values  (raw top eigenvalue — top 2D plot)
// z = displacements.top_eigenvector.cumulative  (bottom 2D plot)
// All three normalised by the same max-abs for equal-scale 3D view.
function parseDataset(json) {
  const xs    = json.displacements.residual_direction.cumulative;
  const ys    = json.sharpness.values;
  const zs    = json.displacements.top_eigenvector.cumulative;
  const steps = json.epochs;
  const n     = Math.min(xs.length, ys.length, zs.length, steps.length);

  let div = 0;
  for (let i = 0; i < n; i++)
    div = Math.max(div, Math.abs(xs[i]), Math.abs(ys[i]), Math.abs(zs[i]));
  if (div === 0) div = 1;

  const points = [];
  const ox = xs[0]/div, oy = ys[0]/div, oz = zs[0]/div;
  
  // Sharpness scale factor for 3D visualization only
  const SHARPNESS_SCALE_3D = 2.0;  // Adjust this value
  
  for (let i = 0; i < n; i++) {
    points.push({ 
      x: xs[i]/div - ox, 
      y: ys[i]/div - oy,           // Original y for 2D plots
      z: zs[i]/div - oz, 
      y3d: (ys[i]/div - oy) * SHARPNESS_SCALE_3D,  // Scaled y for 3D
      epoch: i 
    });
  }

  // Critical threshold for 2D (original scale)
  const criticalY = SHARPNESS_CRITICAL / div - oy;
  
  // Critical threshold for 3D (scaled)
  const criticalY3d = criticalY * SHARPNESS_SCALE_3D;

  // Create relative epochs starting from 0
  const relativeEpochs = steps.map((_, i) => i);
  
  return { points, epochs: relativeEpochs, criticalY, criticalY3d };
}  // ← Make sure this closing brace exists!

// Regions: boundary epoch indices (1-based). Adjust to match your data.
// Regions: boundary epoch indices (0-based, relative epochs)
const REGIONS = [
  { 
    label: 'progressive sharpening',  
    endIdx: 40,  
    color: '#a0c4ff', 
    text: 'Initial Progressive Sharpening: Sharpness steadily increases, approaching the critical threshold.' 
  },
  { 
    label: 'high sharpness',  
    endIdx: 160, 
    color: '#b9fbc0', 
    text: 'High Sharpness: Above the critical threshold, any oscillations along the eigenvector will grow.' 
  },
  { 
    label: 'oscillations',  
    endIdx: 182, 
    color: '#ffd6a5', 
    text: 'Oscillations: Oscillations begin to dominate the gradient here, and the sharpness reduction from the third order terms appear.' 
  },
  { 
    label: 'sharpness reduction',  
    endIdx: 205, 
    color: '#ffadad', 
    text: 'Sharpness Reduction: Large oscillations cause large reductions in sharpness.' 
  },
  { 
    label: 'oscillation reduction',  
    endIdx: 240, 
    color: '#e2b5ff', 
    text: 'Oscillation Reduction: Now that sharpness is below the critical threshold, oscillations slowly decay.' 
  },
  { 
    label: 'cycle repeats',  
    endIdx: null, 
    color: '#d0d0d0', 
    text: 'Cycle Repeats: Without oscillations, the model begins progressively sharpening again, repeating the cycle.' 
  },
];
// The critical threshold plane is built dynamically after data loads (see init)
// so it can use the normalised criticalY value from parseDataset.
let CRITICAL_PLANE = null;

// ---- Curve color (black → blue, weighted toward tip) ----
function curveColor(t) {
  const t2 = Math.pow(t, 1.5);
  return `rgb(0,${Math.round(t2*64)},${Math.round(t2*217+13)})`;
}

// ---- Matrix math (3D) ----
function matMul(a, b) {
  const o = new Array(16).fill(0);
  for (let r=0; r<4; r++)
    for (let c=0; c<4; c++)
      for (let k=0; k<4; k++)
        o[r+c*4] += a[r+k*4] * b[k+c*4];
  return o;
}
function matPerspective(fov, aspect) {
  const f = 1/Math.tan(fov/2), near=0.1, far=100, nf=1/(near-far);
  return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0];
}
function matRotX(a) { const c=Math.cos(a),s=Math.sin(a); return [1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1]; }
function matTranslate(x,y,z) { return [1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]; }

function project3d(wx,wy,wz,mvp,W,H) {
  const x=mvp[0]*wx+mvp[4]*wy+mvp[8]*wz+mvp[12];
  const y=mvp[1]*wx+mvp[5]*wy+mvp[9]*wz+mvp[13];
  const w=mvp[3]*wx+mvp[7]*wy+mvp[11]*wz+mvp[15];
  if (w<=0) return null;
  return { sx:(x/w*0.5+0.5)*W, sy:(-y/w*0.5+0.5)*H };
}

// ---- 2D projection helpers ----
// Map a data value from [dataMin,dataMax] to canvas pixels [pad, size-pad]
function makeScale(dataMin, dataMax, pixelSize, pad) {
  return v => pad + (v - dataMin) / (dataMax - dataMin) * (pixelSize - 2*pad);
}

// makeScaleY flips the y axis so larger values appear higher on screen
// (Canvas 2D has y increasing downward, so we reverse the mapping)
function makeScaleY(dataMin, dataMax, pixelSize, pad) {
  return v => (pixelSize - pad) - (v - dataMin) / (dataMax - dataMin) * (pixelSize - 2*pad);
}

// Draw a 2D curve on ctx, mapping data through scaleX/scaleY
function draw2dCurve(ctx, pts, getX, getY, scaleX, scaleY, alpha, color, lineWidth) {
  if (pts.length < 2) return;
  const SEG = 8;
  const N   = pts.length;
  ctx.lineWidth   = lineWidth;
  ctx.lineCap     = 'round';
  ctx.lineJoin    = 'round';
  ctx.globalAlpha = alpha;

  if (typeof color === 'string') {
    // Single solid color (used for ghost)
    ctx.strokeStyle = color;
    ctx.beginPath();
    let started = false;
    for (let i=0; i<N; i++) {
      const sx = scaleX(getX(pts[i])), sy = scaleY(getY(pts[i]));
      if (!started) { ctx.moveTo(sx,sy); started=true; } else ctx.lineTo(sx,sy);
    }
    ctx.stroke();
  } else {
    // Gradient color along the curve
    for (let i=0; i<N-1; i+=SEG) {
      const end = Math.min(i+SEG+1, N);
      const t   = (i + SEG/2) / (N-1);
      ctx.strokeStyle = curveColor(t);
      ctx.beginPath();
      let started = false;
      for (let j=i; j<end; j++) {
        const sx=scaleX(getX(pts[j])), sy=scaleY(getY(pts[j]));
        if (!started) { ctx.moveTo(sx,sy); started=true; } else ctx.lineTo(sx,sy);
      }
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1.0;
}

// ---- Main ----
async function init() {
  const canvas3d    = document.getElementById('canvas-3d');
  const canvasXY    = document.getElementById('canvas-xy');
  const canvasXZ    = document.getElementById('canvas-xz');
  const epochVal    = document.getElementById('epoch-value');
  const epochThumb  = document.getElementById('epoch-thumb');
  const epochFill   = document.getElementById('epoch-track-fill');
  const epochWrap   = document.getElementById('epoch-track-wrap');
  const rotXSlider  = document.getElementById('rotx-slider');
  const rotXVal     = document.getElementById('rotx-value');
  const zoomSlider  = document.getElementById('zoom-slider');
  const zoomVal     = document.getElementById('zoom-value');
  const loadStatus  = document.getElementById('load-status');

  const ctx3d = canvas3d.getContext('2d');
  const ctxXY = canvasXY.getContext('2d');
  const ctxXZ = canvasXZ.getContext('2d');

  const dpr = window.devicePixelRatio || 1;

  function resizeCanvas(c) {
    const r = c.getBoundingClientRect();
    c.width  = Math.round(r.width  * dpr);
    c.height = Math.round(r.height * dpr);
  }
  function resizeAll() { resizeCanvas(canvas3d); resizeCanvas(canvasXY); resizeCanvas(canvasXZ); }
  resizeAll();
  window.addEventListener('resize', resizeAll);

  // ---- Camera (3D only) ----
  let ax        = -10 * Math.PI / 180;
  let zoom      = 1.2;
  let panX      = 0, panY = 0;
  let autoX     = 0;
  let fixedView = false;

  // Custom epoch slider state
  let epochMin = 1, epochMax = 1, epochCur = 1;

  function setEpoch(n) {
    epochCur = Math.max(epochMin, Math.min(epochMax, Math.round(n)));
    const pct = (epochCur - epochMin) / (epochMax - epochMin);
    epochThumb.style.left = (pct * 100) + '%';
    epochFill.style.width = (pct * 100) + '%';
    rebuild();
  }

  function epochFromPointer(e) {
    const rect = epochWrap.getBoundingClientRect();
    const pct  = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    setEpoch(Math.round(epochMin + pct * (epochMax - epochMin)));  // Round to nearest integer
  }

  let epochDragging = false;
  epochWrap.addEventListener('mousedown', e => { epochDragging=true; epochFromPointer(e); });
  document.addEventListener('mouseup',   () => { epochDragging=false; });
  document.addEventListener('mousemove', e => { if (epochDragging) epochFromPointer(e); });
  // Clicking the region bar jumps to that epoch position
  document.getElementById('region-bar').addEventListener('mousedown', e => {
    epochDragging = true;
    // Region bar is same width as epochWrap, so use epochWrap for coordinate mapping
    epochFromPointer(e);
  });

  let dataset    = null;
  let currentPts = [];
  let ext        = null;
  let ext2d      = null;

  // Zoom log scale helpers — declared here so syncZoomSlider() can be called after load
  const ZOOM_MIN = 0.3, ZOOM_MAX = 2.0;
  function sliderToZoom(s) { return ZOOM_MIN * Math.pow(ZOOM_MAX/ZOOM_MIN, s/1000); }
  function zoomToSlider(z) { return Math.round(Math.log(z/ZOOM_MIN) / Math.log(ZOOM_MAX/ZOOM_MIN) * 1000); }
  function syncZoomSlider() {
    zoomSlider.value = zoomToSlider(zoom);
    zoomVal.textContent = zoom.toFixed(2);
  }

  // ---- Load data ----
  try {
    loadStatus.textContent = 'Loading data…';
    const resp = await fetch(DATA_PATH);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    dataset = parseDataset(await resp.json());

    const n = dataset.points.length;
    epochMin = 0;                    // Start at 0 instead of 1
    epochMax = n - 1;                // Max is n-1 for 0-indexed epochs
    epochVal.textContent = '0';      // Show 0 instead of actual epoch
    setEpoch(0);                     // Start at epoch 0
    loadStatus.textContent = '';  // Simplified message

    let x0=0,x1=0,y0=0,y1=0,z0=0,z1=0;
    for (const p of dataset.points) {
      x0=Math.min(x0,p.x); x1=Math.max(x1,p.x);
      y0=Math.min(y0,p.y); y1=Math.max(y1,p.y);
      z0=Math.min(z0,p.z); z1=Math.max(z1,p.z);
    }
    const pad = v => v>=0 ? v*1.15+0.05 : v*1.15-0.05;
    // ext: padded equal-scale extents for the 3D view
    ext = { x0:pad(x0),x1:pad(x1), y0:pad(y0),y1:pad(y1), z0:pad(z0),z1:pad(z1) };

    // ext2d: tight per-axis extents for 2D plots.
    // x: exact data range, no padding (starts at first point, ends at last).
    // y/z: tiny fixed margin of 2% of each axis's range to avoid clipping at boundary.
    let rx0=dataset.points[0].x, rx1=dataset.points[0].x;
    let ry0=dataset.points[0].y, ry1=dataset.points[0].y;
    let rz0=dataset.points[0].z, rz1=dataset.points[0].z;
    for (const p of dataset.points) {
      rx0=Math.min(rx0,p.x); rx1=Math.max(rx1,p.x);
      ry0=Math.min(ry0,p.y); ry1=Math.max(ry1,p.y);
      rz0=Math.min(rz0,p.z); rz1=Math.max(rz1,p.z);
    }
    const ym = (ry1-ry0)*0.02 || 0.01;
    const zm = (rz1-rz0)*0.02 || 0.01;
    ext2d = {
      x0: rx0,       x1: rx1,
      y0: ry0 - ym,  y1: ry1 + ym,
      z0: rz0 - zm,  z1: rz1 + zm,
    };

    rebuild();
    syncZoomSlider();
    buildRegionBar(dataset.points.length);

    // Build the critical threshold plane
    const PAD3D = 1.15;
    CRITICAL_PLANE = {
      y:         dataset.criticalY3d,  // ← Use scaled critical Y
      x0:        0,
      x1:        ext.x1 * PAD3D,
      z0:        ext.z0 * PAD3D,
      z1:        ext.z1 * PAD3D,
      color:     'rgba(200,40,40,0.12)',
      gridColor: 'rgba(200,40,40,0.3)',
      gridStep:  0.2,
    };
  } catch(e) {
    loadStatus.textContent = `Failed to load: ${e.message}`;
    console.error(e); return;
  }

  // ---- Rebuild ----
  function rebuild() {
    if (!dataset) return;
    const n = epochCur + 1;  // +1 because array is 0-indexed, epochCur is 0-indexed
    epochVal.textContent = epochCur;  // Show the relative epoch number
    currentPts = dataset.points.slice(0, n);
    if (!fixedView) autoX = -currentPts[currentPts.length-1].x;
    updateRegionBar(n);
  }

  // Build the region bar segments once (proportional widths based on index ranges)
  function buildRegionBar(total) {
    const bar = document.getElementById('region-bar');
    bar.innerHTML = '';
    REGIONS.forEach((r, i) => {
      const start = i === 0 ? 1 : (REGIONS[i-1].endIdx ?? total) + 1;
      const end   = r.endIdx ?? total;
      const width = (end - start + 1) / total * 100;
      const seg   = document.createElement('div');
      seg.className = 'region-segment';
      seg.id        = `region-seg-${i}`;
      seg.style.width      = width + '%';
      seg.style.background = r.color;
      seg.title = r.label;
      bar.appendChild(seg);
    });
  }

  // Highlight only the region containing the current epoch index
  function updateRegionBar(currentIdx) {
    if (!dataset) return;
    const total = dataset.points.length;
    let activeRegion = null;
    REGIONS.forEach((r, i) => {
      const start = i === 0 ? 1 : (REGIONS[i-1].endIdx ?? total) + 1;
      const end   = r.endIdx ?? total;
      const seg   = document.getElementById(`region-seg-${i}`);
      if (!seg) return;
      const active = currentIdx >= start && currentIdx <= end;
      seg.classList.toggle('inactive', !active);
      if (active) activeRegion = r;
    });
    const box = document.getElementById('region-text');
    if (box && activeRegion) {
      box.style.borderColor = activeRegion.color;
      box.querySelector('.region-text-label').textContent = activeRegion.label.toUpperCase();
      box.querySelector('.region-text-label').style.color = '#555';
      box.querySelector('.region-text-body').textContent  = activeRegion.text;
    }
  }

  // ---- Events ----
  rotXSlider.addEventListener('input', () => {
    ax = parseFloat(rotXSlider.value) * Math.PI/180;
    rotXVal.textContent = rotXSlider.value + '°';
  });
  zoomSlider.addEventListener('input', () => {
    zoom = sliderToZoom(parseFloat(zoomSlider.value));
    zoomVal.textContent = zoom.toFixed(2);
  });
  canvas3d.addEventListener('wheel', e => {
    e.preventDefault();
    zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom * (1 + e.deltaY * 0.002)));
    syncZoomSlider();
  }, {passive:false});

  let dragging=false, lx=0;
  canvas3d.addEventListener('mousedown', e=>{ dragging=true; lx=e.clientX; });
  canvas3d.addEventListener('mouseup',   ()=>{ dragging=false; });
  canvas3d.addEventListener('mouseleave',()=>{ dragging=false; });
  canvas3d.addEventListener('mousemove', e=>{
    if (!dragging) return;
    panX += (e.clientX-lx)*zoom*0.002; lx=e.clientX;
  });

  const fixBtn = document.getElementById('fix-btn');
  fixBtn.addEventListener('click', () => {
    fixedView = !fixedView;
    fixBtn.textContent      = fixedView ? 'Unfix view' : 'Fix view';
    fixBtn.style.fontWeight = fixedView ? '600' : 'normal';
  });

  // ---- 3D draw helpers ----
  function line3d(x0,y0,z0,x1,y1,z1,mvp,W,H) {
    const a=project3d(x0,y0,z0,mvp,W,H), b=project3d(x1,y1,z1,mvp,W,H);
    if (!a||!b) return;
    ctx3d.beginPath(); ctx3d.moveTo(a.sx,a.sy); ctx3d.lineTo(b.sx,b.sy); ctx3d.stroke();
  }
  function quad3d(corners,mvp,W,H) {
    const pts=corners.map(([x,y,z])=>project3d(x,y,z,mvp,W,H));
    if (pts.some(p=>!p)) return;
    ctx3d.beginPath(); ctx3d.moveTo(pts[0].sx,pts[0].sy);
    for (let i=1;i<4;i++) ctx3d.lineTo(pts[i].sx,pts[i].sy);
    ctx3d.closePath(); ctx3d.fill();
  }

  // ---- Render 2D plot (shared logic for XY and XZ) ----
  function render2d(ctx, canvas, allPts, curPts, getH, yLabel, yColor) {
    const W=canvas.width, H=canvas.height;
    const PAD = Math.round(36*dpr);  // padding for axes

    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);

    // scX maps data x → pixel x, with PAD pixels of margin on left and right.
    // scYown maps data y/z → pixel y, with PAD pixels of margin on top and bottom.
    // makeScale(dataMin, dataMax, pixelTotal, pad) maps dataMin→pad, dataMax→pixelTotal-pad.
    const scX    = makeScale(ext2d.x0, ext2d.x1, W, PAD);
    const yMin   = getH === 'y' ? ext2d.y0 : ext2d.z0;
    const yMax   = getH === 'y' ? ext2d.y1 : ext2d.z1;
    const scYown = makeScaleY(yMin, yMax, H, PAD);

    const getX = p => p.x;
    const getY = p => getH === 'y' ? p.y : p.z;

    // Region backgrounds and dividers
    if (allPts.length > 0) {
      const total = allPts.length;
      const curIdx = curPts.length;
      REGIONS.forEach((r, i) => {
        const startIdx = i === 0 ? 0 : (REGIONS[i-1].endIdx ?? total) - 1;
        const endIdx   = (r.endIdx ?? total) - 1;
        const x0px = scX(allPts[Math.min(startIdx, total-1)].x);
        const x1px = scX(allPts[Math.min(endIdx,   total-1)].x);

        // Active region: colored background
        const regionStart = i === 0 ? 1 : (REGIONS[i-1].endIdx ?? total) + 1;
        const regionEnd   = r.endIdx ?? total;
        const active = curIdx >= regionStart && curIdx <= regionEnd;
        if (active) {
          ctx.globalAlpha = 0.25;
          ctx.fillStyle   = r.color;
          ctx.fillRect(x0px, PAD, x1px - x0px, H - 2*PAD);
          ctx.globalAlpha = 1.0;
        }

        // Divider line at region boundary (not before first region)
        if (i > 0) {
          ctx.globalAlpha = 0.6;
          ctx.strokeStyle = '#333';
          ctx.lineWidth   = 1 * dpr;
          ctx.beginPath(); ctx.moveTo(x0px, PAD); ctx.lineTo(x0px, H-PAD); ctx.stroke();
          ctx.globalAlpha = 1.0;
        }
      });
    }

    // Ghost: full curve
    draw2dCurve(ctx, allPts, getX, getY, scX, scYown, GHOST_ALPHA, '#000033', LINE_WIDTH*dpr);

    // Current slice
    draw2dCurve(ctx, curPts, getX, getY, scX, scYown, 1.0, null, LINE_WIDTH*dpr);

    // Tip dot
    if (curPts.length > 0) {
      const tip = curPts[curPts.length-1];
      const sx  = scX(tip.x), sy = scYown(getY(tip));
      ctx.globalAlpha = 1.0;
      ctx.fillStyle   = '#ffee44';
      ctx.strokeStyle = '#333';
      ctx.lineWidth   = 1.5*dpr;
      ctx.beginPath(); ctx.arc(sx,sy,4*dpr,0,Math.PI*2); ctx.fill(); ctx.stroke();
    }

    // Vertical line at current x position
    if (curPts.length > 0) {
      const tipX = scX(curPts[curPts.length-1].x);
      ctx.globalAlpha = 0.3;
      ctx.strokeStyle = '#888';
      ctx.lineWidth   = 1*dpr;
      ctx.setLineDash([4*dpr, 4*dpr]);
      ctx.beginPath(); ctx.moveTo(tipX, PAD); ctx.lineTo(tipX, H-PAD); ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1.0;
    }

    // Dashed critical threshold line (sharpness = 100) — only on the sharpness plot
    if (getH === 'y' && dataset.criticalY !== undefined) {
      const sy = scYown(dataset.criticalY);
      ctx.globalAlpha = 0.75;
      ctx.strokeStyle = '#222';
      ctx.lineWidth   = 1.5 * dpr;
      ctx.setLineDash([6*dpr, 4*dpr]);
      ctx.beginPath(); ctx.moveTo(PAD, sy); ctx.lineTo(W-PAD, sy); ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1.0;
      // Small label
      ctx.font      = `${12*dpr}px system-ui,sans-serif`;
      ctx.fillStyle = '#222';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText('λ₁ = 100', 300, sy + 30);
      ctx.textBaseline = 'middle';
    }

    // Axis lines along the plot boundary
    ctx.globalAlpha = 1.0;
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth   = 1*dpr;
    ctx.beginPath(); ctx.moveTo(PAD, H-PAD); ctx.lineTo(W-PAD, H-PAD); ctx.stroke(); // x axis
    ctx.beginPath(); ctx.moveTo(PAD, PAD);   ctx.lineTo(PAD, H-PAD);   ctx.stroke(); // y axis

    // Axis labels
    ctx.font      = `${14*dpr}px system-ui,sans-serif`;
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.fillText(AXIS_LABELS[0], W/2, H - 18*dpr);   // x label bottom
    ctx.save();
    ctx.translate(20*dpr, H/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillStyle = yColor;
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }

  // ---- Render 3D ----
  function render3d() {
    const W=canvas3d.width, H=canvas3d.height;
    ctx3d.clearRect(0,0,W,H);
    ctx3d.fillStyle='#fff'; ctx3d.fillRect(0,0,W,H);

    const mvp = matMul(
      matPerspective(Math.PI/4, W/H),
      matMul(matTranslate(autoX+panX, panY, -zoom), matRotX(ax))
    );

    const {x0,x1,y0,y1,z0,z1} = ext;

    // Critical threshold plane (sharpness = 100)
    if (CRITICAL_PLANE) {
      const { y: vy, x0: px0, x1: px1, z0: pz0, z1: pz1,
              color, gridColor, gridStep } = CRITICAL_PLANE;
      // Use the scaled Y value for the plane position
      const scaledY = dataset.criticalY3d;  // ← Use scaled critical Y
      // Filled quad
      ctx3d.fillStyle = color;
      ctx3d.globalAlpha = 1.0;
      quad3d([[px0,scaledY,pz0],[px1,scaledY,pz0],[px1,scaledY,pz1],[px0,scaledY,pz1]], mvp,W,H);
      // Gridlines
      ctx3d.strokeStyle = gridColor;
      ctx3d.lineWidth   = 0.5 * dpr;
      for (let g = Math.ceil(pz0/gridStep)*gridStep; g <= pz1; g += gridStep)
        line3d(px0,scaledY,g, px1,scaledY,g, mvp,W,H);
      for (let g = Math.ceil(px0/gridStep)*gridStep; g <= px1; g += gridStep)
        line3d(g,scaledY,pz0, g,scaledY,pz1, mvp,W,H);
      // "critical threshold" label
      const labelPt = project3d((px0+px1)/2, scaledY, pz1, mvp, W, H);
      if (labelPt) {
        ctx3d.font = `bold ${14*dpr}px system-ui,sans-serif`;
        ctx3d.textAlign = 'center';
        ctx3d.textBaseline = 'bottom';
        ctx3d.strokeStyle = 'rgba(255,255,255,0.85)';
        ctx3d.lineWidth   = 4 * dpr;
        ctx3d.strokeText('critical threshold', labelPt.sx*0.7, labelPt.sy + 30);
        ctx3d.fillStyle = 'rgba(180,30,30,0.9)';
        ctx3d.fillText('critical threshold', labelPt.sx*0.7, labelPt.sy + 30);
      }
    }

    // Axes
    ctx3d.lineWidth=dpr;
    ctx3d.globalAlpha=0.25;
    ctx3d.strokeStyle=AXIS_COLORS[0]; line3d(x0,0,0,0,0,0,mvp,W,H);
    ctx3d.strokeStyle=AXIS_COLORS[1]; line3d(0,y0,0,0,0,0,mvp,W,H);
    ctx3d.strokeStyle=AXIS_COLORS[2]; line3d(0,0,z0,0,0,0,mvp,W,H);
    ctx3d.globalAlpha=1.0;
    ctx3d.strokeStyle=AXIS_COLORS[0]; line3d(0,0,0,x1,0,0,mvp,W,H);
    ctx3d.strokeStyle=AXIS_COLORS[1]; line3d(0,0,0,0,y1,0,mvp,W,H);
    ctx3d.strokeStyle=AXIS_COLORS[2]; line3d(0,0,0,0,0,z1,mvp,W,H);
      ctx3d.font=`bold ${16*dpr}px system-ui,sans-serif`;
      ctx3d.textAlign='center'; ctx3d.textBaseline='middle';
      [[x1,0,0],[0,y1,0],[0,0,z1]].forEach(([tx,ty,tz],i)=>{
        const s=project3d(tx,ty,tz,mvp,W,H); if(!s) return;
        ctx3d.strokeStyle='rgba(255,255,255,0.85)'; ctx3d.lineWidth=4*dpr; ctx3d.strokeText(AXIS_LABELS[i],s.sx,s.sy);
        ctx3d.fillStyle=AXIS_COLORS[i]; ctx3d.fillText(AXIS_LABELS[i],s.sx,s.sy);
      });

    // Ghost
    if (dataset.points.length > 1) {
      const all=dataset.points;
      ctx3d.lineWidth=LINE_WIDTH*dpr; ctx3d.lineCap='round'; ctx3d.lineJoin='round';
      ctx3d.globalAlpha=GHOST_ALPHA; ctx3d.strokeStyle='#000033';
      ctx3d.beginPath(); let started=false;
      for (const p of all) {
        const s=project3d(p.x, p.y3d, p.z, mvp, W, H);  // ← Use p.y3d
        if(!s) continue;
        if(!started){ctx3d.moveTo(s.sx,s.sy);started=true;}else ctx3d.lineTo(s.sx,s.sy);
      }
      ctx3d.stroke(); ctx3d.globalAlpha=1.0;
    }

        // Current curve
    if (currentPts.length > 1) {
      const N=currentPts.length, SEG=8;
      ctx3d.lineWidth=LINE_WIDTH*dpr; ctx3d.lineCap='round'; ctx3d.lineJoin='round'; ctx3d.globalAlpha=1.0;
      for (let i=0;i<N-1;i+=SEG) {
        const end=Math.min(i+SEG+1,N);
        ctx3d.strokeStyle=curveColor((i+SEG/2)/(N-1));
        ctx3d.beginPath(); let started=false;
        for (let j=i;j<end;j++) {
          const s=project3d(currentPts[j].x, currentPts[j].y3d, currentPts[j].z, mvp, W, H);  // ← Use .y3d
          if(!s) continue;
          if(!started){ctx3d.moveTo(s.sx,s.sy);started=true;}else ctx3d.lineTo(s.sx,s.sy);
        }
        ctx3d.stroke();
      }
    }
  }

  // ---- Main render loop ----
  function render() {
    requestAnimationFrame(render);
    if (!dataset) return;
    render3d();
    render2d(ctxXY, canvasXY, dataset.points, currentPts, 'y', AXIS_LABELS[1], AXIS_COLORS[1]);
    render2d(ctxXZ, canvasXZ, dataset.points, currentPts, 'z', AXIS_LABELS[2], AXIS_COLORS[2]);
  }

  render();
}

init();
