// experiments2.js — 3D weight-space trajectory viewer

// ---- Config ----
const DATA_PATH = 'runs/cifar10_sharp/gradient_displacement_data.json';
const MAP_X = 'residual_direction';  // → world X
const MAP_Y = 'sharpness_gradient';  // → world Y (negated)
const MAP_Z = 'top_eigenvector';     // → world Z (negated)
const AXIS_LABELS  = ['residual', 'sharp∇', 'top eig'];
const AXIS_COLORS  = [[0.75,0.10,0.10], [0.05,0.55,0.15], [0.10,0.20,0.80]];
const DRAG_SENS    = 0.004;

// Reference planes — each drawn as a semi-transparent quad spanning the full data extent.
// { axis: 'x'|'y'|'z', value: number|'mid', color: [r,g,b], alpha: 0-1 }
// value: a world-space coordinate, or 'mid' to auto-place at the midpoint of the data range.
const PLANES = [
  { axis: 'y', value: 'mid', color: [0.05, 0.55, 0.15], alpha: 0.35 },  // sharpness midplane
];

// ---- Data ----
function parseDataset(json) {
  const xs = json.displacements[MAP_X].cumulative;
  const ys = json.displacements[MAP_Y].cumulative;
  const zs = json.displacements[MAP_Z].cumulative;
  const epochs = json.epochs;
  const n = Math.min(xs.length, ys.length, zs.length, epochs.length);

  let div = 0;
  for (let i = 0; i < n; i++)
    div = Math.max(div, Math.abs(xs[i]), Math.abs(ys[i]), Math.abs(zs[i]));
  if (div === 0) div = 1;

  const points = [];
  for (let i = 0; i < n; i++)
    points.push({ x: xs[i]/div, y: -ys[i]/div, z: -zs[i]/div, epoch: epochs[i] });

  return { points, epochs };
}

// ---- Matrix math (column-major) ----
function mul(a, b) {
  const o = new Float32Array(16);
  for (let r=0;r<4;r++) for (let c=0;c<4;c++) { let s=0; for (let k=0;k<4;k++) s+=a[r+k*4]*b[k+c*4]; o[r+c*4]=s; } return o;
}
function perspective(fov, asp, n, f) {
  const t=1/Math.tan(fov/2), nf=1/(n-f);
  return new Float32Array([t/asp,0,0,0, 0,t,0,0, 0,0,(f+n)*nf,-1, 0,0,2*f*n*nf,0]);
}
function rotX(a){const c=Math.cos(a),s=Math.sin(a);return new Float32Array([1,0,0,0,0,c,-s,0,0,s,c,0,0,0,0,1]);}
function rotY(a){const c=Math.cos(a),s=Math.sin(a);return new Float32Array([c,0,s,0,0,1,0,0,-s,0,c,0,0,0,0,1]);}
function trans(x,y,z){return new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1]);}

function project(wx,wy,wz,mvp,W,H) {
  const x=mvp[0]*wx+mvp[4]*wy+mvp[8]*wz+mvp[12];
  const y=mvp[1]*wx+mvp[5]*wy+mvp[9]*wz+mvp[13];
  const w=mvp[3]*wx+mvp[7]*wy+mvp[11]*wz+mvp[15];
  if (w<=0) return null;
  return {sx:(x/w*0.5+0.5)*W, sy:(-y/w*0.5+0.5)*H};
}

// ---- WebGL helpers ----
function shader(gl, type, src) {
  const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s);
  if (!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(gl, vert, frag) {
  const p=gl.createProgram();
  gl.attachShader(p,shader(gl,gl.VERTEX_SHADER,vert));
  gl.attachShader(p,shader(gl,gl.FRAGMENT_SHADER,frag));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

// ---- Shaders ----
// Thick curve: screen-space quad per segment, black→blue gradient
const VERT_CURVE = `
  attribute vec3 aPos, aNext; attribute float aSide, aT;
  uniform mat4 uMVP; uniform vec2 uRes; uniform float uThick;
  varying float vT;
  void main() {
    vec4 c=uMVP*vec4(aPos,1.), cn=uMVP*vec4(aNext,1.);
    vec2 d=normalize((cn.xy/cn.w-c.xy/c.w)*uRes);
    gl_Position=vec4(c.xy+aSide*uThick*vec2(-d.y,d.x)/uRes*c.w,c.z,c.w);
    vT=aT;
  }`;
const FRAG_CURVE = `
  precision mediump float; varying float vT;
  void main() {
    float t=pow(vT,1.5);
    gl_FragColor=vec4(0., t*0.25, t*0.85+0.05, 1.0);
  }`;

// Flat color for axes
const VERT_FLAT = `attribute vec3 aPos; uniform mat4 uMVP; void main(){gl_Position=uMVP*vec4(aPos,1.);}`;
const FRAG_FLAT = `precision mediump float; uniform vec3 uColor; uniform float uAlpha; void main(){gl_FragColor=vec4(uColor,uAlpha);}`;

// ---- Main ----
async function init() {
  const canvas      = document.getElementById('curve3d-canvas');
  const epochSlider = document.getElementById('epoch-slider');
  const epochVal    = document.getElementById('epoch-value');
  const rotXSlider  = document.getElementById('rotx-slider');
  const rotXVal     = document.getElementById('rotx-value');
  const rotYSlider  = document.getElementById('roty-slider');
  const rotYVal     = document.getElementById('roty-value');
  const zoomSlider  = document.getElementById('zoom-slider');
  const zoomVal     = document.getElementById('zoom-value');
  const showAxes    = document.getElementById('check-axes');
  const showPlanes  = document.getElementById('check-planes');
  const loadStatus  = document.getElementById('load-status');

  // Canvas sizing
  const dpr = window.devicePixelRatio || 1;
  function resize() {
    const r = canvas.getBoundingClientRect();
    canvas.width  = Math.round(r.width  * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  resize();
  window.addEventListener('resize', resize);

  // WebGL
  const gl = canvas.getContext('webgl');
  if (!gl) { loadStatus.textContent = 'WebGL not supported'; return; }
  gl.enable(gl.DEPTH_TEST);

  // Programs
  const progCurve = program(gl, VERT_CURVE, FRAG_CURVE);
  const progFlat  = program(gl, VERT_FLAT,  FRAG_FLAT);

  const cLoc = {
    aPos:  gl.getAttribLocation(progCurve,'aPos'),
    aNext: gl.getAttribLocation(progCurve,'aNext'),
    aSide: gl.getAttribLocation(progCurve,'aSide'),
    aT:    gl.getAttribLocation(progCurve,'aT'),
    uMVP:  gl.getUniformLocation(progCurve,'uMVP'),
    uRes:  gl.getUniformLocation(progCurve,'uRes'),
    uThick:gl.getUniformLocation(progCurve,'uThick'),
  };
  const fLoc = {
    aPos:  gl.getAttribLocation(progFlat,'aPos'),
    uMVP:  gl.getUniformLocation(progFlat,'uMVP'),
    uColor:gl.getUniformLocation(progFlat,'uColor'),
    uAlpha:gl.getUniformLocation(progFlat,'uAlpha'),
  };

  // Buffers
  const buf = {
    pos:  gl.createBuffer(), next: gl.createBuffer(),
    side: gl.createBuffer(), t:    gl.createBuffer(),
    idx:  gl.createBuffer(), axis: gl.createBuffer(), plane: gl.createBuffer(),
  };
  let idxCount = 0;

  // 2D label overlay
  const label2d = document.createElement('canvas');
  label2d.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;';
  canvas.parentElement.style.position = 'relative';
  canvas.parentElement.appendChild(label2d);
  function resizeLabel() {
    const r = canvas.getBoundingClientRect();
    label2d.width  = Math.round(r.width  * dpr);
    label2d.height = Math.round(r.height * dpr);
    label2d.style.width  = r.width  + 'px';
    label2d.style.height = r.height + 'px';
  }
  resizeLabel();
  window.addEventListener('resize', resizeLabel);

  // ---- Load data ----
  let dataset = null;
  let axisTips = [[1,0,0],[0,1,0],[0,0,1]];

  // ---- Camera (declared here so rebuild() can write autoX before events are wired) ----
  let ax=-10*Math.PI/180, ay=0, zoom=3.5;
  let panX=0, panY=0;
  let autoX=0;

  try {
    const resp = await fetch(DATA_PATH);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    dataset = parseDataset(await resp.json());

    const n = dataset.points.length;
    epochSlider.min = 1; epochSlider.max = n; epochSlider.value = n;
    epochVal.textContent = dataset.epochs[n-1];
    loadStatus.textContent = `Loaded ${n} epochs (${dataset.epochs[0]}–${dataset.epochs[n-1]})`;

    buildAxes();
    rebuild();
  } catch(e) {
    loadStatus.textContent = `Failed to load: ${e.message}`;
    console.error(e); return;
  }

  // ---- Build axis geometry from data extent ----
  function buildAxes() {
    const pts = dataset.points;
    let x0=0,x1=0, y0=0,y1=0, z0=0,z1=0;
    for (const p of pts) {
      x0=Math.min(x0,p.x); x1=Math.max(x1,p.x);
      y0=Math.min(y0,p.y); y1=Math.max(y1,p.y);
      z0=Math.min(z0,p.z); z1=Math.max(z1,p.z);
    }
    const pad = (v) => v >= 0 ? v*1.15+0.05 : v*1.15-0.05;
    x0=pad(x0); x1=pad(x1); y0=pad(y0); y1=pad(y1); z0=pad(z0); z1=pad(z1);
    axisTips = [[x1,0,0],[0,y1,0],[0,0,z1]];
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.axis);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      x0,0,0, 0,0,0,   0,y0,0, 0,0,0,   0,0,z0, 0,0,0,  // negative stubs
      0,0,0,x1,0,0,    0,0,0,0,y1,0,    0,0,0,0,0,z1,   // positive arms
    ]), gl.STATIC_DRAW);

    // Build one quad per plane, spanning the data extent on the two non-normal axes
    const mid = { x: (x0+x1)/2, y: (y0+y1)/2, z: (z0+z1)/2 };
    const planeVerts = [];
    for (const pl of PLANES) {
      const v = pl.value === 'mid' ? mid[pl.axis] : pl.value;
      if (pl.axis === 'y') {
        planeVerts.push(x0,v,z0, x1,v,z0, x1,v,z1, x0,v,z0, x1,v,z1, x0,v,z1);
      } else if (pl.axis === 'x') {
        planeVerts.push(v,y0,z0, v,y1,z0, v,y1,z1, v,y0,z0, v,y1,z1, v,y0,z1);
      } else {
        planeVerts.push(x0,y0,v, x1,y0,v, x1,y1,v, x0,y0,v, x1,y1,v, x0,y1,v);
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.plane);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(planeVerts), gl.STATIC_DRAW);
  }

  // ---- Build curve GPU buffers ----
  function rebuild() {
    if (!dataset) return;
    const n = parseInt(epochSlider.value);
    epochVal.textContent = dataset.epochs[n-1];
    const pts = dataset.points.slice(0, n);
    if (pts.length < 2) { idxCount = 0; return; }

    const N = pts.length, S = N-1;
    const pos  = new Float32Array(S*4*3);
    const next = new Float32Array(S*4*3);
    const side = new Float32Array(S*4);
    const t    = new Float32Array(S*4);
    const idx  = new Uint16Array(S*6);

    for (let i=0; i<S; i++) {
      const p=pts[i], q=pts[i+1], ti=i/(N-1), b=i*4;
      for (let v=0; v<4; v++) {
        const s = v<2 ? p : q;
        pos[ (b+v)*3]  =s.x; pos[ (b+v)*3+1]=s.y; pos[ (b+v)*3+2]=s.z;
        next[(b+v)*3]  =q.x; next[(b+v)*3+1]=q.y; next[(b+v)*3+2]=q.z;
        side[b+v] = v%2===0 ? 1 : -1;
        t[b+v] = ti;
      }
      const j=i*6;
      idx[j]=b; idx[j+1]=b+1; idx[j+2]=b+2;
      idx[j+3]=b+1; idx[j+4]=b+3; idx[j+5]=b+2;
    }
    idxCount = idx.length;
    // Auto-follow: keep tip's x centred by negating it in the view translation
    autoX = -pts[pts.length-1].x;

    const u = gl.DYNAMIC_DRAW;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.pos);  gl.bufferData(gl.ARRAY_BUFFER, pos,  u);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.next); gl.bufferData(gl.ARRAY_BUFFER, next, u);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.side); gl.bufferData(gl.ARRAY_BUFFER, side, u);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.t);    gl.bufferData(gl.ARRAY_BUFFER, t,    u);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buf.idx); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, u);
  }

  function syncSliders() {
    rotXSlider.value = Math.round(ax*180/Math.PI); rotXVal.textContent = rotXSlider.value+'°';
    rotYSlider.value = Math.round(ay*180/Math.PI); rotYVal.textContent = rotYSlider.value+'°';
    zoomSlider.value = Math.round(zoom*10);         zoomVal.textContent = zoom.toFixed(1);
  }
  syncSliders();

  // ---- Events ----
  let dragging=false, lx=0, ly=0;
  canvas.addEventListener('mousedown', e=>{dragging=true; lx=e.clientX; ly=e.clientY;});
  canvas.addEventListener('mouseup',   ()=>{dragging=false;});
  canvas.addEventListener('mouseleave',()=>{dragging=false;});
  canvas.addEventListener('mousemove', e=>{
    if (!dragging) return;
    const dx=e.clientX-lx, dy=e.clientY-ly;
    if (e.shiftKey) {
      panX += dx * zoom * 0.002;
      panY -= dy * zoom * 0.002;
    } else {
      ax+=dx*DRAG_SENS; ay+=dy*DRAG_SENS;
      syncSliders();
    }
    lx=e.clientX; ly=e.clientY;
  });
  canvas.addEventListener('wheel', e=>{
    e.preventDefault();
    zoom=Math.max(0.5,Math.min(50,zoom*(1+e.deltaY*0.001))); syncSliders();
  },{passive:false});

  rotXSlider.addEventListener('input',()=>{ax=parseFloat(rotXSlider.value)*Math.PI/180; rotXVal.textContent=rotXSlider.value+'°';});
  rotYSlider.addEventListener('input',()=>{ay=parseFloat(rotYSlider.value)*Math.PI/180; rotYVal.textContent=rotYSlider.value+'°';});
  zoomSlider.addEventListener('input',()=>{zoom=parseFloat(zoomSlider.value)/10; zoomVal.textContent=zoom.toFixed(1);});
  epochSlider.addEventListener('input', rebuild);

  // ---- Render ----
  function render() {
    requestAnimationFrame(render);
    if (!dataset) return;

    const W=canvas.width, H=canvas.height;
    gl.viewport(0,0,W,H);
    gl.clearColor(1,1,1,1);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);

    const mvp = mul(perspective(Math.PI/4,W/H,0.1,50), mul(trans(autoX+panX, panY, -zoom), mul(rotX(ax),rotY(ay))));

    // Axes
    if (showAxes && showAxes.checked) {
      gl.useProgram(progFlat); gl.uniformMatrix4fv(fLoc.uMVP,false,mvp);
      gl.bindBuffer(gl.ARRAY_BUFFER,buf.axis);
      gl.enableVertexAttribArray(fLoc.aPos); gl.vertexAttribPointer(fLoc.aPos,3,gl.FLOAT,false,0,0);
      for(let i=0;i<3;i++){gl.uniform3fv(fLoc.uColor,AXIS_COLORS[i]);gl.uniform1f(fLoc.uAlpha,0.3);gl.drawArrays(gl.LINES,i*2,2);}
      for(let i=0;i<3;i++){gl.uniform3fv(fLoc.uColor,AXIS_COLORS[i]);gl.uniform1f(fLoc.uAlpha,1.0);gl.drawArrays(gl.LINES,6+i*2,2);}
    }

    // Planes (drawn before curve so curve renders on top)
    if (PLANES.length > 0 && showPlanes && showPlanes.checked) {
      gl.useProgram(progFlat); gl.uniformMatrix4fv(fLoc.uMVP,false,mvp);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf.plane);
      gl.enableVertexAttribArray(fLoc.aPos); gl.vertexAttribPointer(fLoc.aPos,3,gl.FLOAT,false,0,0);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);  // don't write to depth so curve always shows through
      let vOffset = 0;
      for (const pl of PLANES) {
        gl.uniform3fv(fLoc.uColor, pl.color);
        gl.uniform1f(fLoc.uAlpha, pl.alpha);
        gl.drawArrays(gl.TRIANGLES, vOffset, 6);
        vOffset += 6;
      }
      gl.depthMask(true);
      gl.disable(gl.BLEND);
    }

    // Curve
    if (idxCount>0) {
      gl.useProgram(progCurve); gl.uniformMatrix4fv(cLoc.uMVP,false,mvp);
      gl.uniform2f(cLoc.uRes,W,H); gl.uniform1f(cLoc.uThick,3.5*dpr);
      gl.bindBuffer(gl.ARRAY_BUFFER,buf.pos);  gl.enableVertexAttribArray(cLoc.aPos);  gl.vertexAttribPointer(cLoc.aPos, 3,gl.FLOAT,false,0,0);
      gl.bindBuffer(gl.ARRAY_BUFFER,buf.next); gl.enableVertexAttribArray(cLoc.aNext); gl.vertexAttribPointer(cLoc.aNext,3,gl.FLOAT,false,0,0);
      gl.bindBuffer(gl.ARRAY_BUFFER,buf.side); gl.enableVertexAttribArray(cLoc.aSide); gl.vertexAttribPointer(cLoc.aSide,1,gl.FLOAT,false,0,0);
      gl.bindBuffer(gl.ARRAY_BUFFER,buf.t);    gl.enableVertexAttribArray(cLoc.aT);    gl.vertexAttribPointer(cLoc.aT,   1,gl.FLOAT,false,0,0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,buf.idx);
      gl.drawElements(gl.TRIANGLES,idxCount,gl.UNSIGNED_SHORT,0);
    }

    // Axis labels (2D overlay)
    const ctx = label2d.getContext('2d');
    ctx.clearRect(0,0,label2d.width,label2d.height);
    if (showAxes && showAxes.checked) {
      ctx.font = `bold ${13*dpr}px system-ui,sans-serif`;
      ctx.textAlign='center'; ctx.textBaseline='middle';
      const labelColors = ['#b81818','#0a8c25','#1830cc'];
      for (let i=0; i<3; i++) {
        const tip = axisTips[i];
        const s = project(tip[0],tip[1],tip[2],mvp,label2d.width,label2d.height);
        if (!s) continue;
        ctx.strokeStyle='rgba(255,255,255,0.85)'; ctx.lineWidth=4*dpr; ctx.strokeText(AXIS_LABELS[i],s.sx,s.sy);
        ctx.fillStyle=labelColors[i]; ctx.fillText(AXIS_LABELS[i],s.sx,s.sy);
      }
    }
  }

  render();
}

init();
