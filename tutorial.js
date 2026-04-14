// ============================================================================
// TUTORIAL WIDGETS - Inline training demonstrations
// ============================================================================
// Each widget is a self-contained simulation + chart pair embedded in the
// tutorial text. They use the same Simulation, LossChart, and RightChart
// classes as the main playground, but with hardcoded parameters.

import { Simulation } from './simulation.js';
import { LossChart, RightChart, GradProjectionChart } from './visualization.js';
import { MultiSimulation, MultiLossChart, MultiSharpnessChart } from './multi-simulation.js';
import { CHEBYSHEV_DEFAULTS, describeDefaults } from './defaults.js';

// ============================================================================
// BASE EXPERIMENT: Progressive Sharpening + Edge of Stability (5000 steps, 1 eigenvalue)
// ============================================================================

function initBaseExperiment() {
  const startBtn = document.getElementById('base-experiment-start');
  const resetBtn = document.getElementById('base-experiment-reset');
  if (!startBtn) return null; // widget not in DOM

  // Populate info label from defaults
  const infoEl = document.getElementById('base-experiment-info');
  if (infoEl) infoEl.textContent = `tanh, 2 hidden layers (${CHEBYSHEV_DEFAULTS.hiddenDims.join('+')}), Chebyshev degree ${CHEBYSHEV_DEFAULTS.taskParams.degree}, ${CHEBYSHEV_DEFAULTS.taskParams.nTrain} points — runs for 5000 steps`;

  const sim = new Simulation({ maxSteps: 5000, kEigs: 1, stepsPerSecId: null });
  const lossChart = new LossChart('base-experiment-loss', { showTest: false, showEma: false });
  const rightChart = new RightChart('base-experiment-sharpness', { kEigs: 1 });

  sim.onFrameUpdate = () => {
    const state = sim.getState();
    lossChart.update(state.lossHistory, state.testLossHistory, state.eta);
    rightChart.update(state.eigenvalueHistory, state.eta);
  };

  sim.onComplete = () => {
    startBtn.textContent = 'done';
    startBtn.disabled = true;
  };

  sim.onDiverge = (iteration, loss) => {
    startBtn.textContent = 'diverged';
    startBtn.disabled = true;
  };

  const DEFAULT_ETA = CHEBYSHEV_DEFAULTS.eta;
  const lrSlider = document.getElementById('base-experiment-lr-slider');
  const lrValueEl = document.getElementById('base-experiment-lr-value');
  const lrDefaultBtn = document.getElementById('base-experiment-lr-default');

  function setEta(val) {
    if (lrSlider) lrSlider.value = val;
    if (lrValueEl) lrValueEl.textContent = parseFloat(val).toFixed(2);
  }

  // Initialise display to DEFAULT_ETA on load
  setEta(DEFAULT_ETA);

  if (lrSlider) {
    lrSlider.addEventListener('input', () => {
      lrValueEl.textContent = parseFloat(lrSlider.value).toFixed(2);
    });
  }

  if (lrDefaultBtn) {
    lrDefaultBtn.addEventListener('click', () => {
      setEta(DEFAULT_ETA);
    });
  }

  function getEta() {
    return lrSlider ? parseFloat(lrSlider.value) : DEFAULT_ETA;
  }

  startBtn.addEventListener('click', () => {
    if (!sim.isRunning) {
      if (!sim.model) {
        const d = CHEBYSHEV_DEFAULTS;
        sim.captureParams(d.taskKey, d.taskParams, d.activation, d.hiddenDims, getEta(), d.batchSize, d.modelSeed);
      }
      sim.start();
      startBtn.textContent = 'pause';
    } else {
      sim.pause();
      startBtn.textContent = 'resume';
    }
  });

  // Reset clears the simulation and charts but leaves the LR slider where it is
  resetBtn.addEventListener('click', () => {
    sim.reset();
    lossChart.clear();
    rightChart.clear();
    startBtn.textContent = 'start';
    startBtn.disabled = false;
  });

  return sim;
}

// ============================================================================
// MULTI EIGENVALUES: Multiple eigenvalue tracking (no step limit, 3 eigenvalues)
// ============================================================================

function initMultiEigenvalues() {
  const startBtn = document.getElementById('multi-eigenvalues-start');
  const resetBtn = document.getElementById('multi-eigenvalues-reset');
  if (!startBtn) return null;

  // Populate info label from defaults (no explicit lr)
  const infoEl = document.getElementById('multi-eigenvalues-info');
  if (infoEl) infoEl.textContent = `tanh, 2 hidden layers (${CHEBYSHEV_DEFAULTS.hiddenDims.join('+')}), 3 eigenvalues — stop when ready`;

  const sim = new Simulation({ kEigs: 3, stepsPerSecId: null });
  const lossChart = new LossChart('multi-eigenvalues-loss', { showTest: false, showEma: false });
  const rightChart = new RightChart('multi-eigenvalues-sharpness', { kEigs: 3 });

  sim.onFrameUpdate = () => {
    const state = sim.getState();
    lossChart.update(state.lossHistory, state.testLossHistory, state.eta);
    rightChart.update(state.eigenvalueHistory, state.eta);
  };

  sim.onDiverge = (iteration, loss) => {
    startBtn.textContent = 'diverged';
    startBtn.disabled = true;
  };

  const DEFAULT_ETA = CHEBYSHEV_DEFAULTS.eta;
  const lrSlider = document.getElementById('multi-eigenvalues-lr-slider');
  const lrValueEl = document.getElementById('multi-eigenvalues-lr-value');
  const lrDefaultBtn = document.getElementById('multi-eigenvalues-lr-default');

  function setEta(val) {
    if (lrSlider) lrSlider.value = val;
    if (lrValueEl) lrValueEl.textContent = parseFloat(val).toFixed(2);
  }

  setEta(DEFAULT_ETA);

  if (lrSlider) {
    lrSlider.addEventListener('input', () => {
      lrValueEl.textContent = parseFloat(lrSlider.value).toFixed(2);
    });
  }

  if (lrDefaultBtn) {
    lrDefaultBtn.addEventListener('click', () => setEta(DEFAULT_ETA));
  }

  function getEta() {
    return lrSlider ? parseFloat(lrSlider.value) : DEFAULT_ETA;
  }

  startBtn.addEventListener('click', () => {
    if (!sim.isRunning) {
      if (!sim.model) {
        const d = CHEBYSHEV_DEFAULTS;
        sim.captureParams(d.taskKey, d.taskParams, d.activation, d.hiddenDims, getEta(), d.batchSize, d.modelSeed);
      }
      sim.start();
      startBtn.textContent = 'pause';
    } else {
      sim.pause();
      startBtn.textContent = 'resume';
    }
  });

  // Reset clears the simulation and charts but leaves the LR slider where it is
  resetBtn.addEventListener('click', () => {
    sim.reset();
    lossChart.clear();
    rightChart.clear();
    startBtn.textContent = 'start';
    startBtn.disabled = false;
  });

  return sim;
}

// ============================================================================
// GRADIENT FLOW: Three learning rate comparison (3 learning rates, shared charts)
// ============================================================================

function initGradientFlow() {
  const startBtn = document.getElementById('gradient-flow-start');
  const resetBtn = document.getElementById('gradient-flow-reset');
  const xAxisToggle = document.getElementById('gradient-flow-xaxis-toggle');
  if (!startBtn) return null;

  const baseEta = CHEBYSHEV_DEFAULTS.eta;
  const etaMultipliers = [1, 0.75, 0.5];
  const etas = etaMultipliers.map(m => baseEta * m);

  // Populate info label
  const infoEl = document.getElementById('gradient-flow-info');
  if (infoEl) infoEl.textContent = `η = ${etas.map(e => parseFloat(e.toPrecision(4))).join(', ')} — same model, three learning rates`;

  // stepsPerTick proportional to 1/eta so step*eta advances ~equally per tick
  // Highest η gets fewest steps (slowest), lowest η gets most steps (fastest)
  const maxEta = Math.max(...etas);
  const baseTicks = 3; // steps per tick for the highest η (slowest)
  const stepsPerTick = etas.map(e => Math.round(baseTicks * maxEta / e));

  // maxSteps proportional to 1/eta so total effective time is equal
  const baseEffTime = 2000; // step * eta target
  const maxSteps = etas.map(e => Math.round(baseEffTime / e));

  const labels = etas.map(e => `η = ${parseFloat(e.toPrecision(4))}`);
  const dashes = [[6, 3], [2, 2], []]; // dashed (highest η), dotted (middle), solid (lowest)

  const multiSim = new MultiSimulation({
    n: 3,
    etas: etas,
    maxSteps: maxSteps,
    stepsPerTick: stepsPerTick,
    baseParams: {
      taskKey: CHEBYSHEV_DEFAULTS.taskKey,
      taskParams: CHEBYSHEV_DEFAULTS.taskParams,
      activation: CHEBYSHEV_DEFAULTS.activation,
      hiddenDims: CHEBYSHEV_DEFAULTS.hiddenDims,
      batchSize: CHEBYSHEV_DEFAULTS.batchSize,
      modelSeed: CHEBYSHEV_DEFAULTS.modelSeed
    },
    kEigs: 1,
    hessianNumIters: 10,
    hessianMaxIters: 30,
    hessianInterval: 3  // compute Hessian every 3 steps (reduces jitter + cost)
  });

  const lossChart = new MultiLossChart('gradient-flow-loss', labels, { dashes });
  const sharpnessChart = new MultiSharpnessChart('gradient-flow-sharpness', labels, { dashes });

  // Default to η·step view
  let useEffectiveTime = true;
  lossChart.setEffectiveTime(true, etas);
  sharpnessChart.setEffectiveTime(true, etas);

  multiSim.onFrameUpdate = () => {
    lossChart.update(multiSim.lossHistories);
    sharpnessChart.update(multiSim.eigHistories, etas);
  };

  multiSim.onComplete = () => {
    startBtn.textContent = 'done';
    startBtn.disabled = true;
  };

  // X-axis toggle
  if (xAxisToggle) {
    const stepLink = xAxisToggle.querySelector('[data-mode="step"]');
    const teffLink = xAxisToggle.querySelector('[data-mode="teff"]');

    // Start with teff active
    stepLink.classList.remove('active');
    teffLink.classList.add('active');

    function setXMode(mode) {
      useEffectiveTime = mode === 'teff';
      stepLink.classList.toggle('active', mode === 'step');
      teffLink.classList.toggle('active', mode === 'teff');
      lossChart.setEffectiveTime(useEffectiveTime, etas);
      sharpnessChart.setEffectiveTime(useEffectiveTime, etas);
      // Force redraw
      lossChart.update(multiSim.lossHistories);
      sharpnessChart.update(multiSim.eigHistories, etas);
    }

    stepLink.addEventListener('click', (e) => { e.preventDefault(); setXMode('step'); });
    teffLink.addEventListener('click', (e) => { e.preventDefault(); setXMode('teff'); });
  }

  startBtn.addEventListener('click', () => {
    if (!multiSim.isRunning) {
      multiSim.start();
      startBtn.textContent = 'pause';
    } else {
      multiSim.pause();
      startBtn.textContent = 'resume';
    }
  });

  resetBtn.addEventListener('click', () => {
    multiSim.reset();
    lossChart.clear();
    sharpnessChart.clear();
    startBtn.textContent = 'start';
    startBtn.disabled = false;
  });

  return multiSim;
}

// ============================================================================
// EIGENVECTOR WIDGET: Loss + Sharpness + Gradient Projection along top eigenvector
// ============================================================================
// Uses CHEBYSHEV_DEFAULTS. The third chart only starts filling once sharpness
// enters the 5% proximity band around 2/η, at which point the top eigenvector
// is locked in and every subsequent gradient is projected onto it.

function initEigenvectorWidget() {
  const startBtn = document.getElementById('eigenvector-start');
  const resetBtn = document.getElementById('eigenvector-reset');
  if (!startBtn) return null;

  // Info label
  const infoEl = document.getElementById('eigenvector-info');
  if (infoEl) infoEl.textContent = `${describeDefaults(CHEBYSHEV_DEFAULTS)}, Chebyshev degree ${CHEBYSHEV_DEFAULTS.taskParams.degree}, ${CHEBYSHEV_DEFAULTS.taskParams.nTrain} points — eigenvector locked at 5% of 2/η`;

  const sim = new Simulation({
    kEigs: 1,
    stepsPerSecId: null,
    sharpnessProximityThreshold: 0.05
  });

  const lossChart = new LossChart('eigenvector-loss', { showTest: false, showEma: false });
  const sharpnessChart = new RightChart('eigenvector-sharpness', { kEigs: 1 });
  const projChart = new GradProjectionChart('eigenvector-projection');

  sim.onFrameUpdate = () => {
    const state = sim.getState();
    lossChart.update(state.lossHistory, state.testLossHistory, state.eta);
    sharpnessChart.update(state.eigenvalueHistory, state.eta);
    projChart.update(state.gradProjectionHistory, state.eta);

    // Update capture-step annotation if eigenvector was just captured
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
// INITIALIZATION - called after MathJax is ready
// ============================================================================

export function initTutorialWidgets() {
  const w1 = initBaseExperiment();
  const w2 = initMultiEigenvalues();
  const w3 = initGradientFlow();
  const w4 = initEigenvectorWidget();
}
