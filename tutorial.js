// ============================================================================
// TUTORIAL WIDGETS - Inline training demonstrations
// ============================================================================
// Each widget is a self-contained simulation + chart pair embedded in the
// tutorial text. They use the same Simulation, LossChart, and RightChart
// classes as the main playground, but with hardcoded parameters.

import { Simulation } from './simulation.js';
import { LossChart, RightChart } from './visualization.js';
import { MultiSimulation, MultiLossChart, MultiSharpnessChart } from './multi-simulation.js';
import { CHEBYSHEV_DEFAULTS, describeDefaults } from './defaults.js';

// ============================================================================
// WIDGET 1: Progressive Sharpening + Edge of Stability (5000 steps, 1 eigenvalue)
// ============================================================================

function initWidget1() {
  const startBtn = document.getElementById('widget1-start');
  const resetBtn = document.getElementById('widget1-reset');
  if (!startBtn) return null; // widget not in DOM

  // Populate info label from defaults
  const infoEl = document.getElementById('widget1-info');
  if (infoEl) infoEl.textContent = `${describeDefaults(CHEBYSHEV_DEFAULTS)}, Chebyshev degree ${CHEBYSHEV_DEFAULTS.taskParams.degree}, ${CHEBYSHEV_DEFAULTS.taskParams.nTrain} points — runs for 5000 steps`;

  const sim = new Simulation({ maxSteps: 5000, kEigs: 1, stepsPerSecId: null });
  const lossChart = new LossChart('widget1-loss', { showTest: false, showEma: false });
  const rightChart = new RightChart('widget1-sharpness', { kEigs: 1 });

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

  startBtn.addEventListener('click', () => {
    if (!sim.isRunning) {
      if (!sim.model) {
        const d = CHEBYSHEV_DEFAULTS;
        sim.captureParams(d.taskKey, d.taskParams, d.activation, d.hiddenDims, d.eta, d.batchSize, d.modelSeed);
        lossChart.setInitialLoss(0.5);
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
    rightChart.clear();
    startBtn.textContent = 'start';
    startBtn.disabled = false;
  });

  return sim;
}

// ============================================================================
// WIDGET 3: Multiple Eigenvalues (no step limit, 3 eigenvalues)
// ============================================================================

function initWidget3() {
  const startBtn = document.getElementById('widget3-start');
  const resetBtn = document.getElementById('widget3-reset');
  if (!startBtn) return null;

  // Populate info label from defaults
  const infoEl = document.getElementById('widget3-info');
  if (infoEl) infoEl.textContent = `${describeDefaults(CHEBYSHEV_DEFAULTS)}, 3 eigenvalues — stop when ready`;

  const sim = new Simulation({ kEigs: 3, stepsPerSecId: null });
  const lossChart = new LossChart('widget3-loss', { showTest: false, showEma: false });
  const rightChart = new RightChart('widget3-sharpness', { kEigs: 3 });

  sim.onFrameUpdate = () => {
    const state = sim.getState();
    lossChart.update(state.lossHistory, state.testLossHistory, state.eta);
    rightChart.update(state.eigenvalueHistory, state.eta);
  };

  sim.onDiverge = (iteration, loss) => {
    startBtn.textContent = 'diverged';
    startBtn.disabled = true;
  };

  startBtn.addEventListener('click', () => {
    if (!sim.isRunning) {
      if (!sim.model) {
        const d = CHEBYSHEV_DEFAULTS;
        sim.captureParams(d.taskKey, d.taskParams, d.activation, d.hiddenDims, d.eta, d.batchSize, d.modelSeed);
        lossChart.setInitialLoss(0.5);
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
    rightChart.clear();
    startBtn.textContent = 'start';
    startBtn.disabled = false;
  });

  return sim;
}

// ============================================================================
// WIDGET 4: Gradient Flow Comparison (3 learning rates, shared charts)
// ============================================================================

function initWidget4() {
  const startBtn = document.getElementById('widget4-start');
  const resetBtn = document.getElementById('widget4-reset');
  const xAxisToggle = document.getElementById('widget4-xaxis-toggle');
  if (!startBtn) return null;

  const baseEta = CHEBYSHEV_DEFAULTS.eta;
  const etaMultipliers = [1, 0.75, 0.5];
  const etas = etaMultipliers.map(m => baseEta * m);

  // Populate info label
  const infoEl = document.getElementById('widget4-info');
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

  const lossChart = new MultiLossChart('widget4-loss', labels, { dashes });
  const sharpnessChart = new MultiSharpnessChart('widget4-sharpness', labels, { dashes });

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
// INITIALIZATION - called after MathJax is ready
// ============================================================================

export function initTutorialWidgets() {
  const w1 = initWidget1();
  const w3 = initWidget3();
  const w4 = initWidget4();
}
