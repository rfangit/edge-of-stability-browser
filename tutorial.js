// ============================================================================
// TUTORIAL WIDGETS - Inline training demonstrations
// ============================================================================
// Each widget is a self-contained simulation + chart pair embedded in the
// tutorial text. They use the same Simulation, LossChart, and RightChart
// classes as the main playground, but with hardcoded parameters.

import { Simulation } from './simulation.js';
import { LossChart, RightChart } from './visualization.js';

// Default Chebyshev parameters (matching the recommended EoS settings)
const CHEBYSHEV_DEFAULTS = {
  taskKey: 'chebyshev',
  taskParams: { degree: 4, nTrain: 20 },
  activation: 'tanh',
  hiddenDims: [100],
  eta: 0.3,
  batchSize: 20,
  modelSeed: 0
};

// ============================================================================
// WIDGET 1: Progressive Sharpening (1000 steps, 1 eigenvalue)
// ============================================================================

function initWidget1() {
  const startBtn = document.getElementById('widget1-start');
  const resetBtn = document.getElementById('widget1-reset');
  if (!startBtn) return null; // widget not in DOM

  const sim = new Simulation({ maxSteps: 2000, kEigs: 1, stepsPerSecId: null });
  const lossChart = new LossChart('widget1-loss');
  const rightChart = new RightChart('widget1-sharpness');

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
// WIDGET 2: Edge of Stability (continues from widget 1, 3000 more steps)
// ============================================================================

function initWidget2(widget1Sim) {
  const startBtn = document.getElementById('widget2-start');
  const resetBtn = document.getElementById('widget2-reset');
  if (!startBtn) return null;

  // Total steps = 2000 (from widget1) + 3000 = 5000
  const sim = new Simulation({ maxSteps: 5000, kEigs: 1, stepsPerSecId: null });
  const lossChart = new LossChart('widget2-loss');
  const rightChart = new RightChart('widget2-sharpness');

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

  function tryStart() {
    if (!widget1Sim || !widget1Sim.model) {
      startBtn.textContent = 'run widget above first';
      startBtn.disabled = true;
      return false;
    }
    if (widget1Sim.isRunning) {
      startBtn.textContent = 'wait for widget above';
      startBtn.disabled = true;
      return false;
    }
    return true;
  }

  startBtn.addEventListener('click', () => {
    if (!sim.isRunning) {
      if (!sim.model) {
        if (!tryStart()) return;
        const d = CHEBYSHEV_DEFAULTS;
        sim.captureParams(d.taskKey, d.taskParams, d.activation, d.hiddenDims, d.eta, d.batchSize, d.modelSeed);
        sim.continueFrom(widget1Sim);

        // Show the prepended history immediately
        lossChart.setInitialLoss(0.5);
        const state = sim.getState();
        lossChart.update(state.lossHistory, state.testLossHistory, state.eta);
        rightChart.update(state.eigenvalueHistory, state.eta);
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

  const sim = new Simulation({ kEigs: 3, stepsPerSecId: null });
  const lossChart = new LossChart('widget3-loss');
  const rightChart = new RightChart('widget3-sharpness');

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
// WIDGET 4: Gradient Flow Comparison (3 learning rates, 3×2 grid)
// ============================================================================

function initWidget4() {
  const startBtn = document.getElementById('widget4-start');
  const resetBtn = document.getElementById('widget4-reset');
  const xAxisToggle = document.getElementById('widget4-xaxis-toggle');
  if (!startBtn) return null;

  const baseEta = CHEBYSHEV_DEFAULTS.eta; // 0.25
  const etas = [baseEta, baseEta * 0.5, baseEta * 0.25]; // 0.25, 0.125, 0.0625
  const baseSteps = 4000;
  const maxSteps = [baseSteps, baseSteps * 2, baseSteps * 4]; // 4000, 8000, 16000

  const sims = [];
  const lossCharts = [];
  const sharpnessCharts = [];

  for (let i = 0; i < 3; i++) {
    const sim = new Simulation({ maxSteps: maxSteps[i], kEigs: 1, stepsPerSecId: null });
    const lc = new LossChart(`widget4-loss-${i}`);
    const rc = new RightChart(`widget4-sharpness-${i}`);

    sim.onFrameUpdate = () => {
      const state = sim.getState();
      lc.update(state.lossHistory, state.testLossHistory, state.eta);
      rc.update(state.eigenvalueHistory, state.eta);
    };

    const idx = i;
    sim.onComplete = () => {
      checkAllComplete();
    };

    sim.onDiverge = () => {
      checkAllComplete();
    };

    sims.push(sim);
    lossCharts.push(lc);
    sharpnessCharts.push(rc);
  }

  function checkAllComplete() {
    if (sims.every(s => !s.isRunning)) {
      startBtn.textContent = 'done';
      startBtn.disabled = true;
    }
  }

  let useEffectiveTime = false;

  // X-axis toggle (step vs step*η)
  if (xAxisToggle) {
    const stepLink = xAxisToggle.querySelector('[data-mode="step"]');
    const teffLink = xAxisToggle.querySelector('[data-mode="teff"]');

    function setXMode(mode) {
      useEffectiveTime = mode === 'teff';
      stepLink.classList.toggle('active', mode === 'step');
      teffLink.classList.toggle('active', mode === 'teff');
      for (let i = 0; i < 3; i++) {
        lossCharts[i].setEffectiveTime(useEffectiveTime, etas[i]);
        sharpnessCharts[i].setEffectiveTime(useEffectiveTime, etas[i]);
        // Force redraw
        const state = sims[i].getState();
        if (state.lossHistory.length > 0) {
          lossCharts[i].update(state.lossHistory, state.testLossHistory, state.eta);
          sharpnessCharts[i].update(state.eigenvalueHistory, state.eta);
        }
      }
    }

    stepLink.addEventListener('click', (e) => { e.preventDefault(); setXMode('step'); });
    teffLink.addEventListener('click', (e) => { e.preventDefault(); setXMode('teff'); });
  }

  startBtn.addEventListener('click', () => {
    const anyRunning = sims.some(s => s.isRunning);
    if (!anyRunning) {
      // Initialize all if needed
      for (let i = 0; i < 3; i++) {
        if (!sims[i].model) {
          const d = CHEBYSHEV_DEFAULTS;
          sims[i].captureParams(d.taskKey, d.taskParams, d.activation, d.hiddenDims, etas[i], d.batchSize, d.modelSeed);
          lossCharts[i].setInitialLoss(0.5);
        }
        sims[i].start();
      }
      startBtn.textContent = 'pause';
    } else {
      for (let i = 0; i < 3; i++) {
        if (sims[i].isRunning) sims[i].pause();
      }
      startBtn.textContent = 'resume';
    }
  });

  resetBtn.addEventListener('click', () => {
    for (let i = 0; i < 3; i++) {
      sims[i].reset();
      lossCharts[i].clear();
      sharpnessCharts[i].clear();
    }
    startBtn.textContent = 'start';
    startBtn.disabled = false;
  });

  return sims;
}

// ============================================================================
// INITIALIZATION - called after MathJax is ready
// ============================================================================

export function initTutorialWidgets() {
  const w1 = initWidget1();
  const w2 = initWidget2(w1);
  const w3 = initWidget3();
  const w4 = initWidget4();
}
