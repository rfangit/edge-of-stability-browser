// ============================================================================
// MULTI-SIMULATION - Coordinated training of multiple models
// ============================================================================
// Runs N simulations in lockstep (one step per model per frame tick) and
// feeds their histories into shared multi-series charts.

import { MLP } from './model.js';
import { Trainer } from './training.js';
import { TASKS, generateDataset } from './tasks.js';
import { lanczosTopEigenvalues } from './hessian.js';
import { formatTickLabel, baseChartOptions, CHART_FONT } from './chart-utils.js';

// NOTE: IncrementalCache is not currently used here, but could be added
// for downsampling if multi-simulation histories grow very large.
// import { IncrementalCache } from './incremental-cache.js';

// ============================================================================
// CHART HELPERS
// ============================================================================

// Default colors for multi-series
const SERIES_COLORS = [
  'rgb(220, 50, 50)',    // red
  'rgb(40, 130, 180)',   // blue
  'rgb(80, 180, 80)',    // green
];

// ============================================================================
// MULTI LOSS CHART - overlays N train loss curves
// ============================================================================

export class MultiLossChart {
  /**
   * @param {string} canvasId
   * @param {string[]} labels - one label per series (e.g. ['η = 0.4', 'η = 0.3', ...])
   * @param {object} [options]
   * @param {string[]} [options.colors] - colors per series
   * @param {number[][]} [options.dashes] - borderDash per series (e.g. [[], [6,3], [2,2]])
   */
  constructor(canvasId, labels, options = {}) {
    this.n = labels.length;
    this.colors = options.colors || SERIES_COLORS.slice(0, this.n);
    this.dashes = options.dashes || new Array(this.n).fill([]);
    this.useEffectiveTime = false;
    this.etas = new Array(this.n).fill(0.01);
    this.logScale = false;

    const datasets = labels.map((label, i) => ({
      label: label,
      data: [],
      borderColor: this.colors[i],
      backgroundColor: 'transparent',
      borderWidth: 2,
      borderDash: this.dashes[i],
      pointRadius: 0,
      tension: 0,
      order: i
    }));

    const ctx = document.getElementById(canvasId).getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: baseChartOptions()
    });

    // Multi charts may have non-integer x values (η·step); only show integer ticks
    this.chart.options.scales.x.ticks.callback = function(value) {
      if (value !== Math.floor(value)) return '';
      return formatTickLabel(value);
    };

    // Custom legend with dash styles
    const chartRef = this;
    this.chart.options.plugins.legend.labels.generateLabels = function(chart) {
      return labels.map((label, i) => ({
        text: label,
        strokeStyle: chartRef.colors[i],
        fillStyle: chartRef.colors[i],
        lineWidth: 2,
        lineDash: chartRef.dashes[i],
        hidden: false
      }));
    };
  }

  setEffectiveTime(useEffTime, etas) {
    this.useEffectiveTime = useEffTime;
    this.etas = etas;
  }

  /**
   * @param {Array<Array<{iteration: number, loss: number}>>} allHistories - one history per series
   */
  update(allHistories) {
    let xMax = 0;
    let yMax = 0;

    for (let i = 0; i < this.n; i++) {
      const hist = allHistories[i];
      if (!hist || hist.length === 0) {
        this.chart.data.datasets[i].data = [];
        continue;
      }
      const eta = this.etas[i];
      const data = hist.map(p => ({
        x: this.useEffectiveTime ? p.iteration * eta : p.iteration,
        y: p.loss
      }));
      this.chart.data.datasets[i].data = data;

      const lastX = data[data.length - 1].x;
      if (lastX > xMax) xMax = lastX;
      for (const p of hist) {
        if (p.loss > yMax) yMax = p.loss;
      }
    }

    this.chart.options.scales.x.max = xMax > 0 ? Math.ceil(xMax) : undefined;

    if (!this.logScale) {
      const yMaxPadded = yMax * 1.4;
      this.chart.options.scales.y.max = yMaxPadded;
      this.chart.options.scales.y.ticks.callback = function(value) {
        if (Math.abs(value - yMaxPadded) < 1e-10) return '';
        return formatTickLabel(value);
      };
    } else {
      this.chart.options.scales.y.max = undefined;
    }

    this.chart.update('none');
  }

  clear() {
    for (const ds of this.chart.data.datasets) ds.data = [];
    this.chart.options.scales.x.max = undefined;
    this.chart.update('none');
  }
}

// ============================================================================
// MULTI SHARPNESS CHART - overlays N λ₁ curves + N threshold lines
// ============================================================================

export class MultiSharpnessChart {
  /**
   * @param {string} canvasId
   * @param {string[]} labels - one label per series
   * @param {object} [options]
   * @param {string[]} [options.colors] - colors per series
   * @param {number[][]} [options.dashes] - borderDash per series
   */
  constructor(canvasId, labels, options = {}) {
    this.n = labels.length;
    this.colors = options.colors || SERIES_COLORS.slice(0, this.n);
    this.dashes = options.dashes || new Array(this.n).fill([]);
    this.useEffectiveTime = false;
    this.etas = new Array(this.n).fill(0.01);
    this.logScale = false;

    const datasets = [];

    // Threshold lines (solid, semi-transparent black)
    this.thresholdIndices = [];
    for (let i = 0; i < this.n; i++) {
      this.thresholdIndices.push(datasets.length);
      datasets.push({
        label: i === 0 ? '2/η' : '',
        data: [],
        borderColor: 'rgba(0, 0, 0, 0.5)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        order: 10 + i
      });
    }

    // Eigenvalue curves
    this.eigIndices = [];
    for (let i = 0; i < this.n; i++) {
      this.eigIndices.push(datasets.length);
      datasets.push({
        label: labels[i],
        data: [],
        borderColor: this.colors[i],
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: this.dashes[i],
        pointRadius: 0,
        tension: 0,
        order: i
      });
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: baseChartOptions()
    });

    // Multi charts may have non-integer x values (η·step); only show integer ticks
    this.chart.options.scales.x.ticks.callback = function(value) {
      if (value !== Math.floor(value)) return '';
      return formatTickLabel(value);
    };

    // Custom legend: only show eigenvalue curves + one threshold entry
    const chartRef = this;
    this.chart.options.plugins.legend.labels.generateLabels = function(chart) {
      const legendItems = [];
      // Threshold — single entry
      legendItems.push({
        text: '2/η',
        strokeStyle: 'rgba(0, 0, 0, 0.5)',
        fillStyle: 'rgba(0, 0, 0, 0.5)',
        lineWidth: 2,
        hidden: false
      });
      // Eigenvalue curves
      for (let i = 0; i < chartRef.n; i++) {
        legendItems.push({
          text: labels[i],
          strokeStyle: chartRef.colors[i],
          fillStyle: chartRef.colors[i],
          lineWidth: 2,
          lineDash: chartRef.dashes[i],
          hidden: false
        });
      }
      return legendItems;
    };
  }

  setEffectiveTime(useEffTime, etas) {
    this.useEffectiveTime = useEffTime;
    this.etas = etas;
  }

  /**
   * @param {Array<Array<{iteration: number, eigs: number[]}>>} allEigHistories
   * @param {number[]} etas - learning rates per series (for threshold lines)
   */
  update(allEigHistories, etas) {
    this.etas = etas;
    let xMax = 0;
    let yMax = 0;

    // First pass: compute eigenvalue data and find global xMax/yMax
    for (let i = 0; i < this.n; i++) {
      const hist = allEigHistories[i];
      const eta = etas[i];
      const threshold = 2 / eta;
      if (threshold > yMax) yMax = threshold;

      const toX = (iter) => this.useEffectiveTime ? iter * eta : iter;

      // Eigenvalue curve (top eigenvalue only)
      if (hist && hist.length > 0) {
        const data = hist.map(p => {
          const k = p.eigs.length;
          return { x: toX(p.iteration), y: p.eigs[k - 1] };
        });
        this.chart.data.datasets[this.eigIndices[i]].data = data;

        const lastX = data[data.length - 1].x;
        if (lastX > xMax) xMax = lastX;

        for (const p of hist) {
          for (const e of p.eigs) {
            if (e > yMax) yMax = e;
          }
        }
      } else {
        this.chart.data.datasets[this.eigIndices[i]].data = [];
      }
    }

    // Second pass: set threshold lines spanning from 0 to global xMax (ceiled to integer)
    const xMaxCeil = xMax > 0 ? Math.ceil(xMax) : 0;
    for (let i = 0; i < this.n; i++) {
      const threshold = 2 / etas[i];
      if (xMaxCeil > 0) {
        this.chart.data.datasets[this.thresholdIndices[i]].data = [
          { x: 0, y: threshold },
          { x: xMaxCeil, y: threshold }
        ];
      } else {
        this.chart.data.datasets[this.thresholdIndices[i]].data = [];
      }
    }

    this.chart.options.scales.x.max = xMaxCeil || undefined;

    if (!this.logScale) {
      this.chart.options.scales.y.max = yMax * 1.3;
    } else {
      this.chart.options.scales.y.max = undefined;
    }

    this.chart.update('none');
  }

  clear() {
    for (const ds of this.chart.data.datasets) ds.data = [];
    this.chart.options.scales.x.max = undefined;
    this.chart.update('none');
  }
}

// ============================================================================
// MULTI-SIMULATION
// ============================================================================

export class MultiSimulation {
  /**
   * @param {object} options
   * @param {number} options.n - number of parallel simulations
   * @param {number[]} options.etas - learning rate per simulation
   * @param {number[]} [options.maxSteps] - max steps per simulation (null entries = unlimited)
   * @param {number[]} [options.stepsPerTick] - steps per model per outer tick (for proportional speed)
   * @param {object} options.baseParams - shared params: { taskKey, taskParams, activation, hiddenDims, batchSize, modelSeed }
   * @param {number} [options.kEigs=1] - eigenvalues to track
   * @param {number} [options.hessianNumIters=10] - Lanczos iterations
   * @param {number} [options.hessianMaxIters=30] - Lanczos max iterations
   * @param {number} [options.hessianInterval=1] - steps between Hessian computations
   */
  constructor(options) {
    this.n = options.n;
    this.etas = options.etas;
    this.maxSteps = options.maxSteps || new Array(this.n).fill(null);
    this.stepsPerTick = options.stepsPerTick || new Array(this.n).fill(1);
    this.baseParams = options.baseParams;
    this.kEigs = options.kEigs || 1;
    this.hessianNumIters = options.hessianNumIters || 10;
    this.hessianMaxIters = options.hessianMaxIters || 30;
    this.hessianInterval = options.hessianInterval || 1;

    this.isRunning = false;
    this.animationFrameId = null;

    // Per-simulation state
    this.models = new Array(this.n).fill(null);
    this.trainers = new Array(this.n).fill(null);
    this.datasets = new Array(this.n).fill(null);
    this.dataYArrays = new Array(this.n).fill(null);
    this.iterations = new Array(this.n).fill(0);
    this.finished = new Array(this.n).fill(false);

    // Histories
    this.lossHistories = Array.from({ length: this.n }, () => []);
    this.eigHistories = Array.from({ length: this.n }, () => []);

    // Callbacks
    this.onFrameUpdate = null;
    this.onComplete = null;

    // Adaptive stepping
    this.TARGET_FRAME_TIME = 25;
    this.avgStepTime = 0.8;
    this.STEP_TIME_ALPHA = 0.15;
  }

  initialize() {
    const bp = this.baseParams;
    const task = TASKS[bp.taskKey];

    for (let i = 0; i < this.n; i++) {
      // All simulations share the same dataset (generated from same params)
      this.datasets[i] = generateDataset(bp.taskKey, bp.taskParams);
      this.dataYArrays[i] = this.datasets[i].y.map(y => Array.isArray(y) ? y : [y]);

      const layerSizes = [task.inputDim, ...bp.hiddenDims, task.outputDim];
      this.models[i] = new MLP(layerSizes, bp.activation, bp.modelSeed);
      this.trainers[i] = new Trainer(this.models[i], this.etas[i], bp.batchSize, this.datasets[i]);

      this.iterations[i] = 0;
      this.finished[i] = false;
      this.lossHistories[i] = [];
      this.eigHistories[i] = [];
    }
  }

  start() {
    if (this.isRunning) return;
    if (!this.models[0]) this.initialize();
    this.isRunning = true;
    this.runLoop();
  }

  pause() {
    this.isRunning = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  reset() {
    this.pause();
    for (let i = 0; i < this.n; i++) {
      this.models[i] = null;
      this.trainers[i] = null;
      this.datasets[i] = null;
      this.dataYArrays[i] = null;
      this.iterations[i] = 0;
      this.finished[i] = false;
      this.lossHistories[i] = [];
      this.eigHistories[i] = [];
    }
  }

  runLoop() {
    if (!this.isRunning) return;

    const frameStart = performance.now();
    let stepsThisFrame = 0;
    const timeBudget = this.TARGET_FRAME_TIME - 1.5;

    while (true) {
      const elapsed = performance.now() - frameStart;
      if (elapsed + this.avgStepTime > timeBudget && stepsThisFrame > 0) break;

      const stepStart = performance.now();

      // Each model takes stepsPerTick[i] steps per outer tick
      let allFinished = true;
      for (let i = 0; i < this.n; i++) {
        if (this.finished[i]) continue;
        allFinished = false;

        for (let s = 0; s < this.stepsPerTick[i]; s++) {
          if (this.finished[i]) break;

          const loss = this.trainers[i].step();
          this.iterations[i]++;
          this.lossHistories[i].push({ iteration: this.iterations[i], loss });

          // Divergence check
          if (!isFinite(loss) || loss > 100000) {
            this.finished[i] = true;
            break;
          }

          // Hessian eigenvalues
          if (this.iterations[i] % this.hessianInterval === 0) {
            const eigs = lanczosTopEigenvalues(
              this.trainers[i],
              this.datasets[i].x,
              this.dataYArrays[i],
              {
                kEigs: this.kEigs,
                numIters: this.hessianNumIters,
                maxIters: this.hessianMaxIters,
                tolRatio: 0.01
              }
            );
            if (eigs && eigs.eigenvalues) {
              this.eigHistories[i].push({
                iteration: this.iterations[i],
                eigs: eigs.eigenvalues
              });
            }
          }

          // Max steps check
          if (this.maxSteps[i] && this.iterations[i] >= this.maxSteps[i]) {
            this.finished[i] = true;
          }
        }
      }

      const stepTime = performance.now() - stepStart;
      this.avgStepTime = this.STEP_TIME_ALPHA * stepTime + (1 - this.STEP_TIME_ALPHA) * this.avgStepTime;

      stepsThisFrame++;
      if (stepsThisFrame >= 500) break;

      if (allFinished) {
        this.isRunning = false;
        if (this.onFrameUpdate) this.onFrameUpdate();
        if (this.onComplete) this.onComplete();
        return;
      }
    }

    if (this.onFrameUpdate) this.onFrameUpdate();
    this.animationFrameId = requestAnimationFrame(() => this.runLoop());
  }
}
