// ============================================================================
// ANIMATED PLOT - Progressive reveal of pre-computed training data
// ============================================================================
// Loads a JSON file (same format as saved runs) and progressively reveals
// the data in two Chart.js plots (loss + sharpness), then pauses and loops.
// General-purpose — can be used for title animations, inline demos, etc.
//
// Configuration (all in the options object):
//   dataUrl:        string   - path to JSON file
//   lossCanvasId:   string   - canvas element ID for loss chart
//   sharpCanvasId:  string   - canvas element ID for sharpness chart
//   revealDuration: number   - seconds to reveal all data (default: 10)
//   pauseDuration:  number   - seconds to pause on final frame before looping (default: 3)
//   fps:            number   - target frames per second (default: 30)
//   kEigs:          number   - how many eigenvalues to plot (default: auto from data)
//   showThreshold:  boolean  - show 2/η line (default: true)
//   clipSharpness:  boolean  - cap y-axis at 3× threshold (default: true)

import { formatTickLabel, baseChartOptions, CHART_FONT } from './chart-utils.js';

const EIG_COLORS = [
  'rgb(220, 50, 50)',    // λ₁ - red
  'rgb(230, 120, 30)',   // λ₂ - orange
  'rgb(80, 180, 80)',    // λ₃ - green
  'rgb(150, 80, 200)',   // λ₄ - purple
  'rgb(40, 130, 180)',   // λ₅ - blue
];

const LOSS_COLOR = 'rgb(40, 130, 130)';

export class HeroPlot {
  constructor(options = {}) {
    this.dataUrl = options.dataUrl || 'runs/title_plot/run.json';
    this.lossCanvasId = options.lossCanvasId || 'heroLossChart';
    this.sharpCanvasId = options.sharpCanvasId || 'heroSharpnessChart';
    this.revealDuration = options.revealDuration || 10;   // seconds
    this.pauseDuration = options.pauseDuration || 3;      // seconds
    this.fps = options.fps || 30;
    this.kEigs = options.kEigs || null;                   // null = auto from data
    this.showThreshold = options.showThreshold !== false;
    this.clipSharpness = options.clipSharpness !== false;

    this.lossChart = null;
    this.sharpnessChart = null;
    this.data = null;
    this.lossData = [];
    this.eigData = [];
    this.eta = 0.01;
    this.threshold = 0;
    this.totalPoints = 0;
    this.revealedPoints = 0;
    this.timerId = null;
    this.state = 'idle';       // 'idle' | 'revealing' | 'paused'
  }

  async init() {
    try {
      const response = await fetch(this.dataUrl);
      if (!response.ok) {
        console.warn(`HeroPlot: failed to load ${this.dataUrl} (${response.status})`);
        return false;
      }
      this.data = await response.json();
    } catch (e) {
      console.warn('HeroPlot: error loading data:', e);
      return false;
    }

    const params = this.data.params || {};
    this.eta = params.eta || 0.01;
    this.threshold = 2 / this.eta;

    // stepInterval: how many actual training steps each data point represents
    const stepInterval = this.data.stepInterval || 1;

    // Parse loss data
    const lossArr = this.data.loss || [];
    this.lossData = lossArr.map((loss, i) => ({ x: (i + 1) * stepInterval, y: loss }));

    // Parse eigenvalue data
    const eigArr = this.data.eigenvalues || [];
    const dataKEigs = eigArr.length > 0 ? (Array.isArray(eigArr[0]) ? eigArr[0].length : 1) : 0;
    const kEigs = this.kEigs || dataKEigs || 1;

    this.eigData = [];
    for (let k = 0; k < kEigs; k++) {
      this.eigData.push(eigArr.map((eigs, i) => {
        const arr = Array.isArray(eigs) ? eigs : [eigs];
        const eigIdx = arr.length - 1 - k;
        return { x: (i + 1) * stepInterval, y: eigIdx >= 0 ? arr[eigIdx] : 0 };
      }));
    }

    this.totalPoints = Math.max(this.lossData.length, this.eigData.length > 0 ? this.eigData[0].length : 0);

    this._createLossChart();
    this._createSharpnessChart(kEigs);
    this._startReveal();

    return true;
  }

  _createLossChart() {
    const canvas = document.getElementById(this.lossCanvasId);
    if (!canvas) return;

    this.lossChart = new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: { datasets: [{
        label: 'train',
        data: [],
        borderColor: LOSS_COLOR,
        backgroundColor: 'rgba(40, 130, 130, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0
      }]},
      options: baseChartOptions()
    });
  }

  _createSharpnessChart(kEigs) {
    const canvas = document.getElementById(this.sharpCanvasId);
    if (!canvas) return;

    const eigLabels = ['λ₁', 'λ₂', 'λ₃'];
    const datasets = [];

    if (this.showThreshold) {
      datasets.push({
        label: '2/η',
        data: [],
        borderColor: 'rgb(0, 0, 0)',
        borderWidth: 3.5,
        borderDash: [8, 4],
        pointRadius: 0,
        tension: 0,
        order: 0
      });
    }

    for (let k = 0; k < kEigs; k++) {
      datasets.push({
        label: eigLabels[k] || `λ${k + 1}`,
        data: [],
        borderColor: EIG_COLORS[k] || EIG_COLORS[0],
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        order: k + 1
      });
    }

    this.sharpnessChart = new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: { datasets },
      options: baseChartOptions()
    });
  }

  _startReveal() {
    this.revealedPoints = 0;
    this.state = 'revealing';
    const frameInterval = 1000 / this.fps;
    const pointsPerFrame = Math.max(1, Math.ceil(this.totalPoints / (this.revealDuration * this.fps)));

    this.timerId = setInterval(() => {
      if (this.state === 'revealing') {
        this.revealedPoints = Math.min(this.revealedPoints + pointsPerFrame, this.totalPoints);
        this._updateCharts();

        if (this.revealedPoints >= this.totalPoints) {
          this.state = 'paused';
          setTimeout(() => {
            this._clearCharts();
            this.state = 'revealing';
            this.revealedPoints = 0;
          }, this.pauseDuration * 1000);
        }
      }
    }, frameInterval);
  }

  _updateCharts() {
    const n = this.revealedPoints;

    if (this.lossChart) {
      this.lossChart.data.datasets[0].data = this.lossData.slice(0, n);
      if (n > 0) this.lossChart.options.scales.x.max = this.lossData[n - 1].x;

      let maxLoss = 0;
      for (let i = 0; i < n; i++) {
        if (this.lossData[i].y > maxLoss) maxLoss = this.lossData[i].y;
      }
      if (maxLoss > 0) {
        const yMax = maxLoss * 1.4;
        this.lossChart.options.scales.y.max = yMax;
        this.lossChart.options.scales.y.ticks.callback = function(value) {
          if (Math.abs(value - yMax) < 1e-10) return '';
          return formatTickLabel(value);
        };
      }
      this.lossChart.update('none');
    }

    if (this.sharpnessChart) {
      const eigN = Math.min(n, this.eigData.length > 0 ? this.eigData[0].length : 0);
      let dsIdx = 0;

      if (this.showThreshold) {
        if (eigN > 0) {
          const lastX = this.eigData[0][eigN - 1].x;
          this.sharpnessChart.data.datasets[dsIdx].data = [
            { x: 0, y: this.threshold },
            { x: lastX, y: this.threshold }
          ];
        }
        dsIdx++;
      }

      let maxEig = this.threshold;
      for (let k = 0; k < this.eigData.length; k++) {
        this.sharpnessChart.data.datasets[dsIdx + k].data = this.eigData[k].slice(0, eigN);
        for (let i = 0; i < eigN; i++) {
          if (this.eigData[k][i].y > maxEig) maxEig = this.eigData[k][i].y;
        }
      }

      if (eigN > 0) this.sharpnessChart.options.scales.x.max = this.eigData[0][eigN - 1].x;

      let yMax = maxEig * 1.3;
      if (this.clipSharpness) yMax = Math.min(yMax, this.threshold * 3);
      this.sharpnessChart.options.scales.y.max = yMax;

      this.sharpnessChart.update('none');
    }
  }

  _clearCharts() {
    if (this.lossChart) {
      this.lossChart.data.datasets[0].data = [];
      this.lossChart.options.scales.x.max = undefined;
      this.lossChart.update('none');
    }
    if (this.sharpnessChart) {
      for (const ds of this.sharpnessChart.data.datasets) ds.data = [];
      this.sharpnessChart.options.scales.x.max = undefined;
      this.sharpnessChart.update('none');
    }
  }

  destroy() {
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
    if (this.lossChart) {
      this.lossChart.destroy();
      this.lossChart = null;
    }
    if (this.sharpnessChart) {
      this.sharpnessChart.destroy();
      this.sharpnessChart = null;
    }
  }
}
