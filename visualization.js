// ============================================================================
// VISUALIZATION - Chart.js loss and sharpness plotting
// ============================================================================
// Both LossChart and RightChart accept an options object in their constructor
// to control which series are included. Only declared series get datasets and
// legend entries.
//
// LossChart options:
//   showTrain:   bool (default true)  - train loss (raw + EMA)
//   showTest:    bool (default true)  - test loss curve
//   showEma:     bool (default true)  - enable EMA smoothing support
//
// RightChart options:
//   kEigs:       number (default 3)   - how many eigenvalue curves (1-3)
//   showThreshold: bool (default true) - the 2/η dashed line

import { IncrementalCache } from './incremental-cache.js';
import { formatTickLabel, baseChartOptions, CHART_FONT } from './chart-utils.js';

const MAX_PLOT_POINTS = 1000;

// ============================================================================
// LOSS CHART
// ============================================================================

export class LossChart {
  /**
   * @param {string} canvasId
   * @param {object} [options]
   * @param {boolean} [options.showTrain=true]
   * @param {boolean} [options.showTest=true]
   * @param {boolean} [options.showEma=true]
   */
  constructor(canvasId, options = {}) {
    this.showTrain = options.showTrain !== false;
    this.showTest  = options.showTest  !== false;
    this.showEma   = options.showEma   !== false;

    this.logScale = false;
    this.logScaleX = false;
    this.useEffectiveTime = false;
    this.eta = 0.01;
    this.emaWindow = 1;

    this.cache = new IncrementalCache(this.emaWindow, MAX_PLOT_POINTS, 'loss', { loss: 0.5 });

    // Build datasets dynamically. Track indices by name.
    this.idx = {};
    const datasets = [];

    if (this.showTrain) {
      this.idx.trainRaw = datasets.length;
      datasets.push({
        label: this.showEma ? 'train (raw)' : 'train',
        data: [],
        borderColor: this.showEma ? 'rgba(40, 130, 130, 0.3)' : 'rgb(40, 130, 130)',
        backgroundColor: this.showEma ? 'rgba(40, 130, 130, 0.05)' : 'rgba(40, 130, 130, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        order: 4
      });
    }

    if (this.showTrain && this.showEma) {
      this.idx.trainEma = datasets.length;
      datasets.push({
        label: 'train (ema)',
        data: [],
        borderColor: 'rgb(40, 130, 130)',
        backgroundColor: 'rgba(40, 130, 130, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        order: 3,
        hidden: true
      });
    }

    if (this.showTest) {
      this.idx.test = datasets.length;
      datasets.push({
        label: 'test',
        data: [],
        borderColor: 'rgb(220, 80, 80)',
        backgroundColor: 'rgba(220, 80, 80, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        borderDash: [6, 3],
        order: 2,
        hidden: true
      });
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: baseChartOptions()
    });

    // Custom legend: only show series that have data or are relevant
    const chartRef = this;
    this.chart.options.plugins.legend.labels.generateLabels = function(chart) {
      const labels = [];
      if (!chartRef.showTrain) return labels;

      const emaOn = chartRef.showEma && chartRef.emaWindow > 1;

      if (emaOn) {
        labels.push({
          text: 'train (raw)',
          strokeStyle: 'rgba(40, 130, 130, 0.3)',
          fillStyle: 'rgba(40, 130, 130, 0.3)',
          lineWidth: 2, hidden: false
        });
        labels.push({
          text: 'train (ema)',
          strokeStyle: 'rgb(40, 130, 130)',
          fillStyle: 'rgb(40, 130, 130)',
          lineWidth: 2, hidden: false
        });
      } else {
        labels.push({
          text: 'train',
          strokeStyle: 'rgb(40, 130, 130)',
          fillStyle: 'rgb(40, 130, 130)',
          lineWidth: 2, hidden: false
        });
      }

      if (chartRef.showTest && chartRef.idx.test !== undefined) {
        const hasTestData = chart.data.datasets[chartRef.idx.test].data.length > 0;
        if (hasTestData) {
          labels.push({
            text: 'test',
            strokeStyle: 'rgb(220, 80, 80)',
            fillStyle: 'rgb(220, 80, 80)',
            lineWidth: 2, lineDash: [5, 3], hidden: false
          });
        }
      }

      return labels;
    };
  }

  setLogScale(useLog) {
    this.logScale = useLog;
    this.chart.options.scales.y.type = useLog ? 'logarithmic' : 'linear';
    this.chart.update('none');
  }

  setLogScaleX(useLogX) {
    this.logScaleX = useLogX;
    this.chart.options.scales.x.type = useLogX ? 'logarithmic' : 'linear';
    this.chart.options.scales.x.min = useLogX ? 1 : 0;
    this.chart.update('none');
  }

  setEffectiveTime(useEffTime, eta) {
    this.useEffectiveTime = useEffTime;
    this.eta = eta;
    if (this.logScaleX) {
      this.chart.options.scales.x.min = useEffTime ? 0.001 : 1;
    }
  }

  setEmaWindow(window) {
    this.emaWindow = window;
    this.cache.setEmaWindow(window);
  }

  setInitialLoss(initialLoss) {
    this.cache.initEmaValues = { loss: initialLoss };
    this.cache.lastEmaValues = { loss: initialLoss };
  }

  update(lossHistory, testLossHistory = [], eta = this.eta) {
    if (lossHistory.length === 0) return;
    this.eta = eta;

    const { downsampledRaw, downsampledSmoothed, max } = this.cache.update(lossHistory);
    const toX = (iter) => this.useEffectiveTime ? iter * this.eta : iter;

    // Train raw
    if (this.idx.trainRaw !== undefined) {
      const rawData = downsampledRaw.map(p => ({ x: toX(p.iteration), y: p.loss }));
      this.chart.data.datasets[this.idx.trainRaw].data = rawData;

      // Style depends on whether EMA is active
      if (this.showEma && this.emaWindow > 1) {
        this.chart.data.datasets[this.idx.trainRaw].borderColor = 'rgba(40, 130, 130, 0.3)';
        this.chart.data.datasets[this.idx.trainRaw].backgroundColor = 'rgba(40, 130, 130, 0.05)';
      } else {
        this.chart.data.datasets[this.idx.trainRaw].borderColor = 'rgb(40, 130, 130)';
        this.chart.data.datasets[this.idx.trainRaw].backgroundColor = 'rgba(40, 130, 130, 0.1)';
      }
    }

    // Train EMA
    if (this.idx.trainEma !== undefined) {
      if (this.emaWindow > 1) {
        const smoothedData = downsampledSmoothed.map(p => ({ x: toX(p.iteration), y: p.loss }));
        this.chart.data.datasets[this.idx.trainEma].data = smoothedData;
        this.chart.data.datasets[this.idx.trainEma].hidden = false;
      } else {
        this.chart.data.datasets[this.idx.trainEma].data = [];
        this.chart.data.datasets[this.idx.trainEma].hidden = true;
      }
    }

    // Test loss
    if (this.idx.test !== undefined) {
      if (testLossHistory && testLossHistory.length > 0) {
        const testData = testLossHistory.map(p => ({ x: toX(p.iteration), y: p.loss }));
        this.chart.data.datasets[this.idx.test].data = testData;
        this.chart.data.datasets[this.idx.test].hidden = false;
      } else {
        this.chart.data.datasets[this.idx.test].data = [];
        this.chart.data.datasets[this.idx.test].hidden = true;
      }
    }

    // X-axis max
    const currentIteration = lossHistory[lossHistory.length - 1].iteration;
    this.chart.options.scales.x.max = toX(currentIteration);

    // Y-axis max
    if (!this.logScale) {
      let maxLoss = max.loss;
      if (this.idx.test !== undefined && testLossHistory && testLossHistory.length > 0) {
        for (const p of testLossHistory) {
          if (p.loss > maxLoss) maxLoss = p.loss;
        }
      }
      const yMax = maxLoss * 1.4;
      this.chart.options.scales.y.max = yMax;
      this.chart.options.scales.y.ticks.callback = function(value) {
        if (Math.abs(value - yMax) < 1e-10) return '';
        return formatTickLabel(value);
      };
    } else {
      this.chart.options.scales.y.max = undefined;
      this.chart.options.scales.y.ticks.callback = function(value) {
        return formatTickLabel(value);
      };
    }

    this.chart.update('none');
  }

  clear() {
    for (const ds of this.chart.data.datasets) {
      ds.data = [];
    }
    this.chart.options.scales.x.max = undefined;
    this.cache.clear();
    this.chart.update('none');
  }
}

// ============================================================================
// GRAD PROJECTION CHART - Cosine similarity + gradient norm post-EoS capture
// ============================================================================
// Only receives data after the top eigenvector has been captured (when sharpness
// first enters the proximity band around 2/η). Before that, the chart sits empty
// with a placeholder message. Once data arrives it plots two series:
//   - projection  (cosine similarity, left y-axis, [-1, 1])
//   - gradNorm    (gradient ‖g‖, right y-axis)

export class GradProjectionChart {
  /**
   * @param {string} canvasId
   * @param {object} [options]
   * @param {string}  [options.placeholderText] - text shown before capture
   */
  constructor(canvasId, options = {}) {
    this.canvasId = canvasId;
    this.placeholderText = options.placeholderText || 'Waiting for sharpness to reach 2/η…';

    this.logScaleX = false;
    this.useEffectiveTime = false;
    this.eta = 0.01;

    // Two datasets: cosine similarity (left axis) and grad norm (right axis)
    const datasets = [
      {
        label: 'cos(∇L, v̂)',
        data: [],
        borderColor: 'rgb(100, 60, 200)',
        backgroundColor: 'rgba(100, 60, 200, 0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0,
        yAxisID: 'yLeft',
        order: 1
      },
      {
        label: '‖∇L‖',
        data: [],
        borderColor: 'rgb(200, 120, 30)',
        backgroundColor: 'rgba(200, 120, 30, 0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0,
        yAxisID: 'yRight',
        order: 2,
        hidden: true
      }
    ];

    const opts = baseChartOptions();

    // Override scales for dual-axis layout
    opts.scales = {
      x: {
        type: 'linear',
        min: 0,
        ticks: {
          maxRotation: 0,
          font: { size: 14, ...CHART_FONT },
          callback: function(value) { return formatTickLabel(value); }
        }
      },
      yLeft: {
        type: 'linear',
        position: 'left',
        min: -1,
        max: 1,
        ticks: {
          font: { size: 14, ...CHART_FONT },
          callback: function(value) { return formatTickLabel(value); }
        },
        grid: { drawOnChartArea: true }
      },
      yRight: {
        type: 'linear',
        position: 'right',
        beginAtZero: true,
        ticks: {
          font: { size: 14, ...CHART_FONT },
          callback: function(value) { return formatTickLabel(value); }
        },
        grid: { drawOnChartArea: false }
      }
    };

    const placeholderText = this.placeholderText;
    const placeholderPlugin = {
      id: 'gradProjectionPlaceholder',
      afterDraw: (chart) => {
        if (chart.data.datasets[0].data.length > 0) return;
        const { ctx: c, chartArea } = chart;
        if (!chartArea) return;
        c.save();
        c.textAlign = 'center';
        c.textBaseline = 'middle';
        c.fillStyle = '#aaa';
        c.font = `13px Monaco, Consolas, monospace`;
        const cx = (chartArea.left + chartArea.right) / 2;
        const cy = (chartArea.top + chartArea.bottom) / 2;
        c.fillText(placeholderText, cx, cy);
        c.restore();
      }
    };

    const ctx = document.getElementById(canvasId).getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: opts,
      plugins: [placeholderPlugin]
    });
    this.chart.update('none');
  }

  setLogScaleX(useLogX) {
    this.logScaleX = useLogX;
    this.chart.options.scales.x.type = useLogX ? 'logarithmic' : 'linear';
    this.chart.options.scales.x.min = useLogX ? 1 : 0;
    this.chart.update('none');
  }

  setEffectiveTime(useEffTime, eta) {
    this.useEffectiveTime = useEffTime;
    this.eta = eta;
  }

  /**
   * @param {Array<{iteration: number, projection: number, gradNorm: number}>} history
   * @param {number} eta
   */
  update(history, eta = this.eta) {
    if (!history || history.length === 0) return;
    this.eta = eta;

    const toX = (iter) => this.useEffectiveTime ? iter * this.eta : iter;

    this.chart.data.datasets[0].data = history.map(p => ({ x: toX(p.iteration), y: p.projection }));
    this.chart.data.datasets[1].data = history.map(p => ({ x: toX(p.iteration), y: p.gradNorm }));

    // X-axis: start from the first recorded iteration, not zero
    const firstX = toX(history[0].iteration);
    const lastX = toX(history[history.length - 1].iteration);
    this.chart.options.scales.x.min = firstX;
    this.chart.options.scales.x.max = lastX;

    // Right y-axis max (grad norm): auto with 20% headroom
    let maxNorm = 0;
    for (const p of history) {
      if (p.gradNorm > maxNorm) maxNorm = p.gradNorm;
    }
    this.chart.options.scales.yRight.max = maxNorm * 1.2 || 1;

    this.chart.update('none');
  }

  clear() {
    this.chart.data.datasets[0].data = [];
    this.chart.data.datasets[1].data = [];
    this.chart.options.scales.x.min = 0;
    this.chart.options.scales.x.max = undefined;
    this.chart.update('none');
  }
}

// ============================================================================
// RIGHT CHART - Hessian eigenvalues + 2/η stability threshold
// ============================================================================

export class RightChart {
  /**
   * @param {string} canvasId
   * @param {object} [options]
   * @param {number} [options.kEigs=3] - number of eigenvalue curves to show (1-3)
   * @param {boolean} [options.showThreshold=true] - show the 2/η dashed line
   */
  constructor(canvasId, options = {}) {
    this.kEigs = options.kEigs || 3;
    this.showThreshold = options.showThreshold !== false;
    this.clipSharpness = options.clipSharpness !== false; // default: true

    this.logScale = false;
    this.logScaleX = false;
    this.useEffectiveTime = false;
    this.eta = 0.01;

    const eigColors = [
      'rgb(220, 50, 50)',    // λ₁ (largest) - red
      'rgb(230, 120, 30)',   // λ₂ - orange
      'rgb(80, 180, 80)',    // λ₃ - green
      'rgb(150, 80, 200)',   // λ₄ - purple
      'rgb(40, 130, 180)',   // λ₅ - blue
    ];
    const eigLabels = ['λ₁', 'λ₂', 'λ₃', 'λ₄', 'λ₅'];

    // Build datasets dynamically. Track indices by name.
    this.idx = {};
    const datasets = [];

    if (this.showThreshold) {
      this.idx.threshold = datasets.length;
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

    this.idx.eigs = [];
    for (let i = 0; i < this.kEigs; i++) {
      this.idx.eigs.push(datasets.length);
      datasets.push({
        label: eigLabels[i] || `λ${i + 1}`,
        data: [],
        borderColor: eigColors[i] || eigColors[0],
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        order: i + 1
      });
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: baseChartOptions()
    });
  }

  setLogScale(useLog) {
    this.logScale = useLog;
    this.chart.options.scales.y.type = useLog ? 'logarithmic' : 'linear';
    this.chart.update('none');
  }

  setLogScaleX(useLogX) {
    this.logScaleX = useLogX;
    this.chart.options.scales.x.type = useLogX ? 'logarithmic' : 'linear';
    this.chart.options.scales.x.min = useLogX ? 1 : 0;
    this.chart.update('none');
  }

  setEffectiveTime(useEffTime, eta) {
    this.useEffectiveTime = useEffTime;
    this.eta = eta;
  }

  setClipSharpness(clip) {
    this.clipSharpness = clip;
    // Force redraw on next update
    this.chart.update('none');
  }

  /**
   * Update the chart with eigenvalue history.
   * @param {Array<{iteration: number, eigs: number[]}>} eigenvalueHistory
   * @param {number} eta - learning rate (for 2/η line)
   */
  update(eigenvalueHistory, eta = this.eta) {
    this.eta = eta;
    if (!eigenvalueHistory || eigenvalueHistory.length === 0) return;

    const toX = (iter) => this.useEffectiveTime ? iter * this.eta : iter;
    const threshold = 2 / this.eta;

    // Number of eigenvalues in the data (may differ from this.kEigs)
    const dataKEigs = eigenvalueHistory[0].eigs.length;

    // 2/η threshold line
    if (this.idx.threshold !== undefined) {
      const firstX = toX(eigenvalueHistory[0].iteration);
      const lastX = toX(eigenvalueHistory[eigenvalueHistory.length - 1].iteration);
      this.chart.data.datasets[this.idx.threshold].data = [
        { x: firstX, y: threshold },
        { x: lastX, y: threshold }
      ];
    }

    // Eigenvalue curves: eigs are sorted ascending, so eigs[k-1] is the largest
    for (let eigIdx = 0; eigIdx < this.kEigs; eigIdx++) {
      const dsIdx = this.idx.eigs[eigIdx];
      if (eigIdx < dataKEigs) {
        const eigArrayIdx = dataKEigs - 1 - eigIdx;
        const data = eigenvalueHistory.map(point => ({
          x: toX(point.iteration),
          y: point.eigs[eigArrayIdx]
        }));
        this.chart.data.datasets[dsIdx].data = data;
        this.chart.data.datasets[dsIdx].hidden = false;
      } else {
        this.chart.data.datasets[dsIdx].data = [];
        this.chart.data.datasets[dsIdx].hidden = true;
      }
    }

    // X-axis max
    const lastX = toX(eigenvalueHistory[eigenvalueHistory.length - 1].iteration);
    this.chart.options.scales.x.max = lastX;

    // Y-axis: auto-scale with optional clip at 3× threshold
    if (!this.logScale) {
      let maxEig = threshold;
      for (const point of eigenvalueHistory) {
        for (const e of point.eigs) {
          if (e > maxEig) maxEig = e;
        }
      }
      let yMax = maxEig * 1.3;
      if (this.clipSharpness) {
        yMax = Math.min(yMax, threshold * 3);
      }
      this.chart.options.scales.y.max = yMax;
    } else {
      this.chart.options.scales.y.max = undefined;
    }

    this.chart.update('none');
  }

  clear() {
    for (const ds of this.chart.data.datasets) {
      ds.data = [];
    }
    this.chart.options.scales.x.max = undefined;
    this.chart.update('none');
  }
}

// ============================================================================
// LINE CHART — generic single-series line plot against iteration
// ============================================================================
// Used by the deep scalar network widget (and any future widget that needs a
// simple companion chart). Intentionally minimal: one series, optional
// horizontal reference line, optional log y-axis.
//
// Options:
//   label      string   — dataset legend label
//   color      string   — CSS color for the line
//   logY       bool     — log scale on y-axis (default false)
//   refLine    number   — if set, draws a dashed horizontal line at this value
//   refLabel   string   — legend label for the ref line (default '2/η')
//   yLabel     string   — y-axis label text drawn via plugin (optional)

export class LineChart {
  constructor(canvasId, options = {}) {
    this.logY     = options.logY     || false;
    this.refLine  = options.refLine  !== undefined ? options.refLine : null;
    this.refLabel = options.refLabel || '2/η';

    // Create a ref line dataset if either refLine value or refLabel was supplied
    this.hasRefLine = options.refLine !== undefined || options.refLabel !== undefined;

    const datasets = [
      {
        label: options.label || 'value',
        data: [],
        borderColor: options.color || 'rgb(40, 130, 130)',
        backgroundColor: 'rgba(0,0,0,0)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        order: 1
      }
    ];

    if (this.hasRefLine) {
      datasets.push({
        label: this.refLabel,
        data: [],
        borderColor: 'rgb(0,0,0)',
        borderWidth: 2,
        borderDash: [8, 4],
        pointRadius: 0,
        tension: 0,
        order: 0
      });
    }

    const opts = baseChartOptions();
    if (this.logY) {
      opts.scales.y.type = 'logarithmic';
      opts.scales.y.beginAtZero = false;
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    this.chart = new Chart(ctx, { type: 'line', data: { datasets }, options: opts });
  }

  /**
   * @param {Array<{iteration: number, value: number}>} history
   * @param {number} [refLineValue] — updated reference line value (e.g. 2/η changes with η)
   */
  update(history, refLineValue) {
    if (!history || history.length === 0) return;

    this.chart.data.datasets[0].data = history.map(p => ({ x: p.iteration, y: p.value }));

    // Update reference line if present
    if (this.hasRefLine && refLineValue !== undefined) {
      const firstX = history[0].iteration;
      const lastX  = history[history.length - 1].iteration;
      this.chart.data.datasets[1].data = [{ x: firstX, y: refLineValue }, { x: lastX, y: refLineValue }];
    }

    this.chart.options.scales.x.max = history[history.length - 1].iteration;
    this.chart.update('none');
  }

  /** Update only the reference line value without redrawing series data */
  setRefLine(value) {
    this.refLine = value;
  }

  clear() {
    for (const ds of this.chart.data.datasets) ds.data = [];
    this.chart.options.scales.x.max = undefined;
    this.chart.update('none');
  }
}
