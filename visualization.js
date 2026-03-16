// ============================================================================
// VISUALIZATION - Chart.js loss and norm plotting
// ============================================================================

import { IncrementalCache } from './incremental-cache.js';

function formatTickLabel(value) {
  if (value === 0) return '0';
  const abs = Math.abs(value);
  // Use integer form if large enough and close to integer
  if (abs >= 1 && Math.abs(value - Math.round(value)) < 0.01) {
    return String(Math.round(value));
  }
  // 4 significant figures
  return parseFloat(value.toPrecision(4)).toString();
}

const MAX_PLOT_POINTS = 1000;

// ============================================================================
// LOSS CHART - train loss + optional test loss
// ============================================================================

export class LossChart {
  constructor(canvasId) {
    this.logScale = false;
    this.logScaleX = false;
    this.useEffectiveTime = false;
    this.eta = 0.01;
    this.emaWindow = 1;

    this.cache = new IncrementalCache(this.emaWindow, MAX_PLOT_POINTS, 'loss', { loss: 0.5 });

    const ctx = document.getElementById(canvasId).getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'train (raw)',
            data: [],
            borderColor: 'rgba(40, 130, 130, 0.3)',
            backgroundColor: 'rgba(40, 130, 130, 0.05)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0,
            order: 4
          },
          {
            label: 'train (ema)',
            data: [],
            borderColor: 'rgb(40, 130, 130)',
            backgroundColor: 'rgba(40, 130, 130, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0,
            order: 3
          },
          {
            label: 'test',
            data: [],
            borderColor: 'rgb(220, 80, 80)',
            backgroundColor: 'rgba(220, 80, 80, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0,
            borderDash: [6, 3],
            order: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            type: 'linear',
            min: 0,
            ticks: {
              maxRotation: 0,
              font: { size: 14, family: 'Monaco, Consolas, "Courier New", monospace' },
              callback: function(value) { return formatTickLabel(value); }
            }
          },
          y: {
            type: 'linear',
            beginAtZero: true,
            ticks: {
              font: { size: 14, family: 'Monaco, Consolas, "Courier New", monospace' },
              callback: function(value) { return formatTickLabel(value); }
            }
          }
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            align: 'start',
            onClick: () => {},
            labels: {
              usePointStyle: false,
              boxWidth: 40,
              boxHeight: 2,
              font: { size: 12, family: 'Monaco, Consolas, "Courier New", monospace' }
            }
          }
        }
      }
    });

    // Set up legend labels. We use generateLabels to control what appears.
    const chartRef = this;
    this.chart.options.plugins.legend.labels.generateLabels = function(chart) {
      const emaOn = chartRef.emaWindow > 1;
      const hasTest = chart.data.datasets[2].data.length > 0;
      const labels = [];

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
        if (hasTest) {
          labels.push({
            text: 'test',
            strokeStyle: 'rgb(220, 80, 80)',
            fillStyle: 'rgb(220, 80, 80)',
            lineWidth: 2, lineDash: [5, 3], hidden: false
          });
        }
      } else {
        labels.push({
          text: 'train',
          strokeStyle: 'rgb(40, 130, 130)',
          fillStyle: 'rgb(40, 130, 130)',
          lineWidth: 2, hidden: false
        });
        if (hasTest) {
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

    // Raw train data
    const rawData = downsampledRaw.map(point => ({
      x: this.useEffectiveTime ? point.iteration * this.eta : point.iteration,
      y: point.loss
    }));
    this.chart.data.datasets[0].data = rawData;

    // EMA train data
    if (this.emaWindow > 1) {
      const smoothedData = downsampledSmoothed.map(point => ({
        x: this.useEffectiveTime ? point.iteration * this.eta : point.iteration,
        y: point.loss
      }));
      this.chart.data.datasets[1].data = smoothedData;
      this.chart.data.datasets[1].hidden = false;
      this.chart.data.datasets[0].borderColor = 'rgba(40, 130, 130, 0.3)';
      this.chart.data.datasets[0].backgroundColor = 'rgba(40, 130, 130, 0.05)';
    } else {
      this.chart.data.datasets[1].data = [];
      this.chart.data.datasets[1].hidden = true;
      this.chart.data.datasets[0].borderColor = 'rgb(40, 130, 130)';
      this.chart.data.datasets[0].backgroundColor = 'rgba(40, 130, 130, 0.1)';
    }

    // Test loss
    if (testLossHistory && testLossHistory.length > 0) {
      const testData = testLossHistory.map(point => ({
        x: this.useEffectiveTime ? point.iteration * this.eta : point.iteration,
        y: point.loss
      }));
      this.chart.data.datasets[2].data = testData;
      this.chart.data.datasets[2].hidden = false;
    } else {
      this.chart.data.datasets[2].data = [];
      this.chart.data.datasets[2].hidden = true;
    }

    // X-axis max
    const currentIteration = lossHistory[lossHistory.length - 1].iteration;
    const currentMax = this.useEffectiveTime ? currentIteration * this.eta : currentIteration;
    this.chart.options.scales.x.max = currentMax;

    // Y-axis max
    if (!this.logScale) {
      let maxLoss = max.loss;
      if (testLossHistory && testLossHistory.length > 0) {
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
    this.chart.data.datasets[0].data = [];
    this.chart.data.datasets[1].data = [];
    this.chart.data.datasets[2].data = [];
    this.chart.options.scales.x.max = undefined;
    this.cache.clear();
    this.chart.update('none');
  }
}

// ============================================================================
// RIGHT CHART - Hessian eigenvalues + 2/η stability threshold
// ============================================================================

export class RightChart {
  constructor(canvasId) {
    this.logScale = false;
    this.logScaleX = false;
    this.useEffectiveTime = false;
    this.eta = 0.01;

    // Colors for eigenvalue curves
    this.eigColors = [
      'rgb(220, 50, 50)',    // λ₁ (largest) - red
      'rgb(230, 120, 30)',   // λ₂ - orange
      'rgb(80, 180, 80)',    // λ₃ - green
    ];

    const ctx = document.getElementById(canvasId).getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          // Dataset 0: 2/η threshold line (dashed black, thick)
          {
            label: '2/η',
            data: [],
            borderColor: 'rgb(0, 0, 0)',
            borderWidth: 3.5,
            borderDash: [8, 4],
            pointRadius: 0,
            tension: 0,
            order: 0
          },
          // Datasets 1..k: eigenvalue curves
          {
            label: 'λ₁',
            data: [],
            borderColor: this.eigColors[0],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0,
            order: 1
          },
          {
            label: 'λ₂',
            data: [],
            borderColor: this.eigColors[1],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0,
            order: 2
          },
          {
            label: 'λ₃',
            data: [],
            borderColor: this.eigColors[2],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0,
            order: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            type: 'linear',
            min: 0,
            ticks: {
              maxRotation: 0,
              font: { size: 14, family: 'Monaco, Consolas, "Courier New", monospace' },
              callback: function(value) { return formatTickLabel(value); }
            }
          },
          y: {
            type: 'linear',
            beginAtZero: true,
            ticks: {
              font: { size: 14, family: 'Monaco, Consolas, "Courier New", monospace' },
              callback: function(value) { return formatTickLabel(value); }
            }
          }
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            align: 'start',
            onClick: () => {},
            labels: {
              usePointStyle: false,
              boxWidth: 40,
              boxHeight: 2,
              font: { size: 12, family: 'Monaco, Consolas, "Courier New", monospace' }
            }
          }
        }
      }
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

    // Number of eigenvalues tracked (up to 3)
    const kEigs = eigenvalueHistory[0].eigs.length;

    // 2/η threshold line: horizontal line spanning the full x range
    const firstX = toX(eigenvalueHistory[0].iteration);
    const lastX = toX(eigenvalueHistory[eigenvalueHistory.length - 1].iteration);
    this.chart.data.datasets[0].data = [
      { x: firstX, y: threshold },
      { x: lastX, y: threshold }
    ];

    // Eigenvalue curves: eigs are sorted ascending, so eigs[k-1] is the largest
    for (let eigIdx = 0; eigIdx < 3; eigIdx++) {
      if (eigIdx < kEigs) {
        // Map so that dataset 1 = largest eigenvalue, dataset 2 = second largest, etc.
        const eigArrayIdx = kEigs - 1 - eigIdx;
        const data = eigenvalueHistory.map(point => ({
          x: toX(point.iteration),
          y: point.eigs[eigArrayIdx]
        }));
        this.chart.data.datasets[eigIdx + 1].data = data;
        this.chart.data.datasets[eigIdx + 1].hidden = false;
      } else {
        this.chart.data.datasets[eigIdx + 1].data = [];
        this.chart.data.datasets[eigIdx + 1].hidden = true;
      }
    }

    // X-axis max
    this.chart.options.scales.x.max = lastX;

    // Y-axis: show threshold + some headroom
    if (!this.logScale) {
      let maxEig = threshold;
      for (const point of eigenvalueHistory) {
        for (const e of point.eigs) {
          if (e > maxEig) maxEig = e;
        }
      }
      this.chart.options.scales.y.max = maxEig * 1.3;
    } else {
      this.chart.options.scales.y.max = undefined;
    }

    this.chart.update('none');
  }

  clear() {
    for (let i = 0; i < this.chart.data.datasets.length; i++) {
      this.chart.data.datasets[i].data = [];
    }
    this.chart.options.scales.x.max = undefined;
    this.chart.update('none');
  }
}
