// ============================================================================
// UI CHARTS - Chart initialization and controls
// ============================================================================

import { LossChart, NormChart } from '../visualization.js';

/**
 * ChartsManager - handles chart initialization and controls
 */
export class ChartsManager {
  constructor(appState) {
    this.appState = appState;

    // Initialize charts
    this.lossChart = new LossChart('lossChart');
    this.normChart = new NormChart('normChart');

    // Apply saved settings
    this.lossChart.setLogScale(appState.logScale);
    this.normChart.setLogScale(appState.logScale);
    this.lossChart.setLogScaleX(appState.logScaleX);
    this.normChart.setLogScaleX(appState.logScaleX);
    this.lossChart.setShowTheory(appState.showTheory || false);
    this.normChart.setShowTheory(appState.showTheory || false);
    this.lossChart.setEffectiveTime(appState.xAxisMode === 'teff', appState.eta);
    this.normChart.setEffectiveTime(appState.xAxisMode === 'teff', appState.eta);
    this.lossChart.setEmaWindow(appState.emaWindow);
    this.normChart.setEmaWindow(appState.emaWindow);
  }

  /**
   * Setup chart controls (logscale, x-axis mode, EMA, theory)
   */
  setupControls(controlsManager, simulation) {
    this.simulation = simulation;  // Store reference for EMA updates
    this.controlsManager = controlsManager;  // Store reference for parameter access
    this.setupLogScaleXControl();
    this.setupLogScaleControl();
    this.setupShowTheoryControl();
    this.setupXAxisModeControl(controlsManager);
    this.setupEmaControl();
  }

  /**
   * Setup logscale x-axis checkbox
   */
  setupLogScaleXControl() {
    const checkbox = document.getElementById('logScaleXCheckbox');
    checkbox.checked = this.appState.logScaleX;

    checkbox.addEventListener('change', (e) => {
      const useLogX = e.target.checked;
      this.lossChart.setLogScaleX(useLogX);
      this.normChart.setLogScaleX(useLogX);
      this.appState.logScaleX = useLogX;
      this.appState.save();
    });
  }

  /**
   * Setup logscale y-axis checkbox
   */
  setupLogScaleControl() {
    const checkbox = document.getElementById('logScaleCheckbox');
    checkbox.checked = this.appState.logScale;

    checkbox.addEventListener('change', (e) => {
      const useLog = e.target.checked;
      this.lossChart.setLogScale(useLog);
      this.normChart.setLogScale(useLog);
      this.appState.logScale = useLog;
      this.appState.save();
    });
  }

  /**
   * Setup show theory checkbox
   */
  setupShowTheoryControl() {
    const checkbox = document.getElementById('showTheoryCheckbox');
    checkbox.checked = this.appState.showTheory || false;

    // If theory is enabled on page load, compute and show it (preview mode)
    if (checkbox.checked) {
      this.lossChart.setShowTheory(true);
      this.normChart.setShowTheory(true);
      this.computeAndShowTheoryOnly();
    }

    checkbox.addEventListener('change', (e) => {
      const show = e.target.checked;
      this.lossChart.setShowTheory(show);
      this.normChart.setShowTheory(show);
      this.appState.showTheory = show;
      this.appState.save();

      // THEORY VISIBILITY LOGIC:
      // - If simulation has data: show/hide existing theory (already computed with captured params)
      // - If no data yet: compute fresh theory using current slider values (preview mode)

      if (this.simulation) {
        const state = this.simulation.getState();
        const simulationHasData = state.lossHistory.length > 0;

        if (simulationHasData) {
          // Simulation has run - theory already exists with captured params
          // Just show/hide it by updating the charts
          const eta = this.simulation.params ? this.simulation.params.eta : this.appState.eta;
          this.lossChart.update(state.lossHistory, state.theoryLossHistory || [], eta);
          this.normChart.update(state.normHistory, state.theoryNormHistory || [], state.d, eta);
        } else if (show) {
          // Preview mode: no data yet, checkbox enabled -> compute theory with current params
          this.computeAndShowTheoryOnly();
        } else {
          // Preview mode: no data yet, checkbox disabled -> clear preview
          this.lossChart.clear();
          this.normChart.clear();
        }
      } else if (show) {
        // No simulation exists yet -> compute preview
        this.computeAndShowTheoryOnly();
      } else {
        // No simulation exists, checkbox disabled -> clear
        this.lossChart.clear();
        this.normChart.clear();
      }

      // Force legend update
      this.lossChart.chart.update('none');
      this.normChart.chart.update('none');
    });
  }

  /**
   * Compute theory and show it without running simulation
   */
  computeAndShowTheoryOnly() {
    // Get current parameters from controls
    const params = this.controlsManager.getCurrentParams();

    // Import theory computation and ODE solver
    import('../core/theory.js').then(({ calculateAllTheory }) => {
      import('../core/ode-solver.js').then(({ solveODE }) => {
        import('../core/dynamics.js').then(({ computeDerivatives, computeLoss }) => {
          const numTerms = params.alphas.length;
          const d = params.d;

          // Compute c values for all terms
          const cValues = [];
          for (let termIdx = 0; termIdx < numTerms; termIdx++) {
            const theory = calculateAllTheory(params.gammas, params.alphas[termIdx], d);
            cValues.push(theory.c);
          }

          // Initial conditions: y = [a₁, ..., aₐ, b₁, ..., bₙ]
          const y0 = [
            ...Array(d).fill(1),
            ...Array(numTerms).fill(1e-10)
          ];

          // Define derivative function
          const derivativeFn = (t, y) => computeDerivatives(t, y, params.alphas, cValues, d);

          // Adaptive search for convergence time
          const threshold = 0.001;
          let tMax = 1000;
          const maxTMax = 50000;
          const baseDt = 0.1;
          const dtScaleThreshold = 1000;
          let tConvergence = null;

          while (tMax <= maxTMax) {
            const dt = tMax > dtScaleThreshold
              ? baseDt * (tMax / dtScaleThreshold)
              : baseDt;

            const result = solveODE(derivativeFn, y0, tMax, dt);

            // Check final loss
            const finalY = [];
            for (let i = 0; i < y0.length; i++) {
              finalY.push(result.trajectories[i][result.trajectories[i].length - 1]);
            }
            const finalLoss = computeLoss(finalY, params.alphas, cValues, d);

            if (finalLoss < threshold) {
              // Find exact convergence time
              for (let idx = result.times.length - 1; idx >= 0; idx--) {
                const y = [];
                for (let i = 0; i < y0.length; i++) {
                  y.push(result.trajectories[i][idx]);
                }
                const loss = computeLoss(y, params.alphas, cValues, d);

                if (loss >= threshold) {
                  tConvergence = result.times[Math.min(idx + 1, result.times.length - 1)];
                  break;
                }
              }
              if (tConvergence === null) {
                tConvergence = result.times[0];
              }
              break;
            }

            tMax *= 2;
          }

          if (tConvergence === null) {
            tConvergence = maxTMax;
          }

          // Determine plot duration
          const tMaxPlot = numTerms === 1 ? 2 * tConvergence : 1.5 * tConvergence;

          // Final solve
          const dtFinal = tMaxPlot > dtScaleThreshold
            ? baseDt * (tMaxPlot / dtScaleThreshold)
            : baseDt;
          const finalResult = solveODE(derivativeFn, y0, tMaxPlot, dtFinal);

          // Extract loss trajectory
          const theoryLossHistory = finalResult.times.map((t, idx) => {
            const y = [];
            for (let i = 0; i < y0.length; i++) {
              y.push(finalResult.trajectories[i][idx]);
            }
            const loss = computeLoss(y, params.alphas, cValues, d);

            return {
              iteration: t / params.eta,
              loss: loss
            };
          });

          // Extract norm trajectories
          const theoryNormHistory = finalResult.times.map((t, idx) => {
            const dataPoint = { iteration: t / params.eta };

            // W1 diagonal elements (a_i)
            for (let i = 0; i < d; i++) {
              dataPoint[`w1_${i}`] = finalResult.trajectories[i][idx];
            }

            // Individual b values (b, b', b'')
            for (let j = 0; j < numTerms; j++) {
              dataPoint[`b_${j}`] = finalResult.trajectories[d + j][idx];
            }

            // W2 norm: √(Σⱼ bⱼ²) (no √k factor for theory)
            let sumBSquared = 0;
            for (let j = 0; j < numTerms; j++) {
              const b = finalResult.trajectories[d + j][idx];
              sumBSquared += b * b;
            }
            dataPoint.w2NormNormalized = Math.sqrt(sumBSquared);

            return dataPoint;
          });

          // Create dummy experiment data (single point at origin)
          const dummyLossHistory = [{ iteration: 0, loss: 0.5 * numTerms }];
          const dummyNormHistory = [{
            iteration: 0,
            w2NormNormalized: 0,
            ...Object.fromEntries(Array.from({length: d}, (_, i) => [`w1_${i}`, 1]))
          }];

          // Update charts
          this.lossChart.update(dummyLossHistory, theoryLossHistory, params.eta);
          this.normChart.update(dummyNormHistory, theoryNormHistory, d, params.eta);
        });
      });
    });
  }

  /**
   * Setup x-axis mode toggle (step vs teff)
   */
  setupXAxisModeControl(controlsManager) {
    const stepLink = document.getElementById('step-link');
    const teffLink = document.getElementById('teff-link');

    // Set initial state
    if (this.appState.xAxisMode === 'teff') {
      stepLink.classList.remove('active');
      teffLink.classList.add('active');
    } else {
      stepLink.classList.add('active');
      teffLink.classList.remove('active');
    }

    // Create global function for onclick handlers
    window.setXAxisMode = (mode) => {
      // Use simulation's captured eta if available, otherwise fall back to current slider value
      const eta = (this.simulation && this.simulation.params) ? this.simulation.params.eta : controlsManager.appState.eta;

      if (mode === 'step') {
        stepLink.classList.add('active');
        teffLink.classList.remove('active');
        this.lossChart.setEffectiveTime(false, eta);
        this.normChart.setEffectiveTime(false, eta);
        this.appState.xAxisMode = 'step';

        // Update x-axis labels
        document.querySelectorAll('.x-axis-label').forEach(label => {
          label.textContent = 'step';
        });

        // Replot with new x-axis - always replot if we have data
        if (this.simulation) {
          const state = this.simulation.getState();
          if (state.lossHistory.length > 0) {
            this.lossChart.update(state.lossHistory, state.theoryLossHistory || [], eta);
            this.normChart.update(state.normHistory, state.theoryNormHistory || [], state.d, eta);
          } else if (this.appState.showTheory) {
            // Theory-only preview mode - recompute with new x-axis
            this.computeAndShowTheoryOnly();
          }
        } else if (this.appState.showTheory) {
          // No simulation yet but theory is shown - recompute
          this.computeAndShowTheoryOnly();
        }
      } else {
        stepLink.classList.remove('active');
        teffLink.classList.add('active');
        this.lossChart.setEffectiveTime(true, eta);
        this.normChart.setEffectiveTime(true, eta);
        this.appState.xAxisMode = 'teff';

        // Update x-axis labels
        document.querySelectorAll('.x-axis-label').forEach(label => {
          label.innerHTML = '$t_{\\mathrm{eff}} = \\eta \\cdot \\mathrm{step}$';
        });

        // Retypeset MathJax
        if (window.MathJax && window.MathJax.typesetPromise) {
          MathJax.typesetPromise(document.querySelectorAll('.x-axis-label')).catch((err) => console.log(err));
        }

        // Replot with new x-axis - always replot if we have data
        if (this.simulation) {
          const state = this.simulation.getState();
          if (state.lossHistory.length > 0) {
            this.lossChart.update(state.lossHistory, state.theoryLossHistory || [], eta);
            this.normChart.update(state.normHistory, state.theoryNormHistory || [], state.d, eta);
          } else if (this.appState.showTheory) {
            // Theory-only preview mode - recompute with new x-axis
            this.computeAndShowTheoryOnly();
          }
        } else if (this.appState.showTheory) {
          // No simulation yet but theory is shown - recompute
          this.computeAndShowTheoryOnly();
        }
      }

      this.appState.save();
    };
  }

  /**
   * Setup EMA slider
   */
  setupEmaControl() {
    const slider = document.getElementById('emaSlider');
    const valueDisplay = document.getElementById('emaValue');

    // Map slider position [0,100] to window values
    const emaValues = [1, 10, 20, 30, 50, 70, 100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000];

    const positionToWindow = (pos) => {
      const index = Math.round((pos / 100) * (emaValues.length - 1));
      return emaValues[index];
    };

    const windowToPosition = (window) => {
      let closestIndex = 0;
      let closestDist = Math.abs(window - emaValues[0]);
      for (let i = 1; i < emaValues.length; i++) {
        const dist = Math.abs(window - emaValues[i]);
        if (dist < closestDist) {
          closestDist = dist;
          closestIndex = i;
        }
      }
      return (closestIndex / (emaValues.length - 1)) * 100;
    };

    // Set initial value
    slider.value = windowToPosition(this.appState.emaWindow);
    valueDisplay.textContent = this.appState.emaWindow === 1 ? 'off' : this.appState.emaWindow.toString();

    // Handle slider change
    slider.addEventListener('input', (e) => {
      const window = positionToWindow(parseFloat(e.target.value));
      valueDisplay.textContent = window === 1 ? 'off' : window.toString();
      this.lossChart.setEmaWindow(window);
      this.normChart.setEmaWindow(window);
      this.appState.emaWindow = window;
      this.appState.save();

      // Trigger replot if we have data
      if (this.simulation) {
        const state = this.simulation.getState();
        if (state.lossHistory.length > 0) {
          this.lossChart.update(state.lossHistory, state.theoryLossHistory || []);
          this.normChart.update(state.normHistory, state.theoryNormHistory || [], state.d);
        }
      }
    });
  }

  /**
   * Setup label position slider
   */
  setupLabelPositionControl() {
    const slider = document.getElementById('labelPosSlider');
    const valueDisplay = document.getElementById('labelPosValue');
    const label = document.getElementById('paramSizeLabel');

    // Set initial value
    const initialValue = -61;
    slider.value = initialValue;
    valueDisplay.textContent = initialValue.toString();
    label.style.left = `${initialValue}px`;

    // Handle slider change
    slider.addEventListener('input', (e) => {
      const offset = parseFloat(e.target.value);
      valueDisplay.textContent = offset.toString();
      // Adjust left position of the label
      label.style.left = `${offset}px`;
    });
  }

  /**
   * Get chart instances
   */
  getCharts() {
    return {
      lossChart: this.lossChart,
      normChart: this.normChart
    };
  }
}
