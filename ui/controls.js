// ============================================================================
// UI CONTROLS - Slider initialization and event handlers
// ============================================================================

import { LogarithmicSlider, LogarithmicSliderWithOff, formatScientific } from '../utils/sliders.js';
import { calculateAllTheory } from '../core/theory.js';

/**
 * ControlsManager - handles all slider initialization and events
 */
export class ControlsManager {
  constructor(appState, displayManager, onParamsChange) {
    this.appState = appState;
    this.display = displayManager;
    this.onParamsChange = onParamsChange; // Callback when params change

    // Current values
    this.currentD = appState.d;
    this.currentNumTerms = appState.numTerms;
    this.gammaValues = [...appState.gammaValues];
    // Deep copy 2D alpha array
    this.alphaValues = appState.alphaValues.map(row => [...row]);

    // Create sliders
    this.kSlider = new LogarithmicSlider(10, 30000);
    this.etaSlider = new LogarithmicSlider(0.0001, 1);
    this.batchSizeSlider = new LogarithmicSlider(1, 1000);
    this.gammaSlider = new LogarithmicSlider(0.0001, 1);
  }

  /**
   * Initialize all controls
   */
  initialize() {
    this.initializeKSlider();
    this.initializeDSlider();
    this.initializeNumTermsSlider();
    this.initializeEtaSlider();
    this.initializeBatchSizeSlider();
    this.renderGammaSliders();
    this.renderAlphaSliders();
  }

  /**
   * Initialize k slider
   */
  initializeKSlider() {
    const sliderElement = document.getElementById('kSlider');
    const valueElement = document.getElementById('kValue');

    sliderElement.value = this.kSlider.valueToSlider(this.appState.k);
    valueElement.textContent = this.appState.k.toString();

    sliderElement.addEventListener('input', (e) => {
      const value = this.kSlider.sliderToValue(parseFloat(e.target.value));
      valueElement.textContent = value.toString();
      this.appState.k = value;
      this.appState.save();
      this.onParamsChange();
    });
  }

  /**
   * Initialize d slider (linear, 1-5)
   */
  initializeDSlider() {
    const sliderElement = document.getElementById('dSlider');
    const valueElement = document.getElementById('dValue');

    sliderElement.value = this.appState.d;
    valueElement.textContent = this.appState.d.toString();

    sliderElement.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      valueElement.textContent = value.toString();
      this.currentD = value;
      this.appState.d = value;
      this.appState.save();

      // Re-render sliders for new dimension
      this.renderGammaSliders();
      this.renderAlphaSliders();
      this.updateTargetFunctionEquation();
      this.onParamsChange();
    });
  }

  /**
   * Initialize # terms slider (linear, 1-3)
   */
  initializeNumTermsSlider() {
    const sliderElement = document.getElementById('numTermsSlider');
    const valueElement = document.getElementById('numTermsValue');

    sliderElement.value = this.appState.numTerms;
    valueElement.textContent = this.appState.numTerms.toString();

    sliderElement.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      valueElement.textContent = value.toString();
      this.currentNumTerms = value;
      this.appState.numTerms = value;
      this.appState.save();

      // Re-render alpha sliders for new # of terms
      this.renderAlphaSliders();
      this.updateTargetFunctionEquation();
      this.updateTheoryDisplay();
      this.onParamsChange();
    });
  }

  /**
   * Initialize eta slider
   */
  initializeEtaSlider() {
    const sliderElement = document.getElementById('etaSlider');
    const valueElement = document.getElementById('etaValue');

    sliderElement.value = this.etaSlider.valueToSlider(this.appState.eta);
    valueElement.innerHTML = formatScientific(this.appState.eta);

    sliderElement.addEventListener('input', (e) => {
      const value = this.etaSlider.sliderToValue(parseFloat(e.target.value));
      valueElement.innerHTML = formatScientific(value);
      this.appState.eta = value;
      this.appState.save();
      this.onParamsChange();
    });
  }

  /**
   * Initialize batch size slider
   */
  initializeBatchSizeSlider() {
    const sliderElement = document.getElementById('batchSizeSlider');
    const valueElement = document.getElementById('batchSizeValue');

    sliderElement.value = this.batchSizeSlider.valueToSlider(this.appState.batchSize);
    valueElement.innerHTML = formatScientific(this.appState.batchSize);

    sliderElement.addEventListener('input', (e) => {
      const value = this.batchSizeSlider.sliderToValue(parseFloat(e.target.value));
      valueElement.innerHTML = formatScientific(value);
      this.appState.batchSize = value;
      this.appState.save();
    });
  }

  /**
   * Render gamma sliders
   */
  renderGammaSliders() {
    this.renderParameterSliders({
      containerId: 'gammaSliders',
      count: this.currentD,
      values: this.gammaValues,
      stateKey: 'gammaValues',
      label: (i) => `$\\gamma_{${i+1}}$:`,
      parseValue: (pos) => this.gammaSlider.sliderToValue(pos),
      formatValue: (val) => formatScientific(val),
      displayProp: 'innerHTML',
      displayWidth: '60px'
    });
  }

  /**
   * Render alpha sliders
   */
  renderAlphaSliders() {
    const columnsContainer = document.getElementById('alphaColumns');
    columnsContainer.innerHTML = '';

    const primeLabels = ['', "'", "''"];

    for (let termIdx = 0; termIdx < this.currentNumTerms; termIdx++) {
      const column = document.createElement('div');
      column.style.cssText = 'text-align: center; flex: 1;';

      // Column header
      const header = document.createElement('div');
      header.style.cssText = 'font-size: 16px; margin-bottom: 4px; color: #666;';
      header.textContent = `target multi-index${termIdx > 0 ? ' α' + primeLabels[termIdx] : ' α'}`;

      const subtitle = document.createElement('div');
      subtitle.style.cssText = 'margin-bottom: 8px;';
      subtitle.innerHTML = `$$\\boldsymbol{\\alpha}${primeLabels[termIdx]} = (\\alpha_i${primeLabels[termIdx]})$$`;

      const slidersContainer = document.createElement('div');
      slidersContainer.style.cssText = 'display: flex; flex-direction: column; gap: 6px; align-items: center;';

      // Create sliders for this term
      for (let dimIdx = 0; dimIdx < this.currentD; dimIdx++) {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; align-items: center; gap: 8px;';

        const label = document.createElement('span');
        label.style.cssText = 'font-size: 18px; min-width: 40px;';
        label.innerHTML = `$\\alpha${primeLabels[termIdx]}_{${dimIdx+1}}$:`;

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '0';
        slider.max = '3';
        slider.style.width = '120px';
        slider.value = this.alphaValues[termIdx][dimIdx];

        const valueSpan = document.createElement('span');
        valueSpan.style.cssText = 'font-size: 16px; min-width: 30px; text-align: left;';
        valueSpan.textContent = this.alphaValues[termIdx][dimIdx].toString();

        slider.addEventListener('input', (e) => {
          const value = parseInt(e.target.value);
          this.alphaValues[termIdx][dimIdx] = value;
          valueSpan.textContent = value.toString();
          this.appState.alphaValues[termIdx][dimIdx] = value;
          this.appState.save();
          this.updateTargetFunctionEquation();
          this.updateTheoryDisplay();
          this.onParamsChange();
        });

        wrapper.appendChild(label);
        wrapper.appendChild(slider);
        wrapper.appendChild(valueSpan);
        slidersContainer.appendChild(wrapper);
      }

      column.appendChild(header);
      column.appendChild(subtitle);
      column.appendChild(slidersContainer);
      columnsContainer.appendChild(column);
    }

    // Retypeset MathJax for the column headers
    if (window.MathJax && window.MathJax.typesetPromise) {
      MathJax.typesetPromise([columnsContainer]).catch(err => console.log(err));
    }
  }

  /**
   * Generic function to render dynamic parameter sliders
   */
  renderParameterSliders(config) {
    const container = document.getElementById(config.containerId);
    container.innerHTML = '';

    for (let i = 0; i < config.count; i++) {
      const wrapper = document.createElement('div');
      wrapper.style.cssText = 'display: flex; align-items: center; gap: 8px; margin-top: 6px;';

      const label = document.createElement('span');
      label.style.cssText = 'font-size: 18px; min-width: 40px;';
      label.innerHTML = `${config.label(i)}`;

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = config.max !== undefined ? config.max.toString() : '100';
      slider.style.width = '120px';

      const valueSpan = document.createElement('span');
      valueSpan.style.cssText = `font-size: 16px; min-width: ${config.displayWidth}; text-align: left;`;

      // Set initial value
      const currentValue = config.values[i];
      if (config.stateKey === 'alphaValues') {
        slider.value = currentValue;
      } else {
        slider.value = this.gammaSlider.valueToSlider(currentValue);
      }
      valueSpan[config.displayProp] = config.formatValue(currentValue);

      // Add event listener
      slider.addEventListener('input', (e) => {
        const value = config.parseValue(parseFloat(e.target.value));
        config.values[i] = value;
        valueSpan[config.displayProp] = config.formatValue(value);
        this.appState[config.stateKey] = [...config.values];
        this.appState.save();

        // Update theory calculations when alpha or gamma changes
        if (config.stateKey === 'alphaValues') {
          this.updateTargetFunctionEquation();
        } else if (config.stateKey === 'gammaValues') {
          this.updateTheoryCalculations();
        }

        // Trigger theory curve recomputation in preview mode
        this.onParamsChange();
      });

      wrapper.appendChild(label);
      wrapper.appendChild(slider);
      wrapper.appendChild(valueSpan);
      container.appendChild(wrapper);
    }

    // Retypeset MathJax for labels
    if (window.MathJax && window.MathJax.typesetPromise) {
      MathJax.typesetPromise([container]).catch(err => console.log(err));
    }
  }

  /**
   * Update target function equation based on d and numTerms
   */
  updateTargetFunctionEquation() {
    const primeLabels = ['', "'", "''"];
    const terms = [];

    // Build each term
    for (let termIdx = 0; termIdx < this.currentNumTerms; termIdx++) {
      const factors = [];
      for (let dimIdx = 0; dimIdx < this.currentD; dimIdx++) {
        const alpha = this.alphaValues[termIdx][dimIdx];
        const arg = `\\gamma_{${dimIdx+1}}^{-1/2} x_{${dimIdx+1}}`;

        if (alpha === 0) {
          // h_0 = 1, skip
          continue;
        } else if (alpha === 1) {
          // h_1(z) = z, just use the argument
          factors.push(arg);
        } else {
          // h_2 and higher
          factors.push(`h_{${alpha}}(${arg})`);
        }
      }

      // Build product for this term
      let productString;
      if (factors.length === 0) {
        productString = '1';
      } else {
        productString = factors.join(' \\cdot ');
      }

      terms.push(productString);
    }

    // Build equation
    let equation;
    if (this.currentNumTerms === 1) {
      equation = `$$\\begin{align}
f_*(\\mathbf{x}) &= h_{\\boldsymbol{\\alpha}}\\left(\\mathbf{\\Gamma}^{-1/2} \\mathbf{x}\\right) \\\\
&= ${terms[0]}
\\end{align}$$`;
    } else {
      // Multiple terms - show abstract form first, then expansion
      const primeLabels = ['', "'", "''"];

      // Build abstract Hermite notation
      const abstractTerms = [];
      for (let termIdx = 0; termIdx < this.currentNumTerms; termIdx++) {
        abstractTerms.push(`h_{\\boldsymbol{\\alpha}${primeLabels[termIdx]}}\\left(\\mathbf{\\Gamma}^{-1/2} \\mathbf{x}\\right)`);
      }
      const abstractFirst = `f_*(\\mathbf{x}) &= ${abstractTerms[0]}`;
      const abstractAdditional = abstractTerms.slice(1).map(t => `&\\quad + ${t}`).join(' \\\\\n');

      // Build expanded form
      const expandedFirst = `&= ${terms[0]}`;
      const expandedAdditional = terms.slice(1).map(t => `&\\quad + ${t}`).join(' \\\\\n');

      equation = `$$\\begin{align}
${abstractFirst} \\\\
${abstractAdditional} \\\\
${expandedFirst} \\\\
${expandedAdditional}
\\end{align}$$`;
    }

    const container = document.getElementById('targetFunctionEquation');
    container.textContent = equation;

    // Render MathJax
    if (window.MathJax && window.MathJax.typesetPromise) {
      MathJax.typesetPromise([container]).catch((err) => console.log(err));
    } else if (window.MathJax && window.MathJax.Hub) {
      MathJax.Hub.Queue(["Typeset", MathJax.Hub, container]);
    }

    // Update theory calculations
    this.updateTheoryCalculations();
  }

  /**
   * Update all theory calculations
   */
  updateTheoryCalculations() {
    // Build alphas array for all terms
    const alphasArray = [];
    for (let termIdx = 0; termIdx < this.currentNumTerms; termIdx++) {
      alphasArray.push(this.alphaValues[termIdx].slice(0, this.currentD));
    }

    // Always update initialization, dynamics, and modeling ODE formulas
    this.display.updateInitializationFormula(this.currentNumTerms);
    this.display.updateDynamicsFormula(this.currentNumTerms);
    this.display.updateModelingOdeFormula(alphasArray, this.currentD);

    if (this.currentNumTerms > 1) {
      // Multiple terms: compute theory for each term separately
      const theories = [];
      for (let termIdx = 0; termIdx < this.currentNumTerms; termIdx++) {
        theories.push(calculateAllTheory(this.gammaValues, alphasArray[termIdx], this.currentD));
      }

      // Update display - show c, c', c'' in separate columns
      this.display.updateTheoryValuesMultiTerm(theories, alphasArray, this.gammaValues, this.currentD);
    } else {
      // Single term: normal theory display
      const theory = calculateAllTheory(this.gammaValues, alphasArray[0], this.currentD);

      // Update all theory values (including rate constant)
      this.display.updateTheoryValues(theory, alphasArray[0], this.gammaValues, this.currentD);
    }
  }

  /**
   * Update theory display visibility based on numTerms
   */
  updateTheoryDisplay() {
    // Always enable theory checkbox
    const checkbox = document.getElementById('showTheoryCheckbox');
    if (checkbox) {
      checkbox.disabled = false;
    }

    // Update theory calculations
    this.updateTheoryCalculations();
  }

  /**
   * Get current parameter values
   */
  getCurrentParams() {
    // Extract alphas for all terms
    const alphasArray = [];
    for (let termIdx = 0; termIdx < this.currentNumTerms; termIdx++) {
      alphasArray.push(this.alphaValues[termIdx].slice(0, this.currentD));
    }

    return {
      d: this.currentD,
      k: this.appState.k,
      numTerms: this.currentNumTerms,
      gammas: this.gammaValues.slice(0, this.currentD),
      alphas: alphasArray,  // Now 2D array
      eta: this.appState.eta,
      batchSize: this.appState.batchSize
    };
  }
}
