// ============================================================================
// UI DISPLAY - DOM updates and MathJax rendering
// ============================================================================

import { formatForLatex } from '../utils/formatters.js';

/**
 * DisplayManager - handles all DOM updates and MathJax rendering
 */
export class DisplayManager {
  constructor() {
    // Cache DOM elements
    this.modelingOdeFormula = document.getElementById('modelingOdeFormula');
    this.initializationFormula = document.getElementById('initializationFormula');
    this.rateConstantColumns = document.getElementById('rateConstantColumns');
    this.relevantDirectionsValue = document.getElementById('relevantDirectionsValue');
    this.orderOfFStarValue = document.getElementById('orderOfFStarValue');
    this.totalOrderValue = document.getElementById('totalOrderValue');
    this.betaValue = document.getElementById('betaValue');
    this.shapeParamsValue = document.getElementById('shapeParamsValue');
    this.shapeIntegralValue = document.getElementById('shapeIntegralValue');
    this.triseValue = document.getElementById('triseValue');

    // Elements to hide for multi-term
    this.theoryRowSeparator = document.getElementById('theoryRowSeparator');
    this.theoryMetricsRow2 = document.getElementById('theoryMetricsRow2');
    this.theoryRowSeparator2 = document.getElementById('theoryRowSeparator2');
    this.theoryMetricsRow3 = document.getElementById('theoryMetricsRow3');
    this.theoryRiseTime = document.getElementById('theoryRiseTime');
  }

  /**
   * Update initialization formula based on number of terms
   */
  updateInitializationFormula(numTerms) {
    const primeLabels = ['', "'", "''"];
    const bInitLines = [];

    for (let termIdx = 0; termIdx < numTerms; termIdx++) {
      bInitLines.push(`b${primeLabels[termIdx]}(0) &= 0`);
    }

    const formula = `$\\begin{align}
a_i(0) &= 1 \\\\
${bInitLines.join(' \\\\ ')}
\\end{align}$`;

    this.initializationFormula.textContent = formula;
    this.renderMathJax([this.initializationFormula]);
  }

  /**
   * Update dynamics formula based on number of terms
   */
  updateDynamicsFormula(numTerms) {
    const primeLabels = ['', "'", "''"];
    const bDotLines = [];

    for (let termIdx = 0; termIdx < numTerms; termIdx++) {
      bDotLines.push(`\\dot{b}${primeLabels[termIdx]} &= -\\partial_{b${primeLabels[termIdx]}} \\mathcal{L}`);
    }

    // Find the dynamics formula element by ID
    const dynamicsFormula = document.getElementById('dynamicsFormula');

    if (dynamicsFormula) {
      const formula = `$\\begin{align}
\\dot{a}_i &= -\\partial_{a_i} \\mathcal{L} \\\\
${bDotLines.join(' \\\\ ')}
\\end{align}$`;

      dynamicsFormula.textContent = formula;
      this.renderMathJax([dynamicsFormula]);
    }
  }

  /**
   * Update modeling ODE formula based on alpha values (handles multiple terms)
   */
  updateModelingOdeFormula(alphasArray, d) {
    const primeLabels = ['', "'", "''"];
    const numTerms = alphasArray.length;

    // Build each term
    const terms = [];
    for (let termIdx = 0; termIdx < numTerms; termIdx++) {
      const alphaValues = alphasArray[termIdx];
      const factors = [];

      for (let i = 0; i < d; i++) {
        const alpha = alphaValues[i];

        if (alpha === 0) {
          continue;
        } else if (alpha === 1) {
          factors.push(`a_{${i+1}}`);
        } else {
          factors.push(`a_{${i+1}}^{${alpha}}`);
        }
      }

      let productString;
      if (factors.length === 0) {
        productString = '';
      } else {
        productString = factors.join(' \\cdot ');
      }

      let expression;
      if (productString === '') {
        expression = `c${primeLabels[termIdx]} \\cdot b${primeLabels[termIdx]}`;
      } else {
        expression = `c${primeLabels[termIdx]} \\cdot ${productString} \\cdot b${primeLabels[termIdx]}`;
      }

      terms.push(expression);
    }

    // Build formula
    let formula;
    if (numTerms === 1) {
      // Single term - traditional format
      formula = `$$\\begin{align}
\\mathcal{L} &= \\frac{1}{2}\\left(1 - c \\cdot \\prod_{i=1}^{d} a_i^{\\alpha_i} \\cdot b\\right)^2 \\\\
&= \\frac{1}{2}\\left(1 - ${terms[0]}\\right)^2
\\end{align}$$`;
    } else {
      // Multiple terms
      const genericTerms = [];
      const specificTerms = [];

      for (let termIdx = 0; termIdx < numTerms; termIdx++) {
        const prime = primeLabels[termIdx];
        genericTerms.push(`\\frac{1}{2}\\left(1 - c${prime} \\cdot \\prod_{i=1}^{d} a_i^{\\alpha_i${prime}} \\cdot b${prime}\\right)^2`);
        specificTerms.push(`\\frac{1}{2}\\left(1 - ${terms[termIdx]}\\right)^2`);
      }

      // First line: generic form with summation
      // Subsequent lines: specific expansions with + aligned to right of =
      const firstLine = `\\mathcal{L} &= ${genericTerms[0]}`;
      const additionalGeneric = genericTerms.slice(1).map(t => `&\\qquad + ${t}`).join(' \\\\\n');
      const firstSpecific = `&= ${specificTerms[0]}`;
      const additionalSpecific = specificTerms.slice(1).map(t => `&\\qquad + ${t}`).join(' \\\\\n');

      formula = `$$\\begin{align}
${firstLine} \\\\
${additionalGeneric} \\\\
${firstSpecific} \\\\
${additionalSpecific}
\\end{align}$$`;
    }

    this.modelingOdeFormula.textContent = formula;
    this.renderMathJax([this.modelingOdeFormula]);
  }


  /**
   * Update all theory values from theory object (single term case)
   */
  updateTheoryValues(theory, alphaValues, gammaValues, d) {
    // Create single rate constant column
    this.rateConstantColumns.innerHTML = '';

    const column = document.createElement('div');
    column.className = 'theory-metric';
    column.style.flex = '1';

    const label = document.createElement('div');
    label.className = 'theory-metric-label';
    label.textContent = 'rate constant';

    const formula = document.createElement('div');
    formula.className = 'theory-metric-formula';

    // Build formula
    const productFactors = [];
    for (let i = 0; i < d; i++) {
      const alpha = alphaValues[i];
      if (alpha === 0) {
        continue;
      } else if (alpha === 1) {
        productFactors.push(`\\gamma_{${i+1}}`);
      } else {
        productFactors.push(`\\gamma_{${i+1}}^{${alpha}}`);
      }
    }

    let specificProduct;
    if (productFactors.length === 0) {
      specificProduct = '1';
    } else {
      specificProduct = productFactors.join(' ');
    }

    const formulaText = `$\\begin{align}
c &= \\sqrt{\\prod_{i=1}^{d} \\gamma_i^{\\alpha_i}} \\\\
&= \\sqrt{${specificProduct}}
\\end{align}$`;
    formula.innerHTML = formulaText;

    const value = document.createElement('div');
    value.className = 'theory-metric-value';
    const formattedC = formatForLatex(theory.c);
    value.innerHTML = `$${formattedC}$`;

    column.appendChild(label);
    column.appendChild(formula);
    column.appendChild(value);
    this.rateConstantColumns.appendChild(column);

    // Show all other theory rows
    this.theoryRowSeparator.style.display = 'block';
    this.theoryMetricsRow2.style.display = 'flex';
    this.theoryRowSeparator2.style.display = 'block';
    this.theoryMetricsRow3.style.display = 'flex';
    this.theoryRiseTime.style.display = 'block';

    // Number of relevant directions (d')
    const formattedRelevantDirs = formatForLatex(theory.relevantDirections);
    this.relevantDirectionsValue.innerHTML = `$${formattedRelevantDirs}$`;

    // Order of f*
    const formattedOrderFStar = formatForLatex(theory.orderOfFStar);
    this.orderOfFStarValue.innerHTML = `$${formattedOrderFStar}$`;

    // Total order (ℓ)
    const formattedEll = formatForLatex(theory.ell);
    this.totalOrderValue.innerHTML = `$${formattedEll}$`;

    // Beta
    const formattedBeta = formatForLatex(theory.beta);
    this.betaValue.innerHTML = `$${formattedBeta}$`;

    // Shape parameters
    const shapeParamsRows = theory.shapeParams.map((r, i) =>
      `r_{${i+1}} &= ${formatForLatex(r)}`
    ).join(' \\\\ ');
    const allRows = shapeParamsRows + ` \\\\ r_b &= 0`;
    this.shapeParamsValue.innerHTML = `$$\\begin{aligned} ${allRows} \\end{aligned}$$`;

    // Shape integral
    if (theory.shapeIntegral.isUndefined) {
      this.shapeIntegralValue.innerHTML = '—';
    } else {
      const formattedF = formatForLatex(theory.shapeIntegral.value);
      this.shapeIntegralValue.innerHTML = `$${formattedF}$`;
    }

    // Rise time
    if (theory.tRise.isUndefined) {
      this.triseValue.innerHTML = '$\\text{undefined}$';
    } else {
      const formattedTRise = formatForLatex(theory.tRise.value);
      this.triseValue.innerHTML = `$${formattedTRise}$`;
    }

    // Render all MathJax
    this.renderMathJax([
      this.rateConstantColumns,
      this.relevantDirectionsValue,
      this.orderOfFStarValue,
      this.totalOrderValue,
      this.betaValue,
      this.shapeParamsValue,
      this.shapeIntegralValue,
      this.triseValue
    ]);
  }

  /**
   * Update theory values for multiple terms (show c, c', c'' in separate columns)
   */
  updateTheoryValuesMultiTerm(theories, alphasArray, gammaValues, d) {
    const primeLabels = ['', "'", "''"];
    const ordinalLabels = ['first', 'second', 'third'];

    // Clear and rebuild rate constant columns
    this.rateConstantColumns.innerHTML = '';

    theories.forEach((theory, idx) => {
      const column = document.createElement('div');
      column.className = 'theory-metric';
      column.style.flex = '1';

      const label = document.createElement('div');
      label.className = 'theory-metric-label';
      label.textContent = `${ordinalLabels[idx]} rate constant`;

      const formula = document.createElement('div');
      formula.className = 'theory-metric-formula';

      // Build formula for this term
      const productFactors = [];
      for (let i = 0; i < d; i++) {
        const alpha = alphasArray[idx][i];
        if (alpha === 0) {
          continue;
        } else if (alpha === 1) {
          productFactors.push(`\\gamma_{${i+1}}`);
        } else {
          productFactors.push(`\\gamma_{${i+1}}^{${alpha}}`);
        }
      }

      let specificProduct;
      if (productFactors.length === 0) {
        specificProduct = '1';
      } else {
        specificProduct = productFactors.join(' ');
      }

      const formulaText = `$\\begin{align}
c${primeLabels[idx]} &= \\sqrt{\\prod_{i=1}^{d} \\gamma_i^{\\alpha_i${primeLabels[idx]}}} \\\\
&= \\sqrt{${specificProduct}}
\\end{align}$`;
      formula.innerHTML = formulaText;

      const value = document.createElement('div');
      value.className = 'theory-metric-value';
      value.innerHTML = `$${formatForLatex(theory.c)}$`;

      column.appendChild(label);
      column.appendChild(formula);
      column.appendChild(value);
      this.rateConstantColumns.appendChild(column);
    });

    // Hide all rows after the first row
    this.theoryRowSeparator.style.display = 'none';
    this.theoryMetricsRow2.style.display = 'none';
    this.theoryRowSeparator2.style.display = 'none';
    this.theoryMetricsRow3.style.display = 'none';
    this.theoryRiseTime.style.display = 'none';

    // Render MathJax
    this.renderMathJax([this.rateConstantColumns]);
  }

  /**
   * Render MathJax for given elements
   */
  renderMathJax(elements) {
    if (window.MathJax && window.MathJax.typesetPromise) {
      MathJax.typesetPromise(elements).catch((err) => console.log(err));
    } else if (window.MathJax && window.MathJax.Hub) {
      MathJax.Hub.Queue(["Typeset", MathJax.Hub, ...elements]);
    }
  }
}
