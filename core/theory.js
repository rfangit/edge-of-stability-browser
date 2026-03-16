// ============================================================================
// THEORY CALCULATIONS
// ============================================================================
// Pure functions for theoretical predictions

import { computeShapeIntegral } from '../utils/integration.js';

/**
 * Helper: Compute factorial
 */
function factorial(n) {
  if (n <= 1) return 1;
  let result = 1;
  for (let i = 2; i <= n; i++) {
    result *= i;
  }
  return result;
}

/**
 * Calculate number of relevant directions: d' = #{αᵢ > 0}
 */
export function calculateRelevantDirections(alphaValues, d) {
  return alphaValues.slice(0, d).filter(alpha => alpha > 0).length;
}

/**
 * Calculate order of f*: |α| = Σ αᵢ
 */
export function calculateOrderOfFStar(alphaValues, d) {
  return alphaValues.slice(0, d).reduce((sum, alpha) => sum + alpha, 0);
}

/**
 * Calculate total order: ℓ = |α| + 1
 */
export function calculateEll(alphaValues, d) {
  const alphaSum = calculateOrderOfFStar(alphaValues, d);
  return alphaSum + 1;
}

/**
 * Calculate rate constant: c = √(Πᵢ γᵢ^αᵢ)
 */
export function calculateRateConstant(gammaValues, alphaValues, d) {
  let product = 1;
  for (let i = 0; i < d; i++) {
    const gamma = gammaValues[i];
    const alpha = alphaValues[i];
    product *= Math.pow(gamma, alpha);
  }

  return Math.sqrt(product);
}

/**
 * Calculate order prefactor: κ = Πᵢ αᵢ^(αᵢ/2)
 */
export function calculateKappa(alphaValues, d) {
  let product = 1;
  for (let i = 0; i < d; i++) {
    const alpha = alphaValues[i];
    if (alpha > 0) {
      product *= Math.pow(alpha, alpha / 2);
    }
  }
  return product;
}

/**
 * Calculate mean core parameter at init: β = 1/(d_rel+1) Σᵢ |aᵢ(0)|/√αᵢ
 * (with aᵢ(0) = 1, b(0) = 0)
 * where d_rel is the number of relevant directions (alphas > 0)
 */
export function calculateBeta(alphaValues, d) {
  let sum = 0;
  let dRel = 0;
  for (let i = 0; i < d; i++) {
    const alpha = alphaValues[i];
    if (alpha > 0) {
      sum += 1 / Math.sqrt(alpha);
      dRel++;
    }
  }
  return sum / (dRel + 1);
}

/**
 * Calculate shape parameters: rᵢ = aᵢ(0)² / (αᵢ β²)
 * Returns array of non-zero shape parameters
 */
export function calculateShapeParams(alphaValues, d, beta) {
  const shapeParams = [];
  for (let i = 0; i < d; i++) {
    const alpha = alphaValues[i];
    if (alpha > 0) {
      const r_i = 1 / (alpha * beta * beta);
      shapeParams.push(r_i);
    }
  }
  return shapeParams;
}

/**
 * Calculate shape integral F(r)
 * Returns {value, isUndefined}
 */
export function calculateShapeIntegral(shapeParams, alphaValues, d, ell) {
  // r_b = 1e-6 (small epsilon to avoid singularity)
  const rB = 1e-6;

  // Get only the non-zero alphas that correspond to shapeParams
  const nonZeroAlphas = [];
  for (let i = 0; i < d; i++) {
    if (alphaValues[i] > 0) {
      nonZeroAlphas.push(alphaValues[i]);
    }
  }

  return computeShapeIntegral(shapeParams, rB, nonZeroAlphas, ell);
}

/**
 * Calculate rise time (piecewise function)
 * Returns {value, isUndefined}
 */
export function calculateRiseTime(ell, kappa, beta, c, shapeIntegral) {
  if (ell < 1) {
    return { value: NaN, isUndefined: true };
  }

  if (ell === 1) {
    return { value: 1 / (c * c), isUndefined: false };
  }

  if (ell === 2) {
    const trise = -(1 / (kappa * c)) * Math.log(Math.sqrt(kappa * c) * beta);
    return { value: trise, isUndefined: false };
  }

  // ell > 2
  if (shapeIntegral.isUndefined) {
    return { value: NaN, isUndefined: true };
  }

  const trise = (1 / (ell - 2)) * (shapeIntegral.value / (kappa * c)) * (1 / Math.pow(beta, ell - 2));
  return { value: trise, isUndefined: false };
}

/**
 * Calculate all theoretical quantities
 * Returns object with all theory values
 */
export function calculateAllTheory(gammaValues, alphaValues, d) {
  const relevantDirections = calculateRelevantDirections(alphaValues, d);
  const orderOfFStar = calculateOrderOfFStar(alphaValues, d);
  const ell = calculateEll(alphaValues, d);
  const c = calculateRateConstant(gammaValues, alphaValues, d);
  const kappa = calculateKappa(alphaValues, d);
  const beta = calculateBeta(alphaValues, d);
  const shapeParams = calculateShapeParams(alphaValues, d, beta);
  const shapeIntegral = calculateShapeIntegral(shapeParams, alphaValues, d, ell);
  const tRise = calculateRiseTime(ell, kappa, beta, c, shapeIntegral);

  return {
    relevantDirections,
    orderOfFStar,
    ell,
    c,
    kappa,
    beta,
    shapeParams,
    shapeIntegral,
    tRise
  };
}
