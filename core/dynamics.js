// ============================================================================
// DYNAMICS - Gradient descent dynamics for multi-term learning
// ============================================================================

/**
 * Compute derivatives for the learning dynamics
 *
 * State vector: y = [a₁, a₂, ..., aₐ, b₁, b₂, ..., bₙᵤₘₜₑᵣₘₛ]
 *
 * Loss: L = Σⱼ (1/2)(1 - cⱼ·∏ᵢ aᵢ^αⱼᵢ · bⱼ)²
 *
 * Dynamics:
 *   ȧᵢ = -∂L/∂aᵢ
 *   ḃⱼ = -∂L/∂bⱼ
 *
 * @param {number} t - Time (unused, but required for ODE solver interface)
 * @param {Array<number>} y - State [a₁, ..., aₐ, b₁, ..., bₙ]
 * @param {Array<Array<number>>} alphasArray - [[α₁, α₂, ...], [α'₁, α'₂, ...], ...]
 * @param {Array<number>} cValues - [c, c', c'', ...]
 * @param {number} d - Dimension
 * @returns {Array<number>} Derivatives [ȧ₁, ..., ȧₐ, ḃ₁, ..., ḃₙ]
 */
export function computeDerivatives(t, y, alphasArray, cValues, d) {
  const numTerms = alphasArray.length;

  // Extract state variables
  const a = y.slice(0, d);           // [a₁, a₂, ..., aₐ]
  const b = y.slice(d, d + numTerms); // [b₁, b₂, ..., bₙ]

  // Initialize gradients
  const grad_a = Array(d).fill(0);
  const grad_b = Array(numTerms).fill(0);

  // Compute gradients by summing over all terms
  for (let termIdx = 0; termIdx < numTerms; termIdx++) {
    const alphas = alphasArray[termIdx];
    const c = cValues[termIdx];
    const bj = b[termIdx];

    // Compute product: ∏ᵢ aᵢ^αᵢ
    let product = 1;
    for (let i = 0; i < d; i++) {
      if (alphas[i] > 0) {
        product *= Math.pow(a[i], alphas[i]);
      }
    }

    // Error term: (1 - c·∏aᵢ^αᵢ·bⱼ)
    const error = 1 - c * product * bj;

    // ∂L/∂aᵢ for this term
    for (let i = 0; i < d; i++) {
      if (alphas[i] === 0) {
        // No contribution if αᵢ = 0
        continue;
      }

      // Compute product excluding aᵢ: ∏ₖ≠ᵢ aₖ^αₖ
      let productExcludingI = 1;
      for (let k = 0; k < d; k++) {
        if (k !== i && alphas[k] > 0) {
          productExcludingI *= Math.pow(a[k], alphas[k]);
        }
      }

      // ∂L/∂aᵢ += -error · c · αᵢ · aᵢ^(αᵢ-1) · ∏ₖ≠ᵢ aₖ^αₖ · bⱼ
      grad_a[i] += -error * c * alphas[i] * Math.pow(a[i], alphas[i] - 1) * productExcludingI * bj;
    }

    // ∂L/∂bⱼ = -error · c · ∏ᵢ aᵢ^αᵢ
    grad_b[termIdx] = -error * c * product;
  }

  // Return derivatives: ȧ = -∂L/∂a, ḃ = -∂L/∂b (gradient descent)
  return [...grad_a.map(g => -g), ...grad_b.map(g => -g)];
}

/**
 * Compute loss from current state
 *
 * L = Σⱼ (1/2)(1 - cⱼ·∏ᵢ aᵢ^αⱼᵢ · bⱼ)²
 *
 * @param {Array<number>} y - State [a₁, ..., aₐ, b₁, ..., bₙ]
 * @param {Array<Array<number>>} alphasArray - [[α₁, α₂, ...], [α'₁, α'₂, ...], ...]
 * @param {Array<number>} cValues - [c, c', c'', ...]
 * @param {number} d - Dimension
 * @returns {number} Loss value
 */
export function computeLoss(y, alphasArray, cValues, d) {
  const numTerms = alphasArray.length;

  // Extract state variables
  const a = y.slice(0, d);
  const b = y.slice(d, d + numTerms);

  let totalLoss = 0;

  // Sum over all terms
  for (let termIdx = 0; termIdx < numTerms; termIdx++) {
    const alphas = alphasArray[termIdx];
    const c = cValues[termIdx];
    const bj = b[termIdx];

    // Compute product: ∏ᵢ aᵢ^αᵢ
    let product = 1;
    for (let i = 0; i < d; i++) {
      if (alphas[i] > 0) {
        product *= Math.pow(a[i], alphas[i]);
      }
    }

    // Add term: (1/2)(1 - c·∏aᵢ^αᵢ·bⱼ)²
    const error = 1 - c * product * bj;
    totalLoss += 0.5 * error * error;
  }

  return totalLoss;
}
