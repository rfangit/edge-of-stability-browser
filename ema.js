// ============================================================================
// EMA (Exponential Moving Average) utilities
// ============================================================================

/**
 * Apply exponential moving average to a time series
 * @param {Array} data - Array of {iteration, value} or {iteration, ...values} objects
 * @param {number} window - EMA window size (alpha = 1/window)
 * @param {string|Array<string>} valueKeys - Key(s) to smooth (e.g., 'loss' or ['w1NormSq', 'w2NormSq'])
 * @param {Object} initValues - Initial values for EMA (e.g., {loss: 0.5, w1NormSq: 3, w2NormSq: 0})
 * @returns {Array} Smoothed data with same structure
 */
export function applyEMA(data, window, valueKeys, initValues = {}) {
  if (data.length === 0) return [];
  if (window === 1) return data; // No smoothing

  const alpha = 1 / window;
  const keys = Array.isArray(valueKeys) ? valueKeys : [valueKeys];

  // Initialize smoothed values
  const smoothed = [];
  let prevSmoothed = {};

  // Use provided initial values, or fall back to first data point
  for (const key of keys) {
    prevSmoothed[key] = initValues[key] !== undefined ? initValues[key] : data[0][key];
  }

  // Apply EMA: smoothed[i] = alpha * raw[i] + (1 - alpha) * smoothed[i-1]
  for (let i = 0; i < data.length; i++) {
    const point = { iteration: data[i].iteration };

    for (const key of keys) {
      point[key] = alpha * data[i][key] + (1 - alpha) * prevSmoothed[key];
      prevSmoothed[key] = point[key];
    }

    // Copy any other fields
    for (const key in data[i]) {
      if (key !== 'iteration' && !keys.includes(key)) {
        point[key] = data[i][key];
      }
    }

    smoothed.push(point);
  }

  return smoothed;
}

/**
 * Downsample data using stable stride-based sampling
 * Uses power-of-2 strides that only change at specific thresholds,
 * ensuring stable point selection across frames (no flickering)
 * @param {Array} data - Input data array
 * @param {number} maxPoints - Target maximum number of points (will stay between maxPoints/2 and maxPoints)
 * @returns {Array} Downsampled data
 */
export function downsample(data, maxPoints) {
  if (data.length <= maxPoints) return data;

  // Find the stride (power of 2) such that data.length / stride ≈ maxPoints
  let stride = 1;
  while (data.length / stride > maxPoints) {
    stride *= 2;
  }

  // Sample at regular intervals with this stride
  const result = [];
  for (let i = 0; i < data.length; i += stride) {
    result.push(data[i]);
  }

  // Always include the last point if it wasn't included
  if ((data.length - 1) % stride !== 0) {
    result.push(data[data.length - 1]);
  }

  return result;
}

/**
 * Incrementally update EMA with new data point
 * @param {number} lastSmoothed - Previous smoothed value
 * @param {number} newRaw - New raw value
 * @param {number} window - EMA window size
 * @returns {number} New smoothed value
 */
export function emaIncremental(lastSmoothed, newRaw, window) {
  if (window === 1) return newRaw;
  const alpha = 1 / window;
  return alpha * newRaw + (1 - alpha) * lastSmoothed;
}
