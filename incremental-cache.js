// ============================================================================
// INCREMENTAL CACHE - Efficient data structure for incrementally updating
// downsampled and smoothed data without O(n) recomputation
// ============================================================================

/**
 * IncrementalCache maintains downsampled and EMA-smoothed data incrementally.
 * Only processes new data points, avoiding O(n) recomputation on every update.
 */
export class IncrementalCache {
  constructor(emaWindow, maxPlotPoints, valueKeys, initEmaValues) {
    this.emaWindow = emaWindow;
    this.maxPlotPoints = maxPlotPoints;
    this.valueKeys = Array.isArray(valueKeys) ? valueKeys : [valueKeys];
    this.initEmaValues = initEmaValues || {};

    // Downsampling state
    this.stride = 1;
    this.downsampledRaw = [];
    this.downsampledSmoothed = [];

    // EMA state
    this.emaData = [];
    this.lastEmaValues = {}; // Track last EMA value for each key

    // Initialize last EMA values to init values
    for (const key of this.valueKeys) {
      this.lastEmaValues[key] = this.initEmaValues[key] !== undefined
        ? this.initEmaValues[key]
        : 0;
    }

    // Max tracking
    this.rollingMax = {}; // Track max for each key
    for (const key of this.valueKeys) {
      this.rollingMax[key] = -Infinity;
    }

    // Track last processed index
    this.lastProcessedIndex = -1;
  }

  /**
   * Update cache with new data points
   * @param {Array} fullData - Complete data array (including old + new points)
   * @returns {Object} - { downsampledRaw, downsampledSmoothed, max }
   */
  update(fullData) {
    if (fullData.length === 0) {
      return {
        downsampledRaw: [],
        downsampledSmoothed: [],
        max: this.rollingMax
      };
    }

    // Check if we need to rebuild from scratch
    const newStride = this._computeStride(fullData.length);
    const needsRebuild = newStride !== this.stride ||
                        this.lastProcessedIndex >= fullData.length ||
                        this.lastProcessedIndex < 0; // Rebuild if cache was invalidated

    if (needsRebuild) {
      return this._rebuildCache(fullData, newStride);
    }

    // Process only new points incrementally
    for (let i = this.lastProcessedIndex + 1; i < fullData.length; i++) {
      const point = fullData[i];

      // 1. Update EMA incrementally
      const emaPoint = { iteration: point.iteration };
      for (const key of this.valueKeys) {
        const alpha = this.emaWindow === 1 ? 1 : 1 / this.emaWindow;
        const smoothed = alpha * point[key] + (1 - alpha) * this.lastEmaValues[key];
        emaPoint[key] = smoothed;
        this.lastEmaValues[key] = smoothed;

        // Update rolling max
        if (smoothed > this.rollingMax[key]) {
          this.rollingMax[key] = smoothed;
        }
      }
      this.emaData.push(emaPoint);

      // 2. Update downsampled arrays if point falls on stride boundary
      if (i % this.stride === 0) {
        this.downsampledRaw.push(point);
        this.downsampledSmoothed.push(emaPoint);
      }
    }

    // Always include the last point
    const lastIndex = fullData.length - 1;
    if (lastIndex % this.stride !== 0) {
      // Remove previous "last point" if it exists and wasn't on stride boundary
      if (this.downsampledRaw.length > 0 &&
          this.downsampledRaw[this.downsampledRaw.length - 1].iteration !== fullData[lastIndex].iteration) {
        // Check if the last point in our cache is the "forced last point" from previous update
        const lastCachedIteration = this.downsampledRaw[this.downsampledRaw.length - 1].iteration;
        const secondToLastStrideIndex = Math.floor(this.lastProcessedIndex / this.stride) * this.stride;
        if (lastCachedIteration > secondToLastStrideIndex) {
          // Yes, it was a forced last point, remove it
          this.downsampledRaw.pop();
          this.downsampledSmoothed.pop();
        }
      }

      // Add current last point
      this.downsampledRaw.push(fullData[lastIndex]);
      this.downsampledSmoothed.push(this.emaData[this.emaData.length - 1]);
    }

    this.lastProcessedIndex = fullData.length - 1;

    return {
      downsampledRaw: this.downsampledRaw,
      downsampledSmoothed: this.downsampledSmoothed,
      emaData: this.emaData,
      max: this.rollingMax
    };
  }

  /**
   * Rebuild entire cache from scratch (e.g., when stride changes or settings change)
   */
  _rebuildCache(fullData, newStride) {
    this.stride = newStride;
    this.downsampledRaw = [];
    this.downsampledSmoothed = [];
    this.emaData = [];

    // Reset last EMA values to initial values
    for (const key of this.valueKeys) {
      this.lastEmaValues[key] = this.initEmaValues[key] !== undefined
        ? this.initEmaValues[key]
        : fullData[0]?.[key] || 0;
      this.rollingMax[key] = -Infinity;
    }

    // Compute EMA for all points
    const alpha = this.emaWindow === 1 ? 1 : 1 / this.emaWindow;
    for (let i = 0; i < fullData.length; i++) {
      const point = fullData[i];
      const emaPoint = { iteration: point.iteration };

      for (const key of this.valueKeys) {
        const smoothed = alpha * point[key] + (1 - alpha) * this.lastEmaValues[key];
        emaPoint[key] = smoothed;
        this.lastEmaValues[key] = smoothed;

        // Update rolling max
        if (smoothed > this.rollingMax[key]) {
          this.rollingMax[key] = smoothed;
        }
      }
      this.emaData.push(emaPoint);

      // Downsample
      if (i % this.stride === 0) {
        this.downsampledRaw.push(point);
        this.downsampledSmoothed.push(emaPoint);
      }
    }

    // Always include last point
    const lastIndex = fullData.length - 1;
    if (lastIndex % this.stride !== 0 && lastIndex >= 0) {
      this.downsampledRaw.push(fullData[lastIndex]);
      this.downsampledSmoothed.push(this.emaData[lastIndex]);
    }

    this.lastProcessedIndex = fullData.length - 1;

    return {
      downsampledRaw: this.downsampledRaw,
      downsampledSmoothed: this.downsampledSmoothed,
      emaData: this.emaData,
      max: this.rollingMax
    };
  }

  /**
   * Compute stride for downsampling (power of 2)
   */
  _computeStride(dataLength) {
    if (dataLength <= this.maxPlotPoints) return 1;

    let stride = 1;
    while (dataLength / stride > this.maxPlotPoints) {
      stride *= 2;
    }
    return stride;
  }

  /**
   * Update EMA window (requires full rebuild)
   */
  setEmaWindow(newWindow) {
    if (newWindow !== this.emaWindow) {
      this.emaWindow = newWindow;
      // Mark for rebuild on next update
      this.lastProcessedIndex = -1;
    }
  }

  /**
   * Clear cache (e.g., on simulation reset)
   */
  clear() {
    this.stride = 1;
    this.downsampledRaw = [];
    this.downsampledSmoothed = [];
    this.emaData = [];
    this.lastProcessedIndex = -1;

    for (const key of this.valueKeys) {
      this.lastEmaValues[key] = this.initEmaValues[key] !== undefined
        ? this.initEmaValues[key]
        : 0;
      this.rollingMax[key] = -Infinity;
    }
  }
}
