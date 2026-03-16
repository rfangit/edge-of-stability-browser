// ============================================================================
// SLIDER UTILITIES
// ============================================================================
// Logarithmic slider classes and helper functions

/**
 * Generate values with 1 significant figure for logarithmic sliders
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {number[]} Array of values: 1, 2, 3, ..., 9, 10, 20, 30, ..., 90, 100, 200, etc.
 */
export function generateOneSigFigValues(min, max) {
  const values = [];
  let power = Math.floor(Math.log10(min));

  while (true) {
    const base = Math.pow(10, power);
    for (let digit = 1; digit <= 9; digit++) {
      const value = digit * base;
      if (value >= min && value <= max) {
        values.push(value);
      }
      if (value > max) return values;
    }
    power++;
  }
}

/**
 * Format value in scientific notation with superscripts
 * @param {number} value - Value to format
 * @returns {string} Formatted string (e.g., "2×10⁻³")
 */
export function formatScientific(value) {
  if (value === 1) return '1';

  const exponent = Math.floor(Math.log10(value));
  const mantissa = value / Math.pow(10, exponent);
  const roundedMantissa = Math.round(mantissa * 1e10) / 1e10;

  if (Math.abs(roundedMantissa - 1) < 1e-10) {
    if (exponent === 0) return '1';
    return `10<sup>${exponent}</sup>`;
  } else {
    if (exponent === 0) return roundedMantissa.toString();
    return `${roundedMantissa}×10<sup>${exponent}</sup>`;
  }
}

/**
 * Generic logarithmic slider class for mapping slider position to values
 */
export class LogarithmicSlider {
  constructor(min, max) {
    this.values = generateOneSigFigValues(min, max);
  }

  sliderToValue(position) {
    const index = Math.round((position / 100) * (this.values.length - 1));
    return this.values[index];
  }

  valueToSlider(value) {
    let closestIndex = 0;
    let closestDist = Math.abs(value - this.values[0]);
    for (let i = 1; i < this.values.length; i++) {
      const dist = Math.abs(value - this.values[i]);
      if (dist < closestDist) {
        closestDist = dist;
        closestIndex = i;
      }
    }
    return (closestIndex / (this.values.length - 1)) * 100;
  }
}

/**
 * Logarithmic slider that starts at a minimum value (no zero)
 * Just uses the standard logarithmic values [min, max]
 */
export class LogarithmicSliderWithOff {
  constructor(min, max) {
    this.values = generateOneSigFigValues(min, max);
  }

  sliderToValue(position) {
    const index = Math.round((position / 100) * (this.values.length - 1));
    return this.values[index];
  }

  valueToSlider(value) {
    let closestIndex = 0;
    let closestDist = Math.abs(value - this.values[0]);
    for (let i = 1; i < this.values.length; i++) {
      const dist = Math.abs(value - this.values[i]);
      if (dist < closestDist) {
        closestDist = dist;
        closestIndex = i;
      }
    }
    return (closestIndex / (this.values.length - 1)) * 100;
  }
}
