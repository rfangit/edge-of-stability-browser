// ============================================================================
// NUMBER FORMATTING UTILITIES
// ============================================================================
// Single source of truth for all number formatting in the app

/**
 * Format number for LaTeX display with scientific notation
 * - If value >= 0.1: show with 3 decimals, strip trailing zeros
 * - If value < 0.1: show as x.xxx \times 10^k (3 sig figs in mantissa)
 * - Special case: if mantissa rounds to 1.000, show as 10^k
 * - Very special case: if value is exactly a power of 10 (like 0.001), show as 10^k with 1 digit precision
 */
export function formatForLatex(value) {
  if (value === 0) return '0';
  if (!isFinite(value)) return '\\text{undefined}';

  const absValue = Math.abs(value);

  // For values >= 0.1, use fixed decimal notation
  if (absValue >= 0.1) {
    const formatted = value.toFixed(3);
    return formatted.replace(/\.?0+$/, '');
  }

  // For small values, use scientific notation
  const exponent = Math.floor(Math.log10(absValue));
  const mantissa = value / Math.pow(10, exponent);

  // Check if this is a clean power of 10 (like 0.001 = 10^-3)
  const isPowerOfTen = Math.abs(absValue - Math.pow(10, exponent)) < 1e-12;

  if (isPowerOfTen) {
    // Show as 10^k with no mantissa
    return `10^{${exponent}}`;
  }

  // Round mantissa to 3 decimal places (4 total digits including leading digit)
  const roundedMantissa = parseFloat(mantissa.toFixed(3));

  // If mantissa rounds to 1, show as 10^k
  if (Math.abs(roundedMantissa - 1) < 1e-10) {
    return `10^{${exponent}}`;
  }

  // Otherwise show as x.xxx × 10^k
  const mantissaStr = roundedMantissa.toFixed(3).replace(/\.?0+$/, '');
  return `${mantissaStr} \\times 10^{${exponent}}`;
}
