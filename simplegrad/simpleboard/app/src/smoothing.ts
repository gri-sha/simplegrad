/**
 * TensorBoard-style exponential moving average with debiasing.
 *
 * weight in [0, 1). 0 = no smoothing (returns the input). At 0.99 the smoothed
 * value lags far behind the raw signal, matching TensorBoard's slider scale.
 */
export function smoothEMA(values: number[], weight: number): number[] {
  if (weight <= 0 || values.length === 0) return values.slice();

  const out = new Array<number>(values.length);
  let last = 0;
  let numAcc = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      out[i] = v;
      continue;
    }
    last = last * weight + (1 - weight) * v;
    numAcc += 1;
    const debias = 1 - Math.pow(weight, numAcc);
    out[i] = debias > 0 ? last / debias : last;
  }
  return out;
}
