/**
 * Stable per-run color assignment used everywhere a run is rendered.
 */

const PALETTE_VARS = [
  '--color-green',
  '--color-blue',
  '--color-orange',
  '--color-yellow',
  '--color-pink',
];

const FALLBACK = ['#34b87e', '#4474b8', '#f35c2d', '#feba14', '#f5a4c6'];

export function resolvePalette(): string[] {
  if (typeof window === 'undefined') return FALLBACK;
  const styles = getComputedStyle(document.documentElement);
  return PALETTE_VARS.map((v, i) => styles.getPropertyValue(v).trim() || FALLBACK[i]);
}

function hashInt(n: number): number {
  let h = n | 0;
  h = Math.imul(h ^ (h >>> 16), 2246822507);
  h = Math.imul(h ^ (h >>> 13), 3266489909);
  h ^= h >>> 16;
  return h >>> 0;
}

export function colorForRun(runId: number, palette?: string[]): string {
  const p = palette ?? resolvePalette();
  return p[hashInt(runId) % p.length];
}

export function colorForIndex(index: number, palette?: string[]): string {
  const p = palette ?? resolvePalette();
  return p[Math.max(0, index) % p.length];
}
