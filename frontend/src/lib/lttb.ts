/**
 * Largest-Triangle-Three-Buckets downsampling (Steinarsson, 2013): keeps the visual shape
 * of a long series with a fixed number of points. Gaps (null values) are dropped first.
 */
export function lttb(x: ArrayLike<number>, y: ArrayLike<number | null>, threshold: number
                     ): { x: number[]; y: number[] } {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let k = 0; k < x.length; k++) {
    const v = y[k];
    if (v !== null && !Number.isNaN(v)) {
      xs.push(x[k]);
      ys.push(v);
    }
  }
  const n = xs.length;
  if (threshold >= n || threshold < 3) return { x: xs, y: ys };
  const outX = [xs[0]];
  const outY = [ys[0]];
  const every = (n - 2) / (threshold - 2);
  let a = 0;
  for (let i = 0; i < threshold - 2; i++) {
    const avgStart = Math.floor((i + 1) * every) + 1;
    const avgEnd = Math.min(Math.floor((i + 2) * every) + 1, n);
    let avgX = 0;
    let avgY = 0;
    for (let j = avgStart; j < avgEnd; j++) {
      avgX += xs[j];
      avgY += ys[j];
    }
    const count = Math.max(1, avgEnd - avgStart);
    avgX /= count;
    avgY /= count;
    const rangeStart = Math.floor(i * every) + 1;
    const rangeEnd = Math.floor((i + 1) * every) + 1;
    let maxArea = -1;
    let next = rangeStart;
    for (let j = rangeStart; j < rangeEnd; j++) {
      const area = Math.abs((xs[a] - avgX) * (ys[j] - ys[a]) - (xs[a] - xs[j]) * (avgY - ys[a]));
      if (area > maxArea) {
        maxArea = area;
        next = j;
      }
    }
    outX.push(xs[next]);
    outY.push(ys[next]);
    a = next;
  }
  outX.push(xs[n - 1]);
  outY.push(ys[n - 1]);
  return { x: outX, y: outY };
}
