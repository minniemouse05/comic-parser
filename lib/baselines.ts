import type { Box } from "./types";
import { centroid, edgeDistance, sameFrame } from "./features";

/**
 * The three geometric baselines from §4.5 of the paper, ported from
 * notebooks/attribution_code.ipynb. Each returns a score per candidate body;
 * the caller takes the argmax, exactly as `evaluate()` does in the notebook.
 */

const INV = (d: number) => 1 / (d + 1e-8);

export function nearestCentroidScores(textBox: Box, bodies: Box[]): number[] {
  const [tx, ty] = centroid(textBox);
  return bodies.map((b) => {
    const [bx, by] = centroid(b);
    return INV(Math.hypot(tx - bx, ty - by));
  });
}

export function nearestEdgeScores(textBox: Box, bodies: Box[]): number[] {
  return bodies.map((b) => INV(edgeDistance(textBox, b)));
}

/**
 * Nearest edge, but restricted to bodies sharing the bubble's panel — falling
 * back to the whole page when nothing shares it. Bodies outside the winning
 * pool get -Infinity so the argmax can never stray out of it.
 */
export function frameAwareScores(textBox: Box, bodies: Box[], frames: Box[]): number[] {
  const inFrame = bodies.map((b) => sameFrame(textBox, b, frames) === 1);
  const restrict = inFrame.some(Boolean);
  return bodies.map((b, i) =>
    restrict && !inFrame[i] ? Number.NEGATIVE_INFINITY : INV(edgeDistance(textBox, b)),
  );
}
