import type { XgbModel } from "./types";

/**
 * Browser-side inference for the exported XGBoost attribution model.
 *
 * The model is emitted by scripts/export_web_assets.py in a canonical flattened
 * form (leaf values live in `splitCond`, and `left[i] === -1` marks a leaf), and
 * that script asserts this traversal reproduces `clf.predict_proba` to within
 * 1e-6 before it writes the file. So what runs here is the same model that
 * produced the 65.1% number, not a re-fit approximation.
 */

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Standardise a raw feature vector, selecting the model's own columns. */
export function standardize(model: XgbModel, rawFeatures: number[]): number[] {
  const out = new Array<number>(model.cols.length);
  for (let i = 0; i < model.cols.length; i++) {
    out[i] = (rawFeatures[model.cols[i]] - model.scalerMean[i]) / model.scalerScale[i];
  }
  return out;
}

/**
 * Sum of leaf outputs across every tree, in log-odds space.
 *
 * Both sides of the split comparison go through Math.fround first. XGBoost
 * stores thresholds as float32 and narrows its input to float32 too, and that
 * detail is load-bearing rather than pedantic: `same_frame` is binary, so after
 * standardisation it takes exactly two values and XGBoost places its threshold
 * right on the upper one. Comparing in float64 sends every same-panel pair down
 * the wrong branch of the root split.
 */
export function marginOf(model: XgbModel, scaled: number[]): number {
  let margin = model.baseMargin;
  for (const tree of model.trees) {
    let node = 0;
    while (tree.left[node] !== -1) {
      const v = scaled[tree.splitIndex[node]];
      if (Number.isNaN(v)) {
        node = tree.defaultLeft[node] ? tree.left[node] : tree.right[node];
      } else {
        node =
          Math.fround(v) < Math.fround(tree.splitCond[node])
            ? tree.left[node]
            : tree.right[node];
      }
    }
    margin += tree.splitCond[node];
  }
  return margin;
}

/** P(this body is the speaker) for one already-standardised pair. */
export function predictScaled(model: XgbModel, scaled: number[]): number {
  return sigmoid(marginOf(model, scaled));
}

/** P(this body is the speaker) from a raw 10-feature vector. */
export function predict(model: XgbModel, rawFeatures: number[]): number {
  return predictScaled(model, standardize(model, rawFeatures));
}

export function predictBatch(model: XgbModel, rawRows: number[][]): number[] {
  return rawRows.map((r) => predict(model, r));
}
