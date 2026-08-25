import type { Box } from "./types";

/**
 * Direct port of `extract_features_v3` from notebooks/attribution_code.ipynb.
 * The ordering here is load-bearing: it must match FEATURE_NAMES_V3, because the
 * exported StandardScaler and the XGBoost split indices are both positional.
 */
export const FEATURE_NAMES = [
  "edge_dist",
  "centroid_dist",
  "same_frame",
  "iou",
  "above_bonus",
  "rel_x",
  "rel_y",
  "text_area",
  "body_area",
  "h_overlap",
] as const;

type FeatureName = (typeof FEATURE_NAMES)[number];

/** Human-readable gloss for each feature, used by the inspector panel. */
export const FEATURE_DOCS: Record<FeatureName, { label: string; blurb: string; group: string }> = {
  edge_dist: {
    label: "edge_dist",
    blurb: "Closest gap between the two boxes, over the page diagonal. 0 when they touch or overlap.",
    group: "Proximity",
  },
  centroid_dist: {
    label: "centroid_dist",
    blurb: "Centre-to-centre distance, over the page diagonal.",
    group: "Proximity",
  },
  same_frame: {
    label: "same_frame",
    blurb: "1 when both centroids fall inside the same panel. Carries 82% of the model's total gain.",
    group: "Structural",
  },
  iou: {
    label: "iou",
    blurb: "Intersection over union. Non-zero when the bubble sits on top of the body.",
    group: "Proximity",
  },
  above_bonus: {
    label: "above_bonus",
    blurb: "1 when the bubble sits above the body, the usual layout for a speaker.",
    group: "Relative position",
  },
  rel_x: {
    label: "rel_x",
    blurb: "Signed horizontal offset (bubble minus body), over page width.",
    group: "Relative position",
  },
  rel_y: {
    label: "rel_y",
    blurb: "Signed vertical offset (bubble minus body), over page height.",
    group: "Relative position",
  },
  text_area: {
    label: "text_area",
    blurb: "Bubble area as a fraction of the page.",
    group: "Size",
  },
  body_area: {
    label: "body_area",
    blurb: "Body area as a fraction of the page.",
    group: "Size",
  },
  h_overlap: {
    label: "h_overlap",
    blurb: "Horizontal overlap between bubble and body, normalised by bubble width.",
    group: "Structural",
  },
};

export function centroid(b: Box): [number, number] {
  return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
}

/** Minimum gap between two axis-aligned boxes; 0 when they touch or overlap. */
export function edgeDistance(a: Box, b: Box): number {
  const dx = Math.max(0, Math.max(a[0], b[0]) - Math.min(a[2], b[2]));
  const dy = Math.max(0, Math.max(a[1], b[1]) - Math.min(a[3], b[3]));
  return Math.sqrt(dx * dx + dy * dy);
}

export function boxIou(a: Box, b: Box): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const union =
    (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return inter / (union + 1e-8);
}

/** Index of the first frame containing this box's centroid, or -1. */
function containingFrame(box: Box, frames: Box[]): number {
  const [cx, cy] = centroid(box);
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    if (f[0] <= cx && cx <= f[2] && f[1] <= cy && cy <= f[3]) return i;
  }
  return -1;
}

export function sameFrame(textBox: Box, bodyBox: Box, frames: Box[]): number {
  const t = containingFrame(textBox, frames);
  const b = containingFrame(bodyBox, frames);
  return t !== -1 && t === b ? 1 : 0;
}

/**
 * `h_overlap` — how much the bubble and body occupy the same horizontal band.
 *
 * Do not "fix" this to match the paper's Table 2 wording. This is the exact
 * arithmetic `h_overlap_score` uses in attribution_code.ipynb, which is what the
 * shipped model was trained on: it measures the shared span over y, normalised
 * by the two boxes' combined vertical extent. Changing the formula here would
 * feed the classifier a feature it has never seen and silently degrade every
 * prediction on the site.
 */
function hOverlapScore(a: Box, b: Box): number {
  const overlap = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const union = Math.max(a[3], b[3]) - Math.min(a[1], b[1]);
  return overlap / (union + 1e-8);
}

/** The full 10-dimensional geometric feature vector for one (bubble, body) pair. */
export function extractFeatures(
  textBox: Box,
  bodyBox: Box,
  frames: Box[],
  imgW: number,
  imgH: number,
): number[] {
  const diag = Math.sqrt(imgW * imgW + imgH * imgH);
  const [tcx, tcy] = centroid(textBox);
  const [bcx, bcy] = centroid(bodyBox);
  return [
    edgeDistance(textBox, bodyBox) / diag,
    Math.hypot(tcx - bcx, tcy - bcy) / diag,
    sameFrame(textBox, bodyBox, frames),
    boxIou(textBox, bodyBox),
    tcy < bcy ? 1 : 0,
    (tcx - bcx) / imgW,
    (tcy - bcy) / imgH,
    ((textBox[2] - textBox[0]) * (textBox[3] - textBox[1])) / (imgW * imgH),
    ((bodyBox[2] - bodyBox[0]) * (bodyBox[3] - bodyBox[1])) / (imgW * imgH),
    hOverlapScore(textBox, bodyBox),
  ];
}
