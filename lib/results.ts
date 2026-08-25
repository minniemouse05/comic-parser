/**
 * Every number here is transcribed from the committed notebooks' own output
 * cells (notebooks/attribution_code.ipynb §11–§15, notebooks/detection_xgboost.ipynb)
 * and cross-checked against the paper's tables.
 */

export const HEADLINE = {
  accuracy: 0.6511,
  condAccuracy: 0.6985,
  evaluatedPairs: 25475,
  missedByYolo: 1731,
  missedPct: 0.068,
  valVolumes: 22,
  trainVolumes: 87,
  trainImages: 8525,
  valImages: 2077,
  trainSamples: 1958846,
  overNearestEdge: 0.0755,
  overFrameAware: 0.0178,
  overGtRegime: 0.0522,
  sameFrameGain: 0.82,
};

export interface Row {
  label: string;
  values: (number | string)[];
  best?: boolean;
  note?: string;
  group?: string;
}

export const detectionArchitectures = {
  caption:
    "Three YOLO families fine-tuned on Manga109 with identical settings (20 epochs, AdamW, lr 5e-4, batch 8, 640px). YOLOv8s won and became the baseline for everything downstream.",
  columns: ["Model", "mAP50", "mAP50-95", "Precision", "Recall", "F1"],
  rows: [
    { label: "YOLOv8s", values: [0.926, 0.679, 0.906, 0.857, 0.881], best: true },
    { label: "YOLOv11s", values: [0.921, 0.673, 0.904, 0.852, 0.877] },
    { label: "YOLOv26n", values: [0.89, 0.635, 0.874, 0.811, 0.842] },
  ] as Row[],
};

export const detectionAblation = {
  caption:
    "Backbone surgery aimed at texture bias made detection worse. Only CBAM in the neck helped, and only barely — fine-tuning had already done the domain adaptation the architecture changes were meant to force.",
  columns: ["Variant", "mAP50", "mAP50-95", "Precision", "Recall", "F1"],
  rows: [
    { label: "YOLOv8s (baseline)", values: [0.926, 0.679, 0.906, 0.857, 0.881] },
    {
      label: "V1 Shape-biased backbone",
      values: [0.922, 0.672, 0.906, 0.851, 0.878],
      note: "InstanceNorm stem + ViT block after SPPF; +3.18M params, −0.0068 mAP50-95",
    },
    {
      label: "V2 Style-robust C2f",
      values: [0.897, 0.644, 0.889, 0.82, 0.853],
      note: "InstanceNorm + Squeeze-and-Excite in every backbone C2f; worst of the four",
    },
    {
      label: "V3 CBAM neck",
      values: [0.928, 0.686, 0.908, 0.864, 0.886],
      best: true,
      note: "The only variant to beat baseline on every metric — bodies gained more than text (+0.0064 vs −0.0004 AP50)",
    },
  ] as Row[],
};

export const detectionPostFilter = {
  caption:
    "XGBoost over per-detection metadata (confidence, box geometry, position, class) at threshold 0.2. It buys precision with recall — so the attribution stage deliberately uses the unfiltered detector instead.",
  columns: ["System", "Precision", "Recall", "F1", "TP", "FP", "FN"],
  rows: [
    { label: "YOLO only", values: [0.863, 0.882, 0.873, "52,412", "8,332", "6,986"] },
    {
      label: "YOLO + XGBoost filter",
      values: [0.901, 0.86, 0.88, "51,109", "5,603", "8,289"],
      note: "Removes 33% of false positives, keeps 97.5% of true positives — but a missed body is unrecoverable downstream",
    },
  ] as Row[],
};

export const attributionMain = {
  caption:
    "All methods evaluated on the same 25,475 ground-truth pairs across the 22 held-out volumes. 1,731 (6.8%) were never detected by YOLO and count against overall accuracy; conditional accuracy drops those.",
  columns: ["Method", "Accuracy", "Conditional", "Wrong %"],
  rows: [
    { label: "Nearest centroid", values: [0.5558, 0.5963, "37.6%"], group: "Geometric baselines" },
    { label: "Nearest edge", values: [0.5756, 0.6175, "35.6%"], group: "Geometric baselines" },
    { label: "Frame-aware", values: [0.6333, 0.6795, "29.9%"], group: "Geometric baselines" },
    {
      label: "XGBoost — GT coordinates",
      values: [0.5989, 0.6426, "33.3%"],
      group: "Learned, ground-truth training regime",
      note: "Trained on Manga109's own boxes. Loses to the frame-aware heuristic it was meant to beat.",
    },
    {
      label: "XGBoost — GT + confidence weighting",
      values: [0.6028, 0.6468, "32.9%"],
      group: "Learned, ground-truth training regime",
    },
    {
      label: "XGBoost — YOLO-matched",
      values: [0.6508, 0.6982, "28.1%"],
      group: "Learned, YOLO-matched training regime",
      note: "Same features, same model, same hyperparameters. The only change is which boxes the features were computed from.",
    },
    {
      label: "XGBoost — YOLO-matched + confidence weighting",
      values: [0.6511, 0.6985, "28.1%"],
      group: "Learned, YOLO-matched training regime",
      best: true,
    },
  ] as Row[],
};

export const featureAblation = {
  caption:
    "Feature sets, all trained on YOLO-matched, confidence-weighted data. Dropping same_frame costs 2.1 points — and the 8-feature model still lands within a rounding error of the frame-aware heuristic, because edge_dist quietly absorbs the panel signal.",
  columns: ["Feature set", "# feats", "Accuracy", "Conditional", "Val AUC"],
  rows: [
    {
      label: "No same_frame, no h_overlap",
      values: [8, 0.6305, 0.6764, 0.9578],
      note: "edge_dist importance jumps from 0.09 to 0.60 to cover the gap",
    },
    { label: "No same_frame, + h_overlap", values: [9, 0.6323, 0.6784, 0.95826] },
    { label: "same_frame, no h_overlap", values: [9, 0.6496, 0.6969, 0.96389] },
    { label: "same_frame + h_overlap", values: [10, 0.6511, 0.6985, 0.96395], best: true },
  ] as Row[],
};

export const weightingAblation = {
  caption:
    "Sample weighting barely registers once training and inference see the same box distribution. Under the mismatched GT regime it does a little work — which is the tell: weighting was only ever compensating for the mismatch.",
  columns: ["Regime", "Weights", "Val AUC", "Accuracy", "Conditional"],
  rows: [
    { label: "GT coordinates", values: ["Uniform", 0.9645, 0.5989, 0.6426] },
    { label: "GT coordinates", values: ["Page-difficulty", 0.9645, 0.5961, 0.6395] },
    { label: "GT coordinates", values: ["YOLO-confidence", 0.9645, 0.6028, 0.6468] },
    { label: "YOLO-matched", values: ["Uniform", 0.96412, 0.6508, 0.6982] },
    { label: "YOLO-matched", values: ["Page-difficulty", 0.96383, 0.6505, 0.6979] },
    { label: "YOLO-matched", values: ["YOLO-confidence", 0.96395, 0.6511, 0.6985], best: true },
  ] as Row[],
};

export const featureImportance = [
  { name: "same_frame", best: 0.818, ablated: null },
  { name: "edge_dist", best: 0.084, ablated: 0.6 },
  { name: "centroid_dist", best: 0.036, ablated: 0.157 },
  { name: "iou", best: 0.018, ablated: 0.056 },
  { name: "above_bonus", best: 0.014, ablated: 0.079 },
  { name: "body_area", best: 0.011, ablated: 0.022 },
  { name: "rel_y", best: 0.009, ablated: 0.053 },
  { name: "h_overlap", best: 0.005, ablated: null },
  { name: "rel_x", best: 0.004, ablated: 0.02 },
  { name: "text_area", best: 0.002, ablated: 0.003 },
];
