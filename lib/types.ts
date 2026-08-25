/** [x1, y1, x2, y2] in absolute page pixels. */
export type Box = [number, number, number, number];

/** A YOLO detection: box plus its confidence. */
export interface Detection {
  box: Box;
  conf: number;
}

/** One human-annotated speaker -> text link from Manga109Dialog. */
export interface GtPair {
  text: Box;
  speaker: Box;
  /** Index into `texts` of the detection matched to this GT bubble (IoU >= 0.5), or -1 if YOLO missed it. */
  textDet: number;
  /** Index into `bodies` of the detection matched to the GT speaker (IoU >= 0.5), or -1. */
  speakerDet: number;
}

export interface PageRecord {
  id: string;
  book: string;
  page: number;
  width: number;
  height: number;
  /** Path under /public, or null for geometry-only pages. */
  image: string | null;
  /** Credit line to display over the artwork, per the Manga109 terms of use. */
  credit?: string;
  /** Ground-truth panel boundaries (Manga109 `frame` annotations). */
  frames: Box[];
  bodies: Detection[];
  texts: Detection[];
  gt: GtPair[];
  /** Set on the bundled placeholder pages so the UI can say so. */
  synthetic?: boolean;
}

export interface PagesFile {
  generated?: string;
  source?: string;
  note?: string;
  pages: PageRecord[];
}

/** A single boosted tree, flattened into parallel arrays (XGBoost's own layout). */
export interface TreeArrays {
  left: number[];
  right: number[];
  splitIndex: number[];
  splitCond: number[];
  defaultLeft: number[];
}

export interface XgbModel {
  featureNames: string[];
  /** Column indices into the full 10-feature vector that this model consumes. */
  cols: number[];
  /** StandardScaler parameters, in the model's own column order. */
  scalerMean: number[];
  scalerScale: number[];
  /** Bias added in margin (log-odds) space before the sigmoid. */
  baseMargin: number;
  trees: TreeArrays[];
  meta?: Record<string, unknown>;
}

export interface ModelsFile {
  generated?: string;
  models: Record<string, XgbModel>;
}

export type MethodId =
  | "nearest_centroid"
  | "nearest_edge"
  | "frame_aware"
  | "xgb_best"
  | "xgb_nosf";

/** One bubble's attribution result under one method. */
export interface Attribution {
  textIdx: number;
  /** Index into `bodies`, or -1 when there was nothing to pick. */
  bodyIdx: number;
  /** Model probability, or a heuristic score. */
  score: number;
  /** Score for every candidate body, aligned with `bodies`. */
  allScores: number[];
}
