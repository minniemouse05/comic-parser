import type { Attribution, Box, MethodId, PageRecord, XgbModel } from "./types";
import { extractFeatures, boxIou } from "./features";
import { frameAwareScores, nearestCentroidScores, nearestEdgeScores } from "./baselines";
import { predict } from "./xgb";

interface MethodSpec {
  id: MethodId;
  label: string;
  short: string;
  kind: "geometric" | "learned";
  /** Which exported model this method needs, if any. */
  modelKey?: string;
  blurb: string;
  /** Held-out accuracy on the 22 validation volumes (from the paper). */
  accuracy: number;
  condAccuracy: number;
}

export const METHODS: MethodSpec[] = [
  {
    id: "nearest_centroid",
    label: "Nearest centroid",
    short: "centroid",
    kind: "geometric",
    blurb: "Pick the body whose centre is closest to the bubble's centre.",
    accuracy: 0.5558,
    condAccuracy: 0.5963,
  },
  {
    id: "nearest_edge",
    label: "Nearest edge",
    short: "edge",
    kind: "geometric",
    blurb: "Pick the body with the smallest gap to the bubble. A strong, stubborn baseline.",
    accuracy: 0.5756,
    condAccuracy: 0.6175,
  },
  {
    id: "frame_aware",
    label: "Frame-aware",
    short: "frame",
    kind: "geometric",
    blurb:
      "Nearest edge, but only among bodies inside the same panel — falling back to the whole page when the panel is empty.",
    accuracy: 0.6333,
    condAccuracy: 0.6795,
  },
  {
    id: "xgb_nosf",
    label: "XGBoost — no same_frame",
    short: "xgb-8",
    kind: "learned",
    modelKey: "xgb_nosf_nohov",
    blurb:
      "The 8-feature ablation: proximity, position and size only, with both structural features removed.",
    accuracy: 0.6305,
    condAccuracy: 0.6764,
  },
  {
    id: "xgb_best",
    label: "XGBoost — best model",
    short: "xgb-10",
    kind: "learned",
    modelKey: "xgb_ym_v3_conf_fixed300",
    blurb:
      "All 10 features, trained on YOLO-matched coordinates with confidence weighting. The paper's headline model.",
    accuracy: 0.6511,
    condAccuracy: 0.6985,
  },
];

export const METHOD_BY_ID = Object.fromEntries(METHODS.map((m) => [m.id, m])) as Record<
  MethodId,
  MethodSpec
>;

/**
 * Score every (bubble, body) pair on a page and take the argmax per bubble —
 * the same procedure the notebook's `attribute_*` functions use at eval time.
 *
 * `overrides` lets the UI hand in dragged box positions without mutating the
 * underlying page record.
 */
export function attributePage(
  page: PageRecord,
  method: MethodId,
  models: Record<string, XgbModel>,
  overrides?: { bodies?: Box[]; texts?: Box[] },
): Attribution[] {
  const bodies = overrides?.bodies ?? page.bodies.map((d) => d.box);
  const texts = overrides?.texts ?? page.texts.map((d) => d.box);
  const spec = METHOD_BY_ID[method];

  return texts.map((textBox, textIdx) => {
    if (bodies.length === 0) {
      return { textIdx, bodyIdx: -1, score: 0, allScores: [] };
    }

    let allScores: number[];
    if (method === "nearest_centroid") {
      allScores = nearestCentroidScores(textBox, bodies);
    } else if (method === "nearest_edge") {
      allScores = nearestEdgeScores(textBox, bodies);
    } else if (method === "frame_aware") {
      allScores = frameAwareScores(textBox, bodies, page.frames);
    } else {
      const model = spec.modelKey ? models[spec.modelKey] : undefined;
      if (!model) return { textIdx, bodyIdx: -1, score: 0, allScores: [] };
      allScores = bodies.map((b) =>
        predict(model, extractFeatures(textBox, b, page.frames, page.width, page.height)),
      );
    }

    let bodyIdx = 0;
    for (let i = 1; i < allScores.length; i++) {
      if (allScores[i] > allScores[bodyIdx]) bodyIdx = i;
    }
    return { textIdx, bodyIdx, score: allScores[bodyIdx], allScores };
  });
}

type Outcome = "correct" | "wrong" | "missed";

/**
 * Score a page against Manga109Dialog ground truth using the paper's protocol
 * (§4.9): a GT pair YOLO never detected is `missed` and unrecoverable; otherwise
 * the predicted body must overlap the true speaker at IoU >= 0.5.
 */
export function scorePage(
  page: PageRecord,
  attributions: Attribution[],
  overrides?: { bodies?: Box[] },
): { outcomes: Outcome[]; correct: number; wrong: number; missed: number } {
  const bodies = overrides?.bodies ?? page.bodies.map((d) => d.box);
  const byText = new Map(attributions.map((a) => [a.textIdx, a]));
  const outcomes: Outcome[] = [];
  let correct = 0;
  let wrong = 0;
  let missed = 0;

  for (const pair of page.gt) {
    if (pair.textDet < 0) {
      outcomes.push("missed");
      missed++;
      continue;
    }
    const a = byText.get(pair.textDet);
    const predicted = a && a.bodyIdx >= 0 ? bodies[a.bodyIdx] : undefined;
    if (predicted && boxIou(predicted, pair.speaker) >= 0.5) {
      outcomes.push("correct");
      correct++;
    } else {
      outcomes.push("wrong");
      wrong++;
    }
  }
  return { outcomes, correct, wrong, missed };
}
