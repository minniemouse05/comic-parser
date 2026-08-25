/**
 * Proves the browser's XGBoost inference still agrees with scikit-learn.
 *
 * scripts/export_web_assets.py writes tests/parity-fixture.json alongside the
 * models: 200 real feature rows per model plus the probabilities that
 * clf.predict_proba produced for them in Python. This walks the same rows
 * through lib/xgb.ts — the exact code the site ships — and fails if they drift.
 *
 *   npm run check:inference
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { predict } from "../lib/xgb.ts";
import type { XgbModel } from "../lib/types.ts";

const here = dirname(fileURLToPath(import.meta.url));
const TOLERANCE = 1e-6;

interface FixtureEntry {
  probe: number[][];
  expected: number[];
}

function read<T>(path: string): T | null {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as T;
  } catch {
    return null;
  }
}

const models = read<{ models: Record<string, XgbModel> }>(
  join(here, "..", "public", "model", "models.json"),
);
const fixture = read<Record<string, FixtureEntry>>(join(here, "parity-fixture.json"));

if (!models || !fixture) {
  console.log(
    "No exported model or fixture found — skipping.\n" +
      "Run scripts/export_web_assets.py first (see README, 'Regenerating the demo data').",
  );
  process.exit(0);
}

let failed = false;
for (const [key, entry] of Object.entries(fixture)) {
  const model = models.models[key];
  if (!model) {
    console.error(`  ${key}: MISSING from public/model/models.json`);
    failed = true;
    continue;
  }
  let worst = 0;
  for (let i = 0; i < entry.probe.length; i++) {
    worst = Math.max(worst, Math.abs(predict(model, entry.probe[i]) - entry.expected[i]));
  }
  const ok = worst < TOLERANCE;
  failed ||= !ok;
  console.log(
    `  ${ok ? "PASS" : "FAIL"}  ${key}: ${entry.probe.length} rows, ` +
      `max|Δ| vs sklearn = ${worst.toExponential(3)}`,
  );
}

console.log(
  failed
    ? "\nBrowser inference has drifted from scikit-learn."
    : "\nBrowser inference matches scikit-learn.",
);
process.exit(failed ? 1 : 0);
