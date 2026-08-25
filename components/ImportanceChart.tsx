"use client";

import { useState } from "react";
import { featureImportance } from "@/lib/results";

/**
 * Feature importance (gain) for the best model, with the 8-feature ablation as a
 * second view. Switching between them is the point: strip same_frame out and
 * edge_dist inflates from 0.08 to 0.60 to cover for it.
 */
export default function ImportanceChart() {
  const [view, setView] = useState<"best" | "ablated">("best");
  const rows = featureImportance
    .map((f) => ({ name: f.name, value: view === "best" ? f.best : f.ablated }))
    .filter((f): f is { name: string; value: number } => f.value !== null)
    .sort((a, b) => b.value - a.value);
  const top = rows[0]?.name;

  return (
    <div>
      <div className="seg" style={{ marginTop: 24 }}>
        <button aria-pressed={view === "best"} onClick={() => setView("best")}>
          Best model · 10 features
        </button>
        <button aria-pressed={view === "ablated"} onClick={() => setView("ablated")}>
          Ablation · no same_frame
        </button>
      </div>

      <div className="imp">
        {rows.map((r) => (
          <div className="imp-row" key={r.name}>
            <span className="imp-name">{r.name}</span>
            <span className="imp-track">
              <span
                className={`imp-fill${r.name === top ? " hot" : ""}`}
                style={{ width: `${r.value * 100}%` }}
              />
            </span>
            <span className="imp-val">{r.value.toFixed(3)}</span>
          </div>
        ))}
      </div>

      <p className="imp-legend">
        {view === "best"
          ? "same_frame alone carries 82% of the model's total gain."
          : "With same_frame gone, edge_dist absorbs the load — 0.60 gain, up from 0.08."}
      </p>
    </div>
  );
}
