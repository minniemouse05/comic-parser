"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import PageCanvas, { type CanvasToggles } from "./PageCanvas";
import { METHODS, METHOD_BY_ID, attributePage, scorePage } from "@/lib/attribution";
import { FEATURE_DOCS, FEATURE_NAMES, extractFeatures } from "@/lib/features";
import type { Box, MethodId, ModelsFile, PageRecord, PagesFile, XgbModel } from "@/lib/types";

type Loaded = {
  pages: PageRecord[];
  models: Record<string, XgbModel>;
  synthetic: boolean;
};

const fmt = (v: number, d = 3) =>
  Number.isFinite(v) ? v.toFixed(d) : v > 0 ? "∞" : "−∞";

export default function Demo() {
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [pageIdx, setPageIdx] = useState(0);
  const [method, setMethod] = useState<MethodId>("xgb_best");
  const [selectedText, setSelectedText] = useState<number | null>(null);
  const [hoverBody, setHoverBody] = useState<number | null>(null);
  const [toggles, setToggles] = useState<CanvasToggles>({
    art: true,
    frames: true,
    truth: false,
  });
  const [moved, setMoved] = useState<{ bodies?: Box[]; texts?: Box[] } | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [pagesRes, modelsRes] = await Promise.all([
          fetch("/data/pages.json"),
          fetch("/model/models.json").catch(() => null),
        ]);
        if (!pagesRes.ok) throw new Error(`pages.json: ${pagesRes.status}`);
        const pagesFile: PagesFile = await pagesRes.json();
        let models: Record<string, XgbModel> = {};
        if (modelsRes && modelsRes.ok) {
          const mf: ModelsFile = await modelsRes.json();
          models = mf.models ?? {};
        }
        if (!alive) return;
        setData({
          pages: pagesFile.pages ?? [],
          models,
          synthetic: (pagesFile.pages ?? []).some((p) => p.synthetic),
        });
        // Fall back to a method that can actually run.
        if (!models["xgb_ym_v3_conf_fixed300"]) setMethod("frame_aware");
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const page = data?.pages[pageIdx];

  const bodies = useMemo<Box[]>(
    () => moved?.bodies ?? page?.bodies.map((d) => d.box) ?? [],
    [moved, page],
  );
  const texts = useMemo<Box[]>(
    () => moved?.texts ?? page?.texts.map((d) => d.box) ?? [],
    [moved, page],
  );

  const attributions = useMemo(
    () =>
      page && data
        ? attributePage(page, method, data.models, { bodies, texts })
        : [],
    [page, data, method, bodies, texts],
  );

  const tally = useMemo(
    () =>
      page
        ? scorePage(page, attributions, { bodies })
        : { outcomes: [], correct: 0, wrong: 0, missed: 0 },
    [page, attributions, bodies],
  );

  const goPage = useCallback((next: number) => {
    setPageIdx(next);
    setSelectedText(null);
    setMoved(null);
  }, []);

  const onMoveBox = useCallback(
    (kind: "body" | "text", index: number, box: Box) => {
      setMoved((prev) => {
        const base = {
          bodies: prev?.bodies ?? bodies.slice(),
          texts: prev?.texts ?? texts.slice(),
        };
        const arr = kind === "body" ? base.bodies.slice() : base.texts.slice();
        arr[index] = box;
        return kind === "body"
          ? { bodies: arr, texts: base.texts }
          : { bodies: base.bodies, texts: arr };
      });
    },
    [bodies, texts],
  );

  if (error) {
    return (
      <div className="card demo">
        <p className="setup-banner">
          <strong>Could not load the demo data.</strong> {error}
        </p>
      </div>
    );
  }

  if (!data || !page) {
    return (
      <div className="card demo">
        <div className="demo-stage" style={{ minHeight: 300 }}>
          <p className="empty-side">Loading pages…</p>
        </div>
      </div>
    );
  }

  const hasModels = Object.keys(data.models).length > 0;
  const selected = selectedText !== null
    ? attributions.find((a) => a.textIdx === selectedText)
    : undefined;

  // Candidate bodies for the selected bubble, ranked by score.
  const ranked = selected
    ? selected.allScores
        .map((score, bodyIdx) => ({ score, bodyIdx }))
        .sort((a, b) => b.score - a.score)
    : [];
  const maxScore = ranked.length ? Math.max(...ranked.map((r) => r.score)) : 1;
  const learned = METHOD_BY_ID[method].kind === "learned";

  const inspectBody = hoverBody ?? selected?.bodyIdx ?? null;
  const featureRow =
    selectedText !== null && inspectBody !== null && bodies[inspectBody]
      ? extractFeatures(
          texts[selectedText],
          bodies[inspectBody],
          page.frames,
          page.width,
          page.height,
        )
      : null;

  return (
    <div className="card demo">
      {data.synthetic && (
        <p className="setup-banner">
          <strong>Placeholder pages.</strong> These are hand-built layouts, not Manga109.
          Run <code>scripts/export_web_assets.py</code> to swap in real held-out pages,
          detections and the trained model — see the README.
        </p>
      )}
      {!hasModels && (
        <p className="setup-banner">
          <strong>No trained model loaded.</strong> The geometric baselines run as normal;
          the XGBoost options need <code>public/model/models.json</code> from the export
          script.
        </p>
      )}

      <div className="demo-bar">
        <div className="seg">
          <button
            onClick={() => goPage((pageIdx - 1 + data.pages.length) % data.pages.length)}
            aria-label="Previous page"
          >
            ←
          </button>
          <button
            onClick={() => goPage((pageIdx + 1) % data.pages.length)}
            aria-label="Next page"
          >
            →
          </button>
        </div>
        <span style={{ fontSize: 13.5 }}>
          <strong>{page.book}</strong>{" "}
          <span style={{ color: "var(--ink-faint)" }}>
            · page {page.page} · {pageIdx + 1}/{data.pages.length}
          </span>
        </span>

        <div className="tally">
          <span className="ok">✓ {tally.correct}</span>
          <span className="bad">✗ {tally.wrong}</span>
          {tally.missed > 0 && <span className="miss">⊘ {tally.missed} missed by YOLO</span>}
        </div>
      </div>

      <div className="demo-body">
        <div className="demo-stage">
          <PageCanvas
            page={page}
            bodies={bodies}
            texts={texts}
            attributions={attributions}
            toggles={toggles}
            selectedText={selectedText}
            hoverBody={hoverBody}
            onSelectText={setSelectedText}
            onHoverBody={setHoverBody}
            onMoveBox={onMoveBox}
          />
        </div>

        <div className="demo-side">
          <div>
            <p className="side-h">Attribution method</p>
            <div className="methods">
              {METHODS.map((m) => {
                const missing = m.kind === "learned" && (!m.modelKey || !data.models[m.modelKey]);
                return (
                  <button
                    key={m.id}
                    className="method"
                    aria-pressed={method === m.id}
                    disabled={missing}
                    onClick={() => setMethod(m.id)}
                    title={missing ? "Run the export script to enable this model" : m.blurb}
                  >
                    <span className="method-row">
                      <span className="method-name">{m.label}</span>
                      <span className="method-acc">{(m.accuracy * 100).toFixed(1)}%</span>
                    </span>
                    <span className="method-blurb">{m.blurb}</span>
                  </button>
                );
              })}
            </div>
            <p
              style={{
                fontSize: 11.5,
                color: "var(--ink-faint)",
                margin: "8px 0 0",
                lineHeight: 1.45,
              }}
            >
              Percentages are held-out accuracy over all 22 validation volumes, not this page.
            </p>
          </div>

          <div>
            <p className="side-h">Overlay</p>
            <div className="toggles">
              {page.image && (
                <button
                  className="toggle"
                  aria-pressed={toggles.art}
                  onClick={() => setToggles((t) => ({ ...t, art: !t.art }))}
                >
                  <span className="dot" /> Artwork
                </button>
              )}
              <button
                className="toggle"
                aria-pressed={toggles.frames}
                onClick={() => setToggles((t) => ({ ...t, frames: !t.frames }))}
              >
                <span className="dot" /> Panels
              </button>
              <button
                className="toggle"
                aria-pressed={toggles.truth}
                onClick={() => setToggles((t) => ({ ...t, truth: !t.truth }))}
              >
                <span className="dot" /> Ground truth
              </button>
              {moved && (
                <button className="toggle" onClick={() => setMoved(null)}>
                  ↺ Reset boxes
                </button>
              )}
            </div>
          </div>

          <div>
            <p className="side-h">
              {selectedText === null ? "Inspector" : `Bubble ${selectedText} — candidates`}
            </p>

            {selectedText === null ? (
              <p className="empty-side">
                Click a <span style={{ color: "var(--bubble)" }}>bubble</span> to see every
                candidate body scored. Drag any box to move it and watch the scores update —
                the model runs live, in your browser.
              </p>
            ) : (
              <>
                <div className="cand">
                  {ranked.map(({ score, bodyIdx }) => (
                    <button
                      key={bodyIdx}
                      className={`cand-row${selected?.bodyIdx === bodyIdx ? " picked" : ""}`}
                      onPointerEnter={() => setHoverBody(bodyIdx)}
                      onPointerLeave={() => setHoverBody(null)}
                      onClick={() => setHoverBody(bodyIdx)}
                    >
                      <span className="swatch">{bodyIdx}</span>
                      <span className="barwrap">
                        <span
                          className="bar"
                          style={{
                            width: `${Math.max(
                              0,
                              Math.min(100, (score / (maxScore || 1)) * 100),
                            )}%`,
                          }}
                        />
                      </span>
                      <span className="val">
                        {learned ? fmt(score, 3) : fmt(score * 1000, 1)}
                      </span>
                    </button>
                  ))}
                </div>
                <p
                  style={{
                    fontSize: 11.5,
                    color: "var(--ink-faint)",
                    margin: "8px 0 0",
                    lineHeight: 1.45,
                  }}
                >
                  {learned
                    ? "P(this body is the speaker), straight from the gradient-boosted trees."
                    : "Inverse distance ×1000 — a ranking score, not a probability."}
                </p>
              </>
            )}
          </div>

          {featureRow && (
            <div>
              <p className="side-h">
                Features · bubble {selectedText} × body {inspectBody}
              </p>
              <table className="feat-table">
                <tbody>
                  {FEATURE_NAMES.map((name, i) => (
                    <tr key={name} className={name === "same_frame" ? "hot" : undefined}>
                      <td title={FEATURE_DOCS[name].blurb}>{name}</td>
                      <td>{featureRow[i].toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p
                style={{
                  fontSize: 11.5,
                  color: "var(--ink-faint)",
                  margin: "8px 0 0",
                  lineHeight: 1.45,
                }}
              >
                Exactly the vector <code>extract_features_v3</code> builds in the notebook.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
