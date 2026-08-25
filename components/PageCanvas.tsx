"use client";

import { useRef } from "react";
import type { Attribution, Box, PageRecord } from "@/lib/types";
import { boxIou } from "@/lib/features";

export interface CanvasToggles {
  art: boolean;
  frames: boolean;
  truth: boolean;
}

interface Props {
  page: PageRecord;
  bodies: Box[];
  texts: Box[];
  attributions: Attribution[];
  toggles: CanvasToggles;
  selectedText: number | null;
  hoverBody: number | null;
  onSelectText: (i: number | null) => void;
  onHoverBody: (i: number | null) => void;
  onMoveBox: (kind: "body" | "text", index: number, box: Box) => void;
}

const cx = (b: Box) => (b[0] + b[2]) / 2;
const cy = (b: Box) => (b[1] + b[3]) / 2;

export default function PageCanvas({
  page,
  bodies,
  texts,
  attributions,
  toggles,
  selectedText,
  hoverBody,
  onSelectText,
  onHoverBody,
  onMoveBox,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<{
    kind: "body" | "text";
    index: number;
    startX: number;
    startY: number;
    origin: Box;
  } | null>(null);

  const showArt = toggles.art && !!page.image;
  // Stroke widths are in page-pixel units, so scale them to the page size.
  const u = Math.max(page.width, page.height) / 700;

  /** Which GT pair (if any) this detected bubble corresponds to. */
  const gtForText = (textIdx: number) => page.gt.find((g) => g.textDet === textIdx);

  function beginDrag(
    e: React.PointerEvent,
    kind: "body" | "text",
    index: number,
    origin: Box,
  ) {
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    drag.current = { kind, index, startX: e.clientX, startY: e.clientY, origin };
  }

  function onPointerMove(e: React.PointerEvent) {
    const d = drag.current;
    const svg = svgRef.current;
    if (!d || !svg) return;
    const rect = svg.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const sx = page.width / rect.width;
    const sy = page.height / rect.height;
    let dx = (e.clientX - d.startX) * sx;
    let dy = (e.clientY - d.startY) * sy;
    // Keep the box on the page.
    dx = Math.max(-d.origin[0], Math.min(page.width - d.origin[2], dx));
    dy = Math.max(-d.origin[1], Math.min(page.height - d.origin[3], dy));
    onMoveBox(d.kind, d.index, [
      d.origin[0] + dx,
      d.origin[1] + dy,
      d.origin[2] + dx,
      d.origin[3] + dy,
    ]);
  }

  function endDrag(e: React.PointerEvent) {
    if (drag.current) {
      try {
        (e.target as Element).releasePointerCapture(e.pointerId);
      } catch {
        /* pointer already released */
      }
      drag.current = null;
    }
  }

  return (
    <div className={`stage-frame${showArt ? "" : " no-art"}`}>
      {showArt && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={page.image!} alt={`${page.book}, page ${page.page}`} draggable={false} />
      )}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${page.width} ${page.height}`}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onClick={() => onSelectText(null)}
        style={{ touchAction: "none", display: "block" }}
        role="img"
        aria-label={`Attribution overlay for ${page.book} page ${page.page}`}
      >
        {!showArt && (
          <rect
            x={0}
            y={0}
            width={page.width}
            height={page.height}
            fill="var(--bg-raised)"
          />
        )}

        {toggles.frames &&
          page.frames.map((f, i) => (
            <rect
              key={`f${i}`}
              x={f[0]}
              y={f[1]}
              width={f[2] - f[0]}
              height={f[3] - f[1]}
              fill="none"
              stroke="var(--frame)"
              strokeWidth={1.6 * u}
              strokeDasharray={`${7 * u} ${5 * u}`}
              opacity={0.85}
            />
          ))}

        {/* Ground-truth links sit underneath the predictions. */}
        {toggles.truth &&
          page.gt.map((g, i) => (
            <line
              key={`gt${i}`}
              x1={cx(g.text)}
              y1={cy(g.text)}
              x2={cx(g.speaker)}
              y2={cy(g.speaker)}
              stroke="var(--gt)"
              strokeWidth={2.4 * u}
              strokeDasharray={`${5 * u} ${4 * u}`}
              opacity={0.75}
            />
          ))}

        {/* Predicted links. */}
        {attributions.map((a) => {
          if (a.bodyIdx < 0) return null;
          const t = texts[a.textIdx];
          const b = bodies[a.bodyIdx];
          if (!t || !b) return null;
          const gt = gtForText(a.textIdx);
          const state = !gt
            ? "neutral"
            : boxIou(b, gt.speaker) >= 0.5
              ? "correct"
              : "wrong";
          const dim = selectedText !== null && selectedText !== a.textIdx;
          const stroke =
            state === "correct"
              ? "var(--correct)"
              : state === "wrong"
                ? "var(--wrong)"
                : "var(--ink-faint)";
          return (
            <g key={`p${a.textIdx}`} opacity={dim ? 0.18 : 1} pointerEvents="none">
              <line
                x1={cx(t)}
                y1={cy(t)}
                x2={cx(b)}
                y2={cy(b)}
                stroke={stroke}
                strokeWidth={3 * u}
                strokeLinecap="round"
              />
              <circle cx={cx(b)} cy={cy(b)} r={4.5 * u} fill={stroke} />
            </g>
          );
        })}

        {/* Bodies. */}
        {bodies.map((b, i) => {
          const picked =
            selectedText !== null &&
            attributions.find((a) => a.textIdx === selectedText)?.bodyIdx === i;
          const hot = hoverBody === i || picked;
          return (
            <g key={`b${i}`}>
              <rect
                x={b[0]}
                y={b[1]}
                width={b[2] - b[0]}
                height={b[3] - b[1]}
                fill={hot ? "var(--body-fill)" : "transparent"}
                stroke="var(--body)"
                strokeWidth={(hot ? 3.2 : 1.8) * u}
                rx={3 * u}
                style={{ cursor: "grab" }}
                onPointerDown={(e) => beginDrag(e, "body", i, b)}
                onPointerEnter={() => onHoverBody(i)}
                onPointerLeave={() => onHoverBody(null)}
              />
              <text
                x={b[0] + 5 * u}
                y={b[1] + 17 * u}
                fill="var(--body)"
                fontSize={14 * u}
                fontFamily="var(--mono)"
                fontWeight={600}
                pointerEvents="none"
                style={{ paintOrder: "stroke", stroke: "var(--bg-raised)", strokeWidth: 3.5 * u }}
              >
                {i}
              </text>
            </g>
          );
        })}

        {/* Bubbles. */}
        {texts.map((t, i) => {
          const sel = selectedText === i;
          return (
            <rect
              key={`t${i}`}
              x={t[0]}
              y={t[1]}
              width={t[2] - t[0]}
              height={t[3] - t[1]}
              fill={sel ? "var(--bubble-fill)" : "transparent"}
              stroke="var(--bubble)"
              strokeWidth={(sel ? 3.4 : 2) * u}
              rx={3 * u}
              style={{ cursor: "pointer" }}
              onPointerDown={(e) => beginDrag(e, "text", i, t)}
              onClick={(e) => {
                e.stopPropagation();
                onSelectText(sel ? null : i);
              }}
            />
          );
        })}
      </svg>

      {showArt && page.credit && <span className="credit">{page.credit}</span>}
    </div>
  );
}
