import type { Row } from "@/lib/results";
import { Fragment } from "react";

interface Props {
  caption: string;
  columns: string[];
  rows: Row[];
}

const render = (v: number | string) =>
  typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(v < 1 ? 4 : 3)) : v;

export default function ResultTable({ caption, columns, rows }: Props) {
  let lastGroup: string | undefined;

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const newGroup = r.group && r.group !== lastGroup;
            if (r.group) lastGroup = r.group;
            return (
              <Fragment key={i}>
                {newGroup && (
                  <tr className="group-head">
                    <td colSpan={columns.length}>{r.group}</td>
                  </tr>
                )}
                <tr className={r.best ? "best" : undefined}>
                  <td>{r.label}</td>
                  {r.values.map((v, j) => (
                    <td key={j}>{render(v)}</td>
                  ))}
                </tr>
                {r.note && (
                  <tr>
                    <td className="note-cell" colSpan={columns.length}>
                      {r.note}
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
        <caption>{caption}</caption>
      </table>
    </div>
  );
}
