"use client";

import type { DailyUsageItem } from "../../../lib/accountApi";
import { compact, fmt } from "./ui";

const W = 720, H = 220, PL = 44, PR = 8, PT = 10, PB = 26;

/** 把最大值往上取到 1/2/5 × 10^k 的整齊刻度。 */
function niceMax(v: number): number {
  if (v <= 0) return 1000;
  const exp  = Math.pow(10, Math.floor(Math.log10(v)));
  const frac = v / exp;
  const nice = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
  return nice * exp;
}

export default function DailyBars({ days }: { days: DailyUsageItem[] }) {
  const max = niceMax(Math.max(...days.map((d) => d.total_tokens), 0));
  const n   = days.length;
  const slot = (W - PL - PR) / n;
  const bw   = Math.max(2, slot * 0.68);
  const y    = (v: number) => PT + (H - PT - PB) * (1 - v / max);
  const labelEvery = Math.max(1, Math.ceil(n / 7));

  return (
    <svg
      className="acct-chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`最近 ${n} 天每日 token 用量長條圖`}
    >
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <line className="grid" x1={PL} x2={W - PR} y1={y(max * t)} y2={y(max * t)} />
          <text x={PL - 6} y={y(max * t) + 3} textAnchor="end">{compact(max * t)}</text>
        </g>
      ))}

      {days.map((d, i) => {
        const x = PL + slot * i + (slot - bw) / 2;
        const h = Math.max(0, y(0) - y(d.total_tokens));
        const isToday = i === n - 1;
        const zero = d.total_tokens === 0;
        return (
          <g key={d.date}>
            <rect
              className={`bar${isToday ? " today" : ""}${zero ? " zero" : ""}`}
              x={x} y={zero ? y(0) - 2 : y(d.total_tokens)} width={bw} height={zero ? 2 : h} rx={1.5}
            >
              <title>{`${d.date}　${fmt(d.total_tokens)} tokens　${d.turns} 輪對話`}</title>
            </rect>
            {(i % labelEvery === 0 || isToday) && (
              <text x={x + bw / 2} y={H - 8} textAnchor="middle">{d.date.slice(5).replace("-", "/")}</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
