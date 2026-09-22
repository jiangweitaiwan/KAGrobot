"use client";

import { useEffect, useState } from "react";

/**
 * 圓相（ensō）額度環——會員中心唯一刻意做得有辨識度的元素，呼應首頁的圓相。
 * 圓被「寫滿」的程度 = 額度用量。顏色隨用量推進：苔綠（充裕）→ 金（接近上限）→ 紅（已滿）。
 * 進頁時筆觸由空畫到目前用量，是頁面唯一的入場動畫（prefers-reduced-motion 下關閉，見 CSS）。
 */
const R = 100;
const C = 2 * Math.PI * R;

export default function EnsoGauge({
  ratio, caption, unmetered = false,
}: {
  ratio: number;
  caption: string;
  unmetered?: boolean;   // 未啟用限額（QUOTA_ENFORCE=false）：仍畫用量，但不用警示色
}) {
  const r = Math.max(0, Math.min(1, ratio));
  const state = unmetered ? "ok" : r >= 1 ? "bad" : r >= 0.8 ? "warn" : "ok";
  const drawn = r === 0 ? 0 : Math.max(r, 0.012);

  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, []);

  const offset = C * (1 - (mounted ? drawn : 0));
  const pct = Math.round(r * 100);
  const label = r > 0 && pct === 0 ? "<1" : String(pct);

  return (
    <div className="acct-enso" data-state={state} role="img" aria-label={`${caption} 已使用 ${label}%`}>
      <svg viewBox="0 0 240 240" aria-hidden="true">
        <g transform="rotate(-90 120 120)">
          <circle className="track" cx="120" cy="120" r={R} />
          <circle className="brush-soft" cx="121.5" cy="118.5" r={R} strokeDasharray={C} strokeDashoffset={offset} />
          <circle className="brush" cx="120" cy="120" r={R} strokeDasharray={C} strokeDashoffset={offset} />
        </g>
      </svg>
      <div className="acct-enso-center">
        <span className="acct-enso-num">{label}<small>%</small></span>
        <span className="acct-enso-cap">{caption}</span>
      </div>
    </div>
  );
}
