"use client";

import { useEffect, useState, type RefObject } from "react";
import type { PlanKey } from "../../../lib/accountApi";

const nf = new Intl.NumberFormat("zh-TW");
export const fmt = (n: number) => nf.format(Math.round(n));

/** 1_500_000 → "1.5M"，200_000 → "200K"，1_234 → "1.2K" */
export function compact(n: number): string {
  const one = (v: number) => (v >= 10 ? String(Math.round(v)) : String(Math.round(v * 10) / 10));
  const a = Math.abs(n);
  if (a >= 1e6) return `${one(n / 1e6)}M`;
  if (a >= 1e3) return `${one(n / 1e3)}K`;
  return String(Math.round(n));
}

export const PLAN_LABEL: Record<PlanKey, string> = { guest: "訪客", normal: "普通會員", vip: "VIP" };

export const FEATURE_LABEL: Record<string, string> = {
  socratic:   "需求釐清",
  walk:       "慢想探索",
  main_agent: "分析執行",
  suggest:    "概念推薦",
  other:      "其他",
};

export function formatDateTime(ms: number | null | undefined): string {
  if (!ms) return "—";
  return new Date(ms).toLocaleString("zh-TW", {
    year: "numeric", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export function formatDate(ms: number | null | undefined): string {
  if (!ms) return "—";
  return new Date(ms).toLocaleDateString("zh-TW", { year: "numeric", month: "long", day: "numeric" });
}

/** 與 dashboard 使用的 localStorage key 相同（見 app/dashboard/page.tsx）。 */
export const LS_SESSION = "kagrobot_session_id";
export const LS_HISTORY = "kagrobot_history";
export const LS_PHASE   = "kagrobot_phase";

export function useElementSize(ref: RefObject<HTMLElement | null>, fallback = { w: 400, h: 420 }) {
  const [size, setSize] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ w: Math.floor(width) || fallback.w, h: Math.floor(height) || fallback.h });
    });
    ro.observe(el);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ref]);
  return size;
}

/** 兩段式確認按鈕：第一次點擊變成「確定？」，再點一次才執行，5 秒沒動作自動還原。 */
export function ConfirmButton({
  label, confirmLabel = "確定刪除", onConfirm, disabled, className = "acct-btn danger",
}: {
  label: string; confirmLabel?: string; onConfirm: () => void; disabled?: boolean; className?: string;
}) {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 5000);
    return () => clearTimeout(t);
  }, [armed]);

  return (
    <button
      type="button"
      className={armed && !className.includes("danger") ? `${className} danger` : className}
      disabled={disabled}
      onClick={() => { if (armed) { setArmed(false); onConfirm(); } else setArmed(true); }}
      onBlur={() => setArmed(false)}
    >
      {armed ? confirmLabel : label}
    </button>
  );
}
