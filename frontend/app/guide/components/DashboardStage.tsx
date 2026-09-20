"use client";

/**
 * DashboardStage.tsx — Section 1 的「運鏡舞台」。
 *
 * 核心概念：畫面上只有一份 dashboard（固定 1440×900 的虛擬畫布），
 * 切換步驟時不換內容，只移動相機（對畫布套 translate + scale）。
 * 兩份畫面（縮圖 + 放大圖）永遠對不齊，一份畫面不會。
 *
 * 運鏡規則：
 *   - 跨欄位（例如 chat → Global）先退回全景停一拍，再推進目標。
 *     直接從 A 補間到 B，使用者讀不出「我從左邊移到右邊」。
 *   - 同一欄位內換區域（Personal Brain 的剖面 → 送出列）只平移，
 *     不退回全景——因為要傳達的正是「這兩步在同一個面板」。
 *   - 目標完全相同（Detail Panel 的詳情 → 重心）則相機完全不動，
 *     只有面板內部換分頁。這是這個 demo 最想教會的一件事。
 *
 * 退回全景那一拍會畫出聚焦框（.gds-focus-ring），把「等下要放大的
 * 是這一塊」先講清楚，再放大。
 */

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import "../guide-stage.css";
import {
  CANVAS_W, CANVAS_H, HEADER_RECT, SLOT_RECTS, SLOT_ORDER, SLOT_TITLES,
  cameraFor, overviewCamera, sameRect,
  type Rect, type SlotKey,
} from "../dashboardMap";
import { BoardHeader } from "./dashboardSlots";

const PULLBACK_MS = 420;   // 停在全景的時間
const ZOOM_MS     = 720;   // 與 guide-stage.css 的 transition 時間一致

type Props = {
  focusSlot: SlotKey;
  focusRect: Rect;
  stepLabel: string;
  renderSlot: (key: SlotKey, live: boolean, scale: number, settled: boolean) => ReactNode;
};

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

export default function DashboardStage({ focusSlot, focusRect, stepLabel, renderSlot }: Props) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [vp, setVp] = useState({ w: 960, h: 540 });
  const reduced = usePrefersReducedMotion();

  const [phase,   setPhase]   = useState<"overview" | "focus">("focus");
  const [settled, setSettled] = useState(false);

  const prevRect = useRef<Rect | null>(null);
  const prevSlot = useRef<SlotKey | null>(null);

  // ── 量測可視範圍 ─────────────────────────────────────────
  useLayoutEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const apply = () => {
      const r = el.getBoundingClientRect();
      setVp({ w: Math.max(Math.floor(r.width), 320), h: Math.max(Math.floor(r.height), 240) });
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── 運鏡狀態機 ───────────────────────────────────────────
  useEffect(() => {
    const rectUnchanged = sameRect(prevRect.current, focusRect);
    const slotUnchanged = prevSlot.current === focusSlot;
    const isFirst       = prevRect.current === null;
    prevRect.current = focusRect;
    prevSlot.current = focusSlot;

    if (reduced || rectUnchanged) {
      setPhase("focus");
      setSettled(true);
      return;
    }

    // 同一欄位內移動：不退回全景，直接平移過去。
    if (slotUnchanged && !isFirst) {
      setPhase("focus");
      const t = setTimeout(() => setSettled(true), ZOOM_MS + 40);
      return () => clearTimeout(t);
    }

    setSettled(false);
    setPhase("overview");
    const t1 = setTimeout(() => setPhase("focus"), PULLBACK_MS);
    const t2 = setTimeout(() => setSettled(true), PULLBACK_MS + ZOOM_MS + 40);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [focusRect, focusSlot, reduced]);

  const cam = phase === "overview"
    ? overviewCamera(vp.w, vp.h)
    : cameraFor(focusRect, vp.w, vp.h);

  return (
    <div
      className="gds-stage"
      ref={stageRef}
      data-phase={phase}
      data-motion={reduced ? "reduced" : "full"}
      aria-hidden="true"
    >
      <div
        className="gds-canvas"
        style={{
          width:     CANVAS_W,
          height:    CANVAS_H,
          transform: `translate(${cam.tx}px, ${cam.ty}px) scale(${cam.scale})`,
        }}
      >
        <div className="gds-board" />
        <BoardHeader rect={HEADER_RECT} />

        {SLOT_ORDER.map((key) => {
          const rect = SLOT_RECTS[key];
          const live = key === focusSlot;
          return (
            <div
              key={key}
              className="gds-slot"
              data-live={live}
              style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}
            >
              <div className="gds-slot-title">{SLOT_TITLES[key]}</div>
              {renderSlot(key, live, cam.scale, settled)}
            </div>
          );
        })}

        {/* 全景那一拍：先框出等下要放大的區域 */}
        <div
          className="gds-focus-ring"
          style={{
            left:   focusRect.x - 6,
            top:    focusRect.y - 6,
            width:  focusRect.w + 12,
            height: focusRect.h + 12,
          }}
        />
      </div>

      <Minimap focusRect={focusRect} focusSlot={focusSlot} stepLabel={stepLabel} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Minimap — 放大到 2 倍以上時，使用者仍需要知道自己在哪
// ─────────────────────────────────────────────────────────────

function Minimap({
  focusRect, focusSlot, stepLabel,
}: {
  focusRect: Rect; focusSlot: SlotKey; stepLabel: string;
}) {
  return (
    <div className="gds-minimap">
      <svg viewBox={`0 0 ${CANVAS_W} ${CANVAS_H}`}>
        <rect x="0" y="0" width={CANVAS_W} height={CANVAS_H} rx="18"
              fill="rgba(232,238,248,0.03)" stroke="rgba(232,238,248,0.1)" strokeWidth="4" />
        <rect x={HEADER_RECT.x + 8} y={HEADER_RECT.y + 8}
              width={HEADER_RECT.w - 16} height={HEADER_RECT.h - 12}
              rx="8" fill="rgba(232,238,248,0.05)" />
        {SLOT_ORDER.map((key) => {
          const r = SLOT_RECTS[key];
          const on = key === focusSlot;
          return (
            <rect
              key={key}
              x={r.x} y={r.y} width={r.w} height={r.h} rx="12"
              fill={on ? "rgba(201,162,39,0.12)" : "rgba(232,238,248,0.045)"}
              stroke={on ? "rgba(201,162,39,0.55)" : "rgba(232,238,248,0.09)"}
              strokeWidth="4"
            />
          );
        })}
        <rect
          x={focusRect.x} y={focusRect.y} width={focusRect.w} height={focusRect.h}
          rx="10" fill="rgba(201,162,39,0.3)" stroke="#c9a227" strokeWidth="8"
          style={{ transition: "all 720ms cubic-bezier(0.66,0,0.26,1)" }}
        />
      </svg>
      <div className="gds-minimap-caption">{stepLabel}</div>
    </div>
  );
}
