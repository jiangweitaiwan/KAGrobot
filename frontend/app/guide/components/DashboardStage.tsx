"use client";

/**
 * DashboardStage.tsx — Section 1 的「定點導覽舞台」。
 *
 * 改版重點：不再運鏡。整個 dashboard 全景（1440×560 虛擬畫布）從頭到尾
 * 都用同一個 overview 相機顯示——不放大、不位移、背景不調暗。使用者
 * 隨時都看得到完整的五欄佈局，切換步驟只改變哪一欄是「作用中」：
 * 作用中的欄位套用既有的金色框線＋光暈（.gds-slot[data-live="true"]），
 * 內部從骨架換成真內容；其餘欄位維持原樣，不做任何壓暗處理。
 *
 * 這是刻意的取捨：舊版靠鏡頭運動＋壓暗來製造「這一步在這裡」的戲劇性，
 * 但代價是使用者看不清楚 dashboard 全貌、切換時有閃現感。現在反過來，
 * 讓「隨時看得到完整佈局」優先於「炫技的鏡頭語言」。
 *
 * 原本「退全景 → 停一拍 → 推近放大」的狀態機已整個拿掉。唯一留著的
 * 計時器是 SETTLE_MS：切到新欄位後隔一個很短的延遲才掛真元件
 * （GlobalBrainExplorer / HourglassViz 等 WebGL 元件），純粹讓
 * 骨架 → 真內容 的切換不要一瞬間跳出來，跟鏡頭無關。
 */

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import "../guide-stage.css";
import {
  CANVAS_W, CANVAS_H, HEADER_RECT, SLOT_RECTS, SLOT_ORDER, SLOT_TITLES,
  overviewCamera,
  type SlotKey,
} from "../dashboardMap";
import { BoardHeader } from "./dashboardSlots";

const SETTLE_MS = 220; // 切欄位後多久才掛真元件，純粹讓切換不要太突兀，跟鏡頭無關

type Props = {
  focusSlot:  SlotKey;
  stepLabel:  string;
  stepIndex:  number;
  stepCount:  number;
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

export default function DashboardStage({ focusSlot, stepLabel, stepIndex, stepCount, renderSlot }: Props) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [vp, setVp] = useState({ w: 960, h: 540 });
  const reduced = usePrefersReducedMotion();

  const [settled, setSettled] = useState(true);

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

  // ── 切欄位：短暫退回 skeleton，再掛真元件 ───────────────────
  useEffect(() => {
    if (reduced) { setSettled(true); return; }
    setSettled(false);
    const t = setTimeout(() => setSettled(true), SETTLE_MS);
    return () => clearTimeout(t);
  }, [focusSlot, reduced]);

  const cam = overviewCamera(vp.w, vp.h);

  return (
    <div
      className="gds-stage"
      ref={stageRef}
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
      </div>

      <StepDots stepIndex={stepIndex} stepCount={stepCount} stepLabel={stepLabel} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// StepDots — 取代原本的 Minimap。全景本來就一直看得到，不需要縮圖
// 再標示「這一格在畫布的哪裡」；只需要一個很輕量的「第幾步╱目前
// 在做什麼」提示，讓使用者知道進度。
// ─────────────────────────────────────────────────────────────

function StepDots({
  stepIndex, stepCount, stepLabel,
}: {
  stepIndex: number; stepCount: number; stepLabel: string;
}) {
  return (
    <div className="gds-stepdots">
      <div className="gds-stepdots-row">
        {Array.from({ length: stepCount }).map((_, i) => (
          <span key={i} className="gds-stepdot" data-on={i === stepIndex} />
        ))}
      </div>
      <div className="gds-stepdots-caption">{stepLabel}</div>
    </div>
  );
}
