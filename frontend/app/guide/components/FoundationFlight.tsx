"use client";

/**
 * FoundationFlight.tsx — 「加入 Personal Brain」的飛行動畫層。
 *
 * Section 2 要讓使用者看見一件在 dashboard 上其實看不見的事：按下
 * 「加入」之後，那幾個概念去了哪裡。所以這裡把它畫出來——從按下的
 * 位置起飛，走一段拋物線，落在右邊語意地圖上它該在的座標。
 *
 * 用 fixed 定位 + portal 掛到 body：飛行途中會跨越三個欄位的邊界，
 * 任何一層 overflow:hidden 都會把它截掉。
 *
 * 起點刻意取「使用者實際按下的滑鼠座標」而不是某個 DOM 節點——
 * 這樣就不需要知道 DetailPanel 內部把加入按鈕渲染在哪，共用元件
 * 一行都不用改。見 GlobalBrainSection 的 onPointerDownCapture。
 */

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Point } from "../bertopicProject";

export type Flight = {
  id:    string;
  eid:   string;
  label: string;
  color: string;
  from:  Point;   // viewport 座標
  to:    Point;   // viewport 座標
};

const FLY_MS    = 720;
const STAGGER_MS = 110;

type Props = {
  flights: Flight[];
  onLand:  (flight: Flight) => void;
  onDone:  () => void;
};

export default function FoundationFlight({ flights, onLand, onDone }: Props) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const onLandRef = useRef(onLand);
  const onDoneRef = useRef(onDone);
  useEffect(() => { onLandRef.current = onLand; onDoneRef.current = onDone; });

  useEffect(() => {
    if (flights.length === 0) return;
    const timers = flights.map((f, i) =>
      setTimeout(() => onLandRef.current(f), i * STAGGER_MS + FLY_MS),
    );
    const last = setTimeout(
      () => onDoneRef.current(),
      (flights.length - 1) * STAGGER_MS + FLY_MS + 420,
    );
    return () => { timers.forEach(clearTimeout); clearTimeout(last); };
  }, [flights]);

  if (!mounted || flights.length === 0) return null;

  return createPortal(
    <div className="gfl-layer">
      <svg className="gfl-trails">
        {flights.map((f, i) => {
          const mid = arcMid(f.from, f.to);
          return (
            <path
              key={f.id}
              d={`M ${f.from.x} ${f.from.y} Q ${mid.x} ${mid.y} ${f.to.x} ${f.to.y}`}
              stroke={f.color}
              style={{ animationDelay: `${i * STAGGER_MS}ms` }}
            />
          );
        })}
      </svg>

      {flights.map((f, i) => (
        <FlyingChip key={f.id} flight={f} delay={i * STAGGER_MS} />
      ))}
    </div>,
    document.body,
  );
}

function arcMid(from: Point, to: Point): Point {
  return {
    x: (from.x + to.x) / 2,
    y: Math.min(from.y, to.y) - Math.max(56, Math.abs(to.x - from.x) * 0.18),
  };
}

function FlyingChip({ flight, delay }: { flight: Flight; delay: number }) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const { from, to } = flight;
    const mid = arcMid(from, to);

    const anim = el.animate(
      [
        { transform: `translate(${from.x}px, ${from.y}px) translate(-50%, -50%) scale(1)`,  opacity: 0 },
        { transform: `translate(${from.x}px, ${from.y}px) translate(-50%, -50%) scale(1)`,  opacity: 1, offset: 0.08 },
        { transform: `translate(${mid.x}px, ${mid.y}px) translate(-50%, -50%) scale(0.92)`, opacity: 1, offset: 0.55 },
        { transform: `translate(${to.x}px, ${to.y}px) translate(-50%, -50%) scale(0.3)`,    opacity: 0 },
      ],
      { duration: FLY_MS, delay, easing: "cubic-bezier(0.4, 0, 0.25, 1)", fill: "both" },
    );
    return () => anim.cancel();
  }, [flight, delay]);

  return (
    <div
      ref={ref}
      className="gfl-chip"
      style={{ borderColor: flight.color, color: flight.color, opacity: 0 }}
    >
      {flight.label}
    </div>
  );
}
