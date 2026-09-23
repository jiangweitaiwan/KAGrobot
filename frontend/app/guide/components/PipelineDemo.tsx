"use client";

/**
 * PipelineDemo.tsx — Section 1:「完整流程」六步驟示範。
 *
 * 改版重點：不再靠鏡頭運鏡+壓暗來凸顯「這一步在哪裡」，改成一份
 * 完整的 dashboard 骨架全程以全景呈現、不放大不位移、背景不調暗；
 * 作用中的欄位單純用既有的金色框線＋光暈標示（見 DashboardStage.tsx
 * 的說明）。使用者因此隨時能看到完整佈局，減少切步驟時的閃現感。
 *
 * 六個步驟只落在四個相機位置上，這是刻意的：
 *   ③讀定義 / ④加重心 都在 Detail Panel，
 *   ⑤看剖面 / ⑥送出   都在 Personal Brain。
 * 欄位不動、只有面板內部換分頁或冒出確認列，使用者就會自己讀出
 * 「這兩步是在同一個面板做的」——這件事用六張獨立卡片講不出來。
 *
 * 真元件（GlobalBrainExplorer / HourglassViz）只在該欄位作用中、
 * 且短暫延遲（跟鏡頭無關，純粹讓切換不要太突兀）後才掛載，見
 * dashboardSlots.tsx 的說明。
 */

import { useCallback, useEffect, useRef, useState } from "react";
import DashboardStage from "./DashboardStage";
import {
  ChatSlot, GlobalSlot, DetailSlot, SemanticSlot, PersonalSlot,
} from "./dashboardSlots";
import { SLOT_RECTS, type SlotKey } from "../dashboardMap";

type Step = {
  key:          string;
  glyph:        string;
  title:        string;
  slot:         SlotKey;
  /** Detail Panel 兩個步驟共用一格，用這個決定顯示哪個分頁。 */
  detailTab?:   "detail" | "brain";
  /** Personal Brain 步驟⑥才冒出送出確認列。 */
  showConfirm?: boolean;
  where:        string;
  instruction:  string;
};

const STEPS: Step[] = [
  {
    key: "chat", glyph: "💬", title: "提問", slot: "chat",
    where: "最左邊的對話欄",
    instruction: "輸入你關心的問題。系統會依語意到 GlobalBrain 裡找出相關概念，這是整個迴圈的起點。",
  },
  {
    key: "explore", glyph: "🌌", title: "探索", slot: "global",
    where: "中央的 Global Knowledge Field",
    instruction: "拖曳旋轉、滾輪縮放，點任一節點。這裡點的是「價格僵固性」——被標示的是目前語意上最相關的候選。",
  },
  {
    key: "read", glyph: "🔍", title: "閱讀", slot: "detail", detailTab: "detail",
    where: "右邊的 Detail Panel",
    instruction: "讀這個概念的定義與機制，以及支撐它的更基礎概念。相機停在這一格，因為下一步也在這裡。",
  },
  {
    key: "collect", glyph: "➕", title: "收藏", slot: "detail", detailTab: "brain",
    where: "同一個 Detail Panel，切到「重心」分頁",
    instruction: "把有用的概念加進 Personal Brain。每個概念帶一個權重，決定它在後續排名裡的份量。",
  },
  {
    key: "profile", glyph: "⏳", title: "剖面", slot: "personal",
    where: "最右邊的 Personal Brain",
    instruction: "收藏累積之後會收斂出推薦 RC。這裡是「通膨黏性價格假說」，往上延伸得到「升息衝擊分析」。",
  },
  {
    key: "launch", glyph: "⚡", title: "送出", slot: "personal", showConfirm: true,
    where: "同一個 Personal Brain，下方的確認列",
    instruction: "選定 RC 後送出，交給 MainAgent 執行對應的模型計算——不是再開一輪對話。",
  },
];

const AUTOPLAY_MS     = 9000;
const RESUME_AFTER_MS = 12000;

export default function PipelineDemo() {
  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);
  const resumeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (paused) return;
    const t = setInterval(() => setActive((a) => (a + 1) % STEPS.length), AUTOPLAY_MS);
    return () => clearInterval(t);
  }, [paused]);

  useEffect(() => () => { if (resumeTimer.current) clearTimeout(resumeTimer.current); }, []);

  function selectStep(i: number) {
    setActive(i);
    setPaused(true);
    if (resumeTimer.current) clearTimeout(resumeTimer.current);
    resumeTimer.current = setTimeout(() => {
      resumeTimer.current = null;
      setPaused(false);
    }, RESUME_AFTER_MS);
  }

  const step = STEPS[active];

  const renderSlot = useCallback(
    (key: SlotKey, live: boolean, scale: number, settled: boolean) => {
      const rect = SLOT_RECTS[key];
      switch (key) {
        case "chat":
          return <ChatSlot live={live} />;
        case "global":
          return <GlobalSlot live={live} settled={settled} scale={scale} rect={rect} />;
        case "detail":
          return <DetailSlot live={live} tab={step.detailTab ?? "detail"} />;
        case "semantic":
          return <SemanticSlot />;
        case "personal":
          return (
            <PersonalSlot
              live={live}
              settled={settled}
              scale={scale}
              rect={rect}
              showConfirm={!!step.showConfirm}
            />
          );
      }
    },
    [step],
  );

  return (
    <div
      className="guide-pipeline"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => { if (!resumeTimer.current) setPaused(false); }}
    >
      <DashboardStage
        focusSlot={step.slot}
        stepIndex={active}
        stepCount={STEPS.length}
        stepLabel={`${step.glyph} ${step.title}`}
        renderSlot={renderSlot}
      />

      <div className="guide-pipeline-tabs" role="tablist" aria-label="操作六步驟">
        {STEPS.map((s, i) => (
          <button
            key={s.key}
            type="button"
            role="tab"
            aria-selected={active === i}
            className={`guide-pipeline-tab${active === i ? " active" : ""}`}
            onClick={() => selectStep(i)}
          >
            <span className="guide-pipeline-tab-glyph">{s.glyph}</span>
            <span className="guide-pipeline-tab-title">{s.title}</span>
          </button>
        ))}
      </div>

      <div className="guide-pipeline-copy">
        <span className="gds-where">
          <span className="gds-where-dot" />
          {step.where}
        </span>
        <p className="guide-pipeline-instruction">{step.instruction}</p>
      </div>
    </div>
  );
}
