"use client";

/**
 * PipelineDemo.tsx — Section 1:「完整流程」六步驟示範。
 *
 * 六個步驟對應 dashboard 的六個操作環節:
 *   💬 提問 → 🌌 探索 GlobalBrain → 🔍 閱讀 DetailPanel →
 *   ➕ 收藏進 PersonalBrain → ⏳ 檢視 Hourglass 剖面 → ⚡ 送出 RC Skill
 *
 * 步驟②⑤直接複用 dashboard 的真實 3D 元件（GlobalBrainExplorer /
 * HourglassViz）,餵一份縮小版假資料;其餘四步驟是輕量仿製卡片
 * （ChatDemoCard / DetailDemoCard / BrainAddDemo / SkillLaunchDemo）,
 * 只需要「看起來像」,不需要真的可互動狀態機。
 *
 * 每一步都刻意標註「對應 dashboard 區塊」——這是這個 demo 存在的
 * 目的:讓使用者在看示範時就能把畫面對應到真正要操作的地方。
 */

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import ChatDemoCard from "./ChatDemoCard";
import DetailDemoCard from "./DetailDemoCard";
import BrainAddDemo from "./BrainAddDemo";
import SkillLaunchDemo from "./SkillLaunchDemo";
import {
  GUIDE_QUESTION,
  GUIDE_IDS,
  GUIDE_LAYOUT_MINI,
  GUIDE_CONCEPT_DETAIL,
  GUIDE_FOUNDATION_TREE,
  GUIDE_PERSONAL_BRAIN,
} from "../guideData";

const GlobalBrainExplorer = dynamic(() => import("../../components/GlobalBrainExplorer"), { ssr: false });
const HourglassViz        = dynamic(() => import("../../components/HourglassViz"),        { ssr: false });

type Step = {
  key: string;
  glyph: string;
  title: string;
  dashboardRef: string;
  instruction: string;
};

const STEPS: Step[] = [
  {
    key: "chat",
    glyph: "💬",
    title: "提問",
    dashboardRef: "左側對話欄",
    instruction: "在 chat-box 輸入你關心的問題，系統會依語意去 GlobalBrain 裡找相關概念。",
  },
  {
    key: "explore",
    glyph: "🌌",
    title: "探索",
    dashboardRef: "中央 Global Knowledge Field",
    instruction: "拖曳旋轉、滾輪縮放，點擊任一節點——這裡點的是「價格僵固性」。",
  },
  {
    key: "read",
    glyph: "🔍",
    title: "閱讀",
    dashboardRef: "Detail Panel",
    instruction: "讀這個概念的定義、機制，以及它的 Foundation Tree（支撐它的更基礎概念）。",
  },
  {
    key: "collect",
    glyph: "➕",
    title: "收藏",
    dashboardRef: "Detail Panel「重心」分頁",
    instruction: "把覺得有用的 foundation 加入 Personal Brain，之後排名與推薦都會參考這裡的權重。",
  },
  {
    key: "profile",
    glyph: "⏳",
    title: "剖面",
    dashboardRef: "Personal Brain「沙漏」視圖",
    instruction: "收藏幾個概念後，系統會收斂出推薦 RC——這裡是「通膨黏性價格假說」，往上看它能延伸到「升息衝擊分析」。",
  },
  {
    key: "launch",
    glyph: "⚡",
    title: "送出",
    dashboardRef: "Personal Brain 送出確認列",
    instruction: "選定 RC 後按下閃電送出，交給 MainAgent 執行對應的模型計算。",
  },
];

const AUTOPLAY_MS = 6000;
const RESUME_AFTER_MS = 9000;

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

  return (
    <div
      className="guide-pipeline"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => { if (!resumeTimer.current) setPaused(false); }}
    >
      <div className="guide-pipeline-stage">
        {step.key === "chat" && <ChatDemoCard question={GUIDE_QUESTION} />}

        {step.key === "explore" && (
          <div className="guide-demo-frame">
            <div className="guide-demo-label">對應 dashboard 區塊：{step.dashboardRef}</div>
            <div className="guide-3d-frame">
              <GlobalBrainExplorer
                layout={GUIDE_LAYOUT_MINI}
                activeCandidateEids={[GUIDE_IDS.priceStick]}
                width={520}
                height={320}
              />
            </div>
          </div>
        )}

        {step.key === "read" && (
          <DetailDemoCard detail={GUIDE_CONCEPT_DETAIL} foundationTree={GUIDE_FOUNDATION_TREE} />
        )}

        {step.key === "collect" && (
          <BrainAddDemo concepts={GUIDE_PERSONAL_BRAIN.map((c) => ({ eid: c.eid, label: c.label }))} />
        )}

        {step.key === "profile" && (
          <div className="guide-demo-frame">
            <div className="guide-demo-label">對應 dashboard 區塊：{step.dashboardRef}</div>
            <div className="guide-3d-frame">
              <HourglassViz
                focusedEid={GUIDE_IDS.rcInflation}
                focusedLabel="通膨黏性價格假說"
                foundationTree={GUIDE_FOUNDATION_TREE}
                downstreamChains={[[
                  { eid: GUIDE_IDS.rcInflation, label: "通膨黏性價格假說" },
                  { eid: GUIDE_IDS.rcRateShock, label: "升息衝擊分析" },
                ]]}
                personalBrainEids={new Set(GUIDE_PERSONAL_BRAIN.map((c) => c.eid))}
                bertopicPosByEid={new Map()}
                bertopicCentroid={{ x: 0, y: 0 }}
                skillEids={new Set([GUIDE_IDS.rcRateShock])}
                selectedSkillEid={null}
                onSelectSkill={() => {}}
                onFocusNode={() => {}}
                width={520}
                height={320}
              />
            </div>
          </div>
        )}

        {step.key === "launch" && <SkillLaunchDemo rcLabel="升息衝擊分析" />}
      </div>

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
        <p className="guide-pipeline-instruction">{step.instruction}</p>
      </div>
    </div>
  );
}
