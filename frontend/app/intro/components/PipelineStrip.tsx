"use client";

/**
 * PipelineStrip.tsx — 底部「從感受到行動」的五步驟流程圖。
 *
 * 對照真實系統的五個階段：聊天釐清需求 → 慢想探索 → 推薦RC浮現 →
 * 選擇RC Skill → MainAgent執行，各自對應上方分頁的敘事詞。
 * 點擊任一步驟會捲動回分頁區塊，並跳到對應的分頁。
 */

export type PipelineStage = {
  title: string;
  narrative: string;
  tabIndex: number;
};

export const PIPELINE_STAGES: PipelineStage[] = [
  { title: "聊天釐清需求", narrative: "感受", tabIndex: 0 },
  { title: "慢想探索",     narrative: "知識", tabIndex: 1 },
  { title: "推薦RC浮現",   narrative: "信念", tabIndex: 2 },
  { title: "選擇RC Skill", narrative: "行動", tabIndex: 3 },
  { title: "MainAgent執行", narrative: "行動", tabIndex: 3 },
];

type Props = {
  onSelectStage: (tabIndex: number) => void;
};

export default function PipelineStrip({ onSelectStage }: Props) {
  return (
    <div className="intro-pipeline">
      {PIPELINE_STAGES.map((stage, i) => (
        <div className="intro-pipeline-item" key={stage.title}>
          <button
            type="button"
            className="intro-pipeline-node"
            onClick={() => onSelectStage(stage.tabIndex)}
          >
            <span className="intro-pipeline-title">{stage.title}</span>
            <span className="intro-pipeline-narrative">{stage.narrative}</span>
          </button>
          {i < PIPELINE_STAGES.length - 1 && <span className="intro-pipeline-rule" aria-hidden="true" />}
        </div>
      ))}
    </div>
  );
}
