"use client";

/**
 * SkillLaunchDemo.tsx — Section 1 步驟⑥「送出」的輕量仿製卡片。
 *
 * 呈現「⚡ 執行 / 送出」確認列，套用真實 PersonalBrain 送出確認列
 * 的視覺語言，按下後播放一次送出動畫，結束後顯示「已送出」狀態，
 * 銜接 Section 4 的可計算性說明。
 */

import { useState } from "react";

type Props = {
  rcLabel: string;
};

export default function SkillLaunchDemo({ rcLabel }: Props) {
  const [sent, setSent] = useState(false);

  return (
    <div className="guide-demo-frame">
      <div className="guide-demo-label">對應 dashboard 區塊：Personal Brain「送出」確認列</div>
      <div className="guide-skill-launch-demo">
        <div className="guide-skill-pill">⚡ 執行「{rcLabel}」</div>
        <div style={{ display: "flex", gap: 10 }}>
          <button type="button" disabled className="detail-btn-secondary" style={{ opacity: 0.6 }}>
            取消
          </button>
          <button
            type="button"
            className="detail-btn-add"
            onClick={() => setSent(true)}
            disabled={sent}
          >
            {sent ? "已送出 ✓" : "🚀 送出"}
          </button>
        </div>
        {sent && (
          <p className="guide-fade-in" style={{ fontSize: "0.72rem", color: "var(--teal)", margin: 0, textAlign: "center" }}>
            已送入 MainAgent，接著會套用對應的量化模型計算結論（見下方「可計算性」）。
          </p>
        )}
      </div>
    </div>
  );
}
