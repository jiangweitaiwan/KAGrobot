"use client";

/**
 * BrainAddDemo.tsx — Section 1 步驟④「收藏」的輕量仿製卡片。
 *
 * 播放一次「陸續加入 → 標籤飛入重心列表」的小動畫，呼應真實
 * PersonalBrain / DetailPanel「重心」分頁的加入互動，不含真實
 * 狀態管理，也不會真的送出任何請求。
 */

import { useEffect, useState } from "react";

type Props = {
  concepts: { eid: string; label: string }[];
};

export default function BrainAddDemo({ concepts }: Props) {
  const [addedCount, setAddedCount] = useState(0);

  useEffect(() => {
    setAddedCount(0);
    const timers = concepts.map((_, i) =>
      setTimeout(() => setAddedCount((c) => c + 1), 500 + i * 450),
    );
    return () => timers.forEach(clearTimeout);
  }, [concepts]);

  return (
    <div className="guide-demo-frame">
      <div className="guide-demo-label">對應 dashboard 區塊：Detail Panel「重心」分頁</div>
      <div className="guide-brain-add-demo">
        <div className="detail-section-label">
          知識重心
          <span style={{ color: "var(--muted)", marginLeft: 6, fontWeight: 400 }}>
            · 共 {addedCount} 個
          </span>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, minHeight: 26 }}>
          {concepts.slice(0, addedCount).map((c) => (
            <span key={c.eid} className="guide-chip guide-chip-fly-in">
              {c.label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
