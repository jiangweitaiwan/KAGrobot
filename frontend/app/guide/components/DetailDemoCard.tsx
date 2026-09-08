"use client";

/**
 * DetailDemoCard.tsx — Section 1 步驟③「閱讀」的輕量仿製卡片。
 *
 * 呈現真實 DetailPanel 的縮影：定義／機制 + foundation 縮圖列 +
 * 加入按鈕，套用 globals.css 的 .detail-panel 系列 class，純靜態，
 * 不含 FoundationTreeViz 的 D3 互動（那個留給 Section 2 的真元件）。
 */

import type { ConceptDetail, FoundationTreeData } from "../../components/DetailPanel";

type Props = {
  detail: ConceptDetail;
  foundationTree: FoundationTreeData;
};

export default function DetailDemoCard({ detail, foundationTree }: Props) {
  return (
    <div className="guide-demo-frame">
      <div className="guide-demo-label">對應 dashboard 區塊：Detail Panel</div>
      <div className="detail-panel guide-detail-demo">
        <div className="detail-header">
          <div className="detail-title-row">
            <span className="detail-label">Concept</span>
            <span className="detail-concept-name">{detail.rc_name}</span>
          </div>
        </div>

        {detail.definition && (
          <div>
            <div className="detail-section-label">Definition</div>
            <p style={{ margin: 0, fontSize: "0.72rem", color: "#9090c0", lineHeight: 1.6 }}>
              {detail.definition}
            </p>
          </div>
        )}

        {detail.mechanism && (
          <div>
            <div className="detail-section-label">Mechanism</div>
            <p style={{ margin: 0, fontSize: "0.72rem", color: "#9090c0", lineHeight: 1.6 }}>
              {detail.mechanism}
            </p>
          </div>
        )}

        <div>
          <div className="detail-section-label">
            Foundation Tree
            <span style={{ color: "var(--muted)", marginLeft: 6, fontWeight: 400 }}>
              · 點選節點加入
            </span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {foundationTree.foundations.map((f) => (
              <span key={f.element_id} className="guide-chip">
                ◆ {f.label}
              </span>
            ))}
          </div>
        </div>

        <button className="detail-btn-add" disabled style={{ cursor: "default", opacity: 0.85 }}>
          + 加入 {foundationTree.foundations.length} 個 foundation 到 Personal Brain
        </button>
      </div>
    </div>
  );
}
