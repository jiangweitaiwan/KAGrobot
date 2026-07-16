"use client";

import type { RCSemanticScore } from "./DetailPanel";

export type SemanticContextData = {
  query:        string;
  intent_types: string[];
  summary:      string;
};

export type RecommendedFoundation = {
  element_id: string;
  label:      string;
  score:      number;
};

type Props = {
  context:                SemanticContextData | null;
  rcScores:               RCSemanticScore[];   // 已按 score 由高到低排序
  recommendedFoundations: RecommendedFoundation[];
  onAddFoundation:        (eid: string, label: string) => void;
};

const INTENT_LABELS: Record<string, string> = {
  causal_mechanism:      "因果機制",
  equilibrium_analysis:  "均衡分析",
  forecasting:           "預測",
  event_study:           "事件研究",
  structural_estimation: "結構估計",
  policy_evaluation:     "政策評估",
  asset_pricing:         "資產定價",
  descriptive:           "描述性分析",
};

function RecommendedFoundationsRow({
  items, onAdd,
}: {
  items: RecommendedFoundation[];
  onAdd: (eid: string, label: string) => void;
}) {
  if (!items.length) return null;
  return (
    <div style={{ flexShrink: 0 }}>
      <div className="detail-section-label" style={{ marginBottom: 4 }}>
        語意推薦 Foundation
        <span style={{ color: "var(--muted)", marginLeft: 6, fontWeight: 400 }}>
          · 依對話語意匹配 · 點擊加入
        </span>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {items.map((f) => (
          <span
            key={f.element_id}
            style={{
              fontSize: "0.62rem", padding: "2px 8px", borderRadius: 12,
              border:     "1px solid rgba(0,212,170,0.35)",
              background: "rgba(0,212,170,0.06)",
              color:      "var(--teal)",
              cursor:     "pointer",
            }}
            title={`語意分數 ${f.score.toFixed(2)}`}
            onClick={() => onAdd(f.element_id, f.label)}
          >
            + {f.label}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function SemanticPanel({ context, rcScores, recommendedFoundations, onAddFoundation }: Props) {
  const topScores = rcScores.slice(0, 5);
  const maxScore  = Math.max(0.01, ...topScores.map((s) => s.score));

  const hasContext =
    context && (context.query || context.summary || context.intent_types.length > 0);

  return (
    <div className="detail-panel semantic-panel" style={{ gap: 10 }}>
      <div className="detail-label" style={{ flexShrink: 0 }}>Semantic Panel</div>

      {/* ── 上半部：對話語意脈絡 ── */}
      {!hasContext ? (
        <div className="detail-empty">
          <span>對話開始後</span>
          <span>語意上下文將顯示於此</span>
        </div>
      ) : (
        <>
          {context!.intent_types.length > 0 && (
            <div style={{ flexShrink: 0 }}>
              <div className="detail-section-label" style={{ marginBottom: 4 }}>Intent</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {context!.intent_types.map((t) => (
                  <span
                    key={t}
                    style={{
                      fontSize: "0.62rem", padding: "2px 8px", borderRadius: 12,
                      border:     "1px solid rgba(232,160,48,0.35)",
                      background: "rgba(232,160,48,0.08)",
                      color:      "var(--orange)",
                    }}
                  >
                    {INTENT_LABELS[t] ?? t}
                  </span>
                ))}
              </div>
            </div>
          )}

          {context!.summary && (
            <div style={{ flexShrink: 0 }}>
              <div className="detail-section-label" style={{ marginBottom: 4 }}>Summary</div>
              <p style={{ margin: 0, fontSize: "0.68rem", color: "#9090c0", lineHeight: 1.6 }}>
                {context!.summary}
              </p>
            </div>
          )}

          {context!.query && (
            <div style={{ flexShrink: 0 }}>
              <div className="detail-section-label" style={{ marginBottom: 4 }}>Query</div>
              <p style={{
                margin: 0, fontSize: "0.66rem", color: "var(--muted)",
                lineHeight: 1.5, fontStyle: "italic",
              }}>
                “{context!.query}”
              </p>
            </div>
          )}

          {topScores.length > 0 && (
            <div style={{ flexShrink: 0 }}>
              <div className="detail-section-label" style={{ marginBottom: 6 }}>
                RC 匹配度
                <span style={{ color: "var(--muted)", marginLeft: 6, fontWeight: 400 }}>
                  · 紫=embed 橘=intent
                </span>
              </div>
              {topScores.map((s, i) => {
                const embedW  = maxScore > 0 ? ((s.embed_score  ?? 0) / maxScore) * 100 : 0;
                const intentW = maxScore > 0 ? ((s.intent_score ?? 0) / maxScore) * 100 : 0;
                return (
                  <div key={s.eid} style={{ marginBottom: 6 }}>
                    <div style={{
                      display: "flex", justifyContent: "space-between",
                      fontSize: "0.62rem", marginBottom: 2,
                    }}>
                      <span style={{
                        color: "#9090b0", overflow: "hidden", textOverflow: "ellipsis",
                        whiteSpace: "nowrap", maxWidth: 150,
                      }} title={s.label}>
                        {i + 1}. {s.label}
                      </span>
                      <span style={{ color: "#afa9ec" }}>{s.score.toFixed(2)}</span>
                    </div>
                    <div style={{
                      display: "flex", height: 6, borderRadius: 3, overflow: "hidden",
                      background: "rgba(83,74,183,0.1)",
                    }}>
                      <div style={{ width: `${embedW}%`,  background: "var(--purple)" }} />
                      <div style={{ width: `${intentW}%`, background: "var(--orange)" }} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}

      {/* ── 下半部：操作區（語意推薦 Foundation → 加入 Personal Brain）── */}
      <RecommendedFoundationsRow items={recommendedFoundations} onAdd={onAddFoundation} />
    </div>
  );
}