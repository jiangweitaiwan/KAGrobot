"use client";

// RC 全局說明卡片：顯示於聊天氣泡內（比照 SkillWidgetCard 的位置）。
// 樣式沿用 dashboard 既有的 CSS 變數（--purple / --border / --muted / --text / --teal / --orange），
// 不引入新的視覺語言。

export type BriefingLayer = { name: string; node_eids: string[]; role: string };

export type RCBriefingData = {
  problem_frame:    string;
  layers:           BriefingLayer[];
  convergence:      string;
  rc_role:          string;
  contested_points: { eid: string; why: string }[];
  fit_gaps:         string[];
  handoff:          { assumptions: string[]; params_hint: string[] };
  walk_overlap:     string[];
};

export type BriefingState = {
  eid:     string;
  label:   string;
  status:  "loading" | "ready" | "error" | "adopted";
  data?:   RCBriefingData;
  labels?: Record<string, string>;
  error?:  string;
};

const box: React.CSSProperties = {
  marginTop: 8, border: "1px solid rgba(83,74,183,0.4)", borderRadius: 10,
  padding: "10px 12px", display: "flex", flexDirection: "column", gap: 10,
  fontSize: "0.78rem", lineHeight: 1.55, maxWidth: 520,
};
const h: React.CSSProperties = { fontSize: "0.7rem", color: "var(--muted)", marginBottom: 2 };

export default function RCBriefingCard({
  briefing, disabled, onAdopt, onDismiss,
}: {
  briefing: BriefingState;
  disabled: boolean;
  onAdopt: (eid: string, label: string) => void;
  onDismiss: () => void;
}) {
  const { status, data, labels = {}, label, eid } = briefing;
  const name = (id: string) => labels[id] ?? id;

  if (status === "loading") {
    return <div style={box}>正在整理「{label}」的全局脈絡…</div>;
  }
  if (status === "error") {
    return (
      <div style={box}>
        <div>無法產生「{label}」的說明：{briefing.error || "請稍後再試"}</div>
        <div style={{ display: "flex", gap: 8 }}>
          <button type="button" onClick={() => onAdopt(eid, label)} disabled={disabled}>
            略過說明，直接採用
          </button>
          <button type="button" onClick={onDismiss}>關閉</button>
        </div>
      </div>
    );
  }
  if (!data) return null;

  return (
    <div style={box}>
      <div style={{ color: "var(--purple)", fontWeight: 600 }}>{label}：全局說明</div>

      <section>
        <div style={h}>問題脈絡</div>
        <div>{data.problem_frame}</div>
      </section>

      <section>
        <div style={h}>上游結構（由上游到方案）</div>
        {data.layers.map((l, i) => (
          <div key={i} style={{ marginBottom: 6 }}>
            <div><b>{l.name}</b>：{l.role}</div>
            <div style={{ color: "var(--muted)" }}>{l.node_eids.map(name).join("、")}</div>
          </div>
        ))}
        <div>{data.convergence}</div>
      </section>

      <section>
        <div style={h}>方案負責的部分</div>
        <div>{data.rc_role}</div>
      </section>

      <section>
        <div style={h}>需求缺口</div>
        {data.fit_gaps.length === 0
          ? <div style={{ color: "var(--teal)" }}>未發現明顯缺口</div>
          : <ul style={{ margin: 0, paddingLeft: 18 }}>{data.fit_gaps.map((g, i) => <li key={i}>{g}</li>)}</ul>}
      </section>

      {data.contested_points.length > 0 && (
        <section>
          <div style={h}>有爭議的環節</div>
          <ul style={{ margin: 0, paddingLeft: 18, color: "var(--orange)" }}>
            {data.contested_points.map((c, i) => (
              <li key={i}><b>{name(c.eid)}</b>：{c.why}</li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <div style={h}>佐證來源</div>
        <div>
          {data.walk_overlap.length > 0
            ? `慢想探索經過此方案結構內的 ${data.walk_overlap.length} 個節點：${data.walk_overlap.map(name).join("、")}`
            : "僅依語意相似度推薦，慢想路徑沒有經過此方案的結構。"}
        </div>
      </section>

      {status === "adopted" ? (
        <div style={{ color: "var(--teal)" }}>已採用此方案</div>
      ) : (
        <div style={{ display: "flex", gap: 8 }}>
          <button type="button" disabled={disabled} onClick={() => onAdopt(eid, label)}>
            採用此方案
          </button>
          <button type="button" onClick={onDismiss}>看其他方案</button>
        </div>
      )}
    </div>
  );
}
