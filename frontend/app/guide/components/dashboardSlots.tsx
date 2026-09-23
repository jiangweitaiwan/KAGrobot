"use client";

/**
 * dashboardSlots.tsx — Section 1 虛擬 dashboard 上五個欄位的內容。
 *
 * 每個欄位有兩級：
 *   - 非作用中（live=false）：只畫骨架，讓全景圖看起來像 dashboard，
 *     但不掛任何 3D / 狀態機。這不只是效能考量——瀏覽器對同時存在的
 *     WebGL context 有數量上限，而 guide 整頁 Section 2、3 還各要用掉
 *     一個，這裡必須節制。
 *   - 作用中（live=true）：掛上真元件或高保真仿製卡片。
 *
 * 3D 元件另外要求「相機停穩後才掛」（settled，現在跟鏡頭無關，純粹
 * 是切欄位後的短暫延遲），並且以最終螢幕像素尺寸 render、再反向
 * scale 回畫布座標——否則 canvas 被 CSS 縮放會糊掉，DOM 元素則不會
 * 有這個問題。
 *
 * 改版重點（Step 2 / Step 3）：
 *   - ChatSlot：從「1問1答」改成 3 輪來回＋打字指示動畫，對話記錄
 *     區改可捲動並自動貼底，示範真實對話的節奏感。
 *   - GlobalSlot：設定停留一段時間後，透過 GlobalBrainExplorer 新增
 *     的 forceFocusEid prop「模擬點擊」聚焦候選節點——這條路徑跟
 *     使用者真的點擊完全共用同一個 state，所以光暈／聚焦路徑高亮
 *     都是真實效果，不是另外疊一層假動畫。
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import dynamic from "next/dynamic";
import type { Rect } from "../dashboardMap";
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

// ─────────────────────────────────────────────────────────────
// CrispScale — 讓 canvas 類元件以最終螢幕像素尺寸繪製
// ─────────────────────────────────────────────────────────────

function CrispScale({
  scale, width, height, children,
}: {
  scale:  number;
  width:  number;
  height: number;
  children: (pxW: number, pxH: number) => ReactNode;
}) {
  const s = Math.max(scale, 1);
  const pxW = Math.round(width * s);
  const pxH = Math.round(height * s);
  return (
    <div style={{ width, height, overflow: "hidden" }}>
      <div style={{ width: pxW, height: pxH, transform: `scale(${1 / s})`, transformOrigin: "0 0" }}>
        {children(pxW, pxH)}
      </div>
    </div>
  );
}

function Skeleton({ lines = 4 }: { lines?: number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 7, opacity: 0.7 }}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="gds-bar"
          style={{ height: 7, width: `${92 - (i % 3) * 18}%` }}
        />
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Header
// ─────────────────────────────────────────────────────────────

export function BoardHeader({ rect }: { rect: Rect }) {
  return (
    <div
      className="gds-header"
      style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}
    >
      <div>
        <div className="gds-header-title">ZenMind</div>
        <div className="gds-header-meta">socratic · 4 in brain</div>
      </div>
      <div className="gds-header-chips">
        <span className="gds-header-chip">慢想模式</span>
        <span className="gds-header-chip">歷史對話</span>
        <span className="gds-header-chip">開新對話</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Chat — 3 輪來回，AI 訊息前有打字指示，對話記錄可捲動並自動貼底
// ─────────────────────────────────────────────────────────────

type ChatTurn = { role: "user" | "ai"; text: string };

const CHAT_TURNS: ChatTurn[] = [
  { role: "user", text: GUIDE_QUESTION },
  { role: "ai",   text: "這牽涉到「價格為什麼不會馬上調整」。我先到知識場域裡找相關概念。" },
  { role: "user", text: "所以廠商不是想漲就漲、想跌就跌？" },
  { role: "ai",   text: "對，關鍵在「調價本身有成本」。我找到幾個概念可以說明這件事，你可以點進去看。" },
  { role: "user", text: "有具體的模型可以參考嗎？" },
  { role: "ai",   text: "有，Calvo 定價模型是這類理論的經典之一，我一併整理進候選概念。" },
];

export function ChatSlot({ live }: { live: boolean }) {
  const [shown, setShown]   = useState(0);      // 已顯示到第幾則（CHAT_TURNS 的長度）
  const [typing, setTyping] = useState(false);  // 是否顯示「對方正在輸入…」
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!live) { setShown(0); setTyping(false); return; }

    const timers: ReturnType<typeof setTimeout>[] = [];
    let t = 0;

    setShown(1); // 使用者的提問立刻出現，不用等

    for (let i = 1; i < CHAT_TURNS.length; i++) {
      const turn = CHAT_TURNS[i];
      if (turn.role === "ai") {
        t += 500;
        timers.push(setTimeout(() => setTyping(true), t));
        t += 900;
        const idx = i;
        timers.push(setTimeout(() => { setTyping(false); setShown(idx + 1); }, t));
      } else {
        t += 900;
        const idx = i;
        timers.push(setTimeout(() => setShown(idx + 1), t));
      }
    }

    return () => timers.forEach(clearTimeout);
  }, [live]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [shown, typing]);

  return (
    <div className="gds-slot-body">
      <div className="gds-chat-log" ref={logRef}>
        {live ? (
          <>
            {CHAT_TURNS.slice(0, shown).map((m, i) => (
              <div
                key={i}
                className={`gds-msg ${m.role === "user" ? "gds-msg-user" : "gds-msg-ai"} guide-fade-in`}
              >
                {m.text}
              </div>
            ))}
            {typing && (
              <div className="gds-msg gds-msg-ai gds-msg-typing guide-fade-in">
                <span className="gds-typing-dot" />
                <span className="gds-typing-dot" />
                <span className="gds-typing-dot" />
              </div>
            )}
          </>
        ) : (
          <>
            <div className="gds-bar" style={{ height: 22, width: "72%", alignSelf: "flex-end" }} />
            <div className="gds-bar" style={{ height: 32, width: "86%" }} />
            <div className="gds-bar" style={{ height: 22, width: "64%", alignSelf: "flex-end" }} />
            <div className="gds-bar" style={{ height: 28, width: "78%" }} />
          </>
        )}
      </div>
      <div className="gds-composer">
        <span className="gds-composer-text">輸入你的問題…</span>
        <span className="gds-composer-send">送出</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Global Knowledge Field — 停留一段時間後模擬點擊聚焦「價格僵固性」
// ─────────────────────────────────────────────────────────────

const PICK_DELAY_MS = 900;

export function GlobalSlot({
  live, settled, scale, rect,
}: {
  live: boolean; settled: boolean; scale: number; rect: Rect;
}) {
  const w = rect.w - 24;
  const h = rect.h - 42;

  const [picked, setPicked] = useState(false);

  useEffect(() => {
    if (!live || !settled) { setPicked(false); return; }
    const t = setTimeout(() => setPicked(true), PICK_DELAY_MS);
    return () => clearTimeout(t);
  }, [live, settled]);

  if (!live || !settled) {
    return (
      <div className="gds-slot-body">
        <div
          className="gds-block"
          style={{
            flex: 1,
            display: "grid",
            placeItems: "center",
            background:
              "radial-gradient(52% 32% at 50% 44%, rgba(83,74,183,0.22) 0%, transparent 72%)",
          }}
        >
          <svg viewBox="0 0 120 120" style={{ width: "72%", opacity: live ? 0.5 : 0.32 }}>
            <circle cx="60" cy="60" r="44" fill="none" stroke="rgba(127,119,221,0.22)" />
            <ellipse cx="60" cy="60" rx="44" ry="15" fill="none" stroke="rgba(127,119,221,0.14)" />
            {[
              [38, 44], [74, 38], [52, 70], [86, 66], [30, 78],
              [62, 30], [68, 90], [44, 92], [90, 50],
            ].map(([cx, cy], i) => (
              <circle key={i} cx={cx} cy={cy} r={i % 3 === 0 ? 2.6 : 1.6} fill="#7f77dd" opacity={0.55} />
            ))}
            <circle cx="60" cy="58" r="4" fill="#c060ff" opacity="0.8" />
          </svg>
        </div>
      </div>
    );
  }

  return (
    <div className="gds-slot-body" style={{ position: "relative" }}>
      <CrispScale scale={scale} width={w} height={h}>
        {(pxW, pxH) => (
          <GlobalBrainExplorer
            layout={GUIDE_LAYOUT_MINI}
            activeCandidateEids={[GUIDE_IDS.priceStick]}
            forceFocusEid={picked ? GUIDE_IDS.priceStick : null}
            width={pxW}
            height={pxH}
          />
        )}
      </CrispScale>
      {picked && (
        <div className="gds-picked-badge guide-fade-in">
          已選取「價格僵固性」
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Detail Panel（步驟③「詳情」與步驟④「重心」共用同一格）
// ─────────────────────────────────────────────────────────────

const BRAIN_ITEMS = GUIDE_PERSONAL_BRAIN.map((c) => ({ eid: c.eid, label: c.label, weight: c.weight }));

export function DetailSlot({
  live, tab,
}: {
  live: boolean; tab: "detail" | "brain";
}) {
  const [added, setAdded] = useState(0);

  useEffect(() => {
    if (!live || tab !== "brain") { setAdded(0); return; }
    const timers = BRAIN_ITEMS.map((_, i) =>
      setTimeout(() => setAdded((n) => Math.max(n, i + 1)), 320 + i * 420),
    );
    return () => timers.forEach(clearTimeout);
  }, [live, tab]);

  if (!live) {
    return (
      <div className="gds-slot-body">
        <div className="gds-bar" style={{ height: 11, width: "58%" }} />
        <Skeleton lines={5} />
        <div className="gds-block" style={{ height: 54 }} />
        <Skeleton lines={3} />
      </div>
    );
  }

  return (
    <div className="gds-slot-body">
      <div className="gds-tabs">
        <span className="gds-tab" data-on={tab === "detail"}>詳情</span>
        <span className="gds-tab" data-on={tab === "brain"}>重心</span>
      </div>

      {tab === "detail" ? (
        <>
          <div className="gds-detail-name">{GUIDE_CONCEPT_DETAIL.rc_name}</div>
          <div>
            <div className="gds-field-label">定義</div>
            <p className="gds-field-text">{GUIDE_CONCEPT_DETAIL.definition}</p>
          </div>
          <div>
            <div className="gds-field-label">機制</div>
            <p className="gds-field-text">{GUIDE_CONCEPT_DETAIL.mechanism}</p>
          </div>
          <div>
            <div className="gds-field-label">支撐它的概念</div>
            <div className="gds-chiprow">
              {GUIDE_FOUNDATION_TREE.foundations.map((f) => (
                <span key={f.element_id} className="gds-chip">◆ {f.label}</span>
              ))}
            </div>
          </div>
          <div className="gds-addbtn">
            加入 {GUIDE_FOUNDATION_TREE.foundations.length} 個概念到 Personal Brain
          </div>
        </>
      ) : (
        <>
          <div className="gds-field-label">知識重心 · {added} 個</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 9, marginTop: 4 }}>
            {BRAIN_ITEMS.map((c, i) => {
              const on = i < added;
              return (
                <div
                  key={c.eid}
                  className="gds-weight-row"
                  style={{
                    opacity: on ? 1 : 0.18,
                    transform: on ? "translateY(0)" : "translateY(-4px)",
                    transition: "opacity 380ms ease, transform 380ms ease",
                  }}
                >
                  <span style={{ width: 88, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {c.label}
                  </span>
                  <span className="gds-weight-track">
                    <span className="gds-weight-fill" style={{ width: on ? `${c.weight * 100}%` : "0%" }} />
                  </span>
                  <span style={{ opacity: 0.6, width: 24, textAlign: "right" }}>
                    {c.weight.toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="gds-field-text" style={{ marginTop: "auto", opacity: 0.8 }}>
            權重決定這些概念在後續排名裡的份量，隨時可以拉動調整。
          </p>
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Semantic（六步驟不經過，永遠只畫骨架）
// ─────────────────────────────────────────────────────────────

export function SemanticSlot() {
  return (
    <div className="gds-slot-body">
      <Skeleton lines={3} />
      <div className="gds-block" style={{ height: 40 }} />
      <Skeleton lines={4} />
      <div className="gds-block" style={{ height: 40 }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Personal Brain（步驟⑤剖面、步驟⑥送出共用同一格）
// ─────────────────────────────────────────────────────────────

const RC_TAGS = [
  { label: "通膨黏性價格假說", score: 83 },
  { label: "升息衝擊分析",     score: 55 },
];

export function PersonalSlot({
  live, settled, scale, rect, showConfirm,
}: {
  live: boolean; settled: boolean; scale: number; rect: Rect; showConfirm: boolean;
}) {
  const w = rect.w - 24;
  const h = rect.h - 96;

  return (
    <div className="gds-slot-body" style={{ position: "relative" }}>
      <div className="gds-rc-tags">
        {RC_TAGS.map((t, i) => (
          <span key={t.label} className="gds-rc-tag" data-on={live && i === (showConfirm ? 1 : 0)}>
            {i + 1}. {t.label} <span style={{ opacity: 0.6 }}>{t.score}</span>
          </span>
        ))}
      </div>

      {live && settled ? (
        <CrispScale scale={scale} width={w} height={h}>
          {(pxW, pxH) => (
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
              selectedSkillEid={showConfirm ? GUIDE_IDS.rcRateShock : null}
              onSelectSkill={() => {}}
              onFocusNode={() => {}}
              width={pxW}
              height={pxH}
            />
          )}
        </CrispScale>
      ) : (
        <div style={{ width: w, height: h, display: "grid", placeItems: "center" }}>
          <svg viewBox="0 0 100 150" style={{ height: "82%", opacity: live ? 0.45 : 0.3 }}>
            <path d="M14 8 L86 8 L54 75 L86 142 L14 142 L46 75 Z"
                  fill="rgba(127,119,221,0.07)" stroke="rgba(127,119,221,0.28)" />
            <circle cx="50" cy="75" r="4" fill="#00d4aa" opacity="0.75" />
            {[[30, 26], [66, 30], [24, 48], [74, 52], [34, 112], [68, 118], [50, 132]].map(([cx, cy], i) => (
              <circle key={i} cx={cx} cy={cy} r="2" fill="#7f77dd" opacity="0.5" />
            ))}
          </svg>
        </div>
      )}

      {live && showConfirm && (
        <div
          className="guide-fade-in"
          style={{ position: "absolute", left: 10, right: 10, bottom: 14 }}
        >
          <div className="gds-confirm">
            <span className="gds-confirm-label">⚡ 執行「升息衝擊分析」</span>
            <span className="gds-confirm-ghost">取消</span>
            <span className="gds-confirm-go">🚀 送出</span>
          </div>
        </div>
      )}
    </div>
  );
}
