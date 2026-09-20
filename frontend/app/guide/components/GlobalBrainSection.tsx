"use client";

/**
 * GlobalBrainSection.tsx — Section 2:「探索 GlobalBrain，把概念收進來」。
 *
 * 改版重點：多了第三欄。
 *
 *   [ GlobalBrain 3D ]  →  [ Detail Panel ]  →  [ 語意地圖 ]
 *        點一個節點            讀它、按加入         看它落在哪
 *
 * 舊版只有前兩欄，「加入 Personal Brain」按下去之後什麼也沒發生——
 * 使用者讀得到動作，讀不到後果。補上第三欄並讓概念真的飛過去，
 * 這一節要講的因果才完整：你在左邊挑的東西，會改變右邊那張地圖的重心。
 *
 * 三欄都是真元件（GlobalBrainExplorer / DetailPanel / BertopicMap），
 * 餵 guideData 假資料；personalBrain 是這一節自己的 local state，
 * 從空的開始長，所以「BIAS」那條比例列會隨著加入而長出來。
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import "../guide-flow.css";
import type { GlobalLayoutNode } from "../../components/GlobalBrainExplorer";
import type { ConceptDetail, FoundationTreeData, PersonalBrainConcept } from "../../components/DetailPanel";
import { makeBertopicProjector } from "../bertopicProject";
import FoundationFlight, { type Flight } from "./FoundationFlight";
import {
  GUIDE_LAYOUT,
  GUIDE_IDS,
  GUIDE_RC_SEMANTIC_SCORES,
  GUIDE_RC_FOUNDATION_MAP,
  GUIDE_BERTOPIC_LAYOUT,
  guideFetchConceptDetail,
  guideFetchFoundationTree,
} from "../guideData";

const GlobalBrainExplorer = dynamic(() => import("../../components/GlobalBrainExplorer"), { ssr: false });
const DetailPanel         = dynamic(() => import("../../components/DetailPanel"),         { ssr: false });
const BertopicMap         = dynamic(() => import("../../components/BertopicMap"),         { ssr: false });

const MAP_H = 420;

/** eid → 它在語意地圖上的流派顏色，飛行時的 chip 用同一個顏色，
 *  落點才讀得出「它屬於那一群」。 */
const COLOR_BY_EID = new Map(
  GUIDE_BERTOPIC_LAYOUT.nodes.filter((n) => n.eid).map((n) => [n.eid!, n.color]),
);

type Ripple = { id: string; x: number; y: number; color: string };

export default function GlobalBrainSection() {
  const [selectedNode, setSelectedNode]     = useState<GlobalLayoutNode | null>(null);
  const [detail, setDetail]                 = useState<ConceptDetail | null>(null);
  const [foundationTree, setFoundationTree] = useState<FoundationTreeData | null>(null);
  const [loadingTree, setLoadingTree]       = useState(false);

  // 這一節自己的 Personal Brain：從空的開始，讓使用者看它長出來
  const [brain, setBrain] = useState<PersonalBrainConcept[]>([]);

  const [flights, setFlights] = useState<Flight[]>([]);
  const [ripples, setRipples] = useState<Ripple[]>([]);

  const clickPtRef  = useRef<{ x: number; y: number } | null>(null);
  const mapWrapRef  = useRef<HTMLDivElement | null>(null);
  const fieldWrapRef = useRef<HTMLDivElement | null>(null);
  const [mapW, setMapW]     = useState(360);
  const [fieldW, setFieldW] = useState(360);

  // ── 量測語意地圖的寬度（BertopicMap 需要具體數字）──────────
  useLayoutEffect(() => {
    const el = mapWrapRef.current;
    if (!el) return;
    const apply = () => setMapW(Math.max(Math.floor(el.getBoundingClientRect().width), 240));
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useLayoutEffect(() => {
    const el = fieldWrapRef.current;
    if (!el) return;
    const apply = () => setFieldW(Math.max(Math.floor(el.getBoundingClientRect().width), 240));
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const project = useMemo(
    () => makeBertopicProjector(GUIDE_BERTOPIC_LAYOUT, mapW, MAP_H),
    [mapW],
  );

  const handleNodeClick = useCallback(async (node: GlobalLayoutNode) => {
    setSelectedNode(node);
    setLoadingTree(true);
    const [d, t] = await Promise.all([
      guideFetchConceptDetail(node.id),
      guideFetchFoundationTree(node.id),
    ]);
    setDetail(d);
    setFoundationTree(t);
    setLoadingTree(false);
  }, []);

  // ── 加入 → 起飛 ───────────────────────────────────────────
  const handleAddFoundations = useCallback((items: { eid: string; label: string }[]) => {
    const wrap = mapWrapRef.current;
    const origin = clickPtRef.current;
    if (!wrap || !origin) return;

    const mapRect = wrap.getBoundingClientRect();
    const fresh = items.filter((it) => !brain.some((c) => c.eid === it.eid));
    if (fresh.length === 0) return;

    const next: Flight[] = [];
    for (const it of fresh) {
      const p = project(it.eid);
      if (!p) continue;                     // 不在語意地圖上就不演飛行
      next.push({
        id:    `${it.eid}-${Date.now()}`,
        eid:   it.eid,
        label: it.label,
        color: COLOR_BY_EID.get(it.eid) ?? "#00d4aa",
        from:  origin,
        to:    { x: mapRect.left + p.x, y: mapRect.top + p.y },
      });
    }
    if (next.length === 0) return;
    setFlights(next);
  }, [brain, project]);

  const handleLand = useCallback((f: Flight) => {
    setBrain((prev) => {
      if (prev.some((c) => c.eid === f.eid)) return prev;
      return [...prev, {
        eid: f.eid, label: f.label, source: "manual", weight: 0.6, addedAt: Date.now(),
      }];
    });
    const wrap = mapWrapRef.current;
    if (wrap) {
      const r = wrap.getBoundingClientRect();
      setRipples((prev) => [...prev, {
        id: f.id, x: f.to.x - r.left, y: f.to.y - r.top, color: f.color,
      }]);
    }
  }, []);

  const handleFlightsDone = useCallback(() => setFlights([]), []);

  // 漣漪播完就清掉，避免無限累積
  useEffect(() => {
    if (ripples.length === 0) return;
    const t = setTimeout(() => setRipples([]), 900);
    return () => clearTimeout(t);
  }, [ripples]);

  const reset = useCallback(() => {
    setBrain([]);
    setFlights([]);
    setRipples([]);
  }, []);

  return (
    <div className="guide-flow">
      <div className="guide-flow-grid">

        {/* ① 探索 */}
        <div className="guide-flow-col">
          <div className="guide-flow-step">① 在知識場域裡點一個概念</div>
          <div className="guide-3d-frame" ref={fieldWrapRef} style={{ height: MAP_H }}>
            <GlobalBrainExplorer
              layout={GUIDE_LAYOUT}
              activeCandidateEids={[GUIDE_IDS.rcInflation, GUIDE_IDS.rcRateShock]}
              rcFoundationMap={GUIDE_RC_FOUNDATION_MAP}
              onNodeClick={handleNodeClick}
              width={fieldW}
              height={MAP_H}
            />
          </div>
          <ul className="guide-legend-notes">
            <li>核心 article：直覺與情緒反應，例如「風險趨避」。</li>
            <li>教科書 chapter：打底知識，例如「總體經濟學原理」。</li>
            <li>論文 paper：你讀過、蒐集到的具體文獻。</li>
            <li>洋紅標示環：目前語意上最相關的候選 RC。</li>
          </ul>
        </div>

        {/* ② 閱讀並加入 —— onPointerDownCapture 記下按下的位置，當作飛行起點 */}
        <div
          className="guide-flow-col"
          onPointerDownCapture={(e) => { clickPtRef.current = { x: e.clientX, y: e.clientY }; }}
        >
          <div className="guide-flow-step">② 讀它，把支撐它的概念加進來</div>
          <div className="guide-explore-detail" style={{ height: MAP_H }}>
            <DetailPanel
              node={selectedNode}
              detail={detail}
              foundationTree={foundationTree}
              loadingTree={loadingTree}
              personalBrain={brain}
              rcSemanticScores={GUIDE_RC_SEMANTIC_SCORES}
              rcFoundationMap={GUIDE_RC_FOUNDATION_MAP}
              alpha={0.5}
              fetchConceptDetail={guideFetchConceptDetail}
              onAddFoundations={handleAddFoundations}
              onRemove={(eid) => setBrain((p) => p.filter((c) => c.eid !== eid))}
              onUpdateWeight={(eid, weight) =>
                setBrain((p) => p.map((c) => (c.eid === eid ? { ...c, weight } : c)))}
              onAskAboutNode={() => {}}
            />
          </div>
        </div>

        {/* ③ 落地 */}
        <div className="guide-flow-col">
          <div className="guide-flow-step">③ 它落在語意地圖的這裡</div>
          <div className="guide-3d-frame guide-map-wrap" ref={mapWrapRef} style={{ height: MAP_H }}>
            <BertopicMap
              data={GUIDE_BERTOPIC_LAYOUT}
              personalBrain={brain}
              width={mapW}
              height={MAP_H}
            />
            {ripples.map((r) => (
              <span
                key={r.id}
                className="guide-map-ripple"
                style={{ left: r.x, top: r.y, borderColor: r.color }}
              />
            ))}
            {brain.length === 0 && (
              <div className="guide-map-empty">還沒有任何收藏。到左邊挑一個概念加進來。</div>
            )}
          </div>

          <div className="guide-flow-footer">
            <span className="guide-flow-count">{brain.length} 個概念在 Personal Brain</span>
            {brain.length > 0 && (
              <button type="button" className="guide-flow-reset" onClick={reset}>
                重來一次
              </button>
            )}
          </div>

          {brain.length >= 3 && (
            <p className="guide-flow-payoff guide-fade-in">
              收藏集中在同一個流派，底部的比例列就會偏向那一邊。之後的 RC 排名參考的正是這個偏向。
            </p>
          )}
        </div>
      </div>

      <FoundationFlight flights={flights} onLand={handleLand} onDone={handleFlightsDone} />
    </div>
  );
}
