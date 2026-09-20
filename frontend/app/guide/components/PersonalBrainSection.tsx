"use client";

/**
 * PersonalBrainSection.tsx — Section 3:「同一顆大腦，兩種切法」。
 *
 * 改版重點有三個。
 *
 * 一、切換鈕移到這一節手上。原本直接複用 PersonalBrain 外殼，但它的
 *     沙漏／分佈切換是內部 state，示意圖沒辦法跟著動。這裡改成自己
 *     持有 view，再直接掛真正的 HourglassViz / BertopicMap——dashboard
 *     的元件一行都不用改。
 *
 * 二、錐形連接線常駐。從球體切面兩端收斂到下方檢視框的上緣兩角，
 *     任何時候看都讀得出「下面那塊是上面那塊取出來放大的」。比起
 *     只在切換時播一次動畫，常駐的結構更可靠。
 *
 * 三、腰身是兩邊共用的錨點。在沙漏裡點 downstream 節點重新聚焦時，
 *     球體上那顆腰身點會跟著滑到對應方向——這一下才是「同一個空間的
 *     兩種投影」真正的證據。
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import "../guide-flow.css";
import SliceDiagram from "./SliceDiagram";
import type { FoundationTreeData } from "../../components/DetailPanel";
import type { ChainStep } from "../../components/HourglassViz";
import {
  GUIDE_IDS,
  GUIDE_PERSONAL_BRAIN,
  GUIDE_BERTOPIC_LAYOUT,
  GUIDE_RC_PATHS,
  guideFetchFoundationTree,
} from "../guideData";

const HourglassViz = dynamic(() => import("../../components/HourglassViz"), { ssr: false });
const BertopicMap  = dynamic(() => import("../../components/BertopicMap"),  { ssr: false });

const VIEWER_H = 400;

// ── BERTopic 座標查表（沙漏用它決定節點的方位角）────────────
const BERTOPIC_POS = new Map(
  GUIDE_BERTOPIC_LAYOUT.nodes
    .filter((n) => n.eid)
    .map((n) => [n.eid!, { x: n.x, y: n.y }]),
);

const BERTOPIC_CENTROID = (() => {
  const ns = GUIDE_BERTOPIC_LAYOUT.nodes;
  if (ns.length === 0) return { x: 0, y: 0 };
  return {
    x: ns.reduce((s, n) => s + n.x, 0) / ns.length,
    y: ns.reduce((s, n) => s + n.y, 0) / ns.length,
  };
})();

const BERTOPIC_SPAN = (() => {
  const xs = GUIDE_BERTOPIC_LAYOUT.nodes.map((n) => n.x);
  return Math.max(...xs) - Math.min(...xs) || 1;
})();

const BRAIN_EIDS = new Set(GUIDE_PERSONAL_BRAIN.map((c) => c.eid));
const SKILL_EIDS = new Set([GUIDE_IDS.rcRateShock]);

/** 從 guideData 的假 rc-paths 取出以 eid 為起點的 downstream 鏈。 */
function chainsFor(eid: string): ChainStep[][] {
  const data = GUIDE_RC_PATHS[eid];
  if (!data) return [];
  const labelOf = (id: string) => data.nodes.find((n) => n.id === id)?.label ?? id;
  return data.paths
    .filter((p) => p.node_ids[0] === eid && p.node_ids.length > 1)
    .map((p) => p.node_ids.map((id) => ({ eid: id, label: labelOf(id) })));
}

export default function PersonalBrainSection() {
  const [view, setView] = useState<"hourglass" | "2d">("hourglass");
  const [focused, setFocused] = useState({
    eid:   GUIDE_IDS.rcInflation as string,
    label: "通膨黏性價格假說",
  });
  const [tree, setTree] = useState<FoundationTreeData | null>(null);
  const [loading, setLoading] = useState(false);

  const viewerRef = useRef<HTMLDivElement | null>(null);
  const [viewerW, setViewerW] = useState(520);

  useLayoutEffect(() => {
    const el = viewerRef.current;
    if (!el) return;
    const apply = () => setViewerW(Math.max(Math.floor(el.getBoundingClientRect().width), 280));
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    guideFetchFoundationTree(focused.eid).then((t) => {
      if (!alive) return;
      setTree(t);
      setLoading(false);
    });
    return () => { alive = false; };
  }, [focused.eid]);

  const chains = useMemo(() => chainsFor(focused.eid), [focused.eid]);

  // 腰身在球體上的水平偏移：取聚焦 RC 在語意地圖上相對重心的方向
  const waistBias = useMemo(() => {
    const p = BERTOPIC_POS.get(focused.eid);
    if (!p) return 0;
    return ((p.x - BERTOPIC_CENTROID.x) / (BERTOPIC_SPAN / 2)) * 0.8;
  }, [focused.eid]);

  const handleFocusNode = useCallback((eid: string, label: string) => {
    setFocused({ eid, label });
    setView("hourglass");
  }, []);

  return (
    <div className="guide-brain-stack">
      <SliceDiagram view={view} waistLabel={focused.label} waistBias={waistBias} />

      {/* 從切面收斂到檢視框：下面那塊是上面那塊取出來放大的 */}
      <svg className="guide-funnel" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id="guide-funnel-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#00d4aa" stopOpacity="0.14" />
            <stop offset="100%" stopColor="#00d4aa" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <path d="M 28 0 L 0 100 L 100 100 L 72 0 Z" fill="url(#guide-funnel-fill)" />
        <path d="M 28 0 L 0 100" stroke="rgba(0,212,170,0.4)" strokeWidth="0.4"
              vectorEffect="non-scaling-stroke" strokeDasharray="4,4" fill="none" />
        <path d="M 72 0 L 100 100" stroke="rgba(0,212,170,0.4)" strokeWidth="0.4"
              vectorEffect="non-scaling-stroke" strokeDasharray="4,4" fill="none" />
      </svg>

      <div className="guide-viewer">
        <div className="guide-viewer-bar">
          <span className="guide-viewer-focus">聚焦中 · {focused.label}</span>
          <div className="guide-viewer-toggle">
            {([
              { key: "hourglass", label: "沙漏" },
              { key: "2d",        label: "分佈" },
            ] as const).map((v) => (
              <button
                key={v.key}
                type="button"
                data-on={view === v.key}
                onClick={() => setView(v.key)}
              >
                {v.label}
              </button>
            ))}
          </div>
        </div>

        <div className="guide-viewer-body" ref={viewerRef} style={{ height: VIEWER_H }}>
          {view === "hourglass" ? (
            <HourglassViz
              focusedEid={focused.eid}
              focusedLabel={focused.label}
              foundationTree={tree}
              downstreamChains={chains}
              personalBrainEids={BRAIN_EIDS}
              bertopicPosByEid={BERTOPIC_POS}
              bertopicCentroid={BERTOPIC_CENTROID}
              skillEids={SKILL_EIDS}
              selectedSkillEid={null}
              onSelectSkill={() => {}}
              onFocusNode={handleFocusNode}
              loading={loading}
              width={viewerW}
              height={VIEWER_H}
            />
          ) : (
            <BertopicMap
              data={GUIDE_BERTOPIC_LAYOUT}
              personalBrain={GUIDE_PERSONAL_BRAIN}
              width={viewerW}
              height={VIEWER_H}
            />
          )}
        </div>
      </div>

      <div className="guide-slice-copy">
        <div className="guide-slice-copy-item">
          <span className="guide-slice-copy-title">分佈：你的偏好落在哪個方向</span>
          <p>把收藏投影到全域語意地圖上。淡色點是全域概念，亮色點是你的 Personal Brain，底部的比例列就是你的知識偏向。</p>
        </div>
        <div className="guide-slice-copy-item">
          <span className="guide-slice-copy-title">沙漏：單一 RC 的因果剖面</span>
          <p>下半錐是支撐目前 RC 的 foundation，上半錐是它能延伸出的 downstream。節點的方位角取自它在分佈圖上的方向——兩種視圖看的是同一個空間。</p>
        </div>
      </div>

      <p className="guide-note">
        點沙漏上半錐的節點會重新聚焦，球體上的腰身也會跟著移動。這裡是示範資料，到 dashboard 才會接上你真正的收藏。
      </p>
    </div>
  );
}
