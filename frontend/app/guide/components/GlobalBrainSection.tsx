"use client";

/**
 * GlobalBrainSection.tsx — Section 2:「探索 GlobalBrain」的迷你 playground。
 *
 * 左邊是完整版 GlobalBrainExplorer（真元件 + guideData 假資料）,
 * 右邊是真實 DetailPanel,點左邊節點就能在右邊看到詳情——這是
 * Section 1 步驟②③的放大、可持續把玩版本,兩者共用同一批
 * guideData 節點,讓使用者感覺是同一個世界的延伸探索。
 */

import { useCallback, useState } from "react";
import dynamic from "next/dynamic";
import type { GlobalLayoutNode } from "../../components/GlobalBrainExplorer";
import type { ConceptDetail, FoundationTreeData } from "../../components/DetailPanel";
import {
  GUIDE_LAYOUT,
  GUIDE_IDS,
  GUIDE_PERSONAL_BRAIN,
  GUIDE_RC_SEMANTIC_SCORES,
  GUIDE_RC_FOUNDATION_MAP,
  guideFetchConceptDetail,
  guideFetchFoundationTree,
} from "../guideData";

const GlobalBrainExplorer = dynamic(() => import("../../components/GlobalBrainExplorer"), { ssr: false });
const DetailPanel         = dynamic(() => import("../../components/DetailPanel"),         { ssr: false });

export default function GlobalBrainSection() {
  const [selectedNode, setSelectedNode]     = useState<GlobalLayoutNode | null>(null);
  const [detail, setDetail]                 = useState<ConceptDetail | null>(null);
  const [foundationTree, setFoundationTree] = useState<FoundationTreeData | null>(null);
  const [loadingTree, setLoadingTree]       = useState(false);

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

  return (
    <div className="guide-explore-grid">
      <div className="guide-explore-visual">
        <div className="guide-3d-frame guide-3d-frame-tall">
          <GlobalBrainExplorer
            layout={GUIDE_LAYOUT}
            activeCandidateEids={[GUIDE_IDS.rcInflation, GUIDE_IDS.rcRateShock]}
            rcFoundationMap={GUIDE_RC_FOUNDATION_MAP}
            onNodeClick={handleNodeClick}
            width={560}
            height={420}
          />
        </div>
        <ul className="guide-legend-notes">
          <li><strong>核心 article</strong>：直覺與情緒反應，例如「風險趨避」。</li>
          <li><strong>教科書 chapter</strong>：打底知識，例如「總體經濟學原理」。</li>
          <li><strong>論文 paper</strong>：你讀過、蒐集到的具體文獻。</li>
          <li><strong>洋紅標示環</strong>：目前語意上最相關的候選 RC。</li>
        </ul>
      </div>

      <div className="guide-explore-detail">
        <DetailPanel
          node={selectedNode}
          detail={detail}
          foundationTree={foundationTree}
          loadingTree={loadingTree}
          personalBrain={GUIDE_PERSONAL_BRAIN}
          rcSemanticScores={GUIDE_RC_SEMANTIC_SCORES}
          rcFoundationMap={GUIDE_RC_FOUNDATION_MAP}
          alpha={0.5}
          fetchConceptDetail={guideFetchConceptDetail}
          onAddFoundations={() => {}}
          onRemove={() => {}}
          onUpdateWeight={() => {}}
          onAskAboutNode={() => {}}
        />
      </div>
    </div>
  );
}
