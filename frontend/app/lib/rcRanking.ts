// -*- coding: utf-8 -*-
/**
 * lib/rcRanking.ts — RC 排名共用公式。
 *
 * 取代 page.tsx::computeRCRanking 與 DetailPanel.tsx 內部私有
 * computeRanking 兩份原本各自維護、容易分岔的實作。
 *
 * interScore = (S - C) / Z
 *   S = Σ weight(foundation) * support_weight   （僅計入 brain 裡有命中的 foundation）
 *   C = Σ weight(foundation) * critique_weight
 *   Z = Σ (support_weight + critique_weight)     （此 RC 全體 foundation，不論是否命中）
 *   Z = 0（沒有任何 support/critique 質量）時 interScore 記為 0。
 */

export type FoundationPolarity = "support" | "critique" | "other";

export type RCFoundationEntry = {
  eid: string;
  supportWeight: number;
  critiqueWeight: number;
};

export type RCFoundationMapEntry = {
  label: string;
  foundations: RCFoundationEntry[];
  treeNodes?: { element_id: string; label: string; node_type?: string }[];
  treeEdges?: { source: string; target: string; edge_type?: string }[];
};

export type RCFoundationMap = Map<string, RCFoundationMapEntry>;


// ─────────────────────────────────────────────────────────────
// /graph/expand/{eid} 與 /graph/expand/batch 的回傳結構（單一來源）
// 對應後端 ExpandResponse。DetailPanel 會 re-export，既有 import 不變。
// ─────────────────────────────────────────────────────────────

export type FoundationTreeData = {
  anchor_eid:   string;
  anchor_label: string;
  foundations: {
    element_id:      string;
    label:           string;
    depth:           number;
    dominant_type:   FoundationPolarity;
    support_weight:  number;
    critique_weight: number;
  }[];
  nodes: { element_id: string; label: string; node_type: FoundationPolarity }[];
  edges: { source: string; target: string; edge_type: "support" | "critique" }[];
};

function toPolarity(v: unknown): FoundationPolarity {
  return v === "support" || v === "critique" ? v : "other";
}

/** 防禦性正規化：欄位缺漏時補上與後端 schema 相同的預設值
 *  （FoundationItem: depth=0/dominant_type="other"/weights=0；
 *   FoundationTreeNode.node_type 與 Edge.edge_type 預設 "support"）。 */
export function normalizeFoundationTree(raw: any): FoundationTreeData {
  return {
    anchor_eid:   raw?.anchor_eid ?? "",
    anchor_label: raw?.anchor_label ?? raw?.anchor_eid ?? "",
    foundations: (raw?.foundations ?? []).map((f: any) => ({
      element_id:      f.element_id,
      label:           f.label,
      depth:           f.depth ?? 0,
      dominant_type:   toPolarity(f.dominant_type),
      support_weight:  f.support_weight ?? 0,
      critique_weight: f.critique_weight ?? 0,
    })),
    nodes: (raw?.nodes ?? []).map((n: any) => ({
      element_id: n.element_id,
      label:      n.label,
      node_type:  toPolarity(n.node_type ?? "support"),
    })),
    edges: (raw?.edges ?? []).map((e: any) => ({
      source:    e.source,
      target:    e.target,
      edge_type: e.edge_type === "critique" ? "critique" : "support",
    })),
  };
}

/** expand 結果 → rcFoundationMap 的一筆 entry（dashboard 與 guide 共用）。 */
export function foundationTreeToRCEntry(tree: FoundationTreeData): RCFoundationMapEntry {
  return {
    label: tree.anchor_label || tree.anchor_eid,
    foundations: tree.foundations.map((f) => ({
      eid:            f.element_id,
      supportWeight:  f.support_weight,
      critiqueWeight: f.critique_weight,
    })),
    treeNodes: tree.nodes,
    treeEdges: tree.edges,
  };
}


export type RCRanked = {
  eid: string;
  label: string;
  semantic: number;
  interScore: number;
  totalScore: number;
  supportSum: number;
  critiqueSum: number;
  hits: number;        // brain 裡有命中的 foundation 數
  totalFound: number;  // 此 RC 的 foundation 總數
};

export function computeRCRanking(
  weightMap: Map<string, number>,
  semMap: Map<string, number>,
  rcFoundationMap: RCFoundationMap,
  alpha: number,
): RCRanked[] {
  const results: RCRanked[] = [];

  rcFoundationMap.forEach(({ label, foundations }, rcEid) => {
    const semantic = semMap.get(rcEid) ?? 0;

    let S = 0, C = 0, Z = 0, hits = 0;
    for (const f of foundations) {
      const w = weightMap.get(f.eid);
      if (w !== undefined) {
        S += w * f.supportWeight;
        C += w * f.critiqueWeight;
        hits++;
      }
      Z += f.supportWeight + f.critiqueWeight;
    }
    const interScore = Z > 0 ? (S - C) / Z : 0;

    results.push({
      eid: rcEid,
      label,
      semantic,
      interScore,
      totalScore: alpha * semantic + (1 - alpha) * interScore,
      supportSum: S,
      critiqueSum: C,
      hits,
      totalFound: foundations.length,
    });
  });

  return results.sort((a, b) => b.totalScore - a.totalScore);
}