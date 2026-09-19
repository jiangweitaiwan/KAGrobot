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