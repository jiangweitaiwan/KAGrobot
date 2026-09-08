// -*- coding: utf-8 -*-
/**
 * guideData.ts — /guide（操作心法）頁面專用的假資料。
 *
 * 整頁共用同一條敘事線：
 *   提問「升息會怎麼影響物價的黏性？」
 *   → 點選「價格僵固性」（paper 概念）
 *   → 加入 Personal Brain（連同「菜單成本理論」「Calvo 定價模型」）
 *   → 收斂出 RC「通膨黏性價格假說」
 *   → downstream 延伸出 RC「升息衝擊分析」
 *   → 送出後套用量化模型算出結論（見 ComputeSection）
 *
 * 刻意不打任何 API：所有座標、標籤都是手工／決定性生成，讓 Section 1
 * 的六步驟示範、Section 2 的 GlobalBrain 探索、Section 3 的
 * PersonalBrain 探索，三處都能重複出現同一批節點，讓使用者意識到
 * 這是同一件事的不同視角，而不是三份互不相干的教材。
 */

import type { GlobalLayout, GlobalLayoutNode } from "../components/GlobalBrainExplorer";
import type {
  ConceptDetail,
  FoundationTreeData,
  PersonalBrainConcept,
  RCFoundationMap,
  RCSemanticScore,
} from "../components/DetailPanel";
import type { RCRankedItem, RCPathData } from "../components/PersonalBrain";
import type { BertopicLayoutData } from "../components/BertopicMap";

// ─────────────────────────────────────────────────────────────
// 敘事線核心文案 / id
// ─────────────────────────────────────────────────────────────

export const GUIDE_QUESTION = "升息會怎麼影響物價的黏性？";

export const GUIDE_IDS = {
  feel:        "f-risk-averse",
  priceStick:  "k-price-stick",
  menuCost:    "k-menu-cost",
  calvo:       "k-calvo",
  rcInflation: "rc-inflation-stickiness",
  rcRateShock: "rc-rate-shock",
} as const;

// ─────────────────────────────────────────────────────────────
// 決定性座標生成（讓填充節點分佈自然，不用手 key 三十組座標）
// ─────────────────────────────────────────────────────────────

function sphericalFiller(
  count: number,
  radius: number,
  seedOffset: number,
): { x: number; y: number; z: number }[] {
  const pts: { x: number; y: number; z: number }[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (i / Math.max(count - 1, 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * (i + seedOffset);
    pts.push({
      x: Math.cos(theta) * r * radius,
      y: y * radius,
      z: Math.sin(theta) * r * radius,
    });
  }
  return pts;
}

// ─────────────────────────────────────────────────────────────
// GUIDE_LAYOUT — Section 2（完整探索用，約 26 個節點）
// ─────────────────────────────────────────────────────────────

const PAPER_LABELS = [
  "理性預期假說", "動態隨機均衡", "貨幣中立性", "資訊不對稱",
  "行為經濟偏誤", "跨期替代", "流動性偏好", "價格僵固性擴散",
  "供給彈性衝擊", "工資僵固性", "通膨預期形成", "利率傳導機制",
  "匯率傳遞效果", "產出缺口估計",
];
const CHAPTER_LABELS = ["總體經濟學原理", "貨幣理論基礎", "總合供需模型", "IS-LM 模型", "菲利浦曲線", "開放經濟三元悖論"];
const ARTICLE_LABELS = ["公平感", "信任傾向", "損失趨避"];

const FILLER_PAPER: GlobalLayoutNode[] = sphericalFiller(PAPER_LABELS.length, 0.95, 3).map((p, i) => ({
  id: `filler-paper-${i}`,
  name: PAPER_LABELS[i],
  type: "paper",
  source: "paper",
  ...p,
}));

const FILLER_CHAPTER: GlobalLayoutNode[] = sphericalFiller(CHAPTER_LABELS.length, 0.55, 7).map((p, i) => ({
  id: `filler-chapter-${i}`,
  name: CHAPTER_LABELS[i],
  type: "chapter",
  source: "chapter",
  ...p,
}));

const FILLER_ARTICLE: GlobalLayoutNode[] = sphericalFiller(ARTICLE_LABELS.length, 0.22, 11).map((p, i) => ({
  id: `filler-article-${i}`,
  name: ARTICLE_LABELS[i],
  type: "article",
  source: "article",
  ...p,
}));

const NAMED_NODES: GlobalLayoutNode[] = [
  { id: GUIDE_IDS.feel,        name: "風險趨避",         type: "article",           source: "article", x: 0.10, y: 0.16, z: -0.08 },
  { id: GUIDE_IDS.priceStick,  name: "價格僵固性",       type: "paper",             source: "paper",   x: 0.72, y: 0.30, z: -0.40 },
  { id: GUIDE_IDS.menuCost,    name: "菜單成本理論",     type: "paper",             source: "paper",   x: -0.55, y: -0.35, z: -0.60 },
  { id: GUIDE_IDS.calvo,       name: "Calvo 定價模型",   type: "paper",             source: "paper",   x: 0.60, y: -0.50, z: 0.45 },
  { id: GUIDE_IDS.rcInflation, name: "通膨黏性價格假說", type: "resulting_concept", source: "paper",   x: 0.05, y: 0.02, z: 0.06 },
  { id: GUIDE_IDS.rcRateShock, name: "升息衝擊分析",     type: "resulting_concept", source: "paper",   x: 0.20, y: 0.28, z: 0.30 },
];

export const GUIDE_LAYOUT: GlobalLayout = {
  nodes: [...NAMED_NODES, ...FILLER_PAPER, ...FILLER_CHAPTER, ...FILLER_ARTICLE],
  edges: [
    { source: GUIDE_IDS.priceStick,  target: GUIDE_IDS.rcInflation },
    { source: GUIDE_IDS.menuCost,    target: GUIDE_IDS.rcInflation },
    { source: GUIDE_IDS.calvo,       target: GUIDE_IDS.rcInflation },
    { source: GUIDE_IDS.feel,        target: GUIDE_IDS.rcInflation },
    { source: GUIDE_IDS.rcInflation, target: GUIDE_IDS.rcRateShock },
    ...FILLER_PAPER.slice(0, 6).map((n) => ({ source: n.id, target: GUIDE_IDS.priceStick })),
  ],
};

// ─────────────────────────────────────────────────────────────
// GUIDE_LAYOUT_MINI — Section 1 步驟②用的縮小版
// ─────────────────────────────────────────────────────────────

export const GUIDE_LAYOUT_MINI: GlobalLayout = {
  nodes: [...NAMED_NODES, ...FILLER_PAPER.slice(0, 2)],
  edges: GUIDE_LAYOUT.edges.filter(
    (e) => NAMED_NODES.some((n) => n.id === e.source) && NAMED_NODES.some((n) => n.id === e.target),
  ),
};

// ─────────────────────────────────────────────────────────────
// ConceptDetail — 「價格僵固性」的假詳情
// ─────────────────────────────────────────────────────────────

export const GUIDE_CONCEPT_DETAIL: ConceptDetail = {
  found: true,
  eid: GUIDE_IDS.priceStick,
  rc_name: "價格僵固性",
  definition: "廠商在成本或需求改變後，不會立刻調整價格，而是維持一段時間不變的現象。",
  mechanism: "價格調整本身有成本（菜單成本）與資訊蒐集成本，廠商會權衡「調價的收益」與「不調價的損失」，只有當落差夠大時才會重新定價。",
  key_assumptions: "廠商是理性但面對調價成本的決策者；市場並非完全競爭，廠商對自己的價格有一定的控制權。",
  function: "解釋貨幣政策為何在短期內對實質產出有影響（而非只影響物價）。",
  analysis_types: ["causal_mechanism", "policy_evaluation"],
  concept_type: "paper",
  papers: [
    { title: "Calvo (1983) Staggered Prices in a Utility-Maximizing Framework", premises: "廠商依外生機率隨機重新定價", constraints: "忽略廠商規模差異" },
  ],
};

// ─────────────────────────────────────────────────────────────
// FoundationTreeData — anchor＝「通膨黏性價格假說」
// ─────────────────────────────────────────────────────────────

export const GUIDE_FOUNDATION_TREE: FoundationTreeData = {
  anchor_eid: GUIDE_IDS.rcInflation,
  anchor_label: "通膨黏性價格假說",
  foundations: [
    { element_id: GUIDE_IDS.priceStick, label: "價格僵固性" },
    { element_id: GUIDE_IDS.menuCost,   label: "菜單成本理論" },
    { element_id: GUIDE_IDS.calvo,      label: "Calvo 定價模型" },
    { element_id: GUIDE_IDS.feel,       label: "風險趨避" },
  ],
  nodes: [
    { element_id: GUIDE_IDS.rcInflation, label: "通膨黏性價格假說" },
    { element_id: GUIDE_IDS.priceStick,  label: "價格僵固性" },
    { element_id: GUIDE_IDS.menuCost,    label: "菜單成本理論" },
    { element_id: GUIDE_IDS.calvo,       label: "Calvo 定價模型" },
    { element_id: GUIDE_IDS.feel,        label: "風險趨避" },
  ],
  edges: [
    { source: GUIDE_IDS.priceStick, target: GUIDE_IDS.rcInflation },
    { source: GUIDE_IDS.menuCost,   target: GUIDE_IDS.rcInflation },
    { source: GUIDE_IDS.calvo,      target: GUIDE_IDS.rcInflation },
    { source: GUIDE_IDS.feel,       target: GUIDE_IDS.rcInflation },
  ],
};

// ─────────────────────────────────────────────────────────────
// PersonalBrainConcept — 已加入的收藏（含步驟④剛加入的幾筆）
// ─────────────────────────────────────────────────────────────

export const GUIDE_PERSONAL_BRAIN: PersonalBrainConcept[] = [
  { eid: GUIDE_IDS.priceStick, label: "價格僵固性",     source: "manual",       weight: 0.8,  addedAt: Date.now() - 60_000 },
  { eid: GUIDE_IDS.menuCost,   label: "菜單成本理論",   source: "manual",       weight: 0.65, addedAt: Date.now() - 30_000 },
  { eid: GUIDE_IDS.calvo,      label: "Calvo 定價模型", source: "llm_suggest",  weight: 0.5,  addedAt: Date.now() - 20_000 },
  { eid: GUIDE_IDS.feel,       label: "風險趨避",       source: "path_confirm", weight: 0.4,  addedAt: Date.now() - 10_000 },
];

// ─────────────────────────────────────────────────────────────
// RCFoundationMap / RCSemanticScore（DetailPanel 排名預覽需要）
// ─────────────────────────────────────────────────────────────

export const GUIDE_RC_FOUNDATION_MAP: RCFoundationMap = new Map([
  [GUIDE_IDS.rcInflation, {
    label: "通膨黏性價格假說",
    foundationEids: GUIDE_FOUNDATION_TREE.foundations.map((f) => f.element_id),
    treeNodes: GUIDE_FOUNDATION_TREE.nodes,
    treeEdges: GUIDE_FOUNDATION_TREE.edges,
  }],
  [GUIDE_IDS.rcRateShock, {
    label: "升息衝擊分析",
    foundationEids: [GUIDE_IDS.rcInflation],
  }],
]);

export const GUIDE_RC_SEMANTIC_SCORES: RCSemanticScore[] = [
  { eid: GUIDE_IDS.rcInflation, label: "通膨黏性價格假說", score: 0.86, embed_score: 0.52, intent_score: 0.34 },
  { eid: GUIDE_IDS.rcRateShock, label: "升息衝擊分析",     score: 0.61, embed_score: 0.38, intent_score: 0.23 },
];

// ─────────────────────────────────────────────────────────────
// RC Ranking（PersonalBrain Top-3 標籤用）
// ─────────────────────────────────────────────────────────────

export const GUIDE_RC_RANKING: RCRankedItem[] = [
  { eid: GUIDE_IDS.rcInflation, label: "通膨黏性價格假說", totalScore: 0.83 },
  { eid: GUIDE_IDS.rcRateShock, label: "升息衝擊分析",     totalScore: 0.55 },
];

// ─────────────────────────────────────────────────────────────
// RCPathData — downstream 鏈（通膨黏性價格假說 → 升息衝擊分析）
// ─────────────────────────────────────────────────────────────

export const GUIDE_RC_PATHS: Record<string, RCPathData> = {
  [GUIDE_IDS.rcInflation]: {
    eid: GUIDE_IDS.rcInflation,
    label: "通膨黏性價格假說",
    nodes: [
      { id: GUIDE_IDS.rcInflation, label: "通膨黏性價格假說", node_type: "resulting_concept", concept_type: "resulting_concept", matched: true, has_skill: false },
      { id: GUIDE_IDS.rcRateShock, label: "升息衝擊分析",     node_type: "resulting_concept", concept_type: "resulting_concept", matched: true, has_skill: true },
    ],
    edges: [{ id: "e1", source: GUIDE_IDS.rcInflation, target: GUIDE_IDS.rcRateShock }],
    paths: [{
      path_id: 1,
      node_ids: [GUIDE_IDS.rcInflation, GUIDE_IDS.rcRateShock],
      edge_ids: ["e1"],
      final_label: "升息衝擊分析",
      intent_score: 0.7,
    }],
  },
  [GUIDE_IDS.rcRateShock]: {
    eid: GUIDE_IDS.rcRateShock,
    label: "升息衝擊分析",
    nodes: [
      { id: GUIDE_IDS.rcRateShock, label: "升息衝擊分析", node_type: "resulting_concept", concept_type: "resulting_concept", matched: true, has_skill: true },
    ],
    edges: [],
    paths: [],
  },
};

export async function guideFetchRCPaths(eid: string): Promise<RCPathData | null> {
  return GUIDE_RC_PATHS[eid] ?? null;
}

export async function guideFetchFoundationTree(eid: string): Promise<FoundationTreeData | null> {
  if (eid === GUIDE_IDS.rcInflation) return GUIDE_FOUNDATION_TREE;
  return null;
}

export async function guideFetchConceptDetail(eid: string): Promise<ConceptDetail | null> {
  if (eid === GUIDE_IDS.priceStick) return GUIDE_CONCEPT_DETAIL;
  const fallbackName = GUIDE_LAYOUT.nodes.find((n) => n.id === eid)?.name ?? eid;
  return {
    found: true,
    eid,
    rc_name: fallbackName,
    definition: "（操作心法示範資料，尚無詳細內容——到 dashboard 探索真實資料會有完整說明）",
    mechanism: null,
    key_assumptions: null,
    function: null,
    analysis_types: [],
    concept_type: null,
    papers: [],
  };
}

// ─────────────────────────────────────────────────────────────
// BertopicLayoutData — 假 2D 分佈
// ─────────────────────────────────────────────────────────────

const BERTOPIC_TOPICS = [
  { id: 0, label: "0_貨幣政策", color: "#00e8ff", words: ["利率", "央行", "傳導"] },
  { id: 1, label: "1_定價行為", color: "#b48cff", words: ["菜單成本", "僵固", "調價"] },
  { id: 2, label: "2_預期形成", color: "#ff9a52", words: ["理性預期", "適應性預期"] },
];

function bertopicFiller(count: number, topicId: number, cx: number, cy: number, spread: number) {
  const out = [];
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2;
    const r = spread * (0.4 + 0.6 * ((i * 37) % 10) / 10);
    out.push({
      eid: null,
      name: `filler-${topicId}-${i}`,
      x: cx + Math.cos(a) * r,
      y: cy + Math.sin(a) * r,
      topic_id: topicId,
      topic_label: BERTOPIC_TOPICS[topicId].label,
      color: BERTOPIC_TOPICS[topicId].color,
    });
  }
  return out;
}

export const GUIDE_BERTOPIC_LAYOUT: BertopicLayoutData = {
  nodes: [
    { eid: GUIDE_IDS.priceStick, name: "價格僵固性",     x: 5.2,  y: 1.1, topic_id: 1, topic_label: "1_定價行為", color: "#b48cff" },
    { eid: GUIDE_IDS.menuCost,   name: "菜單成本理論",   x: 5.6,  y: 1.6, topic_id: 1, topic_label: "1_定價行為", color: "#b48cff" },
    { eid: GUIDE_IDS.calvo,      name: "Calvo 定價模型", x: 4.9,  y: 0.7, topic_id: 1, topic_label: "1_定價行為", color: "#b48cff" },
    { eid: GUIDE_IDS.feel,       name: "風險趨避",       x: -2.1, y: 3.0, topic_id: 2, topic_label: "2_預期形成", color: "#ff9a52" },
    ...bertopicFiller(10, 0, -3, -1, 1.6),
    ...bertopicFiller(8, 1, 5.2, 1.2, 1.4),
    ...bertopicFiller(8, 2, -2, 3, 1.6),
  ],
  topics: BERTOPIC_TOPICS,
};
