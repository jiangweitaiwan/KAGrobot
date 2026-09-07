// -*- coding: utf-8 -*-
/**
 * demoData.ts — /intro 頁面專用的手工示範資料。
 *
 * 刻意不打任何 API：這裡的座標、標籤都是手工排列，追求視覺上乾淨、
 * 可控的敘事效果（感受在核心、知識在外圍、信念是幾條知識線索收斂
 * 而成），而不是精確反映真實圖譜的分佈。與正式系統的
 * GlobalBrainExplorer / HourglassViz 使用的座標系統一致（單位球
 * 半徑 0~1 附近），方便日後如果要接真實資料時座標概念不必重改。
 */

export type DemoNode = {
  id: string;
  label: string;
  kind: "feel" | "knowledge";
  x: number;
  y: number;
  z: number;
};

export type DemoEdge = { source: string; target: string };

// ── 感受（article／核心，邊緣系統）──────────────────────────
export const FEEL_NODES: DemoNode[] = [
  { id: "f1", label: "稀缺性直覺", kind: "feel", x: 0.06, y: 0.13, z: -0.08 },
  { id: "f2", label: "公平感",     kind: "feel", x: -0.15, y: -0.05, z: 0.10 },
  { id: "f3", label: "風險趨避",   kind: "feel", x: 0.11, y: -0.15, z: 0.06 },
  { id: "f4", label: "信任傾向",   kind: "feel", x: -0.08, y: 0.10, z: -0.14 },
];

// ── 知識（paper／外層，皮質）─────────────────────────────────
export const KNOWLEDGE_NODES: DemoNode[] = [
  { id: "k1",  label: "Calvo 定價模型",   kind: "knowledge", x: 0.78,  y: 0.42,  z: -0.35 },
  { id: "k2",  label: "理性預期假說",     kind: "knowledge", x: -0.65, y: 0.58,  z: 0.30 },
  { id: "k3",  label: "動態隨機均衡",     kind: "knowledge", x: 0.30,  y: -0.85, z: 0.40 },
  { id: "k4",  label: "菜單成本理論",     kind: "knowledge", x: -0.42, y: -0.70, z: -0.55 },
  { id: "k5",  label: "貨幣中立性",       kind: "knowledge", x: 0.88,  y: -0.20, z: 0.35 },
  { id: "k6",  label: "資訊不對稱",       kind: "knowledge", x: -0.80, y: -0.10, z: -0.45 },
  { id: "k7",  label: "行為經濟偏誤",     kind: "knowledge", x: 0.15,  y: 0.90,  z: -0.30 },
  { id: "k8",  label: "跨期替代",         kind: "knowledge", x: -0.20, y: -0.35, z: 0.90 },
  { id: "k9",  label: "流動性偏好",       kind: "knowledge", x: 0.55,  y: 0.25,  z: 0.75 },
  { id: "k10", label: "價格僵固性",       kind: "knowledge", x: -0.55, y: 0.30,  z: -0.75 },
];

export const ALL_SPHERE_NODES: DemoNode[] = [...FEEL_NODES, ...KNOWLEDGE_NODES];

// ── 信念:這幾個知識/感受節點的線索收斂成一個新的信念節點 ──────
// (擴充到 7 個節點,收斂網絡更密集)
export const FOUNDATION_IDS = ["k1", "k4", "k10", "f3", "k6", "k9", "f1"];

export const BELIEF_NODE = {
  id: "belief",
  label: "通膨黏性價格假說",
  x: 0.15,
  y: 0.05,
  z: 1.9,
};

export const BELIEF_EDGES: DemoEdge[] = FOUNDATION_IDS.map((id) => ({
  source: id,
  target: BELIEF_NODE.id,
}));

// ── 行動:信念驅動出的下一步(座標放在 belief 更外側,象徵推理落地延伸出去)
export const ACTION_NODE = {
  id: "action",
  label: "模擬升息衝擊路徑",
  x: 0.4,
  y: 0.6,
  z: 2.75,
};