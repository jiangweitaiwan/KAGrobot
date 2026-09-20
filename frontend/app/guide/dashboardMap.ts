// -*- coding: utf-8 -*-
/**
 * dashboardMap.ts — dashboard 的「虛擬座標系」。
 *
 * Section 1 的運鏡效果建立在一個假設上：整個 dashboard 是一張固定尺寸
 * 的畫布（CANVAS_W × CANVAS_H），五個欄位在上面各佔一塊固定矩形。
 * 切換步驟時我們不換畫面，只移動「相機」——也就是對畫布套一個
 * translate + scale。因為放大前後是同一個 DOM，使用者才讀得出
 * 「這一步就在剛剛那張全景圖的哪個位置」。
 *
 * 這裡的比例刻意對齊 dashboard/page.tsx 的 .content-area：
 *   chat-column / brain-panel(Global) / DetailPanel / SemanticPanel /
 *   brain-panel(Personal)，左到右五欄，上面壓一條 header。
 * 數值不需要跟線上 CSS 像素完全相同，只要相對關係一致即可——
 * 使用者記住的是「Detail Panel 在右邊數來第三欄」這件事。
 */

export const CANVAS_W = 1440;
export const CANVAS_H = 560;

export type Rect = { x: number; y: number; w: number; h: number };

export type SlotKey = "chat" | "global" | "detail" | "semantic" | "personal";

export const HEADER_RECT: Rect = { x: 0, y: 0, w: CANVAS_W, h: 48 };

/**
 * 五欄的完整矩形。
 *
 * 高度刻意壓成 484（而不是照抄 1080p 螢幕上那種接近 800 的細長欄位）：
 * 細長欄位放進寬螢幕舞台時，放大倍率會被高度卡死，看起來等於沒放大。
 * 壓扁之後每一欄都能以約兩倍放大填滿舞台，聚焦框也就等於整欄，
 * 不需要再切子區域，六個步驟落在四個相機位置上。
 */
export const SLOT_RECTS: Record<SlotKey, Rect> = {
  chat:     { x: 16,   y: 60, w: 300, h: 484 },
  global:   { x: 328,  y: 60, w: 384, h: 484 },
  detail:   { x: 724,  y: 60, w: 248, h: 484 },
  semantic: { x: 984,  y: 60, w: 184, h: 484 },
  personal: { x: 1180, y: 60, w: 244, h: 484 },
};

export const SLOT_ORDER: SlotKey[] = ["chat", "global", "detail", "semantic", "personal"];

export const SLOT_TITLES: Record<SlotKey, string> = {
  chat:     "對話",
  global:   "Global Knowledge Field",
  detail:   "Detail Panel",
  semantic: "Semantic",
  personal: "Personal Brain",
};

// ─────────────────────────────────────────────────────────────
// 相機
// ─────────────────────────────────────────────────────────────

export type Camera = { scale: number; tx: number; ty: number };

/** 可視範圍（viewport）能完整容納整張畫布的縮放倍率。 */
export function baseScale(vpW: number, vpH: number): number {
  return Math.min(vpW / CANVAS_W, vpH / CANVAS_H);
}

/**
 * 把 rect 置中放大到 viewport 裡。
 * 下限是 baseScale（永遠不會比全景還小），上限 3（再大就只剩色塊、
 * 失去「這是 dashboard 的一角」的辨識度）。
 */
export function cameraFor(rect: Rect, vpW: number, vpH: number, pad = 28): Camera {
  const base = baseScale(vpW, vpH);
  const fit  = Math.min(vpW / (rect.w + pad * 2), vpH / (rect.h + pad * 2));
  const scale = Math.max(base, Math.min(fit, 3));
  return {
    scale,
    tx: vpW / 2 - scale * (rect.x + rect.w / 2),
    ty: vpH / 2 - scale * (rect.y + rect.h / 2),
  };
}

/** 全景相機：整張畫布置中。 */
export function overviewCamera(vpW: number, vpH: number): Camera {
  const scale = baseScale(vpW, vpH);
  return {
    scale,
    tx: vpW / 2 - scale * (CANVAS_W / 2),
    ty: vpH / 2 - scale * (CANVAS_H / 2),
  };
}

export function sameRect(a: Rect | null, b: Rect | null): boolean {
  if (!a || !b) return false;
  return a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
}
