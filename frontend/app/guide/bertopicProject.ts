// -*- coding: utf-8 -*-
/**
 * bertopicProject.ts — 算出某個 eid 在 BertopicMap 上的像素座標。
 *
 * 為什麼不是跟 BertopicMap 要：
 *   BertopicMap 的 xScale / yScale 建立在 useEffect 內部，畫完就消失，
 *   沒有對外介面。Section 2 的「foundation 飛進語意地圖」需要知道落點，
 *   而 guide 不該為了一個示範動畫去改共用元件的 API——真正的 dashboard
 *   不需要這個能力。
 *
 * 所以這裡鏡像一份同樣的數學。這是刻意接受的耦合：若日後 BertopicMap
 * 的 PAD 或 domain padding 改了，這裡只會讓飛行終點偏個幾像素，不會壞掉。
 * 兩邊的常數請一起改。
 *
 * 另外，BertopicMap 掛了 d3.zoom。使用者若在 guide 裡縮放那張圖，
 * 這裡算出的座標就會偏——示範情境下不處理，落點偏移遠小於動畫本身的
 * 可讀性損失。
 */

import type { BertopicLayoutData } from "../components/BertopicMap";

/** 與 BertopicMap.tsx 的 PAD 常數一致。 */
const PAD = { top: 28, right: 20, bottom: 32, left: 32 };

export type Point = { x: number; y: number };

/**
 * 回傳一個 eid → SVG 內像素座標的查詢函式（相對於 BertopicMap 容器左上角）。
 * 找不到該 eid 時回傳 null。
 */
export function makeBertopicProjector(
  data:   BertopicLayoutData,
  width:  number,
  height: number,
): (eid: string) => Point | null {
  const W = width  - PAD.left - PAD.right;
  const H = height - PAD.top  - PAD.bottom;

  if (data.nodes.length === 0 || W <= 0 || H <= 0) return () => null;

  const xs = data.nodes.map((n) => n.x);
  const ys = data.nodes.map((n) => n.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);

  const xPad = (x1 - x0) * 0.05;
  const yPad = (y1 - y0) * 0.05;

  const dx0 = x0 - xPad, dx1 = x1 + xPad;
  const dy0 = y0 - yPad, dy1 = y1 + yPad;

  const byEid = new Map(data.nodes.filter((n) => n.eid).map((n) => [n.eid!, n]));

  return (eid: string) => {
    const n = byEid.get(eid);
    if (!n) return null;
    const tx = (n.x - dx0) / (dx1 - dx0 || 1);
    const ty = (n.y - dy0) / (dy1 - dy0 || 1);
    return {
      x: PAD.left + tx * W,
      y: PAD.top  + (1 - ty) * H,   // yScale 的 range 是 [H, 0]
    };
  };
}
