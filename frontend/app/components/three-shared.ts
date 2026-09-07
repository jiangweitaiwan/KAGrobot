// -*- coding: utf-8 -*-
/**
 * three-shared.ts — GlobalBrainExplorer 與 HourglassViz 共用的 Three.js helper。
 *
 * 這些函式原本寫在 GlobalBrainExplorer.tsx 內部，抽到這裡讓 HourglassViz.tsx
 * 可以複用同一套 sprite / dispose 邏輯，避免兩份場景的視覺語言（label 樣式、
 * glow 質感、hover ring）不小心走鐘。
 *
 * GlobalBrainExplorer.tsx 之後應該改成從這裡 import，而不是保留自己的私有版本，
 * 這是這次重構順手做的清理，行為完全不變。
 *
 * ── Mipmap 修正（新增）──────────────────────────────────────
 *   所有 CanvasTexture 一律關閉 generateMipmaps，並改用 LinearFilter。
 *   原因：makeLabelSprite 的 canvas 尺寸是 320x72（非二的次方，NPOT），
 *   部分瀏覽器/驅動在為 NPOT 貼圖產生 mipmap 時，會誤觸 texImage3D 上傳
 *   路徑並丟出「texImage3D: FLIP_Y or PREMULTIPLY_ALPHA isn't allowed
 *   for uploading 3D textures」這個 INVALID_OPERATION 錯誤，導致該次
 *   render pass 後續的 draw call 一併被拖垮（畫面整個空白，不只是那個
 *   NPOT 貼圖消失）。這些 sprite 貼圖本來就不會被縮到很小，不需要 mip
 *   chain，直接關閉最乾淨，也順便省一點記憶體與生成 mipmap 的成本。
 */

import * as THREE from "three";

// ─────────────────────────────────────────────────────────────
// 決定性亂數（fallback 用）
// ─────────────────────────────────────────────────────────────

/** 簡單字串 hash，供 mulberry32 當 seed。與 GlobalBrainExplorer 原本
 *  `abs(hash(eid)) % 2**32` 的精神一致，只是抽成獨立函式。 */
export function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  }
  return h >>> 0;
}

/** mulberry32：輕量決定性 PRNG，同一 seed 永遠吐出同一序列，
 *  用來在缺乏真實座標（例如 bertopic_layout.json 尚未收錄該 eid）時，
 *  給節點一個穩定、不會每次 render 都跳動的 fallback 角度。 */
export function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ─────────────────────────────────────────────────────────────
// Canvas 繪製 helper
// ─────────────────────────────────────────────────────────────

export function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y,     x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x,     y + h, r);
  ctx.arcTo(x,     y + h, x,     y,     r);
  ctx.arcTo(x,     y,     x + w, y,     r);
  ctx.closePath();
}

/** 統一套用在所有 CanvasTexture 上的設定：關閉 mipmap，避免 NPOT 貼圖
 *  觸發 texImage3D 錯誤並拖垮整個 render pass（見檔頭說明）。 */
function configureCanvasTexture(tex: THREE.CanvasTexture): void {
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
}

export function makeLabelSprite(text: string, colorHex: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 320; canvas.height = 72;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgba(13,18,32,0.85)";
  roundRectPath(ctx, 2, 2, canvas.width - 4, canvas.height - 4, 14);
  ctx.fill();
  ctx.strokeStyle = "rgba(232,238,248,0.18)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = colorHex;
  ctx.font = "26px \"IBM Plex Mono\", monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  configureCanvasTexture(tex);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.62, 0.14, 1);
  return spr;
}

export function makeRingSprite(colorHex: string, lineWidth = 5): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = colorHex;
  ctx.beginPath();
  ctx.arc(32, 32, 24, 0, Math.PI * 2);
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  configureCanvasTexture(tex);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  return new THREE.Sprite(mat);
}

export function makeGlowSprite(colorHex: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext("2d")!;
  const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0,    `${colorHex}cc`);
  grad.addColorStop(0.35, `${colorHex}55`);
  grad.addColorStop(0.7,  `${colorHex}15`);
  grad.addColorStop(1,    `${colorHex}00`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(canvas);
  configureCanvasTexture(tex);
  const mat = new THREE.SpriteMaterial({
    map: tex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  return new THREE.Sprite(mat);
}

export function makeFlatDotSprite(colorHex: string, strokeHex = "#e8eef8"): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 32;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = colorHex;
  ctx.beginPath();
  ctx.arc(16, 16, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = strokeHex;
  ctx.lineWidth = 2;
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  configureCanvasTexture(tex);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.05, 0.05, 1);
  return spr;
}



/** 方塊 sprite：RC（Resulting Concept）節點專用主體圖示，
 *  與其他圓點節點做形狀區隔，讓使用者一眼認出「這是可執行方案」。 */
export function makeSquareDotSprite(colorHex: string, strokeHex = "#e8eef8"): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 32;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = colorHex;
  roundRectPath(ctx, 5, 5, 22, 22, 4);
  ctx.fill();
  ctx.strokeStyle = strokeHex;
  ctx.lineWidth = 2;
  roundRectPath(ctx, 5, 5, 22, 22, 4);
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.05, 0.05, 1);
  return spr;
}

/** 大閃電 sprite：ACTION 節點專用主體圖示。
 *  與 makeBadgeSprite（RC 旁的小徽章、「送入 MainAgent」開關）不同——
 *  這是節點本體，筆畫更粗、輪廓更飽滿，縮小到列表尺寸也能辨識。 */
export function makeLightningNodeSprite(colorHex: string, strokeHex = "#fff3c4"): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 32;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = colorHex;
  ctx.beginPath();
  ctx.moveTo(19, 2);
  ctx.lineTo(8, 18);
  ctx.lineTo(15, 18);
  ctx.lineTo(12, 30);
  ctx.lineTo(25, 12);
  ctx.lineTo(17, 12);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = strokeHex;
  ctx.lineWidth = 1.6;
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.05, 0.05, 1);
  return spr;
}
/** 空心版 flat dot：只畫描邊、中心透空，供「尚未加入 Personal Brain 的
 *  foundation」使用，與實心版做出「已選 / 未選」的視覺區隔。 */
export function makeHollowDotSprite(strokeHex: string, lineWidth = 3): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 32;
  const ctx = canvas.getContext("2d")!;
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = strokeHex;
  ctx.beginPath();
  ctx.arc(16, 16, 9, 0, Math.PI * 2);
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  configureCanvasTexture(tex);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.05, 0.05, 1);
  return spr;
}

/** 小閃電圖示，標記「此節點有可執行的 RC Skill」。 */
export function makeBadgeSprite(colorHex: string, strokeHex = "#fff3c4"): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 32;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = colorHex;
  ctx.beginPath();
  ctx.moveTo(18, 3);
  ctx.lineTo(9, 18);
  ctx.lineTo(15, 18);
  ctx.lineTo(13, 29);
  ctx.lineTo(24, 13);
  ctx.lineTo(17, 13);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = strokeHex;
  ctx.lineWidth = 1.2;
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  configureCanvasTexture(tex);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.045, 0.045, 1);
  return spr;
}
// ─────────────────────────────────────────────────────────────
// Dispose helper
// ─────────────────────────────────────────────────────────────

export function disposeObject3D(root: THREE.Object3D) {
  root.traverse((child: any) => {
    if (child.geometry) {
      child.geometry.dispose();
    }
    if (child.material) {
      const mats: THREE.Material[] = Array.isArray(child.material)
        ? child.material
        : [child.material];
      mats.forEach((m: any) => {
        if (m.map) m.map.dispose();
        m.dispose();
      });
    }
  });
}

export function disposeGroupChildren(group: THREE.Group) {
  group.children.forEach((child) => disposeObject3D(child));
}

/** 死路終點的 ✕ 標記，供 GlobalBrainExplorer 的 walk 視覺化使用。 */
export function makeCrossSprite(colorHex: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 32;
  const ctx = canvas.getContext("2d")!;
  ctx.strokeStyle = colorHex;
  ctx.lineWidth = 3.5;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(9, 9); ctx.lineTo(23, 23);
  ctx.moveTo(23, 9); ctx.lineTo(9, 23);
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.05, 0.05, 1);
  return spr;
}
