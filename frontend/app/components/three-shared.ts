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
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.05, 0.05, 1);
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
