"use client";

/**
 * IntroGlobalBrain.tsx — /intro 頁面的簡化版 3D 球體(單一場景版)。
 *
 * ── 本次改版重點 ──────────────────────────────────────────────
 *   1. 全面放大節點/光暈/字級,搭配外部改為全寬長方形視窗,畫面更飽滿,
 *      同時把 camera 拉近、放大點擊熱區(PICK_PIXEL_RADIUS),避免節點
 *      在較大尺寸下反而難點到。
 *   2. 新增 RC（Resulting Concept）節點:foundation tree 收斂出的信念
 *      (belief)再往下一步,是「可被執行」的具體方案,座落在 belief 與
 *      action 之間。RC 節點右上角掛一個閃電徽章(視覺語言沿用正式系統
 *      HourglassViz 的 makeBadgeSprite),代表「送入 MainAgent」的開關:
 *        - 點擊 RC 節點本體 → 跳回「信念」分頁(回頭看它是哪個信念收斂出來的)
 *        - 點擊閃電徽章     → 跳到「行動」分頁,並沿 belief→RC→action
 *                              的路徑播放一次高亮的「送出脈衝」動畫,
 *                              動畫結束後場景維持一條較暗的常駐脈衝,
 *                              象徵「這條路徑已啟用」。
 *
 * 其餘結構(四分頁球體、可點擊拾取、pause-then-resume 自轉)沿用前一版。
 */

import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import {
  makeGlowSprite,
  makeFlatDotSprite,
  makeLabelSprite,
  makeBadgeSprite,
  makeSquareDotSprite,
  makeLightningNodeSprite,
  disposeObject3D,
  disposeGroupChildren,
  hashStr,
  mulberry32,
} from "../../components/three-shared";
import {
  ALL_SPHERE_NODES,
  FOUNDATION_IDS,
  BELIEF_NODE,
  RC_NODE,
  ACTION_NODE,
  type DemoNode,
} from "../demoData";

export type IntroBrainMode = "feel" | "knowledge" | "belief" | "action";

type Props = {
  mode: IntroBrainMode;
  onRegionClick?: (mode: IntroBrainMode) => void;
  width: number;
  height: number;
};

type PickTarget = { group: IntroBrainMode; pos: THREE.Vector3 };

type Pulse = { curve: THREE.CatmullRomCurve3; dot: THREE.Sprite; phase: number };

const WORLD_SCALE = 2.65;

// ── 基底色:所有節點預設用這個,亮度足以看出整顆球體的形狀 ──────
const BASE_COLOR = new THREE.Color(0x6b7aa0);
const BASE_OPACITY = 0.65;
const BASE_STROKE = "#8a97b8";

const FEEL_COLOR = new THREE.Color(0x00e8ff);
const KNOWLEDGE_COLOR = new THREE.Color(0xff9a52);
const BELIEF_COLOR = new THREE.Color(0xff5fd1);
const RC_COLOR_HEX = "#f0a83c";
const ACTION_COLOR = new THREE.Color(0xffcc4d);

// RC 節點右上角閃電徽章相對於節點本體的世界座標偏移量
const RC_BADGE_OFFSET = 0.2;

const PICK_PIXEL_RADIUS = 32;
const ROTATE_SPEED = 0.0012;
const ROTATE_PAUSE_MS = 2200;

function toWorld(n: { x: number; y: number; z: number }): THREE.Vector3 {
  return new THREE.Vector3(n.x * WORLD_SCALE, n.y * WORLD_SCALE, n.z * WORLD_SCALE);
}

/** 依序連接一串節點,每一段都插入一個決定性偏移的彎折點,取代直線,
 *  整體組成一條 CatmullRomCurve3。同一組節點/seed 每次 render 結果一致。 */
function buildWindingPath(points: THREE.Vector3[], seed: number): THREE.CatmullRomCurve3 {
  const rng = mulberry32(seed);
  const all: THREE.Vector3[] = [points[0]];
  for (let i = 0; i < points.length - 1; i++) {
    const from = points[i];
    const to = points[i + 1];
    const dir = to.clone().sub(from);
    const dist = dir.length();
    dir.normalize();
    const arbitrary = Math.abs(dir.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
    const perp = new THREE.Vector3().crossVectors(dir, arbitrary).normalize();
    const bend = (rng() - 0.5) * dist * 0.5;
    const mid = from.clone().lerp(to, 0.5).add(perp.clone().multiplyScalar(bend));
    all.push(mid, to);
  }
  return new THREE.CatmullRomCurve3(all);
}

export default function IntroGlobalBrain({ mode, onRegionClick, width, height }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rotGroupRef = useRef<THREE.Group | null>(null);
  const contentGroupRef = useRef<THREE.Group | null>(null);
  const breathingRef = useRef<THREE.Sprite[]>([]);
  const pulseRef = useRef<Pulse | null>(null);
  const burstRef = useRef<Pulse | null>(null);
  const pauseUntilRef = useRef(0);
  const animFrameRef = useRef<number>(0);
  const widthRef = useRef(width);
  const heightRef = useRef(height);
  const pickTargetsRef = useRef<PickTarget[]>([]);
  const onRegionClickRef = useRef(onRegionClick);

  useEffect(() => { widthRef.current = width; heightRef.current = height; }, [width, height]);
  useEffect(() => { onRegionClickRef.current = onRegionClick; }, [onRegionClick]);

  // ── 可點擊目標:位置固定不變,不隨 mode 重算 ──────────────────
  const pickTargets = useMemo<PickTarget[]>(() => {
    const arr: PickTarget[] = ALL_SPHERE_NODES.map((n) => ({
      group: (n.kind === "feel" ? "feel" : "knowledge") as IntroBrainMode,
      pos: toWorld(n),
    }));
    arr.push({ group: "belief", pos: toWorld(BELIEF_NODE) });
    arr.push({ group: "action", pos: toWorld(ACTION_NODE) });

    // RC 節點本體 → 點了跳回「信念」;RC 旁的閃電徽章 → 點了跳到「行動」
    const rcWorld = toWorld(RC_NODE);
    arr.push({ group: "belief", pos: rcWorld });
    arr.push({
      group: "action",
      pos: rcWorld.clone().add(new THREE.Vector3(RC_BADGE_OFFSET, RC_BADGE_OFFSET, 0)),
    });
    return arr;
  }, []);
  useEffect(() => { pickTargetsRef.current = pickTargets; }, [pickTargets]);

  // ── Effect A: mount once — renderer/scene/camera/靜態星點/靜態
  //    neural mesh/事件監聽/render loop ──────────────────────────
  useEffect(() => {
    if (!canvasRef.current) return;

    const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true, alpha: true });
    renderer.setSize(widthRef.current, heightRef.current);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setClearColor(0x000000, 0);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(52, widthRef.current / Math.max(heightRef.current, 1), 0.01, 100);
    camera.position.set(0, 0.15, 3.9);
    cameraRef.current = camera;

    const rotGroup = new THREE.Group();
    scene.add(rotGroup);
    rotGroupRef.current = rotGroup;

    // 背景微光星點
    const bgCount = 400;
    const bgPos = new Float32Array(bgCount * 3);
    for (let i = 0; i < bgCount; i++) {
      bgPos[i * 3] = (Math.random() * 2 - 1) * 6;
      bgPos[i * 3 + 1] = (Math.random() * 2 - 1) * 6;
      bgPos[i * 3 + 2] = (Math.random() * 2 - 1) * 6;
    }
    const bgGeo = new THREE.BufferGeometry();
    bgGeo.setAttribute("position", new THREE.BufferAttribute(bgPos, 3));
    scene.add(new THREE.Points(bgGeo, new THREE.PointsMaterial({
      color: 0x2a3550, size: 0.01, transparent: true, opacity: 0.3, depthWrite: false,
    })));

    // ── 靜態 neural mesh:每個節點連到最近的 2 個鄰居,純裝飾,
    //    讓球體在任何 mode 下都讀得出「這是一個網絡」而不只是散點。
    const meshPositions = ALL_SPHERE_NODES.map((n) => toWorld(n));
    const meshLinePts: number[] = [];
    meshPositions.forEach((p, i) => {
      const nearest = meshPositions
        .map((q, j) => ({ j, d: i === j ? Infinity : p.distanceTo(q) }))
        .sort((a, b) => a.d - b.d)
        .slice(0, 2);
      nearest.forEach(({ j }) => {
        meshLinePts.push(p.x, p.y, p.z, meshPositions[j].x, meshPositions[j].y, meshPositions[j].z);
      });
    });
    const meshGeo = new THREE.BufferGeometry();
    meshGeo.setAttribute("position", new THREE.Float32BufferAttribute(meshLinePts, 3));
    rotGroup.add(new THREE.LineSegments(
      meshGeo,
      new THREE.LineBasicMaterial({ color: 0x3a4868, transparent: true, opacity: 0.18 }),
    ));

    scene.add(new THREE.AmbientLight(0xffffff, 1));

    const contentGroup = new THREE.Group();
    rotGroup.add(contentGroup);
    contentGroupRef.current = contentGroup;

    let lastT = 0;
    function animate(now: number) {
      animFrameRef.current = requestAnimationFrame(animate);
      if (now - lastT < 1000 / 30) return;
      lastT = now;

      const spinning = now > pauseUntilRef.current;
      rotGroup.rotation.y += spinning ? ROTATE_SPEED : 0;
      rotGroup.rotation.x = Math.sin(now * 0.00008) * 0.08;

      breathingRef.current.forEach((spr) => {
        const pulse = 0.55 + 0.45 * Math.sin(now * 0.0025);
        (spr.material as THREE.SpriteMaterial).opacity = pulse;
      });

      // 常駐環境脈衝(慢速循環,象徵這條路徑目前「已啟用」)
      if (pulseRef.current) {
        const p = pulseRef.current;
        p.phase = (p.phase + 0.006) % 1;
        p.dot.position.copy(p.curve.getPointAt(p.phase));
        (p.dot.material as THREE.SpriteMaterial).opacity = 1 - Math.abs(p.phase - 0.5) * 1.2;
      }

      // 送出脈衝(一次性、較亮較快,象徵閃電剛被按下的瞬間)
      if (burstRef.current) {
        const b = burstRef.current;
        if (b.phase < 1) {
          b.phase = Math.min(1, b.phase + 0.022);
          b.dot.position.copy(b.curve.getPointAt(b.phase));
          (b.dot.material as THREE.SpriteMaterial).opacity = 0.35 + 0.65 * Math.sin(b.phase * Math.PI);
        } else {
          (b.dot.material as THREE.SpriteMaterial).opacity *= 0.9;
        }
      }

      renderer.render(scene, camera);
    }
    animFrameRef.current = requestAnimationFrame(animate);

    // ── 點擊:螢幕空間找最近的可點擊目標 ─────────────────────
    function onClick(e: MouseEvent) {
      const rot = rotGroupRef.current;
      const cam = cameraRef.current;
      if (!rot || !cam || !canvasRef.current) return;
      const r = canvasRef.current.getBoundingClientRect();
      const px = e.clientX - r.left;
      const py = e.clientY - r.top;

      let bestGroup: IntroBrainMode | null = null;
      let bestDist = PICK_PIXEL_RADIUS * PICK_PIXEL_RADIUS;
      pickTargetsRef.current.forEach((t) => {
        const wp = t.pos.clone();
        rot.localToWorld(wp);
        const proj = wp.project(cam);
        if (Math.abs(proj.z) > 1) return;
        const sx = ((proj.x + 1) / 2) * widthRef.current;
        const sy = ((1 - proj.y) / 2) * heightRef.current;
        const dx = sx - px, dy = sy - py;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDist) { bestDist = d2; bestGroup = t.group; }
      });
      if (bestGroup) onRegionClickRef.current?.(bestGroup);
    }

    const canvas = canvasRef.current;
    canvas.addEventListener("click", onClick);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      canvas.removeEventListener("click", onClick);
      disposeObject3D(scene);
      renderer.dispose();
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      rotGroupRef.current = null;
      contentGroupRef.current = null;
      breathingRef.current = [];
      pulseRef.current = null;
      burstRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Effect B: resize ──
  useEffect(() => {
    const renderer = rendererRef.current;
    const camera = cameraRef.current;
    if (!renderer || !camera || width <= 0 || height <= 0) return;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }, [width, height]);

  // ── Effect C: mode 變化 — 重建節點/連結線,並讓自轉暫停一下 ──────
  useEffect(() => {
    const contentGroup = contentGroupRef.current;
    if (!contentGroup) return;

    disposeGroupChildren(contentGroup);
    contentGroup.clear();
    breathingRef.current = [];
    pulseRef.current = null;
    burstRef.current = null;
    pauseUntilRef.current = performance.now() + ROTATE_PAUSE_MS;

    const nodeById = new Map<string, DemoNode>(ALL_SPHERE_NODES.map((n) => [n.id, n]));
    const foundationSet = new Set(FOUNDATION_IDS);

    // ── 球體上的一般節點(感受 + 知識)─────────────────────────
    ALL_SPHERE_NODES.forEach((n) => {
      const isFeelActive = mode === "feel" && n.kind === "feel";
      const isKnowledgeActive = mode === "knowledge" && n.kind === "knowledge";
      const isFoundationActive = mode === "belief" && foundationSet.has(n.id);
      const active = isFeelActive || isKnowledgeActive || isFoundationActive;

      let color = BASE_COLOR;
      if (isFeelActive) color = FEEL_COLOR;
      else if (isKnowledgeActive) color = KNOWLEDGE_COLOR;
      else if (isFoundationActive) color = BELIEF_COLOR;

      const pos = toWorld(n);
      const hex = `#${color.getHexString()}`;
      const dot = makeFlatDotSprite(hex, active ? "#ffffff" : BASE_STROKE);
      const scale = active ? 0.13 : 0.075;
      dot.scale.set(scale, scale, 1);
      dot.position.copy(pos);
      dot.material.opacity = active ? 1 : BASE_OPACITY;
      contentGroup.add(dot);

      if (active) {
        const glow = makeGlowSprite(hex);
        glow.position.copy(pos);
        glow.scale.set(0.42, 0.42, 1);
        contentGroup.add(glow);

        const label = makeLabelSprite(n.label, hex);
        label.position.set(pos.x, pos.y + 0.2, pos.z);
        label.scale.set(0.56, 0.12, 1);
        contentGroup.add(label);
      }
    });

    const beliefPos = toWorld(BELIEF_NODE);
    const rcPos = toWorld(RC_NODE);
    const actionPos = toWorld(ACTION_NODE);
    const beliefActive = mode === "belief";
    const beliefSecondary = mode === "action";
    const actionActive = mode === "action";

    // ── 信念節點(球體上永遠存在)─────────────────────────────
    {
      const active = beliefActive || beliefSecondary;
      const dot = makeLightningNodeSprite(
        actionActive ? "#ffcc4d" : `#${BASE_COLOR.getHexString()}`,
        actionActive ? "#ffffff" : BASE_STROKE,
      );
      const scale = actionActive ? 0.26 : 0.12;   // ← 閃電形狀視覺重心偏移，需要更大 scale 才夠醒目
      dot.scale.set(scale, scale, 1);
      dot.position.copy(beliefPos);
      dot.material.opacity = beliefActive ? 1 : beliefSecondary ? 0.85 : BASE_OPACITY;
      contentGroup.add(dot);

      if (active) {
        const glow = makeGlowSprite("#ff5fd1");
        glow.position.copy(beliefPos);
        const glowScale = beliefActive ? 0.85 : 0.55;
        glow.scale.set(glowScale, glowScale, 1);
        contentGroup.add(glow);
        breathingRef.current.push(glow);

        const label = makeLabelSprite(BELIEF_NODE.label, "#ffd8f2");
        label.position.set(beliefPos.x, beliefPos.y + (beliefActive ? 0.24 : 0.2), beliefPos.z);
        label.scale.set(0.62, 0.13, 1);
        contentGroup.add(label);
      }
    }

    // ── RC 節點(球體上永遠存在;belief/action 模式時展開為主要節點,
    //    並掛上閃電徽章 — 送入 MainAgent 的開關)────────────────
    {
      const rcVisible = mode === "belief" || mode === "action";
      const dot = makeSquareDotSprite(
        rcVisible ? RC_COLOR_HEX : `#${BASE_COLOR.getHexString()}`,
        rcVisible ? "#ffffff" : BASE_STROKE,
      );
      const scale = rcVisible ? 0.22 : 0.11;
      dot.scale.set(scale, scale, 1);
      dot.position.copy(rcPos);
      dot.material.opacity = rcVisible ? 1 : BASE_OPACITY;
      contentGroup.add(dot);

      if (rcVisible) {
        const glow = makeGlowSprite(RC_COLOR_HEX);
        glow.position.copy(rcPos);
        glow.scale.set(0.55, 0.55, 1);
        contentGroup.add(glow);
        breathingRef.current.push(glow);

        const label = makeLabelSprite(RC_NODE.label, "#ffd9a0");
        label.position.set(rcPos.x, rcPos.y + 0.22, rcPos.z);
        label.scale.set(0.66, 0.13, 1);
        contentGroup.add(label);

        // 閃電:送入 MainAgent 的開關。action 模式時代表「已按下」。
        const firing = mode === "action";
        const badge = makeBadgeSprite(firing ? "#ffffff" : RC_COLOR_HEX, firing ? RC_COLOR_HEX : "#fff3c4");
        badge.scale.set(0.14, 0.14, 1);
        badge.position.set(rcPos.x + RC_BADGE_OFFSET, rcPos.y + RC_BADGE_OFFSET, rcPos.z);
        contentGroup.add(badge);

        if (firing) {
          const bglow = makeGlowSprite("#ffcc4d");
          bglow.position.copy(badge.position);
          bglow.scale.set(0.42, 0.42, 1);
          contentGroup.add(bglow);
          breathingRef.current.push(bglow);
        }
      }
    }

    // ── 行動節點(球體上永遠存在)─────────────────────────────
    {
      const dot = makeFlatDotSprite(
        actionActive ? "#ffcc4d" : `#${BASE_COLOR.getHexString()}`,
        actionActive ? "#ffffff" : BASE_STROKE,
      );
      const scale = actionActive ? 0.14 : 0.075;
      dot.scale.set(scale, scale, 1);
      dot.position.copy(actionPos);
      dot.material.opacity = actionActive ? 1 : BASE_OPACITY;
      contentGroup.add(dot);

      if (actionActive) {
        const glow = makeGlowSprite("#ffcc4d");
        glow.position.copy(actionPos);
        glow.scale.set(0.55, 0.55, 1);
        contentGroup.add(glow);
        breathingRef.current.push(glow);

        const label = makeLabelSprite(ACTION_NODE.label, "#fff3c4");
        label.position.set(actionPos.x, actionPos.y + 0.22, actionPos.z);
        label.scale.set(0.62, 0.13, 1);
        contentGroup.add(label);
      }
    }

    // ── 信念模式:foundation → belief 的曲折收斂線,
    //    以及 belief → RC 的預告連結(虛線,尚未送出)────────────
    if (mode === "belief") {
      FOUNDATION_IDS.forEach((id) => {
        const n = nodeById.get(id);
        if (!n) return;
        const curve = buildWindingPath([toWorld(n), beliefPos], hashStr(id));
        const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(30));
        const mat = new THREE.LineDashedMaterial({
          color: 0xff5fd1, dashSize: 0.05, gapSize: 0.04, transparent: true, opacity: 0.5,
        });
        const line = new THREE.Line(geo, mat);
        line.computeLineDistances();
        contentGroup.add(line);
      });

      const previewPath = buildWindingPath([beliefPos, rcPos], hashStr("belief-rc-preview"));
      const previewGeo = new THREE.BufferGeometry().setFromPoints(previewPath.getPoints(24));
      const previewMat = new THREE.LineDashedMaterial({
        color: 0xf0a83c, dashSize: 0.05, gapSize: 0.05, transparent: true, opacity: 0.45,
      });
      const previewLine = new THREE.Line(previewGeo, previewMat);
      previewLine.computeLineDistances();
      contentGroup.add(previewLine);
    }

    // ── 行動模式:belief → RC → action 的曲折路徑 + 送出脈衝 ────
    if (mode === "action") {
      const sendPath = buildWindingPath([beliefPos, rcPos, actionPos], hashStr("belief-rc-action"));
      const geo = new THREE.BufferGeometry().setFromPoints(sendPath.getPoints(48));
      contentGroup.add(new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ color: 0xffcc4d, transparent: true, opacity: 0.75 }),
      ));

      // 常駐環境光點(慢速循環,象徵這條路徑「已啟用」)
      const ambientDot = makeFlatDotSprite("#ffcc4d");
      ambientDot.scale.set(0.06, 0.06, 1);
      contentGroup.add(ambientDot);
      pulseRef.current = { curve: sendPath, dot: ambientDot, phase: 0 };

      // 送出脈衝:一次性、更亮更快,象徵閃電剛被按下的瞬間
      const burstDot = makeGlowSprite("#fff3c4");
      burstDot.scale.set(0.3, 0.3, 1);
      contentGroup.add(burstDot);
      burstRef.current = { curve: sendPath, dot: burstDot, phase: 0 };
    }
  }, [mode]);

  return (
    <div style={{ position: "relative", width, height }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block", width: "100%", height: "100%", cursor: "pointer" }}
      />
      <div style={{
        position: "absolute", bottom: 10, left: 12,
        fontSize: "0.6rem", color: "rgba(232,238,248,0.32)",
        letterSpacing: "0.06em", fontFamily: "monospace",
        pointerEvents: "none", userSelect: "none",
      }}>
        點擊球體上的區域,切換視角 · ▲方塊=推薦方案 ▲大閃電=行動,可直接點擊聚焦
      </div>
    </div>
  );
}
