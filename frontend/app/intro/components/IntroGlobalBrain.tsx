"use client";

/**
 * IntroGlobalBrain.tsx — /intro 頁面的簡化版 3D 球體(單一場景版)。
 *
 * 與前一版的差異:
 *   - 不再有獨立的 IntroHourglass 場景。belief / action 現在是這顆球體
 *     座標系裡兩個真實存在的節點(見 demoData.ts 的 x/y/z),四個分頁
 *     (感受/知識/信念/行動)共用同一顆持續自轉的球體,差別只在於
 *     「目前 highlight 哪一群」。
 *   - 所有節點永遠以統一的「基底色」正常顯示(不再把非選中節點壓到
 *     快看不見),球體任何時刻都看得出是一個完整、有網絡感的結構;
 *     選中的那一群才升級成該分頁的 accent 色 + glow + label。
 *   - 新增可點擊:點擊任一群節點(不論目前是否 highlight)都能透過
 *     onRegionClick 通知外部切換分頁,採用與正式系統
 *     GlobalBrainExplorer/HourglassViz 一致的「螢幕空間最近節點」
 *     pick 手法。純點擊,不做 hover 提示。
 *   - 切換分頁(不論是點擊或 autoplay 觸發)時,球體自轉會短暫暫停
 *     ~2.2 秒讓使用者好好看清楚,之後恢復緩慢自轉。
 *   - foundation → belief 的收斂線,以及 belief → action 的連結線,
 *     改用 CatmullRomCurve3 產生的「曲折路徑」,而不是直線,視覺上
 *     更像神經傳導路徑。
 */

import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import {
  makeGlowSprite,
  makeFlatDotSprite,
  makeLabelSprite,
  disposeObject3D,
  disposeGroupChildren,
  hashStr,
  mulberry32,
} from "../../components/three-shared";
import {
  ALL_SPHERE_NODES,
  FOUNDATION_IDS,
  BELIEF_NODE,
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

const WORLD_SCALE = 2.4;

// ── 基底色:所有節點預設用這個,亮度足以看出整顆球體的形狀 ──────
const BASE_COLOR = new THREE.Color(0x6b7aa0);
const BASE_OPACITY = 0.65;
const BASE_STROKE = "#8a97b8";

const FEEL_COLOR = new THREE.Color(0x00e8ff);
const KNOWLEDGE_COLOR = new THREE.Color(0xff9a52);
const BELIEF_COLOR = new THREE.Color(0xff5fd1);
const ACTION_COLOR = new THREE.Color(0xffcc4d);

const PICK_PIXEL_RADIUS = 26;
const ROTATE_SPEED = 0.0012;
const ROTATE_PAUSE_MS = 2200;

function toWorld(n: { x: number; y: number; z: number }): THREE.Vector3 {
  return new THREE.Vector3(n.x * WORLD_SCALE, n.y * WORLD_SCALE, n.z * WORLD_SCALE);
}

/** 產生一條有兩個彎折點的曲線,取代直線,視覺上更像神經傳導路徑。
 *  bend 的方向與幅度由 seed 決定,同一對節點每次 render 結果一致。 */
function buildWindingCurve(from: THREE.Vector3, to: THREE.Vector3, seed: number): THREE.CatmullRomCurve3 {
  const rng = mulberry32(seed);
  const dir = to.clone().sub(from);
  const dist = dir.length();
  dir.normalize();
  const arbitrary = Math.abs(dir.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  const perp = new THREE.Vector3().crossVectors(dir, arbitrary).normalize();

  const bend1 = (rng() - 0.5) * dist * 0.55;
  const bend2 = (rng() - 0.5) * dist * 0.55;

  const p1 = from.clone().lerp(to, 0.33).add(perp.clone().multiplyScalar(bend1));
  const p2 = from.clone().lerp(to, 0.67).add(perp.clone().multiplyScalar(bend2));

  return new THREE.CatmullRomCurve3([from, p1, p2, to]);
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

    const camera = new THREE.PerspectiveCamera(50, widthRef.current / Math.max(heightRef.current, 1), 0.01, 100);
    camera.position.set(0, 0.2, 4.4);
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

      if (pulseRef.current) {
        const p = pulseRef.current;
        p.phase = (p.phase + 0.006) % 1;
        p.dot.position.copy(p.curve.getPointAt(p.phase));
        (p.dot.material as THREE.SpriteMaterial).opacity = 1 - Math.abs(p.phase - 0.5) * 1.2;
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
      const scale = active ? 0.09 : 0.055;
      dot.scale.set(scale, scale, 1);
      dot.position.copy(pos);
      dot.material.opacity = active ? 1 : BASE_OPACITY;
      contentGroup.add(dot);

      if (active) {
        const glow = makeGlowSprite(hex);
        glow.position.copy(pos);
        glow.scale.set(0.32, 0.32, 1);
        contentGroup.add(glow);

        const label = makeLabelSprite(n.label, hex);
        label.position.set(pos.x, pos.y + 0.18, pos.z);
        label.scale.set(0.5, 0.11, 1);
        contentGroup.add(label);
      }
    });

    const beliefPos = toWorld(BELIEF_NODE);
    const actionPos = toWorld(ACTION_NODE);
    const beliefActive = mode === "belief";
    const beliefSecondary = mode === "action";
    const actionActive = mode === "action";

    // ── 信念節點(球體上永遠存在)─────────────────────────────
    {
      const active = beliefActive || beliefSecondary;
      const dot = makeFlatDotSprite(
        active ? "#ff5fd1" : `#${BASE_COLOR.getHexString()}`,
        active ? "#ffffff" : BASE_STROKE,
      );
      const scale = beliefActive ? 0.13 : beliefSecondary ? 0.09 : 0.06;
      dot.scale.set(scale, scale, 1);
      dot.position.copy(beliefPos);
      dot.material.opacity = beliefActive ? 1 : beliefSecondary ? 0.85 : BASE_OPACITY;
      contentGroup.add(dot);

      if (active) {
        const glow = makeGlowSprite("#ff5fd1");
        glow.position.copy(beliefPos);
        const glowScale = beliefActive ? 0.7 : 0.45;
        glow.scale.set(glowScale, glowScale, 1);
        contentGroup.add(glow);
        breathingRef.current.push(glow);

        const label = makeLabelSprite(BELIEF_NODE.label, "#ffd8f2");
        label.position.set(beliefPos.x, beliefPos.y + (beliefActive ? 0.22 : 0.18), beliefPos.z);
        label.scale.set(0.58, 0.13, 1);
        contentGroup.add(label);
      }
    }

    // ── 行動節點(球體上永遠存在)─────────────────────────────
    {
      const dot = makeFlatDotSprite(
        actionActive ? "#ffcc4d" : `#${BASE_COLOR.getHexString()}`,
        actionActive ? "#ffffff" : BASE_STROKE,
      );
      const scale = actionActive ? 0.1 : 0.055;
      dot.scale.set(scale, scale, 1);
      dot.position.copy(actionPos);
      dot.material.opacity = actionActive ? 1 : BASE_OPACITY;
      contentGroup.add(dot);

      if (actionActive) {
        const glow = makeGlowSprite("#ffcc4d");
        glow.position.copy(actionPos);
        glow.scale.set(0.45, 0.45, 1);
        contentGroup.add(glow);
        breathingRef.current.push(glow);

        const label = makeLabelSprite(ACTION_NODE.label, "#fff3c4");
        label.position.set(actionPos.x, actionPos.y + 0.2, actionPos.z);
        label.scale.set(0.58, 0.13, 1);
        contentGroup.add(label);
      }
    }

    // ── 信念模式:foundation → belief 的曲折收斂線 ────────────
    if (mode === "belief") {
      FOUNDATION_IDS.forEach((id) => {
        const n = nodeById.get(id);
        if (!n) return;
        const curve = buildWindingCurve(toWorld(n), beliefPos, hashStr(id));
        const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(30));
        const mat = new THREE.LineDashedMaterial({
          color: 0xff5fd1, dashSize: 0.05, gapSize: 0.04, transparent: true, opacity: 0.5,
        });
        const line = new THREE.Line(geo, mat);
        line.computeLineDistances();
        contentGroup.add(line);
      });
    }

    // ── 行動模式:belief → action 的曲折連結線 + 接力光點 ────
    if (mode === "action") {
      const curve = buildWindingCurve(beliefPos, actionPos, hashStr("belief-action"));
      const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(30));
      contentGroup.add(new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ color: 0xffcc4d, transparent: true, opacity: 0.75 }),
      ));

      const pulseDot = makeFlatDotSprite("#ffcc4d");
      pulseDot.scale.set(0.05, 0.05, 1);
      contentGroup.add(pulseDot);
      pulseRef.current = { curve, dot: pulseDot, phase: 0 };
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
        點擊球體上的區域,切換視角
      </div>
    </div>
  );
}