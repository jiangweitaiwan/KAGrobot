"use client";

/**
 * PersonalBrain.tsx
 *
 * 職責：
 *   1. 渲染 personal foundations 的 fMRI 疊加（3D 密度雲 + 紫色光暈）
 *   2. 顯示根據 score(RC) 算出的 Top-3 RC 標籤
 *   3. 點選 RC 標籤 → 亮出對應 downstream path（從 /graph/rc-paths/{eid} 取得）
 *   4. 確認 → 送入 MainAgent
 *
 * 接收整個 globalLayout（共享渲染基礎），不需要重新 build 密度雲。
 *
 * ── 效能優化（重點）──────────────────────────────────────────
 *   1. 移除自動旋轉：場景不再持續自轉，改為純使用者驅動。
 *   2. 按需渲染（on-demand rendering）：只有在「真的有東西在動」
 *      （拖曳中、fMRI 呼吸動畫、path glow 動畫）或畫面被標記為
 *      dirty 時才送一幀進 GPU，其餘時間迴圈直接 return，不渲染。
 *   3. 可見性偵測：分頁切到背景（visibilitychange）或面板捲出
 *      可視範圍（IntersectionObserver）時，完全暫停渲染。
 *   4. 渲染幀率上限 ~24fps（原本跟著 rAF 跑滿 60fps）。
 *   5. devicePixelRatio 上限降至 1.25（原本 2）。
 *   6. Bloom 後製採用「降解析度版本」：UnrealBloomPass 的取樣
 *      解析度改用 width/height 的 0.6 倍建構，而非全解析度。
 *      因為 bloom 本質是模糊效果，降低取樣解析度視覺上幾乎無感，
 *      但能大幅降低多重降採樣＋高斯模糊的 fill-rate 成本。
 *   7. Cleanup 時完整 dispose 所有 geometry / material / texture，
 *      避免面板 resize 觸發場景重建時造成 GPU 記憶體持續累積。
 */

import { useEffect, useRef, useMemo, useState, useCallback } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import type { GlobalLayout, GlobalLayoutNode } from "./GlobalBrainExplorer";
import type { PersonalBrainConcept } from "./DetailPanel";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type RCRankedItem = {
  eid:        string;
  label:      string;
  totalScore: number;
};

export type RCPathData = {
  eid:   string;
  label: string;
  nodes: { id: string; label: string; node_type: string; matched: boolean }[];
  edges: { id: string; source: string; target: string }[];
  paths: { path_id: number; node_ids: string[]; edge_ids: string[]; final_label: string }[];
};

type Props = {
  layout:        GlobalLayout | null;
  personalBrain: PersonalBrainConcept[];
  rcRanking:     RCRankedItem[];           // Top-3 from page.tsx
  onConfirmRC:   (rcEid: string, rcLabel: string, pathId: number) => void;
  fetchRCPaths:  (eid: string) => Promise<RCPathData | null>;
  width?:        number;
  height?:       number;
};

// ─────────────────────────────────────────────────────────────
// Design tokens (distinct from GlobalBrainExplorer — warmer purple)
// ─────────────────────────────────────────────────────────────

const C = {
  bg:     0x030108,
  bg_css: "#030108",

  cloud_rc:    new THREE.Color(0xc060ff),
  cloud_other: new THREE.Color(0x7020b0),
  fmri:        new THREE.Color(0x7f77dd),
  path_active: new THREE.Color(0x4f8ef7),

  bloom_strength:  2.6,
  bloom_radius:    0.55,
  bloom_threshold: 0.18,
};

const CLOUD = { particles_per: 300, spread: 0.07, size: 0.014 };
const WORLD_SCALE     = 2.2;
const PROXY_THRESHOLD = 0.12;

const TAG_COLORS = ["#c060ff", "#7f77dd", "#534AB7"];

// ── 效能參數 ─────────────────────────────────────────────────
const MAX_PIXEL_RATIO  = 1.25;       // 原本 2，降低 fill-rate 成本
const RENDER_FRAME_MS  = 1000 / 24;  // 渲染幀率上限 ~24fps
const BLOOM_RES_SCALE  = 0.6;        // Bloom 降解析度版本：取樣解析度 ×0.6

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function nodeToWorld(n: GlobalLayoutNode) {
  return new THREE.Vector3(n.x * WORLD_SCALE, n.y * WORLD_SCALE, n.z * WORLD_SCALE);
}

function seededRand(seed: number) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
}

function buildCloud(nodes: GlobalLayoutNode[], brainEids: Set<string>) {
  const filtered = nodes.filter((n) => brainEids.has(n.id));
  const total    = filtered.length * CLOUD.particles_per;
  const positions = new Float32Array(total * 3);
  const colors    = new Float32Array(total * 3);
  const sizes     = new Float32Array(total);
  const nodeIndex = new Int32Array(total);

  let ptr = 0;
  filtered.forEach((node, ni) => {
    const isRC   = (node.type || "").toLowerCase().includes("resulting");
    const base   = isRC ? C.cloud_rc : C.cloud_other;
    const rand   = seededRand(ni * 7919 + 3);
    const center = nodeToWorld(node);
    for (let i = 0; i < CLOUD.particles_per; i++) {
      const theta = rand() * Math.PI * 2;
      const phi   = Math.acos(2 * rand() - 1);
      const r     = CLOUD.spread * Math.cbrt(rand());
      positions[ptr * 3 + 0] = center.x + r * Math.sin(phi) * Math.cos(theta);
      positions[ptr * 3 + 1] = center.y + r * Math.sin(phi) * Math.sin(theta);
      positions[ptr * 3 + 2] = center.z + r * Math.cos(phi);
      colors[ptr * 3 + 0] = base.r;
      colors[ptr * 3 + 1] = base.g;
      colors[ptr * 3 + 2] = base.b;
      sizes[ptr]     = CLOUD.size * (0.5 + rand() * 0.8);
      nodeIndex[ptr] = ni;
      ptr++;
    }
  });
  return { positions, colors, sizes, nodeIndex, filteredNodes: filtered };
}

function buildFMRITexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const ctx  = canvas.getContext("2d")!;
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0,    "rgba(127,119,221,0.95)");
  grad.addColorStop(0.35, "rgba(83,74,183,0.55)");
  grad.addColorStop(0.7,  "rgba(38,33,92,0.20)");
  grad.addColorStop(1,    "rgba(38,33,92,0.0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(canvas);
}

/** 完整釋放單一 Object3D 的 geometry / material / texture（含子節點）。*/
function disposeObject3D(root: THREE.Object3D) {
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

// ─────────────────────────────────────────────────────────────
// PersonalBrain
// ─────────────────────────────────────────────────────────────

export default function PersonalBrain({
  layout,
  personalBrain,
  rcRanking,
  onConfirmRC,
  fetchRCPaths,
  width  = 480,
  height = 480,
}: Props) {
  const canvasRef     = useRef<HTMLCanvasElement | null>(null);
  const rendererRef   = useRef<THREE.WebGLRenderer | null>(null);
  const composerRef   = useRef<EffectComposer | null>(null);
  const cameraRef     = useRef<THREE.PerspectiveCamera | null>(null);
  const cloudMeshRef  = useRef<THREE.Points | null>(null);
  const fmriGroupRef  = useRef<THREE.Group | null>(null);
  const pathGroupRef  = useRef<THREE.Group | null>(null);
  const animFrameRef  = useRef<number>(0);
  const rotGroup      = useRef<THREE.Group | null>(null);
  const isDragging    = useRef(false);
  const prevMouse     = useRef({ x: 0, y: 0 });
  const fmriTexRef    = useRef<THREE.CanvasTexture | null>(null);

  // ── 效能：可見性 / 按需渲染狀態 ──────────────────────────
  const isVisibleRef   = useRef(true);   // 分頁可見 && 面板在視窗內
  const needsRenderRef = useRef(true);   // 畫面是否為 dirty，需要重繪一次
  const lastFrameTsRef = useRef(0);      // 上一次實際渲染的時間戳

  // RC tag state
  const [selectedRCEid,   setSelectedRCEid]   = useState<string | null>(null);
  const [selectedPathId,  setSelectedPathId]   = useState<number | null>(null);
  const [rcPathData,      setRCPathData]       = useState<RCPathData | null>(null);
  const [loadingRC,       setLoadingRC]        = useState(false);
  const [pulseEids,       setPulseEids]        = useState<string[]>([]); // for fMRI pulse effect

  const brainEids = useMemo(
    () => new Set(personalBrain.map((c) => c.eid)),
    [personalBrain],
  );
  const weightMap = useMemo(
    () => new Map(personalBrain.map((c) => [c.eid, c.weight])),
    [personalBrain],
  );

  const nodeById = useMemo(() => {
    if (!layout) return new Map<string, GlobalLayoutNode>();
    return new Map(layout.nodes.map((n) => [n.id, n]));
  }, [layout]);

  // ── fMRI pulse when brain changes ────────────────────────
  const prevBrainRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    const prev = prevBrainRef.current;
    const newEids = [...brainEids].filter((e) => !prev.has(e));
    if (newEids.length > 0) {
      setPulseEids(newEids);
      const t = setTimeout(() => setPulseEids([]), 1800);
      prevBrainRef.current = new Set(brainEids);
      return () => clearTimeout(t);
    }
    prevBrainRef.current = new Set(brainEids);
  }, [brainEids]);

  // ── Init scene ────────────────────────────────────────────
  useEffect(() => {
    if (!canvasRef.current || !layout || layout.nodes.length === 0) return;

    const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
    renderer.setClearColor(C.bg, 1);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 100);
    camera.position.set(0, 0, 5.5);
    cameraRef.current = camera;

    const rg = new THREE.Group();
    scene.add(rg);
    rotGroup.current = rg;

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    // ── Bloom：降解析度版本 ───────────────────────────────────
    // bloom 本質是模糊效果，取樣解析度降低視覺上幾乎無感，
    // 但能省下大量多重降採樣＋高斯模糊的 fill-rate 成本。
    const bloomRes = new THREE.Vector2(
      Math.max(1, Math.round(width  * BLOOM_RES_SCALE)),
      Math.max(1, Math.round(height * BLOOM_RES_SCALE)),
    );
    const bloomPass = new UnrealBloomPass(bloomRes, C.bloom_strength, C.bloom_radius, C.bloom_threshold);
    composer.addPass(bloomPass);
    composerRef.current = composer;

    // Background dim stars
    const bgPos  = new Float32Array(5000 * 3);
    const bgRand = seededRand(99);
    for (let i = 0; i < 5000; i++) {
      bgPos[i * 3 + 0] = (bgRand() * 2 - 1) * 3.5;
      bgPos[i * 3 + 1] = (bgRand() * 2 - 1) * 3.5;
      bgPos[i * 3 + 2] = (bgRand() * 2 - 1) * 3.5;
    }
    const bgGeo = new THREE.BufferGeometry();
    bgGeo.setAttribute("position", new THREE.BufferAttribute(bgPos, 3));
    const bgMat = new THREE.PointsMaterial({ color: 0x0a0020, size: 0.004, transparent: true, opacity: 0.3, depthWrite: false });
    scene.add(new THREE.Points(bgGeo, bgMat));

    // cloud (only brain nodes)
    const { positions, colors, sizes, nodeIndex } = buildCloud(layout.nodes, brainEids);
    if (positions.length > 0) {
      const cloudGeo = new THREE.BufferGeometry();
      cloudGeo.setAttribute("position",  new THREE.BufferAttribute(positions.slice(), 3));
      cloudGeo.setAttribute("color",     new THREE.BufferAttribute(colors.slice(),    3));
      cloudGeo.setAttribute("size",      new THREE.BufferAttribute(sizes.slice(),     1));
      cloudGeo.setAttribute("nodeIndex", new THREE.BufferAttribute(nodeIndex,         1));

      const cloudMat = new THREE.ShaderMaterial({
        vertexShader: `attribute float size; attribute vec3 color; varying vec3 vColor;
          void main() { vColor = color; vec4 mvPos = modelViewMatrix * vec4(position,1.0);
            gl_PointSize = size * (300.0 / -mvPos.z); gl_Position = projectionMatrix * mvPos; }`,
        fragmentShader: `varying vec3 vColor;
          void main() { vec2 uv = gl_PointCoord - vec2(0.5); float d = length(uv);
            if (d > 0.5) discard; float a = smoothstep(0.5, 0.1, d) * 0.9;
            gl_FragColor = vec4(vColor, a); }`,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      });
      const cloudMesh = new THREE.Points(cloudGeo, cloudMat);
      rg.add(cloudMesh);
      cloudMeshRef.current = cloudMesh;
    }

    fmriTexRef.current = buildFMRITexture();

    // fMRI group
    const fmriGroup = new THREE.Group();
    rg.add(fmriGroup);
    fmriGroupRef.current = fmriGroup;

    // path group (lines for selected RC paths)
    const pathGroup = new THREE.Group();
    rg.add(pathGroup);
    pathGroupRef.current = pathGroup;

    scene.add(new THREE.AmbientLight(0xffffff, 1));

    // 場景剛建立，標記需要畫第一幀
    needsRenderRef.current = true;

    // ── Animation loop（按需渲染版）────────────────────────────
    // 不再自動旋轉。每幀只做：
    //   1) 可見性檢查 → 不可見直接 return
    //   2) 幀率上限節流（~24fps）
    //   3) fMRI 呼吸動畫（仍在動 → 需要持續渲染）
    //   4) 只有「畫面真的有變化」才呼叫 composer.render()
    let frameCount = 0;
    function animate(now: number) {
      animFrameRef.current = requestAnimationFrame(animate);

      if (!isVisibleRef.current) return;
      if (now - lastFrameTsRef.current < RENDER_FRAME_MS) return;
      lastFrameTsRef.current = now;
      frameCount++;

      // pulse fMRI sprites（仍在動 → 需要持續渲染）
      let hasPulsingFmri = false;
      if (fmriGroupRef.current && fmriGroupRef.current.children.length > 0) {
        hasPulsingFmri = true;
        const t = frameCount * 0.01;
        fmriGroupRef.current.children.forEach((child: THREE.Object3D) => {
          if (child instanceof THREE.Sprite) {
            child.material.opacity = child.userData.baseOpacity * (0.85 + 0.15 * Math.sin(t + child.userData.phase));
          }
        });
      }

      if (needsRenderRef.current || isDragging.current || hasPulsingFmri) {
        composerRef.current?.render();
        needsRenderRef.current = false;
      }
    }
    animFrameRef.current = requestAnimationFrame(animate);

    // drag to rotate
    function onMouseDown(e: MouseEvent) {
      isDragging.current = true;
      prevMouse.current = { x: e.clientX, y: e.clientY };
      needsRenderRef.current = true;
    }
    function onMouseUp()  { isDragging.current = false; needsRenderRef.current = true; }
    function onMouseMove(e: MouseEvent) {
      if (!isDragging.current || !rotGroup.current) return;
      rotGroup.current.rotation.y += (e.clientX - prevMouse.current.x) * 0.008;
      rotGroup.current.rotation.x += (e.clientY - prevMouse.current.y) * 0.008;
      prevMouse.current = { x: e.clientX, y: e.clientY };
      needsRenderRef.current = true;
    }
    function onTouchStart(e: TouchEvent) {
      isDragging.current = true;
      prevMouse.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      needsRenderRef.current = true;
    }
    function onTouchMove(e: TouchEvent) {
      if (!isDragging.current || !rotGroup.current) return;
      rotGroup.current.rotation.y += (e.touches[0].clientX - prevMouse.current.x) * 0.008;
      rotGroup.current.rotation.x += (e.touches[0].clientY - prevMouse.current.y) * 0.008;
      prevMouse.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      needsRenderRef.current = true;
    }
    canvasRef.current.addEventListener("mousedown",  onMouseDown);
    canvasRef.current.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("mouseup",   onMouseUp);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("touchmove", onTouchMove, { passive: true });
    window.addEventListener("touchend",  onMouseUp);

    // ── 可見性偵測：分頁背景 / 面板捲出視窗外時暫停渲染 ──────────
    function onVisibilityChange() {
      if (document.hidden) {
        isVisibleRef.current = false;
      } else {
        isVisibleRef.current = true;
        needsRenderRef.current = true; // 回到前景時補畫一幀
      }
    }
    document.addEventListener("visibilitychange", onVisibilityChange);

    let io: IntersectionObserver | null = null;
    if (typeof IntersectionObserver !== "undefined" && canvasRef.current) {
      io = new IntersectionObserver(
        ([entry]) => {
          isVisibleRef.current = entry.isIntersecting && !document.hidden;
          if (isVisibleRef.current) needsRenderRef.current = true;
        },
        { threshold: 0.05 },
      );
      io.observe(canvasRef.current);
    }

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener("mouseup",   onMouseUp);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend",  onMouseUp);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      io?.disconnect();

      // ── 完整釋放 GPU 資源，避免 resize 觸發重建時持續累積 ──────
      disposeObject3D(scene);
      fmriTexRef.current?.dispose();
      renderer.dispose();
      composer.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, width, height]);

  // ── fMRI layer: update when brain or pulse changes ────────
  useEffect(() => {
    if (!fmriGroupRef.current || !fmriTexRef.current || !layout) return;
    const group = fmriGroupRef.current;
    disposeObject3D(group);
    group.clear();

    const tex = fmriTexRef.current;
    const pulseSet = new Set(pulseEids);

    personalBrain.forEach((c, idx) => {
      const node = nodeById.get(c.eid);
      if (!node) return;

      const isPulsing   = pulseSet.has(c.eid);
      const w           = Math.max(0.1, Math.min(1.0, c.weight));
      const baseScale   = 0.35 + w * 0.45;
      const baseOpacity = isPulsing ? 0.85 : 0.4 + w * 0.35;

      const mat = new THREE.SpriteMaterial({
        map:         tex,
        transparent: true,
        opacity:     baseOpacity,
        blending:    THREE.AdditiveBlending,
        depthWrite:  false,
        color:       isPulsing ? new THREE.Color(0x00d4aa) : C.fmri,
      });
      const spr = new THREE.Sprite(mat);
      spr.position.copy(nodeToWorld(node));
      spr.scale.setScalar(baseScale);
      spr.userData.baseOpacity = baseOpacity;
      spr.userData.phase       = idx * 0.7;
      group.add(spr);
    });

    needsRenderRef.current = true;
  }, [personalBrain, nodeById, pulseEids, layout]);

  // ── Path lines: update when selected RC path changes ─────
  useEffect(() => {
    if (!pathGroupRef.current || !layout) return;
    const group = pathGroupRef.current;
    disposeObject3D(group);
    group.clear();

    if (!rcPathData || selectedPathId === null) {
      needsRenderRef.current = true;
      return;
    }

    const path = rcPathData.paths.find((p) => p.path_id === selectedPathId);
    if (!path) {
      needsRenderRef.current = true;
      return;
    }

    const nodeSet  = new Set(path.node_ids);
    const edgeSet  = new Set(path.edge_ids);

    // Draw edge lines
    const pts: number[] = [];
    for (const edge of rcPathData.edges) {
      if (!edgeSet.has(edge.id)) continue;
      const srcNode = layout.nodes.find((n) => n.id === edge.source);
      const tgtNode = layout.nodes.find((n) => n.id === edge.target);
      if (!srcNode || !tgtNode) continue;
      const s = nodeToWorld(srcNode);
      const t = nodeToWorld(tgtNode);
      pts.push(s.x, s.y, s.z, t.x, t.y, t.z);
    }
    if (pts.length > 0) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts), 3));
      const mat = new THREE.LineBasicMaterial({ color: C.path_active, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false });
      group.add(new THREE.LineSegments(geo, mat));
    }

    // Glow sprites for path nodes
    const canvas2d = document.createElement("canvas");
    canvas2d.width = canvas2d.height = 64;
    const ctx  = canvas2d.getContext("2d")!;
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, "rgba(79,142,247,0.9)");
    grad.addColorStop(0.4, "rgba(79,142,247,0.3)");
    grad.addColorStop(1, "rgba(79,142,247,0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);
    const glowTex = new THREE.CanvasTexture(canvas2d);

    for (const nid of nodeSet) {
      const node = layout.nodes.find((n) => n.id === nid);
      if (!node) continue;
      const mat = new THREE.SpriteMaterial({ map: glowTex, transparent: true, opacity: 0.65, blending: THREE.AdditiveBlending, depthWrite: false });
      const spr = new THREE.Sprite(mat);
      spr.position.copy(nodeToWorld(node));
      spr.scale.setScalar(0.4);
      group.add(spr);
    }

    needsRenderRef.current = true;
  }, [rcPathData, selectedPathId, layout]);

  // ── RC tag click ─────────────────────────────────────────
  const handleTagClick = useCallback(async (eid: string) => {
    if (eid === selectedRCEid) {
      // deselect
      setSelectedRCEid(null);
      setSelectedPathId(null);
      setRCPathData(null);
      return;
    }
    setSelectedRCEid(eid);
    setSelectedPathId(null);
    setRCPathData(null);
    setLoadingRC(true);
    try {
      const data = await fetchRCPaths(eid);
      setRCPathData(data);
      if (data && data.paths.length > 0) setSelectedPathId(data.paths[0].path_id);
    } finally {
      setLoadingRC(false);
    }
  }, [selectedRCEid, fetchRCPaths]);

  const handleConfirm = useCallback(() => {
    if (!selectedRCEid || selectedPathId === null) return;
    const tag = rcRanking.find((r) => r.eid === selectedRCEid);
    onConfirmRC(selectedRCEid, tag?.label ?? selectedRCEid, selectedPathId);
  }, [selectedRCEid, selectedPathId, rcRanking, onConfirmRC]);

  // ─────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────

  return (
    <div style={{ position: "relative", width, height, borderRadius: "20px", overflow: "hidden", background: C.bg_css, cursor: "grab", flexShrink: 0 }}>
      <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />

      {/* Title */}
      <div style={{ position: "absolute", top: 10, left: 12, fontSize: "0.6rem", color: "rgba(127,119,221,0.4)", letterSpacing: "0.15em", textTransform: "uppercase", fontFamily: "monospace", pointerEvents: "none", userSelect: "none" }}>
        Personal Brain
      </div>

      {/* Brain count */}
      {personalBrain.length > 0 && (
        <div style={{ position: "absolute", top: 10, right: 12, fontSize: "0.6rem", color: "rgba(127,119,221,0.5)", fontFamily: "monospace", pointerEvents: "none", userSelect: "none" }}>
          {personalBrain.length} concepts
        </div>
      )}

      {/* Empty hint */}
      {!layout || personalBrain.length === 0 ? (
        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 6, color: "rgba(127,119,221,0.15)", fontSize: "0.65rem", letterSpacing: "0.14em", textTransform: "uppercase", fontFamily: "monospace", pointerEvents: "none" }}>
          <span>Personal Brain</span>
          <span style={{ fontSize: "0.55rem" }}>探索 concept，加入 foundations</span>
        </div>
      ) : null}

      {/* ── RC Tag row ── */}
      {rcRanking.length > 0 && (
        <div style={{
          position: "absolute", top: 30, left: 0, right: 0,
          display: "flex", justifyContent: "center", gap: 6, padding: "4px 12px",
          flexWrap: "wrap", pointerEvents: "auto",
        }}>
          {rcRanking.slice(0, 3).map((rc, i) => {
            const isActive = selectedRCEid === rc.eid;
            const color    = TAG_COLORS[i];
            return (
              <button
                key={rc.eid}
                onClick={() => handleTagClick(rc.eid)}
                style={{
                  fontSize: "0.65rem", padding: "3px 10px", borderRadius: 16,
                  border:      `1px solid ${isActive ? color : "rgba(127,119,221,0.3)"}`,
                  background:  isActive ? `rgba(127,119,221,0.15)` : "rgba(3,1,8,0.7)",
                  color:       isActive ? color : "rgba(175,169,236,0.6)",
                  fontFamily:  "monospace", cursor: "pointer",
                  transition:  "all 0.2s ease",
                  boxShadow:   isActive ? `0 0 8px ${color}44` : "none",
                  letterSpacing: "0.03em",
                  backdropFilter: "blur(4px)",
                }}
              >
                {loadingRC && isActive ? "…" : `${i + 1}. ${rc.label}`}
                <span style={{ marginLeft: 4, fontSize: "0.56rem", opacity: 0.7 }}>
                  {(rc.totalScore * 100).toFixed(0)}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* ── Path sub-selector (when RC selected) ── */}
      {selectedRCEid && rcPathData && rcPathData.paths.length > 0 && (
        <div style={{
          position: "absolute", bottom: selectedRCEid ? 52 : 10,
          left: 0, right: 0, padding: "0 12px",
          display: "flex", flexDirection: "column", gap: 5, pointerEvents: "auto",
        }}>
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap", justifyContent: "center" }}>
            {rcPathData.paths.map((p) => {
              const isActive = selectedPathId === p.path_id;
              return (
                <button
                  key={p.path_id}
                  onClick={() => setSelectedPathId(p.path_id)}
                  style={{
                    fontSize: "0.62rem", padding: "2px 8px", borderRadius: 12,
                    border:     `1px solid ${isActive ? "#4f8ef7" : "rgba(79,142,247,0.25)"}`,
                    background: isActive ? "rgba(79,142,247,0.15)" : "rgba(3,1,8,0.6)",
                    color:      isActive ? "#4f8ef7" : "rgba(79,142,247,0.5)",
                    fontFamily: "monospace", cursor: "pointer",
                    backdropFilter: "blur(4px)",
                  }}
                >
                  ↓ {p.final_label || `Path ${p.path_id + 1}`}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Confirm button ── */}
      {selectedRCEid && selectedPathId !== null && (
        <div style={{ position: "absolute", bottom: 10, left: 12, right: 12, pointerEvents: "auto" }}>
          <button
            onClick={handleConfirm}
            style={{
              width: "100%", padding: "8px",
              background: "linear-gradient(135deg, rgba(48,16,96,0.85), rgba(13,48,80,0.85))",
              border:      "1px solid #7f77dd", color: "#afa9ec",
              fontSize:    "0.75rem", fontWeight: 600, fontFamily: "monospace",
              borderRadius: 8, cursor: "pointer", letterSpacing: "0.04em",
              backdropFilter: "blur(6px)",
              transition: "box-shadow 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.boxShadow = "0 0 14px rgba(127,119,221,0.3)")}
            onMouseLeave={(e) => (e.currentTarget.style.boxShadow = "none")}
          >
            確認「{rcRanking.find((r) => r.eid === selectedRCEid)?.label ?? ""}」→ 開始分析
          </button>
        </div>
      )}

      {/* Pulse indicator */}
      {pulseEids.length > 0 && (
        <div style={{ position: "absolute", bottom: selectedRCEid ? 70 : 10, right: 12, fontSize: "0.6rem", color: "var(--teal)", fontFamily: "monospace", pointerEvents: "none", display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--teal)", display: "inline-block", animation: "pb-pulse 0.6s ease-in-out 3" }} />
          +{pulseEids.length} activated
        </div>
      )}

      <style>{`
        @keyframes pb-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.2;transform:scale(1.8)} }
      `}</style>
    </div>
  );
}