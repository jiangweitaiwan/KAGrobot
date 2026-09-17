"use client";

/**
 * GlobalBrainExplorer.tsx（三層殼層版 + 效能優化版 + three-shared 共用版）
 *
 * 殼層隱喻（由核心向外）：
 *   article  → 內核  — 青白色，最核心的基礎概念
 *   chapter  → 中層  — 紫色，教科書概念
 *   paper    → 外層  — 暖橘色，論文專精概念
 *
 * 互動邏輯不變（聚焦候選、foundation 路徑、drag-to-rotate 等）。
 *
 * ── three-shared 共用重構 ────────────────────────────────────
 *   makeLabelSprite / makeRingSprite / makeGlowSprite / makeFlatDotSprite /
 *   disposeObject3D / disposeGroupChildren 原本是這個檔案的私有函式，
 *   現在抽到 three-shared.ts，供 HourglassViz.tsx 共用同一套 sprite
 *   視覺語言，避免兩個場景的質感各自漂移。行為完全不變，純搬移，
 *   不影響下方效能優化的任何邏輯。
 *
 * ── 效能優化（維持不變）────────────────────────────────────
 *   1. 移除自動旋轉：場景不再持續自轉，改為純使用者驅動。
 *   2. 按需渲染（on-demand rendering）：只有在「真的有東西在動」
 *      （拖曳中、有接力光點動畫、聚焦光暈呼吸）或畫面被標記為
 *      dirty 時才送一幀進 GPU，其餘時間迴圈直接 return，不渲染。
 *   3. 可見性偵測：分頁切到背景（visibilitychange）或面板捲出
 *      可視範圍（IntersectionObserver）時，完全暫停渲染。
 *   4. 渲染幀率上限 ~24fps（原本跟著 rAF 跑滿 60fps）。
 *   5. devicePixelRatio 上限降至 1.25（原本 2），大幅降低
 *      fill-rate 成本，一般螢幕無感，Retina/4K 螢幕仍清晰可用。
 *   6. Hover 最近節點偵測降頻至 ~10Hz（原本 ~30Hz），減少主執行緒
 *      上的暴力投影計算。
 *   7. Cleanup 時完整 dispose 所有 geometry / material / texture，
 *      避免面板 resize 觸發場景重建時造成 GPU 記憶體持續累積。
 *
 * ── Renderer 生命週期重構（新增）──────────────────────────────
 *   原本 renderer/scene/camera/背景星點/marker points/edges/分區
 *   標籤/事件監聽/render loop 全部塞在同一個 useEffect，依賴陣列
 *   包含 [layout, width, height, nodeWorldPositions, clusterStats,
 *   nodeIdxMap]。這代表每次點擊推薦 RC 導致 layout 略有變化、或單純
 *   面板 resize，都會把整顆 WebGLRenderer dispose 掉再重新 new 一次
 *   ——這是這幾個操作裡最貴的一步，沒必要每次都做。
 *
 *   現在拆成三層生命週期：
 *     1) Effect A（mount once，依賴 []）
 *        建立 renderer / scene / camera / 背景星點 / ambient light /
 *        hoverRing / pathGroup / ringGroup / focusGroup 容器 /
 *        事件監聽 / render loop。canvas 掛載後只建一次。
 *        事件處理器裡需要讀的易變資料（layout、nodeWorldPositions、
 *        width、height、onNodeClick）一律改讀對應的 ref。
 *     2) Effect B（resize，依賴 [width, height]）
 *        只呼叫 renderer.setSize() + 更新 camera.aspect +
 *        updateProjectionMatrix()，不 dispose 任何東西。
 *     3) Effect C（layout 資料重建，依賴 [layout, nodeWorldPositions,
 *        clusterStats, nodeIdxMap]）
 *        只清空／重建 layoutGroup（marker points、neural mesh 邊、
 *        分區標籤），renderer 本身完全不受影響。
 *   原本就已獨立、不碰 renderer 的「Active candidate 標示環」與
 *   「聚焦候選：光暈 + foundation 路徑」兩個 effect 維持不變。
 */

import { useEffect, useRef, useMemo, useState } from "react";
import * as THREE from "three";
import type { RCFoundationMap } from "./DetailPanel";
import {
  makeLabelSprite,
  makeRingSprite,
  makeGlowSprite,
  makeFlatDotSprite,
  makeHollowDotSprite,
  makeCrossSprite,          // ← 新增
  disposeObject3D,
  disposeGroupChildren,
} from "./three-shared";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type GlobalLayoutNode = {
  id:      string;
  name:    string;
  type:    string;
  source?: string;
  category?: string | null;        
  category_color?: string | null;  
  x:       number;
  y:       number;
  z:       number;
};

export type GlobalLayoutEdge = {
  source: string;
  target: string;
};

export type GlobalLayout = {
  nodes: GlobalLayoutNode[];
  edges: GlobalLayoutEdge[];
};

export type WalkIllumination = {
  pathId: number;
  color: string;
  visitedEids: string[];
  frontierPreview: { eid: string; isRc: boolean }[];
  frontierHistory: { fromEid: string; candidates: { eid: string; isRc: boolean }[] }[];
  status: "pending" | "active" | "dead_end";
};

type Props = {
  layout: GlobalLayout | null;
  activeCandidateEids: string[];
  walkIllumination?: WalkIllumination[];
  rcFoundationMap?: RCFoundationMap;
  onNodeClick?: (node: GlobalLayoutNode) => void;
  dimBackground?: boolean;      // ← 新增：腦內模式開關
  width?: number;
  height?: number;
  externalPause?: boolean; 
};

type PulseInfo = {
  from:  THREE.Vector3;
  to:    THREE.Vector3;
  dot:   THREE.Sprite;
  phase: number;
};

// ─────────────────────────────────────────────────────────────
// Cluster key（三層）
// ─────────────────────────────────────────────────────────────

type ClusterKey = "article" | "chapter" | "paper";

function clusterKeyOf(node: GlobalLayoutNode): ClusterKey {
  const s = (node.source || "paper").toLowerCase();
  if (s === "article") return "article";
  if (s === "chapter") return "chapter";
  return "paper";
}

function isRC(node: GlobalLayoutNode): boolean {
  return (node.type || "").toLowerCase().includes("resulting");
}

// ─────────────────────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────────────────────

const PALETTE = {
  bg:     0x06080f,
  bg_css: "#06080f",

  article: new THREE.Color(0x00e8ff),
  chapter: new THREE.Color(0xb48cff),
  paper:   new THREE.Color(0xff9a52),

  activeCandidate: new THREE.Color(0xff5fd1),  // ← 原 0x00e0b8，改洋紅，跟 walk 色系區隔
  focusGlow:       new THREE.Color(0x5fe6cc),
  foundationPath:  new THREE.Color(0xffb84d),

  strokeCss: "#e8eef8",
};

// ── 新增：腦內模式的背景淡出色 ──
const DIM_COLOR = new THREE.Color(0x262b38);
const DIM_SIZE_MULT = 0.55;

const CLUSTER_COLOR: Record<ClusterKey, THREE.Color> = {
  article: PALETTE.article,
  chapter: PALETTE.chapter,
  paper:   PALETTE.paper,
};

const CLUSTER_LABEL: Record<ClusterKey, string> = {
  article: "核心概念",
  chapter: "教科書概念",
  paper:   "論文概念",
};

const CLUSTER_LABEL_COLOR: Record<ClusterKey, string> = {
  article: "#66eeff",
  chapter: "#c9aaff",
  paper:   "#ffb37a",
};

const SIZE = {
  rc:                  0.16,
  other:               0.10,
  highlightMultiplier: 1.55,
  focusMultiplier:     2.0,
};

const WORLD_SCALE       = 2.2;
const ZOOM_MIN          = 2.0;
const ZOOM_MAX          = 9.5;
const ZOOM_DEFAULT      = 5.5;
const PICK_PIXEL_RADIUS = 16;
const LABEL_Y_OFFSET    = 0.34;

// ── 效能參數 ─────────────────────────────────────────────────
const MAX_PIXEL_RATIO   = 1.25;  // 原本 2，降低 fill-rate 成本
const RENDER_FRAME_MS   = 1000 / 24; // 渲染幀率上限 ~24fps
const HOVER_PICK_EVERY  = 6;     // 原本 2（~30Hz）→ 6（~10Hz）

// ─────────────────────────────────────────────────────────────
// Helpers（只留本檔案專屬的，共用 sprite/dispose helper 已搬到 three-shared.ts）
// ─────────────────────────────────────────────────────────────

function nodeToWorld(n: GlobalLayoutNode): THREE.Vector3 {
  return new THREE.Vector3(n.x * WORLD_SCALE, n.y * WORLD_SCALE, n.z * WORLD_SCALE);
}

function categoryColor(node: GlobalLayoutNode): THREE.Color {
  return CLUSTER_COLOR[clusterKeyOf(node)];
}

const CIRCLED = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩","⑪","⑫","⑬","⑭","⑮","⑯","⑰","⑱","⑲","⑳"];
function circledNum(n: number): string {
  return CIRCLED[n - 1] ?? `${n}.`;
}

// ─────────────────────────────────────────────────────────────
// GlobalBrainExplorer
// ─────────────────────────────────────────────────────────────

export default function GlobalBrainExplorer({
  layout,
  activeCandidateEids,
  walkIllumination,
  rcFoundationMap,
  onNodeClick,
  dimBackground = false,   
  externalPause = false,
  width  = 480,
  height = 480,
}: Props) {
  const canvasRef  = useRef<HTMLCanvasElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  const sceneRef        = useRef<THREE.Scene | null>(null);
  const rendererRef     = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef       = useRef<THREE.PerspectiveCamera | null>(null);
  const rotGroupRef     = useRef<THREE.Group | null>(null);
  const layoutGroupRef  = useRef<THREE.Group | null>(null);   // ← layout 相關內容容器（Effect C 管理）
  const markerPointsRef = useRef<THREE.Points | null>(null);
  const pathGroupRef    = useRef<THREE.Group | null>(null);
  const ringGroupRef    = useRef<THREE.Group | null>(null);
  const focusGroupRef   = useRef<THREE.Group | null>(null);
  const hoverRingRef    = useRef<THREE.Sprite | null>(null);
  const animFrameRef    = useRef<number>(0);

  const isDragging     = useRef(false);
  const prevMouse      = useRef({ x: 0, y: 0 });
  const mousePixel     = useRef({ x: -9999, y: -9999 });
  const pinchStartDist = useRef(0);
  const pinchStartZ    = useRef(ZOOM_DEFAULT);
  const pulsesRef      = useRef<PulseInfo[]>([]);
  const hoveredIdxRef  = useRef<number | null>(null);

  // ── 效能：可見性 / 按需渲染狀態 ──────────────────────────
  const isVisibleRef   = useRef(true);   // 分頁可見 && 面板在視窗內
  const needsRenderRef = useRef(true);   // 畫面是否為 dirty，需要重繪一次
  const lastFrameTsRef = useRef(0);      // 上一次實際渲染的時間戳

  // ── 供 Effect A（mount-once）事件處理器讀取「當下最新值」，
  //    避免 stale closure，同時不需要把這些值放進 Effect A 的依賴陣列 ──
  const layoutRef              = useRef(layout);
  const nodeWorldPositionsRef  = useRef<THREE.Vector3[]>([]);
  const widthRef               = useRef(width);
  const heightRef              = useRef(height);
  const onNodeClickRef         = useRef(onNodeClick);
  const activeCandidateSetRef  = useRef<Set<string>>(new Set());
  const externalPauseRef = useRef(false);
  const walkGroupRef  = useRef<THREE.Group | null>(null);
  const walkPulsesRef = useRef<PulseInfo[]>([]);

  const edgesMaterialRef       = useRef<THREE.LineBasicMaterial | null>(null);
  const clusterLabelSpritesRef = useRef<THREE.Sprite[]>([]);

  useEffect(() => { layoutRef.current = layout; }, [layout]);
  useEffect(() => { widthRef.current = width; heightRef.current = height; }, [width, height]);
  useEffect(() => { onNodeClickRef.current = onNodeClick; }, [onNodeClick]);
  useEffect(() => {
    externalPauseRef.current = !!externalPause;
    needsRenderRef.current = true;   // 解除暫停時，補畫一幀反映當下真實狀態
  }, [externalPause]);
  const [focusedCandidateEid, setFocusedCandidateEid] = useState<string | null>(null);

  useEffect(() => {
    if (focusedCandidateEid && !activeCandidateEids.includes(focusedCandidateEid)) {
      setFocusedCandidateEid(null);
    }
  }, [activeCandidateEids, focusedCandidateEid]);

  const nodeIdxMap = useMemo(() => {
    if (!layout) return new Map<string, number>();
    return new Map(layout.nodes.map((n, i) => [n.id, i]));
  }, [layout]);

  const nodeWorldPositions = useMemo(() => {
    if (!layout) return [] as THREE.Vector3[];
    return layout.nodes.map((n) => nodeToWorld(n));
  }, [layout]);

  useEffect(() => { nodeWorldPositionsRef.current = nodeWorldPositions; }, [nodeWorldPositions]);

  // ── 分群統計（三層）────────────────────────────────────────
  const clusterStats = useMemo(() => {
    if (!layout || layout.nodes.length === 0) return [];
    const groups: Record<ClusterKey, { count: number; sum: THREE.Vector3 }> = {
      article: { count: 0, sum: new THREE.Vector3() },
      chapter: { count: 0, sum: new THREE.Vector3() },
      paper:   { count: 0, sum: new THREE.Vector3() },
    };
    layout.nodes.forEach((n, i) => {
      const k = clusterKeyOf(n);
      groups[k].count++;
      groups[k].sum.add(nodeWorldPositions[i]);
    });
    const total = layout.nodes.length;
    return (Object.keys(groups) as ClusterKey[])
      .filter((k) => groups[k].count > 0)
      .map((k) => ({
        key:      k,
        count:    groups[k].count,
        pct:      total > 0 ? (groups[k].count / total) * 100 : 0,
        centroid: groups[k].sum.clone().multiplyScalar(1 / groups[k].count),
      }));
  }, [layout, nodeWorldPositions]);

  const activeCandidateSet = useMemo(
    () => new Set(activeCandidateEids),
    [activeCandidateEids],
  );
  const walkActiveEidSet = useMemo(() => {
    const s = new Set<string>();
    (walkIllumination ?? []).forEach((wp) => {
      wp.visitedEids.forEach((e) => s.add(e));
      wp.frontierPreview.forEach((c) => s.add(c.eid));
      wp.frontierHistory.forEach((step) => step.candidates.forEach((c) => s.add(c.eid)));
    });
    return s;
  }, [walkIllumination]);  

  useEffect(() => { activeCandidateSetRef.current = activeCandidateSet; }, [activeCandidateSet]);

  const focusedTreeData = useMemo(() => {
    if (!rcFoundationMap || !focusedCandidateEid) {
      return { highlightEids: new Set<string>(), edges: [] as { source: string; target: string }[] };
    }
    const entry = rcFoundationMap.get(focusedCandidateEid);
    if (!entry) {
      return { highlightEids: new Set<string>(), edges: [] as { source: string; target: string }[] };
    }
    const highlightEids = new Set<string>([focusedCandidateEid]);
    entry.foundationEids.forEach((f) => highlightEids.add(f));
    (entry.treeNodes ?? []).forEach((n) => highlightEids.add(n.element_id));
    return { highlightEids, edges: entry.treeEdges ?? [] };
  }, [rcFoundationMap, focusedCandidateEid]);

  // ═════════════════════════════════════════════════════════
  // Effect A — mount once：renderer / scene / camera / 背景星點 /
  //            ambient light / hoverRing / pathGroup / ringGroup /
  //            focusGroup 容器 / 事件監聽 / render loop。
  //            依賴陣列固定為 []，canvas 掛載後只建一次。
  // ═════════════════════════════════════════════════════════
  useEffect(() => {
    if (!canvasRef.current) return;

    const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true, alpha: false });
    renderer.setSize(widthRef.current, heightRef.current);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
    renderer.setClearColor(PALETTE.bg, 1);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      60,
      widthRef.current / Math.max(heightRef.current, 1),
      0.01,
      100,
    );
    camera.position.set(0, 0, ZOOM_DEFAULT);
    cameraRef.current = camera;

    const rg = new THREE.Group();
    scene.add(rg);
    rotGroupRef.current = rg;

    // 背景星點（靜態，不隨 layout 變化，只建一次）
    const bgCount = 1200;
    const bgPos   = new Float32Array(bgCount * 3);
    for (let i = 0; i < bgCount; i++) {
      bgPos[i * 3 + 0] = (Math.random() * 2 - 1) * 9;
      bgPos[i * 3 + 1] = (Math.random() * 2 - 1) * 9;
      bgPos[i * 3 + 2] = (Math.random() * 2 - 1) * 9;
    }
    const bgGeo = new THREE.BufferGeometry();
    bgGeo.setAttribute("position", new THREE.BufferAttribute(bgPos, 3));
    const bgMat = new THREE.PointsMaterial({
      color: 0x2a3550, size: 0.01, transparent: true, opacity: 0.35, depthWrite: false,
    });
    scene.add(new THREE.Points(bgGeo, bgMat));

    // ── layout 相關內容的容器：marker points / edges / 分區標籤，
    //    交給 Effect C 專門管理，這裡只建立空容器 ──────────────
    const layoutGroup = new THREE.Group();
    rg.add(layoutGroup);
    layoutGroupRef.current = layoutGroup;

    // path / ring / focus groups（既有的「聚焦候選」「標示環」effect 會操作這些，
    // 它們本來就不碰 renderer，維持原樣不變）
    const pathGroup  = new THREE.Group(); rg.add(pathGroup);  pathGroupRef.current  = pathGroup;
    const ringGroup  = new THREE.Group(); rg.add(ringGroup);  ringGroupRef.current  = ringGroup;
    const focusGroup = new THREE.Group(); rg.add(focusGroup); focusGroupRef.current = focusGroup;
    const walkGroup = new THREE.Group(); rg.add(walkGroup); walkGroupRef.current = walkGroup;
    // hover ring
    const hoverRing = makeRingSprite("#7a90c0");
    hoverRing.visible = false;
    hoverRing.scale.set(0.22, 0.22, 1);
    rg.add(hoverRing);
    hoverRingRef.current = hoverRing;

    scene.add(new THREE.AmbientLight(0xffffff, 1));

    // 場景剛建立，標記需要畫第一幀
    needsRenderRef.current = true;

    // ── 螢幕空間點擊判定（讀 layoutRef / nodeWorldPositionsRef，避免 stale closure）──
    function worldToScreenPx(worldPos: THREE.Vector3): { x: number; y: number } | null {
      const cam = cameraRef.current;
      if (!cam) return null;
      const proj = worldPos.clone().project(cam);
      if (Math.abs(proj.z) > 1) return null;
      return {
        x: ((proj.x + 1) / 2) * widthRef.current,
        y: ((1 - proj.y) / 2) * heightRef.current,
      };
    }

    function findNearestNodeIndex(px: number, py: number): number | null {
      const currentLayout = layoutRef.current;
      if (!rotGroupRef.current || !currentLayout) return null;
      let bestDist = PICK_PIXEL_RADIUS * PICK_PIXEL_RADIUS;
      let bestIdx: number | null = null;
      currentLayout.nodes.forEach((_, i) => {
        const wp = nodeWorldPositionsRef.current[i]?.clone();
        if (!wp) return;
        rotGroupRef.current!.localToWorld(wp);
        const sp = worldToScreenPx(wp);
        if (!sp) return;
        const dx = sp.x - px, dy = sp.y - py;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDist) { bestDist = d2; bestIdx = i; }
      });
      return bestIdx;
    }

    // ── Animation loop（按需渲染版）────────────────────────────
    let frameCount = 0;
    function animate(now: number) {
      animFrameRef.current = requestAnimationFrame(animate);

      if (!isVisibleRef.current || externalPauseRef.current) return;
      if (now - lastFrameTsRef.current < RENDER_FRAME_MS) return;
      lastFrameTsRef.current = now;
      frameCount++;

      // hover 最近節點偵測（降頻）
      if (frameCount % HOVER_PICK_EVERY === 0 && !isDragging.current) {
        const { x, y } = mousePixel.current;
        const idx = findNearestNodeIndex(x, y);
        hoveredIdxRef.current = idx;
        const hRing = hoverRingRef.current;
        if (hRing) {
          if (idx !== null) {
            const wp = nodeWorldPositionsRef.current[idx].clone();
            rotGroupRef.current!.localToWorld(wp);
            hRing.position.copy(wp);
            hRing.visible = true;
            if (tooltipRef.current) {
              const sp = worldToScreenPx(wp);
              if (sp) {
                const node = layoutRef.current?.nodes[idx];
                const layerLabel = node ? CLUSTER_LABEL[clusterKeyOf(node)] : "";
                const categoryLabel = node?.category ? `  ·  ${node.category}` : "";
                tooltipRef.current.textContent = node
                  ? `${node.name}  ·  ${layerLabel}${categoryLabel}`
                  : "";                
                tooltipRef.current.style.left    = `${sp.x + 12}px`;
                tooltipRef.current.style.top     = `${sp.y - 8}px`;
                tooltipRef.current.style.opacity = "1";
              }
            }
          } else {
            hRing.visible = false;
            if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
          }
        }
        needsRenderRef.current = true; // hover 狀態變了，補畫一幀
      }

      // 聚焦光暈呼吸（仍在動 → 需要持續渲染）
      let hasBreathing = false;
      if (focusGroupRef.current && focusGroupRef.current.children.length > 0) {
        hasBreathing = true;
        const t = frameCount * 0.04;
        const pulse = 0.85 + 0.15 * Math.sin(t);
        focusGroupRef.current.children.forEach((child) => {
          if (child instanceof THREE.Sprite) {
            child.material.opacity = (child.userData.baseOpacity ?? 1) * pulse;
          }
        });
      }

      // 接力光點動畫（仍在動 → 需要持續渲染）
      const hasActivePulses = pulsesRef.current.length > 0 || walkPulsesRef.current.length > 0;
      pulsesRef.current.forEach((p) => {
        p.phase = (p.phase + 0.012) % 1;
        p.dot.position.lerpVectors(p.from, p.to, p.phase);
      });
      walkPulsesRef.current.forEach((p) => {
        p.phase = (p.phase + 0.012) % 1;
        p.dot.position.lerpVectors(p.from, p.to, p.phase);
      });

      // 只有在「畫面真的有變化」時才送進 GPU：
      //   - 標記為 dirty（互動、resize、資料更新後手動觸發）
      //   - 拖曳中
      //   - 有持續動畫（呼吸光暈 / 接力光點）
      if (needsRenderRef.current || isDragging.current || hasBreathing || hasActivePulses) {
        renderer.render(scene, camera);
        needsRenderRef.current = false;
      }
    }
    animFrameRef.current = requestAnimationFrame(animate);

    // ── 事件 ─────────────────────────────────────────────────
    const canvas = canvasRef.current!;
    function getBoundedRect() { return canvas.getBoundingClientRect(); }
    let mouseDownPos = { x: 0, y: 0 };

    function onMouseDown(e: MouseEvent) {
      isDragging.current = true;
      prevMouse.current  = { x: e.clientX, y: e.clientY };
      mouseDownPos       = { x: e.clientX, y: e.clientY };
      needsRenderRef.current = true;
    }
    function onMouseUp(e: MouseEvent) {
      const dx = Math.abs(e.clientX - mouseDownPos.x);
      const dy = Math.abs(e.clientY - mouseDownPos.y);
      if (dx < 4 && dy < 4) {
        const r   = getBoundedRect();
        const px  = e.clientX - r.left;
        const py  = e.clientY - r.top;
        const idx = findNearestNodeIndex(px, py);
        const currentLayout = layoutRef.current;
        if (idx !== null && currentLayout) {
          const clickedNode = currentLayout.nodes[idx];
          if (activeCandidateSetRef.current.has(clickedNode.id)) {
            setFocusedCandidateEid((prev) =>
              prev === clickedNode.id ? prev : clickedNode.id,
            );
          }
          onNodeClickRef.current?.(clickedNode);
        } else {
          setFocusedCandidateEid(null);
        }
      }
      isDragging.current = false;
      needsRenderRef.current = true;
    }
    function onMouseMove(e: MouseEvent) {
      const r = getBoundedRect();
      mousePixel.current = { x: e.clientX - r.left, y: e.clientY - r.top };
      if (!isDragging.current || !rotGroupRef.current) return;
      rotGroupRef.current.rotation.y += (e.clientX - prevMouse.current.x) * 0.008;
      rotGroupRef.current.rotation.x += (e.clientY - prevMouse.current.y) * 0.008;
      prevMouse.current = { x: e.clientX, y: e.clientY };
      needsRenderRef.current = true;
    }
    function onMouseLeave() {
      isDragging.current = false;
      mousePixel.current = { x: -9999, y: -9999 };
      if (tooltipRef.current)    tooltipRef.current.style.opacity = "0";
      if (hoverRingRef.current)  hoverRingRef.current.visible = false;
      needsRenderRef.current = true;
    }
    function onWheel(e: WheelEvent) {
      e.preventDefault();
      const cam = cameraRef.current;
      if (!cam) return;
      cam.position.z = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, cam.position.z + e.deltaY * 0.01));
      needsRenderRef.current = true;
    }
    function getTouchDist(e: TouchEvent) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      return Math.hypot(dx, dy);
    }
    function onTouchStart(e: TouchEvent) {
      if (e.touches.length === 2) {
        pinchStartDist.current = getTouchDist(e);
        pinchStartZ.current    = cameraRef.current?.position.z ?? ZOOM_DEFAULT;
      } else {
        isDragging.current = true;
        prevMouse.current  = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
      needsRenderRef.current = true;
    }
    function onTouchMove(e: TouchEvent) {
      if (e.touches.length === 2 && cameraRef.current) {
        const dist = getTouchDist(e);
        const scale = pinchStartDist.current / Math.max(dist, 1);
        cameraRef.current.position.z = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, pinchStartZ.current * scale));
      } else if (e.touches.length === 1 && isDragging.current && rotGroupRef.current) {
        rotGroupRef.current.rotation.y += (e.touches[0].clientX - prevMouse.current.x) * 0.008;
        rotGroupRef.current.rotation.x += (e.touches[0].clientY - prevMouse.current.y) * 0.008;
        prevMouse.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
      needsRenderRef.current = true;
    }
    function onTouchEnd() { isDragging.current = false; needsRenderRef.current = true; }

    canvas.addEventListener("mousedown",  onMouseDown);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("wheel",      onWheel, { passive: false });
    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("mouseup",    onMouseUp);
    window.addEventListener("mousemove",  onMouseMove);
    window.addEventListener("touchmove",  onTouchMove, { passive: true });
    window.addEventListener("touchend",   onTouchEnd);

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
    if (typeof IntersectionObserver !== "undefined") {
      io = new IntersectionObserver(
        ([entry]) => {
          isVisibleRef.current = entry.isIntersecting && !document.hidden;
          if (isVisibleRef.current) needsRenderRef.current = true;
        },
        { threshold: 0.05 },
      );
      io.observe(canvas);
    }

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      canvas.removeEventListener("mousedown",  onMouseDown);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      canvas.removeEventListener("wheel",      onWheel);
      canvas.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("mouseup",    onMouseUp);
      window.removeEventListener("mousemove",  onMouseMove);
      window.removeEventListener("touchmove",  onTouchMove);
      window.removeEventListener("touchend",   onTouchEnd);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      io?.disconnect();

      // ── 完整釋放 GPU 資源，避免 unmount 時持續累積 ──────
      disposeObject3D(scene);
      renderer.dispose();

      rendererRef.current     = null;
      sceneRef.current        = null;
      cameraRef.current       = null;
      rotGroupRef.current     = null;
      layoutGroupRef.current  = null;
      pathGroupRef.current    = null;
      ringGroupRef.current    = null;
      focusGroupRef.current   = null;
      hoverRingRef.current    = null;
      markerPointsRef.current = null;
      edgesMaterialRef.current       = null;
      clusterLabelSpritesRef.current = [];
      walkGroupRef.current = null;
    };
    // 依賴陣列固定為 []：只在 canvas 掛載／卸載時執行一次。
    // 易變資料（layout/nodeWorldPositions/width/height/onNodeClick）
    // 一律透過 ref 讀取，見上方各個 xxxRef 的同步 effect。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ═════════════════════════════════════════════════════════
  // Effect B — resize：只更新 renderer size / camera aspect，
  //            不 dispose、不重建任何場景物件。
  // ═════════════════════════════════════════════════════════
  useEffect(() => {
    const renderer = rendererRef.current;
    const camera   = cameraRef.current;
    if (!renderer || !camera || width <= 0 || height <= 0) return;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    needsRenderRef.current = true;
  }, [width, height]);

  // ═════════════════════════════════════════════════════════
  // Effect C — layout 資料變化：只重建 layoutGroup 底下的
  //            marker points / neural mesh 邊 / 分區標籤，
  //            renderer / scene / camera 完全不受影響。
  // ═════════════════════════════════════════════════════════
  useEffect(() => {
    const layoutGroup = layoutGroupRef.current;
    if (!layoutGroup || !layout || layout.nodes.length === 0) return;

    disposeGroupChildren(layoutGroup);
    layoutGroup.clear();
    markerPointsRef.current = null;
    edgesMaterialRef.current = null;
    clusterLabelSpritesRef.current = [];
    // 節點標記
    const n = layout.nodes.length;
    const positions = new Float32Array(n * 3);
    const colors    = new Float32Array(n * 3);
    const sizes     = new Float32Array(n);

    layout.nodes.forEach((node, i) => {
      const p = nodeWorldPositions[i];
      positions[i * 3] = p.x; positions[i * 3 + 1] = p.y; positions[i * 3 + 2] = p.z;
      const c = categoryColor(node);
      colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
      sizes[i] = isRC(node) ? SIZE.rc : SIZE.other;
    });

    const markerGeo = new THREE.BufferGeometry();
    markerGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    markerGeo.setAttribute("color",    new THREE.BufferAttribute(colors, 3));
    markerGeo.setAttribute("size",     new THREE.BufferAttribute(sizes, 1));

    const markerMat = new THREE.ShaderMaterial({
      vertexShader: /* glsl */`
        attribute float size; attribute vec3 color; varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mvPos.z);
          gl_Position  = projectionMatrix * mvPos;
        }`,
      fragmentShader: /* glsl */`
        varying vec3 vColor;
        void main() {
          vec2 uv = gl_PointCoord - vec2(0.5);
          float d = length(uv);
          if (d > 0.5) discard;
          if (d > 0.40) {
            gl_FragColor = vec4(0.91, 0.93, 0.97, 1.0);
          } else {
            gl_FragColor = vec4(vColor, 1.0);
          }
        }`,
      transparent: false,
      depthWrite:  true,
    });

    const markerPoints = new THREE.Points(markerGeo, markerMat);
    layoutGroup.add(markerPoints);
    markerPointsRef.current = markerPoints;

    // 神經網格（edges）
    if (layout.edges.length > 0) {
      const linePos: number[] = [];
      layout.edges.forEach((e) => {
        const si = nodeIdxMap.get(e.source);
        const ti = nodeIdxMap.get(e.target);
        if (si === undefined || ti === undefined) return;
        const sp = nodeWorldPositions[si], tp = nodeWorldPositions[ti];
        linePos.push(sp.x, sp.y, sp.z, tp.x, tp.y, tp.z);
      });
      if (linePos.length > 0) {
        const meshGeo = new THREE.BufferGeometry();
        meshGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(linePos), 3));
        const meshMat = new THREE.LineBasicMaterial({ color: 0x3a4868, transparent: true, opacity: 0.10 });
        layoutGroup.add(new THREE.LineSegments(meshGeo, meshMat));
        edgesMaterialRef.current = meshMat;   // ← 供腦內模式動態調整透明度
      }
    }

    // 分區浮動標籤（三層）
    clusterStats.forEach((c) => {
      const label = makeLabelSprite(
        `${CLUSTER_LABEL[c.key]} ${c.pct.toFixed(0)}%`,
        CLUSTER_LABEL_COLOR[c.key],
      );
      label.position.set(c.centroid.x, c.centroid.y + LABEL_Y_OFFSET, c.centroid.z);
      layoutGroup.add(label);
      clusterLabelSpritesRef.current.push(label);   // ← 供腦內模式淡出
    });

    needsRenderRef.current = true;
  }, [layout, nodeWorldPositions, clusterStats, nodeIdxMap]);

  // ── Active candidate 標示環（不碰 renderer，維持原樣）──────
  useEffect(() => {
    const ringGroup = ringGroupRef.current;
    if (!ringGroup || !layout) return;
    disposeGroupChildren(ringGroup); ringGroup.clear();
  
    activeCandidateEids.forEach((eid) => {
      if (eid === focusedCandidateEid) return;
      const idx = nodeIdxMap.get(eid);
      if (idx === undefined) return;
      const ring = makeRingSprite("#ff5fd1", 4);  // ← was activeNodeColors?.get(eid) ?? "#00e0b8"
      ring.position.copy(nodeWorldPositions[idx]);
      const sz = (isRC(layout.nodes[idx]) ? SIZE.rc : SIZE.other) * SIZE.highlightMultiplier;
      ring.scale.set(sz * 1.5, sz * 1.5, 1);
      ringGroup.add(ring);
    });
  
    needsRenderRef.current = true;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCandidateEids, focusedCandidateEid, nodeIdxMap, nodeWorldPositions, layout]);

  // ── 聚焦候選：光暈 + foundation 路徑（不碰 renderer，維持原樣）──
  useEffect(() => {
    const focusGroup = focusGroupRef.current;
    const pathGroup  = pathGroupRef.current;
    const markerPts  = markerPointsRef.current;
    if (!focusGroup || !pathGroup || !layout) return;

    disposeGroupChildren(focusGroup); focusGroup.clear();
    disposeGroupChildren(pathGroup);  pathGroup.clear();
    pulsesRef.current = [];

    // 還原所有節點顏色
    if (markerPts) {
      const colorAttr = markerPts.geometry.attributes.color as THREE.BufferAttribute;
      const colorArr  = colorAttr.array as Float32Array;
      const sizeAttr  = markerPts.geometry.attributes.size as THREE.BufferAttribute;
      const sizeArr   = sizeAttr.array as Float32Array;
    
      layout.nodes.forEach((node, i) => {
        const isWalkActive = walkActiveEidSet.has(node.id);
        const shouldDim     = dimBackground && !isWalkActive;
        const c = shouldDim ? DIM_COLOR : categoryColor(node);
        const baseSize = isRC(node) ? SIZE.rc : SIZE.other;
        colorArr[i * 3] = c.r; colorArr[i * 3 + 1] = c.g; colorArr[i * 3 + 2] = c.b;
        sizeArr[i] = shouldDim ? baseSize * DIM_SIZE_MULT : baseSize;
      });

      // active candidates → 青綠
      activeCandidateEids.forEach((eid) => {
        const idx = nodeIdxMap.get(eid);
        if (idx === undefined) return;
        const c = PALETTE.activeCandidate;
        colorArr[idx * 3] = c.r; colorArr[idx * 3 + 1] = c.g; colorArr[idx * 3 + 2] = c.b;
        sizeArr[idx] *= SIZE.highlightMultiplier;
      });

      // 聚焦中的 foundation 路徑 → 琥珀色
      if (focusedCandidateEid) {
        focusedTreeData.highlightEids.forEach((eid) => {
          const idx = nodeIdxMap.get(eid);
          if (idx === undefined) return;
          const isCand = eid === focusedCandidateEid;
          const c = isCand ? PALETTE.activeCandidate : PALETTE.foundationPath;
          colorArr[idx * 3] = c.r; colorArr[idx * 3 + 1] = c.g; colorArr[idx * 3 + 2] = c.b;
          sizeArr[idx] = (isRC(layout.nodes[idx]) ? SIZE.rc : SIZE.other) * SIZE.highlightMultiplier;
        });
      }

      colorAttr.needsUpdate = true;
      sizeAttr.needsUpdate  = true;
      // ── 腦內模式：神經網格線與分區標籤一併淡出，讓激活節點更跳出 ──
      if (edgesMaterialRef.current) {
        edgesMaterialRef.current.opacity = dimBackground ? 0.03 : 0.10;
      }
      clusterLabelSpritesRef.current.forEach((spr) => {
        (spr.material as THREE.SpriteMaterial).opacity = dimBackground ? 0.06 : 0.28;
      });   
    }

    needsRenderRef.current = true;

    if (!focusedCandidateEid) return;

    // 聚焦光暈
    const focusIdx = nodeIdxMap.get(focusedCandidateEid);
    if (focusIdx !== undefined) {
      const glow = makeGlowSprite("#5fe6cc");
      glow.position.copy(nodeWorldPositions[focusIdx]);
      const baseSz = (isRC(layout.nodes[focusIdx]) ? SIZE.rc : SIZE.other) * SIZE.focusMultiplier;
      glow.scale.set(baseSz * 4, baseSz * 4, 1);
      glow.userData.baseOpacity = 1;
      focusGroup.add(glow);

      const ring = makeRingSprite("#5fe6cc", 3);
      ring.position.copy(nodeWorldPositions[focusIdx]);
      ring.scale.set(baseSz * 1.8, baseSz * 1.8, 1);
      ring.userData.baseOpacity = 0.9;
      focusGroup.add(ring);
    }

    // foundation 路徑線（琥珀色）
    const { edges } = focusedTreeData;
    if (edges.length > 0) {
      const linePos: number[] = [];
      edges.forEach((e) => {
        const si = nodeIdxMap.get(e.source);
        const ti = nodeIdxMap.get(e.target);
        if (si === undefined || ti === undefined) return;
        const sp = nodeWorldPositions[si], tp = nodeWorldPositions[ti];
        linePos.push(sp.x, sp.y, sp.z, tp.x, tp.y, tp.z);
      });
      if (linePos.length > 0) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(linePos), 3));
        const mat = new THREE.LineBasicMaterial({ color: 0xffb84d, transparent: true, opacity: 0.8 });
        pathGroup.add(new THREE.LineSegments(geo, mat));
      }

      // 接力光點
      edges.forEach((e, ei) => {
        const si = nodeIdxMap.get(e.source);
        const ti = nodeIdxMap.get(e.target);
        if (si === undefined || ti === undefined) return;
        const from = nodeWorldPositions[si].clone();
        const to   = nodeWorldPositions[ti].clone();
        const dot  = makeFlatDotSprite("#ffb84d");
        pathGroup.add(dot);
        pulsesRef.current.push({ from, to, dot, phase: (ei / edges.length) % 1 });
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusedCandidateEid, focusedTreeData, activeCandidateEids, nodeIdxMap, nodeWorldPositions, layout, dimBackground, walkActiveEidSet]);
  useEffect(() => {
    const walkGroup = walkGroupRef.current;
    if (!walkGroup || !layout) return;
  
    disposeGroupChildren(walkGroup);
    walkGroup.clear();
    walkPulsesRef.current = [];
  
    if (!walkIllumination || walkIllumination.length === 0) {
      needsRenderRef.current = true;
      return;
    }
  
    walkIllumination.forEach((wp) => {
      const idxs = wp.visitedEids
        .map((eid) => nodeIdxMap.get(eid))
        .filter((i): i is number => i !== undefined);
      if (idxs.length === 0) return;
  
      const isDead = wp.status === "dead_end";
  
      // ── 主幹：雙層線（外層柔光模擬加粗 + 內層高飽和實線）──
      if (idxs.length > 1) {
        const pts: number[] = [];
        for (let i = 0; i < idxs.length - 1; i++) {
          const a = nodeWorldPositions[idxs[i]], b = nodeWorldPositions[idxs[i + 1]];
          pts.push(a.x, a.y, a.z, b.x, b.y, b.z);
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  
        const glowMat = new THREE.LineBasicMaterial({
          color: wp.color, transparent: true, opacity: isDead ? 0.08 : 0.28,
          blending: THREE.AdditiveBlending, depthWrite: false,
        });
        walkGroup.add(new THREE.LineSegments(geo, glowMat));
  
        const coreMat = new THREE.LineBasicMaterial({
          color: wp.color, transparent: true, opacity: isDead ? 0.3 : 0.95,
        });
        walkGroup.add(new THREE.LineSegments(geo, coreMat));
  
        if (!isDead) {
          // 沿線加密光點，加強「粗線」的視覺密度
          for (let i = 0; i < idxs.length - 1; i++) {
            const from = nodeWorldPositions[idxs[i]];
            const to   = nodeWorldPositions[idxs[i + 1]];
            [0.3, 0.6, 0.85].forEach((t) => {
              const dot = makeGlowSprite(wp.color);
              dot.position.lerpVectors(from, to, t);
              dot.scale.set(0.12, 0.12, 1);
              (dot.material as THREE.SpriteMaterial).opacity = 0.35;
              walkGroup.add(dot);
            });
          }
          // 接力光點（動畫）
          for (let i = 0; i < idxs.length - 1; i++) {
            const from = nodeWorldPositions[idxs[i]].clone();
            const to   = nodeWorldPositions[idxs[i + 1]].clone();
            const dot  = makeFlatDotSprite(wp.color);
            walkGroup.add(dot);
            walkPulsesRef.current.push({ from, to, dot, phase: (i / (idxs.length - 1)) % 1 });
          }
        }
      }
  
      // ── 每個走過的節點：常駐編號標籤（新→舊 opacity 漸層，最低 0.32）──
      idxs.forEach((idx, i) => {
        const isStart = i === 0;
        const pos     = nodeWorldPositions[idx];
        const node    = layout.nodes[idx];
  
        if (isStart) {
          const startRing = makeHollowDotSprite(wp.color, 2.5);
          startRing.position.copy(pos);
          startRing.scale.set(0.05, 0.05, 1);
          walkGroup.add(startRing);
        }
  
        const rawLabel   = node?.name ?? wp.visitedEids[i];
        const shortLabel = rawLabel.length > 14 ? rawLabel.slice(0, 13) + "…" : rawLabel;
        const label = makeLabelSprite(`${circledNum(i + 1)} ${shortLabel}`, wp.color);
        const minOpacity = 0.32;
        const opacity = idxs.length > 1
          ? minOpacity + (1 - minOpacity) * (i / (idxs.length - 1))
          : 1;
        (label.material as THREE.SpriteMaterial).opacity = isDead ? opacity * 0.5 : opacity;
        label.position.set(pos.x, pos.y + 0.16, pos.z);
        label.scale.set(0.46, 0.1, 1);
        walkGroup.add(label);
      });
  
      // ── 目前節點：光暈 + 環 ──
      const curIdx = idxs[idxs.length - 1];
      const curPos = nodeWorldPositions[curIdx];
      if (!isDead) {
        const glow = makeGlowSprite(wp.color);
        glow.position.copy(curPos);
        glow.scale.set(0.5, 0.5, 1);
        walkGroup.add(glow);
      }
      const ring = makeRingSprite(wp.color, isDead ? 2 : 4);
      ring.position.copy(curPos);
      ring.scale.set(0.24, 0.24, 1);
      if (isDead) (ring.material as THREE.SpriteMaterial).opacity = 0.3;
      walkGroup.add(ring);
  
      // ── 死路：加上 ✕ 標記 ──
      if (isDead) {
        const cross = makeCrossSprite("#e0665f");
        cross.position.copy(curPos);
        cross.scale.set(0.09, 0.09, 1);
        walkGroup.add(cross);
      }
  
      // ── 歷史分支（樹突，淡）── 不變
      wp.frontierHistory.forEach((step) => {
        const fromIdx = nodeIdxMap.get(step.fromEid);
        if (fromIdx === undefined) return;
        const fromPos = nodeWorldPositions[fromIdx];
        step.candidates.forEach((c) => {
          const idx = nodeIdxMap.get(c.eid);
          if (idx === undefined) return;
          const toPos = nodeWorldPositions[idx];
          const geo = new THREE.BufferGeometry().setFromPoints([fromPos, toPos]);
          const mat = new THREE.LineDashedMaterial({
            color: wp.color, dashSize: 0.04, gapSize: 0.03, transparent: true, opacity: 0.16,
          });
          const line = new THREE.Line(geo, mat);
          line.computeLineDistances();
          walkGroup.add(line);
          const hollow = makeHollowDotSprite(wp.color, 2);
          hollow.position.copy(toPos);
          hollow.scale.set(0.03, 0.03, 1);
          walkGroup.add(hollow);
        });
      });
  
      // ── 目前這一步的候選（較亮）── 不變
      if (wp.status === "active") {
        wp.frontierPreview.forEach((c) => {
          const idx = nodeIdxMap.get(c.eid);
          if (idx === undefined) return;
          const toPos = nodeWorldPositions[idx];
          const geo = new THREE.BufferGeometry().setFromPoints([curPos, toPos]);
          const mat = new THREE.LineDashedMaterial({
            color: wp.color, dashSize: 0.05, gapSize: 0.04, transparent: true, opacity: 0.5,
          });
          const line = new THREE.Line(geo, mat);
          line.computeLineDistances();
          walkGroup.add(line);
          const hollow = makeHollowDotSprite(wp.color, 3);
          hollow.position.copy(toPos);
          hollow.scale.set(0.04, 0.04, 1);
          walkGroup.add(hollow);
        });
      }
    });
  
    needsRenderRef.current = true;
  }, [walkIllumination, nodeIdxMap, nodeWorldPositions, layout]);
  // ── Zoom buttons ───────────────────────────────────────────
  function zoomBy(delta: number) {
    const cam = cameraRef.current;
    if (!cam) return;
    cam.position.z = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, cam.position.z + delta));
    needsRenderRef.current = true;
  }

  // ─────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────

  return (
    <div style={{
      position: "relative", width, height,
      borderRadius: "16px", overflow: "hidden",
      background: PALETTE.bg_css, flexShrink: 0,
    }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block", width: "100%", height: "100%", cursor: "crosshair" }}
      />

      {/* Tooltip */}
      <div ref={tooltipRef} style={{
        position:      "absolute",
        pointerEvents: "none",
        background:    "rgba(13,18,32,0.92)",
        border:        "1px solid rgba(232,238,248,0.2)",
        borderRadius:  "6px",
        padding:       "4px 10px",
        fontSize:      "11px",
        color:         "#e8eef8",
        fontFamily:    "monospace",
        whiteSpace:    "nowrap",
        maxWidth:      "240px",
        overflow:      "hidden",
        textOverflow:  "ellipsis",
        opacity:       0,
        transition:    "opacity 0.12s ease",
        zIndex:        10,
        top:           0,
        left:          0,
      }} />

      {/* 縮放按鈕 */}
      <div style={{
        position: "absolute", bottom: 12, right: 12,
        display: "flex", flexDirection: "column", gap: 4,
        pointerEvents: "auto",
      }}>
        {[{ label: "+", delta: -1.2 }, { label: "−", delta: 1.2 }].map(({ label, delta }) => (
          <button
            key={label}
            onClick={() => zoomBy(delta)}
            style={{
              width: 28, height: 28, padding: 0,
              border: "1px solid rgba(232,238,248,0.2)",
              borderRadius: 6,
              background: "rgba(13,18,32,0.75)",
              color: "#e8eef8",
              fontSize: "16px",
              lineHeight: "1",
              cursor: "pointer",
              fontFamily: "monospace",
              backdropFilter: "blur(4px)",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* 標題 */}
      <div style={{
        position:      "absolute",
        top:           10,
        left:          12,
        fontSize:      "10px",
        color:         "rgba(232,238,248,0.4)",
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        fontFamily:    "monospace",
        pointerEvents: "none",
        userSelect:    "none",
      }}>
        Global Knowledge Field
      </div>

      {/* 圖例（三層）*/}
      <div style={{
        position:      "absolute",
        top:           10,
        right:         12,
        display:       "flex",
        flexDirection: "column",
        gap:           "4px",
        pointerEvents: "none",
        userSelect:    "none",
      }}>
        {([
          { fill: "#00e8ff", label: "核心概念",   hint: "article" },
          { fill: "#b48cff", label: "教科書概念", hint: "chapter" },
          { fill: "#ff9a52", label: "論文概念",   hint: "paper"   },
          { fill: "#ff5fd1", label: "候選命中",   hint: "" },   
          { fill: "#ffb84d", label: "聚焦路徑",   hint: ""        },
        ] as { fill: string; label: string; hint: string }[]).map(({ fill, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <svg width="9" height="9" viewBox="0 0 9 9">
              <circle cx="4.5" cy="4.5" r="3.5" fill={fill} stroke="#e8eef8" strokeWidth="0.75" />
            </svg>
            <span style={{
              fontSize:      "9px",
              color:         fill,
              fontFamily:    "monospace",
              letterSpacing: "0.08em",
            }}>
              {label}
            </span>
          </div>
        ))}
      </div>

      {/* Active 候選計數 / 聚焦狀態 */}
      {activeCandidateEids.length > 0 && (
        <div style={{
          position:      "absolute",
          bottom:        10,
          left:          12,
          fontSize:      "10px",
          color: focusedCandidateEid ? "#5fe6cc" : "#ff5fd1",
          letterSpacing: "0.1em",
          fontFamily:    "monospace",
          pointerEvents: "none",
          userSelect:    "none",
          display:       "flex",
          alignItems:    "center",
          gap:           "6px",
        }}>
          <span style={{
            display:      "inline-block",
            width:        6,
            height:       6,
            borderRadius: "50%",
            background: focusedCandidateEid ? "#5fe6cc" : "#ff5fd1",
            animation:    "gbrain-pulse 1.8s ease-in-out infinite",
          }} />
          {focusedCandidateEid
            ? "聚焦中 · 點其他候選或空白處切換"
            : `${activeCandidateEids.length} region${activeCandidateEids.length > 1 ? "s" : ""} activated`}
        </div>
      )}

      {!layout && (
        <div style={{
          position:      "absolute",
          inset:         0,
          display:       "flex",
          alignItems:    "center",
          justifyContent:"center",
          color:         "rgba(232,238,248,0.3)",
          fontSize:      "11px",
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          fontFamily:    "monospace",
          pointerEvents: "none",
        }}>
          知識場域載入中…
        </div>
      )}

      <style>{`
        @keyframes gbrain-pulse { 0%,100%{opacity:1}50%{opacity:0.25} }
      `}</style>
    </div>
  );
}
