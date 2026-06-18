"use client";

/**
 * GlobalBrainExplorer.tsx（白底 / 精準互動版）
 *
 * 這次改版重點：
 *   1. 背景改白，渲染從「黑底發光粒子雲」改成「實心填色 + 深色描邊」的
 *      扁平標記，移除 additive blending / EffectComposer + UnrealBloomPass
 *      （白底下 bloom 沒有效果，索性整套拿掉，順便減少 bundle 跟運算量）。
 *   2. 點擊／hover 判定改成「螢幕空間最近點搜尋」（worldToScreenPx +
 *      findNearestNodeIndex），取代固定世界座標半徑的 proxy sphere +
 *      raycaster。點擊熱區永遠對齊視覺位置，且不受縮放距離影響。
 *   3. 新增滑鼠滾輪／雙指 pinch 縮放，camera.position.z 在
 *      ZOOM_MIN ~ ZOOM_MAX 之間夾限；右下角也有 +/− 按鈕。
 *   4. 新增 hover 高亮外框，點擊前先預覽會點到哪一顆。
 *   5. 新增 foundation_tree 完整路徑視覺化：吃 rcFoundationMap 裡的
 *      treeNodes / treeEdges（後端 GraphProjector 早就有送，只是先前
 *      page.tsx 沒接），把候選（樹梢）到 foundation（根系）的完整鏈路
 *      用琥珀色描邊 + 接力光點呈現，方向沿用後端 BFS 既有的「上游→下游」，
 *      不需反轉。
 *   6. 新增分區浮動標籤（source × 是否為 Resulting_Concept，最多四群）
 *      與即時百分比；用既有 layout.edges 畫神經網格細線。
 *
 * 與舊版的相容性：
 *   - GlobalLayoutNode / GlobalLayoutEdge / GlobalLayout 型別維持不變。
 *   - Props 只多了一個可選的 rcFoundationMap，其餘參數與呼叫方式不變。
 */

import { useEffect, useRef, useMemo } from "react";
import * as THREE from "three";
import type { RCFoundationMap } from "./DetailPanel";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type GlobalLayoutNode = {
  id:      string;
  name:    string;
  type:    string;
  source?: string;
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

type Props = {
  layout:              GlobalLayout | null;
  activeCandidateEids: string[];          // SocratesChatBot 命中的候選
  rcFoundationMap?:    RCFoundationMap;   // 候選 RC eid → 完整 foundation 路徑
  onNodeClick?:        (node: GlobalLayoutNode) => void;
  width?:              number;
  height?:             number;
};

type PulseInfo = {
  from:  THREE.Vector3;
  to:    THREE.Vector3;
  dot:   THREE.Sprite;
  phase: number;
};

// ─────────────────────────────────────────────────────────────
// Design tokens（白底版）
// ─────────────────────────────────────────────────────────────

const PALETTE = {
  bg:     0xffffff,
  bg_css: "#ffffff",

  resultsRC:    new THREE.Color(0xff6010),
  resultsOther: new THREE.Color(0xb03800),
  chapterRC:    new THREE.Color(0x7020b0),  // 原本的「次要紫」改當主色，白底對比較夠
  chapterOther: new THREE.Color(0x4a1570),  // 更深的紫，給次要群

  activeCandidate: new THREE.Color(0x009e82),  // 比 #00d4aa 深一點，白底對比較好
  foundationPath:  new THREE.Color(0xe8a030),  // 琥珀色，foundation／路徑節點高亮

  strokeCss: "#1a2840",
};

type ClusterKey = "results_rc" | "results_other" | "chapter_rc" | "chapter_other";

const CLUSTER_COLOR: Record<ClusterKey, THREE.Color> = {
  results_rc:    PALETTE.resultsRC,
  results_other: PALETTE.resultsOther,
  chapter_rc:    PALETTE.chapterRC,
  chapter_other: PALETTE.chapterOther,
};

const CLUSTER_LABEL: Record<ClusterKey, string> = {
  results_rc:    "成果概念 · 外殼",
  results_other: "成果支線 · 外殼",
  chapter_rc:    "核心概念 · 內殼",
  chapter_other: "基礎概念 · 內殼",
};

const CLUSTER_LABEL_COLOR: Record<ClusterKey, string> = {
  results_rc:    "#c24a08",
  results_other: "#8a2c00",
  chapter_rc:    "#5a1888",
  chapter_other: "#3a1058",
};

const SIZE = {
  rc:                  0.16,
  other:               0.10,
  highlightMultiplier: 1.55,
};

const WORLD_SCALE       = 2.2;
const ZOOM_MIN          = 2.0;
const ZOOM_MAX          = 9.5;
const ZOOM_DEFAULT      = 5.5;
const PICK_PIXEL_RADIUS = 16;     // 點擊／hover 容許的螢幕像素半徑
const LABEL_Y_OFFSET    = 0.34;

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function nodeToWorld(n: GlobalLayoutNode): THREE.Vector3 {
  return new THREE.Vector3(n.x * WORLD_SCALE, n.y * WORLD_SCALE, n.z * WORLD_SCALE);
}

function isRC(node: GlobalLayoutNode): boolean {
  return (node.type || "").toLowerCase().includes("resulting");
}

function clusterKeyOf(node: GlobalLayoutNode): ClusterKey {
  const src = (node.source || "results").toLowerCase();
  const rc  = isRC(node);
  if (src === "chapter") return rc ? "chapter_rc" : "chapter_other";
  return rc ? "results_rc" : "results_other";
}

function categoryColor(node: GlobalLayoutNode): THREE.Color {
  return CLUSTER_COLOR[clusterKeyOf(node)];
}

function roundRectPath(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function makeLabelSprite(text: string, colorHex: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 320; canvas.height = 72;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgba(255,255,255,0.88)";
  roundRectPath(ctx, 2, 2, canvas.width - 4, canvas.height - 4, 14);
  ctx.fill();
  ctx.strokeStyle = "rgba(20,30,50,0.18)";
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

function makeRingSprite(colorHex: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.lineWidth = 5;
  ctx.strokeStyle = colorHex;
  ctx.beginPath();
  ctx.arc(32, 32, 24, 0, Math.PI * 2);
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  return new THREE.Sprite(mat);
}

function makeFlatDotSprite(colorHex: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 32;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = colorHex;
  ctx.beginPath();
  ctx.arc(16, 16, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = PALETTE.strokeCss;
  ctx.lineWidth = 2;
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.05, 0.05, 1);
  return spr;
}

function disposeSceneChild(child: THREE.Object3D) {
  if (child instanceof THREE.Sprite) {
    child.material.map?.dispose();
    child.material.dispose();
  } else if (child instanceof THREE.Line) {
    child.geometry.dispose();
    (child.material as THREE.Material).dispose();
  }
}

function disposeGroupChildren(group: THREE.Group) {
  group.children.forEach(disposeSceneChild);
}

// ─────────────────────────────────────────────────────────────
// GlobalBrainExplorer
// ─────────────────────────────────────────────────────────────

export default function GlobalBrainExplorer({
  layout,
  activeCandidateEids,
  rcFoundationMap,
  onNodeClick,
  width  = 480,
  height = 480,
}: Props) {
  const canvasRef  = useRef<HTMLCanvasElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  const sceneRef        = useRef<THREE.Scene | null>(null);
  const rendererRef     = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef       = useRef<THREE.PerspectiveCamera | null>(null);
  const rotGroupRef     = useRef<THREE.Group | null>(null);
  const markerPointsRef = useRef<THREE.Points | null>(null);
  const pathGroupRef    = useRef<THREE.Group | null>(null);
  const ringGroupRef    = useRef<THREE.Group | null>(null);
  const hoverRingRef    = useRef<THREE.Sprite | null>(null);
  const animFrameRef    = useRef<number>(0);

  const isDragging     = useRef(false);
  const prevMouse      = useRef({ x: 0, y: 0 });
  const mousePixel     = useRef({ x: -9999, y: -9999 });
  const pinchStartDist = useRef(0);
  const pinchStartZ    = useRef(ZOOM_DEFAULT);
  const pulsesRef      = useRef<PulseInfo[]>([]);
  const hoveredIdxRef  = useRef<number | null>(null);

  const nodeIdxMap = useMemo(() => {
    if (!layout) return new Map<string, number>();
    return new Map(layout.nodes.map((n, i) => [n.id, i]));
  }, [layout]);

  const nodeWorldPositions = useMemo(() => {
    if (!layout) return [] as THREE.Vector3[];
    return layout.nodes.map((n) => nodeToWorld(n));
  }, [layout]);

  const clusterStats = useMemo(() => {
    if (!layout || layout.nodes.length === 0) return [];
    const groups: Record<ClusterKey, { count: number; sum: THREE.Vector3 }> = {
      results_rc:    { count: 0, sum: new THREE.Vector3() },
      results_other: { count: 0, sum: new THREE.Vector3() },
      chapter_rc:    { count: 0, sum: new THREE.Vector3() },
      chapter_other: { count: 0, sum: new THREE.Vector3() },
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

  // 統整目前要高亮的 foundation／路徑節點與邊（候選 + 完整 foundation_tree）
  const activeTreeData = useMemo(() => {
    if (!rcFoundationMap || activeCandidateEids.length === 0) {
      return { highlightEids: new Set<string>(), edges: [] as { source: string; target: string }[] };
    }
    const highlightEids = new Set<string>();
    const edgeMap = new Map<string, { source: string; target: string }>();
    activeCandidateEids.forEach((eid) => {
      const entry = rcFoundationMap.get(eid);
      if (!entry) return;
      highlightEids.add(eid);
      entry.foundationEids.forEach((f) => highlightEids.add(f));
      (entry.treeNodes ?? []).forEach((n) => highlightEids.add(n.element_id));
      (entry.treeEdges ?? []).forEach((e) => {
        const key = `${e.source}__${e.target}`;
        if (!edgeMap.has(key)) edgeMap.set(key, e);
      });
    });
    return { highlightEids, edges: [...edgeMap.values()] };
  }, [rcFoundationMap, activeCandidateEids]);

  // ── Init scene（layout / width / height 變動才重建）──────────
  useEffect(() => {
    if (!canvasRef.current || !layout || layout.nodes.length === 0) return;

    const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(PALETTE.bg, 1);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 100);
    camera.position.set(0, 0, ZOOM_DEFAULT);
    cameraRef.current = camera;

    const rg = new THREE.Group();
    scene.add(rg);
    rotGroupRef.current = rg;

    // ── 節點標記：一顆概念一個點，實心填色 + 深色描邊 ──────────
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
          if (d > 0.42) {
            gl_FragColor = vec4(0.102, 0.157, 0.251, 1.0);
          } else {
            gl_FragColor = vec4(vColor, 1.0);
          }
        }`,
      transparent: false,
      depthWrite:  true,
    });

    const markerPoints = new THREE.Points(markerGeo, markerMat);
    rg.add(markerPoints);
    markerPointsRef.current = markerPoints;

    // ── 神經網格：既有 layout.edges，畫成低透明度細線 ──────────
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
        const meshMat = new THREE.LineBasicMaterial({ color: 0x1a2840, transparent: true, opacity: 0.06 });
        rg.add(new THREE.LineSegments(meshGeo, meshMat));
      }
    }

    // ── 分區浮動標籤 ───────────────────────────────────────────
    clusterStats.forEach((c) => {
      const label = makeLabelSprite(`${CLUSTER_LABEL[c.key]} ${c.pct.toFixed(0)}%`, CLUSTER_LABEL_COLOR[c.key]);
      label.position.set(c.centroid.x, c.centroid.y + LABEL_Y_OFFSET, c.centroid.z);
      rg.add(label);
    });

    // ── path group（foundation 路徑線 + 接力光點）─────────────
    const pathGroup = new THREE.Group();
    rg.add(pathGroup);
    pathGroupRef.current = pathGroup;

    // ── ring group（active candidate 標示環）───────────────────
    const ringGroup = new THREE.Group();
    rg.add(ringGroup);
    ringGroupRef.current = ringGroup;

    // ── hover 高亮外框 ─────────────────────────────────────────
    const hoverRing = makeRingSprite("#2a4a8a");
    hoverRing.visible = false;
    hoverRing.scale.set(0.22, 0.22, 1);
    rg.add(hoverRing);
    hoverRingRef.current = hoverRing;

    scene.add(new THREE.AmbientLight(0xffffff, 1));

    // ── 螢幕空間點擊判定 helper（定義在 useEffect 內，讀 refs）──
    function worldToScreenPx(worldPos: THREE.Vector3): { x: number; y: number } | null {
      const cam = cameraRef.current;
      if (!cam) return null;
      const proj = worldPos.clone().project(cam);
      if (Math.abs(proj.z) > 1) return null;
      return {
        x: ((proj.x + 1) / 2) * width,
        y: ((1 - proj.y) / 2) * height,
      };
    }

    function findNearestNodeIndex(px: number, py: number): number | null {
      if (!rotGroupRef.current || !layout) return null;
      let bestDist = PICK_PIXEL_RADIUS * PICK_PIXEL_RADIUS;
      let bestIdx: number | null = null;
      layout.nodes.forEach((_, i) => {
        const wp = nodeWorldPositions[i].clone();
        rotGroupRef.current!.localToWorld(wp);
        const sp = worldToScreenPx(wp);
        if (!sp) return;
        const dx = sp.x - px, dy = sp.y - py;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDist) { bestDist = d2; bestIdx = i; }
      });
      return bestIdx;
    }

    // ── Animation loop ─────────────────────────────────────────
    let frameCount = 0;
    function animate() {
      animFrameRef.current = requestAnimationFrame(animate);
      frameCount++;

      // 自動慢旋
      if (rotGroupRef.current && !isDragging.current) {
        rotGroupRef.current.rotation.y += 0.0008;
        rotGroupRef.current.rotation.x += 0.0002;
      }

      // hover 更新（每 2 幀做一次，節省開銷）
      if (frameCount % 2 === 0) {
        const { x, y } = mousePixel.current;
        const idx = findNearestNodeIndex(x, y);
        hoveredIdxRef.current = idx;
        const hRing = hoverRingRef.current;
        if (hRing) {
          if (idx !== null) {
            const wp = nodeWorldPositions[idx].clone();
            rotGroupRef.current!.localToWorld(wp);
            hRing.position.copy(wp);
            hRing.visible = true;
            if (tooltipRef.current) {
              const sp = worldToScreenPx(wp);
              if (sp) {
                tooltipRef.current.textContent = layout.nodes[idx].name;
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
      }

      // 接力光點動畫
      const pulses = pulsesRef.current;
      if (pulses.length > 0) {
        pulses.forEach((p) => {
          p.phase = (p.phase + 0.012) % 1;
          p.dot.position.lerpVectors(p.from, p.to, p.phase);
        });
      }

      renderer.render(scene, camera);
    }
    animate();

    // ── 滑鼠事件 ───────────────────────────────────────────────
    const canvas = canvasRef.current!;

    function getBoundedRect() { return canvas.getBoundingClientRect(); }

    let mouseDownPos = { x: 0, y: 0 };

    function onMouseDown(e: MouseEvent) {
      isDragging.current = true;
      prevMouse.current  = { x: e.clientX, y: e.clientY };
      mouseDownPos       = { x: e.clientX, y: e.clientY };
    }
    function onMouseUp(e: MouseEvent) {
      const dx = Math.abs(e.clientX - mouseDownPos.x);
      const dy = Math.abs(e.clientY - mouseDownPos.y);
      if (dx < 4 && dy < 4 && onNodeClick) {
        const r   = getBoundedRect();
        const px  = e.clientX - r.left;
        const py  = e.clientY - r.top;
        const idx = findNearestNodeIndex(px, py);
        if (idx !== null) onNodeClick(layout.nodes[idx]);
      }
      isDragging.current = false;
    }
    function onMouseMove(e: MouseEvent) {
      const r = getBoundedRect();
      mousePixel.current = { x: e.clientX - r.left, y: e.clientY - r.top };
      if (!isDragging.current || !rotGroupRef.current) return;
      rotGroupRef.current.rotation.y += (e.clientX - prevMouse.current.x) * 0.008;
      rotGroupRef.current.rotation.x += (e.clientY - prevMouse.current.y) * 0.008;
      prevMouse.current = { x: e.clientX, y: e.clientY };
    }
    function onMouseLeave() {
      isDragging.current = false;
      mousePixel.current = { x: -9999, y: -9999 };
      if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
      if (hoverRingRef.current) hoverRingRef.current.visible = false;
    }

    // ── 縮放：滑鼠滾輪 ─────────────────────────────────────────
    function onWheel(e: WheelEvent) {
      e.preventDefault();
      const cam = cameraRef.current;
      if (!cam) return;
      cam.position.z = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, cam.position.z + e.deltaY * 0.01));
    }

    // ── 縮放：觸控 pinch ────────────────────────────────────────
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
    }
    function onTouchEnd() { isDragging.current = false; }

    canvas.addEventListener("mousedown",  onMouseDown);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("wheel",      onWheel, { passive: false });
    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("mouseup",    onMouseUp);
    window.addEventListener("mousemove",  onMouseMove);
    window.addEventListener("touchmove",  onTouchMove, { passive: true });
    window.addEventListener("touchend",   onTouchEnd);

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
      renderer.dispose();
      markerGeo.dispose();
      markerMat.dispose();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, width, height, nodeWorldPositions, clusterStats, nodeIdxMap]);

  // ── Active candidate 標示環 + foundation 路徑（資料變動時重建）
  useEffect(() => {
    const rg        = rotGroupRef.current;
    const ringGroup = ringGroupRef.current;
    const pathGroup = pathGroupRef.current;
    if (!rg || !ringGroup || !pathGroup || !layout) return;

    // 清除舊有
    disposeGroupChildren(ringGroup); ringGroup.clear();
    disposeGroupChildren(pathGroup); pathGroup.clear();
    pulsesRef.current = [];

    // ── Active candidate 青色外環 ────────────────────────────────
    activeCandidateEids.forEach((eid) => {
      const idx = nodeIdxMap.get(eid);
      if (idx === undefined) return;
      const ring = makeRingSprite("#009e82");
      ring.position.copy(nodeWorldPositions[idx]);
      const sz = (isRC(layout.nodes[idx]) ? SIZE.rc : SIZE.other) * SIZE.highlightMultiplier;
      ring.scale.set(sz * 1.5, sz * 1.5, 1);
      ringGroup.add(ring);
    });

    // ── Foundation 路徑線 + 接力光點（activeTreeData から）──────
    const { highlightEids, edges } = activeTreeData;

    // 路徑節點改成琥珀色小標記（覆寫 markerPoints 的顏色）
    const markerPts = markerPointsRef.current;
    if (markerPts) {
      const colorAttr = markerPts.geometry.attributes.color as THREE.BufferAttribute;
      const colorArr  = colorAttr.array as Float32Array;
      // 先還原所有顏色
      layout.nodes.forEach((node, i) => {
        const c = categoryColor(node);
        colorArr[i * 3] = c.r; colorArr[i * 3 + 1] = c.g; colorArr[i * 3 + 2] = c.b;
      });
      // 再高亮路徑節點
      highlightEids.forEach((eid) => {
        const idx = nodeIdxMap.get(eid);
        if (idx === undefined) return;
        const c = activeCandidateEids.includes(eid) ? PALETTE.activeCandidate : PALETTE.foundationPath;
        colorArr[idx * 3] = c.r; colorArr[idx * 3 + 1] = c.g; colorArr[idx * 3 + 2] = c.b;
      });
      colorAttr.needsUpdate = true;

      // 同時放大路徑節點的 size
      const sizeAttr = markerPts.geometry.attributes.size as THREE.BufferAttribute;
      const sizeArr  = sizeAttr.array as Float32Array;
      layout.nodes.forEach((node, i) => { sizeArr[i] = isRC(node) ? SIZE.rc : SIZE.other; });
      highlightEids.forEach((eid) => {
        const idx = nodeIdxMap.get(eid);
        if (idx === undefined) return;
        sizeArr[idx] *= SIZE.highlightMultiplier;
      });
      sizeAttr.needsUpdate = true;
    }

    // 路徑邊線（琥珀色描線）
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
        const mat = new THREE.LineBasicMaterial({ color: 0xe8a030, transparent: true, opacity: 0.75 });
        pathGroup.add(new THREE.LineSegments(geo, mat));
      }

      // 接力光點：每段 edge 一顆，phase 偏移讓流動感更連貫
      edges.forEach((e, ei) => {
        const si = nodeIdxMap.get(e.source);
        const ti = nodeIdxMap.get(e.target);
        if (si === undefined || ti === undefined) return;
        const from = nodeWorldPositions[si].clone();
        const to   = nodeWorldPositions[ti].clone();
        const dot  = makeFlatDotSprite("#e8a030");
        pathGroup.add(dot);
        pulsesRef.current.push({ from, to, dot, phase: (ei / edges.length) % 1 });
      });
    }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCandidateEids, activeTreeData, nodeIdxMap, nodeWorldPositions, layout]);

  // ── Zoom 按鈕 handler ──────────────────────────────────────
  function zoomBy(delta: number) {
    const cam = cameraRef.current;
    if (!cam) return;
    cam.position.z = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, cam.position.z + delta));
  }

  // ─────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────

  return (
    <div style={{ position: "relative", width, height, borderRadius: "16px", overflow: "hidden", background: PALETTE.bg_css, flexShrink: 0 }}>
      <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%", cursor: "crosshair" }} />

      {/* Tooltip */}
      <div ref={tooltipRef} style={{
        position:      "absolute",
        pointerEvents: "none",
        background:    "rgba(255,255,255,0.92)",
        border:        "1px solid rgba(26,40,64,0.25)",
        borderRadius:  "6px",
        padding:       "4px 10px",
        fontSize:      "11px",
        color:         "#1a2840",
        fontFamily:    "monospace",
        whiteSpace:    "nowrap",
        maxWidth:      "220px",
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
              border: "1px solid rgba(26,40,64,0.25)",
              borderRadius: 6,
              background: "rgba(255,255,255,0.85)",
              color: "#1a2840",
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
        color:         "rgba(26,40,64,0.35)",
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        fontFamily:    "monospace",
        pointerEvents: "none",
        userSelect:    "none",
      }}>
        Global Knowledge Field
      </div>

      {/* 圖例 */}
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
          { fill: "#ff6010", label: "成果概念" },
          { fill: "#b03800", label: "成果支線" },
          { fill: "#7020b0", label: "核心概念" },
          { fill: "#4a1570", label: "基礎支線" },
          { fill: "#009e82", label: "候選命中" },
          { fill: "#e8a030", label: "推理路徑" },
        ] as { fill: string; label: string }[]).map(({ fill, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <svg width="9" height="9" viewBox="0 0 9 9">
              <circle cx="4.5" cy="4.5" r="3.5" fill={fill} stroke="#1a2840" strokeWidth="1" />
            </svg>
            <span style={{ fontSize: "9px", color: fill, fontFamily: "monospace", letterSpacing: "0.08em" }}>{label}</span>
          </div>
        ))}
      </div>

      {/* Active 候選計數 */}
      {activeCandidateEids.length > 0 && (
        <div style={{
          position:      "absolute",
          bottom:        10,
          left:          12,
          fontSize:      "10px",
          color:         "#009e82",
          letterSpacing: "0.1em",
          fontFamily:    "monospace",
          pointerEvents: "none",
          userSelect:    "none",
          display:       "flex",
          alignItems:    "center",
          gap:           "6px",
        }}>
          <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "#009e82", animation: "gbrain-pulse 1.8s ease-in-out infinite" }} />
          {activeCandidateEids.length} region{activeCandidateEids.length > 1 ? "s" : ""} activated
        </div>
      )}

      {!layout && (
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "rgba(26,40,64,0.3)", fontSize: "11px",
          letterSpacing: "0.12em", textTransform: "uppercase",
          fontFamily: "monospace", pointerEvents: "none",
        }}>
          知識場域載入中…
        </div>
      )}

      <style>{`@keyframes gbrain-pulse { 0%,100%{opacity:1}50%{opacity:0.25} }`}</style>
    </div>
  );
}
