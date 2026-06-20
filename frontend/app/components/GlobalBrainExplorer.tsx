"use client";

/**
 * GlobalBrainExplorer.tsx（三層殼層版）
 *
 * 殼層隱喻（由核心向外）：
 *   article  → 內核  — 青白色，最核心的基礎概念
 *   chapter  → 中層  — 紫色，教科書概念
 *   paper    → 外層  — 暖橘色，論文專精概念
 *
 * 互動邏輯不變（聚焦候選、foundation 路徑、drag-to-rotate 等）。
 */

import { useEffect, useRef, useMemo, useState } from "react";
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
  activeCandidateEids: string[];
  rcFoundationMap?:    RCFoundationMap;
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

  // 三層概念色
  article: new THREE.Color(0x00e8ff),   // 青白 — 內核
  chapter: new THREE.Color(0xb48cff),   // 紫   — 中層
  paper:   new THREE.Color(0xff9a52),   // 暖橘 — 外層

  activeCandidate: new THREE.Color(0x00e0b8),  // 青綠 — 命中候選
  focusGlow:       new THREE.Color(0x5fe6cc),  // 聚焦光暈
  foundationPath:  new THREE.Color(0xffb84d),  // 琥珀 — 聚焦路徑

  strokeCss: "#e8eef8",
};

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

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function nodeToWorld(n: GlobalLayoutNode): THREE.Vector3 {
  return new THREE.Vector3(n.x * WORLD_SCALE, n.y * WORLD_SCALE, n.z * WORLD_SCALE);
}

function categoryColor(node: GlobalLayoutNode): THREE.Color {
  return CLUSTER_COLOR[clusterKeyOf(node)];
}

function roundRectPath(
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

function makeLabelSprite(text: string, colorHex: string): THREE.Sprite {
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

function makeRingSprite(colorHex: string, lineWidth = 5): THREE.Sprite {
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

function makeGlowSprite(colorHex: string): THREE.Sprite {
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

  const activeCandidateSetRef = useRef(activeCandidateSet);
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

  // ── Init scene ─────────────────────────────────────────────
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

    // 背景星點
    const bgCount = 1200;
    const bgPos   = new Float32Array(bgCount * 3);
    for (let i = 0; i < bgCount; i++) {
      bgPos[i * 3 + 0] = (Math.random() * 2 - 1) * 9;
      bgPos[i * 3 + 1] = (Math.random() * 2 - 1) * 9;
      bgPos[i * 3 + 2] = (Math.random() * 2 - 1) * 9;
    }
    const bgGeo = new THREE.BufferGeometry();
    bgGeo.setAttribute("position", new THREE.BufferAttribute(bgPos, 3));
    scene.add(new THREE.Points(bgGeo, new THREE.PointsMaterial({
      color: 0x2a3550, size: 0.01, transparent: true, opacity: 0.35, depthWrite: false,
    })));

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
    rg.add(markerPoints);
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
        rg.add(new THREE.LineSegments(meshGeo, meshMat));
      }
    }

    // 分區浮動標籤（三層）
    clusterStats.forEach((c) => {
      const label = makeLabelSprite(
        `${CLUSTER_LABEL[c.key]} ${c.pct.toFixed(0)}%`,
        CLUSTER_LABEL_COLOR[c.key],
      );
      label.position.set(c.centroid.x, c.centroid.y + LABEL_Y_OFFSET, c.centroid.z);
      rg.add(label);
    });

    // path / ring / focus groups
    const pathGroup  = new THREE.Group(); rg.add(pathGroup);  pathGroupRef.current  = pathGroup;
    const ringGroup  = new THREE.Group(); rg.add(ringGroup);  ringGroupRef.current  = ringGroup;
    const focusGroup = new THREE.Group(); rg.add(focusGroup); focusGroupRef.current = focusGroup;

    // hover ring
    const hoverRing = makeRingSprite("#7a90c0");
    hoverRing.visible = false;
    hoverRing.scale.set(0.22, 0.22, 1);
    rg.add(hoverRing);
    hoverRingRef.current = hoverRing;

    scene.add(new THREE.AmbientLight(0xffffff, 1));

    // ── 螢幕空間點擊判定 ──────────────────────────────────────
    function worldToScreenPx(worldPos: THREE.Vector3): { x: number; y: number } | null {
      const cam = cameraRef.current;
      if (!cam) return null;
      const proj = worldPos.clone().project(cam);
      if (Math.abs(proj.z) > 1) return null;
      return { x: ((proj.x + 1) / 2) * width, y: ((1 - proj.y) / 2) * height };
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

      if (rotGroupRef.current && !isDragging.current) {
        rotGroupRef.current.rotation.y += 0.0008;
        rotGroupRef.current.rotation.x += 0.0002;
      }

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
                const node = layout?.nodes[idx];
                const layerLabel = node ? CLUSTER_LABEL[clusterKeyOf(node)] : "";
                tooltipRef.current.textContent = node
                  ? `${node.name}  ·  ${layerLabel}`
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
      }

      // 聚焦光暈呼吸
      if (focusGroupRef.current && focusGroupRef.current.children.length > 0) {
        const t = frameCount * 0.04;
        const pulse = 0.85 + 0.15 * Math.sin(t);
        focusGroupRef.current.children.forEach((child) => {
          if (child instanceof THREE.Sprite) {
            child.material.opacity = (child.userData.baseOpacity ?? 1) * pulse;
          }
        });
      }

      // 接力光點動畫
      pulsesRef.current.forEach((p) => {
        p.phase = (p.phase + 0.012) % 1;
        p.dot.position.lerpVectors(p.from, p.to, p.phase);
      });

      renderer.render(scene, camera);
    }
    animate();

    // ── 事件 ─────────────────────────────────────────────────
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
      if (dx < 4 && dy < 4) {
        const r   = getBoundedRect();
        const px  = e.clientX - r.left;
        const py  = e.clientY - r.top;
        const idx = findNearestNodeIndex(px, py);
        if (idx !== null && layout) {
          const clickedNode = layout.nodes[idx];
          if (activeCandidateSetRef.current.has(clickedNode.id)) {
            setFocusedCandidateEid((prev) =>
              prev === clickedNode.id ? prev : clickedNode.id,
            );
          }
          onNodeClick?.(clickedNode);
        } else {
          setFocusedCandidateEid(null);
        }
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
      if (tooltipRef.current)    tooltipRef.current.style.opacity = "0";
      if (hoverRingRef.current)  hoverRingRef.current.visible = false;
    }
    function onWheel(e: WheelEvent) {
      e.preventDefault();
      const cam = cameraRef.current;
      if (!cam) return;
      cam.position.z = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, cam.position.z + e.deltaY * 0.01));
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

  // ── Active candidate 標示環 ────────────────────────────────
  useEffect(() => {
    const ringGroup = ringGroupRef.current;
    if (!ringGroup || !layout) return;
    disposeGroupChildren(ringGroup); ringGroup.clear();

    activeCandidateEids.forEach((eid) => {
      if (eid === focusedCandidateEid) return;
      const idx = nodeIdxMap.get(eid);
      if (idx === undefined) return;
      const ring = makeRingSprite("#00e0b8", 4);
      ring.position.copy(nodeWorldPositions[idx]);
      const sz = (isRC(layout.nodes[idx]) ? SIZE.rc : SIZE.other) * SIZE.highlightMultiplier;
      ring.scale.set(sz * 1.5, sz * 1.5, 1);
      ringGroup.add(ring);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCandidateEids, focusedCandidateEid, nodeIdxMap, nodeWorldPositions, layout]);

  // ── 聚焦候選：光暈 + foundation 路徑 ──────────────────────
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
        const c = categoryColor(node);
        colorArr[i * 3] = c.r; colorArr[i * 3 + 1] = c.g; colorArr[i * 3 + 2] = c.b;
        sizeArr[i] = isRC(node) ? SIZE.rc : SIZE.other;
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
    }

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
  }, [focusedCandidateEid, focusedTreeData, activeCandidateEids, nodeIdxMap, nodeWorldPositions, layout]);

  // ── Zoom buttons ───────────────────────────────────────────
  function zoomBy(delta: number) {
    const cam = cameraRef.current;
    if (!cam) return;
    cam.position.z = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, cam.position.z + delta));
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
          { fill: "#00e0b8", label: "候選命中",   hint: ""        },
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
          color:         focusedCandidateEid ? "#5fe6cc" : "#00e0b8",
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
            background:   focusedCandidateEid ? "#5fe6cc" : "#00e0b8",
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