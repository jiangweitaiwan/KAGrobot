"use client";

/**
 * HourglassViz.tsx — 沙漏式垂直剖面圖（取代 RCLineageViz）。
 *
 * 視覺結構：
 *   waist（腰身）  = 目前聚焦的 RC eid，固定在原點 (0,0,0)
 *   上半錐（開口向上，發散）= downstream 鏈，可能有多條 path，
 *                              越靠近頂端 rim 代表鏈上越深的節點
 *   下半錐（開口向下，發散）= foundation_tree，rim（最深的一層）
 *                              對應 BERTopic 2D 平面
 *
 * 座標策略（與 PersonalBrain 討論定案）：
 *   - 半徑（radius）：由「深度佔該半錐最大深度的比例」決定，
 *     保證最深一層一定落在 rim 上，形成乾淨的錐形輪廓，
 *     不受樹狀結構深淺不一影響。
 *   - 角度（angle）：優先取該節點在 BERTopic 2D layout 裡、
 *     相對於全域重心的方位角，讓沙漏裡的節點方向大致呼應
 *     它在語意地形裡的位置；查無座標（bertopic_layout.json
 *     尚未收錄，未來會持續擴充）時，退化為以 eid 為 seed 的
 *     決定性亂數角度，避免同一節點每次 render 角度亂跳。
 *
 * Foundation 節點分兩種樣式（沿用 RCLineageViz 既有的視覺語言）：
 *   - foundation-selected：已加入 Personal Brain，實心亮紫。
 *   - foundation-other：RC 的其他 foundation，空心描邊。
 *   Hourglass 本身不處理「加入 Personal Brain」這個動作——
 *   這個互動已經在 DetailPanel 的 FoundationTreeViz 做得很完整，
 *   這裡刻意保持唯讀 + 導覽（hover 顯示資訊，downstream 節點可點擊
 *   重新聚焦），避免同一份「勾選加入」狀態在兩個元件各自維護。
 *
 * 效能與互動模式（drag 旋轉、wheel/pinch 縮放、按需渲染）
 * 直接沿用 GlobalBrainExplorer 已經驗證過的 pattern，
 * sprite 生成與 dispose helper 抽到 three-shared.ts 共用。
 */

import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { FoundationTreeData } from "./DetailPanel";
import {
  makeLabelSprite,
  makeGlowSprite,
  makeFlatDotSprite,
  makeHollowDotSprite,
  makeBadgeSprite,
  disposeObject3D,
  disposeGroupChildren,
  hashStr,
  mulberry32,
} from "./three-shared";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type ChainStep = { eid: string; label: string };

export type BertopicPos = { x: number; y: number };

type HKind =
  | "waist"
  | "foundation-selected"
  | "foundation-other"
  | "foundation-mid"
  | "downstream"
  | "downstream-final";

type HNode = {
  id: string;
  label: string;
  kind: HKind;
  depth: number;      // 負 = foundation（下半錐）、0 = waist、正 = downstream（上半錐）
  hasSkill: boolean;  // 是否有對應的 RC Skill（僅 waist / downstream 可能為 true）
  x: number; y: number; z: number;
};

type HEdge = { source: string; target: string; kind: "foundation" | "downstream" };

type Props = {
  focusedEid: string;
  focusedLabel: string;
  foundationTree: FoundationTreeData | null;
  downstreamChains: ChainStep[][];
  personalBrainEids: Set<string>;
  bertopicPosByEid: Map<string, BertopicPos>;
  bertopicCentroid: BertopicPos;
  skillEids: Set<string>;
  selectedSkillEid: string | null;        // ← 取代 onLaunchSkill
  onSelectSkill: (eid: string, label: string) => void;  // ← 只負責選取，不送出
  loading?: boolean;
  onFocusNode: (eid: string, label: string) => void;
  width: number;
  height: number;
};

// ─────────────────────────────────────────────────────────────
// Design tokens（延續 RCLineageViz / GlobalBrainExplorer 的色彩語言）
// ─────────────────────────────────────────────────────────────

const T = {
  waist:              "#00d4aa",
  waist_glow:         "#00d4aa",
  foundation_selected:"#c9aaff",
  foundation_other:   "#7f77dd",
  foundation_mid:     "#3a3a5a",
  downstream:         "#4f8ef7",
  downstream_final:   "#a8c8ff",
  edge_foundation:    "#534ab7",
  edge_downstream:    "#2f5a8a",
  cone_wire:          0x3a4a68,
  text_waist:         "#00ffcc",
  text_downstream:    "#a8c8ff",
  text_foundation:    "#c9aaff",
  text_dim:           "#6a7a9a",
  skill_badge:        "#ffcc4d",
  skill_badge_stroke: "#fff3c4",
};

// ─────────────────────────────────────────────────────────────
// 佈局常數
// ─────────────────────────────────────────────────────────────

const RIM_RADIUS   = 1.6;   // 兩個錐形 rim 的半徑
const HALF_HEIGHT  = 1.9;   // 每個錐形的高度（waist → rim）
const ZOOM_MIN     = 1.6;
const ZOOM_MAX      = 9;
const ZOOM_DEFAULT  = 4.4;
const PICK_PIXEL_RADIUS = 18;
const MAX_PIXEL_RATIO   = 1.5;
const RENDER_FRAME_MS   = 1000 / 30;

// ─────────────────────────────────────────────────────────────
// buildHourglassGraph — 純函式，方便日後單獨測試
// ─────────────────────────────────────────────────────────────

export function buildHourglassGraph(
  focusedEid: string,
  focusedLabel: string,
  foundationTree: FoundationTreeData | null,
  downstreamChains: ChainStep[][],
  personalBrainEids: Set<string>,
  bertopicPosByEid: Map<string, BertopicPos>,
  bertopicCentroid: BertopicPos,
  skillEids: Set<string> = new Set(),
): { nodes: HNode[]; edges: HEdge[] } {
  type Draft = { id: string; label: string; kind: HKind; depth: number };
  const draftMap = new Map<string, Draft>();
  const edges: HEdge[] = [];

  draftMap.set(focusedEid, { id: focusedEid, label: focusedLabel, kind: "waist", depth: 0 });

  // ── foundation：從 waist 往上 BFS 指派深度（負數）──────────
  if (foundationTree) {
    const leafEids = new Set(foundationTree.foundations.map((f) => f.element_id));
    const childrenOf = new Map<string, string[]>(); // target(下游) → sources(上游)
    for (const e of foundationTree.edges) {
      if (!childrenOf.has(e.target)) childrenOf.set(e.target, []);
      childrenOf.get(e.target)!.push(e.source);
      edges.push({ source: e.source, target: e.target, kind: "foundation" });
    }

    const depthOf = new Map<string, number>([[focusedEid, 0]]);
    const queue: string[] = [focusedEid];
    while (queue.length) {
      const cur = queue.shift()!;
      const d = depthOf.get(cur)!;
      for (const parent of childrenOf.get(cur) ?? []) {
        if (!depthOf.has(parent)) {
          depthOf.set(parent, d - 1);
          queue.push(parent);
        }
      }
    }

    for (const n of foundationTree.nodes) {
      if (n.element_id === focusedEid) continue;
      const isLeaf = leafEids.has(n.element_id);
      const kind: HKind = !isLeaf
        ? "foundation-mid"
        : personalBrainEids.has(n.element_id)
          ? "foundation-selected"
          : "foundation-other";
      draftMap.set(n.element_id, {
        id: n.element_id,
        label: n.label,
        kind,
        depth: depthOf.get(n.element_id) ?? -1,
      });
    }
  }

  // ── downstream：每條鏈第 i 個節點 depth = i（正數）─────────
  for (const chain of downstreamChains) {
    const finalId = chain[chain.length - 1]?.eid;
    for (let i = 1; i < chain.length; i++) {
      const step = chain[i];
      const isFinal = step.eid === finalId;
      if (!draftMap.has(step.eid)) {
        draftMap.set(step.eid, {
          id: step.eid,
          label: step.label,
          kind: isFinal ? "downstream-final" : "downstream",
          depth: i,
        });
      } else if (isFinal) {
        // 若此節點在別條鏈是中繼、但在這條鏈是終點，升級為 final 樣式
        const existing = draftMap.get(step.eid)!;
        if (existing.kind === "downstream") existing.kind = "downstream-final";
      }
      edges.push({ source: chain[i - 1].eid, target: step.eid, kind: "downstream" });
    }
  }

  // ── 各半錐最大深度（決定 radius 正規化基準）─────────────────
  let maxFoundationDepth = 0;
  let maxDownstreamDepth = 0;
  draftMap.forEach((d) => {
    if (d.depth < 0) maxFoundationDepth = Math.max(maxFoundationDepth, -d.depth);
    if (d.depth > 0) maxDownstreamDepth = Math.max(maxDownstreamDepth, d.depth);
  });

  // ── 角度：優先 BERTopic 方位角，查無座標則決定性 fallback ───
  function resolveAngle(eid: string): number {
    const pos = bertopicPosByEid.get(eid);
    if (pos) {
      return Math.atan2(pos.y - bertopicCentroid.y, pos.x - bertopicCentroid.x);
    }
    const rng = mulberry32(hashStr(eid));
    return rng() * Math.PI * 2;
  }

  const nodes: HNode[] = [];
  draftMap.forEach((d) => {
    const hasSkill = skillEids.has(d.id);
    if (d.depth === 0) {
      nodes.push({ ...d, hasSkill, x: 0, y: 0, z: 0 });
      return;
    }
    const maxAbs = d.depth < 0 ? maxFoundationDepth : maxDownstreamDepth;
    const ratio = maxAbs > 0 ? Math.abs(d.depth) / maxAbs : 0;
    const radius = RIM_RADIUS * ratio;
    const y = Math.sign(d.depth) * HALF_HEIGHT * ratio;
    const angle = resolveAngle(d.id);
    nodes.push({
      ...d,
      hasSkill,
      x: radius * Math.cos(angle),
      y,
      z: radius * Math.sin(angle),
    });
  });

  return { nodes, edges };
}

// ─────────────────────────────────────────────────────────────
// 錐形線框（示意用，不擋互動）
// ─────────────────────────────────────────────────────────────

function buildConeWireframe(apexY: number, rimY: number, radius: number): THREE.Group {
  const group = new THREE.Group();
  const material = new THREE.LineDashedMaterial({
    color: T.cone_wire, dashSize: 0.06, gapSize: 0.05,
    transparent: true, opacity: 0.55,
  });

  // rim 圓（近似封閉多邊形）
  const rimSegments = 48;
  const rimPts: THREE.Vector3[] = [];
  for (let i = 0; i <= rimSegments; i++) {
    const a = (i / rimSegments) * Math.PI * 2;
    rimPts.push(new THREE.Vector3(radius * Math.cos(a), rimY, radius * Math.sin(a)));
  }
  const rimGeo = new THREE.BufferGeometry().setFromPoints(rimPts);
  const rimLine = new THREE.Line(rimGeo, material);
  rimLine.computeLineDistances();
  group.add(rimLine);

  // 子午線（apex → rim）
  const meridians = 8;
  for (let i = 0; i < meridians; i++) {
    const a = (i / meridians) * Math.PI * 2;
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, apexY, 0),
      new THREE.Vector3(radius * Math.cos(a), rimY, radius * Math.sin(a)),
    ]);
    const line = new THREE.Line(geo, material);
    line.computeLineDistances();
    group.add(line);
  }

  return group;
}

// ─────────────────────────────────────────────────────────────
// 節點外觀（sprite 選擇）
// ─────────────────────────────────────────────────────────────

function spriteForKind(kind: HKind): THREE.Sprite {
  switch (kind) {
    case "waist":
      return makeFlatDotSprite(T.waist, "#00ffcc");
    case "foundation-selected":
      return makeFlatDotSprite(T.foundation_selected, "#e8ddff");
    case "foundation-other":
      return makeHollowDotSprite(T.foundation_other, 3);
    case "foundation-mid":
      return makeFlatDotSprite(T.foundation_mid, "#5a5a8a");
    case "downstream-final":
      return makeFlatDotSprite(T.downstream_final, "#dbe8ff");
    case "downstream":
    default:
      return makeFlatDotSprite(T.downstream, "#a8c8ff");
  }
}

function alwaysShowLabel(kind: HKind): boolean {
  return kind === "waist" || kind === "downstream-final" || kind === "foundation-selected";
}

function labelColor(kind: HKind): string {
  switch (kind) {
    case "waist": return T.text_waist;
    case "downstream": case "downstream-final": return T.text_downstream;
    case "foundation-selected": return T.text_foundation;
    default: return T.text_dim;
  }
}

// ─────────────────────────────────────────────────────────────
// HourglassViz
// ─────────────────────────────────────────────────────────────

export default function HourglassViz({
  focusedEid,
  focusedLabel,
  foundationTree,
  downstreamChains,
  personalBrainEids,
  bertopicPosByEid,
  bertopicCentroid,
  skillEids,
  selectedSkillEid,
  onSelectSkill,
  loading = false,
  onFocusNode,
  width,
  height,
}: Props) {
  const canvasRef  = useRef<HTMLCanvasElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  const sceneRef     = useRef<THREE.Scene | null>(null);
  const rendererRef  = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef    = useRef<THREE.PerspectiveCamera | null>(null);
  const rotGroupRef  = useRef<THREE.Group | null>(null);
  const animFrameRef = useRef<number>(0);

  const isDragging     = useRef(false);
  const prevMouse       = useRef({ x: 0, y: 0 });
  const mousePixel      = useRef({ x: -9999, y: -9999 });
  const pinchStartDist  = useRef(0);
  const pinchStartZ     = useRef(ZOOM_DEFAULT);
  const isVisibleRef    = useRef(true);
  const needsRenderRef  = useRef(true);
  const lastFrameTsRef  = useRef(0);
  const skillGlowRef    = useRef<THREE.Sprite | null>(null); // ← 新增：選取中 skill 的呼吸光暈

  const graph = useMemo(
    () => buildHourglassGraph(
      focusedEid, focusedLabel, foundationTree, downstreamChains,
      personalBrainEids, bertopicPosByEid, bertopicCentroid, skillEids,
    ),
    [focusedEid, focusedLabel, foundationTree, downstreamChains, personalBrainEids, bertopicPosByEid, bertopicCentroid, skillEids],
  );

  useEffect(() => {
    if (!canvasRef.current || !focusedEid) return;

    const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
    renderer.setClearColor(0x000000, 0);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(55, width / height, 0.01, 100);
    camera.position.set(0, 0.3, ZOOM_DEFAULT);
    cameraRef.current = camera;

    const rg = new THREE.Group();
    scene.add(rg);
    rotGroupRef.current = rg;

    // ── 錐形線框（上半 downstream + 下半 foundation）──────────
    rg.add(buildConeWireframe(0, HALF_HEIGHT, RIM_RADIUS));
    rg.add(buildConeWireframe(0, -HALF_HEIGHT, RIM_RADIUS));

    // ── edges ──────────────────────────────────────────────
    const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
    const edgeLinePts: number[] = [];
    const edgeColors: number[] = [];
    const fColor = new THREE.Color(T.edge_foundation);
    const dColor = new THREE.Color(T.edge_downstream);
    graph.edges.forEach((e) => {
      const s = nodeById.get(e.source);
      const t = nodeById.get(e.target);
      if (!s || !t) return;
      edgeLinePts.push(s.x, s.y, s.z, t.x, t.y, t.z);
      const c = e.kind === "foundation" ? fColor : dColor;
      edgeColors.push(c.r, c.g, c.b, c.r, c.g, c.b);
    });
    if (edgeLinePts.length > 0) {
      const edgeGeo = new THREE.BufferGeometry();
      edgeGeo.setAttribute("position", new THREE.Float32BufferAttribute(edgeLinePts, 3));
      edgeGeo.setAttribute("color", new THREE.Float32BufferAttribute(edgeColors, 3));
      const edgeMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.65 });
      rg.add(new THREE.LineSegments(edgeGeo, edgeMat));
    }

    // ── nodes（sprite + 可選 label）───────────────────────────
    const nodeGroup = new THREE.Group();
    rg.add(nodeGroup);
    const spritesById = new Map<string, THREE.Sprite>();
    const badgePositions = new Map<string, { x: number; y: number; z: number }>();

    graph.nodes.forEach((n) => {
      const spr = spriteForKind(n.kind);
      const scale = n.kind === "waist" ? 0.11 : n.kind === "downstream-final" ? 0.08 : 0.06;
      spr.scale.set(scale, scale, 1);
      spr.position.set(n.x, n.y, n.z);
      spr.userData.eid = n.id;
      nodeGroup.add(spr);
      spritesById.set(n.id, spr);

      if (n.kind === "waist") {
        const glow = makeGlowSprite("#00d4aa");
        glow.position.set(n.x, n.y, n.z);
        glow.scale.set(0.5, 0.5, 1);
        nodeGroup.add(glow);
      }

      if (alwaysShowLabel(n.kind)) {
        const label = makeLabelSprite(
          n.label.length > 16 ? n.label.slice(0, 15) + "…" : n.label,
          labelColor(n.kind),
        );
        const lift = n.kind === "waist" ? 0.16 : 0.1;
        label.position.set(n.x, n.y + lift, n.z);
        label.scale.set(0.5, 0.11, 1);
        nodeGroup.add(label);
      }

      // ── skill 徽章：獨立於 kind 的小閃電圖示，標記「可選取分析」──
      if (n.hasSkill) {
        const isSelected = n.id === selectedSkillEid;
        const badge = makeBadgeSprite(
          isSelected ? "#ffffff" : T.skill_badge,
          isSelected ? T.skill_badge : T.skill_badge_stroke,
        );
        const bx = n.x + scale * 0.9;
        const by = n.y + scale * 0.9;
        const bz = n.z;
        badge.position.set(bx, by, bz);
        badge.userData.eid = n.id;
        nodeGroup.add(badge);
        badgePositions.set(n.id, { x: bx, y: by, z: bz });

        if (isSelected) {
          const glow = makeGlowSprite(T.skill_badge);
          glow.position.set(bx, by, bz);
          glow.scale.set(0.22, 0.22, 1);
          nodeGroup.add(glow);
          skillGlowRef.current = glow;
        }
      }
    });

    needsRenderRef.current = true;

    // ── 螢幕空間最近節點偵測（沿用 GlobalBrainExplorer 手法）──
    function worldToScreenPx(worldPos: THREE.Vector3): { x: number; y: number } | null {
      const cam = cameraRef.current;
      if (!cam) return null;
      const proj = worldPos.clone().project(cam);
      if (Math.abs(proj.z) > 1) return null;
      return { x: ((proj.x + 1) / 2) * width, y: ((1 - proj.y) / 2) * height };
    }

    function findNearestNode(px: number, py: number): HNode | null {
      if (!rotGroupRef.current) return null;
      let bestDist = PICK_PIXEL_RADIUS * PICK_PIXEL_RADIUS;
      let best: HNode | null = null;
      graph.nodes.forEach((n) => {
        const wp = new THREE.Vector3(n.x, n.y, n.z);
        rotGroupRef.current!.localToWorld(wp);
        const sp = worldToScreenPx(wp);
        if (!sp) return;
        const dx = sp.x - px, dy = sp.y - py;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDist) { bestDist = d2; best = n; }
      });
      return best;
    }


    const BADGE_PICK_PIXEL_RADIUS = 12;

    function findNearestSkillBadge(px: number, py: number): HNode | null {
      if (!rotGroupRef.current) return null;
      let bestDist = BADGE_PICK_PIXEL_RADIUS * BADGE_PICK_PIXEL_RADIUS;
      let best: HNode | null = null;
      graph.nodes.forEach((n) => {
        if (!n.hasSkill) return;
        const pos = badgePositions.get(n.id);
        if (!pos) return;
        const wp = new THREE.Vector3(pos.x, pos.y, pos.z);
        rotGroupRef.current!.localToWorld(wp);
        const sp = worldToScreenPx(wp);
        if (!sp) return;
        const dx = sp.x - px, dy = sp.y - py;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDist) { bestDist = d2; best = n; }
      });
      return best;
    }
    // ── render loop（按需渲染，沿用 GlobalBrainExplorer 節流策略）──
    function animate(now: number) {
      animFrameRef.current = requestAnimationFrame(animate);
      if (!isVisibleRef.current) return;
      if (now - lastFrameTsRef.current < RENDER_FRAME_MS) return;
      lastFrameTsRef.current = now;

      let hasBreathing = false;
      if (skillGlowRef.current) {
        hasBreathing = true;
        const pulse = 0.55 + 0.45 * Math.sin(now * 0.004);
        (skillGlowRef.current.material as THREE.SpriteMaterial).opacity = pulse;
      }

      if (needsRenderRef.current || isDragging.current || hasBreathing) {
        renderer.render(scene, camera);
        needsRenderRef.current = false;
      }
    }


    // ── 互動事件 ────────────────────────────────────────────
    const canvas = canvasRef.current!;
    let mouseDownPos = { x: 0, y: 0 };
    function getRect() { return canvas.getBoundingClientRect(); }

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
        const r  = getRect();
        const px = e.clientX - r.left;
        const py = e.clientY - r.top;

        const badgeHit = findNearestSkillBadge(px, py);
        if (badgeHit) {
          onSelectSkill(badgeHit.id, badgeHit.label);   // ← 改為選取
        } else {
          const hit = findNearestNode(px, py);
          if (hit && (hit.kind === "downstream" || hit.kind === "downstream-final")) {
            onFocusNode(hit.id, hit.label);
          }
        }
      }
      isDragging.current = false;
      needsRenderRef.current = true;
    }
    function onMouseMove(e: MouseEvent) {
      const r = getRect();
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
      if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
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
        const dist  = getTouchDist(e);
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

    // hover tooltip（低頻輪詢即可，沿用滑鼠位置 + rAF 節流已經夠用）
    function onMouseMoveForTooltip(e: MouseEvent) {
      if (isDragging.current || !tooltipRef.current) return;
      const r  = getRect();
      const px = e.clientX - r.left;
      const py = e.clientY - r.top;

      const badgeHit = findNearestSkillBadge(px, py);
      const hit = badgeHit ?? findNearestNode(px, py);
      if (hit) {
        const kindLabel =
          hit.kind === "waist" ? "◉ 目前聚焦" :
          hit.kind === "downstream-final" ? "▲ downstream 終點（點擊可聚焦）" :
          hit.kind === "downstream" ? "↑ downstream（點擊可聚焦）" :
          hit.kind === "foundation-selected" ? "◆ foundation · 已在 Personal Brain" :
          hit.kind === "foundation-other" ? "◇ foundation · 尚未加入" :
          "· 中繼概念";
        const skillLine = hit.hasSkill
          ? `<br/><span style="opacity:0.85;color:${T.skill_badge}">⚡ 點擊圖示以選取此 skill</span>`
          : "";
        tooltipRef.current.innerHTML = `<strong>${hit.label}</strong><br/><span style="opacity:0.65">${kindLabel}</span>${skillLine}`;
        tooltipRef.current.style.display = "block";
        tooltipRef.current.style.opacity = "1";
        tooltipRef.current.style.left = `${px + 12}px`;
        tooltipRef.current.style.top  = `${py - 8}px`;
      } else {
        tooltipRef.current.style.opacity = "0";
      }
    }

    canvas.addEventListener("mousedown",  onMouseDown);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("wheel",      onWheel, { passive: false });
    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    canvas.addEventListener("mousemove",  onMouseMoveForTooltip);
    window.addEventListener("mouseup",    onMouseUp);
    window.addEventListener("mousemove",  onMouseMove);
    window.addEventListener("touchmove",  onTouchMove, { passive: true });
    window.addEventListener("touchend",   onTouchEnd);

    function onVisibilityChange() {
      isVisibleRef.current = !document.hidden;
      if (isVisibleRef.current) needsRenderRef.current = true;
    }
    document.addEventListener("visibilitychange", onVisibilityChange);

    let io: IntersectionObserver | null = null;
    if (typeof IntersectionObserver !== "undefined") {
      io = new IntersectionObserver(([entry]) => {
        isVisibleRef.current = entry.isIntersecting && !document.hidden;
        if (isVisibleRef.current) needsRenderRef.current = true;
      }, { threshold: 0.05 });
      io.observe(canvas);
    }

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      canvas.removeEventListener("mousedown",  onMouseDown);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      canvas.removeEventListener("wheel",      onWheel);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("mousemove",  onMouseMoveForTooltip);
      window.removeEventListener("mouseup",    onMouseUp);
      window.removeEventListener("mousemove",  onMouseMove);
      window.removeEventListener("touchmove",  onTouchMove);
      window.removeEventListener("touchend",   onTouchEnd);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      io?.disconnect();

      disposeObject3D(scene);
      renderer.dispose();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph, width, height, focusedEid, onFocusNode, onSelectSkill, selectedSkillEid]);

  return (
    <div style={{ position: "relative", width, height }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block", width: "100%", height: "100%", cursor: "grab" }}
      />
      <div
        ref={tooltipRef}
        style={{
          display: "none", position: "absolute", pointerEvents: "none",
          background: "rgba(13,18,32,0.92)", border: "1px solid rgba(127,119,221,0.3)",
          borderRadius: 8, padding: "6px 10px", fontSize: 11, color: "#c8daf0",
          fontFamily: "IBM Plex Mono, monospace", maxWidth: 220, lineHeight: 1.5,
          opacity: 0, transition: "opacity 0.12s ease", zIndex: 10,
        }}
      />
      {loading && (
        <div style={{
          position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(3,1,8,0.4)", color: "rgba(127,119,221,0.6)", fontSize: 11,
          fontFamily: "monospace", letterSpacing: "0.1em", pointerEvents: "none",
        }}>
          載入剖面資料…
        </div>
      )}
      {!focusedEid && !loading && (
        <div style={{
          position: "absolute", inset: 0, display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center", gap: 6,
          color: "rgba(127,119,221,0.2)", fontSize: 11, letterSpacing: "0.14em",
          textTransform: "uppercase", fontFamily: "monospace", pointerEvents: "none",
        }}>
          <span>選擇上方推薦 RC</span>
          <span style={{ fontSize: 10 }}>檢視它的沙漏剖面</span>
        </div>
      )}
    </div>
  );
}
