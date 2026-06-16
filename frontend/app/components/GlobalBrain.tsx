"use client";

/**
 * GlobalBrain.tsx — 全局知識圖譜 3D 密度雲視覺化
 *
 * 顏色層次：
 *   Layer 1 base    → 橘/紫粒子雲（恆定背景）
 *   Layer 2 personal→ 紫色 fMRI 熱點疊加（依 weight 決定 sprite scale）
 *   Layer 3 query   → 青綠光暈（每次查詢重置）
 *
 * 新增 props：
 *   personalBrainEids  → fMRI 疊加層資料
 *   onNodeClick        → 點擊節點時通知外部（供 Detail Panel 使用）
 */

import { useEffect, useRef, useMemo } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

// ─────────────────────────────────────────────────────────────
// Exported types
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

export type FoundationTree = {
  nodes: { element_id: string; label: string }[];
  edges: { source: string; target: string }[];
};

type Props = {
  layout:               GlobalLayout | null;
  activeCandidateEids:  string[];
  activePathNodeEids:   string[];
  activePathEdges:      { source: string; target: string }[];
  activeFoundationTree: FoundationTree;
  // ── 新增 ──
  personalBrainEids:    { eid: string; weight: number }[];
  onNodeClick?:         (node: GlobalLayoutNode) => void;
  // ──────────
  width?:               number;
  height?:              number;
};

// ─────────────────────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────────────────────

const C = {
  bg:     0x000000,
  bg_css: "#000000",

  cloud_results_rc:    new THREE.Color(0xff6010),
  cloud_results_other: new THREE.Color(0xb03800),
  cloud_chapter_rc:    new THREE.Color(0xc060ff),
  cloud_chapter_other: new THREE.Color(0x7020b0),

  active_candidate:  new THREE.Color(0x00d4aa),
  active_path:       new THREE.Color(0x4f8ef7),
  active_foundation: new THREE.Color(0x00d4aa),
  // Layer 2: personal brain fMRI colour
  personal_brain:    new THREE.Color(0x7f77dd),

  bloom_strength:  2.4,
  bloom_radius:    0.5,
  bloom_threshold: 0.20,
};

const CLOUD = {
  particles_per_rc:    600,
  particles_per_other: 200,
  spread_rc:           0.06,
  spread_other:        0.10,
  size_rc:             0.018,
  size_other:          0.010,
};

const BG_PARTICLE_COUNT = 8000;
const BG_SPREAD         = 3.5;
const WORLD_SCALE       = 2.2;
const PROXY_THRESHOLD   = 0.12;

// fMRI sprite config
const FMRI_BASE_SCALE  = 0.45;   // scale at weight=1.0
const FMRI_BASE_OPACITY = 0.55;

// ─────────────────────────────────────────────────────────────
// Helpers (unchanged from original)
// ─────────────────────────────────────────────────────────────

function nodeToWorld(n: GlobalLayoutNode): THREE.Vector3 {
  return new THREE.Vector3(
    n.x * WORLD_SCALE,
    n.y * WORLD_SCALE,
    n.z * WORLD_SCALE,
  );
}

function isRC(node: GlobalLayoutNode): boolean {
  return (node.type || "").toLowerCase().includes("resulting");
}

function getNodeColors(node: GlobalLayoutNode): { rc: THREE.Color; other: THREE.Color } {
  const src = (node.source || "results").toLowerCase();
  if (src === "chapter") return { rc: C.cloud_chapter_rc, other: C.cloud_chapter_other };
  return { rc: C.cloud_results_rc, other: C.cloud_results_other };
}

function seededRand(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function buildDensityCloud(nodes: GlobalLayoutNode[]): {
  positions: Float32Array;
  colors:    Float32Array;
  sizes:     Float32Array;
  isRCArr:   Float32Array;
  nodeIndex: Int32Array;
} {
  const totalParticles = nodes.reduce(
    (acc, n) => acc + (isRC(n) ? CLOUD.particles_per_rc : CLOUD.particles_per_other),
    0,
  );
  const positions = new Float32Array(totalParticles * 3);
  const colors    = new Float32Array(totalParticles * 3);
  const sizes     = new Float32Array(totalParticles);
  const isRCArr   = new Float32Array(totalParticles);
  const nodeIndex = new Int32Array(totalParticles);

  let ptr = 0;
  nodes.forEach((node, ni) => {
    const rc        = isRC(node);
    const count     = rc ? CLOUD.particles_per_rc    : CLOUD.particles_per_other;
    const spread    = rc ? CLOUD.spread_rc           : CLOUD.spread_other;
    const baseSize  = rc ? CLOUD.size_rc             : CLOUD.size_other;
    const palette   = getNodeColors(node);
    const baseColor = rc ? palette.rc : palette.other;
    const rand      = seededRand(ni * 7919 + 1);
    const center    = nodeToWorld(node);

    for (let i = 0; i < count; i++) {
      const theta = rand() * Math.PI * 2;
      const phi   = Math.acos(2 * rand() - 1);
      const r     = spread * Math.cbrt(rand());
      positions[ptr * 3 + 0] = center.x + r * Math.sin(phi) * Math.cos(theta);
      positions[ptr * 3 + 1] = center.y + r * Math.sin(phi) * Math.sin(theta);
      positions[ptr * 3 + 2] = center.z + r * Math.cos(phi);
      colors[ptr * 3 + 0] = baseColor.r;
      colors[ptr * 3 + 1] = baseColor.g;
      colors[ptr * 3 + 2] = baseColor.b;
      sizes[ptr]     = baseSize * (0.5 + rand() * 0.8);
      isRCArr[ptr]   = rc ? 1.0 : 0.0;
      nodeIndex[ptr] = ni;
      ptr++;
    }
  });
  return { positions, colors, sizes, isRCArr, nodeIndex };
}

function buildBgParticles(): Float32Array {
  const pos  = new Float32Array(BG_PARTICLE_COUNT * 3);
  const rand = seededRand(42);
  for (let i = 0; i < BG_PARTICLE_COUNT; i++) {
    pos[i * 3 + 0] = (rand() * 2 - 1) * BG_SPREAD;
    pos[i * 3 + 1] = (rand() * 2 - 1) * BG_SPREAD;
    pos[i * 3 + 2] = (rand() * 2 - 1) * BG_SPREAD;
  }
  return pos;
}

function buildLineSegments(
  edges: { source: string; target: string }[],
  nodeById: Map<string, GlobalLayoutNode>,
): Float32Array | null {
  const pts: number[] = [];
  for (const e of edges) {
    const src = nodeById.get(e.source);
    const tgt = nodeById.get(e.target);
    if (!src || !tgt) continue;
    const s = nodeToWorld(src);
    const t = nodeToWorld(tgt);
    pts.push(s.x, s.y, s.z, t.x, t.y, t.z);
  }
  if (pts.length === 0) return null;
  return new Float32Array(pts);
}

// Build a purple radial-gradient canvas texture for fMRI sprites
function buildFMRITexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const ctx  = canvas.getContext("2d")!;
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0,   "rgba(127, 119, 221, 0.95)");
  grad.addColorStop(0.35, "rgba(83, 74, 183, 0.55)");
  grad.addColorStop(0.7,  "rgba(38, 33, 92, 0.20)");
  grad.addColorStop(1,    "rgba(38, 33, 92, 0.0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(canvas);
}

// ─────────────────────────────────────────────────────────────
// GlobalBrain
// ─────────────────────────────────────────────────────────────

export default function GlobalBrain({
  layout,
  activeCandidateEids,
  activePathNodeEids,
  activePathEdges,
  activeFoundationTree,
  personalBrainEids,
  onNodeClick,
  width  = 480,
  height = 480,
}: Props) {
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const sceneRef      = useRef<THREE.Scene | null>(null);
  const rendererRef   = useRef<THREE.WebGLRenderer | null>(null);
  const composerRef   = useRef<EffectComposer | null>(null);
  const cameraRef     = useRef<THREE.PerspectiveCamera | null>(null);
  const cloudMeshRef  = useRef<THREE.Points | null>(null);
  const glowGroupRef  = useRef<THREE.Group | null>(null);
  // Layer 2: personal brain fMRI group
  const fmriGroupRef  = useRef<THREE.Group | null>(null);
  const animFrameRef  = useRef<number>(0);
  const isDragging    = useRef(false);
  const prevMouse     = useRef({ x: 0, y: 0 });
  const rotGroup      = useRef<THREE.Group | null>(null);

  const foundationLinesRef = useRef<THREE.LineSegments | null>(null);
  const pathLinesRef       = useRef<THREE.LineSegments | null>(null);

  const proxyMeshesRef = useRef<THREE.Mesh[]>([]);
  const raycasterRef   = useRef<THREE.Raycaster | null>(null);
  const mouseNDCRef    = useRef(new THREE.Vector2(-9999, -9999));

  // Cached fMRI texture (built once)
  const fmriTextureRef = useRef<THREE.CanvasTexture | null>(null);

  const nodeById = useMemo(() => {
    if (!layout) return new Map<string, GlobalLayoutNode>();
    return new Map(layout.nodes.map((n) => [n.id, n]));
  }, [layout]);

  const nodeIdxMap = useMemo(() => {
    if (!layout) return new Map<string, number>();
    return new Map(layout.nodes.map((n, i) => [n.id, i]));
  }, [layout]);

  // ── 初始化場景 ────────────────────────────────────────────
  useEffect(() => {
    if (!canvasRef.current || !layout || layout.nodes.length === 0) return;

    const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(C.bg, 1);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 100);
    camera.position.set(0, 0, 5.5);
    cameraRef.current = camera;

    const rg = new THREE.Group();
    scene.add(rg);
    rotGroup.current = rg;

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(width, height),
      C.bloom_strength, C.bloom_radius, C.bloom_threshold,
    );
    composer.addPass(bloomPass);
    composerRef.current = composer;

    // bg particles
    const bgPos = buildBgParticles();
    const bgGeo = new THREE.BufferGeometry();
    bgGeo.setAttribute("position", new THREE.BufferAttribute(bgPos, 3));
    const bgMat = new THREE.PointsMaterial({ color: 0x111111, size: 0.005, transparent: true, opacity: 0.25, depthWrite: false });
    scene.add(new THREE.Points(bgGeo, bgMat));

    // density cloud
    const { positions, colors, sizes, isRCArr, nodeIndex } = buildDensityCloud(layout.nodes);
    const cloudGeo = new THREE.BufferGeometry();
    cloudGeo.setAttribute("position",  new THREE.BufferAttribute(positions.slice(), 3));
    cloudGeo.setAttribute("color",     new THREE.BufferAttribute(colors.slice(),    3));
    cloudGeo.setAttribute("size",      new THREE.BufferAttribute(sizes.slice(),     1));
    cloudGeo.setAttribute("isRC",      new THREE.BufferAttribute(isRCArr.slice(),   1));
    cloudGeo.setAttribute("nodeIndex", new THREE.BufferAttribute(nodeIndex,         1));

    const cloudMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 } },
      vertexShader: /* glsl */`
        attribute float size;
        attribute vec3  color;
        attribute float isRC;
        varying vec3    vColor;
        varying float   vIsRC;
        void main() {
          vColor = color; vIsRC = isRC;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mvPosition.z);
          gl_Position  = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: /* glsl */`
        varying vec3  vColor;
        varying float vIsRC;
        void main() {
          vec2 uv = gl_PointCoord - vec2(0.5);
          float alpha;
          if (vIsRC > 0.5) {
            float d = length(uv);
            if (d > 0.5) discard;
            alpha = smoothstep(0.5, 0.1, d) * 0.85;
          } else {
            float d = abs(uv.x) + abs(uv.y);
            if (d > 0.48) discard;
            alpha = smoothstep(0.48, 0.12, d) * 0.70;
          }
          gl_FragColor = vec4(vColor, alpha);
        }
      `,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });

    const cloudMesh = new THREE.Points(cloudGeo, cloudMat);
    cloudMesh.userData.nodeIndex  = nodeIndex;
    cloudMesh.userData.baseColors = colors.slice();
    rg.add(cloudMesh);
    cloudMeshRef.current = cloudMesh;

    // Layer 3: query glow group
    const glowGroup = new THREE.Group();
    rg.add(glowGroup);
    glowGroupRef.current = glowGroup;

    // Layer 2: personal brain fMRI group
    const fmriGroup = new THREE.Group();
    rg.add(fmriGroup);
    fmriGroupRef.current = fmriGroup;

    // Build and cache fMRI texture
    fmriTextureRef.current = buildFMRITexture();

    // Foundation lines
    const ftGeo = new THREE.BufferGeometry();
    ftGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
    const ftMat = new THREE.LineBasicMaterial({ color: C.active_foundation, transparent: true, opacity: 0.0, depthWrite: false, blending: THREE.AdditiveBlending });
    const ftLines = new THREE.LineSegments(ftGeo, ftMat);
    rg.add(ftLines);
    foundationLinesRef.current = ftLines;

    // Path lines
    const plGeo = new THREE.BufferGeometry();
    plGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
    const plMat = new THREE.LineBasicMaterial({ color: C.active_path, transparent: true, opacity: 0.0, depthWrite: false, blending: THREE.AdditiveBlending });
    const plLines = new THREE.LineSegments(plGeo, plMat);
    rg.add(plLines);
    pathLinesRef.current = plLines;

    // Invisible proxy meshes for raycasting
    const proxyGeo = new THREE.SphereGeometry(PROXY_THRESHOLD, 6, 6);
    const proxyMat = new THREE.MeshBasicMaterial({ visible: false, transparent: true, opacity: 0 });
    const proxies: THREE.Mesh[] = [];
    layout.nodes.forEach((node, ni) => {
      const proxy = new THREE.Mesh(proxyGeo, proxyMat);
      proxy.position.copy(nodeToWorld(node));
      proxy.userData.nodeIndex = ni;
      rg.add(proxy);
      proxies.push(proxy);
    });
    proxyMeshesRef.current = proxies;
    raycasterRef.current = new THREE.Raycaster();
    scene.add(new THREE.AmbientLight(0x110800, 2));

    // Animation loop
    let frameCount = 0;
    function animate() {
      animFrameRef.current = requestAnimationFrame(animate);
      frameCount++;

      if (rotGroup.current && !isDragging.current) {
        rotGroup.current.rotation.y += 0.0008;
        rotGroup.current.rotation.x += 0.0002;
      }

      // Pulse query glow sprites
      if (frameCount % 2 === 0 && glowGroupRef.current) {
        const t     = frameCount * 0.02;
        const pulse = 0.85 + 0.15 * Math.sin(t);
        glowGroupRef.current.children.forEach((child) => {
          if (child instanceof THREE.Sprite)
            child.material.opacity = child.userData.baseOpacity * pulse;
        });
      }

      // Gently pulse fMRI sprites (slower, subtler)
      if (frameCount % 3 === 0 && fmriGroupRef.current) {
        const t     = frameCount * 0.008;
        const pulse = 0.88 + 0.12 * Math.sin(t);
        fmriGroupRef.current.children.forEach((child) => {
          if (child instanceof THREE.Sprite)
            child.material.opacity = child.userData.baseOpacity * pulse;
        });
      }

      // Tooltip raycasting (hover)
      if (!isDragging.current && layout && raycasterRef.current && cameraRef.current && proxyMeshesRef.current.length > 0 && tooltipRef.current) {
        raycasterRef.current.setFromCamera(mouseNDCRef.current, cameraRef.current);
        const hits = raycasterRef.current.intersectObjects(proxyMeshesRef.current, false);
        if (hits.length > 0) {
          const ni   = hits[0].object.userData.nodeIndex as number;
          const node = layout.nodes[ni];
          tooltipRef.current.textContent = node.name;
          tooltipRef.current.style.opacity = "1";
        } else {
          tooltipRef.current.style.opacity = "0";
        }
      }

      composerRef.current?.render();
    }
    animate();

    // Mouse / touch events
    const canvas = canvasRef.current;

    function updateNDC(clientX: number, clientY: number) {
      const r = canvasRef.current?.getBoundingClientRect();
      if (!r) return;
      mouseNDCRef.current.x =  ((clientX - r.left) / r.width)  * 2 - 1;
      mouseNDCRef.current.y = -((clientY - r.top)  / r.height) * 2 + 1;
    }

    // ── Click handler (new) — fires onNodeClick ────────────
    function onCanvasClick(e: MouseEvent) {
      if (!layout || !raycasterRef.current || !cameraRef.current) return;
      updateNDC(e.clientX, e.clientY);
      raycasterRef.current.setFromCamera(mouseNDCRef.current, cameraRef.current);
      const hits = raycasterRef.current.intersectObjects(proxyMeshesRef.current, false);
      if (hits.length > 0 && onNodeClick) {
        const ni   = hits[0].object.userData.nodeIndex as number;
        onNodeClick(layout.nodes[ni]);
      }
    }

    function onMouseDown(e: MouseEvent) {
      isDragging.current = true;
      prevMouse.current  = { x: e.clientX, y: e.clientY };
      if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
    }
    function onMouseMove(e: MouseEvent) {
      updateNDC(e.clientX, e.clientY);
      if (!isDragging.current || !rotGroup.current) return;
      const dx = e.clientX - prevMouse.current.x;
      const dy = e.clientY - prevMouse.current.y;
      rotGroup.current.rotation.y += dx * 0.008;
      rotGroup.current.rotation.x += dy * 0.008;
      prevMouse.current = { x: e.clientX, y: e.clientY };
    }
    // Distinguish drag vs click: only fire click if mouse barely moved
    let mouseDownPos = { x: 0, y: 0 };
    function onMouseDownCapture(e: MouseEvent) { mouseDownPos = { x: e.clientX, y: e.clientY }; }
    function onMouseUpCapture(e: MouseEvent) {
      const dx = Math.abs(e.clientX - mouseDownPos.x);
      const dy = Math.abs(e.clientY - mouseDownPos.y);
      if (dx < 4 && dy < 4) onCanvasClick(e);
      isDragging.current = false;
    }
    function onMouseLeave() {
      isDragging.current = false;
      mouseNDCRef.current.set(-9999, -9999);
      if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
    }
    function onTouchStart(e: TouchEvent) {
      isDragging.current = true;
      prevMouse.current  = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }
    function onTouchMove(e: TouchEvent) {
      if (!isDragging.current || !rotGroup.current) return;
      const dx = e.touches[0].clientX - prevMouse.current.x;
      const dy = e.touches[0].clientY - prevMouse.current.y;
      rotGroup.current.rotation.y += dx * 0.008;
      rotGroup.current.rotation.x += dy * 0.008;
      prevMouse.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }

    canvas.addEventListener("mousedown",  onMouseDown);
    canvas.addEventListener("mousedown",  onMouseDownCapture);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("mouseup",    onMouseUpCapture);
    window.addEventListener("mousemove",  onMouseMove);
    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchmove",  onTouchMove,  { passive: true });
    window.addEventListener("touchend",   () => { isDragging.current = false; });

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      canvas.removeEventListener("mousedown",  onMouseDown);
      canvas.removeEventListener("mousedown",  onMouseDownCapture);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      canvas.removeEventListener("mouseup",    onMouseUpCapture);
      window.removeEventListener("mousemove",  onMouseMove);
      canvas.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove",  onTouchMove);
      renderer.dispose();
      composer.dispose();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, width, height]);

  // ── Layer 2: personal brain fMRI sprites ─────────────────
  useEffect(() => {
    if (!fmriGroupRef.current || !layout) return;

    const group = fmriGroupRef.current;
    // Dispose old sprites
    group.children.slice().forEach((child) => {
      if (child instanceof THREE.Sprite) {
        child.material.dispose();
        group.remove(child);
      }
    });

    if (personalBrainEids.length === 0) return;

    const tex = fmriTextureRef.current;
    if (!tex) return;

    personalBrainEids.forEach(({ eid, weight }) => {
      const node = nodeById.get(eid);
      if (!node) return;

      const pos        = nodeToWorld(node);
      const clampW     = Math.max(0.1, Math.min(1.0, weight));
      const spriteScale = FMRI_BASE_SCALE * (0.5 + clampW * 0.5) * (isRC(node) ? 1.2 : 0.9);
      const opacity    = FMRI_BASE_OPACITY * (0.6 + clampW * 0.4);

      const mat = new THREE.SpriteMaterial({
        map:         tex,
        transparent: true,
        opacity,
        blending:    THREE.AdditiveBlending,
        depthWrite:  false,
      });
      const sprite = new THREE.Sprite(mat);
      sprite.position.copy(pos);
      sprite.scale.setScalar(spriteScale);
      sprite.userData.baseOpacity = opacity;
      group.add(sprite);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [personalBrainEids, layout, nodeById]);

  // ── Layer 3: query candidate glow ─────────────────────────
  useEffect(() => {
    if (!cloudMeshRef.current || !layout || !glowGroupRef.current) return;

    const geo        = cloudMeshRef.current.geometry;
    const colorAttr  = geo.attributes.color as THREE.BufferAttribute;
    const colorArr   = colorAttr.array as Float32Array;
    const baseColors = cloudMeshRef.current.userData.baseColors as Float32Array;
    const niArr      = cloudMeshRef.current.userData.nodeIndex as Int32Array;

    const activeIdxSet = new Set(
      activeCandidateEids.map((eid) => nodeIdxMap.get(eid)).filter((i): i is number => i !== undefined),
    );

    colorArr.set(baseColors);
    const ac = C.active_candidate;
    for (let i = 0; i < niArr.length; i++) {
      if (activeIdxSet.has(niArr[i])) {
        colorArr[i * 3 + 0] = ac.r * 1.8;
        colorArr[i * 3 + 1] = ac.g * 1.8;
        colorArr[i * 3 + 2] = ac.b * 1.8;
      }
    }
    colorAttr.needsUpdate = true;

    const glowGroup = glowGroupRef.current;
    glowGroup.clear();
    if (activeIdxSet.size === 0) return;

    const canvas2d = document.createElement("canvas");
    canvas2d.width = canvas2d.height = 64;
    const ctx  = canvas2d.getContext("2d")!;
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0,   "rgba(0, 212, 170, 0.9)");
    grad.addColorStop(0.4, "rgba(0, 212, 170, 0.3)");
    grad.addColorStop(1,   "rgba(0, 212, 170, 0.0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);
    const glowTex = new THREE.CanvasTexture(canvas2d);

    activeIdxSet.forEach((ni) => {
      const node      = layout.nodes[ni];
      const pos       = nodeToWorld(node);
      const spriteMat = new THREE.SpriteMaterial({ map: glowTex, transparent: true, opacity: 0.75, blending: THREE.AdditiveBlending, depthWrite: false });
      const sprite    = new THREE.Sprite(spriteMat);
      sprite.position.copy(pos);
      sprite.scale.setScalar(isRC(node) ? 0.55 : 0.35);
      sprite.userData.baseOpacity = 0.75;
      glowGroup.add(sprite);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCandidateEids, layout, nodeIdxMap]);

  // ── Foundation tree lines ─────────────────────────────────
  useEffect(() => {
    const lines = foundationLinesRef.current;
    if (!lines || !layout) return;
    const mat = lines.material as THREE.LineBasicMaterial;
    if (activeFoundationTree.edges.length === 0) {
      mat.opacity = 0;
      lines.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
      return;
    }
    const posArr = buildLineSegments(activeFoundationTree.edges, nodeById);
    if (!posArr) { mat.opacity = 0; return; }
    lines.geometry.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
    (lines.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    mat.color.copy(C.active_foundation);
    mat.opacity = 0.65;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeFoundationTree, layout, nodeById]);

  // ── Path lines ────────────────────────────────────────────
  useEffect(() => {
    const lines = pathLinesRef.current;
    if (!lines || !layout) return;
    const mat = lines.material as THREE.LineBasicMaterial;
    if (activePathEdges.length === 0) {
      mat.opacity = 0;
      lines.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
      return;
    }
    const posArr = buildLineSegments(activePathEdges, nodeById);
    if (!posArr) { mat.opacity = 0; return; }
    lines.geometry.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
    (lines.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    mat.color.copy(C.active_path);
    mat.opacity = 0.75;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePathEdges, layout, nodeById]);

  // ── Path node cloud highlight ─────────────────────────────
  useEffect(() => {
    if (!cloudMeshRef.current || !layout) return;
    const geo        = cloudMeshRef.current.geometry;
    const colorAttr  = geo.attributes.color as THREE.BufferAttribute;
    const colorArr   = colorAttr.array as Float32Array;
    const baseColors = cloudMeshRef.current.userData.baseColors as Float32Array;
    const niArr      = cloudMeshRef.current.userData.nodeIndex as Int32Array;

    colorArr.set(baseColors);

    const candidateIdxSet = new Set(activeCandidateEids.map((eid) => nodeIdxMap.get(eid)).filter((i): i is number => i !== undefined));
    const ac = C.active_candidate;
    for (let i = 0; i < niArr.length; i++) {
      if (candidateIdxSet.has(niArr[i])) {
        colorArr[i * 3 + 0] = ac.r * 1.8;
        colorArr[i * 3 + 1] = ac.g * 1.8;
        colorArr[i * 3 + 2] = ac.b * 1.8;
      }
    }

    const pathIdxSet = new Set(activePathNodeEids.map((eid) => nodeIdxMap.get(eid)).filter((i): i is number => i !== undefined));
    const ap = C.active_path;
    for (let i = 0; i < niArr.length; i++) {
      if (pathIdxSet.has(niArr[i])) {
        colorArr[i * 3 + 0] = ap.r * 1.8;
        colorArr[i * 3 + 1] = ap.g * 1.8;
        colorArr[i * 3 + 2] = ap.b * 1.8;
      }
    }

    const ftNodeSet = new Set(
      (activeFoundationTree.nodes ?? []).map((n) => nodeIdxMap.get(n.element_id)).filter((i): i is number => i !== undefined),
    );
    for (let i = 0; i < niArr.length; i++) {
      if (ftNodeSet.has(niArr[i]) && !candidateIdxSet.has(niArr[i])) {
        colorArr[i * 3 + 0] = ac.r * 1.2;
        colorArr[i * 3 + 1] = ac.g * 1.2;
        colorArr[i * 3 + 2] = ac.b * 1.2;
      }
    }

    colorAttr.needsUpdate = true;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePathNodeEids, activeFoundationTree, activeCandidateEids, layout, nodeIdxMap]);

  // ── Render ────────────────────────────────────────────────
  const hasActiveSelection = activePathNodeEids.length > 0 || activeFoundationTree.edges.length > 0;

  return (
    <div style={{ position: "relative", width, height, borderRadius: "20px", overflow: "hidden", background: C.bg_css, cursor: "crosshair", flexShrink: 0 }}>
      <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />

      <div ref={tooltipRef} style={{ position: "absolute", bottom: 40, left: "50%", transform: "translateX(-50%)", background: "rgba(0,0,0,0.88)", border: "1px solid rgba(255,96,16,0.4)", borderRadius: "6px", padding: "5px 12px", fontSize: "0.72rem", color: "#f0c090", fontFamily: "monospace", letterSpacing: "0.06em", pointerEvents: "none", opacity: 0, transition: "opacity 0.15s ease", whiteSpace: "nowrap", maxWidth: "80%", overflow: "hidden", textOverflow: "ellipsis", zIndex: 10 }} />

      {!layout && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "#3a1a00", fontSize: "0.8rem", letterSpacing: "0.12em", textTransform: "uppercase", fontFamily: "monospace", pointerEvents: "none" }}>
          知識場域載入中…
        </div>
      )}

      <div style={{ position: "absolute", top: 10, left: 12, fontSize: "0.6rem", color: "#3a1800", letterSpacing: "0.15em", textTransform: "uppercase", fontFamily: "monospace", pointerEvents: "none", userSelect: "none" }}>
        Global Knowledge Field
      </div>

      {/* Legend */}
      <div style={{ position: "absolute", top: 10, right: 12, display: "flex", flexDirection: "column", gap: "4px", pointerEvents: "none", userSelect: "none" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <svg width="9" height="9" viewBox="0 0 9 9"><circle cx="4.5" cy="4.5" r="4" fill="#ff6010" /></svg>
          <span style={{ fontSize: "0.58rem", color: "#ff6010", fontFamily: "monospace", letterSpacing: "0.1em" }}>RESEARCH·RC</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <svg width="9" height="9" viewBox="0 0 9 9"><polygon points="4.5,0.5 8.5,4.5 4.5,8.5 0.5,4.5" fill="#b03800" /></svg>
          <span style={{ fontSize: "0.58rem", color: "#b03800", fontFamily: "monospace", letterSpacing: "0.1em" }}>RESEARCH</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <svg width="9" height="9" viewBox="0 0 9 9"><circle cx="4.5" cy="4.5" r="4" fill="#c060ff" /></svg>
          <span style={{ fontSize: "0.58rem", color: "#c060ff", fontFamily: "monospace", letterSpacing: "0.1em" }}>TEXTBOOK·RC</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <svg width="9" height="9" viewBox="0 0 9 9"><polygon points="4.5,0.5 8.5,4.5 4.5,8.5 0.5,4.5" fill="#7020b0" /></svg>
          <span style={{ fontSize: "0.58rem", color: "#7020b0", fontFamily: "monospace", letterSpacing: "0.1em" }}>TEXTBOOK</span>
        </div>
        {personalBrainEids.length > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 2 }}>
            <svg width="9" height="9" viewBox="0 0 9 9"><circle cx="4.5" cy="4.5" r="4" fill="#7f77dd" opacity="0.8" /></svg>
            <span style={{ fontSize: "0.58rem", color: "#7f77dd", fontFamily: "monospace", letterSpacing: "0.1em" }}>PERSONAL</span>
          </div>
        )}
      </div>

      {activeCandidateEids.length > 0 && (
        <div style={{ position: "absolute", bottom: hasActiveSelection ? 32 : 10, left: 12, fontSize: "0.65rem", color: "#00d4aa", letterSpacing: "0.1em", fontFamily: "monospace", pointerEvents: "none", userSelect: "none", display: "flex", alignItems: "center", gap: "6px", transition: "bottom 0.2s ease" }}>
          <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "#00d4aa", animation: "brain-pulse 1.8s ease-in-out infinite" }} />
          {activeCandidateEids.length} region{activeCandidateEids.length > 1 ? "s" : ""} activated
        </div>
      )}

      {hasActiveSelection && (
        <div style={{ position: "absolute", bottom: 10, left: 12, fontSize: "0.65rem", color: "#4f8ef7", letterSpacing: "0.1em", fontFamily: "monospace", pointerEvents: "none", userSelect: "none", display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ display: "inline-block", width: 20, height: 1, background: "linear-gradient(90deg, #00d4aa, #4f8ef7)" }} />
          path active
        </div>
      )}

      {personalBrainEids.length > 0 && (
        <div style={{ position: "absolute", bottom: hasActiveSelection ? 52 : 10, right: 12, fontSize: "0.62rem", color: "#7f77dd", letterSpacing: "0.1em", fontFamily: "monospace", pointerEvents: "none", userSelect: "none", display: "flex", alignItems: "center", gap: "5px" }}>
          <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "#7f77dd", opacity: 0.8 }} />
          {personalBrainEids.length} personal
        </div>
      )}

      <style>{`
        @keyframes brain-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
      `}</style>
    </div>
  );
}
