"use client";

/**
 * GlobalBrainExplorer.tsx
 *
 * 職責：
 *   - 渲染全局知識圖譜 3D 密度雲
 *   - 使用者點選 concept → 通知外部（onNodeClick）
 *   - 高亮 activeCandidateEids（SocratesChatBot 命中的候選）
 *
 * 已移除（交由 PersonalBrain 處理）：
 *   - personalBrainEids fMRI layer
 *   - activePathNodeEids / activePathEdges / activeFoundationTree
 *   - RC path 確認邏輯
 */

import { useEffect, useRef, useMemo } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

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
  activeCandidateEids: string[];          // SocratesChatBot 命中的候選（橘/青高亮）
  onNodeClick?:        (node: GlobalLayoutNode) => void;
  width?:              number;
  height?:             number;
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

  active_candidate: new THREE.Color(0x00d4aa),

  bloom_strength:  2.0,
  bloom_radius:    0.4,
  bloom_threshold: 0.22,
};

const CLOUD = {
  particles_per_rc:    600,
  particles_per_other: 200,
  spread_rc:   0.06,
  spread_other: 0.10,
  size_rc:     0.018,
  size_other:  0.010,
};

const BG_PARTICLE_COUNT = 8000;
const BG_SPREAD         = 3.5;
const WORLD_SCALE       = 2.2;
const PROXY_THRESHOLD   = 0.12;

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function nodeToWorld(n: GlobalLayoutNode): THREE.Vector3 {
  return new THREE.Vector3(n.x * WORLD_SCALE, n.y * WORLD_SCALE, n.z * WORLD_SCALE);
}

function isRC(node: GlobalLayoutNode): boolean {
  return (node.type || "").toLowerCase().includes("resulting");
}

function getNodeColors(node: GlobalLayoutNode) {
  const src = (node.source || "results").toLowerCase();
  if (src === "chapter")
    return { rc: C.cloud_chapter_rc, other: C.cloud_chapter_other };
  return { rc: C.cloud_results_rc, other: C.cloud_results_other };
}

function seededRand(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function buildDensityCloud(nodes: GlobalLayoutNode[]) {
  const total = nodes.reduce(
    (acc, n) => acc + (isRC(n) ? CLOUD.particles_per_rc : CLOUD.particles_per_other), 0,
  );
  const positions = new Float32Array(total * 3);
  const colors    = new Float32Array(total * 3);
  const sizes     = new Float32Array(total);
  const isRCArr   = new Float32Array(total);
  const nodeIndex = new Int32Array(total);

  let ptr = 0;
  nodes.forEach((node, ni) => {
    const rc       = isRC(node);
    const count    = rc ? CLOUD.particles_per_rc    : CLOUD.particles_per_other;
    const spread   = rc ? CLOUD.spread_rc           : CLOUD.spread_other;
    const baseSize = rc ? CLOUD.size_rc             : CLOUD.size_other;
    const palette  = getNodeColors(node);
    const baseColor = rc ? palette.rc : palette.other;
    const rand     = seededRand(ni * 7919 + 1);
    const center   = nodeToWorld(node);

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

// ─────────────────────────────────────────────────────────────
// GlobalBrainExplorer
// ─────────────────────────────────────────────────────────────

export default function GlobalBrainExplorer({
  layout,
  activeCandidateEids,
  onNodeClick,
  width  = 480,
  height = 480,
}: Props) {
  const canvasRef  = useRef<SVGSVGElement | null>(null);
  const _canvasRef = useRef<HTMLCanvasElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  const sceneRef     = useRef<THREE.Scene | null>(null);
  const rendererRef  = useRef<THREE.WebGLRenderer | null>(null);
  const composerRef  = useRef<EffectComposer | null>(null);
  const cameraRef    = useRef<THREE.PerspectiveCamera | null>(null);
  const cloudMeshRef = useRef<THREE.Points | null>(null);
  const glowGroupRef = useRef<THREE.Group | null>(null);
  const animFrameRef = useRef<number>(0);
  const isDragging   = useRef(false);
  const prevMouse    = useRef({ x: 0, y: 0 });
  const rotGroup     = useRef<THREE.Group | null>(null);

  const proxyMeshesRef = useRef<THREE.Mesh[]>([]);
  const raycasterRef   = useRef<THREE.Raycaster | null>(null);
  const mouseNDCRef    = useRef(new THREE.Vector2(-9999, -9999));

  // canvas ref (HTMLCanvasElement)
  const realCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const nodeById = useMemo(() => {
    if (!layout) return new Map<string, GlobalLayoutNode>();
    return new Map(layout.nodes.map((n) => [n.id, n]));
  }, [layout]);

  const nodeIdxMap = useMemo(() => {
    if (!layout) return new Map<string, number>();
    return new Map(layout.nodes.map((n, i) => [n.id, i]));
  }, [layout]);

  // ── Init scene ────────────────────────────────────────────
  useEffect(() => {
    if (!realCanvasRef.current || !layout || layout.nodes.length === 0) return;

    const renderer = new THREE.WebGLRenderer({
      canvas:    realCanvasRef.current,
      antialias: true,
      alpha:     false,
    });
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
    composer.addPass(new UnrealBloomPass(
      new THREE.Vector2(width, height),
      C.bloom_strength, C.bloom_radius, C.bloom_threshold,
    ));
    composerRef.current = composer;

    // bg particles
    const bgGeo = new THREE.BufferGeometry();
    bgGeo.setAttribute("position", new THREE.BufferAttribute(buildBgParticles(), 3));
    scene.add(new THREE.Points(bgGeo, new THREE.PointsMaterial({
      color: 0x111111, size: 0.005, transparent: true, opacity: 0.25, depthWrite: false,
    })));

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
        attribute float size; attribute vec3 color; attribute float isRC;
        varying vec3 vColor; varying float vIsRC;
        void main() {
          vColor = color; vIsRC = isRC;
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mvPos.z);
          gl_Position  = projectionMatrix * mvPos;
        }`,
      fragmentShader: /* glsl */`
        varying vec3 vColor; varying float vIsRC;
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
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });

    const cloudMesh = new THREE.Points(cloudGeo, cloudMat);
    cloudMesh.userData.nodeIndex  = nodeIndex;
    cloudMesh.userData.baseColors = colors.slice();
    rg.add(cloudMesh);
    cloudMeshRef.current = cloudMesh;

    // glow group (candidate highlights)
    const glowGroup = new THREE.Group();
    rg.add(glowGroup);
    glowGroupRef.current = glowGroup;

    // proxy meshes for raycasting
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
    raycasterRef.current   = new THREE.Raycaster();
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
      if (frameCount % 2 === 0 && glowGroupRef.current) {
        const t = frameCount * 0.02;
        glowGroupRef.current.children.forEach((child) => {
          if (child instanceof THREE.Sprite)
            child.material.opacity = child.userData.baseOpacity * (0.85 + 0.15 * Math.sin(t));
        });
      }
      // tooltip raycasting
      if (!isDragging.current && layout && raycasterRef.current && cameraRef.current
          && proxyMeshesRef.current.length > 0 && tooltipRef.current) {
        raycasterRef.current.setFromCamera(mouseNDCRef.current, cameraRef.current);
        const hits = raycasterRef.current.intersectObjects(proxyMeshesRef.current, false);
        if (hits.length > 0) {
          const ni   = hits[0].object.userData.nodeIndex as number;
          tooltipRef.current.textContent = layout.nodes[ni].name;
          tooltipRef.current.style.opacity = "1";
        } else {
          tooltipRef.current.style.opacity = "0";
        }
      }
      composerRef.current?.render();
    }
    animate();

    // Events
    const canvas = realCanvasRef.current;

    function updateNDC(clientX: number, clientY: number) {
      const r = realCanvasRef.current?.getBoundingClientRect();
      if (!r) return;
      mouseNDCRef.current.x =  ((clientX - r.left) / r.width)  * 2 - 1;
      mouseNDCRef.current.y = -((clientY - r.top)  / r.height) * 2 + 1;
    }

    let mouseDownPos = { x: 0, y: 0 };

    function onMouseDownCapture(e: MouseEvent) { mouseDownPos = { x: e.clientX, y: e.clientY }; isDragging.current = true; prevMouse.current = { x: e.clientX, y: e.clientY }; }
    function onMouseUpCapture(e: MouseEvent) {
      const dx = Math.abs(e.clientX - mouseDownPos.x);
      const dy = Math.abs(e.clientY - mouseDownPos.y);
      if (dx < 4 && dy < 4 && layout && raycasterRef.current && cameraRef.current) {
        updateNDC(e.clientX, e.clientY);
        raycasterRef.current.setFromCamera(mouseNDCRef.current, cameraRef.current);
        const hits = raycasterRef.current.intersectObjects(proxyMeshesRef.current, false);
        if (hits.length > 0 && onNodeClick) {
          onNodeClick(layout.nodes[hits[0].object.userData.nodeIndex as number]);
        }
      }
      isDragging.current = false;
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
    function onMouseLeave() {
      isDragging.current = false;
      mouseNDCRef.current.set(-9999, -9999);
      if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
    }
    function onTouchStart(e: TouchEvent) { isDragging.current = true; prevMouse.current = { x: e.touches[0].clientX, y: e.touches[0].clientY }; }
    function onTouchMove(e: TouchEvent) {
      if (!isDragging.current || !rotGroup.current) return;
      rotGroup.current.rotation.y += (e.touches[0].clientX - prevMouse.current.x) * 0.008;
      rotGroup.current.rotation.x += (e.touches[0].clientY - prevMouse.current.y) * 0.008;
      prevMouse.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }

    canvas.addEventListener("mousedown",  onMouseDownCapture);
    canvas.addEventListener("mouseup",    onMouseUpCapture);
    canvas.addEventListener("mouseleave", onMouseLeave);
    window.addEventListener("mousemove",  onMouseMove);
    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchmove",  onTouchMove,  { passive: true });
    window.addEventListener("touchend",   () => { isDragging.current = false; });

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      canvas.removeEventListener("mousedown",  onMouseDownCapture);
      canvas.removeEventListener("mouseup",    onMouseUpCapture);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      window.removeEventListener("mousemove",  onMouseMove);
      canvas.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove",  onTouchMove);
      renderer.dispose();
      composer.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, width, height]);

  // ── Candidate highlight ───────────────────────────────────
  useEffect(() => {
    if (!cloudMeshRef.current || !layout) return;
    const geo       = cloudMeshRef.current.geometry;
    const colorAttr = geo.attributes.color as THREE.BufferAttribute;
    const colorArr  = colorAttr.array as Float32Array;
    const baseColors = cloudMeshRef.current.userData.baseColors as Float32Array;
    const niArr      = cloudMeshRef.current.userData.nodeIndex as Int32Array;

    colorArr.set(baseColors);
    const activeSet = new Set(
      activeCandidateEids.map((eid) => nodeIdxMap.get(eid)).filter((i): i is number => i !== undefined),
    );
    const ac = C.active_candidate;
    for (let i = 0; i < niArr.length; i++) {
      if (activeSet.has(niArr[i])) {
        colorArr[i * 3 + 0] = ac.r * 1.8;
        colorArr[i * 3 + 1] = ac.g * 1.8;
        colorArr[i * 3 + 2] = ac.b * 1.8;
      }
    }
    colorAttr.needsUpdate = true;

    // Glow sprites
    const glowGroup = glowGroupRef.current;
    if (!glowGroup) return;
    glowGroup.clear();
    if (activeSet.size === 0) return;

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

    activeSet.forEach((ni) => {
      const node = layout.nodes[ni];
      const mat  = new THREE.SpriteMaterial({ map: glowTex, transparent: true, opacity: 0.75, blending: THREE.AdditiveBlending, depthWrite: false });
      const spr  = new THREE.Sprite(mat);
      spr.position.copy(nodeToWorld(node));
      spr.scale.setScalar(isRC(node) ? 0.55 : 0.35);
      spr.userData.baseOpacity = 0.75;
      glowGroup.add(spr);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCandidateEids, layout, nodeIdxMap]);

  return (
    <div style={{ position: "relative", width, height, borderRadius: "20px", overflow: "hidden", background: C.bg_css, cursor: "crosshair", flexShrink: 0 }}>
      <canvas ref={realCanvasRef} style={{ display: "block", width: "100%", height: "100%" }} />

      <div ref={tooltipRef} style={{
        position: "absolute", bottom: 40, left: "50%", transform: "translateX(-50%)",
        background: "rgba(0,0,0,0.88)", border: "1px solid rgba(255,96,16,0.4)",
        borderRadius: "6px", padding: "5px 12px", fontSize: "0.72rem", color: "#f0c090",
        fontFamily: "monospace", letterSpacing: "0.06em", pointerEvents: "none",
        opacity: 0, transition: "opacity 0.15s ease", whiteSpace: "nowrap",
        maxWidth: "80%", overflow: "hidden", textOverflow: "ellipsis", zIndex: 10,
      }} />

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
        {[
          { shape: "circle", fill: "#ff6010", label: "RESEARCH·RC" },
          { shape: "diamond", fill: "#b03800", label: "RESEARCH" },
          { shape: "circle", fill: "#c060ff", label: "TEXTBOOK·RC" },
          { shape: "diamond", fill: "#7020b0", label: "TEXTBOOK" },
        ].map(({ shape, fill, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <svg width="9" height="9" viewBox="0 0 9 9">
              {shape === "circle"
                ? <circle cx="4.5" cy="4.5" r="4" fill={fill} />
                : <polygon points="4.5,0.5 8.5,4.5 4.5,8.5 0.5,4.5" fill={fill} />}
            </svg>
            <span style={{ fontSize: "0.58rem", color: fill, fontFamily: "monospace", letterSpacing: "0.1em" }}>{label}</span>
          </div>
        ))}
      </div>

      {activeCandidateEids.length > 0 && (
        <div style={{ position: "absolute", bottom: 10, left: 12, fontSize: "0.65rem", color: "#00d4aa", letterSpacing: "0.1em", fontFamily: "monospace", pointerEvents: "none", userSelect: "none", display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "#00d4aa", animation: "brain-pulse 1.8s ease-in-out infinite" }} />
          {activeCandidateEids.length} region{activeCandidateEids.length > 1 ? "s" : ""} activated
        </div>
      )}

      <style>{`@keyframes brain-pulse { 0%,100%{opacity:1}50%{opacity:0.3} }`}</style>
    </div>
  );
}
