"use client";

/**
 * GlobalBrain.tsx — 全局知識圖譜 3D 密度雲視覺化
 *
 * 變更：
 *   - 密度雲改為亮橘色系（Resulting_Concept: #e87030，其他: #7a3010）
 *   - Bloom 參數調整（threshold 降至 0.25，strength 升至 2.2）
 *   - 新增 hover tooltip：用不可見代理 Mesh 做 raycasting，命中顯示 node.name
 *     threshold = 0.12，拖曳旋轉時隱藏
 */

import { useEffect, useRef, useMemo } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type GlobalLayoutNode = {
  id:   string;
  name: string;
  type: string;
  x:    number;
  y:    number;
  z:    number;
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
  activePathEids:      string[];
  width?:              number;
  height?:             number;
};

// ─────────────────────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────────────────────

const C = {
  bg:               0x020810,
  bg_css:           "#020810",
  // ── 密度雲基礎色（亮橘色系）──
  cloud_rc:         new THREE.Color(0xe87030),   // Resulting_Concept：亮橘
  cloud_other:      new THREE.Color(0x7a3010),   // 其他：深橘
  // ── 激活色 ──
  active_candidate: new THREE.Color(0x00d4aa),
  active_path:      [
    new THREE.Color(0x4f8ef7),
    new THREE.Color(0x00d4aa),
    new THREE.Color(0xe8a030),
    new THREE.Color(0xe85d60),
    new THREE.Color(0xa060f0),
  ],
  // ── bloom ──
  bloom_strength:   2.2,
  bloom_radius:     0.5,
  bloom_threshold:  0.25,
};

// 密度雲參數
const CLOUD = {
  particles_per_rc:    60,
  particles_per_other: 20,
  spread_rc:           0.06,
  spread_other:        0.10,
  size_rc:             0.018,
  size_other:          0.010,
};

// 背景粒子場
const BG_PARTICLE_COUNT = 12000;
const BG_SPREAD         = 3.5;

// 路徑粒子流
const PATH_PARTICLE_COUNT = 180;
const PATH_PARTICLE_SPEED = 0.0018;

// 座標縮放
const WORLD_SCALE = 2.2;

// Raycasting
const PROXY_THRESHOLD = 0.12;

// ─────────────────────────────────────────────────────────────
// Helpers
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

function seededRand(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

// ─────────────────────────────────────────────────────────────
// Build density cloud geometry
// ─────────────────────────────────────────────────────────────

function buildDensityCloud(nodes: GlobalLayoutNode[]): {
  positions: Float32Array;
  colors:    Float32Array;
  sizes:     Float32Array;
  nodeIndex: Int32Array;
} {
  const totalParticles = nodes.reduce(
    (acc, n) => acc + (isRC(n) ? CLOUD.particles_per_rc : CLOUD.particles_per_other),
    0,
  );

  const positions = new Float32Array(totalParticles * 3);
  const colors    = new Float32Array(totalParticles * 3);
  const sizes     = new Float32Array(totalParticles);
  const nodeIndex = new Int32Array(totalParticles);

  let ptr = 0;
  nodes.forEach((node, ni) => {
    const rc        = isRC(node);
    const count     = rc ? CLOUD.particles_per_rc    : CLOUD.particles_per_other;
    const spread    = rc ? CLOUD.spread_rc           : CLOUD.spread_other;
    const baseSize  = rc ? CLOUD.size_rc             : CLOUD.size_other;
    const baseColor = rc ? C.cloud_rc                : C.cloud_other;
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
      nodeIndex[ptr] = ni;
      ptr++;
    }
  });

  return { positions, colors, sizes, nodeIndex };
}

// ─────────────────────────────────────────────────────────────
// Build background particle geometry
// ─────────────────────────────────────────────────────────────

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
// GlobalBrain component
// ─────────────────────────────────────────────────────────────

export default function GlobalBrain({
  layout,
  activeCandidateEids,
  activePathEids,
  width  = 480,
  height = 480,
}: Props) {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const tooltipRef   = useRef<HTMLDivElement>(null);

  // Three.js refs
  const sceneRef        = useRef<THREE.Scene | null>(null);
  const rendererRef     = useRef<THREE.WebGLRenderer | null>(null);
  const composerRef     = useRef<EffectComposer | null>(null);
  const cameraRef       = useRef<THREE.PerspectiveCamera | null>(null);
  const cloudMeshRef    = useRef<THREE.Points | null>(null);
  const glowGroupRef    = useRef<THREE.Group | null>(null);
  const pathLineRef     = useRef<THREE.LineSegments | null>(null);
  const pathParticles   = useRef<THREE.Points | null>(null);
  const pathProgress    = useRef<Float32Array | null>(null);
  const pathCurve       = useRef<THREE.CatmullRomCurve3 | null>(null);
  const animFrameRef    = useRef<number>(0);
  const isDragging      = useRef(false);
  const prevMouse       = useRef({ x: 0, y: 0 });
  const rotGroup        = useRef<THREE.Group | null>(null);

  // Raycasting refs
  const proxyMeshesRef  = useRef<THREE.Mesh[]>([]);
  const raycasterRef    = useRef<THREE.Raycaster | null>(null);
  const mouseNDCRef     = useRef(new THREE.Vector2(-9999, -9999));

  // node id → index map
  const nodeIdxMap = useMemo(() => {
    if (!layout) return new Map<string, number>();
    return new Map(layout.nodes.map((n, i) => [n.id, i]));
  }, [layout]);

  // ── 初始化 Three.js 場景 ──────────────────────────────────
  useEffect(() => {
    if (!canvasRef.current || !layout || layout.nodes.length === 0) return;

    // ── renderer ──
    const renderer = new THREE.WebGLRenderer({
      canvas:    canvasRef.current,
      antialias: true,
      alpha:     false,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(C.bg, 1);
    rendererRef.current = renderer;

    // ── scene & camera ──
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 100);
    camera.position.set(0, 0, 5.5);
    cameraRef.current = camera;

    // ── rotation group ──
    const rg = new THREE.Group();
    scene.add(rg);
    rotGroup.current = rg;

    // ── bloom composer ──
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(width, height),
      C.bloom_strength,
      C.bloom_radius,
      C.bloom_threshold,
    );
    composer.addPass(bloomPass);
    composerRef.current = composer;

    // ── Layer 0：背景粒子場 ──
    const bgPos = buildBgParticles();
    const bgGeo = new THREE.BufferGeometry();
    bgGeo.setAttribute("position", new THREE.BufferAttribute(bgPos, 3));
    const bgMat = new THREE.PointsMaterial({
      color:       0x2a1a0a,   // 深橘棕背景粒子
      size:        0.006,
      transparent: true,
      opacity:     0.3,
      depthWrite:  false,
    });
    scene.add(new THREE.Points(bgGeo, bgMat));

    // ── Layer 1：密度雲 ──
    const { positions, colors, sizes, nodeIndex } = buildDensityCloud(layout.nodes);

    const cloudGeo = new THREE.BufferGeometry();
    cloudGeo.setAttribute("position", new THREE.BufferAttribute(positions.slice(), 3));
    cloudGeo.setAttribute("color",    new THREE.BufferAttribute(colors.slice(),    3));
    cloudGeo.setAttribute("size",     new THREE.BufferAttribute(sizes.slice(),     1));
    cloudGeo.setAttribute("nodeIndex", new THREE.BufferAttribute(nodeIndex, 1));

    const cloudMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 } },
      vertexShader: /* glsl */`
        attribute float size;
        attribute vec3 color;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mvPosition.z);
          gl_Position  = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: /* glsl */`
        varying vec3 vColor;
        void main() {
          float d = length(gl_PointCoord - vec2(0.5));
          if (d > 0.5) discard;
          float alpha = smoothstep(0.5, 0.1, d);
          gl_FragColor = vec4(vColor, alpha * 0.85);
        }
      `,
      transparent: true,
      depthWrite:  false,
      blending:    THREE.AdditiveBlending,
    });

    const cloudMesh = new THREE.Points(cloudGeo, cloudMat);
    cloudMesh.userData.nodeIndex  = nodeIndex;
    cloudMesh.userData.baseColors = colors.slice();
    rg.add(cloudMesh);
    cloudMeshRef.current = cloudMesh;

    // ── Layer 2：激活光暈群組 ──
    const glowGroup = new THREE.Group();
    rg.add(glowGroup);
    glowGroupRef.current = glowGroup;

    // ── Layer 3：路徑線段 ──
    const pathLineGeo = new THREE.BufferGeometry();
    const pathLineMat = new THREE.LineBasicMaterial({
      color:       0x00d4aa,
      transparent: true,
      opacity:     0.0,
      depthWrite:  false,
      blending:    THREE.AdditiveBlending,
    });
    const pathLine = new THREE.LineSegments(pathLineGeo, pathLineMat);
    rg.add(pathLine);
    pathLineRef.current = pathLine;

    // ── Layer 4：路徑粒子流 ──
    const ppGeo = new THREE.BufferGeometry();
    const ppPos = new Float32Array(PATH_PARTICLE_COUNT * 3);
    ppGeo.setAttribute("position", new THREE.BufferAttribute(ppPos, 3));
    const ppMat = new THREE.PointsMaterial({
      color:       0x00d4aa,
      size:        0.025,
      transparent: true,
      opacity:     0.0,
      depthWrite:  false,
      blending:    THREE.AdditiveBlending,
    });
    const ppPoints = new THREE.Points(ppGeo, ppMat);
    rg.add(ppPoints);
    pathParticles.current = ppPoints;
    pathProgress.current  = new Float32Array(PATH_PARTICLE_COUNT).map(
      (_, i) => i / PATH_PARTICLE_COUNT,
    );

    // ── Layer 5：不可見代理 Mesh（供 Raycasting 使用）──
    const proxyGeo = new THREE.SphereGeometry(PROXY_THRESHOLD, 6, 6);
    const proxyMat = new THREE.MeshBasicMaterial({
      visible:     false,   // 不渲染，僅用於碰撞偵測
      transparent: true,
      opacity:     0,
    });
    const proxies: THREE.Mesh[] = [];
    layout.nodes.forEach((node, ni) => {
      const proxy = new THREE.Mesh(proxyGeo, proxyMat);
      const pos   = nodeToWorld(node);
      proxy.position.copy(pos);
      proxy.userData.nodeIndex = ni;
      rg.add(proxy);
      proxies.push(proxy);
    });
    proxyMeshesRef.current = proxies;

    // ── Raycaster ──
    const raycaster = new THREE.Raycaster();
    raycaster.params.Mesh = {};
    raycasterRef.current = raycaster;

    scene.add(new THREE.AmbientLight(0x331100, 2));

    // ── Animation loop ──
    let frameCount = 0;
    function animate() {
      animFrameRef.current = requestAnimationFrame(animate);
      frameCount++;

      // 緩慢自轉
      if (rotGroup.current && !isDragging.current) {
        rotGroup.current.rotation.y += 0.0008;
        rotGroup.current.rotation.x += 0.0002;
      }

      // 路徑粒子流
      if (
        pathCurve.current &&
        pathParticles.current &&
        pathProgress.current &&
        (pathParticles.current.material as THREE.PointsMaterial).opacity > 0
      ) {
        const ppPositions = (
          pathParticles.current.geometry.attributes.position as THREE.BufferAttribute
        ).array as Float32Array;
        for (let i = 0; i < PATH_PARTICLE_COUNT; i++) {
          pathProgress.current[i] = (pathProgress.current[i] + PATH_PARTICLE_SPEED) % 1.0;
          const pt = pathCurve.current.getPoint(pathProgress.current[i]);
          ppPositions[i * 3 + 0] = pt.x;
          ppPositions[i * 3 + 1] = pt.y;
          ppPositions[i * 3 + 2] = pt.z;
        }
        (pathParticles.current.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
      }

      // 光暈脈動
      if (frameCount % 2 === 0 && glowGroupRef.current) {
        const t     = frameCount * 0.02;
        const pulse = 0.85 + 0.15 * Math.sin(t);
        glowGroupRef.current.children.forEach((child) => {
          if (child instanceof THREE.Sprite) {
            child.material.opacity = child.userData.baseOpacity * pulse;
          }
        });
      }

      // Raycasting（拖曳中跳過）
      if (
        !isDragging.current &&
        layout &&
        raycasterRef.current &&
        cameraRef.current &&
        proxyMeshesRef.current.length > 0 &&
        tooltipRef.current
      ) {
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

    // ── 滑鼠事件 ──
    const canvas = canvasRef.current;
    const rect   = canvas.getBoundingClientRect();

    function updateNDC(clientX: number, clientY: number) {
      const r = canvasRef.current?.getBoundingClientRect();
      if (!r) return;
      mouseNDCRef.current.x =  ((clientX - r.left) / r.width)  * 2 - 1;
      mouseNDCRef.current.y = -((clientY - r.top)  / r.height) * 2 + 1;
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
    function onMouseUp() { isDragging.current = false; }
    function onMouseLeave() {
      isDragging.current = false;
      mouseNDCRef.current.set(-9999, -9999);
      if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
    }

    function onTouchStart(e: TouchEvent) {
      isDragging.current = true;
      prevMouse.current  = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      if (tooltipRef.current) tooltipRef.current.style.opacity = "0";
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
    canvas.addEventListener("mouseleave", onMouseLeave);
    window.addEventListener("mousemove",  onMouseMove);
    window.addEventListener("mouseup",    onMouseUp);
    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchmove",  onTouchMove,  { passive: true });
    window.addEventListener("touchend",   onMouseUp);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      canvas.removeEventListener("mousedown",  onMouseDown);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      window.removeEventListener("mousemove",  onMouseMove);
      window.removeEventListener("mouseup",    onMouseUp);
      canvas.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove",  onTouchMove);
      window.removeEventListener("touchend",   onMouseUp);
      renderer.dispose();
      composer.dispose();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, width, height]);

  // ── 激活 candidate 節點 ──────────────────────────────────
  useEffect(() => {
    if (!cloudMeshRef.current || !layout || !glowGroupRef.current) return;

    const geo        = cloudMeshRef.current.geometry;
    const colorAttr  = geo.attributes.color as THREE.BufferAttribute;
    const colorArr   = colorAttr.array as Float32Array;
    const baseColors = cloudMeshRef.current.userData.baseColors as Float32Array;
    const niArr      = cloudMeshRef.current.userData.nodeIndex as Int32Array;

    const activeIdxSet = new Set(
      activeCandidateEids
        .map((eid) => nodeIdxMap.get(eid))
        .filter((i): i is number => i !== undefined),
    );

    colorArr.set(baseColors);

    const activeColor = C.active_candidate;
    for (let i = 0; i < niArr.length; i++) {
      if (activeIdxSet.has(niArr[i])) {
        colorArr[i * 3 + 0] = activeColor.r * 1.6;
        colorArr[i * 3 + 1] = activeColor.g * 1.6;
        colorArr[i * 3 + 2] = activeColor.b * 1.6;
      }
    }
    colorAttr.needsUpdate = true;

    const glowGroup = glowGroupRef.current;
    glowGroup.clear();
    if (activeIdxSet.size === 0) return;

    const canvas2d = document.createElement("canvas");
    canvas2d.width  = 64;
    canvas2d.height = 64;
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
      const spriteMat = new THREE.SpriteMaterial({
        map:         glowTex,
        transparent: true,
        opacity:     0.7,
        blending:    THREE.AdditiveBlending,
        depthWrite:  false,
      });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.copy(pos);
      sprite.scale.setScalar(isRC(node) ? 0.55 : 0.35);
      sprite.userData.baseOpacity = 0.7;
      glowGroup.add(sprite);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCandidateEids, layout, nodeIdxMap]);

  // ── 激活 path 路徑 ───────────────────────────────────────
  useEffect(() => {
    if (!layout || !pathLineRef.current || !pathParticles.current) return;

    const pathMat = pathLineRef.current.material as THREE.LineBasicMaterial;
    const ppMat   = pathParticles.current.material as THREE.PointsMaterial;

    if (activePathEids.length < 2) {
      pathMat.opacity = 0;
      ppMat.opacity   = 0;
      pathCurve.current = null;
      return;
    }

    const pathPoints: THREE.Vector3[] = [];
    activePathEids.forEach((eid) => {
      const ni = nodeIdxMap.get(eid);
      if (ni !== undefined) pathPoints.push(nodeToWorld(layout.nodes[ni]));
    });
    if (pathPoints.length < 2) return;

    const pathColor = C.active_path[0];
    pathMat.color.copy(pathColor);
    ppMat.color.copy(pathColor);

    const linePositions: number[] = [];
    for (let i = 0; i < pathPoints.length - 1; i++) {
      linePositions.push(
        pathPoints[i].x, pathPoints[i].y, pathPoints[i].z,
        pathPoints[i + 1].x, pathPoints[i + 1].y, pathPoints[i + 1].z,
      );
    }
    const lineGeo = pathLineRef.current.geometry;
    lineGeo.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(linePositions), 3),
    );
    lineGeo.attributes.position.needsUpdate = true;
    pathMat.opacity = 0.65;

    pathCurve.current = new THREE.CatmullRomCurve3(pathPoints, false, "catmullrom", 0.5);
    if (pathProgress.current) {
      for (let i = 0; i < PATH_PARTICLE_COUNT; i++) {
        pathProgress.current[i] = i / PATH_PARTICLE_COUNT;
      }
    }
    ppMat.opacity = 0.85;
    ppMat.size    = 0.022;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePathEids, layout, nodeIdxMap]);

  // ─────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────

  return (
    <div
      style={{
        position:     "relative",
        width,
        height,
        borderRadius: "20px",
        overflow:     "hidden",
        background:   C.bg_css,
        cursor:       "grab",
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ display: "block", width: "100%", height: "100%" }}
      />

      {/* Node name tooltip */}
      <div
        ref={tooltipRef}
        style={{
          position:      "absolute",
          bottom:        40,
          left:          "50%",
          transform:     "translateX(-50%)",
          background:    "rgba(2, 8, 16, 0.85)",
          border:        "1px solid rgba(232, 112, 48, 0.4)",
          borderRadius:  "6px",
          padding:       "5px 12px",
          fontSize:      "0.72rem",
          color:         "#f0c090",
          fontFamily:    "monospace",
          letterSpacing: "0.06em",
          pointerEvents: "none",
          opacity:       0,
          transition:    "opacity 0.15s ease",
          whiteSpace:    "nowrap",
          maxWidth:      "80%",
          overflow:      "hidden",
          textOverflow:  "ellipsis",
          zIndex:        10,
        }}
      />

      {/* 空狀態遮罩 */}
      {!layout && (
        <div style={{
          position:      "absolute",
          inset:         0,
          display:       "flex",
          alignItems:    "center",
          justifyContent: "center",
          color:          "#3a2010",
          fontSize:       "0.8rem",
          letterSpacing:  "0.12em",
          textTransform:  "uppercase",
          fontFamily:     "monospace",
          pointerEvents:  "none",
        }}>
          知識場域載入中…
        </div>
      )}

      {/* 角落標籤 */}
      <div style={{
        position:      "absolute",
        top:           10,
        left:          12,
        fontSize:      "0.6rem",
        color:         "#4a2010",
        letterSpacing: "0.15em",
        textTransform: "uppercase",
        fontFamily:    "monospace",
        pointerEvents: "none",
        userSelect:    "none",
      }}>
        Global Knowledge Field
      </div>

      {/* 激活狀態指示 */}
      {activeCandidateEids.length > 0 && (
        <div style={{
          position:      "absolute",
          bottom:        10,
          left:          12,
          fontSize:      "0.65rem",
          color:         "#00d4aa",
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
            background:   "#00d4aa",
            boxShadow:    "0 0 6px #00d4aa",
            animation:    "brain-pulse 1.8s ease-in-out infinite",
          }} />
          {activeCandidateEids.length} region{activeCandidateEids.length > 1 ? "s" : ""} activated
        </div>
      )}

      {activePathEids.length > 0 && (
        <div style={{
          position:      "absolute",
          bottom:        10,
          right:         12,
          fontSize:      "0.65rem",
          color:         "#4f8ef7",
          letterSpacing: "0.1em",
          fontFamily:    "monospace",
          pointerEvents: "none",
          userSelect:    "none",
        }}>
          ↝ path active
        </div>
      )}

      <style>{`
        @keyframes brain-pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
}
