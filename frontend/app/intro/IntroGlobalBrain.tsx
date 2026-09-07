"use client";

/**
 * IntroGlobalBrain.tsx — /intro 頁面的簡化版 3D 球體。
 *
 * 與正式的 GlobalBrainExplorer.tsx 刻意不共用元件本體：那顆元件耦合了
 * 大量這裡用不到的機制（hover pick、click 選中、walk illumination、
 * resize observer、多層 cache ref）。這裡只需要「依 mode 顯示不同的
 * 高亮型態＋緩慢自轉＋信念收斂動畫」，獨立寫一份精簡版更清楚。
 *
 * 三種 mode：
 *   - "feel"      感受：只有核心的 FEEL_NODES 亮青白色，其餘全部淡出
 *   - "knowledge" 知識：核心淡出，KNOWLEDGE_NODES 亮暖橘色
 *   - "belief"    信念：FOUNDATION_IDS 那幾個節點亮起，並用虛線向
 *                 BELIEF_NODE 收斂，收斂點呼吸發光
 *
 * sprite / dispose helper 沿用 three-shared.ts，維持與正式系統一致的
 * 視覺語言（同一套光暈、圓點畫法）。
 */

import { useEffect, useRef } from "react";
import * as THREE from "three";
import {
  makeGlowSprite,
  makeFlatDotSprite,
  makeLabelSprite,
  disposeObject3D,
  disposeGroupChildren,
} from "../../components/three-shared";
import {
  ALL_SPHERE_NODES,
  FOUNDATION_IDS,
  BELIEF_NODE,
  type DemoNode,
} from "../demoData";

export type IntroBrainMode = "feel" | "knowledge" | "belief";

type Props = {
  mode: IntroBrainMode;
  width: number;
  height: number;
};

const WORLD_SCALE = 2.4;
const DIM_COLOR = new THREE.Color(0x262b38);
const FEEL_COLOR = new THREE.Color(0x00e8ff);
const KNOWLEDGE_COLOR = new THREE.Color(0xff9a52);
const BELIEF_COLOR = new THREE.Color(0xff5fd1);

function toWorld(n: { x: number; y: number; z: number }): THREE.Vector3 {
  return new THREE.Vector3(n.x * WORLD_SCALE, n.y * WORLD_SCALE, n.z * WORLD_SCALE);
}

export default function IntroGlobalBrain({ mode, width, height }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rotGroupRef = useRef<THREE.Group | null>(null);
  const contentGroupRef = useRef<THREE.Group | null>(null);
  const breathingRef = useRef<THREE.Sprite[]>([]);
  const animFrameRef = useRef<number>(0);
  const widthRef = useRef(width);
  const heightRef = useRef(height);

  useEffect(() => {
    widthRef.current = width;
    heightRef.current = height;
  }, [width, height]);

  // ── Effect A: mount once — renderer/scene/camera/render loop ──
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

    // 背景微光星點，呼應正式系統的暗底質感
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

    scene.add(new THREE.AmbientLight(0xffffff, 1));

    const contentGroup = new THREE.Group();
    rotGroup.add(contentGroup);
    contentGroupRef.current = contentGroup;

    let lastT = 0;
    function animate(now: number) {
      animFrameRef.current = requestAnimationFrame(animate);
      if (now - lastT < 1000 / 30) return;
      lastT = now;

      // 唯一的持續動態：緩慢自轉，其餘一律靜止（動效克制原則）
      rotGroup.rotation.y += 0.0012;
      rotGroup.rotation.x = Math.sin(now * 0.00008) * 0.08;

      breathingRef.current.forEach((spr) => {
        const pulse = 0.55 + 0.45 * Math.sin(now * 0.0025);
        (spr.material as THREE.SpriteMaterial).opacity = pulse;
      });

      renderer.render(scene, camera);
    }
    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      disposeObject3D(scene);
      renderer.dispose();
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      rotGroupRef.current = null;
      contentGroupRef.current = null;
      breathingRef.current = [];
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

  // ── Effect C: mode 變化 — 重建節點/edge，不動 renderer ──
  useEffect(() => {
    const contentGroup = contentGroupRef.current;
    if (!contentGroup) return;

    disposeGroupChildren(contentGroup);
    contentGroup.clear();
    breathingRef.current = [];

    const foundationSet = new Set(FOUNDATION_IDS);
    const nodeById = new Map<string, DemoNode>(ALL_SPHERE_NODES.map((n) => [n.id, n]));

    ALL_SPHERE_NODES.forEach((n) => {
      const isFeel = n.kind === "feel";
      const isKnowledge = n.kind === "knowledge";

      let active = false;
      let color = DIM_COLOR;
      if (mode === "feel" && isFeel) { active = true; color = FEEL_COLOR; }
      if (mode === "knowledge" && isKnowledge) { active = true; color = KNOWLEDGE_COLOR; }
      if (mode === "belief" && foundationSet.has(n.id)) { active = true; color = BELIEF_COLOR; }

      const pos = toWorld(n);
      const dot = makeFlatDotSprite(`#${color.getHexString()}`, active ? "#ffffff" : "#4a5468");
      const scale = active ? 0.09 : 0.05;
      dot.scale.set(scale, scale, 1);
      dot.position.copy(pos);
      dot.material.opacity = active ? 1 : 0.35;
      contentGroup.add(dot);

      if (active) {
        const glow = makeGlowSprite(`#${color.getHexString()}`);
        glow.position.copy(pos);
        glow.scale.set(0.32, 0.32, 1);
        contentGroup.add(glow);

        const label = makeLabelSprite(n.label, `#${color.getHexString()}`);
        label.position.set(pos.x, pos.y + 0.18, pos.z);
        label.scale.set(0.5, 0.11, 1);
        contentGroup.add(label);
      }
    });

    if (mode === "belief") {
      // 收斂虛線：foundation → belief
      const beliefPos = new THREE.Vector3(0, 0, 1.9);
      const linePts: number[] = [];
      FOUNDATION_IDS.forEach((id) => {
        const n = nodeById.get(id);
        if (!n) return;
        const from = toWorld(n);
        linePts.push(from.x, from.y, from.z, beliefPos.x, beliefPos.y, beliefPos.z);
      });
      if (linePts.length > 0) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(linePts, 3));
        const mat = new THREE.LineDashedMaterial({
          color: 0xff5fd1, dashSize: 0.05, gapSize: 0.04, transparent: true, opacity: 0.55,
        });
        const lines = new THREE.LineSegments(geo, mat);
        lines.computeLineDistances();
        contentGroup.add(lines);
      }

      const beliefGlow = makeGlowSprite("#ff5fd1");
      beliefGlow.position.copy(beliefPos);
      beliefGlow.scale.set(0.7, 0.7, 1);
      contentGroup.add(beliefGlow);
      breathingRef.current.push(beliefGlow);

      const beliefDot = makeFlatDotSprite("#ff5fd1", "#ffffff");
      beliefDot.position.copy(beliefPos);
      beliefDot.scale.set(0.13, 0.13, 1);
      contentGroup.add(beliefDot);

      const beliefLabel = makeLabelSprite(BELIEF_NODE.label, "#ffd8f2");
      beliefLabel.position.set(beliefPos.x, beliefPos.y + 0.22, beliefPos.z);
      beliefLabel.scale.set(0.58, 0.13, 1);
      contentGroup.add(beliefLabel);
    }
  }, [mode]);

  return (
    <canvas
      ref={canvasRef}
      style={{ display: "block", width: "100%", height: "100%" }}
    />
  );
}
