"use client";

/**
 * IntroHourglass.tsx — /intro 頁面「行動」分頁的簡化沙漏剖面圖。
 *
 * 呼應正式系統的 HourglassViz.tsx，但只保留敘事需要的最小結構：
 *   waist（腰身） = 剛形成的信念（BELIEF_NODE）
 *   下錐（發散）  = 支撐這個信念的 foundation（FOUNDATION_IDS）
 *   上錐（收斂到一點）= 信念驅動出的具體行動（ACTION_NODE）
 *
 * 不需要互動（點擊聚焦、skill 徽章選取等），純粹是靜態旋轉 + 一次性
 * 的「信念 → 行動」箭頭光點動畫，作為 pipeline 敘事的收尾畫面。
 */

import { useEffect, useRef } from "react";
import * as THREE from "three";
import {
  makeGlowSprite,
  makeFlatDotSprite,
  makeLabelSprite,
  disposeObject3D,
} from "../../components/three-shared";
import { FOUNDATION_IDS, BELIEF_NODE, ACTION_NODE, ALL_SPHERE_NODES } from "../demoData";

type Props = { width: number; height: number };

const RIM_RADIUS = 1.3;
const HALF_HEIGHT = 1.5;

function buildConeWireframe(rimY: number, radius: number): THREE.Group {
  const group = new THREE.Group();
  const material = new THREE.LineDashedMaterial({
    color: 0x3a4a68, dashSize: 0.06, gapSize: 0.05, transparent: true, opacity: 0.5,
  });

  const segs = 40;
  const rimPts: THREE.Vector3[] = [];
  for (let i = 0; i <= segs; i++) {
    const a = (i / segs) * Math.PI * 2;
    rimPts.push(new THREE.Vector3(radius * Math.cos(a), rimY, radius * Math.sin(a)));
  }
  const rimLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(rimPts), material);
  rimLine.computeLineDistances();
  group.add(rimLine);

  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(radius * Math.cos(a), rimY, radius * Math.sin(a)),
    ]);
    const line = new THREE.Line(geo, material);
    line.computeLineDistances();
    group.add(line);
  }
  return group;
}

export default function IntroHourglass({ width, height }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const widthRef = useRef(width);
  const heightRef = useRef(height);
  const animFrameRef = useRef<number>(0);

  useEffect(() => { widthRef.current = width; heightRef.current = height; }, [width, height]);

  useEffect(() => {
    if (!canvasRef.current) return;

    const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true, alpha: true });
    renderer.setSize(widthRef.current, heightRef.current);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setClearColor(0x000000, 0);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, widthRef.current / Math.max(heightRef.current, 1), 0.01, 100);
    camera.position.set(0, 0.3, 4.2);
    cameraRef.current = camera;

    const rotGroup = new THREE.Group();
    scene.add(rotGroup);

    rotGroup.add(buildConeWireframe(HALF_HEIGHT, RIM_RADIUS));
    rotGroup.add(buildConeWireframe(-HALF_HEIGHT, RIM_RADIUS));
    scene.add(new THREE.AmbientLight(0xffffff, 1));

    // ── waist：信念 ──
    const waistGlow = makeGlowSprite("#00d4aa");
    waistGlow.scale.set(0.6, 0.6, 1);
    rotGroup.add(waistGlow);
    const waistDot = makeFlatDotSprite("#00d4aa", "#00ffcc");
    waistDot.scale.set(0.13, 0.13, 1);
    rotGroup.add(waistDot);
    const waistLabel = makeLabelSprite(BELIEF_NODE.label, "#00ffcc");
    waistLabel.position.set(0, 0.2, 0);
    waistLabel.scale.set(0.6, 0.13, 1);
    rotGroup.add(waistLabel);

    // ── 下錐：foundation（散開在 rim 上）──
    const nodeById = new Map(ALL_SPHERE_NODES.map((n) => [n.id, n]));
    const foundationPositions: THREE.Vector3[] = [];
    FOUNDATION_IDS.forEach((id, i) => {
      const n = nodeById.get(id);
      const angle = (i / FOUNDATION_IDS.length) * Math.PI * 2;
      const pos = new THREE.Vector3(RIM_RADIUS * Math.cos(angle), -HALF_HEIGHT, RIM_RADIUS * Math.sin(angle));
      foundationPositions.push(pos);

      const dot = makeFlatDotSprite("#7f77dd", "#e8ddff");
      dot.position.copy(pos);
      dot.scale.set(0.07, 0.07, 1);
      rotGroup.add(dot);

      const label = makeLabelSprite(n?.label ?? id, "#c9aaff");
      label.position.set(pos.x, pos.y - 0.16, pos.z);
      label.scale.set(0.46, 0.1, 1);
      rotGroup.add(label);

      const edgeGeo = new THREE.BufferGeometry().setFromPoints([pos, new THREE.Vector3(0, 0, 0)]);
      const edgeMat = new THREE.LineBasicMaterial({ color: 0x534ab7, transparent: true, opacity: 0.5 });
      rotGroup.add(new THREE.Line(edgeGeo, edgeMat));
    });

    // ── 上錐：行動（單點收斂於頂點）──
    const actionPos = new THREE.Vector3(0, HALF_HEIGHT, 0);
    const actionEdge = new THREE.LineBasicMaterial({ color: 0xffcc4d, transparent: true, opacity: 0.7 });
    rotGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), actionPos]),
      actionEdge,
    ));
    const actionGlow = makeGlowSprite("#ffcc4d");
    actionGlow.position.copy(actionPos);
    actionGlow.scale.set(0.45, 0.45, 1);
    rotGroup.add(actionGlow);
    const actionDot = makeFlatDotSprite("#ffcc4d", "#fff3c4");
    actionDot.position.copy(actionPos);
    actionDot.scale.set(0.1, 0.1, 1);
    rotGroup.add(actionDot);
    const actionLabel = makeLabelSprite(ACTION_NODE.label, "#ffcc4d");
    actionLabel.position.set(actionPos.x, actionPos.y + 0.2, actionPos.z);
    actionLabel.scale.set(0.58, 0.13, 1);
    rotGroup.add(actionLabel);

    // ── 接力光點：信念 → 行動，象徵推理落地 ──
    const pulseDot = makeFlatDotSprite("#ffcc4d");
    pulseDot.scale.set(0.05, 0.05, 1);
    rotGroup.add(pulseDot);
    let pulsePhase = 0;

    let lastT = 0;
    function animate(now: number) {
      animFrameRef.current = requestAnimationFrame(animate);
      if (now - lastT < 1000 / 30) return;
      lastT = now;

      rotGroup.rotation.y += 0.0014;

      const breathe = 0.6 + 0.4 * Math.sin(now * 0.0022);
      (waistGlow.material as THREE.SpriteMaterial).opacity = breathe;

      pulsePhase = (pulsePhase + 0.006) % 1;
      pulseDot.position.lerpVectors(new THREE.Vector3(0, 0, 0), actionPos, pulsePhase);
      (pulseDot.material as THREE.SpriteMaterial).opacity = 1 - Math.abs(pulsePhase - 0.5) * 1.2;

      renderer.render(scene, camera);
    }
    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      disposeObject3D(scene);
      renderer.dispose();
      rendererRef.current = null;
      cameraRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const renderer = rendererRef.current;
    const camera = cameraRef.current;
    if (!renderer || !camera || width <= 0 || height <= 0) return;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }, [width, height]);

  return <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />;
}
