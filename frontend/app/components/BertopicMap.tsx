"use client";

/**
 * BertopicMap.tsx — BERTopic 2D 流派投影圖
 *
 * 職責：
 *   - 用 D3 繪製 BERTopic UMAP 2D 散點圖
 *   - 所有 concept 以淡色小點呈現語義地形
 *   - Personal Brain concepts 以亮色大點疊加，顯示「知識偏向」
 *   - Hover 顯示 concept name + 流派
 *   - 流派色塊以 convex hull label 標示（D3 convexHull）
 *
 * 設計語言延續 globals.css（暗底、IBM Plex Mono、var(--purple) 等）。
 */

import { useEffect, useRef, useMemo } from "react";
import * as d3 from "d3";
import type { PersonalBrainConcept } from "./DetailPanel";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type BertopicNode = {
  eid:         string | null;
  name:        string;
  x:           number;
  y:           number;
  topic_id:    number;
  topic_label: string;
  color:       string;
};

export type BertopicTopic = {
  id:    number;
  label: string;
  color: string;
  words: string[];
};

export type BertopicLayoutData = {
  nodes:  BertopicNode[];
  topics: BertopicTopic[];
};

type Props = {
  data:          BertopicLayoutData;
  personalBrain: PersonalBrainConcept[];
  width:         number;
  height:        number;
};

// ─────────────────────────────────────────────────────────────
// Design tokens（與現有 dark palette 一致）
// ─────────────────────────────────────────────────────────────

const T = {
  bg:           "#06080f",
  dot_bg:       "rgba(60,80,120,0.25)",   // 背景點（未選中）
  dot_unmatched:"rgba(40,50,80,0.18)",    // eid=null 的點
  brain_stroke: "#ffffff",
  label_bg:     "rgba(13,18,32,0.82)",
  label_border: "rgba(232,238,248,0.15)",
  label_text:   "#c8daf0",
  topic_label:  "rgba(232,238,248,0.22)",
  axis_tick:    "rgba(120,140,180,0.18)",
  tooltip_bg:   "rgba(13,18,32,0.95)",
};

const PAD = { top: 28, right: 20, bottom: 32, left: 32 };

// ─────────────────────────────────────────────────────────────
// BertopicMap
// ─────────────────────────────────────────────────────────────

export default function BertopicMap({
  data,
  personalBrain,
  width,
  height,
}: Props) {
  const svgRef     = useRef<SVGSVGElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  // Personal Brain の eid セット（重畳表示用）
  const brainEidSet = useMemo(
    () => new Set(personalBrain.map((c) => c.eid)),
    [personalBrain],
  );
  const brainWeightMap = useMemo(
    () => new Map(personalBrain.map((c) => [c.eid, c.weight])),
    [personalBrain],
  );

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const W = width  - PAD.left - PAD.right;
    const H = height - PAD.top  - PAD.bottom;

    // ── 尺度 ─────────────────────────────────────────────────
    const xExt = d3.extent(data.nodes, (d) => d.x) as [number, number];
    const yExt = d3.extent(data.nodes, (d) => d.y) as [number, number];

    // 留一點 padding 讓邊界點不被裁掉
    const xPad = (xExt[1] - xExt[0]) * 0.05;
    const yPad = (yExt[1] - yExt[0]) * 0.05;

    const xScale = d3.scaleLinear().domain([xExt[0] - xPad, xExt[1] + xPad]).range([0, W]);
    const yScale = d3.scaleLinear().domain([yExt[0] - yPad, yExt[1] + yPad]).range([H, 0]);

    // ── root SVG ─────────────────────────────────────────────
    const root = svg
      .attr("width",  width)
      .attr("height", height)
      .style("background", T.bg);

    const g = root.append("g").attr("transform", `translate(${PAD.left},${PAD.top})`);

    // ── zoom ─────────────────────────────────────────────────
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 8])
      .on("zoom", (event) => {
        g.attr("transform",
          `translate(${PAD.left + event.transform.x},${PAD.top + event.transform.y}) scale(${event.transform.k})`
        );
      });
    root.call(zoom);

    // ── 極細軸線（裝飾用）────────────────────────────────────
    g.append("g")
      .attr("transform", `translate(0,${H})`)
      .call(
        d3.axisBottom(xScale).ticks(5).tickSize(-H)
          .tickFormat(() => "")
      )
      .call((ax) => {
        ax.select(".domain").remove();
        ax.selectAll("line").attr("stroke", T.axis_tick);
      });

    g.append("g")
      .call(
        d3.axisLeft(yScale).ticks(5).tickSize(-W)
          .tickFormat(() => "")
      )
      .call((ax) => {
        ax.select(".domain").remove();
        ax.selectAll("line").attr("stroke", T.axis_tick);
      });

    // ── topic cluster labels（取各 topic 的重心放標籤）────────
    const topicMap = new Map(data.topics.map((t) => [t.id, t]));

    // 計算各 topic 重心
    const topicCentroids = new Map<number, { cx: number; cy: number }>();
    const topicGroups = d3.group(data.nodes.filter((n) => n.topic_id !== -1), (n) => n.topic_id);
    topicGroups.forEach((nodes, tid) => {
      const cx = d3.mean(nodes, (n) => xScale(n.x))!;
      const cy = d3.mean(nodes, (n) => yScale(n.y))!;
      topicCentroids.set(tid, { cx, cy });
    });

    // 流派 label（半透明浮水印）
    topicCentroids.forEach(({ cx, cy }, tid) => {
      const topic = topicMap.get(tid);
      if (!topic) return;
      // 取流派名稱中最有辨識度的部分（去掉 BERTopic 自動加的 topic id 前綴）
      const displayLabel = topic.label.replace(/^\d+_/, "").replace(/_/g, " ");
      g.append("text")
        .attr("x", cx).attr("y", cy)
        .attr("text-anchor", "middle")
        .attr("font-size", "9px")
        .attr("font-family", "IBM Plex Mono, monospace")
        .attr("letter-spacing", "0.12em")
        .attr("text-transform", "uppercase")
        .attr("fill", topic.color)
        .attr("opacity", 0.28)
        .attr("pointer-events", "none")
        .text(displayLabel.toUpperCase());
    });

    // ── 背景點（所有 concept，依 topic 著色）─────────────────
    g.append("g").attr("class", "bg-dots")
      .selectAll<SVGCircleElement, BertopicNode>("circle")
      .data(data.nodes.filter((n) => !brainEidSet.has(n.eid ?? "")))
      .join("circle")
      .attr("cx",   (d) => xScale(d.x))
      .attr("cy",   (d) => yScale(d.y))
      .attr("r",    1.8)
      .attr("fill", (d) => {
        if (!d.eid) return T.dot_unmatched;
        // 用 topic 色但極度淡化
        const base = d3.color(d.color);
        return base ? base.copy({ opacity: 0.22 }).formatRgb() : T.dot_bg;
      })
      .attr("pointer-events", "none");

    // ── Personal Brain 疊加點 ─────────────────────────────────
    const brainNodes = data.nodes.filter(
      (n) => n.eid && brainEidSet.has(n.eid)
    );

    // 光暈 glow（底層）
    g.append("g").attr("class", "brain-glow")
      .selectAll<SVGCircleElement, BertopicNode>("circle")
      .data(brainNodes)
      .join("circle")
      .attr("cx",    (d) => xScale(d.x))
      .attr("cy",    (d) => yScale(d.y))
      .attr("r",     (d) => {
        const w = brainWeightMap.get(d.eid!) ?? 0.5;
        return 6 + w * 8;
      })
      .attr("fill",  (d) => {
        const base = d3.color(d.color);
        return base ? base.copy({ opacity: 0.12 }).formatRgb() : "transparent";
      })
      .attr("stroke",       "none")
      .attr("pointer-events", "none");

    // 主點
    const brainDots = g.append("g").attr("class", "brain-dots")
      .selectAll<SVGCircleElement, BertopicNode>("circle")
      .data(brainNodes)
      .join("circle")
      .attr("cx",           (d) => xScale(d.x))
      .attr("cy",           (d) => yScale(d.y))
      .attr("r",            (d) => {
        const w = brainWeightMap.get(d.eid!) ?? 0.5;
        return 4 + w * 4;
      })
      .attr("fill",         (d) => d.color)
      .attr("stroke",       T.brain_stroke)
      .attr("stroke-width", 1)
      .attr("stroke-opacity", 0.6)
      .style("cursor", "default");

    // 名稱標籤（浮在點旁邊）
    g.append("g").attr("class", "brain-labels")
      .selectAll<SVGTextElement, BertopicNode>("text")
      .data(brainNodes)
      .join("text")
      .attr("x",           (d) => xScale(d.x) + 7)
      .attr("y",           (d) => yScale(d.y) + 4)
      .attr("font-size",   "7px")
      .attr("font-family", "IBM Plex Mono, monospace")
      .attr("fill",        (d) => d.color)
      .attr("opacity",     0.85)
      .attr("pointer-events", "none")
      .text((d) => d.name.length > 18 ? d.name.slice(0, 17) + "…" : d.name);

    // ── Tooltip on hover ──────────────────────────────────────
    brainDots
      .on("mouseenter", (event: MouseEvent, d) => {
        if (!tooltipRef.current) return;
        const tip = tooltipRef.current;
        const topic = topicMap.get(d.topic_id);
        tip.innerHTML = [
          `<strong style="color:${d.color}">${d.name}</strong>`,
          `<span style="color:rgba(200,218,240,0.55);font-size:0.65em">${
            topic?.label.replace(/^\d+_/, "").replace(/_/g, " ") ?? "Noise"
          }</span>`,
          topic?.words?.length
            ? `<span style="color:rgba(160,180,220,0.4);font-size:0.6em">${topic.words.join(" · ")}</span>`
            : "",
        ].filter(Boolean).join("<br/>");
        tip.style.display = "block";
        const rect = (event.currentTarget as SVGElement).closest("svg")!.getBoundingClientRect();
        const svgEl = (event.currentTarget as SVGElement).closest("svg")!;
        const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(
          (svgEl as SVGSVGElement).getScreenCTM()!.inverse()
        );
        // px 相對於 SVG 本身（含 PAD）
        tip.style.left = `${event.offsetX + 14}px`;
        tip.style.top  = `${event.offsetY - 10}px`;
      })
      .on("mousemove", (event: MouseEvent) => {
        if (!tooltipRef.current) return;
        tooltipRef.current.style.left = `${event.offsetX + 14}px`;
        tooltipRef.current.style.top  = `${event.offsetY - 10}px`;
      })
      .on("mouseleave", () => {
        if (tooltipRef.current) tooltipRef.current.style.display = "none";
      });

    // ── 圖例（右上角，最多顯示 6 個流派）────────────────────
    const legendTopics = data.topics
      .filter((t) => t.id !== -1)
      .slice(0, 8);

    const legendG = g.append("g").attr("class", "legend")
      .attr("transform", `translate(${W - 4}, 4)`);

    legendTopics.forEach((topic, i) => {
      const row = legendG.append("g").attr("transform", `translate(0,${i * 14})`);
      row.append("circle").attr("r", 3.5).attr("fill", topic.color).attr("cx", -6).attr("cy", 0);
      row.append("text")
        .attr("x",           -12)
        .attr("y",           4)
        .attr("text-anchor", "end")
        .attr("font-size",   "7px")
        .attr("font-family", "IBM Plex Mono, monospace")
        .attr("fill",        topic.color)
        .attr("opacity",     0.55)
        .text(topic.label.replace(/^\d+_/, "").replace(/_/g, " ").slice(0, 16));
    });

    // ── 右下角：Brain 中各流派比例 bar ───────────────────────
    if (brainNodes.length > 0) {
      const topicCounts = new Map<number, number>();
      brainNodes.forEach((n) => {
        topicCounts.set(n.topic_id, (topicCounts.get(n.topic_id) ?? 0) + 1);
      });

      const sorted = [...topicCounts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4);

      const barG = g.append("g").attr("class", "brain-bias")
        .attr("transform", `translate(0,${H + 8})`);

      barG.append("text")
        .attr("font-size",   "7px")
        .attr("font-family", "IBM Plex Mono, monospace")
        .attr("fill",        "rgba(127,119,221,0.45)")
        .attr("letter-spacing", "0.1em")
        .text("BIAS ▸");

      let barX = 42;
      sorted.forEach(([tid, cnt]) => {
        const topic = topicMap.get(tid);
        if (!topic) return;
        const label = topic.label.replace(/^\d+_/, "").replace(/_/g, " ").slice(0, 12);
        const pct   = Math.round((cnt / brainNodes.length) * 100);

        barG.append("rect")
          .attr("x",      barX).attr("y",  -6)
          .attr("width",  pct * 1.2).attr("height", 6)
          .attr("rx",     2)
          .attr("fill",   topic.color)
          .attr("opacity", 0.55);

        barG.append("text")
          .attr("x",           barX + pct * 1.2 + 4)
          .attr("y",           0)
          .attr("font-size",   "6.5px")
          .attr("font-family", "IBM Plex Mono, monospace")
          .attr("fill",        topic.color)
          .attr("opacity",     0.7)
          .text(`${label} ${pct}%`);

        barX += pct * 1.2 + label.length * 4.2 + 16;
      });
    }

  }, [data, width, height, brainEidSet, brainWeightMap]);

  // empty state
  if (!data.nodes.length) {
    return (
      <div style={{
        width, height,
        display:        "flex",
        alignItems:     "center",
        justifyContent: "center",
        color:          "rgba(127,119,221,0.2)",
        fontSize:       "0.65rem",
        letterSpacing:  "0.14em",
        fontFamily:     "monospace",
        background:     T.bg,
        borderRadius:   "20px",
      }}>
        BERTopic layout 載入中…
      </div>
    );
  }

  return (
    <div style={{ position: "relative", width, height, flexShrink: 0 }}>
      <svg
        ref={svgRef}
        style={{ display: "block", width: "100%", height: "100%", borderRadius: "20px" }}
      />
      <div
        ref={tooltipRef}
        style={{
          display:       "none",
          position:      "absolute",
          pointerEvents: "none",
          background:    T.tooltip_bg,
          border:        "1px solid rgba(83,74,183,0.35)",
          borderRadius:  "8px",
          padding:       "7px 11px",
          fontSize:      "11px",
          color:         "#c8daf0",
          fontFamily:    "IBM Plex Mono, monospace",
          lineHeight:    "1.6",
          maxWidth:      "200px",
          zIndex:        20,
        }}
      />
      {/* 右上角標題 */}
      <div style={{
        position:      "absolute",
        top:           8,
        left:          12,
        fontSize:      "0.58rem",
        color:         "rgba(127,119,221,0.35)",
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        fontFamily:    "monospace",
        pointerEvents: "none",
      }}>
        Semantic Field · BERTopic
      </div>
    </div>
  );
}
