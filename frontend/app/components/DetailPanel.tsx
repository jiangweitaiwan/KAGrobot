"use client";

import { useState, useCallback } from "react";
import type { GlobalLayoutNode } from "./GlobalBrain";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type PersonalBrainConcept = {
  eid:       string;
  label:     string;
  source:    "manual" | "llm_suggest" | "path_confirm";
  weight:    number;   // 0–1
  addedAt:   number;
};

export type ConceptDetail = {
  found:          boolean;
  eid:            string;
  rc_name:        string;
  description:    string | null;
  function:       string | null;
  analysis_types: string[];
  concept_type:   string | null;
  papers: {
    title:       string;
    premises:    string | null;
    constraints: string | null;
  }[];
};

export type LLMSuggestion = {
  eid:        string;
  label:      string;
  reason:     string;
  confidence: number;
};

export type RCScore = {
  eid:          string;
  label:        string;
  base_score:   number;
  personal_score: number;
};

type Props = {
  node:           GlobalLayoutNode | null;
  detail:         ConceptDetail | null;
  loading:        boolean;
  personalBrain:  PersonalBrainConcept[];
  suggestions:    LLMSuggestion[];
  rcScores:       RCScore[];
  onAdd:          (eid: string, label: string, source: PersonalBrainConcept["source"]) => void;
  onRemove:       (eid: string) => void;
  onUpdateWeight: (eid: string, weight: number) => void;
  onSuggestionAdd:(s: LLMSuggestion) => void;
};

// ─────────────────────────────────────────────────────────────
// DetailPanel
// ─────────────────────────────────────────────────────────────

export default function DetailPanel({
  node,
  detail,
  loading,
  personalBrain,
  suggestions,
  rcScores,
  onAdd,
  onRemove,
  onUpdateWeight,
  onSuggestionAdd,
}: Props) {
  const [showWeightSlider, setShowWeightSlider] = useState(false);

  const brainEntry = node
    ? personalBrain.find((c) => c.eid === node.id) ?? null
    : null;
  const isInBrain = !!brainEntry;

  const handleWeightChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!node) return;
      onUpdateWeight(node.id, parseFloat(e.target.value));
    },
    [node, onUpdateWeight],
  );

  // ── empty state ────────────────────────────────────────────
  if (!node) {
    return (
      <div className="detail-panel">
        <div className="detail-label" style={{ flexShrink: 0 }}>Detail Panel</div>
        <div className="detail-empty">
          <span>點擊 Global Brain</span>
          <span>查閱 concept 詳情</span>
        </div>
      </div>
    );
  }

  // ── loading state ──────────────────────────────────────────
  const showLoader = loading && !detail;

  // ── derived values ─────────────────────────────────────────
  const nodeSource = (node.source || "results").toLowerCase();
  const nodeType   = (node.type || "").toLowerCase();
  const isRC       = nodeType.includes("resulting");

  const topScores = rcScores
    .slice()
    .sort((a, b) => b.personal_score - a.personal_score)
    .slice(0, 3);

  return (
    <div className="detail-panel">

      {/* ── Header ── */}
      <div className="detail-header">
        <div className="detail-title-row">
          <span className="detail-label">Detail Panel</span>
          <span className="detail-concept-name">
            {detail?.rc_name ?? node.name}
          </span>
        </div>
        {isInBrain && (
          <span style={{
            fontSize: "0.62rem",
            color: "var(--teal)",
            flexShrink: 0,
            alignSelf: "flex-start",
            marginTop: "14px",
          }}>
            ◉ in brain
          </span>
        )}
      </div>

      {/* ── Loading bar ── */}
      {showLoader && (
        <div className="detail-loading-bar" style={{ marginTop: 2 }} />
      )}

      {/* ── Identity rows ── */}
      <div className="detail-section">
        <div className="detail-section-label">Identity</div>
        <div className="detail-row">
          <span className="detail-row-key">Type</span>
          <span className="detail-row-value">
            {isRC ? "Resulting Concept" : (node.type || "Concept")}
          </span>
        </div>
        <div className="detail-row">
          <span className="detail-row-key">Source</span>
          <span className="detail-row-value">
            {nodeSource === "chapter" ? "Textbook" : "Research"}
          </span>
        </div>
        {detail?.analysis_types && detail.analysis_types.length > 0 && (
          <div className="detail-row">
            <span className="detail-row-key">Analysis</span>
            <span className="detail-row-value">
              {detail.analysis_types.slice(0, 2).join(", ")}
            </span>
          </div>
        )}
        {detail?.papers && detail.papers.length > 0 && (
          <div className="detail-row">
            <span className="detail-row-key">Papers</span>
            <span className="detail-row-value">{detail.papers.length} matched</span>
          </div>
        )}
        <div className="detail-row">
          <span className="detail-row-key">In brain</span>
          <span className={`detail-row-value ${isInBrain ? "in-brain" : ""}`}>
            {isInBrain
              ? `● Yes · weight ${brainEntry!.weight.toFixed(2)}`
              : "—"}
          </span>
        </div>
      </div>

      {/* ── RC Score bars ── */}
      {topScores.length > 0 && (
        <div className="detail-section">
          <div className="detail-section-label">RC relevance</div>
          {topScores.map((s) => {
            const delta = s.personal_score - s.base_score;
            const hasPersonal = Math.abs(delta) > 0.005;
            return (
              <div key={s.eid} className="score-bar-wrap">
                <span className="score-bar-label" title={s.label}>{s.label}</span>
                <div className="score-bar-track">
                  <div
                    className={`score-bar-fill ${hasPersonal ? "personal" : ""}`}
                    style={{ width: `${s.personal_score * 100}%` }}
                  />
                </div>
                <span className="score-bar-num">
                  {s.personal_score.toFixed(2)}
                </span>
                {hasPersonal && (
                  <span className="score-bar-delta">
                    {delta > 0 ? "▲" : "▼"} {Math.abs(delta).toFixed(2)}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Theory background ── */}
      {detail && (
        <div className="detail-section">
          <div className="detail-section-label">Theory</div>
          {detail.description && (
            <p className="detail-theory-text" style={{ marginTop: 0 }}>
              {detail.description.slice(0, 200)}
              {detail.description.length > 200 ? "…" : ""}
            </p>
          )}
          {detail.papers?.[0]?.premises && (
            <>
              <div className="detail-theory-label">Premises</div>
              <p className="detail-theory-text" style={{ margin: 0 }}>
                {detail.papers[0].premises!.slice(0, 160)}
                {detail.papers[0].premises!.length > 160 ? "…" : ""}
              </p>
            </>
          )}
          {detail.papers?.[0]?.constraints && (
            <>
              <div className="detail-theory-label">Constraints</div>
              <p className="detail-theory-text" style={{ margin: 0 }}>
                {detail.papers[0].constraints!.slice(0, 160)}
                {detail.papers[0].constraints!.length > 160 ? "…" : ""}
              </p>
            </>
          )}
        </div>
      )}

      {/* ── LLM Suggestions ── */}
      {suggestions.length > 0 && (
        <div className="detail-section">
          <div className="detail-section-label">
            LLM suggested
            <span style={{ color: "var(--muted)", marginLeft: 4, fontWeight: 400 }}>
              · click to add
            </span>
          </div>
          <div className="suggestion-chips">
            {suggestions.map((s) => {
              const alreadyIn = personalBrain.some((c) => c.eid === s.eid);
              return (
                <button
                  key={s.eid}
                  className={`suggestion-chip ${alreadyIn ? "confirmed" : "pending"}`}
                  onClick={() => !alreadyIn && onSuggestionAdd(s)}
                  title={`${s.reason} (${(s.confidence * 100).toFixed(0)}%)`}
                  disabled={alreadyIn}
                >
                  {alreadyIn ? "✓ " : "+ "}
                  {s.label}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Weight slider (conditional) ── */}
      {isInBrain && showWeightSlider && (
        <div className="detail-section">
          <div className="weight-row">
            <span>Weight</span>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={brainEntry!.weight}
              onChange={handleWeightChange}
            />
            <span className="weight-val">{brainEntry!.weight.toFixed(2)}</span>
          </div>
        </div>
      )}

      {/* ── Actions ── */}
      <div className="detail-actions">
        {!isInBrain ? (
          <button
            className="detail-btn-add"
            onClick={() => node && onAdd(node.id, node.name, "manual")}
          >
            + Add to personal brain
          </button>
        ) : (
          <>
            <button
              className="detail-btn-add in-brain"
              disabled
            >
              ◉ In personal brain
            </button>
            <div className="detail-btn-row">
              <button
                className="detail-btn-secondary detail-btn-weight"
                onClick={() => setShowWeightSlider((v) => !v)}
              >
                {showWeightSlider ? "Hide weight" : "Adjust weight"}
              </button>
              <button
                className="detail-btn-secondary detail-btn-remove"
                onClick={() => node && onRemove(node.id)}
              >
                Remove
              </button>
            </div>
          </>
        )}
      </div>

    </div>
  );
}
