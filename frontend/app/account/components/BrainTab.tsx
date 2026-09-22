"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import BertopicMap, { type BertopicLayoutData } from "../../components/BertopicMap";
import type { PersonalBrainConcept } from "../../components/DetailPanel";
import { accountApi, type MyUsage } from "../../../lib/accountApi";
import { ConfirmButton, fmt, useElementSize } from "./ui";

const SOURCE_LABEL: Record<PersonalBrainConcept["source"], string> = {
  manual:       "手動加入",
  llm_suggest:  "語意推薦",
  path_confirm: "路徑確認",
};

const MAP_HEIGHT = 440;

export default function BrainTab({ uid, usage }: { uid: string; usage: MyUsage | null }) {
  const [concepts, setConcepts] = useState<PersonalBrainConcept[] | null>(null);
  const [order, setOrder]       = useState<string[]>([]);
  const [layout, setLayout]     = useState<BertopicLayoutData | null>(null);
  const [error, setError]       = useState("");
  const [query, setQuery]       = useState("");

  // 拖曳滑桿時列表不可跳動，所以排序只在「載入」與「放開滑桿（commit）」時重算。
  const conceptsRef  = useRef<PersonalBrainConcept[]>([]);
  const committedRef = useRef<Map<string, number>>(new Map());   // 伺服器上已保存的權重，避免重複 PATCH
  const mapBoxRef   = useRef<HTMLDivElement>(null);
  const mapSize     = useElementSize(mapBoxRef, { w: 420, h: MAP_HEIGHT });

  const resort = useCallback((list: PersonalBrainConcept[]) => {
    setOrder([...list].sort((a, b) => b.weight - a.weight).map((c) => c.eid));
  }, []);

  const replace = useCallback((list: PersonalBrainConcept[]) => {
    conceptsRef.current = list;
    setConcepts(list);
  }, []);

  useEffect(() => {
    let alive = true;
    accountApi.brain(uid)
      .then((list) => {
        if (!alive) return;
        committedRef.current = new Map(list.map((c) => [c.eid, c.weight]));
        replace(list); resort(list);
      })
      .catch((err) => { if (alive) { setError(err instanceof Error ? err.message : "讀取 Personal Brain 失敗"); replace([]); } });
    accountApi.bertopicLayout()
      .then((d) => alive && setLayout(d))
      .catch(() => { /* 分佈圖只是輔助視覺，失敗時保留清單即可 */ });
    return () => { alive = false; };
  }, [uid, replace, resort]);

  const byEid = useMemo(() => new Map((concepts ?? []).map((c) => [c.eid, c])), [concepts]);

  const visibleEids = useMemo(() => {
    const q = query.trim().toLowerCase();
    return order.filter((eid) => {
      const c = byEid.get(eid);
      return c && (!q || c.label.toLowerCase().includes(q));
    });
  }, [order, byEid, query]);

  function setWeightLocal(eid: string, weight: number) {
    replace(conceptsRef.current.map((c) => (c.eid === eid ? { ...c, weight } : c)));
  }

  async function commitWeight(eid: string) {
    const c = conceptsRef.current.find((x) => x.eid === eid);
    if (!c || committedRef.current.get(eid) === c.weight) return;
    try {
      await accountApi.setBrainWeight(uid, eid, c.weight);
      committedRef.current.set(eid, c.weight);
      resort(conceptsRef.current);
    } catch (err) {
      setError(err instanceof Error ? err.message : "儲存權重失敗");
    }
  }

  async function remove(eid: string) {
    const before = conceptsRef.current;
    replace(before.filter((c) => c.eid !== eid));
    setOrder((prev) => prev.filter((e) => e !== eid));
    try {
      await accountApi.removeBrainConcept(uid, eid);
    } catch (err) {
      setError(err instanceof Error ? err.message : "移除失敗");
      replace(before);
      resort(before);
    }
  }

  function exportJson() {
    const payload = JSON.stringify({ exported_at: new Date().toISOString(), concepts: conceptsRef.current }, null, 2);
    const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `personal-brain-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (concepts === null) return <div className="acct-loading">載入 Personal Brain…</div>;

  if (concepts.length === 0) {
    return (
      <div className="acct-empty">
        <strong>Personal Brain 還是空的</strong>
        <span>在對話頁點選知識圖上的概念，把 Foundation 加入，這裡就會呈現你的知識偏向。</span>
        <Link href="/dashboard" className="acct-btn primary">前往對話頁</Link>
        {error && <span className="acct-error">{error}</span>}
      </div>
    );
  }

  const limit    = usage?.limits.brain_max_concepts ?? 0;
  const capped   = !!usage?.enforced && limit > 0;
  const fillRate = capped ? concepts.length / limit : 0;

  return (
    <div className="acct-brain">
      <div className="acct-brain-list">
        <div className="acct-toolbar" style={{ marginBottom: 0 }}>
          <span>
            {capped ? `${fmt(concepts.length)} / ${fmt(limit)} 個概念` : `${fmt(concepts.length)} 個概念`}
          </span>
          <button type="button" className="acct-btn" onClick={exportJson}>匯出 JSON</button>
        </div>

        {capped && (
          <div className={`acct-meter${fillRate >= 1 ? " bad" : fillRate >= 0.8 ? " warn" : ""}`} aria-hidden="true">
            <span style={{ width: `${Math.min(100, fillRate * 100)}%` }} />
          </div>
        )}

        <input
          className="acct-input"
          placeholder="搜尋概念"
          aria-label="搜尋概念"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {error && <p className="acct-error">{error}</p>}

        <ul className="acct-concepts">
          {visibleEids.map((eid) => {
            const c = byEid.get(eid)!;
            return (
              <li key={eid} className="acct-concept">
                <span className="acct-concept-name" title={c.label}>
                  {c.label}<span className="acct-concept-src">{SOURCE_LABEL[c.source]}</span>
                </span>
                <ConfirmButton label="移除" confirmLabel="確定移除" className="acct-btn quiet" onConfirm={() => remove(eid)} />
                <label className="acct-slider">
                  <span className="sr-only">{c.label} 的權重</span>
                  <input
                    type="range" min={0.1} max={1} step={0.05} value={c.weight}
                    onChange={(e) => setWeightLocal(eid, parseFloat(e.target.value))}
                    onPointerUp={() => commitWeight(eid)}
                    onKeyUp={() => commitWeight(eid)}
                    onBlur={() => commitWeight(eid)}
                  />
                  <output>{c.weight.toFixed(2)}</output>
                </label>
              </li>
            );
          })}
          {visibleEids.length === 0 && <li className="acct-note" style={{ padding: 14 }}>沒有符合「{query}」的概念。</li>}
        </ul>
      </div>

      <div className="acct-map" ref={mapBoxRef} style={{ height: MAP_HEIGHT }}>
        {layout ? (
          <BertopicMap data={layout} personalBrain={concepts} width={mapSize.w} height={MAP_HEIGHT} />
        ) : (
          <div className="acct-loading">載入知識分佈圖…</div>
        )}
      </div>
    </div>
  );
}
