# -*- coding: utf-8 -*-
"""graph_service/walk_start_pool.py — Random Walk Mode 的起點檢索池。

【設計變更】
起點候選不再限定「leaf foundation concept」，而是放寬為「所有非 RC 的
Concept」——起點本身可以是任何中繼概念，不需要是某個 RC 的最終上游。

interScore 計算方式比照 frontend DetailPanel.tsx::computeRanking 的精神：
  - candidate 本身就在使用者的 Personal Brain 裡 → interScore 固定為 1.0。
    這是使用者明確選中的訊號，視為離散、確定性的最強權重，不用使用者
    自訂的 0.1~1.0 weight 稀釋，也不需要再去算它的 foundation set。
  - 否則 interScore = candidate 的 foundation set 中，被 Personal Brain
    命中的加權平均（weighted_sum(foundation 權重) / foundation 數）。

foundation set 的取得方式：不另外實作一套 upstream BFS，直接重用
graph_service/bfs_coordinator.py::BFSCoordinator.get_expand_cached_async()
——這與 RC foundation tree（/graph/expand/{eid}）走的是同一條路徑
（AdjacencyCache 鄰接層 + _expand_cache 結果層），只是起點從 RC 換成
任意非 RC concept；CachedUpstreamBFS.run() 本身沒有限制起點必須是 RC，
可以直接復用。

Two-stage 排序（避免對全池每個候選都跑一次 BFS）：
  1. 純語意排序（記憶體 dot product，便宜）取 top-n_sem。
  2. 聯集使用者 Personal Brain 裡本身合法（非 RC）的 candidate eid——
     確保使用者明確選中的概念一定進入候選，不會因語意分數不夠高被
     語意 prefilter 刷掉（它們的 interScore 必然是 1.0，理應排到前面）。
  3. 只對這個 shortlist（通常 <100）平行呼叫 coordinator 的 upstream
     BFS，取得各自 foundation set，計算 interScore，與 semantic 加權
     合併後重新排序，取最終 top-k。

Embedding 語料維持 name + (function or definition)，與 walk_frontier.py
的候選描述語料一致。

【Embedding self-healing GCS cache】
全量非 RC concept 的 SBERT encode 在每次冷啟動都重算太貴。改成：
  1. load() 時先向 Neo4j 取得目前完整的候選 eid 清單（權威來源）。
  2. 嘗試從 GCS 下載既有 embedding cache pkl（{eid: {name, embedding}}）。
  3. 兩者取差集，只對「pkl 裡缺少的 eid」（新概念、或第一次啟動整批
     缺）呼叫 SBERT encode。
  4. 若有任何新算的 embedding，合併後把完整結果重新上傳回 GCS，讓下次
     啟動可以純下載、不需要再 encode 這批。
  5. GCS 下載/上傳失敗一律 fallback 成「全量現算、不快取」，不讓這件
     事拖垮 random walk 的可用性（同 AdjacencyCache / GlobalLayoutBuilder
     的 best-effort 哲學）。

與 CandidateRetriever 的關係不變：職責不同、故意不合併——見前版檔頭
說明，這裡索引的是「起點候選」，語料與範圍都與 RC 專屬索引不同。
"""

from __future__ import annotations

import asyncio
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import numpy as np

from graph_service.walk_state import WalkNodeRef

if TYPE_CHECKING:
    from retrieval.encoder import SbertOnnxEncoder
    from graph_service.bfs_coordinator import BFSCoordinator

_BATCH_SIZE = 64
_DEFAULT_N_SEM = 8
_DEFAULT_GCS_OBJECT = "concept_name_function_embeddings.pkl"


class StartConceptPool:
    """非 RC concept 的語意搜尋索引，供 random walk 起點選取使用。

    Parameters
    ----------
    driver          : Neo4j driver
    encoder         : SbertOnnxEncoder 實例
    coordinator     : BFSCoordinator，用來對 shortlist candidate 取得
                       foundation set（upstream BFS 結果），與 RC
                       foundation tree 共用同一層快取。
    gcs_bucket_name : embedding cache pkl 所在的 GCS bucket。留空則不使用
                       GCS，每次啟動都全量現算（等同無快取行為）。
    gcs_object_name : pkl 物件名稱。
    """

    def __init__(
        self,
        driver: Any,
        encoder: "SbertOnnxEncoder",
        coordinator: "BFSCoordinator",
        *,
        gcs_bucket_name: str = "",
        gcs_object_name: str = _DEFAULT_GCS_OBJECT,
    ) -> None:
        self._driver          = driver
        self._encoder         = encoder
        self._coordinator     = coordinator
        self._gcs_bucket_name = gcs_bucket_name
        self._gcs_object_name = gcs_object_name

        self._index: List[Dict[str, Any]] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """建立/還原 embedding index（見檔頭 self-healing 流程說明）。

        同步、可能耗時（Neo4j 查詢 + 視情況 SBERT encode + GCS I/O），
        呼叫端應在背景執行緒或啟動階段呼叫，不要在 event loop 上直接跑。
        """
        rows   = self._fetch_candidate_concepts()
        cached = self._download_embedding_cache()

        missing_rows = [r for r in rows if r["eid"] not in cached]
        if missing_rows:
            print(f"[StartConceptPool] {len(missing_rows)}/{len(rows)} concepts "
                  f"missing from embedding cache, encoding…")
            newly_encoded = self._encode_rows(missing_rows)
            cached.update(newly_encoded)
            self._upload_embedding_cache(cached)
        else:
            print(f"[StartConceptPool] embedding cache hit for all {len(rows)} concepts.")

        # 以 Neo4j 目前的候選集合為準組裝最終 index：cached 裡可能還留著
        # 已被刪除節點的舊 embedding，這裡刻意不帶入，避免推薦到不存在的 eid。
        row_by_eid = {r["eid"]: r for r in rows}
        self._index = [
            {
                "eid":       eid,
                "name":      row_by_eid[eid]["name"] or eid,
                "type":      row_by_eid[eid].get("type"),
                "embedding": cached[eid]["embedding"],
            }
            for eid in row_by_eid
            if eid in cached
        ]
        self._loaded = True
        print(f"[StartConceptPool] loaded {len(self._index)} non-RC concepts.")

    async def retrieve(
        self,
        query_text:    str,
        brain_weights: Optional[Dict[str, float]] = None,
        k:             int = 3,
        alpha:         float = 0.6,
        n_sem:         int = _DEFAULT_N_SEM,
    ) -> List[WalkNodeRef]:
        """Two-stage 排序，回傳 top-k 起點候選（依
        alpha*semantic + (1-alpha)*interScore 排序）。

        見檔頭「Two-stage 排序」與「interScore 計算方式」說明。
        """
        if not self._loaded:
            await asyncio.to_thread(self.load)
        if not self._index:
            return []

        weights   = brain_weights or {}
        query_emb = self._encoder.encode(query_text)[0]

        semantic_scores: Dict[str, float] = {
            item["eid"]: float(np.dot(query_emb, item["embedding"]))
            for item in self._index
        }

        ranked_by_semantic = sorted(
            semantic_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        shortlist_eids: Set[str] = {eid for eid, _ in ranked_by_semantic[:n_sem]}

        # 使用者 Personal Brain 裡本身是合法候選（非 RC）的 eid 一併納入，
        # 確保它們不會因語意分數落在 top-n_sem 之外而被漏掉——這些一定會
        # 被 _inter_score 的 self-select 特例給滿分，理應出現在最終排序裡。
        index_eids      = set(semantic_scores.keys())
        shortlist_eids  |= (set(weights.keys()) & index_eids)

        if not shortlist_eids:
            return []

        foundation_sets = await self._collect_foundation_sets(shortlist_eids)

        name_by_eid = {item["eid"]: item["name"] for item in self._index}
        scored: List[tuple] = []
        for eid in shortlist_eids:
            semantic = semantic_scores.get(eid, 0.0)
            inter    = self._inter_score(eid, foundation_sets.get(eid, []), weights)
            total    = alpha * semantic + (1 - alpha) * inter
            scored.append((total, eid))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            WalkNodeRef(eid=eid, label=name_by_eid.get(eid, eid))
            for _score, eid in scored[:k]
        ]

    @property
    def size(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Private — interScore
    # ------------------------------------------------------------------

    @staticmethod
    def _inter_score(
        candidate_eid:   str,
        foundation_eids: List[str],
        weights:         Dict[str, float],
    ) -> float:
        if candidate_eid in weights:
            return 1.0
        n = len(foundation_eids)
        if n == 0:
            return 0.0
        weighted_sum = sum(weights.get(f, 0.0) for f in foundation_eids)
        return weighted_sum / n

    # ------------------------------------------------------------------
    # Private — foundation set（重用 BFSCoordinator，不另寫 upstream 邏輯）
    # ------------------------------------------------------------------

    async def _collect_foundation_sets(self, eids: Set[str]) -> Dict[str, List[str]]:
        """對 shortlist 平行呼叫 coordinator 的 upstream BFS（走既有 expand
        cache，與 RC foundation tree 共用同一層鄰接／結果快取），取得各自
        的 foundation eid 清單。"""
        eid_list = list(eids)
        results = await asyncio.gather(
            *[self._coordinator.get_expand_cached_async(eid) for eid in eid_list]
        )
        return {
            eid: [f["element_id"] for f in r.get("final_foundations", [])]
            for eid, r in zip(eid_list, results)
        }

    # ------------------------------------------------------------------
    # Private — Neo4j：候選集合（所有非 RC concept）
    # ------------------------------------------------------------------

    def _fetch_candidate_concepts(self) -> List[Dict[str, Any]]:
        with self._driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Concept)
                WHERE c.type <> 'Resulting_Concept' OR c.type IS NULL
                RETURN elementId(c) AS eid,
                    c.name       AS name,
                    c.type       AS type,
                    c.definition AS definition,
                    c.function   AS function
                """
            ).data()
        return rows

    # ------------------------------------------------------------------
    # Private — SBERT encode（只對缺少的部分呼叫）
    # ------------------------------------------------------------------

    def _encode_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        texts = [
            f"{r['name']} {r.get('function') or r.get('definition') or ''}".strip()
            for r in rows
        ]

        embs: List[np.ndarray] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            embs.append(self._encoder.encode(texts[i:i + _BATCH_SIZE]))
        emb_matrix = np.vstack(embs) if embs else np.zeros((0, 0))

        return {
            r["eid"]: {"name": r["name"] or r["eid"], "embedding": emb_matrix[idx]}
            for idx, r in enumerate(rows)
        }

    # ------------------------------------------------------------------
    # Private — GCS cache（self-healing：缺什麼補什麼，補完整批回寫）
    # ------------------------------------------------------------------

    def _download_embedding_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self._gcs_bucket_name:
            return {}
        try:
            from google.cloud import storage as gcs_storage
            client = gcs_storage.Client()
            bucket = client.bucket(self._gcs_bucket_name)
            blob   = bucket.blob(self._gcs_object_name)
            if not blob.exists(client):
                print(f"[StartConceptPool] no embedding cache at "
                      f"gs://{self._gcs_bucket_name}/{self._gcs_object_name}, "
                      f"will encode from scratch.")
                return {}
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                tmp_path = tmp.name
            blob.download_to_filename(tmp_path)
            with open(tmp_path, "rb") as f:
                raw = pickle.load(f)
            Path(tmp_path).unlink(missing_ok=True)
            print(f"[StartConceptPool] loaded embedding cache ({len(raw)} entries) "
                  f"from gs://{self._gcs_bucket_name}/{self._gcs_object_name}")
            return raw
        except Exception as exc:
            print(f"[StartConceptPool] embedding cache download failed: {exc}, "
                  f"will encode from scratch.")
            return {}

    def _upload_embedding_cache(self, cache: Dict[str, Dict[str, Any]]) -> None:
        if not self._gcs_bucket_name:
            return
        try:
            from google.cloud import storage as gcs_storage
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                tmp_path = tmp.name
            with open(tmp_path, "wb") as f:
                pickle.dump(cache, f)
            client = gcs_storage.Client()
            bucket = client.bucket(self._gcs_bucket_name)
            blob   = bucket.blob(self._gcs_object_name)
            blob.upload_from_filename(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            print(f"[StartConceptPool] uploaded embedding cache ({len(cache)} entries) "
                  f"to gs://{self._gcs_bucket_name}/{self._gcs_object_name}")
        except Exception as exc:
            print(f"[StartConceptPool] embedding cache upload failed "
                  f"(non-fatal, will re-encode next startup): {exc}")