# -*- coding: utf-8 -*-
"""retrieval/layout.py — 全域知識圖譜 3D 佈局建構器。

GlobalLayoutBuilder 負責：
  1. 從 GCS 或本地讀取 results_dict / chapter_results_dict pkl
  2. 合併兩份資料並做 PCA，取得每個概念的語意方向
  3. 依「大腦殼層」隱喻把方向映射成 3D 座標：
     chapter（textbook，基礎概念）→ 內殼／邊緣系統
     results（paper，專精概念）   → 外殼／皮質
     殼層內的角度位置則由 embedding 方向決定
  4. 與 Neo4j Concept 節點對齊，輸出前端渲染用的 nodes / edges 結構

快取由呼叫端（app.state.layout_cache）管理，本類別本身不做快取。
"""

from __future__ import annotations

import os
import pickle
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from google.cloud import storage as gcs_storage
from sklearn.decomposition import PCA

from retrieval.encoder import SbertOnnxEncoder


class GlobalLayoutBuilder:
    """建構全域知識圖譜的 3D 節點座標與邊結構。

    佈局採用「大腦殼層」隱喻：
      - chapter（textbook 來源，基礎概念）落在內殼，半徑較小，
        對應邊緣系統 / 基礎概念。
      - results（paper 來源，專精概念）落在外殼，半徑較大，
        對應皮質 / 專精概念。
      - 兩個殼層共用同一組 PCA 基底，只取「方向」不取「大小」，
        所以語意相近的 chapter / results 概念即使半徑不同，
        角度方向也會接近──讓推理路徑呈現「由內殼往外殼生長」
        的視覺效果。

    Parameters
    ----------
    driver  : Neo4j driver
    encoder : SbertOnnxEncoder（保留，供未來擴充使用）
    settings: Settings 物件（用於讀取 GCS bucket 名稱等設定）
    """

    # ── 殼層半徑（大腦隱喻）──────────────────────────────────
    # chapter → 內殼（邊緣系統，基礎概念）
    # results → 外殼（皮質，專精概念）
    # 二者皆為離散值，不做連續漸層；_RADIUS_JITTER 讓每個節點的
    # 半徑有 ±10% 隨機抖動，殼層因此呈現「帶狀區域」而非數學上
    # 完美的球面。
    _R_CORE        = 0.35
    _R_CORTEX      = 1.00
    _RADIUS_JITTER = 0.10

    def __init__(self, driver, encoder: SbertOnnxEncoder, settings) -> None:
        self._driver   = driver
        self._encoder  = encoder
        self._settings = settings

    def build(
        self,
        pkl_path:                str,
        *,
        chapter_pkl_path:        Optional[str] = None,
        gcs_bucket_name:         Optional[str] = None,
        gcs_object_name:         str           = "results_dict.pkl",
        chapter_gcs_object_name: str           = "chapter_results_dict.pkl",
    ) -> Dict[str, Any]:
        """建構並回傳 {nodes, edges} 結構。

        快取邏輯由呼叫端負責，本方法每次都重新計算。
        """
        bucket = (
            gcs_bucket_name
            or getattr(self._settings, "gcs_bucketforresult", "")
            or os.environ.get("GCS_BUCKETFORRESULT", "")
        ).strip()

        # ── 下載 pkl ──────────────────────────────────────────
        local_results_path = self._download_pkl(bucket, gcs_object_name, pkl_path)
        local_chapter_path = self._download_pkl(
            bucket, chapter_gcs_object_name, chapter_pkl_path or "",
        )

        # ── 解析兩份 pkl ──────────────────────────────────────
        results_concepts = self._load_results_pkl(local_results_path)
        chapter_concepts = self._load_chapter_pkl(local_chapter_path)

        # results 優先，chapter 補充不重疊的
        results_names = {c["name"] for c in results_concepts}
        all_concepts  = results_concepts + [
            c for c in chapter_concepts if c["name"] not in results_names
        ]
        print(
            f"[GlobalLayoutBuilder] concepts: {len(results_concepts)} results, "
            f"{len(chapter_concepts)} chapter, "
            f"{len(all_concepts)} merged"
        )

        # ── PCA 取語意方向（共用基底，只取方向不取大小）───────
        name_to_direction, name_to_source = self._pca_to_3d(all_concepts)

        # ── 查 Neo4j，對齊 eid，套上殼層半徑 ───────────────────
        nodes_out, edges_out = self._build_graph(name_to_direction, name_to_source)

        return {"nodes": nodes_out, "edges": edges_out}

    # ------------------------------------------------------------------
    # pkl 載入
    # ------------------------------------------------------------------

    @staticmethod
    def _download_pkl(bucket: str, gcs_obj: str, fallback_local: str) -> str:
        if not bucket:
            return fallback_local
        try:
            client = gcs_storage.Client()
            b      = client.bucket(bucket)
            blob   = b.blob(gcs_obj)
            if not blob.exists(client):
                raise FileNotFoundError(
                    f"GCS object not found: gs://{bucket}/{gcs_obj}"
                )
            with tempfile.NamedTemporaryFile(
                prefix=f"pkl_{gcs_obj.replace('/', '_')}_",
                suffix=".pkl",
                delete=False,
            ) as tmp:
                tmp_path = tmp.name
            blob.download_to_filename(tmp_path)
            return tmp_path
        except Exception as exc:
            print(
                f"[GlobalLayoutBuilder] GCS load failed for {gcs_obj}, "
                f"fallback to '{fallback_local}': {exc}"
            )
            return fallback_local

    @staticmethod
    def _load_results_pkl(path: str) -> List[Dict[str, Any]]:
        with open(path, "rb") as f:
            raw = pickle.load(f)

        by_name: Dict[str, List[np.ndarray]] = defaultdict(list)
        for _paper, concepts in raw.items():
            for c in concepts:
                emb_dict = c["alignment"]["embeddings"]
                all_embs: List[np.ndarray] = []
                for emb_list in emb_dict.values():
                    for emb in emb_list:
                        arr = np.array(emb)
                        if arr.shape == (384,):
                            all_embs.append(arr)
                if all_embs:
                    by_name[c["cluster_name"]].append(np.mean(all_embs, axis=0))

        return [
            {"name": name, "source": "results", "embedding": np.mean(embs, axis=0)}
            for name, embs in by_name.items()
        ]

    @staticmethod
    def _load_chapter_pkl(path: str) -> List[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            print("[GlobalLayoutBuilder] chapter pkl not found, skipping.")
            return []
        try:
            with open(path, "rb") as f:
                raw = pickle.load(f)
            by_name: Dict[str, np.ndarray] = {}
            for _chap, items in raw.items():
                for item in items:
                    cname = item["concept"]
                    emb   = np.array(item["embeddings"])
                    if emb.shape == (384,):
                        if cname not in by_name:
                            by_name[cname] = emb
                        else:
                            by_name[cname] = (by_name[cname] + emb) / 2
            return [
                {"name": name, "source": "chapter", "embedding": emb}
                for name, emb in by_name.items()
            ]
        except Exception as exc:
            print(f"[GlobalLayoutBuilder] failed to load chapter pkl: {exc}")
            return []

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    @staticmethod
    def _pca_to_3d(
        concepts: List[Dict[str, Any]],
    ) -> tuple[Dict[str, tuple], Dict[str, str]]:
        """PCA 降到 3D，並把每個概念正規化成單位方向向量。

        chapter 跟 results 的 embedding 合併在一起做同一次 PCA
        （共用基底），藉此保留跨來源的語意一致性：即使一個概念落在
        內殼、另一個落在外殼，只要語意相近，PCA 算出來的方向就會
        接近。這裡刻意不做舊版的逐軸 max-abs 正規化（那樣保留的是
        座標「大小」），而是把整個 3D 向量正規化成單位長度，只保留
        「方向」──大小（也就是殼層半徑）改由 _build_graph() 依
        source 指定。
        """
        embs_matrix = np.array([c["embedding"] for c in concepts])
        try:
            pca       = PCA(n_components=3, random_state=42)
            coords_3d = pca.fit_transform(embs_matrix)
        except Exception:
            centered  = embs_matrix - embs_matrix.mean(axis=0)
            _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
            coords_3d = centered @ Vt[:3].T

        # 整體向量單位化（取代逐軸 max-abs 正規化）：只留方向，捨棄大小。
        norms      = np.linalg.norm(coords_3d, axis=1, keepdims=True)
        norms      = np.clip(norms, a_min=1e-9, a_max=None)
        directions = coords_3d / norms

        name_to_direction = {
            c["name"]: (
                float(directions[idx, 0]),
                float(directions[idx, 1]),
                float(directions[idx, 2]),
            )
            for idx, c in enumerate(concepts)
        }
        name_to_source = {c["name"]: c["source"] for c in concepts}
        return name_to_direction, name_to_source

    # ------------------------------------------------------------------
    # 殼層半徑
    # ------------------------------------------------------------------

    @classmethod
    def _shell_radius(cls, source: str, rng: np.random.Generator) -> float:
        """依 source 決定殼層半徑：chapter → 內殼，results → 外殼。

        加一點隨機抖動（±_RADIUS_JITTER），讓殼層呈現「帶狀區域」
        而不是數學上完美的球面。
        """
        base   = cls._R_CORE if source == "chapter" else cls._R_CORTEX
        jitter = 1.0 + rng.uniform(-cls._RADIUS_JITTER, cls._RADIUS_JITTER)
        return base * jitter

    @staticmethod
    def _random_unit_direction(rng: np.random.Generator) -> tuple:
        """在球面上均勻取一個隨機單位方向（供未匹配到 embedding 的節點使用）。

        用標準常態分布取樣再正規化，而不是直接在球座標角度上均勻
        取樣，避免在極點附近過度集中。
        """
        raw      = rng.normal(size=3)
        raw_norm = float(np.linalg.norm(raw))
        if raw_norm <= 1e-9:
            return (1.0, 0.0, 0.0)
        return tuple((raw / raw_norm).tolist())

    # ------------------------------------------------------------------
    # Neo4j 對齊
    # ------------------------------------------------------------------

    def _build_graph(
        self,
        name_to_direction: Dict[str, tuple],
        name_to_source:    Dict[str, str],
    ) -> tuple[List[dict], List[dict]]:
        with self._driver.session() as session:
            concept_rows = session.run(
                """
                MATCH (c:Concept)
                RETURN elementId(c) AS eid,
                       c.name       AS name,
                       c.type       AS type
                """
            ).data()

            direct_edges = session.run(
                """
                MATCH (a:Concept)-[:REASONING]->(b:Concept)
                RETURN elementId(a) AS src, elementId(b) AS tgt
                """
            ).data()

            paper_edges = session.run(
                """
                MATCH (a:Concept)-[:INSPIRES|INDIRECT_INSPIRES|REASONING]->
                      (:Paper)-[:PRODUCES]->(b:Concept)
                RETURN elementId(a) AS src, elementId(b) AS tgt
                """
            ).data()

        eid_set:      set         = set()
        matched_eids: set         = set()
        nodes_out:    List[dict]  = []

        for row in concept_rows:
            eid   = row["eid"]
            name  = row["name"] or ""
            ctype = row["type"] or "other"
            eid_set.add(eid)

            # 每個節點用 eid 的 hash 當種子，確保抖動／隨機方向是
            # 確定性的（同一個節點每次重建佈局都會落在同一個位置）。
            rng = np.random.default_rng(abs(hash(eid)) % (2 ** 32))

            direction = name_to_direction.get(name)
            source    = name_to_source.get(name, "results")

            if direction:
                matched_eids.add(eid)
            else:
                # 未匹配到 embedding：沒有方向資訊，延用「預設 results」
                # 的邏輯（放到外殼／皮質），方向則在球面上隨機取一個。
                source    = "results"
                direction = self._random_unit_direction(rng)

            radius = self._shell_radius(source, rng)
            nodes_out.append({
                "id":     eid,
                "name":   name,
                "type":   ctype,
                "source": source,
                "x":      direction[0] * radius,
                "y":      direction[1] * radius,
                "z":      direction[2] * radius,
            })

        print(
            f"[GlobalLayoutBuilder] Neo4j concepts: {len(concept_rows)}, "
            f"matched: {len(matched_eids)}, "
            f"unmatched: {len(concept_rows) - len(matched_eids)}"
        )

        seen_edges: set        = set()
        edges_out:  List[dict] = []
        for e in direct_edges + paper_edges:
            src, tgt = e["src"], e["tgt"]
            if src not in eid_set or tgt not in eid_set:
                continue
            key = (min(src, tgt), max(src, tgt))
            if key not in seen_edges:
                seen_edges.add(key)
                edges_out.append({"source": src, "target": tgt})

        return nodes_out, edges_out
