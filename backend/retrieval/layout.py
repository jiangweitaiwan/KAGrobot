# -*- coding: utf-8 -*-
"""retrieval/layout.py — 全域知識圖譜 3D 佈局建構器。

殼層隱喻（由核心向外）：
  article  → 內核（_R_CORE）
  chapter  → 中層（_R_MID）
  paper    → 外層（_R_CORTEX）
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
    # ── 殼層半徑（由核心向外）─────────────────────────────────
    _R_CORE        = 0.20   # article（最核心）
    _R_MID         = 0.60   # chapter（教科書）
    _R_CORTEX      = 1.00   # paper（論文）
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
        article_pkl_path:        Optional[str] = None,   # ← 新增
        gcs_bucket_name:         Optional[str] = None,
        gcs_object_name:         str           = "results_dict.pkl",
        chapter_gcs_object_name: str           = "chapter_results_dict.pkl",
        article_gcs_object_name: str           = "article_results_dict.pkl",  # ← 新增
    ) -> Dict[str, Any]:
        bucket = (
            gcs_bucket_name
            or getattr(self._settings, "gcs_bucketforresult", "")
            or os.environ.get("GCS_BUCKETFORRESULT", "")
        ).strip()

        local_results_path = self._download_pkl(bucket, gcs_object_name,         pkl_path)
        local_chapter_path = self._download_pkl(bucket, chapter_gcs_object_name, chapter_pkl_path or "")
        local_article_path = self._download_pkl(bucket, article_gcs_object_name, article_pkl_path or "")  # ← 新增

        # 三份 pkl 格式已統一，共用同一個 loader
        paper_concepts   = self._load_pkl(local_results_path,  source="paper")
        chapter_concepts = self._load_pkl(local_chapter_path,  source="chapter")
        article_concepts = self._load_pkl(local_article_path,  source="article")

        # merge：article > chapter > paper（核心優先）
        all_concepts = self._merge_concepts(
            article_concepts, chapter_concepts, paper_concepts
        )

        print(
            f"[GlobalLayoutBuilder] concepts: "
            f"{len(article_concepts)} article, "
            f"{len(chapter_concepts)} chapter, "
            f"{len(paper_concepts)} paper, "
            f"{len(all_concepts)} merged"
        )

        name_to_direction, name_to_source = self._pca_to_3d(all_concepts)
        nodes_out, edges_out = self._build_graph(name_to_direction, name_to_source)

        return {"nodes": nodes_out, "edges": edges_out}

    # ------------------------------------------------------------------
    # Merge（article > chapter > paper）
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_concepts(
        article:  List[Dict[str, Any]],
        chapter:  List[Dict[str, Any]],
        paper:    List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """依優先級合併，同名 concept 以高優先級來源為準。"""
        merged: Dict[str, Dict[str, Any]] = {}

        # 低優先到高優先依序寫入，後寫覆蓋前寫
        for concept in paper + chapter + article:
            merged[concept["name"]] = concept

        return list(merged.values())

    # ------------------------------------------------------------------
    # 統一 pkl loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pkl(path: str, source: str) -> List[Dict[str, Any]]:
        """通用 pkl loader，適用 article / chapter / paper 三種格式。

        每個 item 格式：
        {
            "index": int,
            "concept": str,
            "embeddings": {
                "definition": np.ndarray (384,),
                "mechanism":  np.ndarray (384,),
            }
        }

        embedding 取 definition + mechanism 的平均。
        若某個 key 不存在或 shape 不符，只用存在的那個。
        兩個都不存在則跳過。
        """
        if not path or not os.path.exists(path):
            print(f"[GlobalLayoutBuilder] pkl not found, skipping: {path}")
            return []

        try:
            with open(path, "rb") as f:
                raw = pickle.load(f)
        except Exception as exc:
            print(f"[GlobalLayoutBuilder] failed to load pkl '{path}': {exc}")
            return []

        by_name: Dict[str, List[np.ndarray]] = defaultdict(list)

        for _key, items in raw.items():
            for item in items:
                cname = item.get("concept", "").strip()
                if not cname:
                    continue

                emb_dict = item.get("embeddings", {})

                valid_embs: List[np.ndarray] = []
                for key in ("definition", "mechanism"):
                    emb = emb_dict.get(key)
                    if emb is None:
                        continue
                    arr = np.array(emb)
                    if arr.shape == (384,):
                        valid_embs.append(arr)

                if not valid_embs:
                    continue

                # definition + mechanism 平均
                avg_emb = np.mean(valid_embs, axis=0)
                by_name[cname].append(avg_emb)

        # 同名 concept 跨 key 再做一次平均
        return [
            {
                "name":      name,
                "source":    source,
                "embedding": np.mean(embs, axis=0),
            }
            for name, embs in by_name.items()
        ]

    # ------------------------------------------------------------------
    # GCS download（不變）
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

    # ------------------------------------------------------------------
    # PCA（不變）
    # ------------------------------------------------------------------

    @staticmethod
    def _pca_to_3d(
        concepts: List[Dict[str, Any]],
    ) -> tuple[Dict[str, tuple], Dict[str, str]]:
        embs_matrix = np.array([c["embedding"] for c in concepts])
        try:
            pca       = PCA(n_components=3, random_state=42)
            coords_3d = pca.fit_transform(embs_matrix)
        except Exception:
            centered  = embs_matrix - embs_matrix.mean(axis=0)
            _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
            coords_3d = centered @ Vt[:3].T

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
    # 殼層半徑（新增 article）
    # ------------------------------------------------------------------

    @classmethod
    def _shell_radius(cls, source: str, rng: np.random.Generator) -> float:
        base_map = {
            "article": cls._R_CORE,
            "chapter": cls._R_MID,
            "paper":   cls._R_CORTEX,
        }
        base   = base_map.get(source, cls._R_CORTEX)
        jitter = 1.0 + rng.uniform(-cls._RADIUS_JITTER, cls._RADIUS_JITTER)
        return base * jitter

    @staticmethod
    def _random_unit_direction(rng: np.random.Generator) -> tuple:
        raw      = rng.normal(size=3)
        raw_norm = float(np.linalg.norm(raw))
        if raw_norm <= 1e-9:
            return (1.0, 0.0, 0.0)
        return tuple((raw / raw_norm).tolist())

    # ------------------------------------------------------------------
    # Neo4j 對齊（不變，_shell_radius 已支援三種 source）
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
                MATCH (a:Concept)-[:COMPOSES]->(b:Concept)
                RETURN elementId(a) AS src, elementId(b) AS tgt
                """
            ).data()

            paper_edges = session.run(
                """
                MATCH (a:Concept)-[:INSPIRES|INDIRECT_INSPIRES|COMPOSES]->
                      (:Paper)-[:PRODUCES]->(b:Concept)
                RETURN elementId(a) AS src, elementId(b) AS tgt
                """
            ).data()

        eid_set:      set        = set()
        matched_eids: set        = set()
        nodes_out:    List[dict] = []

        for row in concept_rows:
            eid   = row["eid"]
            name  = row["name"] or ""
            ctype = row["type"] or "other"
            eid_set.add(eid)

            rng = np.random.default_rng(abs(hash(eid)) % (2 ** 32))

            direction = name_to_direction.get(name)
            source    = name_to_source.get(name, "paper")

            if direction:
                matched_eids.add(eid)
            else:
                source    = "paper"
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