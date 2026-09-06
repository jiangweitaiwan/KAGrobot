# -*- coding: utf-8 -*-
"""retrieval/rc_category_loader.py — 從 GCS 載入 RC 分類標籤。

資料來源：另一個專案把所有 RC 丟進 BERTopic 分類器後，
輸出 bertopic_rc_categories.json 到 GCS：
    {
      "categories": [
        {"rc_name": "Calvo (1983) Staggered Prices Model",
         "category": "Momentum Investing",
         "topic_id": 1, "color": "#f78250"},
        ...
      ],
      "generated_at": "...",
      "rc_count": 36,
      "topic_count": 2,
      "unmatched_count": 0,
      "config_id": "config"
    }

此處只負責下載、解析 categories、依 rc_name 建索引，不做任何分類邏輯——
分類本身完全交給外部專案離線產生，設計哲學同 routes_brain.py
的 _load_bertopic_layout。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict


class RCCategoryLoader:
    def __init__(self, bucket_name: str, gcs_object: str = "bertopic_rc_categories.json") -> None:
        self._bucket_name = bucket_name
        self._gcs_object = gcs_object

    def load(self) -> Dict[str, Dict[str, Any]]:
        """回傳 { normalized_rc_name: {category, topic_id, color} }。"""
        raw = self._load_raw()
        result: Dict[str, Dict[str, Any]] = {}
        for item in raw.get("categories", []):
            name = (item.get("rc_name") or "").strip()
            if not name:
                continue
            result[self._normalize(name)] = {
                "category": item.get("category", ""),
                "topic_id": item.get("topic_id"),
                "color":    item.get("color"),
            }
        print(f"[RCCategoryLoader] indexed {len(result)} rc categories")
        return result

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()

    def _load_raw(self) -> dict:
        if self._bucket_name:
            try:
                from google.cloud import storage as gcs_storage
                client = gcs_storage.Client()
                bucket = client.bucket(self._bucket_name)
                blob = bucket.blob(self._gcs_object)
                if blob.exists(client):
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                        tmp_path = tmp.name
                    blob.download_to_filename(tmp_path)
                    with open(tmp_path, encoding="utf-8") as f:
                        data = json.load(f)
                    Path(tmp_path).unlink(missing_ok=True)
                    print(f"[RCCategoryLoader] loaded gs://{self._bucket_name}/{self._gcs_object}")
                    return data
            except Exception as exc:
                print(f"[RCCategoryLoader] GCS load failed: {exc}")

        local_path = Path(__file__).resolve().parent.parent.parent / "bertopic_rc_categories.json"
        if local_path.exists():
            with open(local_path, encoding="utf-8") as f:
                return json.load(f)

        print("[RCCategoryLoader] no bertopic_rc_categories.json found, returning empty")
        return {"categories": []}