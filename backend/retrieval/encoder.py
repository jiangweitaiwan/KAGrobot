# -*- coding: utf-8 -*-
"""retrieval/encoder.py — SBERT ONNX 編碼器。"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class SbertOnnxEncoder:
    """以 ONNX Runtime 執行 SBERT multilingual 模型的編碼器。"""

    def __init__(self, model_path: str, tokenizer_dir: str) -> None:
        opts = ort.SessionOptions()
        opts.enable_mem_pattern   = False
        opts.enable_cpu_mem_arena = False
        self._session   = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        sentences: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """將文字編碼為向量。回傳 shape (N, dim) 的 float32 陣列。"""
        if isinstance(sentences, str):
            sentences = [sentences]

        ids_list, mask_list = [], []
        for s in sentences:
            ids, mask = self._tokenize(s)
            ids_list.append(ids)
            mask_list.append(mask)

        input_ids      = np.vstack(ids_list).astype(np.int64)
        attention_mask = np.vstack(mask_list).astype(np.int64)

        outputs    = self._session.run(None, {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        })
        embeddings = self._mean_pooling(outputs[0], attention_mask)

        if normalize:
            norms      = np.clip(
                np.linalg.norm(embeddings, axis=1, keepdims=True),
                a_min=1e-9, a_max=None,
            )
            embeddings = embeddings / norms

        return embeddings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize(
        self,
        text: str,
        max_length: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        enc = self._tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return enc["input_ids"], enc["attention_mask"]

    @staticmethod
    def _mean_pooling(
        token_embeddings: np.ndarray,
        attention_mask:   np.ndarray,
    ) -> np.ndarray:
        mask     = attention_mask[..., None].astype(np.float32)
        sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return (token_embeddings * mask).sum(axis=1) / sum_mask
