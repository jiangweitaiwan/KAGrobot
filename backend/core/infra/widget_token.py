# -*- coding: utf-8 -*-
"""core/widget_token.py — HTML Widget 的無狀態（stateless）簽章 token。

設計取捨（已與使用者確認）：
  - 不存 Firestore：widget token 生命週期短（預設 1 小時），
    dispatch endpoint（POST /skill/widget/dispatch）在互動期間可能被
    高頻呼叫（例如拖曳 slider 即時試算），每次都打一次 Firestore
    不划算。改用 HMAC 簽章的 stateless token：伺服器端一把密鑰簽發，
    payload 直接編碼 uid / rc_name / 允許呼叫的 compute functions /
    過期時間；dispatch 時只需驗簽章與時效，不需要查資料庫。
  - 代價：無法主動撤銷單一 token（例如使用者帳號被刪除的當下，
    尚未過期的 token 理論上仍可用到自然過期為止）。因為 TTL 很短
    （預設 3600 秒），且 payload 內的 allowed_functions 是 skill 作者
    在 manifest.yaml 宣告的白名單子集，不是任意程式碼，這個風險
    在目前的產品階段可接受。若之後有「立即撤銷」的硬需求，
    再改回 Firestore 版本（介面簽名不需要變，呼叫端不受影響）。

密鑰來源：settings.widget_token_secret（環境變數 WIDGET_TOKEN_SECRET）。
部署時務必設定為真隨機值；本機開發若未設定，issue_token 會拒絕簽發
（見 _require_secret），避免用空字串當密鑰產生「形同虛設」的簽章。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

_DEFAULT_TTL_S = 3600


class WidgetTokenError(ValueError):
    """密鑰未設定或格式錯誤時拋出，供呼叫端轉譯成 4xx。"""


def _require_secret(secret: str) -> None:
    if not secret or not secret.strip():
        raise WidgetTokenError(
            "WIDGET_TOKEN_SECRET 未設定，拒絕簽發 widget token。"
            "請在部署環境設定一個真隨機值（例如 openssl rand -hex 32）。"
        )


def issue_token(
    secret: str,
    *,
    uid:               str,
    rc_name:           str,
    allowed_functions: Dict[str, str],
    ttl_s:             int = _DEFAULT_TTL_S,
) -> str:
    """簽發一個 widget token。

    Parameters
    ----------
    allowed_functions : {action_name: "module_path:func_name"}，
                         來自 HtmlWidgetSpec.compute_functions，
                         是這個 widget 被允許呼叫的 compute function 白名單。
    ttl_s             : token 有效秒數，預設 1 小時。
    """
    _require_secret(secret)

    payload: Dict[str, Any] = {
        "uid":     uid,
        "rc_name": rc_name,
        "fns":     allowed_functions,
        "exp":     int(time.time()) + ttl_s,
    }
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    sig = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def verify_token(secret: str, token: str) -> Optional[Dict[str, Any]]:
    """驗證 token，成功回傳 payload dict，失敗（簽章不符/過期/格式錯誤）回傳 None。

    絕不拋出例外——呼叫端只需要判斷 None 就回 401，不需要 try/except。
    """
    if not secret or not token or "." not in token:
        return None

    try:
        body, sig = token.rsplit(".", 1)
        expected = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None

        payload = json.loads(base64.urlsafe_b64decode(body.encode("ascii")).decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None
