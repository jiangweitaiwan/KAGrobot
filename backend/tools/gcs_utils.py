# -*- coding: utf-8 -*-
"""tools/gcs_utils.py — 共用的 GCS 上傳 + v4 簽名 URL helper。

原本 `_upload_and_sign` / `_generate_signed_url` 寫在 tools/skill_code.py
內、只服務 get_sandbox_file 工具。現在 show_interactive_html 工具
（tools/skill_widget.py）也需要同一套「上傳 bytes → 拿簽名 URL」的邏輯，
差別只在 content_type 與過期時間，因此抽成這裡的共用函式，兩邊都改成
呼叫這裡，避免簽名邏輯分岔維護。

行為與原本完全相同：lazy import google.cloud.storage，絕不在模組載入
時就要求 GCP 憑證存在。
"""

from __future__ import annotations

from datetime import timedelta


def upload_and_sign(
    bucket_name:      str,
    object_key:       str,
    data:             bytes,
    *,
    content_type:     str = "application/octet-stream",
    expiration_hours: int = 24,
) -> str:
    """把 bytes 上傳到指定 GCS bucket/object_key，回傳 v4 簽名 URL。"""
    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(object_key)
    blob.upload_from_string(data, content_type=content_type)

    return generate_signed_url(blob, expiration=timedelta(hours=expiration_hours))


def generate_signed_url(blob, expiration: timedelta) -> str:
    """產生 v4 簽名 URL。

    重要 GCP 細節：Cloud Run 上的 Application Default Credentials
    沒有私鑰，無法在本地直接簽名，必須透過 IAM Credentials API 的
    signBlob 遠端簽名（google-cloud-storage 會自動處理，但需要
    Cloud Run 的服務帳號被授予對「自己」的
    roles/iam.serviceAccountTokenCreator，否則會拋出 403）。
    """
    import google.auth
    from google.auth.transport.requests import Request

    credentials, _ = google.auth.default()
    if not credentials.valid:
        credentials.refresh(Request())

    return blob.generate_signed_url(
        version="v4",
        expiration=expiration,
        method="GET",
        service_account_email=credentials.service_account_email,
        access_token=credentials.token,
    )
