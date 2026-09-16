# -*- coding: utf-8 -*-
"""tools/skill_widget.py — show_interactive_html 工具（工廠函數）。

流程（見與使用者確認過的設計）：
  1. LLM 判斷需要讓使用者用互動表單輸入參數時，呼叫此工具。
  2. 工具讀取 widget 的 HTML 全文，注入 widget_token / 預填參數，
     上傳到 GCS，拿到 1 小時效期的簽名 URL。
  3. 工具把這筆 {widget_id, title, url, ...} 寫進呼叫端注入的
     turn_widget_state（mutable dict closure，同 sandbox_state /
     attachment_state 的 pattern），並回傳一個「已完成」的字串結果
     （status="displayed_waiting_for_user"）——這是一輪完整、正常的
     tool round-trip，ToolMessage 不會懸空，LangGraph 的
     router_after_tools 會偵測到這個 tool 名稱並讓這輪對話提早結束
     （見 core/main_agent.py）。
  4. 使用者在前端 iframe 內操作、即時試算（走 POST /skill/widget/dispatch，
     不經過這裡也不經過 LLM），最終按下「送出」或「取消」。
  5. 前端把結果組成一段新的使用者文字訊息（例如
     "[widget_submission: widget_id=risk_input]\n{...json...}"）送進
     既有的 /chat，直接沿用 OrchestratorNodes.main_agent() 既有的
     history 延續機制——這裡刻意不重造 ToolMessage 配對，不需要任何
     「resume」專用的 graph node 或狀態欄位。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model
from typing_extensions import Literal

from core.skill_loader import HtmlWidgetSpec
from core.widget_token import issue_token
from tools.gcs_utils import upload_and_sign

from settings import get_settings

def make_show_interactive_html_tool(
    *,
    skill_dir:           Path,
    skill_name:          str,
    widgets:              List[HtmlWidgetSpec],
    bucket_name:          str,
    uid:                  str,
    widget_token_secret:  str,
    widget_token_ttl_s:   int,
    turn_widget_state:    dict,
    frontend_origin:      str = "",
) -> StructuredTool:
    """建立 show_interactive_html 工具。

    Parameters
    ----------
    skill_dir           : 此 skill 本地已解壓的根目錄（RCSkill.skill_dir）。
    skill_name           : 供 description / widget_token payload 顯示用。
    widgets               : RCSkill.html_widgets，此 skill 所有可用 widget 的索引。
    bucket_name           : GCS bucket 名稱（settings.skill_outputs_bucket_name，
                             與 get_sandbox_file 共用同一個 bucket，
                             以 "{uid}/widgets/..." 前綴區隔，不需要另開 bucket）。
    uid                   : 已驗證的使用者 id，寫入 widget_token payload，
                             也用於 GCS 物件路徑命名空間。
    widget_token_secret   : settings.widget_token_secret，未設定時
                             core.widget_token.issue_token 會直接拋出
                             WidgetTokenError（見該模組說明）。
    widget_token_ttl_s    : widget token 與 GCS 簽名 URL 的有效秒數，
                             兩者刻意保持一致，token 過期後 URL 通常也
                             該視為過期（即使簽名 URL 本身可能還沒到期，
                             使用者也不該再送出試算請求）。
    turn_widget_state     : {"item": None}，mutable dict closure，
                             工具呼叫成功後寫入 {"item": {...}}，
                             chat() 結束後讀出來組成回傳值的 pending_widget。
    frontend_origin        : settings.frontend_origin，注入進 widget HTML 的
                             window.__SKILL_PARENT_ORIGIN__，供 widget 內
                             postMessage 時當 targetOrigin 使用。
                             【重要】widget iframe 用
                             sandbox="allow-scripts"（無 allow-same-origin），
                             其 window.location.origin 是 opaque 的字串
                             "null"，widget HTML 不能拿它當 targetOrigin
                             （會導致 postMessage 永遠送不到 parent，因為
                             parent 的真實 origin 絕不會等於 "null"）。
                             必須由後端明確注入真正的前端 origin。
                             未設定時退回 "*"（訊息會送給任何 embed 這個
                             簽名 URL 的頁面接收，生產環境務必設定
                             FRONTEND_ORIGIN 環境變數避免這個風險）。
    """
    widget_ids = tuple(w.widget_id for w in widgets)
    by_id      = {w.widget_id: w for w in widgets}
    parent_origin = frontend_origin.strip() or "*"

    ShowWidgetInput = create_model(
        "show_interactive_html_input",
        widget_id=(
            Literal[widget_ids],
            Field(description="要顯示的互動元件 id，須為下方可用清單裡的其中一個"),
        ),
        prefill_params_json=(
            str,
            Field(default="{}", description="要預填進表單的參數，合法 JSON 字串，不需要則傳 '{}'"),
        ),
    )

    def show_interactive_html(widget_id: str, prefill_params_json: str = "{}") -> str:
        widget = by_id.get(widget_id)
        if widget is None:
            return json.dumps(
                {"ok": False, "error": f"未知 widget_id: {widget_id}"}, ensure_ascii=False
            )

        try:
            html = (skill_dir / widget.rel_path).read_text(encoding="utf-8")
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": f"讀取 widget 檔案失敗：{exc}"}, ensure_ascii=False
            )

        try:
            prefill = json.loads(prefill_params_json) if prefill_params_json else {}
        except json.JSONDecodeError:
            prefill = {}

        try:
            token = issue_token(
                widget_token_secret,
                uid=uid,
                rc_name=skill_name,
                allowed_functions=widget.compute_functions,
                ttl_s=widget_token_ttl_s,
            )
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": f"簽發 widget token 失敗：{exc}"}, ensure_ascii=False
            )
        dispatch_url = f"{get_settings().backend_base}/skill/widget/dispatch"

        injected_html = (
            "<script>\n"
            f"  window.__SKILL_WIDGET_TOKEN__ = {json.dumps(token)};\n"
            f"  window.__SKILL_WIDGET_PARAMS__ = {json.dumps(prefill, ensure_ascii=False)};\n"
            f"  window.__SKILL_PARENT_ORIGIN__ = {json.dumps(parent_origin)};\n"
            f"  window.__SKILL_DISPATCH_URL__ = {json.dumps(dispatch_url)};\n"
            "  (function () {\n"
            "    var __lastH = 0;\n"
            "    function __postHeight() {\n"
            "      var h = Math.ceil(document.documentElement.getBoundingClientRect().height);\n"
            "      if (h === __lastH) return;\n"
            "      __lastH = h;\n"
            "      try {\n"
            "        window.parent.postMessage(\n"
            "          { type: 'skill_widget_resize', height: h },\n"
            "          window.__SKILL_PARENT_ORIGIN__\n"
            "        );\n"
            "      } catch (e) {}\n"
            "    }\n"
            "    if (window.ResizeObserver) {\n"
            "      new ResizeObserver(__postHeight).observe(document.documentElement);\n"
            "    } else {\n"
            "      setInterval(__postHeight, 400);\n"
            "    }\n"
            "    window.addEventListener('load', __postHeight);\n"
            "    setTimeout(__postHeight, 0);\n"
            "  })();\n"
            "</script>\n"
            + html
        )

        object_key = f"{uid}/widgets/{skill_name}_{widget_id}_{int(time.time() * 1000)}.html"
        try:
            url = upload_and_sign(
                bucket_name, object_key, injected_html.encode("utf-8"),
                content_type="text/html; charset=utf-8",
                expiration_hours=max(1, widget_token_ttl_s // 3600) or 1,
            )
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": f"上傳互動元件失敗：{exc}"}, ensure_ascii=False
            )

        turn_widget_state["item"] = {
            "widget_id":         widget_id,
            "title":             widget.title,
            "url":               url,
            "created_at":        int(time.time() * 1000),
            "expires_in_hours":  max(1, widget_token_ttl_s // 3600) or 1,
        }

        return json.dumps({
            "ok":         True,
            "widget_id":  widget_id,
            "status":     "displayed_waiting_for_user",
            "note":       (
                "此互動元件已顯示給使用者，本輪對話到此結束。"
                "不要假設使用者已經提交資料，下一輪對話你會收到使用者的提交結果"
                "（訊息以 [widget_submission: widget_id=...] 開頭）或取消通知。"
            ),
        }, ensure_ascii=False)

    widget_list_desc = "、".join(
        f"'{w.widget_id}'（{w.title}：{w.description}）" for w in widgets
    )

    return StructuredTool.from_function(
        func=show_interactive_html,
        name="show_interactive_html",
        description=(
            f"顯示此 Skill（{skill_name}）提供的互動式表單/計算器，讓使用者親自輸入"
            "參數並即時試算，比起你直接用文字詢問參數更直覺。呼叫後這輪對話會結束，"
            "等待使用者操作完成；下一輪你會收到使用者的提交結果或取消通知。"
            f"可用 widget_id：{widget_list_desc}"
        ),
        args_schema=ShowWidgetInput,
    )
