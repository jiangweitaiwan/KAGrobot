"use client";

/**
 * SkillWidgetCard.tsx — 顯示 show_interactive_html 產出的互動元件。
 *
 * 設計要點（與後端 tools/skill_widget.py 對應）：
 *   - iframe 一律 `sandbox="allow-scripts"`，刻意不給 `allow-same-origin`：
 *     widget HTML 是不受信任內容（由 skill 作者提供、注入了 widget_token），
 *     不能讓它拿到 parent 頁面的 cookie / localStorage / Firebase token。
 *   - 即時試算（拖曳 slider 等）完全在 iframe 內部透過
 *     POST /skill/widget/dispatch 完成，不經過這個元件、也不經過 LLM。
 *   - 最終「送出」由 iframe 內的 JS 呼叫
 *     `window.parent.postMessage({ type: 'skill_widget_submit', data }, origin)`
 *     觸發，這裡監聽 message 事件並比對 event.source，避免任意頁面
 *     偽造訊息。
 *   - 「取消」按鈕：使用者可以不填完表單就放棄，送出
 *     `{ cancelled: true }`，讓對話可以繼續而不是卡住（比照
 *     walk_awaiting_answer 鎖住輸入框、但至少留一個退路）。
 *   - 過期判斷比照 AttachmentCard.isAttachmentExpired 的邏輯（同一份
 *     created_at + expires_in_hours 結構），過期後顯示灰階提示，
 *     不再渲染 iframe（後端簽名 URL 本身通常也已經失效）。
 *   - 高度：後端在 show_interactive_html 注入的 <script> 裡統一加了
 *     ResizeObserver + postMessage（見 tools/skill_widget.py），每個
 *     widget 不需要作者自己實作就會回報 `{ type: 'skill_widget_resize',
 *     height }`。這裡監聽同一個 message 事件，動態調整 iframe 高度，
 *     取代原本寫死的 minHeight。widget 是不受信任內容，回報的高度做
 *     上下限 clamp，避免異常/惡意值撐爆聊天版面。
 */

import { useEffect, useRef, useState } from "react";
import { isAttachmentExpired, type Attachment } from "./AttachmentCard";

export type PendingWidget = {
  widget_id:        string;
  title:            string;
  url:              string;
  created_at:       number;   // unix ms
  expires_in_hours: number;
};

type Props = {
  widget:   PendingWidget;
  disabled: boolean;                                   // 已提交/取消後鎖住，避免重複觸發
  onSubmit: (widgetId: string, data: unknown) => void;
  onCancel: (widgetId: string) => void;
  maxWidthPx?: number;
};

const MIN_HEIGHT = 120;
const MAX_HEIGHT = 1400;
const DEFAULT_HEIGHT = 320;

export function SkillWidgetCard({ widget, disabled, onSubmit, onCancel, maxWidthPx }: Props) {
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const expired   = isAttachmentExpired(widget as unknown as Attachment);

  const [height, setHeight] = useState(DEFAULT_HEIGHT);

  useEffect(() => {
    function onMessage(e: MessageEvent) {
      if (e.source !== iframeRef.current?.contentWindow) return;

      if (e.data?.type === "skill_widget_submit") {
        onSubmit(widget.widget_id, e.data.data);
        return;
      }

      if (e.data?.type === "skill_widget_resize") {
        const h = Number(e.data.height);
        if (Number.isFinite(h) && h > 0) {
          setHeight(Math.min(Math.max(h, MIN_HEIGHT), MAX_HEIGHT));
        }
      }
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [widget.widget_id, onSubmit]);

  // 每次換一個新 widget（新的 url）時，高度重置回預設值，
  // 避免沿用上一個 widget 殘留的高度造成畫面跳動或空白。
  useEffect(() => {
    setHeight(DEFAULT_HEIGHT);
  }, [widget.url]);

  if (expired) {
    return (
      <div style={{
        border: "1px solid rgba(232,238,248,0.15)", borderRadius: 10,
        padding: "10px 12px", color: "var(--muted)", fontSize: "0.75rem",
        opacity: 0.6, margin: "6px 0",
      }}>
        「{widget.title}」表單已過期，若仍需要，請重新詢問。
      </div>
    );
  }

  return (
    <div style={{
      border: "1px solid rgba(83,74,183,0.35)", borderRadius: 12,
      overflow: "hidden", margin: "6px 0", background: "rgba(13,18,32,0.5)",
      maxWidth: maxWidthPx ?? 480,
    }}>
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "6px 10px", fontSize: "0.72rem", color: "var(--purple)",
        borderBottom: "1px solid rgba(83,74,183,0.2)",
      }}>
        <span>🧾 {widget.title}</span>
      </div>

      <iframe
        ref={iframeRef}
        src={widget.url}
        sandbox="allow-scripts"
        title={widget.title}
        style={{
          width: "100%",
          height,
          border: "none",
          display: "block",
          transition: "height 0.15s ease",
        }}
      />

      <div style={{ display: "flex", justifyContent: "flex-end", padding: "6px 10px" }}>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onCancel(widget.widget_id)}
          style={{
            fontSize: "0.68rem", padding: "4px 12px", borderRadius: 8,
            border: "1px solid rgba(232,112,48,0.4)", background: "transparent",
            color: "var(--orange)", cursor: disabled ? "default" : "pointer",
            opacity: disabled ? 0.5 : 1,
          }}
        >
          取消
        </button>
      </div>
    </div>
  );
}