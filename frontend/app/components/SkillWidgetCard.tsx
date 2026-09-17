"use client";

/**
 * SkillWidgetCard.tsx — 顯示 show_interactive_html 產出的互動元件。
 *
 * ...(原本註解不變)...
 *
 * ── 動態高度上限（新增）───────────────────────────────────────
 *   MAX_HEIGHT 原本是寫死的 1400px，在較矮的視窗（筆電、縮小視窗）
 *   下常常還是比可視高度高，導致 widget 下半部（甚至「取消」按鈕）
 *   被裁到畫面外、且外層沒有捲軸可以碰到。
 *
 *   現在改成 useViewportMaxHeight()：以 window.innerHeight 扣掉一段
 *   預留給版面 chrome（header/footer/外距）的空間，隨視窗 resize 動態
 *   重新計算，並拿這個值同時 clamp「iframe 回報的高度」與「卡片本身
 *   的 maxHeight」。卡片內部的 iframe 容器另外套用 overflow-y: auto
 *   作為保底——即使極端情況下內容仍然比算出來的上限高，也是卡片自己
 *   出現內部捲軸，而不是整塊被切掉、構不到按鈕。
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

const MIN_HEIGHT     = 120;
const DEFAULT_HEIGHT = 320;

// 卡片本身扣掉 header（約 30px）+ footer（約 40px）+ 上下邊框，
// 供 useViewportMaxHeight 算「iframe 還能用多少高度」時扣除。
const CARD_CHROME_HEIGHT = 80;

// 視窗留給 overlay 外距、避免卡片整個貼到螢幕邊緣的安全邊距
// （對應 WidgetOverlay 的 padding: 20 上下各一份 + 一點緩衝）。
const VIEWPORT_MARGIN = 120;

/** 依目前視窗高度動態算出「整張卡片」可以用的最大高度，
 *  隨 resize 即時更新。SSR 階段 window 不存在，先給一個保守預設值，
 *  掛載後的第一個 effect 會立刻校正成真實值。 */
function useViewportMaxHeight(margin: number): number {
  const [maxH, setMaxH] = useState<number>(() =>
    typeof window !== "undefined"
      ? Math.max(window.innerHeight - margin, MIN_HEIGHT + CARD_CHROME_HEIGHT)
      : 800,
  );

  useEffect(() => {
    function update() {
      setMaxH(Math.max(window.innerHeight - margin, MIN_HEIGHT + CARD_CHROME_HEIGHT));
    }
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [margin]);

  return maxH;
}

export function SkillWidgetCard({ widget, disabled, onSubmit, onCancel, maxWidthPx }: Props) {
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const expired   = isAttachmentExpired(widget as unknown as Attachment);

  const [height, setHeight] = useState(DEFAULT_HEIGHT);

  // 整張卡片能用的最大高度（隨視窗 resize 動態更新）
  const cardMaxHeight = useViewportMaxHeight(VIEWPORT_MARGIN);
  // iframe 本身能用的最大高度 = 卡片上限扣掉 header/footer
  const iframeMaxHeight = Math.max(cardMaxHeight - CARD_CHROME_HEIGHT, MIN_HEIGHT);

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
          setHeight(Math.min(Math.max(h, MIN_HEIGHT), iframeMaxHeight));
        }
      }
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [widget.widget_id, onSubmit, iframeMaxHeight]);

  // 每次換一個新 widget（新的 url）時，高度重置回預設值，
  // 避免沿用上一個 widget 殘留的高度造成畫面跳動或空白。
  useEffect(() => {
    setHeight(Math.min(DEFAULT_HEIGHT, iframeMaxHeight));
  }, [widget.url, iframeMaxHeight]);

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
      border:        "1px solid rgba(83,74,183,0.35)",
      borderRadius:  12,
      overflow:      "hidden",
      margin:        "6px 0",
      background:    "rgba(13,18,32,0.5)",
      maxWidth:      maxWidthPx ?? 480,
      maxHeight:     cardMaxHeight,
      display:       "flex",
      flexDirection: "column",
    }}>
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "6px 10px", fontSize: "0.72rem", color: "var(--purple)",
        borderBottom: "1px solid rgba(83,74,183,0.2)",
        flexShrink: 0,
      }}>
        <span>🧾 {widget.title}</span>
      </div>

      {/* iframe 容器：獨立的可捲動區域。正常情況下 iframe 高度已經被
         iframeMaxHeight clamp 過、不會超出，這層 overflow-y 是保底，
         避免任何極端狀況（resize 事件延遲、內容異常）讓按鈕構不到。 */}
      <div style={{ overflowY: "auto", minHeight: 0 }}>
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
      </div>

      <div style={{
        display: "flex", justifyContent: "flex-end", padding: "6px 10px",
        flexShrink: 0, borderTop: "1px solid rgba(83,74,183,0.15)",
      }}>
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