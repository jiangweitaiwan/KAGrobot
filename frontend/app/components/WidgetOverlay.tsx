"use client";

/**
 * WidgetOverlay.tsx — 讓 SkillWidgetCard 跳出 chat-column 的限制,
 * 疊加在 GlobalBrain + DetailPanel + SemanticPanel 三欄之上。
 *
 * rect 由 page.tsx 用 getBoundingClientRect() 量測三欄聯集算出；
 * rect 為 null(尚未量測完成,或視窗寬度 < 1200px、chat-column 已消失
 * 的斷點)時退回全螢幕置中,確保任何螢幕寬度都能操作表單。
 *
 * ── 可捲動置中（修正）─────────────────────────────────────────
 *   外層容器原本用 alignItems/justifyContent: center 置中卡片，
 *   但當卡片（尤其是內容較高的 widget）比容器還高時，flexbox 用
 *   align-items: center 置中會讓內容「往上下兩側溢出」且使用者
 *   捲不到超出的部分（經典的 flexbug #9：置中的 flex item 若比
 *   容器大，捲動軸抵達不了被裁掉的那一側）。
 *
 *   改法：外層容器保留 overflow-y: auto，但拿掉 align/justify
 *   center，改成內層包一個 margin: auto 的 wrapper 來置中——這個
 *   寫法在內容小於容器時效果等同置中，內容大於容器時則會從最上方
 *   開始正常往下捲動，兩種情況都不會有構不到的區域。
 */

import { SkillWidgetCard, type PendingWidget } from "./SkillWidgetCard";

export type OverlayRect = { top: number; left: number; width: number; height: number };

type Props = {
  rect:     OverlayRect | null;
  widget:   PendingWidget;
  disabled: boolean;
  onSubmit: (widgetId: string, data: unknown) => void;
  onCancel: (widgetId: string) => void;
};

const OVERLAY_MAX_WIDTH = 720;

export default function WidgetOverlay({ rect, widget, disabled, onSubmit, onCancel }: Props) {
  const positionStyle: React.CSSProperties = rect
    ? { top: rect.top, left: rect.left, width: rect.width, height: rect.height }
    : { inset: 0 };

  return (
    <div
      style={{
        position:       "fixed",
        ...positionStyle,
        zIndex:          900,
        display:         "flex",
        padding:         20,
        boxSizing:       "border-box",
        background:      "rgba(3,1,8,0.75)",
        backdropFilter:  "blur(3px)",
        overflowY:       "auto",   // ← 容器本身可捲動，避免內容被裁掉又構不到
      }}
    >
      {/* margin: auto 置中寫法：內容小於容器時置中效果與原本相同，
         內容大於容器時可正常從頂部往下捲動（不會有捲不到的區塊）。 */}
      <div style={{ margin: "auto", maxWidth: "100%" }}>
        <SkillWidgetCard
          widget={widget}
          disabled={disabled}
          onSubmit={onSubmit}
          onCancel={onCancel}
          maxWidthPx={OVERLAY_MAX_WIDTH}
        />
      </div>
    </div>
  );
}