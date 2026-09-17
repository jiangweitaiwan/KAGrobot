"use client";

/**
 * WidgetOverlay.tsx — 讓 SkillWidgetCard 跳出 chat-column 的限制,
 * 疊加在 GlobalBrain + DetailPanel + SemanticPanel 三欄之上。
 *
 * rect 由 page.tsx 用 getBoundingClientRect() 量測三欄聯集算出；
 * rect 為 null(尚未量測完成,或視窗寬度 < 1200px、chat-column 已消失
 * 的斷點)時退回全螢幕置中,確保任何螢幕寬度都能操作表單。
 *
 * 這裡只負責「疊層 + 定位 + 置中卡片」，實際互動邏輯（submit/cancel）
 * 委派給既有的 SkillWidgetCard，不重複實作。
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
        alignItems:      "center",
        justifyContent:  "center",
        padding:         20,
        boxSizing:       "border-box",
        background:      "rgba(3,1,8,0.75)",
        backdropFilter:  "blur(3px)",
      }}
    >
      <SkillWidgetCard
        widget={widget}
        disabled={disabled}
        onSubmit={onSubmit}
        onCancel={onCancel}
        maxWidthPx={OVERLAY_MAX_WIDTH}
      />
    </div>
  );
}