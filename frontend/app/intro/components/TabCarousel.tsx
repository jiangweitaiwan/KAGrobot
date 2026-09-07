"use client";

/**
 * TabCarousel.tsx — 上下堆疊版。
 *
 * 版面順序(由上到下):
 *   1. 視覺(IntroGlobalBrain) — 全寬長方形,畫面主角
 *   2. 四個小方形分頁鈕 — 置中,緊貼視覺下方
 *   3. 說明文字
 *
 * children 回傳 [visual, copy] 的 tuple,取代前一版依賴 CSS grid
 * 兩欄自動排版的做法——排版邏輯完全收斂在這個元件的 flex-column
 * 結構裡,呼叫端不需要再猜自己會被排進哪一欄。
 *
 * autoplay 與 pause 邏輯仍由 page.tsx 的 useAutoplayTab 負責,這裡
 * 純粹是受控的展示層。
 */

export type IntroTab = { key: string; label: string; accent: string; glyph: string };

type Props = {
  tabs: IntroTab[];
  active: number;
  onSelect: (i: number) => void;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  children: (activeIndex: number) => [React.ReactNode, React.ReactNode];
};

export default function TabCarousel({ tabs, active, onSelect, onMouseEnter, onMouseLeave, children }: Props) {
  const [visual, copy] = children(active);

  return (
    <div className="intro-tab-stack" onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
      <div className="intro-stage-visual">{visual}</div>

      <div className="intro-tab-list" role="tablist" aria-label="人腦層次">
        {tabs.map((t, i) => (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={active === i}
            title={t.label}
            className={`intro-tab-square${active === i ? " active" : ""}`}
            style={active === i ? {
              borderColor: t.accent,
              color: t.accent,
              boxShadow: `0 0 14px ${t.accent}55`,
            } : undefined}
            onClick={() => onSelect(i)}
          >
            {t.glyph}
          </button>
        ))}
      </div>

      <div className="intro-stage-copy">{copy}</div>
    </div>
  );
}
