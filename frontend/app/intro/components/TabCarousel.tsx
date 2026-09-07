"use client";

/**
 * TabCarousel.tsx — 「感受／知識／信念／行動」分頁切換的展示層。
 *
 * 刻意做成受控元件（active/onSelect 由外部提供）：autoplay 與 pause
 * 邏輯統一交給 page.tsx 的 useAutoplayTab hook 管理，因為底部
 * PipelineStrip 點擊時也需要驅動同一份 active 狀態並觸發同樣的
 * 「暫停後恢復」行為，兩邊各自管理一份狀態容易漂移。
 */

export type IntroTab = { key: string; label: string; accent: string };

type Props = {
  tabs: IntroTab[];
  active: number;
  onSelect: (i: number) => void;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  children: (activeIndex: number) => React.ReactNode;
};

export default function TabCarousel({ tabs, active, onSelect, onMouseEnter, onMouseLeave, children }: Props) {
  return (
    <div className="intro-tab-carousel" onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
      <div className="intro-tab-list" role="tablist" aria-label="人腦層次">
        {tabs.map((t, i) => (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={active === i}
            className={`intro-tab-btn${active === i ? " active" : ""}`}
            style={active === i ? { borderColor: t.accent, color: t.accent } : undefined}
            onClick={() => onSelect(i)}
          >
            {t.label}
          </button>
        ))}
      </div>
      {children(active)}
    </div>
  );
}
