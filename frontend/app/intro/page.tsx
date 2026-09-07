"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { Space_Grotesk, Cormorant_Garamond } from "next/font/google";
import Link from "next/link";
import TabCarousel, { type IntroTab } from "./components/TabCarousel";
import PipelineStrip from "./components/PipelineStrip";
import "./intro.css";
import type { IntroBrainMode } from "./components/IntroGlobalBrain";


const IntroGlobalBrain = dynamic(() => import("./components/IntroGlobalBrain"), { ssr: false });

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
});

// 古典襯線字體:用在引言、舉例卡、機制註腳等「手稿感」文字上,
// 與 Space Grotesk 的現代標題形成「古老人文問題 × 現代智能」的對比。
const cormorant = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["500", "600"],
  style: ["normal", "italic"],
  variable: "--font-serif",
});

const TABS: IntroTab[] = [
  { key: "feel", label: "感受", accent: "#00e8ff", glyph: "感" },
  { key: "knowledge", label: "知識", accent: "#ff9a52", glyph: "知" },
  { key: "belief", label: "信念", accent: "#ff5fd1", glyph: "信" },
  { key: "action", label: "行動", accent: "#ffcc4d", glyph: "行" },
];

const TAB_COPY = [
  {
    headline: "感受:邊緣系統的原型直覺",
    mechanism: "邊緣系統(limbic system)是大腦掌管情緒與本能反應的區域,座落在演化上最古老的位置。神經科學研究指出,像杏仁核這類邊緣結構,會在還沒經過理性分析前就先對情境做出快速的情緒評估——這些直覺不是憑空出現的,而是長期生活經驗被大腦不斷內化、濃縮後留下的『總結性』判斷。",
    example: "『稀缺性直覺』就是一個例子:看到限量商品或倒數計時,大部分人會不自覺地想搶購,這種反應早在意識到自己在『分析』之前就已經發生。在 ZenMind 裡,這類概念被放在最核心的 article 層。",
    anatomy: "對應機制:article concept · 核心殼層(邊緣系統)",
  },
  {
    headline: "知識:大腦皮質累積的具體知識",
    mechanism: "大腦皮質,尤其是負責高階認知的新皮質(neocortex),是儲存可驗證、可教授知識的地方。它比邊緣系統晚演化出現,負責邏輯推理、長期記憶與抽象思考——這也是為什麼具體的理論、模型、研究文獻,會被歸類在皮質對應的外層。",
    example: "『Calvo 定價模型』是一篇具體、可以被檢驗的經濟學理論,對應到皮質裡儲存的一個知識模組。慢想(random walk)實際上就是在這一層裡遊走,蒐集與需求相關的線索。",
    anatomy: "對應機制:paper concept · 外層殼層(大腦皮質)",
  },
  {
    headline: "信念:神經同步放電,線索收斂成迴路",
    mechanism: "神經科學裡有一條經典的 Hebb 法則:『一起放電的神經元,會連在一起』(cells that fire together wire together)。當來自不同腦區的訊號——邊緣系統的直覺、皮質累積的知識——同時、反覆地被激活,它們之間的突觸連結就會被強化,形成一個穩定的迴路。這正是大腦形成新概念與鞏固記憶的核心機制之一。",
    example: "當『稀缺性直覺』『菜單成本理論』『價格僵固性』『Calvo 定價模型』等幾條分散的線索同步指向同一個方向,foundation tree 就會收斂成一個新的信念——『通膨黏性價格假說』。畫面中那條通往 RC 的虛線,是這個信念已經預備好的下一步,就像神經迴路穩定之後,自然會有一條通路準備被使用。",
    anatomy: "對應機制:foundation tree 收斂 → 推薦 RC(突觸連結強化)",
  },
  {
    headline: "行動:前額葉把信念轉譯成執行",
    mechanism: "前額葉皮質(prefrontal cortex)負責大腦的『執行功能』(executive function)——把已經形成的判斷與意圖,轉換成具體的行動計畫與動作執行。信念不會自動變成行動,中間需要一個「可被執行」的具體方案(RC),再由這一層按下開關,把它送出去。",
    example: "『通膨黏性價格假說』收斂之後,系統先產生一個可執行的 RC——『升息衝擊分析』。按下它旁邊的閃電,才真正把判斷送進 MainAgent 執行,對應前額葉『決定要做』與『真正去做』之間的最後一道開關,畫面上會看到一道光沿著信念 → RC → 行動的路徑送出去。",
    anatomy: "對應機制:select RC skill(閃電開關)→ MainAgent 執行(前額葉執行功能)",
  },
];

function usePanelSize(ref: React.RefObject<HTMLDivElement>) {
  const [size, setSize] = useState({ w: 480, h: 480 });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setSize({ w: Math.floor(width) || 480, h: Math.floor(height) || 480 });
      }
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

const AUTOPLAY_MS = 5000;
const RESUME_AFTER_MS = 8000;

export default function IntroPage() {
  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);
  const resumeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stageSectionRef = useRef<HTMLDivElement>(null);
  const visualRef = useRef<HTMLDivElement>(null);
  const visualSize = usePanelSize(visualRef as React.RefObject<HTMLDivElement>);

  useEffect(() => {
    if (paused) return;
    const t = setInterval(() => setActive((a) => (a + 1) % TABS.length), AUTOPLAY_MS);
    return () => clearInterval(t);
  }, [paused]);

  useEffect(() => () => { if (resumeTimer.current) clearTimeout(resumeTimer.current); }, []);

  const selectTab = useCallback((i: number) => {
    setActive(i);
    setPaused(true);
    if (resumeTimer.current) clearTimeout(resumeTimer.current);
    resumeTimer.current = setTimeout(() => {
      resumeTimer.current = null;
      setPaused(false);
    }, RESUME_AFTER_MS);
  }, []);

  const handlePipelineSelect = useCallback((tabIndex: number) => {
    selectTab(tabIndex);
    stageSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [selectTab]);

  return (
    <div className={`intro-root ${spaceGrotesk.variable} ${cormorant.variable}`}>
      <div className="intro-inner">
        <section className="intro-hero">
          <svg className="intro-enso" viewBox="0 0 400 400" aria-hidden="true">
            <circle cx="200" cy="200" r="170" />
          </svg>
          <span className="intro-eyebrow">ZenMind</span>
          <h1 className="intro-headline">ZenMind：建立感知到行動的因果關係</h1>
          <p className="intro-subhead">
            禪，是一顆包羅萬象的心。這個系統把龐雜、抽象的感知收攏成脈絡清晰的信念，
            再讓信念驅動出具體可執行的行動——從感受出發、經知識累積、至信念收斂，
            最終圓滿成為行動，是同一顆心智完整的因果鏈。往下滑，看看四個層次分別
            對應 ZenMind 裡的哪一種心智活動。
          </p>
          <div className="intro-hero-actions">
            <Link href="/" className="intro-btn-primary">開始使用</Link>
            <button
              type="button"
              className="intro-btn-secondary"
              onClick={() => stageSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
            >
              往下看
            </button>
          </div>
        </section>

        <div className="intro-ornament" aria-hidden="true">✦</div>

        <section className="intro-stage-section" ref={stageSectionRef}>
          <h2 className="intro-section-title">四個層次</h2>
          <p className="intro-section-lede">
            人腦從邊緣系統的直覺，到皮質的具體知識，再到神經網絡同步放電形成的信念，
            最終驅動行動—— ZenMind 的殼層結構與這個歷程一一對應。
          </p>

          <TabCarousel
            tabs={TABS}
            active={active}
            onSelect={selectTab}
            onMouseEnter={() => setPaused(true)}
            onMouseLeave={() => { if (!resumeTimer.current) setPaused(false); }}
          >
            {(activeIndex) => {
              const copy = TAB_COPY[activeIndex];
              const mode = TABS[activeIndex].key as IntroBrainMode;
              return [
                <div className="intro-stage-visual-inner" ref={visualRef} key="visual">
                  <IntroGlobalBrain
                    mode={mode}
                    onRegionClick={(clicked) => {
                      const idx = TABS.findIndex((t) => t.key === clicked);
                      if (idx !== -1) selectTab(idx);
                    }}
                    width={visualSize.w}
                    height={visualSize.h}
                  />
                </div>,
                <div className="intro-stage-copy-inner" key="copy">
                  <h3 className="intro-tab-headline">{copy.headline}</h3>
                  <p className="intro-tab-body">{copy.mechanism}</p>
                  <div className="intro-tab-example">
                    <span className="intro-tab-example-label">舉例</span>
                    <p>{copy.example}</p>
                  </div>
                  <p className="intro-tab-anatomy">{copy.anatomy}</p>
                </div>,
              ];
            }}
          </TabCarousel>
        </section>

        <div className="intro-ornament" aria-hidden="true">✦</div>

        <section className="intro-pipeline-section">
          <h2 className="intro-section-title">從感受到行動</h2>
          <p className="intro-section-lede">
            這四個層次不只是比喻，它們對應到系統實際運作的五個步驟。點一個步驟，回去看看它是哪個層次。
          </p>
          <PipelineStrip onSelectStage={handlePipelineSelect} />
        </section>
      </div>
    </div>
  );
}
