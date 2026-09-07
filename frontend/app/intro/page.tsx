"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { Space_Grotesk } from "next/font/google";
import Link from "next/link";
import TabCarousel, { type IntroTab } from "./components/TabCarousel";
import PipelineStrip from "./components/PipelineStrip";
import "./intro.css";

const IntroGlobalBrain = dynamic(() => import("./components/IntroGlobalBrain"), { ssr: false });
const IntroHourglass = dynamic(() => import("./components/IntroHourglass"), { ssr: false });

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
});

const TABS: IntroTab[] = [
  { key: "feel", label: "感受", accent: "#00e8ff" },
  { key: "knowledge", label: "知識", accent: "#ff9a52" },
  { key: "belief", label: "信念", accent: "#ff5fd1" },
  { key: "action", label: "行動", accent: "#ffcc4d" },
];

const TAB_COPY = [
  {
    headline: "感受：最核心、最抽象的直覺",
    body: "邊緣系統藏在大腦最深處，負責稀缺、公平、風險這類還沒被驗證、卻左右每個判斷的原型情感。在 GlobalBrain 裡，這是最靠近核心的 article 概念層。",
    anatomy: "對應機制：article concept · 核心殼層",
  },
  {
    headline: "知識：向外擴散的皮質",
    body: "越靠近皮質表層，概念越具體、越可驗證——一篇篇論文、一個個模型，構成可以被檢驗的知識庫。慢想（random walk）就是在這一層裡遊走，蒐集與需求相關的線索。",
    anatomy: "對應機制：paper concept · 外層殼層",
  },
  {
    headline: "信念：線索收斂，神經被激活",
    body: "當散落各處的知識線索指向同一個方向，foundation tree 開始收斂成一個推薦的 Resulting Concept——就像多個神經元同步放電，一個信念於是成形。",
    anatomy: "對應機制：foundation tree 收斂 → 推薦 RC",
  },
  {
    headline: "行動：信念驅動推理落地",
    body: "信念不會停在腦中。選定一個 RC 之後，MainAgent 依據這個信念與其 foundation 執行具體分析——沙漏的腰身是剛形成的信念，上錐收斂成一次可執行的行動。",
    anatomy: "對應機制：select RC skill → MainAgent 執行",
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
    <div className={`intro-root ${spaceGrotesk.variable}`}>
      <div className="intro-inner">
        <section className="intro-hero">
          <span className="intro-eyebrow">KAG Robot</span>
          <h1 className="intro-headline">一顆會思考的知識腦</h1>
          <p className="intro-subhead">
            這個系統把推理過程模擬成人腦的樣子：從最核心的直覺出發，
            向外累積知識，讓線索收斂成信念，再由信念驅動具體的行動。
            往下滑，看看四個層次分別對應 GlobalBrain 裡的哪一種激活型態。
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

        <section className="intro-stage-section" ref={stageSectionRef}>
          <h2 className="intro-section-title">四個層次</h2>
          <p className="intro-section-lede">
            人腦從邊緣系統的直覺，到皮質的具體知識，再到神經網絡同步放電形成的信念，
            最終驅動行動——GlobalBrain 的殼層結構與這個歷程一一對應。
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
              const brainMode = activeIndex === 0 ? "feel" : activeIndex === 1 ? "knowledge" : "belief";
              return (
                <>
                  <div className="intro-stage-visual" ref={visualRef}>
                    {activeIndex === 3 ? (
                      <IntroHourglass width={visualSize.w} height={visualSize.h} />
                    ) : (
                      <IntroGlobalBrain mode={brainMode} width={visualSize.w} height={visualSize.h} />
                    )}
                  </div>
                  <div className="intro-stage-copy">
                    <h3 className="intro-tab-headline">{copy.headline}</h3>
                    <p className="intro-tab-body">{copy.body}</p>
                    <p className="intro-tab-anatomy">{copy.anatomy}</p>
                  </div>
                </>
              );
            }}
          </TabCarousel>
        </section>

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
