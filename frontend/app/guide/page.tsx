"use client";

/**
 * /guide — 「操作心法」頁面。
 *
 * 服務對象:準備要用 dashboard 的人（或卡關回來查手冊的人）,
 * 不是還沒登入、只看行銷敘事的訪客——所以調性比 /intro 更像
 * 說明書,但視覺語言延續同一套古典禪風（沿用 intro.css 變數與
 * class,不另外長出第三套風格）。
 *
 * 四個區塊對應四個決策點:
 *   1. 完整流程 demo（六步驟,可點擊切換,每步都標註對應 dashboard 區塊,
 *      步驟②⑤複用真元件,其餘輕量仿製）
 *   2. 探索 GlobalBrain（真元件 playground）
 *   3. 觀測 Personal Brain（切片示意圖 + 真元件 playground）
 *   4. 可計算性（RC 送出後,套用模型算出結論;部分欄位先佔位）
 *
 * 四個區塊共用 guideData.ts 同一條敘事線（升息→物價黏性→
 * 通膨黏性價格假說→升息衝擊分析）,讓使用者意識到這是同一件事的
 * 不同視角,而不是四份互不相干的教材。
 */

import Link from "next/link";
import "../intro/intro.css";
import "./guide.css";
import PipelineDemo from "./components/PipelineDemo";
import GlobalBrainSection from "./components/GlobalBrainSection";
import PersonalBrainSection from "./components/PersonalBrainSection";
import ComputeSection from "./components/ComputeSection";

export default function GuidePage() {
  return (
    <div className="intro-root">
      <div className="intro-inner">
        <section className="intro-hero" style={{ minHeight: "auto", paddingBottom: 20 }}>
          <span className="intro-eyebrow">ZenMind · 操作心法</span>
          <h1 className="intro-headline" style={{ maxWidth: "20ch" }}>
            在混亂的世界裡，收斂出屬於你的信念
          </h1>
          <p className="intro-subhead">
            GlobalBrain 是混亂但完整的知識場域；下面四個區塊，帶你走一遍把混亂收斂成
            專屬 Personal Brain、再讓信念落地成行動的完整迴圈。
          </p>
          <div className="intro-hero-actions">
            <Link href="/dashboard" className="intro-btn-primary">前往 dashboard</Link>
            <Link href="/" className="intro-btn-secondary">← 回首頁</Link>
          </div>
        </section>

        <section className="guide-section">
          <span className="guide-section-kicker">01 · 完整流程</span>
          <h2 className="intro-section-title">六個步驟，一個迴圈</h2>
          <p className="intro-section-lede">
            點擊任一步驟，看它在 dashboard 上長什麼樣子。這六步是「探索混亂 → 收斂信念 → 化為行動」的具體實現。
          </p>
          <PipelineDemo />
        </section>

        <section className="guide-section">
          <span className="guide-section-kicker">02 · 探索 GlobalBrain</span>
          <h2 className="intro-section-title">混亂但完整的知識場域</h2>
          <p className="intro-section-lede">
            左邊挑一個概念，中間讀它，按下加入——右邊會看見它落在語意地圖的哪裡。
          </p>
          <GlobalBrainSection />
        </section>

        <section className="guide-section">
          <span className="guide-section-kicker">03 · 觀測 Personal Brain</span>
          <h2 className="intro-section-title">同一顆大腦，兩種切法</h2>
          <p className="intro-section-lede">
            BERTopic 是水平切片，看你的收藏在語意地形裡的分佈；Hourglass 是垂直切片，看單一 RC 的因果剖面。
          </p>
          <PersonalBrainSection />
        </section>

        <section className="guide-section">
          <span className="guide-section-kicker">04 · 可計算性</span>
          <h2 className="intro-section-title">送出之後，系統真的在算東西</h2>
          <p className="intro-section-lede">
            選定 RC 並送出後，MainAgent 會套用對應的量化模型，算出具體結論，而不是又一輪空泛的對話。
          </p>
          <ComputeSection />
        </section>
      </div>
    </div>
  );
}
