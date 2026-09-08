"use client";

/**
 * PersonalBrainSection.tsx — Section 3:「探索 Personal Brain」。
 *
 * 上半是切片示意圖（SliceDiagram）,把「同一顆大腦,兩種切法」的
 * 比喻畫出來;下半直接複用真實 PersonalBrain 元件（含它內建的
 * 「沙漏／分佈」切換鈕）,餵 guideData 假資料——資料與 Section 1/2
 * 共用同一批節點,讓使用者意識到這是同一個世界的第三種視角。
 */

import dynamic from "next/dynamic";
import SliceDiagram from "./SliceDiagram";
import {
  GUIDE_LAYOUT,
  GUIDE_PERSONAL_BRAIN,
  GUIDE_RC_RANKING,
  GUIDE_BERTOPIC_LAYOUT,
  guideFetchRCPaths,
  guideFetchFoundationTree,
} from "../guideData";

const PersonalBrain = dynamic(() => import("../../components/PersonalBrain"), { ssr: false });

export default function PersonalBrainSection() {
  return (
    <div className="guide-personal-section">
      <SliceDiagram />

      <div className="guide-slice-copy">
        <div className="guide-slice-copy-item">
          <span className="guide-slice-copy-title">BERTopic · 水平切片</span>
          <p>把你的收藏投影到全域語意地圖上，看你的偏好座落在哪個方向——淡色點是全域概念，亮色點是你的 Personal Brain。</p>
        </div>
        <div className="guide-slice-copy-item">
          <span className="guide-slice-copy-title">Hourglass · 垂直切片</span>
          <p>針對目前聚焦的 RC，垂直切出它的因果剖面——下半錐是支撐它的 foundation，上半錐是它能延伸出的 downstream，角度呼應 BERTopic 裡的語意方向。</p>
        </div>
      </div>

      <div className="guide-3d-frame guide-3d-frame-tall">
        <PersonalBrain
          layout={GUIDE_LAYOUT}
          bertopicLayout={GUIDE_BERTOPIC_LAYOUT}
          personalBrain={GUIDE_PERSONAL_BRAIN}
          rcRanking={GUIDE_RC_RANKING}
          fetchRCPaths={guideFetchRCPaths}
          fetchFoundationTree={guideFetchFoundationTree}
          onLaunchSkill={() => {}}
          width={560}
          height={420}
        />
      </div>

      <p className="guide-note">
        這裡是操作心法示範，資料固定不變；到 dashboard 才會接上你真正的對話與收藏。試著點上方「沙漏／分佈」切換鈕，或點沙漏裡的 downstream 節點重新聚焦。
      </p>
    </div>
  );
}
