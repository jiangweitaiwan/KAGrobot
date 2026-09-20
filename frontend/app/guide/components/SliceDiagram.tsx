"use client";

/**
 * SliceDiagram.tsx — 「同一顆大腦，兩種切法」示意圖。
 *
 * 改版重點：接受 view，與下方真正的檢視器連動。
 *   view="2d"        → 水平圓盤亮起、微微抽離，垂直柱狀退到背景
 *   view="hourglass" → 垂直柱狀亮起，水平圓盤退到背景
 *
 * 光是把兩個切面畫在一起，說服力不夠；真正讓人相信「這是同一顆大腦」
 * 的是腰身那顆點——它在球體上、在語意地圖上、在沙漏的腰身，是同一個
 * RC、同一個顏色。使用者在沙漏裡重新聚焦時，它會跟著移動。
 */

type Props = {
  view:        "2d" | "hourglass";
  waistLabel:  string;
  /** 腰身在球體上的水平偏移（-1 ~ 1），用來呼應目前聚焦的 RC 在語意
   *  地圖上的方向。重新聚焦時這顆點會滑過去。 */
  waistBias?:  number;
};

const TEAL = "#00d4aa";

export default function SliceDiagram({ view, waistLabel, waistBias = 0 }: Props) {
  const on2d   = view === "2d";
  const onHour = view === "hourglass";
  const waistX = 200 + Math.max(-1, Math.min(1, waistBias)) * 62;

  return (
    <svg viewBox="0 0 400 250" className="guide-slice-svg" role="img" aria-label="大腦切片示意圖">
      {/* 球體輪廓 */}
      <ellipse cx="200" cy="125" rx="110" ry="98" fill="none"
               stroke="rgba(232,238,248,0.16)" strokeWidth="1.5" />
      <ellipse cx="200" cy="125" rx="110" ry="36" fill="none"
               stroke="rgba(232,238,248,0.1)" strokeWidth="1" strokeDasharray="3,4" />

      {/* 垂直柱狀切片（Hourglass） */}
      <g className="guide-slice-layer" data-on={onHour}
         style={{ transform: onHour ? "translateY(0)" : "translateY(4px)" }}>
        <path
          d="M 178 32 L 222 32 L 236 218 L 164 218 Z"
          fill={onHour ? "rgba(255,204,77,0.14)" : "rgba(255,204,77,0.05)"}
          stroke="#ffcc4d"
          strokeWidth={onHour ? 1.8 : 1}
        />
        <text x="200" y="240" textAnchor="middle" fontSize="10.5" fill="#ffcc4d"
              letterSpacing="0.05em">
          垂直切片 · 單一 RC 的因果剖面
        </text>
      </g>

      {/* 水平切片（BERTopic） */}
      <g className="guide-slice-layer" data-on={on2d}
         style={{ transform: on2d ? "translateY(-6px)" : "translateY(0)" }}>
        <ellipse cx="200" cy="125" rx="102" ry="25"
                 fill={on2d ? "rgba(0,232,255,0.16)" : "rgba(0,232,255,0.05)"}
                 stroke="#00e8ff" strokeWidth={on2d ? 1.8 : 1} />
        <text x="200" y="20" textAnchor="middle" fontSize="10.5" fill="#66eeff"
              letterSpacing="0.05em">
          水平切片 · 收藏在語意地形裡的分佈
        </text>
      </g>

      {/* 腰身：兩種切面共用的錨點 */}
      <g style={{ transform: `translateX(${waistX - 200}px)`, transition: "transform 620ms cubic-bezier(.4,0,.2,1)" }}>
        <circle cx="200" cy="125" r="9" fill={TEAL} opacity="0.18" />
        <circle cx="200" cy="125" r="4.5" fill={TEAL} />
        <text x="200" y="112" textAnchor="middle" fontSize="9" fill="#7fe6cc">
          {waistLabel}
        </text>
      </g>
    </svg>
  );
}
