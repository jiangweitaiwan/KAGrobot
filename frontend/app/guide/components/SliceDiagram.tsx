/**
 * SliceDiagram.tsx — 「同一顆大腦，兩種切法」示意圖。
 *
 * 純 SVG,不含任何 3D 運算:一顆簡化球體疊上兩個切面——
 * 水平圓盤（BERTopic）與垂直柱狀（Hourglass）,搭配虛線示意
 * 切割方向與腰身位置。
 */

export default function SliceDiagram() {
  return (
    <svg viewBox="0 0 400 260" className="guide-slice-svg" role="img" aria-label="大腦切片示意圖">
      {/* 球體輪廓 */}
      <ellipse cx="200" cy="130" rx="110" ry="100" fill="none" stroke="rgba(232,238,248,0.18)" strokeWidth="1.5" />
      <ellipse cx="200" cy="130" rx="110" ry="38" fill="none" stroke="rgba(232,238,248,0.12)" strokeWidth="1" strokeDasharray="3,4" />

      {/* 水平切片（BERTopic） */}
      <ellipse cx="200" cy="130" rx="102" ry="26" fill="rgba(0,232,255,0.10)" stroke="#00e8ff" strokeWidth="1.5" />
      <text x="200" y="80" textAnchor="middle" fontSize="11" fill="#66eeff" letterSpacing="0.05em">
        BERTopic · 水平切片
      </text>

      {/* 垂直柱狀切片（Hourglass） */}
      <path
        d="M 178 34 L 222 34 L 236 226 L 164 226 Z"
        fill="rgba(255,204,77,0.08)"
        stroke="#ffcc4d"
        strokeWidth="1.5"
      />
      <text x="200" y="246" textAnchor="middle" fontSize="11" fill="#ffcc4d" letterSpacing="0.05em">
        Hourglass · 垂直切片
      </text>

      {/* 腰身：兩種切面共同的參照點 */}
      <circle cx="200" cy="130" r="5" fill="#00d4aa" />
      <text x="200" y="118" textAnchor="middle" fontSize="9" fill="#7fe6cc">
        腰身 · 目前聚焦的 RC
      </text>
    </svg>
  );
}
