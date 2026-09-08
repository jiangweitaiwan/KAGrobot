"use client";

/**
 * ChatDemoCard.tsx — Section 1 步驟①「提問」的輕量仿製卡片。
 *
 * 只做「看起來像 dashboard 的 chat-box」的靜態展示，套用真實
 * globals.css 的 .chat-box / .msg class，不掛任何 state machine
 * 或 API，文字內容固定播放一次淡入即可。
 */

import { useEffect, useState } from "react";

type Props = {
  question: string;
};

export default function ChatDemoCard({ question }: Props) {
  const [showReply, setShowReply] = useState(false);

  useEffect(() => {
    setShowReply(false);
    const t = setTimeout(() => setShowReply(true), 600);
    return () => clearTimeout(t);
  }, [question]);

  return (
    <div className="guide-demo-frame">
      <div className="guide-demo-label">對應 dashboard 區塊：左側對話欄</div>
      <div className="chat-box guide-chat-demo">
        <div className="msg user">{question}</div>
        {showReply && (
          <div className="msg ai guide-fade-in">
            這聽起來牽涉到「價格為什麼不會馬上調整」——讓我先在知識場域裡找找相關的概念。
          </div>
        )}
      </div>
    </div>
  );
}
