"use client";

/**
 * AttachmentCard.tsx — 圖片／檔案附件的縮圖卡 + 內建 lightbox。
 *
 * 設計背景：
 *   MainAgent 透過 get_sandbox_file 產生的圖表／檔案，後端已經結構化成
 *   ChatResponse.attachments（ref/kind/url/filename/created_at/
 *   expires_in_hours），不再依賴模型把網址正確寫進回覆文字。前端只需要
 *   依 ref 把訊息文字裡的 [[attachment:N]] 佔位符換成這裡的卡片即可
 *   （見 page.tsx 的 splitByPlaceholder）。
 *
 * 兩種呈現：
 *   - kind === "image" → 縮圖（裁成小方塊），點擊開全螢幕 lightbox。
 *   - kind === "file"  → 檔案圖示 + 檔名 + 下載連結。
 *
 * 過期處理：
 *   簽名 URL 效期 24 小時，純前端用 created_at + expires_in_hours 換算，
 *   過期後卡片變灰階並顯示「已過期」徽章，不可再點擊放大／下載
 *   （連結本身在雲端那端也已經失效，這裡只是提前告知使用者，不去
 *   猜測或重新產生新連結）。
 */

import { useState } from "react";

export type Attachment = {
  ref:              string;
  kind:             "image" | "file";
  url:              string;
  filename:         string;
  created_at:       number;   // unix ms
  expires_in_hours: number;
};

export function isAttachmentExpired(a: Attachment): boolean {
  return Date.now() > a.created_at + a.expires_in_hours * 3600_000;
}

// ─────────────────────────────────────────────────────────────
// AttachmentCard
// ─────────────────────────────────────────────────────────────

export function AttachmentCard({ attachment }: { attachment: Attachment }) {
  const [open, setOpen] = useState(false);
  const expired = isAttachmentExpired(attachment);
  const isImage = attachment.kind === "image";

  return (
    <>
      <div
        onClick={() => { if (!expired && isImage) setOpen(true); }}
        title={expired ? `${attachment.filename} · 連結已過期` : attachment.filename}
        style={{
          position:      "relative",
          display:       "inline-flex",
          flexDirection: "column",
          width:         96,
          flexShrink:    0,
          borderRadius:  10,
          overflow:      "hidden",
          border:        "1px solid rgba(83,74,183,0.3)",
          background:    "rgba(13,18,32,0.6)",
          cursor:        expired ? "default" : (isImage ? "zoom-in" : "default"),
          margin:        "4px 6px 4px 0",
          verticalAlign: "top",
          opacity:       expired ? 0.5 : 1,
          transition:    "opacity 0.15s ease",
        }}
      >
        {isImage ? (
          <div style={{ width: "100%", height: 72, background: "#06080f", overflow: "hidden" }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={attachment.url}
              alt={attachment.filename}
              style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
            />
          </div>
        ) : (
          <div style={{
            width: "100%", height: 72, display: "flex",
            alignItems: "center", justifyContent: "center", fontSize: 26,
          }}>
            📄
          </div>
        )}

        <div style={{
          padding:      "4px 6px",
          fontSize:     "0.58rem",
          color:        "#9090c0",
          overflow:     "hidden",
          textOverflow: "ellipsis",
          whiteSpace:   "nowrap",
        }}>
          {attachment.filename}
        </div>

        {expired ? (
          <div style={{
            position:      "absolute",
            top:           4,
            right:         4,
            fontSize:      "0.5rem",
            background:    "rgba(162,45,45,0.85)",
            color:         "#fff",
            padding:       "1px 5px",
            borderRadius:  8,
            letterSpacing: "0.02em",
          }}>
            已過期
          </div>
        ) : !isImage ? (
          <a
            href={attachment.url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            style={{
              display:      "block",
              textAlign:    "center",
              fontSize:     "0.56rem",
              padding:      "3px 0",
              color:        "var(--teal)",
              borderTop:    "1px solid rgba(83,74,183,0.2)",
              textDecoration: "none",
            }}
          >
            下載
          </a>
        ) : null}
      </div>

      {/* ── Lightbox（僅圖片、未過期才會開啟）── */}
      {open && isImage && !expired && (
        <div
          onClick={() => setOpen(false)}
          style={{
            position:       "fixed",
            inset:          0,
            background:     "rgba(3,1,8,0.88)",
            display:        "flex",
            alignItems:     "center",
            justifyContent: "center",
            zIndex:         1000,
            cursor:         "zoom-out",
            padding:        24,
            backdropFilter: "blur(2px)",
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              display:       "flex",
              flexDirection: "column",
              alignItems:    "center",
              gap:           10,
              maxWidth:      "90vw",
              maxHeight:     "90vh",
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={attachment.url}
              alt={attachment.filename}
              style={{
                maxWidth:      "90vw",
                maxHeight:     "78vh",
                objectFit:     "contain",
                borderRadius:  8,
                border:        "1px solid rgba(232,238,248,0.15)",
                background:    "#06080f",
              }}
            />
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <span style={{ color: "#c8daf0", fontSize: "0.75rem", fontFamily: "monospace" }}>
                {attachment.filename}
              </span>
              <a
                href={attachment.url}
                target="_blank"
                rel="noopener noreferrer"
                style={{ fontSize: "0.75rem", color: "var(--teal)", textDecoration: "none" }}
              >
                下載
              </a>
              <button onClick={() => setOpen(false)} style={{ fontSize: "0.7rem" }}>
                關閉
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// ─────────────────────────────────────────────────────────────
// AttachmentTray — 整個 session 至今所有附件的橫向彙總列（可折疊）
// ─────────────────────────────────────────────────────────────

export function AttachmentTray({ attachments }: { attachments: Attachment[] }) {
  const [collapsed, setCollapsed] = useState(true);

  if (attachments.length === 0) return null;

  return (
    <div style={{
      border:        "1px solid rgba(232,112,48,0.15)",
      borderRadius:  10,
      background:    "rgba(13,18,32,0.5)",
      flexShrink:    0,
      overflow:      "hidden",
    }}>
      <button
        type="button"
        onClick={() => setCollapsed((v) => !v)}
        style={{
          width:          "100%",
          display:        "flex",
          justifyContent: "space-between",
          alignItems:     "center",
          padding:        "4px 10px",
          background:     "transparent",
          border:         "none",
          color:          "var(--orange)",
          fontSize:       "0.62rem",
          letterSpacing:  "0.08em",
          cursor:         "pointer",
        }}
      >
        <span>📎 本次分析產出 · {attachments.length}</span>
        <span>{collapsed ? "▸" : "▾"}</span>
      </button>
      {!collapsed && (
        <div style={{ display: "flex", overflowX: "auto", gap: 4, padding: "0 8px 8px" }}>
          {attachments.map((a) => (
            <AttachmentCard key={`${a.ref}-${a.created_at}`} attachment={a} />
          ))}
        </div>
      )}
    </div>
  );
}
