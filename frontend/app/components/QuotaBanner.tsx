"use client";

import Link from "next/link";
import type { QuotaDetail } from "../../lib/quotaError";

/** 對話被 429 擋下時，顯示在輸入框下方：說明原因、何時恢復、下一步怎麼做。 */
export default function QuotaBanner({ detail, onDismiss }: { detail: QuotaDetail; onDismiss: () => void }) {
  const cta =
    detail.upgrade_hint === "register" ? "註冊帳號"
    : detail.upgrade_hint === "vip"    ? "查看方案"
    : null;

  return (
    <div
      role="alert"
      style={{
        marginTop: 8, padding: "10px 12px", fontSize: "0.74rem", lineHeight: 1.6,
        border: "1px solid rgba(196,87,74,0.5)", background: "rgba(196,87,74,0.08)",
        borderRadius: 10, display: "flex", flexDirection: "column", gap: 8,
      }}
    >
      <span>{detail.message}</span>
      <div style={{ display: "flex", gap: 8 }}>
        {cta && (
          <Link
            href="/account#overview"
            style={{
              fontSize: "0.72rem", padding: "4px 10px", borderRadius: 8, textDecoration: "none",
              border: "1px solid var(--orange)", color: "var(--orange)",
            }}
          >
            {cta}
          </Link>
        )}
        <button type="button" onClick={onDismiss} style={{ fontSize: "0.72rem", padding: "4px 10px" }}>知道了</button>
      </div>
    </div>
  );
}
