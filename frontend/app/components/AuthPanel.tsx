"use client";

/**
 * AuthPanel.tsx — 會員系統的 UI 進入點。
 *
 * 顯示位置：header 右側（見 page.tsx）。
 *   - 匿名使用者 → 顯示「訪客模式」徽章 + 「登入 / 建立帳號」按鈕，
 *     點擊開啟 modal，分「建立帳號」「登入」兩個分頁，各自都提供
 *     Google 與 Email 兩種方式；「建立帳號」分頁在匿名狀態下走的是
 *     upgrade（linkWithPopup/linkWithCredential），uid 不變、資料延續。
 *   - 正式帳號 → 顯示 email/displayName + 「登出」。
 *
 * 視覺語言沿用 globals.css 既有的 CSS variables（--purple/--teal/
 * --orange/--panel-2/--border/--text/--muted），不引入新的調色盤，
 * 讓這個元件看起來像產品原本就有的一部分，而不是外掛上去的第三方元件。
 */

import { useState } from "react";
import { useAuth } from "./AuthProvider";

type Mode = "signup" | "signin";
type Method = "google" | "email";

export default function AuthPanel() {
  const { user, error, clearError, signInGoogle, signInEmail, signUpEmail,
           upgradeWithGoogle, upgradeWithEmail, signOutUser } = useAuth();

  const [open, setOpen]   = useState(false);
  const [mode, setMode]   = useState<Mode>("signup");
  const [busy, setBusy]   = useState(false);

  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [name, setName]         = useState("");

  if (!user) return null;

  const closeModal = () => {
    setOpen(false);
    clearError();
    setEmail(""); setPassword(""); setName("");
  };

  async function run(method: Method) {
    setBusy(true);
    try {
      if (mode === "signup") {
        if (user!.isAnonymous) {
          // 升級：保留匿名期間累積的 Personal Brain / 對話紀錄
          if (method === "google") await upgradeWithGoogle();
          else await upgradeWithEmail(email, password, name);
        } else {
          if (method === "google") await signInGoogle();
          else await signUpEmail(email, password, name);
        }
      } else {
        if (method === "google") await signInGoogle();
        else await signInEmail(email, password);
      }
      closeModal();
    } catch {
      // 錯誤訊息已由 AuthProvider 設定進 context.error，這裡不需要重複處理
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      {user.isAnonymous ? (
        <>
          <span style={{
            fontSize: "0.62rem", padding: "3px 9px", borderRadius: 12,
            border: "1px solid rgba(122,138,154,0.35)",
            color: "var(--muted)", letterSpacing: "0.04em",
          }}>
            訪客模式
          </span>
          <button type="button" onClick={() => { setMode("signup"); setOpen(true); }}>
            登入 / 建立帳號
          </button>
        </>
      ) : (
        <>
          <span style={{ fontSize: "0.7rem", color: "var(--text)", opacity: 0.8 }}>
            {user.displayName || user.email}
          </span>
          <button type="button" onClick={signOutUser}>登出</button>
        </>
      )}

      {open && (
        <div
          onClick={closeModal}
          style={{
            position: "fixed", inset: 0, background: "rgba(3,1,8,0.72)",
            display: "flex", alignItems: "center", justifyContent: "center",
            zIndex: 1000, padding: 20,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 340, maxWidth: "100%",
              background: "var(--panel)", border: "1px solid var(--border)",
              borderRadius: 14, padding: 20,
              display: "flex", flexDirection: "column", gap: 14,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{
                fontSize: "0.62rem", letterSpacing: "0.18em", textTransform: "uppercase",
                color: "var(--orange)",
              }}>
                {user.isAnonymous ? "帳號" : "帳號"}
              </span>
              <button type="button" onClick={closeModal} style={{ padding: "2px 8px" }}>✕</button>
            </div>

            {user.isAnonymous && (
              <p style={{ margin: 0, fontSize: "0.7rem", color: "var(--muted)", lineHeight: 1.6 }}>
                你目前以訪客身分使用，「建立帳號」會把這台裝置累積的
                Personal Brain 與對話紀錄一併保留到新帳號。
              </p>
            )}

            <div style={{ display: "flex", gap: 6 }}>
              <TabButton active={mode === "signup"} onClick={() => setMode("signup")}>
                {user.isAnonymous ? "建立帳號" : "註冊新帳號"}
              </TabButton>
              <TabButton active={mode === "signin"} onClick={() => setMode("signin")}>
                登入既有帳號
              </TabButton>
            </div>

            <button
              type="button"
              disabled={busy}
              onClick={() => run("google")}
              style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}
            >
              <GoogleG /> 使用 Google {mode === "signup" ? "建立帳號" : "登入"}
            </button>

            <div style={{ display: "flex", alignItems: "center", gap: 8, opacity: 0.4 }}>
              <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
              <span style={{ fontSize: "0.6rem" }}>或使用 Email</span>
              <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {mode === "signup" && (
                <input
                  placeholder="顯示名稱（選填）"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  style={inputStyle}
                />
              )}
              <input
                type="email"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                style={inputStyle}
              />
              <input
                type="password"
                placeholder="密碼（至少 6 碼）"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={inputStyle}
              />
              <button
                type="button"
                disabled={busy || !email || password.length < 6}
                onClick={() => run("email")}
                className="detail-btn-add"
              >
                {mode === "signup" ? "建立帳號" : "登入"}
              </button>
            </div>

            {error && (
              <p style={{ margin: 0, fontSize: "0.68rem", color: "#e85d60" }}>{error}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────

function TabButton({ active, onClick, children }: {
  active: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        flex: 1, fontSize: "0.66rem", padding: "6px 4px",
        border: `1px solid ${active ? "var(--orange)" : "var(--border)"}`,
        background: active ? "rgba(232,112,48,0.1)" : "transparent",
        color: active ? "var(--orange)" : "var(--muted)",
      }}
    >
      {children}
    </button>
  );
}

const inputStyle: React.CSSProperties = {
  border: "1px solid var(--border)",
  background: "rgba(13,18,32,0.8)",
  color: "var(--text)",
  borderRadius: 8,
  padding: "8px 10px",
  fontSize: "0.75rem",
  fontFamily: "IBM Plex Mono, monospace",
};

function GoogleG() {
  return (
    <svg width="14" height="14" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
      <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
      <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
      <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
    </svg>
  );
}
