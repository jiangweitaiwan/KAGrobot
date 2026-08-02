"use client";

/**
 * AuthProvider.tsx — 會員系統的核心 context。
 *
 * 策略：訪客即用、隨時可升級。
 *   - App 掛載時若沒有任何登入狀態，自動 signInAnonymously()，
 *     使用者完全不需要先看到登入畫面就能開始使用（session/brain
 *     等既有資料流全部改以 Firebase uid 取代原本 localStorage 產生的 uid）。
 *   - 匿名使用者可隨時「升級為正式帳號」（Google 或 Email/密碼），
 *     升級用 linkWithPopup / linkWithCredential——uid 完全不變，
 *     後端 users/{uid} 底下的所有資料（brain、sessions）自動延續，
 *     不需要任何資料搬遷。
 *   - 已經是正式帳號的使用者，也可以直接「登入」到別的帳號
 *     （signInWithPopup / signInWithEmailAndPassword），這會切換到
 *     不同的 uid，等同換一個人使用這台裝置。
 *
 * onIdTokenChanged（而非 onAuthStateChanged）：token 每次刷新
 * （包含 linkWithCredential/linkWithPopup 升級後）都会重新觸發，
 * 確保 user.isAnonymous 等衍生狀態即時反映最新登入方式。
 */

import {
  createContext, useContext, useEffect, useMemo, useRef, useState,
  type ReactNode,
} from "react";
import {
  GoogleAuthProvider,
  EmailAuthProvider,
  onIdTokenChanged,
  signInAnonymously,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  linkWithPopup,
  linkWithCredential,
  updateProfile,
  signOut,
  type User,
} from "firebase/auth";
import { auth } from "../../lib/firebase";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type AuthUser = {
  uid:          string;
  email:        string | null;
  displayName:  string | null;
  isAnonymous:  boolean;
};

type AuthContextValue = {
  user:    AuthUser | null;
  loading: boolean;
  error:   string | null;
  clearError: () => void;

  signInGoogle:      () => Promise<void>;
  signInEmail:       (email: string, password: string) => Promise<void>;
  signUpEmail:       (email: string, password: string, displayName: string) => Promise<void>;
  upgradeWithGoogle: () => Promise<void>;
  upgradeWithEmail:  (email: string, password: string, displayName: string) => Promise<void>;
  signOutUser:       () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

// ─────────────────────────────────────────────────────────────
// 錯誤訊息轉換（Firebase 錯誤代碼 → 繁體中文）
// ─────────────────────────────────────────────────────────────

function toErrorMessage(err: unknown): string {
  const code = (err as { code?: string })?.code ?? "";
  const table: Record<string, string> = {
    "auth/email-already-in-use":  "這個 Email 已經被註冊過了，請改用登入。",
    "auth/invalid-email":         "Email 格式不正確。",
    "auth/weak-password":         "密碼太短，至少需要 6 個字元。",
    "auth/wrong-password":        "密碼不正確。",
    "auth/invalid-credential":    "帳號或密碼不正確。",
    "auth/user-not-found":        "找不到這個帳號，請確認 Email 或改用註冊。",
    "auth/too-many-requests":     "嘗試次數過多，請稍後再試。",
    "auth/popup-closed-by-user":  "登入視窗已關閉，請再試一次。",
    "auth/credential-already-in-use":
      "這個帳號已經綁定過其他使用者，請直接用它登入（會切換到該帳號原本的資料）。",
    "auth/network-request-failed": "網路連線異常，請檢查網路後再試。",
  };
  return table[code] ?? "發生未預期的錯誤，請稍後再試。";
}

function toAuthUser(fbUser: User): AuthUser {
  return {
    uid:         fbUser.uid,
    email:       fbUser.email,
    displayName: fbUser.displayName,
    isAnonymous: fbUser.isAnonymous,
  };
}

// ─────────────────────────────────────────────────────────────
// Provider
// ─────────────────────────────────────────────────────────────

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser]       = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  // 避免 signInAnonymously 在 onIdTokenChanged 快速觸發多次時被呼叫多次
  const anonInFlight = useRef(false);

  useEffect(() => {
    const unsub = onIdTokenChanged(auth, async (fbUser) => {
      if (fbUser) {
        setUser(toAuthUser(fbUser));
        setLoading(false);
        return;
      }

      // 完全沒有登入狀態（第一次造訪、或使用者剛登出）→ 自動匿名登入，
      // 讓「先用再說」的體驗不被登入畫面擋住。
      if (anonInFlight.current) return;
      anonInFlight.current = true;
      try {
        await signInAnonymously(auth);
        // onIdTokenChanged 會因此再次觸發，上面 fbUser 分支會設定 user。
      } catch (err) {
        setError(toErrorMessage(err));
        setLoading(false);
      } finally {
        anonInFlight.current = false;
      }
    });
    return () => unsub();
  }, []);

  const clearError = () => setError(null);

  const googleProvider = useMemo(() => new GoogleAuthProvider(), []);

  const signInGoogle = async () => {
    setError(null);
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (err) {
      setError(toErrorMessage(err));
      throw err;
    }
  };

  const signInEmail = async (email: string, password: string) => {
    setError(null);
    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch (err) {
      setError(toErrorMessage(err));
      throw err;
    }
  };

  const signUpEmail = async (email: string, password: string, displayName: string) => {
    setError(null);
    try {
      const cred = await createUserWithEmailAndPassword(auth, email, password);
      if (displayName.trim()) {
        await updateProfile(cred.user, { displayName: displayName.trim() });
      }
    } catch (err) {
      setError(toErrorMessage(err));
      throw err;
    }
  };

  // ── 升級：匿名 → 正式帳號，uid 保持不變 ──────────────────
  const upgradeWithGoogle = async () => {
    setError(null);
    const current = auth.currentUser;
    if (!current) return;
    try {
      await linkWithPopup(current, googleProvider);
    } catch (err) {
      setError(toErrorMessage(err));
      throw err;
    }
  };

  const upgradeWithEmail = async (email: string, password: string, displayName: string) => {
    setError(null);
    const current = auth.currentUser;
    if (!current) return;
    try {
      const credential = EmailAuthProvider.credential(email, password);
      await linkWithCredential(current, credential);
      if (displayName.trim()) {
        await updateProfile(current, { displayName: displayName.trim() });
      }
    } catch (err) {
      setError(toErrorMessage(err));
      throw err;
    }
  };

  const signOutUser = async () => {
    setError(null);
    await signOut(auth);
    // onIdTokenChanged 會偵測到 fbUser === null，自動重新匿名登入，
    // 讓使用者可以無縫繼續以訪客身分使用。
  };

  const value: AuthContextValue = {
    user, loading, error, clearError,
    signInGoogle, signInEmail, signUpEmail,
    upgradeWithGoogle, upgradeWithEmail,
    signOutUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
