"use client";

/**
 * lib/apiFetch.ts — 帶 Firebase ID token 的 fetch 包裝。
 *
 * 後端所有受保護路由（/chat、/me、/user/{uid}/brain/*、/graph/expand/* 等）
 * 都透過 Depends(require_auth) 驗證 Authorization: Bearer <id_token>，
 * 因此前端呼叫後端一律要走這支函式，不要直接用裸的 fetch()。
 *
 * Token 快取／自動刷新交給 firebase/auth 本身處理：
 * `getIdToken()`（不強制 refresh）會在 token 尚未過期時直接回傳快取值，
 * 只有接近過期（Firebase SDK 內部判斷，一般是剩餘 <5 分鐘）才會真的發送
 * refresh 請求，所以這裡每次呼叫都取一次是安全、低成本的。
 *
 * 若呼叫回傳 401（多半是 token 剛好在請求當下失效，例如帳號被刪除、
 * 或極少數 clock skew 邊界情況），會強制 refresh 一次 token 後重試一次；
 * 仍失敗就把原始 401 response 交還給呼叫端自行處理。
 */

import { auth } from "./firebase";

async function withAuthHeader(init: RequestInit, forceRefresh: boolean): Promise<RequestInit> {
  const user  = auth.currentUser;
  const token = user ? await user.getIdToken(forceRefresh) : null;

  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);

  return { ...init, headers };
}

export async function apiFetch(url: string, init: RequestInit = {}): Promise<Response> {
  // 尚未有任何使用者（AuthProvider 還在做匿名登入的第一拍）時，
  // 等一下下再試一次，避免頁面剛載入時第一批請求都漏帶 token。
  if (!auth.currentUser) {
    await new Promise((resolve) => setTimeout(resolve, 300));
  }

  const first = await withAuthHeader(init, false);
  const res   = await fetch(url, first);

  if (res.status !== 401 || !auth.currentUser) {
    return res;
  }

  const retried = await withAuthHeader(init, true);
  return fetch(url, retried);
}
