"use client";

/**
 * lib/sseFetch.ts — 以 fetch + ReadableStream 手動解析 SSE。
 *
 * 為什麼不用瀏覽器原生 EventSource：
 *   EventSource 不支援自訂 header，沒辦法帶上 apiFetch 附加的
 *   Authorization: Bearer <firebase_id_token>。這個專案所有受保護路由
 *   一律靠這個 header 驗證（見 lib/apiFetch.ts），random walk 的
 *   SSE 呼叫沒有理由自成一格繞過這條路徑，所以改用 fetch 讀
 *   streaming response body，自己拆 "data: ...\n\n" frame。
 *
 * 支援 GET 與 POST：
 *   /walk/stream/{walk_id} 是 GET；/walk/answer/{walk_id} 是 POST
 *   （提交使用者回答後，直接把回應當成新的一段 SSE stream 繼續讀，
 *   對應後端 core/walk_stream.py 的「resume = 開一個新 stream_walk_turns()
 *   generator」設計，不需要另外一個「resume token」機制）。
 */

import { apiFetch } from "./apiFetch";

export type SSEEvent = Record<string, any>;

/**
 * 開一段 SSE 讀取，逐個 event 呼叫 onEvent，直到伺服器端結束這次
 * response（對應後端一次 stream_walk_turns() generator 的自然結束：
 * 停在 user_question 或 walk_complete）。
 *
 * 呼叫端負責在 unmount / 重新觸發時傳入 AbortSignal 取消連線，
 * 避免元件卸載後還在背景讀一個沒人要看的 stream。
 */
export async function streamSSE(
  url: string,
  onEvent: (event: SSEEvent) => void,
  init: RequestInit = {},
): Promise<void> {
  const res = await apiFetch(url, init);
  if (!res.ok || !res.body) {
    throw new Error(`SSE 連線失敗：HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let sepIndex: number;
      while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
        const rawFrame = buffer.slice(0, sepIndex);
        buffer = buffer.slice(sepIndex + 2);
        emitFrame(rawFrame, onEvent);
      }
    }
    // 收尾：若最後一個 frame 沒有以 \n\n 結尾但仍有內容，嘗試解析一次
    if (buffer.trim()) emitFrame(buffer, onEvent);
  } finally {
    reader.releaseLock();
  }
}

function emitFrame(rawFrame: string, onEvent: (event: SSEEvent) => void): void {
  const dataLines = rawFrame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());

  if (dataLines.length === 0) return;

  try {
    const payload = JSON.parse(dataLines.join("\n"));
    onEvent(payload);
  } catch (err) {
    console.warn("[sseFetch] 無法解析 SSE frame:", rawFrame, err);
  }
}
