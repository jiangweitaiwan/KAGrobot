"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { accountApi, type MyUsage } from "./accountApi";

/**
 * 目前使用者的等級 / 額度 / 用量。dashboard 與會員中心共用。
 * enabled=false（例如尚未取得登入 uid）時不發請求。
 * refresh() 可在每輪對話結束後呼叫，讓額度指示即時更新；失敗時保留上一份資料。
 */
export function useMyUsage(enabled: boolean) {
  const [usage, setUsage]     = useState<MyUsage | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const alive = useRef(true);

  useEffect(() => () => { alive.current = false; }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await accountApi.usage();
      if (alive.current) { setUsage(data); setError(null); }
    } catch (err) {
      if (alive.current) setError(err instanceof Error ? err.message : "讀取用量失敗");
    } finally {
      if (alive.current) setLoading(false);
    }
  }, []);

  useEffect(() => { if (enabled) refresh(); }, [enabled, refresh]);

  return { usage, error, loading, refresh };
}
