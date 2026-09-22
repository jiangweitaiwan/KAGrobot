"use client";

import { useCallback, useEffect, useState } from "react";
import { adminApi, type VipRow } from "../../../lib/adminApi";
import { ConfirmButton, fmt } from "../../account/components/ui";
import { PlanChip, daysLeft, extendedExpiry, userLabel } from "./adminUi";

export default function AdminVips({
  onOpenUser, onChanged, refreshKey,
}: { onOpenUser: (uid: string) => void; onChanged: () => void; refreshKey: number }) {
  const [vips, setVips]   = useState<VipRow[] | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy]   = useState<string | null>(null);

  const load = useCallback(async () => {
    try { setVips(await adminApi.vips()); setError(""); }
    catch (err) { setError(err instanceof Error ? err.message : "讀取 VIP 名單失敗"); setVips((p) => p ?? []); }
  }, []);

  useEffect(() => { load(); }, [load, refreshKey]);

  async function act(uid: string, fn: () => Promise<unknown>) {
    setBusy(uid); setError("");
    try { await fn(); await load(); onChanged(); }
    catch (err) { setError(err instanceof Error ? err.message : "操作失敗"); }
    finally { setBusy(null); }
  }

  if (vips === null) return <div className="acct-loading">載入 VIP 名單…</div>;

  const expiredCount = vips.filter((v) => v.expired).length;

  return (
    <div>
      <p className="acct-note" style={{ margin: "0 0 14px" }}>
        已到期者排在最前面：到期後系統會自動當作普通會員，但名單裡仍保留，方便你決定要延長或清理。
        {expiredCount > 0 && `　目前有 ${expiredCount} 位已到期。`}
      </p>
      {error && <p className="acct-error">{error}</p>}

      {vips.length === 0 ? (
        <div className="acct-empty">
          <strong>目前沒有 VIP</strong>
          <span>到「使用者」分頁找到對象，即可開通 VIP。</span>
        </div>
      ) : (
        <div className="adm-table-wrap">
          <table className="adm-table">
            <thead>
              <tr>
                <th scope="col">使用者</th><th scope="col">狀態</th>
                <th scope="col" className="num">本月 token</th><th scope="col"><span className="sr-only">操作</span></th>
              </tr>
            </thead>
            <tbody>
              {vips.map((v) => {
                const isBusy = busy === v.uid;
                const left = v.plan_expires_at ? daysLeft(v.plan_expires_at) : null;
                return (
                  <tr key={v.uid}>
                    <td>
                      <button type="button" className="adm-link" onClick={() => onOpenUser(v.uid)}>
                        {userLabel({ ...v, is_anonymous: false })}
                      </button>
                    </td>
                    <td>
                      {v.expired ? <PlanChip plan="vip" expired />
                        : left === null ? <span className="adm-chip vip">不過期</span>
                        : <span className={`adm-chip ${left <= 7 ? "soon" : "vip"}`}>剩 {left} 天</span>}
                      {v.plan_expires_at && <span className="adm-sub">{new Date(v.plan_expires_at).toLocaleDateString("zh-TW")}</span>}
                    </td>
                    <td className="num">{fmt(v.month_tokens)}</td>
                    <td className="adm-actions">
                      {v.plan_expires_at !== null && (
                        <ConfirmButton
                          label="延長 30 天" confirmLabel="確定延長" className="acct-btn"
                          disabled={isBusy}
                          onConfirm={() => act(v.uid, () => adminApi.setPlan(v.uid, "vip", extendedExpiry(v.plan_expires_at, 30)))}
                        />
                      )}
                      <ConfirmButton
                        label="改為普通" confirmLabel="確定改為普通" className="acct-btn quiet"
                        disabled={isBusy}
                        onConfirm={() => act(v.uid, () => adminApi.setPlan(v.uid, "normal", null))}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
