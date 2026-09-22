"use client";

import { useState } from "react";
import { useAuth } from "../../components/AuthProvider";
import { accountApi, type MyUsage, type PlanKey, type PlanLimits } from "../../../lib/accountApi";
import EnsoGauge from "./EnsoGauge";
import { LS_HISTORY, LS_PHASE, LS_SESSION, PLAN_LABEL, compact, fmt, formatDate, formatDateTime } from "./ui";

const PLAN_ORDER: PlanKey[] = ["guest", "normal", "vip"];

function quotaCell(p: PlanLimits): string {
  return p.period === "lifetime" ? `${compact(p.token_limit)} 一次性` : `${compact(p.token_limit)} / 月`;
}

export default function OverviewTab({ usage }: { usage: MyUsage }) {
  const { user, signOutUser } = useAuth();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [typed, setTyped]             = useState("");
  const [deleting, setDeleting]       = useState(false);
  const [deleteError, setDeleteError] = useState("");

  const isGuest  = usage.plan === "guest";
  const isLife   = usage.period === "lifetime";
  const unmetered = !usage.enforced;

  async function deleteAccount() {
    setDeleting(true);
    setDeleteError("");
    try {
      await accountApi.deleteAccount();
      [LS_SESSION, LS_HISTORY, LS_PHASE].forEach((k) => localStorage.removeItem(k));
      await signOutUser();
      window.location.href = "/";
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "刪除失敗，請稍後再試");
      setDeleting(false);
    }
  }

  const planPlans = PLAN_ORDER
    .map((k) => usage.plans.find((p) => p.plan === k))
    .filter((p): p is PlanLimits => !!p);

  return (
    <div>
      {isGuest && (
        <p className="acct-callout">
          你目前是訪客。對話與 Personal Brain 綁在這個瀏覽器的訪客身分上，清除瀏覽資料就找不回來。
          點右上角「登入 / 建立帳號」註冊，現有資料會完整保留，並取得每月額度與慢想模式。
        </p>
      )}
      {!isGuest && !unmetered && usage.exceeded && (
        <p className="acct-callout bad">
          {isLife ? "試用額度已用完。" : `本月額度已用完，將於 ${formatDate(usage.resets_at)} 重置。`}
          {usage.plan === "normal" ? "升級 VIP 可取得更高額度。" : ""}
        </p>
      )}
      {!usage.exceeded && !unmetered && usage.warn && (
        <p className="acct-callout">額度已使用超過 {Math.round(usage.ratio * 100)}%，接近上限時對話會被暫停。</p>
      )}

      <div className="acct-hero">
        <EnsoGauge ratio={usage.ratio} caption={isLife ? "試用額度" : "本月額度"} unmetered={unmetered} />

        <div>
          <p className="acct-plan-name">{PLAN_LABEL[usage.plan]}</p>
          <p className="acct-plan-meta">
            {usage.plan === "vip"
              ? usage.plan_expires_at ? `有效至 ${formatDate(usage.plan_expires_at)}` : "長期有效"
              : isGuest ? "一次性試用，用完不會重置"
              : "額度每月 1 日重置"}
            {usage.is_admin ? "　管理者" : ""}
          </p>

          <dl className="acct-figures">
            <div><dt>已使用</dt><dd>{fmt(usage.used)} <small>tokens</small></dd></div>
            <div><dt>{isLife ? "試用額度" : "本月上限"}</dt><dd>{fmt(usage.limit)} <small>tokens</small></dd></div>
            <div><dt>剩餘</dt><dd>{fmt(usage.remaining)} <small>tokens</small></dd></div>
            <div><dt>{isLife ? "重置" : "下次重置"}</dt><dd>{isLife ? "不重置" : formatDateTime(usage.resets_at)}</dd></div>
          </dl>

          {unmetered && (
            <p className="acct-note" style={{ marginTop: 16 }}>
              目前只記錄用量，尚未啟用上限，因此不會有對話被暫停。
            </p>
          )}
        </div>
      </div>

      <section className="acct-section">
        <h2>方案內容</h2>
        <p className="lede">
          VIP 目前為邀請制，需要請聯絡管理員開通；開通後會立刻套用，不必重新登入。
        </p>
        <div className="acct-plans-wrap">
          <table className="acct-plans">
            <thead>
              <tr>
                <th scope="col"><span className="sr-only">項目</span></th>
                {planPlans.map((p) => (
                  <th key={p.plan} scope="col" className={p.plan === usage.plan ? "is-current" : ""}>
                    {PLAN_LABEL[p.plan]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {([
                ["Token 額度",       (p: PlanLimits) => quotaCell(p)],
                ["慢想模式",         (p: PlanLimits) => (p.walk_allowed ? "可使用" : "需註冊")],
                ["慢想深度上限",     (p: PlanLimits) => (p.walk_allowed ? `${p.walk_max_depth} 步` : "—")],
                ["Personal Brain",   (p: PlanLimits) => `${fmt(p.brain_max_concepts)} 個概念`],
              ] as const).map(([label, cell]) => (
                <tr key={label}>
                  <th scope="row">{label}</th>
                  {planPlans.map((p) => (
                    <td key={p.plan} className={p.plan === usage.plan ? "is-current" : ""}>{cell(p)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="acct-section">
        <h2>帳號</h2>
        <dl className="acct-kv" style={{ marginTop: 14 }}>
          <dt>名稱</dt><dd>{user?.displayName || "—"}</dd>
          <dt>Email</dt><dd>{user?.email || (isGuest ? "訪客沒有 Email" : "—")}</dd>
          <dt>使用者 ID</dt><dd>{user?.uid}</dd>
        </dl>
        <button type="button" className="acct-btn danger" onClick={() => { setTyped(""); setDeleteError(""); setConfirmOpen(true); }}>
          刪除帳號與所有資料
        </button>
      </section>

      {confirmOpen && (
        <div className="acct-modal-back" onClick={() => !deleting && setConfirmOpen(false)}>
          <div className="acct-modal" role="dialog" aria-modal="true" aria-labelledby="del-title" onClick={(e) => e.stopPropagation()}>
            <h3 id="del-title">刪除帳號</h3>
            <p>
              會永久刪除所有歷史對話、Personal Brain 與用量紀錄，並登出所有裝置。此操作無法復原。
              請輸入「刪除」確認。
            </p>
            <input
              className="acct-input"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              placeholder="刪除"
              autoFocus
              disabled={deleting}
            />
            {deleteError && <span className="acct-error">{deleteError}</span>}
            <div className="acct-modal-actions">
              <button type="button" className="acct-btn" onClick={() => setConfirmOpen(false)} disabled={deleting}>取消</button>
              <button type="button" className="acct-btn danger" onClick={deleteAccount} disabled={deleting || typed.trim() !== "刪除"}>
                {deleting ? "刪除中…" : "永久刪除"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
