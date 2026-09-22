"use client";

import { useCallback, useEffect, useState } from "react";
import { adminApi, type AdminUserDetail, type AuditEntry } from "../../../lib/adminApi";
import type { DailyUsageItem } from "../../../lib/accountApi";
import DailyBars from "../../account/components/DailyBars";
import { ConfirmButton, FEATURE_LABEL, PLAN_LABEL, fmt, formatDateTime } from "../../account/components/ui";
import { PlanChip, daysLeft, describePlan, endOfLocalDay, extendedExpiry, toDateInput, userLabel } from "./adminUi";

export default function UserDetail({
  uid, onClose, onChanged,
}: { uid: string; onClose: () => void; onChanged: () => void }) {
  const [detail, setDetail] = useState<AdminUserDetail | null>(null);
  const [daily, setDaily]   = useState<DailyUsageItem[] | null>(null);
  const [audit, setAudit]   = useState<AuditEntry[] | null>(null);
  const [error, setError]   = useState("");

  // 等級編輯表單
  const [planChoice, setPlanChoice] = useState<"normal" | "vip">("normal");
  const [forever, setForever]       = useState(true);
  const [dateStr, setDateStr]       = useState("");
  const [saving, setSaving]         = useState(false);
  const [saveError, setSaveError]   = useState("");
  const [saved, setSaved]           = useState(false);

  const load = useCallback(async () => {
    try {
      const d = await adminApi.user(uid);
      setDetail(d);
      setPlanChoice(d.plan === "vip" ? "vip" : "normal");
      setForever(!(d.plan === "vip" && d.plan_expires_at));
      setDateStr(d.plan === "vip" && d.plan_expires_at ? toDateInput(d.plan_expires_at) : "");
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "讀取使用者失敗");
    }
    adminApi.audit({ targetUid: uid, limit: 10 }).then(setAudit).catch(() => setAudit([]));
  }, [uid]);

  useEffect(() => {
    setDetail(null); setDaily(null); setAudit(null); setSaved(false); setSaveError("");
    load();
    adminApi.userDaily(uid, 30).then(setDaily).catch(() => setDaily([]));
  }, [uid, load]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const isGuest    = detail?.identity.is_anonymous ?? false;
  const currentExp = detail?.plan === "vip" ? detail.plan_expires_at : null;
  const newExp     = planChoice === "vip" && !forever ? endOfLocalDay(dateStr) : null;

  let formError = "";
  if (planChoice === "vip" && !forever) {
    if (newExp === null) formError = "請選擇到期日。";
    else if (newExp <= Date.now()) formError = "到期日必須在未來。";
  }
  const unchanged = !!detail && (
    planChoice === "normal" ? detail.plan === "normal"
    : detail.plan === "vip" && (forever ? currentExp === null : newExp !== null && toDateInput(newExp) === (currentExp ? toDateInput(currentExp) : ""))
  );

  function quickExtend(days: number) {
    setPlanChoice("vip"); setForever(false);
    setDateStr(toDateInput(extendedExpiry(currentExp, days)));
    setSaved(false);
  }

  async function save() {
    if (formError || !detail) return;
    setSaving(true); setSaveError(""); setSaved(false);
    try {
      await adminApi.setPlan(uid, planChoice, planChoice === "vip" ? (forever ? null : newExp) : null);
      await load();
      setSaved(true);
      onChanged();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "儲存失敗");
    } finally {
      setSaving(false);
    }
  }

  const u = detail?.usage;
  const features = u ? Object.entries(u.breakdown.by_feature).filter(([, v]) => v.total_tokens > 0).sort((a, b) => b[1].total_tokens - a[1].total_tokens) : [];
  const featureMax = features[0]?.[1].total_tokens ?? 1;
  const meterState = !u ? "" : !u.enforced ? "" : u.exceeded ? " bad" : u.warn ? " warn" : "";

  return (
    <div className="adm-drawer-back" onClick={onClose}>
      <aside className="adm-drawer" role="dialog" aria-modal="true" aria-label="使用者詳情" onClick={(e) => e.stopPropagation()}>
        <div className="adm-drawer-head">
          <div style={{ minWidth: 0 }}>
            <div className="adm-drawer-title">{detail ? userLabel({ ...detail.identity }) : "載入中…"}</div>
            {detail && <div className="adm-sub" style={{ marginLeft: 0 }}>{detail.identity.uid}</div>}
          </div>
          <button type="button" className="acct-btn quiet" onClick={onClose} aria-label="關閉">✕</button>
        </div>

        {error && <p className="acct-error">{error}</p>}
        {!detail && !error && <div className="acct-loading">載入中…</div>}

        {detail && u && (
          <>
            <dl className="acct-kv">
              <dt>目前等級</dt><dd><PlanChip plan={detail.plan} />{detail.plan === "vip" && (
                <span className="adm-sub">{detail.plan_expires_at ? `${new Date(detail.plan_expires_at).toLocaleDateString("zh-TW")}（剩 ${Math.max(0, daysLeft(detail.plan_expires_at))} 天）` : "不過期"}</span>
              )}</dd>
              <dt>Email</dt><dd>{detail.identity.email ?? "—"}</dd>
              <dt>註冊</dt><dd>{formatDateTime(detail.identity.created_at)}</dd>
              <dt>最近登入</dt><dd>{formatDateTime(detail.identity.last_sign_in_at)}</dd>
              {detail.identity.is_admin && (<><dt>身分</dt><dd><span className="adm-chip admin">管理者</span></dd></>)}
            </dl>

            <section className="adm-block">
              <h3>會員等級</h3>
              {isGuest ? (
                <p className="acct-note">訪客不能設定等級，需要對方先註冊帳號。</p>
              ) : (
                <>
                  <div className="adm-radio" role="radiogroup" aria-label="會員等級">
                    {(["normal", "vip"] as const).map((p) => (
                      <label key={p} className={planChoice === p ? "on" : ""}>
                        <input type="radio" name="plan" checked={planChoice === p} onChange={() => { setPlanChoice(p); setSaved(false); }} />
                        {PLAN_LABEL[p]}
                      </label>
                    ))}
                  </div>

                  {planChoice === "vip" && (
                    <div className="adm-expiry">
                      <label className="adm-check">
                        <input type="checkbox" checked={forever} onChange={(e) => { setForever(e.target.checked); setSaved(false); }} />
                        不過期
                      </label>
                      {!forever && (
                        <>
                          <input type="date" className="acct-input" value={dateStr} aria-label="VIP 到期日"
                            onChange={(e) => { setDateStr(e.target.value); setSaved(false); }} />
                          <div className="adm-quick">
                            {[30, 90, 365].map((d) => (
                              <button key={d} type="button" className="acct-btn quiet" onClick={() => quickExtend(d)}>
                                +{d === 365 ? "1 年" : `${d} 天`}
                              </button>
                            ))}
                          </div>
                          <span className="acct-note">延長從「目前到期日」與「今天」較晚者起算。</span>
                        </>
                      )}
                    </div>
                  )}

                  {(formError || saveError) && <p className="acct-error">{formError || saveError}</p>}
                  {saved && !saveError && <p className="acct-note" style={{ color: "var(--teal)" }}>已儲存，立即生效。</p>}

                  <div style={{ marginTop: 10 }}>
                    <ConfirmButton
                      label="儲存等級變更"
                      confirmLabel={`確定改為 ${describePlan(planChoice, planChoice === "vip" ? (forever ? null : newExp) : null)}`}
                      className="acct-btn primary"
                      disabled={saving || unchanged || !!formError}
                      onConfirm={save}
                    />
                  </div>
                </>
              )}
            </section>

            <section className="adm-block">
              <h3>用量</h3>
              <div className={`acct-meter${meterState}`} aria-hidden="true"><span style={{ width: `${Math.min(100, u.ratio * 100)}%` }} /></div>
              <p className="acct-note" style={{ margin: "6px 0 10px" }}>
                {u.period === "lifetime" ? "試用累計" : "本月"}已用 {fmt(u.used)} / {fmt(u.limit)}（{Math.round(u.ratio * 100)}%）
                {!u.enforced && "，目前未啟用限額"}
              </p>
              {features.map(([name, v]) => (
                <div className="acct-feature" key={name}>
                  <span className="name">{FEATURE_LABEL[name] ?? name}</span>
                  <div className="acct-meter" aria-hidden="true"><span style={{ width: `${(v.total_tokens / featureMax) * 100}%` }} /></div>
                  <span className="val">{fmt(v.total_tokens)}</span>
                </div>
              ))}
              <div style={{ marginTop: 12 }}>
                {daily ? <DailyBars days={daily} /> : <div className="acct-loading">載入中…</div>}
              </div>
            </section>

            <section className="adm-block">
              <h3>這位使用者的操作紀錄</h3>
              {audit === null ? <div className="acct-loading">載入中…</div>
                : audit.length === 0 ? <p className="acct-note">沒有紀錄。</p>
                : (
                  <ul className="adm-audit">
                    {audit.map((e) => (
                      <li key={e.id}>
                        <span>{describePlan(e.before.plan, e.before.plan_expires_at)} → {describePlan(e.after.plan, e.after.plan_expires_at)}</span>
                        <span className="adm-sub">{formatDateTime(e.at)}　{e.admin_email ?? e.admin_uid}</span>
                      </li>
                    ))}
                  </ul>
                )}
            </section>
          </>
        )}
      </aside>
    </div>
  );
}
