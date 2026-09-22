"use client";

/**
 * lib/adminApi.ts — 管理後台用的後端呼叫。型別對應 backend api/routes/admin.py。
 * 真正的權限檢查在後端（require_admin）；前端的 is_admin 判斷只是決定要不要顯示頁面。
 */

import { apiFetch } from "./apiFetch";
import { ApiError, apiErrorFromResponse } from "./quotaError";
import type { DailyUsageItem, MyUsage, PlanKey, UsageBucket } from "./accountApi";

const BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");
const JSON_HEADERS = { "Content-Type": "application/json" };

export type AdminIdentity = {
  uid:             string;
  email:           string | null;
  display_name:    string | null;
  is_anonymous:    boolean;
  is_admin:        boolean;
  disabled:        boolean;
  created_at:      number | null;
  last_sign_in_at: number | null;
};

export type AdminUserRow = AdminIdentity & {
  plan:            PlanKey;
  plan_expires_at: number | null;
  month_tokens:    number;
  lifetime_tokens: number;
};

export type AdminUserList = { users: AdminUserRow[]; next_page_token: string | null };

export type AdminUserDetail = {
  identity:        AdminIdentity;
  plan:            PlanKey;
  plan_expires_at: number | null;
  usage:           MyUsage;
};

export type AdminOverview = {
  period:           string;               // "202609"
  total_tokens:     number;
  input_tokens:     number;
  output_tokens:    number;
  reasoning_tokens: number;
  calls:            number;
  turns:            number;
  by_feature:       Record<string, UsageBucket>;
  by_plan:          Record<string, { total_tokens: number; turns: number }>;
  vip_active:       number;
};

export type TopUser = {
  uid: string; email: string | null; display_name: string | null;
  plan: PlanKey; total_tokens: number; turns: number;
};

export type VipRow = {
  uid: string; email: string | null; display_name: string | null;
  plan_expires_at: number | null; expired: boolean; month_tokens: number;
};

export type AuditEntry = {
  id: string; admin_uid: string | null; admin_email: string | null;
  action: string; target_uid: string;
  before: { plan?: string; plan_expires_at?: number | null };
  after:  { plan?: string; plan_expires_at?: number | null };
  at: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await apiFetch(`${BASE}${path}`, init);
  if (!res.ok) throw await apiErrorFromResponse(res);
  return (await res.json()) as T;
}

const qs = (params: Record<string, string | number | boolean | null | undefined>) => {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v !== null && v !== undefined && v !== "" && v !== false) p.set(k, String(v)); });
  const s = p.toString();
  return s ? `?${s}` : "";
};

export const adminApi = {
  overview:      (period?: string) => request<AdminOverview>(`/admin/overview${qs({ period })}`),
  overviewDaily: (days: number)    => request<{ days: DailyUsageItem[] }>(`/admin/overview/daily?days=${days}`).then((r) => r.days),
  topUsers:      (period?: string, limit = 10) =>
    request<{ period: string; users: TopUser[] }>(`/admin/top-users${qs({ period, limit })}`).then((r) => r.users),

  users: (opts: { q?: string; includeGuests?: boolean; pageToken?: string | null; limit?: number }) =>
    request<AdminUserList>(`/admin/users${qs({
      q: opts.q, include_guests: opts.includeGuests, page_token: opts.pageToken, limit: opts.limit,
    })}`),
  user:      (uid: string) => request<AdminUserDetail>(`/admin/users/${encodeURIComponent(uid)}`),
  userDaily: (uid: string, days: number) =>
    request<{ days: DailyUsageItem[] }>(`/admin/users/${encodeURIComponent(uid)}/usage/daily?days=${days}`).then((r) => r.days),
  setPlan: (uid: string, plan: "normal" | "vip", expiresAt: number | null) =>
    request<{ uid: string; plan: string; plan_expires_at: number | null }>(
      `/admin/users/${encodeURIComponent(uid)}/plan`,
      { method: "PUT", headers: JSON_HEADERS, body: JSON.stringify({ plan, expires_at: plan === "vip" ? expiresAt : null }) },
    ),

  vips:  () => request<{ vips: VipRow[] }>("/admin/vips").then((r) => r.vips),
  audit: (opts: { limit?: number; targetUid?: string } = {}) =>
    request<{ entries: AuditEntry[] }>(`/admin/audit${qs({ limit: opts.limit, target_uid: opts.targetUid })}`).then((r) => r.entries),
};

/** 後端 top-users 在缺少 Firestore 索引時回 503 + code=index_required。 */
export function asIndexRequired(err: unknown): string | null {
  if (!(err instanceof ApiError) || err.status !== 503) return null;
  const d = err.detail as { code?: string; message?: string } | null;
  return d?.code === "index_required" ? (d.message ?? "需要建立 Firestore 索引") : null;
}

/** "202609" → 上一個月 "202608"（跨年正確）。 */
export function shiftPeriod(key: string, delta: number): string {
  const y = Number(key.slice(0, 4)), m = Number(key.slice(4, 6)) - 1 + delta;
  const d = new Date(Date.UTC(y, m, 1));
  return `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
}
