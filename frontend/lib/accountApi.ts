"use client";

/**
 * lib/accountApi.ts — 會員中心用的後端呼叫（一律經 apiFetch 帶登入憑證）。
 * 型別對應 backend api/routes/account.py、users.py、user_brain.py。
 */

import { apiFetch } from "./apiFetch";
import { apiErrorFromResponse } from "./quotaError";
import type { PersonalBrainConcept } from "../app/components/DetailPanel";
import type { BertopicLayoutData } from "../app/components/BertopicMap";

const BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");

export type PlanKey = "guest" | "normal" | "vip";

export type UsageBucket = {
  total_tokens:     number;
  input_tokens:     number;
  output_tokens:    number;
  reasoning_tokens: number;
  calls:            number;
};

export type UsageBreakdown = UsageBucket & {
  turns:      number;
  by_feature: Record<string, UsageBucket>;
};

export type PlanLimits = {
  plan:               PlanKey;
  period:             "lifetime" | "monthly";
  token_limit:        number;
  walk_allowed:       boolean;
  walk_max_depth:     number;
  brain_max_concepts: number;
};

export type MyUsage = {
  plan:            PlanKey;
  plan_expires_at: number | null;
  is_anonymous:    boolean;
  is_admin:        boolean;
  enforced:        boolean;          // false = 目前只計量、尚未限額
  period:          "lifetime" | "monthly";
  period_key:      string | null;
  resets_at:       number | null;
  used:            number;
  limit:           number;
  remaining:       number;
  ratio:           number;
  warn:            boolean;
  exceeded:        boolean;
  breakdown:       UsageBreakdown;
  limits:          PlanLimits;       // 目前實際生效
  plans:           PlanLimits[];     // 各等級名義方案
};

export type DailyUsageItem = UsageBucket & { date: string; turns: number };

export type SessionSummary = {
  session_id:   string;
  title:        string;
  phase:        string | null;
  summary:      string;
  updated_at:   number | null;
  created_at:   number | null;
  total_tokens: number;
};

export type Profile = {
  uid:             string;
  email:           string | null;
  display_name:    string | null;
  is_anonymous:    boolean;
  provider:        string;
  plan:            PlanKey;
  plan_expires_at: number | null;
  created_at:      number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await apiFetch(`${BASE}${path}`, init);
  if (!res.ok) throw await apiErrorFromResponse(res);
  return (await res.json()) as T;
}

const JSON_HEADERS = { "Content-Type": "application/json" };

export const accountApi = {
  usage:      ()                  => request<MyUsage>("/me/usage"),
  dailyUsage: (days: number)      => request<{ days: DailyUsageItem[] }>(`/me/usage/daily?days=${days}`).then((r) => r.days),
  profile:    ()                  => request<Profile>("/me"),
  setDisplayName: (display_name: string) =>
    request<Profile>("/me", { method: "PATCH", headers: JSON_HEADERS, body: JSON.stringify({ display_name }) }),
  deleteAccount: ()               => request<{ ok: boolean }>("/me", { method: "DELETE" }),

  sessions:      ()               => request<{ sessions: SessionSummary[] }>("/me/sessions?limit=100").then((r) => r.sessions),
  renameSession: (sid: string, title: string) =>
    request<{ ok: boolean }>(`/me/sessions/${encodeURIComponent(sid)}`, {
      method: "PATCH", headers: JSON_HEADERS, body: JSON.stringify({ title }),
    }),
  deleteSession: (sid: string)    =>
    request<{ ok: boolean }>(`/me/sessions/${encodeURIComponent(sid)}`, { method: "DELETE" }),

  brain:       (uid: string)      =>
    request<{ concepts: PersonalBrainConcept[]; updatedAt: number }>(`/user/${uid}/brain`).then((r) => r.concepts),
  setBrainWeight: (uid: string, eid: string, weight: number) =>
    request(`/user/${uid}/brain/concepts/${encodeURIComponent(eid)}`, {
      method: "PATCH", headers: JSON_HEADERS, body: JSON.stringify({ weight }),
    }),
  removeBrainConcept: (uid: string, eid: string) =>
    request(`/user/${uid}/brain/concepts/${encodeURIComponent(eid)}`, { method: "DELETE" }),
  bertopicLayout: ()              => request<BertopicLayoutData>("/graph/bertopic-layout"),
};
