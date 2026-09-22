/**
 * lib/quotaError.ts — 後端結構化錯誤（429 額度超限 / 409 Personal Brain 上限）的解析。
 *
 * 後端 detail 是物件（見 backend core/quota.py 與 api/routes/user_brain.py）：
 *   429 → { code: "quota_exceeded" | "guest_trial_exhausted", message, plan, used, limit,
 *           period, resets_at, upgrade_hint }
 *   409 → { code: "brain_limit", message, plan, limit }
 * 其餘錯誤的 detail 通常是字串。apiErrorFromResponse 把兩種形狀都收進同一個 ApiError，
 * 呼叫端用 asQuotaError / asBrainLimitError 判斷是不是自己要處理的那種。
 */

export type QuotaDetail = {
  code:         "quota_exceeded" | "guest_trial_exhausted";
  message:      string;
  plan:         "guest" | "normal" | "vip";
  used:         number;
  limit:        number;
  period:       "lifetime" | "monthly";
  resets_at:    number | null;     // unix ms；訪客為 null
  upgrade_hint: "register" | "vip" | null;
};

export type BrainLimitDetail = {
  code:    "brain_limit";
  message: string;
  plan:    string;
  limit:   number;
};

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.name   = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export async function apiErrorFromResponse(res: Response): Promise<ApiError> {
  let detail: unknown = null;
  try {
    const body = await res.json();
    detail = body?.detail ?? body;
  } catch {
    /* 非 JSON 回應（例如 proxy 的 502 頁面），保持 null */
  }
  const message =
    typeof detail === "string" ? detail
    : (detail as { message?: string } | null)?.message ?? `API error ${res.status}`;
  return new ApiError(res.status, detail, message);
}

export function asQuotaError(err: unknown): QuotaDetail | null {
  if (!(err instanceof ApiError) || err.status !== 429) return null;
  const d = err.detail as Partial<QuotaDetail> | null;
  return d && (d.code === "quota_exceeded" || d.code === "guest_trial_exhausted")
    ? (d as QuotaDetail)
    : null;
}

export function asBrainLimitError(err: unknown): BrainLimitDetail | null {
  if (!(err instanceof ApiError) || err.status !== 409) return null;
  const d = err.detail as Partial<BrainLimitDetail> | null;
  return d && d.code === "brain_limit" ? (d as BrainLimitDetail) : null;
}
