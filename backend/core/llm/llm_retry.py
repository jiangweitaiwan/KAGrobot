from __future__ import annotations

import os
import random
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _extract_status_code(exc: BaseException) -> Optional[int]:
    # Many SDKs expose status in different shapes; probe common ones.
    for attr in ("status_code", "code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val

    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val

    return None


def _looks_retryable_message(msg: str) -> bool:
    m = msg.lower()
    needles = (
        "503",
        "service unavailable",
        "unavailable",
        "server is busy",
        "too many requests",
        "rate limit",
        "temporarily",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection refused",
        "tls",
    )
    return any(n in m for n in needles)


def _is_retryable_exception(exc: BaseException) -> bool:
    code = _extract_status_code(exc)
    if code in (429, 500, 502, 503, 504):
        return True

    # google.api_core.exceptions.ServiceUnavailable / TooManyRequests
    try:
        from google.api_core import exceptions as gexc  # type: ignore

        if isinstance(exc, (gexc.ServiceUnavailable, gexc.TooManyRequests, gexc.InternalServerError)):
            return True
        if isinstance(exc, (gexc.DeadlineExceeded, gexc.Aborted, gexc.ResourceExhausted)):
            return True
    except Exception:
        pass

    return _looks_retryable_message(str(exc))


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: Optional[int] = None,
    initial_backoff_s: Optional[float] = None,
    max_backoff_s: Optional[float] = None,
    jitter_ratio: Optional[float] = None,
) -> T:
    """
    通用 retry 包裝器：針對常見暫時性錯誤（尤其 503/429）做指數退避 + jitter。

    可用環境變數覆寫：
      - LLM_RETRY_MAX_ATTEMPTS (default: 6)
      - LLM_RETRY_INITIAL_BACKOFF_S (default: 0.6)
      - LLM_RETRY_MAX_BACKOFF_S (default: 8.0)
      - LLM_RETRY_JITTER_RATIO (default: 0.25)
    """
    attempts = max_attempts if max_attempts is not None else _env_int("LLM_RETRY_MAX_ATTEMPTS", 6)
    attempts = max(1, attempts)

    base = initial_backoff_s if initial_backoff_s is not None else _env_float("LLM_RETRY_INITIAL_BACKOFF_S", 0.6)
    base = max(0.0, base)

    cap = max_backoff_s if max_backoff_s is not None else _env_float("LLM_RETRY_MAX_BACKOFF_S", 8.0)
    cap = max(0.0, cap)

    jitter = jitter_ratio if jitter_ratio is not None else _env_float("LLM_RETRY_JITTER_RATIO", 0.25)
    jitter = max(0.0, min(1.0, jitter))

    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if i >= attempts - 1 or not _is_retryable_exception(exc):
                raise

            # Exponential backoff with bounded jitter.
            backoff = min(cap, base * (2**i))
            if jitter > 0 and backoff > 0:
                backoff = backoff * (1 - jitter) + random.random() * backoff * (2 * jitter)
            time.sleep(backoff)

    # Unreachable in practice; keep mypy happy.
    assert last_exc is not None
    raise last_exc

