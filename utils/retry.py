"""
utils/retry.py — Exponential-backoff retry decorator.

Usage:
    from utils.retry import with_retry

    @with_retry(max_attempts=4, base_delay=2.0, exceptions=(IOError,))
    def fetch_something():
        ...

    # Or as a one-off call wrapper:
    result = with_retry()(my_function)(arg1, arg2)

Retried exception types default to a broad set of transient network /
rate-limit errors common across yfinance, Alpaca, httpx, and requests.
"""
from __future__ import annotations
import time
import functools
from typing import Callable, Tuple, Type
from loguru import logger

# Default exception types that warrant a retry
_DEFAULT_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    IOError,
    OSError,
    TimeoutError,
    ConnectionError,
)

# Lazily extend with library-specific types if the libraries are installed
def _default_exc_tuple() -> Tuple[Type[Exception], ...]:
    extra: list[Type[Exception]] = []
    try:
        from requests.exceptions import RequestException
        extra.append(RequestException)
    except ImportError:
        pass
    try:
        from httpx import HTTPError
        extra.append(HTTPError)
    except ImportError:
        pass
    try:
        from alpaca.common.exceptions import APIError
        extra.append(APIError)
    except ImportError:
        pass
    return _DEFAULT_EXCEPTIONS + tuple(extra)


def with_retry(
    max_attempts: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] | None = None,
) -> Callable:
    """Return a decorator that retries the wrapped function with exponential backoff.

    Parameters
    ----------
    max_attempts : total number of calls (1 = no retry)
    base_delay   : seconds to wait before the first retry
    max_delay    : cap on the wait between retries
    backoff      : multiplier applied to delay after each failure
    exceptions   : exception types to catch; defaults to common network errors
    """
    exc_types = exceptions or _default_exc_tuple()

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exc_types as e:
                    last_exc = e
                    if attempt == max_attempts:
                        break
                    logger.warning(
                        f"{fn.__qualname__} failed (attempt {attempt}/{max_attempts}): "
                        f"{type(e).__name__}: {e}  — retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff, max_delay)
            logger.error(
                f"{fn.__qualname__} failed after {max_attempts} attempts: {last_exc}"
            )
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
