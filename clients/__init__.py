"""Image provider API clients."""

import time


def elapsed_ms(started_at: float) -> int:
    """Milliseconds elapsed since a time.monotonic() timestamp, for logging."""
    return int((time.monotonic() - started_at) * 1000)
