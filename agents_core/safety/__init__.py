"""Safety gates: HITL confirmation + rate limits (research §9 rules 6)."""

from .confirmation import ConfirmationRequired, require_confirmation
from .rate_limits import Limit, RateLimitExceeded, RateLimits

__all__ = [
    "ConfirmationRequired",
    "require_confirmation",
    "Limit",
    "RateLimits",
    "RateLimitExceeded",
]
