"""Independent TAF / RALCP reproduction for De-En (NAACL 2025)."""

__version__ = "0.1.0"

from .config import StreamingConfig
from .laal import average_lagging, laal, laal_raw, source_completion_cutoff
from .majority import exact_majority, extend_only
from .policy import (
    AuthorRALCPTAFPolicy,
    RALCPPolicy,
    TAFPolicy,
    WaitKPolicy,
    build_policy,
)
from .ralcp import author_ralcp, relaxed_longest_common_prefix

__all__ = [
    "StreamingConfig",
    "AuthorRALCPTAFPolicy",
    "TAFPolicy",
    "RALCPPolicy",
    "WaitKPolicy",
    "build_policy",
    "exact_majority",
    "extend_only",
    "author_ralcp",
    "relaxed_longest_common_prefix",
    "laal",
    "laal_raw",
    "average_lagging",
    "source_completion_cutoff",
    "__version__",
]
