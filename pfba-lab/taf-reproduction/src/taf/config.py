"""Configuration for streaming TAF / RALCP reproduction runs."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

VALID_POLICIES = ("ralcp_taf", "ralcp+taf", "taf", "ralcp", "waitk")


@dataclass
class StreamingConfig:
    # Policy: ralcp_taf (default; authors' agents/ralcp_taf.py protocol),
    # taf (paper Eq. 7 exact-majority interpretation), ralcp (RALCP baseline,
    # no LLM), waitk (Wait-K-Stride-N baseline).  "ralcp+taf" is a
    # backward-compatible alias for "ralcp_taf".
    policy: str = "ralcp_taf"

    # TAF sampling (paper Sec. 4.2).
    n: int = 10
    horizon: int = 10
    top_k: int = 10
    tau: float = 0.6

    # RALCP (paper Sec. 4.2; De-En beam width 40).  ``ralcp_fraction`` is the
    # authors' --agree-thres: support >= len(candidates) * fraction.
    beam_width: int = 40
    ralcp_fraction: float = 0.5
    # Retained for backward compatibility; the author protocol always votes
    # on word prefixes (no exact-majority gate).
    ralcp_content: bool = False
    # Authors' --min-start: source units required before the first write.
    min_start: int = 1

    # Wait-K-Stride-N (paper Sec. 4.2).
    wait_k: int = 3
    wait_stride: int = 1

    # Decoding / streaming.
    max_new_tokens: int = 64
    initial_read: int = 1
    max_steps: int = 100_000
    seed: int = 13
    src_unit: str = "word"  # word for De/En, char for Zh/Ja
    tgt_unit: str = "word"  # target voting / latency unit; char for Zh/Ja

    # Instrumentation.
    cuda_sync: bool = True

    def __post_init__(self) -> None:
        if self.policy not in VALID_POLICIES:
            raise ValueError(f"policy must be one of {VALID_POLICIES}, got {self.policy!r}")
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if not 0.0 <= self.ralcp_fraction <= 1.0:
            raise ValueError("ralcp_fraction must be in [0, 1]")
        if self.wait_k < 0:
            raise ValueError("wait_k must be non-negative")
        if self.wait_stride <= 0:
            raise ValueError("wait_stride must be positive")
        if self.initial_read < 1:
            raise ValueError("initial_read must be >= 1")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.src_unit not in ("word", "char"):
            raise ValueError("src_unit must be 'word' or 'char'")
        if self.tgt_unit not in ("word", "char"):
            raise ValueError("tgt_unit must be 'word' or 'char'")
        if self.min_start < 1:
            raise ValueError("min_start must be >= 1")

    @property
    def effective_beam_width(self) -> int:
        """Per-prediction MT beam width for RALCP+TAF.

        The paper requires n * beam == RALCP beam width.  For the default
        n=10, beam=4 -> 40.  For n values that do not divide 40 we use floor
        and expose candidate_budget so the deviation is auditable.
        """
        return max(1, self.beam_width // self.n)

    @property
    def candidate_budget(self) -> int:
        return self.n * self.effective_beam_width

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["effective_beam_width"] = self.effective_beam_width
        data["candidate_budget"] = self.candidate_budget
        return data
