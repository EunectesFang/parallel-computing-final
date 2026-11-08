"""Optional COMET adapter.

Paper footnote 2 points to ``Unbabel/XCOMET-XXL``.  COMET is never simulated:
if the package or checkpoint is unavailable, the run records ``score: null``
plus a machine-readable reason.
"""
from __future__ import annotations

from statistics import mean
from typing import Sequence


class CometScorer:
    def __init__(
        self,
        model=None,
        model_name: str | None = None,
        status: str = "unavailable",
        reason: str = "",
        gpus: int = 0,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.status = status
        self.reason = reason
        self.gpus = gpus

    @classmethod
    def unavailable(cls, reason: str, model_name: str | None = None) -> "CometScorer":
        return cls(model=None, model_name=model_name, status="unavailable", reason=reason)

    @classmethod
    def load(cls, model_name: str | None, device: str | None = None) -> "CometScorer":
        if not model_name:
            return cls.unavailable("no --comet-model specified; COMET disabled")
        gpus = 0
        try:
            import torch  # type: ignore

            if device in (None, "auto"):
                gpus = 1 if torch.cuda.is_available() else 0
            elif str(device).startswith("cuda"):
                gpus = 1
        except Exception:
            gpus = 0
        try:
            from comet import download_model, load_from_checkpoint  # type: ignore

            checkpoint = download_model(model_name)
            model = load_from_checkpoint(checkpoint)
            return cls(
                model=model,
                model_name=model_name,
                status="loaded",
                reason="",
                gpus=gpus,
            )
        except Exception as exc:  # noqa: BLE001 - report, never hide
            return cls.unavailable(
                f"COMET could not be loaded ({type(exc).__name__}: {exc})", model_name
            )

    @property
    def kind(self) -> str:
        if self.model_name is None:
            return "none"
        if "XCOMET" in self.model_name or "xcomet" in self.model_name:
            return "paper-cited (XCOMET-XXL)"
        return "substitution"

    def score(
        self,
        sources: Sequence[str],
        hypotheses: Sequence[str],
        references: Sequence[str],
    ) -> dict:
        if self.model is None:
            return {
                "score": None,
                "status": self.status,
                "reason": self.reason,
                "model": self.model_name,
                "kind": self.kind,
            }
        if not (len(sources) == len(hypotheses) == len(references)):
            return {
                "score": None,
                "status": "error",
                "reason": "source/hypothesis/reference lengths differ",
                "model": self.model_name,
                "kind": self.kind,
            }
        try:
            data = [
                {"src": s, "mt": h, "ref": r}
                for s, h, r in zip(sources, hypotheses, references)
            ]
            output = self.model.predict(data, batch_size=8, gpus=self.gpus)
            scores = output["scores"] if isinstance(output, dict) else output.scores
            return {
                "score": float(mean(float(x) for x in scores)),
                "status": "ok",
                "n": len(scores),
                "model": self.model_name,
                "kind": self.kind,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "score": None,
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "model": self.model_name,
                "kind": self.kind,
            }
