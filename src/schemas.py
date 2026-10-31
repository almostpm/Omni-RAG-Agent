"""schemas.py - Pydantic v2 models for Phase 2 (grounded QA with citations).

LLMAnswer / LLMSource : what the LLM must emit (validated).
Source / QAResult     : what the pipeline returns (citations key on (source, page)).
"""
from __future__ import annotations

import re
import sys
from typing import List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

Confidence = Literal["high", "medium", "low"]


class LLMSource(BaseModel):
    """A citation as emitted by the LLM. No chunk_id: the model must not invent ids."""

    # [NOOK-P2-11-FIX] Pydantic v2 only: ConfigDict, field_validator, model_validate.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    source: str = Field(..., min_length=1,
                        description="Exact PDF filename from the passage header, e.g. report.pdf")
    page: int = Field(..., ge=1,
                      description="1-indexed page number from the passage header")

    @field_validator("page", mode="before")
    @classmethod
    def _coerce_page(cls, v: object) -> object:
        """Accept 3, 3.0, '3', 'p.3', 'page 3' -> 3."""
        if isinstance(v, bool):
            raise ValueError("page must be an integer, not a bool")
        if isinstance(v, str):
            digits = re.sub(r"[^0-9]", "", v)
            if not digits:
                raise ValueError(f"cannot parse page from {v!r}")
            return int(digits)
        return v


class LLMAnswer(BaseModel):
    """Structured answer the LLM must return as JSON."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    answer: str = Field(..., min_length=1,
                        description="Concise answer grounded ONLY in the provided context")
    answerable: bool = Field(True,
                             description="false if the context does not contain the answer")
    confidence: Confidence = Field("low", description="high | medium | low")
    sources: List[LLMSource] = Field(default_factory=list,
                                     description="Only passages actually used; [] if not answerable")

    @field_validator("confidence", mode="before")
    @classmethod
    def _norm_conf(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("answer")
    @classmethod
    def _answer_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer must not be blank")
        return v

    @model_validator(mode="after")
    def _no_sources_when_unanswerable(self) -> "LLMAnswer":
        # [NOOK-P2-13-FIX] "not found" answers must never carry citations.
        if not self.answerable:
            self.sources = []
        return self


class Source(BaseModel):
    """A validated citation. Identity key is (source, page); chunk_id is best-effort."""

    model_config = ConfigDict(extra="ignore")

    source: str
    page: int = Field(..., ge=1)
    chunk_id: Optional[str] = None  # [NOOK-P2-5-FIX] optional: retrieve() does not return it


class QAResult(BaseModel):
    """Final pipeline output."""

    model_config = ConfigDict(extra="ignore")

    question: str
    answer: str
    answerable: bool
    used_context: bool                      # False => LLM was never called (score gate)
    confidence: Confidence = "low"
    sources: List[Source] = Field(default_factory=list)
    top_score: Optional[float] = None
    threshold: float
    num_retrieved: int = 0
    fallback_raw: bool = False              # True => LLM JSON failed twice, raw text returned
    warnings: List[str] = Field(default_factory=list)


if __name__ == "__main__":
    a = LLMAnswer.model_validate({
        "answer": "Demo answer.",
        "confidence": "HIGH",
        "sources": [{"source": "a.pdf", "page": "p.3"}],
    })
    assert a.confidence == "high" and a.sources[0].page == 3
    u = LLMAnswer.model_validate({"answer": "Not found.", "answerable": False,
                                  "sources": [{"source": "a.pdf", "page": 1}]})
    assert u.sources == []
    try:
        LLMAnswer.model_validate({"answer": "   ", "sources": []})
        print("FAIL: blank answer accepted", file=sys.stderr)
        sys.exit(1)
    except ValidationError:
        pass
    print("schemas OK:", a.model_dump_json())
    sys.exit(0)