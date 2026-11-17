"""Strict binary judgment admission, independent of a caller's JSON Schema."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class BinaryJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    extracted_final_answer: str = Field(min_length=1, pattern=r"[^\s\x1c-\x1f]")
    reasoning: str = Field(min_length=1, pattern=r"[^\s\x1c-\x1f]")
    correct: Literal["yes", "no"]
    confidence: StrictInt = Field(ge=0, le=100)


def score(value: dict[str, Any]) -> tuple[float, bool]:
    """Reject invalid or extra fields before assigning an exact yes/no score."""
    judgment = BinaryJudgment.model_validate(value)
    correct = judgment.correct == "yes"
    return float(correct), correct
