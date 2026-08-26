from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


ALLOWED_ACTIONS = Literal[
    "impute_missing",
    "remove_duplicates",
    "convert_to_date",
    "standardize_categories"
]


class CleaningAction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    action: ALLOWED_ACTIONS
    column: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)

class CleaningPlan(BaseModel):
    summary: str
    actions: list[CleaningAction]