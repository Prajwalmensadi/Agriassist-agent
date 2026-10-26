from typing import Literal
from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    """Validated proposal from the model; application code must enforce policy."""

    case_summary: str = Field(description="Brief summary of farmer-reported facts.")
    missing_information: list[str] = Field(default_factory=list)
    next_action: Literal[
        "ask_question",
        "search_knowledge_base",
        "generate_advisory",
        "request_expert_review",
        "record_farmer_action",
        "schedule_follow_up",
        "close_case",
    ]
    evidence_ids: list[str] = Field(default_factory=list)
    uncertainty: str = ""
    expert_review_required: bool = False
