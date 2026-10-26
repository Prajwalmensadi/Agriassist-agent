from agriassist.agent import health_check
from agriassist.schemas import AgentDecision


def test_health_check():
    assert health_check() == "AgriAssist scaffold ready"


def test_agent_decision_schema_accepts_permitted_action():
    decision = AgentDecision(
        case_summary="Farmer reports yellowing paddy leaves.",
        missing_information=["growth stage"],
        next_action="ask_question",
    )
    assert decision.next_action == "ask_question"


def test_agent_decision_schema_rejects_unknown_action():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentDecision(
            case_summary="Example",
            next_action="prescribe_pesticide",
        )
