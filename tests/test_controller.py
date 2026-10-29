import json
from types import SimpleNamespace

import pytest

import pytest

from agriassist.controller import ConversationState, handle_message
from agriassist.knowledge import RetrievedChunk


def chunk(
    text="Rice sheath blight is described as lesions on the leaf sheath.",
    *,
    filename="8.pdf",
    page=2,
    geography="Odisha; applicability to coastal Karnataka unverified.",
):
    return RetrievedChunk(
        text=text,
        source_filename=filename,
        page_number=page,
        metadata={
            "source_title": "Rice sheath blight article",
            "topic_category": "rice_disease",
            "geographic_applicability": geography,
            "verification_status": "provisional; authority unverified",
        },
        distance=0.21,
    )


class Retriever:
    def __init__(self, results=None, error=None):
        self.results = list(results or [])
        self.error = error
        self.calls = []

    def __call__(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error:
            raise self.error
        return self.results


class Ollama:
    def __init__(self, answer="The article describes lesions on leaf sheaths."):
        self.answer = answer
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=self.answer))


def test_supported_paddy_question_retrieves_and_answers_with_geographic_caveat():
    retriever = Retriever([chunk()])
    model = Ollama()
    response = handle_message(
        "What symptoms are described for paddy sheath blight?",
        retriever=retriever,
        ollama_client=model,
    )
    assert response.state == "answered"
    assert response.evidence == (chunk(),)
    assert "not locally validated for coastal Karnataka" in response.answer
    assert response.answer.count("coastal Karnataka") == 1
    assert response.evidence[0].source_filename == "8.pdf"
    assert response.evidence[0].page_number == 2
    assert response.evidence[0].distance == 0.21
    assert response.evidence[0].metadata["verification_status"] == "provisional; authority unverified"
    assert len(retriever.calls) == len(model.calls) == 1


def test_existing_geographic_warning_is_not_repeated():
    model = Ollama("This Odisha study is not locally validated for coastal Karnataka.")
    response = handle_message(
        "What symptoms are described for paddy sheath blight?",
        retriever=Retriever([chunk()]),
        ollama_client=model,
    )
    assert response.answer.count("coastal Karnataka") == 1


def test_unsupported_crop_is_rejected_before_retrieval():
    retriever = Retriever([chunk()])
    response = handle_message("How do I treat tomato disease?", retriever=retriever)
    assert response.state == "unsupported_scope"
    assert "paddy (rice) only" in response.answer
    assert not retriever.calls


def test_pending_crop_clarification_resumes_after_confirmation():
    retriever = Retriever([chunk()])
    model = Ollama()
    first = handle_message(
        "What symptoms of sheath blight are described?", retriever=retriever, ollama_client=model
    )
    original_question = first.conversation.pending_question
    second = handle_message(
        "yes", conversation=first.conversation, retriever=retriever, ollama_client=model
    )
    assert first.state == "clarification_required"
    assert original_question == "What symptoms of sheath blight are described?"
    assert second.state == "answered"
    assert second.conversation.pending_question is None
    assert retriever.calls == [
        ("What symptoms of sheath blight are described?", {"top_k": 4, "crop": "paddy"})
    ]


def test_negative_confirmation_does_not_retrieve_or_clear_question():
    first = handle_message("What does sheath blight look like?", retriever=Retriever())
    declined = handle_message("no", conversation=first.conversation, retriever=Retriever())
    assert declined.state == "clarification_required"
    assert declined.conversation.pending_question == first.conversation.pending_question
    assert "Which crop did you mean?" in declined.answer


def test_new_question_replaces_pending_question_and_clears_stale_confirmation():
    retriever = Retriever([chunk("Rice micronutrient deficiency symptoms include chlorosis.", filename="11-14.pdf", page=2)])
    model = Ollama("The passage describes chlorosis.")
    first = handle_message("What does sheath blight look like?", retriever=retriever)
    assert first.state == "clarification_required"
    next_turn = handle_message(
        "What micronutrient symptoms are described for paddy?",
        conversation=first.conversation,
        retriever=retriever,
        ollama_client=model,
    )
    assert next_turn.state == "answered"
    assert next_turn.conversation.pending_question is None
    assert "sheath blight" not in retriever.calls[-1][0]


def test_field_symptom_report_asks_for_missing_distribution_first():
    response = handle_message("My paddy leaves are yellow.", retriever=Retriever())
    assert response.state == "clarification_required"
    assert "whole field" in response.answer
    assert response.conversation.pending_kind == "symptom_context"


def test_symptom_context_reply_resumes_original_question():
    retriever = Retriever([chunk("Yellow symptoms appear in patches in the field.")])
    model = Ollama("The source mentions symptoms in patches.")
    first = handle_message("My paddy leaves are yellow.", retriever=retriever)
    resumed = handle_message(
        "patches", conversation=first.conversation, retriever=retriever, ollama_client=model
    )
    assert resumed.state == "answered"
    assert "Farmer-reported symptom distribution: patches" in retriever.calls[0][0]


def test_new_question_during_symptom_clarification_drops_pending_context():
    retriever = Retriever([chunk("Rice sheath blight field lesions are described in the article.")])
    model = Ollama("The article describes field lesions.")
    first = handle_message("My paddy leaves are yellow.", retriever=retriever)
    next_turn = handle_message(
        "What does the sheath blight article say?",
        conversation=first.conversation,
        retriever=retriever,
        ollama_client=model,
    )
    assert next_turn.state == "answered"
    assert next_turn.conversation.pending_question is None
    assert "yellow" not in retriever.calls[-1][0].casefold()


def test_empty_retrieval_is_insufficient_evidence_and_skips_model():
    model = Ollama()
    response = handle_message(
        "What are the symptoms of sheath blight in paddy?",
        retriever=Retriever(),
        ollama_client=model,
    )
    assert response.state == "insufficient_evidence"
    assert not model.calls


def test_retrieval_exception_returns_safe_error():
    response = handle_message(
        "What are the symptoms of sheath blight in paddy?",
        retriever=Retriever(error=RuntimeError("internal detail")),
    )
    assert response.state == "error"
    assert "internal detail" not in response.answer


def test_irrelevant_results_are_not_used_as_answer_evidence():
    model = Ollama()
    irrelevant = chunk("Iron toxicity is associated with waterlogged fields.", filename="11-14.pdf")
    irrelevant = RetrievedChunk(
        text=irrelevant.text,
        source_filename=irrelevant.source_filename,
        page_number=irrelevant.page_number,
        metadata={**irrelevant.metadata, "source_title": "Rice micronutrient article"},
        distance=irrelevant.distance,
    )
    response = handle_message(
        "What does sheath blight look like in paddy?",
        retriever=Retriever([irrelevant]),
        ollama_client=model,
    )
    assert response.state == "insufficient_evidence"
    assert not model.calls


def test_generation_uses_mocked_ollama_and_retrieved_passages_only():
    model = Ollama("The article reports sheath lesions.")
    response = handle_message(
        "What symptoms are described for paddy sheath blight?",
        retriever=Retriever([chunk()]),
        ollama_client=model,
    )
    assert response.state == "answered"
    assert model.calls[0]["model"] == "qwen2.5:3b"
    assert "leaf sheath" in model.calls[0]["messages"][1]["content"]


def test_fabricated_model_citation_is_rejected():
    response = handle_message(
        "What symptoms are described for paddy sheath blight?",
        retriever=Retriever([chunk()]),
        ollama_client=Ollama("The answer is in other.pdf page 8."),
    )
    assert response.state == "insufficient_evidence"
    assert "could not be verified" in response.answer


def test_high_risk_query_is_conservatively_refused_without_generation():
    model = Ollama()
    response = handle_message(
        "What pesticide dose should I apply to paddy?",
        retriever=Retriever([chunk()]),
        ollama_client=model,
    )
    assert response.state == "insufficient_evidence"
    assert "extension officer" in response.answer
    assert len(response.answer.split()) > 20
    assert not model.calls


def test_follow_up_uses_recent_conversation_context():
    retriever = Retriever([chunk("The field pattern is described as a bird's nest.")])
    model = Ollama("The cited text describes a bird's nest field pattern.")
    first = handle_message(
        "What symptoms are described for paddy sheath blight?",
        retriever=Retriever([chunk()]),
        ollama_client=Ollama(),
    )
    second = handle_message(
        "What field pattern does it have?",
        conversation=first.conversation,
        retriever=retriever,
        ollama_client=model,
    )
    assert second.state == "answered"
    assert "sheath blight" in retriever.calls[0][0].casefold()
    assert "recent_conversation_context" in model.calls[0]["messages"][1]["content"]


def _answered_first_turn():
    retriever = Retriever([chunk()])
    model = Ollama("The document describes lesions on the leaf sheath.")
    clarification = handle_message(
        "What symptoms of sheath blight are described in the available documents?",
        retriever=retriever,
        ollama_client=model,
    )
    return handle_message(
        "yes", conversation=clarification.conversation,
        retriever=retriever, ollama_client=model,
    ), retriever, model


@pytest.mark.parametrize(
    "question",
    [
        "Is this information validated for coastal Karnataka?",
        "Is it proven locally?",
    ],
)
def test_geographic_validation_follow_up_answers_locality_first_without_repeating_symptoms(question):
    first, retriever, model = _answered_first_turn()
    call_count = len(model.calls)
    response = handle_message(
        question,
        conversation=first.conversation,
        retriever=retriever,
        ollama_client=model,
    )
    assert response.state == "answered"
    assert response.answer.startswith("The available metadata does not establish")
    assert "coastal Karnataka" in response.answer
    assert "lesions" not in response.answer
    assert len(model.calls) == call_count
    assert response.evidence[0].source_filename == "8.pdf"
    assert response.evidence[0].page_number == 2
    assert response.evidence[0].metadata["geographic_applicability"].startswith("Odisha")


def test_cause_follow_up_focuses_on_cause_from_evidence():
    evidence = chunk("Sheath blight is caused by Rhizoctonia solani.")
    retriever = Retriever([evidence])
    first = handle_message(
        "What symptoms of sheath blight are described for paddy?",
        retriever=Retriever([chunk()]),
        ollama_client=Ollama("The source describes lesions on the leaf sheath."),
    )
    cause_model = Ollama("The document identifies Rhizoctonia solani as the cause.")
    response = handle_message(
        "What causes it?",
        conversation=first.conversation,
        retriever=retriever,
        ollama_client=cause_model,
    )
    payload = json.loads(cause_model.calls[0]["messages"][1]["content"])
    assert response.state == "answered"
    assert response.answer.startswith("The document identifies")
    assert payload["farmer_question"] == "What causes it?"
    assert "Rhizoctonia solani" in payload["retrieved_passages"][0]["text"]
    assert payload["recent_conversation_context"]


def test_simpler_wording_follow_up_uses_prior_answer_without_repeating_it():
    first, _, _ = _answered_first_turn()
    model = Ollama("The fungus causes marks on the plant's leaf sheath.")
    response = handle_message(
        "Explain that more simply.",
        conversation=first.conversation,
        retriever=Retriever([chunk("Sheath blight symptoms include lesions on rice leaf sheaths.")]),
        ollama_client=model,
    )
    assert response.state == "answered"
    assert response.answer.startswith("The fungus causes marks on the plant's leaf sheath.")
    assert response.answer.count("coastal Karnataka") == 1
    assert "Explain that more simply" in model.calls[0]["messages"][1]["content"]
    assert "do not repeat an earlier answer" in model.calls[0]["messages"][0]["content"]
    assert "leaf sheath" in model.calls[0]["messages"][1]["content"]


def test_invalid_retrieved_provenance_is_not_presented():
    response = handle_message(
        "What symptoms are described for paddy sheath blight?",
        retriever=Retriever([chunk(filename="unlisted.pdf")]),
        ollama_client=Ollama(),
    )
    assert response.state == "insufficient_evidence"
    assert "valid source and page provenance" in response.answer


@pytest.mark.parametrize("message", ["", "  "])
def test_empty_user_turn_requests_clarification(message):
    assert handle_message(message).state == "clarification_required"
