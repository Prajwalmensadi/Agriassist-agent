from types import SimpleNamespace

import pytest

from agriassist.chat import (
    ChatServiceError,
    LLM_MODEL,
    _system_prompt,
    answer_question,
)
from agriassist.knowledge import RetrievedChunk


def evidence_chunk(text="The passage describes paddy leaf symptoms."):
    return RetrievedChunk(
        text=text,
        source_filename="8.pdf",
        page_number=2,
        metadata={
            "source_title": "Rice sheath blight article",
            "geographic_applicability": "Odisha; applicability to coastal Karnataka unverified.",
            "verification_status": "provisional; authority unverified",
            "crop_scope": "paddy",
        },
        distance=0.2,
    )


class StubRetriever:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def __call__(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.results


class StubOllama:
    def __init__(self, answer="The passage says this. "):
        self.answer = answer
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=self.answer))


def test_unclear_crop_asks_before_retrieval_or_generation():
    retriever = StubRetriever([evidence_chunk()])
    model = StubOllama()

    result = answer_question("What should I do about yellow leaves?", retriever=retriever, ollama_client=model)

    assert result.status == "clarification"
    assert "paddy (rice)" in result.answer
    assert retriever.calls == []
    assert model.calls == []


def test_pending_paddy_confirmation_resumes_original_question_after_yes():
    original = "What symptoms of sheath blight are described in the available documents?"
    retriever = StubRetriever([evidence_chunk("Sheath blight symptoms include lesions on rice leaf sheaths.")])
    model = StubOllama("The passage describes lesions on leaf sheaths.")

    first = answer_question(original, retriever=retriever, ollama_client=model)
    resumed = answer_question(
        "yes",
        pending_question=first.pending_question,
        retriever=retriever,
        ollama_client=model,
    )

    assert first.status == "clarification"
    assert first.pending_question == original
    assert resumed.status == "answered"
    assert resumed.pending_question is None
    assert resumed.paddy_confirmed is True
    assert retriever.calls == [(original, {"top_k": 4, "crop": "paddy"})]
    assert len(model.calls) == 1


@pytest.mark.parametrize("confirmation", ["yes, paddy", "rice", "paddy"])
def test_clear_paddy_confirmations_resume_pending_question(confirmation):
    original = "What symptoms of sheath blight are described?"
    retriever = StubRetriever([evidence_chunk("Sheath blight symptoms include lesions.")])
    model = StubOllama("The passage describes lesions.")
    first = answer_question(original, retriever=retriever, ollama_client=model)

    resumed = answer_question(
        confirmation,
        pending_question=first.pending_question,
        retriever=retriever,
        ollama_client=model,
    )

    assert resumed.status == "answered"
    assert resumed.pending_question is None
    assert retriever.calls[-1][0] == original


def test_negative_confirmation_clears_pending_question_and_asks_crop():
    original = "What symptoms of sheath blight are described?"
    retriever = StubRetriever([evidence_chunk()])
    model = StubOllama()
    first = answer_question(original, retriever=retriever, ollama_client=model)

    declined = answer_question(
        "no",
        pending_question=first.pending_question,
        retriever=retriever,
        ollama_client=model,
    )

    assert declined.status == "clarification"
    assert declined.pending_question == original
    assert "Which crop did you mean?" in declined.answer
    assert retriever.calls == []
    assert model.calls == []


def test_new_non_paddy_request_abandons_pending_question():
    retriever = StubRetriever([evidence_chunk()])
    model = StubOllama()
    first = answer_question("What symptoms of sheath blight are described?", retriever=retriever)

    result = answer_question(
        "What symptoms appear on tomato?",
        pending_question=first.pending_question,
        retriever=retriever,
        ollama_client=model,
    )

    assert result.status == "unsupported_crop"
    assert result.pending_question is None
    assert retriever.calls == []
    assert model.calls == []


def test_new_question_replaces_pending_question_instead_of_inheriting_it():
    original = "What symptoms of sheath blight are described?"
    new_question = "What nutrient deficiencies are described?"
    nutrient_evidence = RetrievedChunk(
        text="The article describes iron and zinc deficiencies in rice.",
        source_filename="11-14.pdf",
        page_number=1,
        metadata={"source_title": "Rice nutrient deficiencies", "topic_category": "nutrient_management"},
        distance=0.2,
    )
    retriever = StubRetriever([nutrient_evidence])
    model = StubOllama("The article discusses iron and zinc deficiencies.")
    first = answer_question(original, retriever=retriever, ollama_client=model)

    replacement = answer_question(
        new_question,
        pending_question=first.pending_question,
        retriever=retriever,
        ollama_client=model,
    )
    resumed = answer_question(
        "yes",
        pending_question=replacement.pending_question,
        retriever=retriever,
        ollama_client=model,
    )

    assert replacement.status == "clarification"
    assert replacement.pending_question == new_question
    assert resumed.status == "answered"
    assert retriever.calls[-1][0] == new_question
    assert retriever.calls[-1][0] != original


@pytest.mark.parametrize("crop", ["tomato", "coconut", "arecanut"])
def test_unsupported_crop_is_rejected_politely(crop):
    retriever = StubRetriever([evidence_chunk()])
    model = StubOllama()

    result = answer_question(f"How do I manage {crop} disease?", retriever=retriever, ollama_client=model)

    assert result.status == "unsupported_crop"
    assert "supports paddy (rice) only" in result.answer
    assert retriever.calls == []
    assert model.calls == []


@pytest.mark.parametrize("crop_term", ["paddy", "rice", "PADDY", "RiCe"])
def test_explicit_paddy_or_rice_is_validated_and_retrieved(crop_term):
    retriever = StubRetriever([evidence_chunk()])
    model = StubOllama()

    result = answer_question(
        f"What symptoms appear in {crop_term}?", retriever=retriever, ollama_client=model
    )

    assert result.status == "answered"
    assert result.paddy_confirmed is True
    assert retriever.calls[0][1]["crop"] == "paddy"
    assert len(model.calls) == 1


def test_follow_up_can_use_previously_confirmed_paddy_scope():
    retriever = StubRetriever([evidence_chunk()])
    model = StubOllama()

    result = answer_question(
        "What does the passage say about it?",
        paddy_confirmed=True,
        retriever=retriever,
        ollama_client=model,
    )

    assert result.status == "answered"
    assert result.paddy_confirmed is True
    assert retriever.calls[0][1]["crop"] == "paddy"


def test_empty_retrieval_does_not_call_model_or_make_advice():
    retriever = StubRetriever()
    model = StubOllama()

    result = answer_question("What symptoms are reported for paddy?", retriever=retriever, ollama_client=model)

    assert result.status == "no_evidence"
    assert "do not establish an answer" in result.answer
    assert "won’t generate an agricultural recommendation" in result.answer
    assert model.calls == []


def test_irrelevant_retrieval_does_not_call_model_or_show_irrelevant_citation():
    nutrient_chunk = RetrievedChunk(
        text="A paddy nutrient article discusses micronutrients.",
        source_filename="11-14.pdf",
        page_number=1,
        metadata={"source_title": "Nutrient management", "topic_category": "nutrient_management"},
        distance=0.1,
    )
    retriever = StubRetriever([nutrient_chunk])
    model = StubOllama()

    result = answer_question(
        "What causes sheath blight in paddy?", retriever=retriever, ollama_client=model
    )

    assert result.status == "no_evidence"
    assert result.evidence == ()
    assert "do not establish an answer" in result.answer
    assert model.calls == []


def test_high_risk_input_returns_extension_referral_without_generation():
    retriever = StubRetriever([evidence_chunk()])
    model = StubOllama()

    result = answer_question(
        "What pesticide dose should I spray on paddy?", retriever=retriever, ollama_client=model
    )

    assert result.status == "high_risk"
    assert "provisional" in result.answer
    assert "extension officer" in result.answer
    assert result.evidence == ()
    assert model.calls == []


def test_generation_uses_only_retrieved_passages_and_fixed_local_model():
    passage = evidence_chunk("Rice sheath blight may show lesions on leaf sheaths.")
    retriever = StubRetriever([passage])
    model = StubOllama("The passage describes lesions on leaf sheaths.")

    result = answer_question(
        "What symptoms are described for paddy sheath blight?",
        retriever=retriever,
        ollama_client=model,
    )

    call = model.calls[0]
    assert result.answer == "The passage describes lesions on leaf sheaths."
    assert call["model"] == LLM_MODEL == "qwen2.5:3b"
    assert "untrusted source data" in call["messages"][0]["content"]
    assert "lesions on leaf sheaths" in call["messages"][1]["content"]
    assert "Odisha" in call["messages"][1]["content"]
    assert result.evidence == (passage,)


def test_generation_prompt_requests_concise_farmer_friendly_answers():
    prompt = _system_prompt().casefold()
    assert "40–80 words" in prompt
    assert "simple english" in prompt
    assert "3–4 concise bullets" in prompt
    assert "avoid generic introductions" in prompt


def test_generation_prompt_preserves_grounding_and_follow_up_safety():
    prompt = _system_prompt().casefold()
    assert "only claims directly supported" in prompt
    assert "do not diagnose" in prompt
    assert "do not prescribe or recommend pesticides" in prompt
    assert "never present it as locally validated advice" in prompt
    assert "do not invent citations or add a separate source list" in prompt
    assert "answer only the new request" in prompt
    assert "do not repeat the earlier answer" in prompt
    assert "the latest question determines the task" in prompt
    assert "put its direct answer in the first sentence" in prompt
    assert "answer only the new request" in prompt
    assert "do not repeat an earlier answer" in prompt
    assert "for a cause question, focus on the cause" in prompt
    assert "do not invent an action absent from the evidence" in prompt
    assert "for a request to explain something more simply" in prompt


def test_insufficient_evidence_response_is_concise_and_does_not_overclaim():
    model = StubOllama()
    result = answer_question(
        "What are paddy symptoms of an unknown condition?",
        retriever=StubRetriever(),
        ollama_client=model,
    )
    assert result.status == "no_evidence"
    assert len(result.answer.split()) < 40
    assert "do not establish an answer" in result.answer
    assert "won’t generate" in result.answer
    assert not model.calls


def test_generated_specific_dose_is_replaced_with_safe_referral():
    model = StubOllama("Apply chemical X at 2 ml per liter every 7 days.")

    result = answer_question(
        "What symptoms are described for paddy?",
        retriever=StubRetriever([evidence_chunk()]),
        ollama_client=model,
    )

    assert result.status == "answered"
    assert "I can’t provide chemical or treatment instructions" in result.answer
    assert "2 ml" not in result.answer


def test_retrieval_failure_becomes_farmer_safe_error():
    def broken_retriever(*args, **kwargs):
        raise RuntimeError("internal storage path")

    with pytest.raises(ChatServiceError, match="knowledge index is unavailable"):
        answer_question("Paddy leaf symptoms?", retriever=broken_retriever, ollama_client=StubOllama())


def test_missing_persistent_index_has_clear_error(monkeypatch, tmp_path):
    import agriassist.chat as chat

    monkeypatch.setattr(chat, "INDEX_PATH", tmp_path / "missing-index")

    with pytest.raises(ChatServiceError, match="index has not been built"):
        answer_question("Paddy leaf symptoms?", ollama_client=StubOllama())


def test_ollama_failure_does_not_leak_exception_details():
    class FailingOllama:
        def chat(self, **kwargs):
            raise RuntimeError("private internal detail")

    with pytest.raises(ChatServiceError, match="local language model") as error:
        answer_question(
            "What symptoms appear in paddy?",
            retriever=StubRetriever([evidence_chunk()]),
            ollama_client=FailingOllama(),
        )
    assert "private internal detail" not in str(error.value)
