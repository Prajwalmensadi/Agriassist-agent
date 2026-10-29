"""Session-scoped controller coordinating the bounded AgriAssist chat workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .chat import ChatServiceError, ChatResult, answer_question
from .context import CropContext, SymptomDistribution, assess_crop_context
from .knowledge import ALLOWED_PDF_FILENAMES, RetrievedChunk, retrieve


ResponseState = Literal[
    "clarification_required",
    "unsupported_scope",
    "insufficient_evidence",
    "answered",
    "error",
]

_FIELD_SYMPTOM_RE = re.compile(
    r"\b(?:my|our)\s+(?:paddy|rice)\b.{0,100}\b(?:yellow(?:ing)?|spots?|wilting|drying|"
    r"stunted|lesions?|discolou?red|curling)\b",
    re.I,
)
_DISTRIBUTION_WORDS = {
    "whole field": SymptomDistribution.WHOLE_FIELD,
    "everywhere": SymptomDistribution.WHOLE_FIELD,
    "patch": SymptomDistribution.PATCHES,
    "some plants": SymptomDistribution.SOME_PLANTS,
    "few plants": SymptomDistribution.SOME_PLANTS,
    "field edge": SymptomDistribution.FIELD_EDGE,
    "unknown": SymptomDistribution.UNKNOWN,
    "don't know": SymptomDistribution.UNKNOWN,
    "do not know": SymptomDistribution.UNKNOWN,
}
_CITATION_RE = re.compile(r"([\w.+ -]+\.pdf)\s*(?:,?\s*(?:page|p\.?)\s*(\d+))?", re.I)


@dataclass
class ConversationState:
    """Small per-browser-session state; no farmer identity or persistent memory."""

    paddy_confirmed: bool = False
    pending_question: str | None = None
    pending_kind: Literal["crop_scope", "symptom_context"] | None = None
    context_notes: list[str] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ControllerResponse:
    state: ResponseState
    answer: str
    evidence: tuple[RetrievedChunk, ...] = ()
    conversation: ConversationState = field(default_factory=ConversationState)
    validation_notes: tuple[str, ...] = ()


def _validated_evidence(evidence: tuple[RetrievedChunk, ...]) -> bool:
    return all(
        item.source_filename in ALLOWED_PDF_FILENAMES
        and isinstance(item.page_number, int)
        and not isinstance(item.page_number, bool)
        and item.page_number > 0
        for item in evidence
    )


def _invalid_model_citations(answer: str, evidence: tuple[RetrievedChunk, ...]) -> bool:
    available = {(item.source_filename, item.page_number) for item in evidence}
    for match in _CITATION_RE.finditer(answer):
        filename, page = match.groups()
        if page is None or (filename.strip(), int(page)) not in available:
            return True
    return False


def _needs_geographic_caveat(evidence: tuple[RetrievedChunk, ...]) -> bool:
    for item in evidence:
        region = str(item.metadata.get("geographic_applicability", "unknown")).casefold()
        if "coastal karnataka" not in region or any(
            place in region for place in ("odisha", "telangana", "india", "unknown")
        ):
            return True
    return False


def _map_chat_status(result: ChatResult) -> ResponseState:
    if result.status == "clarification":
        return "clarification_required"
    if result.status == "unsupported_crop":
        return "unsupported_scope"
    if result.status in {"no_evidence", "high_risk"}:
        return "insufficient_evidence"
    return "answered"


def _crop_context_question(question: str) -> str | None:
    if not _FIELD_SYMPTOM_RE.search(question) or "?" in question:
        return None
    symptom = "reported crop symptoms"
    lower = question.casefold()
    for term in ("yellowing", "yellow", "spots", "wilting", "drying", "stunted", "lesions", "curling"):
        if term in lower:
            symptom = term
            break
    context = CropContext(crop="paddy", symptoms=[symptom])
    assessment = assess_crop_context(context)
    if "symptom_distribution" not in assessment.missing_fields:
        return None
    return "Are the symptoms across the whole field or mainly in patches or some plants?"


def _context_answer_value(reply: str) -> SymptomDistribution | None:
    normalized = " ".join(reply.casefold().strip().split())
    return next((value for phrase, value in _DISTRIBUTION_WORDS.items() if phrase in normalized), None)


def handle_message(
    message: str,
    *,
    conversation: ConversationState | None = None,
    top_k: int = 4,
    retriever: Callable[..., list[RetrievedChunk]] = retrieve,
    ollama_client: Any | None = None,
) -> ControllerResponse:
    """Handle one turn with deterministic scope, evidence, and output policies."""

    state = conversation or ConversationState()
    if not isinstance(message, str) or not message.strip():
        return ControllerResponse(
            "clarification_required", "Please enter a question about paddy (rice).", conversation=state
        )
    turn = message.strip()

    # A brief answer resolves only the pending field-context question. A new
    # substantive question replaces it, preventing stale context from leaking.
    if state.pending_kind == "symptom_context" and state.pending_question:
        distribution = _context_answer_value(turn)
        if distribution is not None:
            state.context_notes.append(f"Farmer-reported symptom distribution: {distribution.value}.")
            turn = f"{state.pending_question}\n" + "\n".join(state.context_notes)
            state.pending_question = None
            state.pending_kind = None
        elif "?" not in turn and len(turn.split()) <= 3:
            return ControllerResponse(
                "clarification_required",
                "Please tell me whether symptoms are across the whole field, in patches, on some plants, at the field edge, or unknown.",
                conversation=state,
            )
        else:
            state.pending_question = None
            state.pending_kind = None
            state.context_notes.clear()
            state.history.clear()

    if state.pending_kind is None and not state.pending_question and not state.paddy_confirmed:
        # Crop scope confirmation remains owned by the existing validator and
        # chat workflow. Symptom intake starts only after paddy scope is explicit.
        if re.search(r"\b(?:paddy|rice)\b", turn, re.I):
            clarification = _crop_context_question(turn)
            if clarification:
                state.paddy_confirmed = True
                state.pending_question = turn
                state.pending_kind = "symptom_context"
                return ControllerResponse("clarification_required", clarification, conversation=state)

    # For a clarified distribution include it as reported context for the
    # retriever/model, while retaining the same bounded existing pipeline.
    chat_pending = state.pending_question if state.pending_kind == "crop_scope" else None
    context = state.history[-4:] if state.paddy_confirmed else []
    try:
        result = answer_question(
            turn,
            paddy_confirmed=state.paddy_confirmed,
            pending_question=chat_pending,
            top_k=top_k,
            retriever=retriever,
            ollama_client=ollama_client,
            conversation_context=context,
        )
    except ChatServiceError:
        return ControllerResponse(
            "error",
            "AgriAssist could not complete this request. Please check the local services and try again.",
            conversation=state,
        )

    state.paddy_confirmed = result.paddy_confirmed
    if result.pending_question:
        state.pending_question = result.pending_question
        state.pending_kind = "crop_scope"
    else:
        state.pending_question = None
        state.pending_kind = None

    response_state = _map_chat_status(result)
    evidence = result.evidence
    if evidence and not _validated_evidence(evidence):
        return ControllerResponse(
            "insufficient_evidence",
            "Retrieved material did not include valid source and page provenance, so I cannot present it as evidence.",
            conversation=state,
            validation_notes=("invalid_evidence_provenance",),
        )
    if response_state == "answered" and not evidence:
        return ControllerResponse(
            "insufficient_evidence",
            "The available paddy documents do not establish an answer to this question.",
            conversation=state,
            validation_notes=("empty_evidence",),
        )
    if response_state == "answered" and _invalid_model_citations(result.answer, evidence):
        return ControllerResponse(
            "insufficient_evidence",
            "The generated response included a source reference that could not be verified against retrieved pages; I will not repeat that claim.",
            evidence=evidence,
            conversation=state,
            validation_notes=("invalid_model_citation",),
        )

    answer = result.answer
    if response_state == "answered" and _needs_geographic_caveat(evidence):
        if not re.search(r"coastal Karnataka|locally validated|local applicability", answer, re.I):
            answer += "\n\nGeographic limitation: these provisional sources are not locally validated for coastal Karnataka."

    if response_state in {"answered", "insufficient_evidence"}:
        state.history.append({"user": turn, "assistant": answer})
        state.history = state.history[-8:]
    return ControllerResponse(response_state, answer, evidence, state)
