"""Deterministic crop-context intake for agricultural advisory cases."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GrowthStage(str, Enum):
    """Common crop stages, with explicit values for uncertainty or other stages."""

    GERMINATION = "germination"
    SEEDLING = "seedling"
    TILLERING = "tillering"
    PANICLE_INITIATION = "panicle initiation"
    FLOWERING = "flowering"
    GRAIN_FILLING = "grain filling"
    MATURITY = "maturity"
    UNKNOWN = "unknown"
    OTHER = "other"


class SymptomDistribution(str, Enum):
    """Where in the crop or field symptoms have been observed."""

    WHOLE_FIELD = "whole field"
    PATCHES = "patches"
    SOME_PLANTS = "some plants"
    FIELD_EDGE = "field edge"
    UNKNOWN = "unknown"
    OTHER = "other"


class CropContext(BaseModel):
    """Farmer-reported crop facts; unset fields remain distinct from unknown values."""

    model_config = ConfigDict(str_strip_whitespace=True)

    location: str | None = Field(default=None, description="Farm region or locality; no exact address needed.")
    crop: str | None = Field(default=None, description="Supported crop: paddy (rice).")
    growth_stage: GrowthStage | None = None
    symptoms: list[str] | None = Field(default=None, description="Observed symptoms, without diagnosis.")
    symptom_distribution: SymptomDistribution | None = None
    recent_field_events: list[str] | None = Field(
        default=None,
        description="Farmer-reported recent events or actions; an empty list means none reported.",
    )
    symptom_duration: str | None = Field(default=None, description="How long symptoms have been present, if known.")

    @field_validator("location", "symptom_duration", mode="before")
    @classmethod
    def blank_text_is_unset(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("crop", mode="before")
    @classmethod
    def normalize_supported_crop(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("crop must be 'paddy' or 'rice'; empty crop names are not allowed")

        normalized = value.strip().casefold()
        if normalized not in {"paddy", "rice"}:
            raise ValueError(
                f"unsupported crop '{value.strip()}'; AgriAssist currently supports paddy (rice) only"
            )
        return "paddy"

    @field_validator("symptoms", mode="before")
    @classmethod
    def clean_symptoms(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            cleaned = [
                item.strip() if isinstance(item, str) else item
                for item in value
                if not isinstance(item, str) or item.strip()
            ]
            return cleaned or None
        return value

    @field_validator("recent_field_events", mode="before")
    @classmethod
    def clean_field_events(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return [
                item.strip() if isinstance(item, str) else item
                for item in value
                if not isinstance(item, str) or item.strip()
            ]
        return value


class ContextAssessment(BaseModel):
    """Intake status and one prioritized question; it contains no diagnosis or advice."""

    missing_fields: list[str]
    next_question: str | None
    sufficient_for_retrieval: bool


_MISSING_FIELD_QUESTIONS = (
    ("crop", "Which crop are you growing?"),
    ("symptoms", "What changes or symptoms have you observed on the crop?"),
    ("symptom_distribution", "Are the symptoms across the whole field or mainly in patches or some plants?"),
    ("growth_stage", "What growth stage is the crop at, if you know? You can say ‘unknown’."),
    ("location", "Which district or region is the field in?"),
    ("recent_field_events", "Have there been any recent field events or actions, such as flooding or fertilizer application?"),
    ("symptom_duration", "How long have you noticed these symptoms, if you know?"),
)


def assess_crop_context(context: CropContext) -> ContextAssessment:
    """Find missing context and choose one question without diagnosing or advising."""

    missing_fields = [
        field_name
        for field_name, _ in _MISSING_FIELD_QUESTIONS
        if getattr(context, field_name) is None
    ]
    questions = dict(_MISSING_FIELD_QUESTIONS)
    next_question = next(
        (questions[field_name] for field_name in missing_fields if field_name in questions),
        None,
    )

    return ContextAssessment(
        missing_fields=missing_fields,
        next_question=next_question,
        sufficient_for_retrieval=bool(context.crop and context.symptoms),
    )
