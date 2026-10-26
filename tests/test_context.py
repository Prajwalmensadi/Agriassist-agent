import pytest
from pydantic import ValidationError

from agriassist.context import (
    CropContext,
    GrowthStage,
    SymptomDistribution,
    assess_crop_context,
)


def test_empty_report_requests_crop_first_and_cannot_retrieve():
    assessment = assess_crop_context(CropContext())

    assert assessment.missing_fields[0] == "crop"
    assert assessment.next_question == "Which crop are you growing?"
    assert assessment.sufficient_for_retrieval is False


def test_yellow_paddy_leaves_prompts_for_distribution_without_diagnosis():
    context = CropContext(crop="paddy", symptoms=["yellow leaves"])

    assessment = assess_crop_context(context)

    assert "symptom_distribution" in assessment.missing_fields
    assert assessment.next_question == (
        "Are the symptoms across the whole field or mainly in patches or some plants?"
    )
    assert assessment.sufficient_for_retrieval is True
    assert not hasattr(assessment, "diagnosis")
    assert not hasattr(assessment, "recommendation")


def test_partially_completed_report_prioritizes_one_question():
    context = CropContext(
        crop="paddy",
        symptoms=["yellow leaves"],
        symptom_distribution=SymptomDistribution.PATCHES,
    )

    assessment = assess_crop_context(context)

    assert assessment.missing_fields == [
        "growth_stage",
        "location",
        "recent_field_events",
        "symptom_duration",
    ]
    assert assessment.next_question == "What growth stage is the crop at, if you know? You can say ‘unknown’."
    assert assessment.sufficient_for_retrieval is True


def test_complete_context_has_no_missing_fields_or_question():
    context = CropContext(
        location="Udupi district",
        crop="paddy",
        growth_stage=GrowthStage.TILLERING,
        symptoms=["yellow leaves"],
        symptom_distribution=SymptomDistribution.PATCHES,
        recent_field_events=["heavy rain"],
        symptom_duration="three days",
    )

    assessment = assess_crop_context(context)

    assert assessment.missing_fields == []
    assert assessment.next_question is None
    assert assessment.sufficient_for_retrieval is True


def test_explicitly_unknown_growth_stage_is_not_reasked():
    context = CropContext(
        crop="paddy",
        symptoms=["yellow leaves"],
        growth_stage=GrowthStage.UNKNOWN,
    )

    assessment = assess_crop_context(context)

    assert "growth_stage" not in assessment.missing_fields
    assert assessment.next_question != "What growth stage is the crop at, if you know? You can say ‘unknown’."


def test_whitespace_values_are_treated_as_unset():
    context = CropContext(
        location="  ",
        crop=" \t",
        symptoms=[" ", "\n"],
        symptom_duration=" ",
    )

    assessment = assess_crop_context(context)

    assert context.location is None
    assert context.crop is None
    assert context.symptoms is None
    assert context.symptom_duration is None
    assert assessment.sufficient_for_retrieval is False
    assert assessment.next_question == "Which crop are you growing?"


def test_invalid_growth_stage_is_rejected():
    with pytest.raises(ValidationError):
        CropContext(growth_stage="fruiting")


def test_crop_and_symptom_report_is_retrievable_without_diagnosis():
    context = CropContext(crop="rice", symptoms=["brown leaf tips"])

    assessment = assess_crop_context(context)

    assert assessment.sufficient_for_retrieval is True
    assert assessment.next_question is not None
    assert assessment.missing_fields
    assert not hasattr(assessment, "diagnosis")
