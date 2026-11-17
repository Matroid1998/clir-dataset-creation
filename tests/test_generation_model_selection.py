"""Decision boundaries and provenance for automatic generator selection."""

import pytest

from clir_bench.core import generation_models as models


@pytest.mark.parametrize("advantage,expected", [
    (-1, models.HAIKU_MODEL), (0, models.HAIKU_MODEL),
    (.49, models.HAIKU_MODEL), (.5, models.QWEN_MODEL), (.51, models.QWEN_MODEL),
])
def test_qwen_requires_at_least_half_a_point(monkeypatch, advantage, expected):
    row = {"recommended_model": models.QWEN_MODEL, "paired_documents": 10,
           "haiku_minus_qwen_paired": -advantage,
           # Opposing unpaired means must not override the shared-input result.
           "haiku_mean": 30, "qwen_mean": 35}
    monkeypatch.setattr(models, "_table", lambda: ({("corpus", "lookup", "en"): row}, "sha"))
    choice = models.select_generator("corpus", "lookup", "en")
    assert choice.model == expected
    assert choice.basis == "paired_target_mean"
    assert choice.metadata()["generator_selection_qwen_advantage"] == advantage


def test_sparse_comparison_uses_the_table_means(monkeypatch):
    row = {"recommended_model": models.QWEN_MODEL, "paired_documents": 2,
           "haiku_minus_qwen_paired": -10, "haiku_mean": 30, "qwen_mean": 30.25}
    monkeypatch.setattr(models, "_table", lambda: ({("corpus", "lookup", "en"): row}, "sha"))
    choice = models.select_generator("corpus", "lookup", "en")
    assert choice.model == models.HAIKU_MODEL
    assert choice.basis == "observed_target_mean"


def test_missing_cell_falls_back_to_haiku():
    choice = models.select_generator("eurlex", "source_finding", "en")
    assert choice.model == models.HAIKU_MODEL
    assert choice.basis == "no_data"


def test_explicit_model_does_not_read_the_comparison_table(monkeypatch):
    monkeypatch.setattr(models, "_table", lambda: pytest.fail("Explicit generator consulted table"))
    choice = models.select_generator("un", "lookup", "ru", model="provider/manual")
    assert choice.model == "provider/manual" and choice.metadata() == {}


@pytest.mark.parametrize("corpus,mode,language,expected", [
    ("un", "practitioner", "ru", models.QWEN_MODEL),
    ("un", "claim_verification", "ru", models.HAIKU_MODEL),
    ("un", "lookup", "ar", models.HAIKU_MODEL),
    ("eurlex", "lookup", "de", models.QWEN_MODEL),
    ("eurlex", "conceptual", "es", models.HAIKU_MODEL),
    ("gp", "constraint_search", "ja", models.QWEN_MODEL),
    ("gp", "source_finding", "ru", models.HAIKU_MODEL),
])
def test_packaged_table_applies_corpus_language_and_mode(corpus, mode, language, expected):
    assert models.select_generator(corpus, mode, language).model == expected


def test_response_cache_filters_use_real_provider_ids():
    assert models.replay_models(["auto", "provider/manual"]) == [
        models.HAIKU_MODEL, models.QWEN_MODEL, "provider/manual"]
