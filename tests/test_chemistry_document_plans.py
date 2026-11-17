"""Chemistry generation selects exact documents and follows their source languages."""

from collections import Counter
from types import SimpleNamespace

import pytest

from clir_bench.core import corpus
from clir_bench.domains.chemistry.domain import SPEC
from clir_bench.domains.chemistry.qac.plans import PLANS
from clir_bench.domains.chemistry.schema import SCHEMA
from clir_bench.domains.chemistry.vocabulary import MODES


@pytest.fixture
def documents():
    rows = [
        {"id": identity, "publication_number": "EP-1234567-A1", "language": language,
         "context": f"Abstract for {identity}"}
        for identity, language in (("fr-a", "fr"), ("fr-b", "fr"), ("en-a", "en"),
                                   ("de-a", "de"), ("es-a", "es"), ("zh-a", "zh"),
                                   ("ja-a", "ja"), ("ru-a", "ru"), ("ar-a", "ar"))
    ]
    rows.append(dict(rows[0], id="empty-fr", context=""))
    return {"EP-1234567-A1": rows}


def build(name, grouped, **options):
    args = SimpleNamespace(seed=42, questions=100, pool=None, langs=None, modes=None,
                           priority_langs=None, decider_model=None, exclude_from=None)
    args.__dict__.update(options)
    return PLANS[name](
        context=SimpleNamespace(schema=SCHEMA, languages=SPEC.languages),
        source=SPEC.source("gp"), grouped=grouped, args=args,
    )


@pytest.mark.parametrize("name", PLANS)
def test_plans_preserve_distinct_same_language_documents_and_derive_language(name, documents):
    plan = build(name, documents)
    by_id = {row["id"]: row for rows in documents.values() for row in rows}
    assert len(plan) == 8 * len(MODES)
    assert len({(item.document_id, item.mode) for item in plan}) == len(plan)
    assert {item.document_id for item in plan} == {
        "en-a", "fr-a", "fr-b", "de-a", "es-a", "zh-a", "ja-a", "ru-a",
    }
    assert all(item.language == by_id[item.document_id]["language"] for item in plan)
    assert all(item.strategy == 3 for item in plan)
    assert Counter(item.mode for item in plan) == dict.fromkeys(MODES, 8)
    assert build(name, {key: list(reversed(rows)) for key, rows in documents.items()}) == plan


@pytest.mark.parametrize("name", PLANS)
def test_language_filters_select_existing_documents_without_language_expansion(name, documents):
    plan = build(name, documents, langs=["fr"])
    assert {item.document_id for item in plan} == {"fr-a", "fr-b"}
    assert {item.language for item in plan} == {"fr"}
    french_only = {key: [row for row in rows if row["language"] == "fr"]
                   for key, rows in documents.items()}
    assert build(name, french_only, langs=["en"]) == []


@pytest.mark.parametrize("name", PLANS)
def test_japanese_and_russian_language_filters_select_their_source_documents(name, documents):
    plan = build(name, documents, langs=["ja", "ru"])
    assert {item.document_id for item in plan} == {"ja-a", "ru-a"}
    assert {item.language for item in plan} == {"ja", "ru"}


@pytest.mark.parametrize("name", PLANS)
@pytest.mark.parametrize("option", ["langs", "priority_langs"])
def test_unsupported_language_filters_are_rejected_even_with_no_documents(name, option):
    with pytest.raises(ValueError, match="Unsupported chemistry"):
        build(name, {}, **{option: ["fr", "ar"]})


@pytest.mark.parametrize("name", PLANS)
def test_question_counts_apply_to_distinct_documents_per_mode_and_decider(name, documents):
    fixed = build(name, documents, questions=3, modes=["lookup", "comparison"])
    assert Counter(item.mode for item in fixed) == {"lookup": 3, "comparison": 3}
    automatic = build(name, documents, questions=3, decider_model="jev")
    assert len(automatic) == len({item.document_id for item in automatic}) == 3


def test_coverage_priority_uses_actual_document_language(documents):
    plan = build("coverage", documents, questions=2, priority_langs=["fr"], modes=["lookup"])
    assert {item.document_id for item in plan} == {"fr-a", "fr-b"}


@pytest.mark.parametrize("name", PLANS)
def test_endpoint_exclusions_leave_uncovered_siblings_available(name, documents, tmp_path):
    path = tmp_path / "previous.csv"
    corpus.write_rows(path, [{"publication_number": "EP-1234567-A1", "source_document_id": "fr-a"}],
                      ("publication_number", "source_document_id"))
    plan = build(name, documents, langs=["fr"], exclude_from=path)
    assert {item.document_id for item in plan} == {"fr-b"}
    corpus.write_rows(path, [{"publication_number": "EP-1234567-A1"}], ("publication_number",))
    assert build(name, documents, exclude_from=path) == []


@pytest.mark.parametrize("name", PLANS)
@pytest.mark.parametrize("defect", ["missing_id", "duplicate_id"])
def test_ambiguous_document_identities_fail_before_generation(name, documents, defect):
    row = documents["EP-1234567-A1"][0]
    if defect == "missing_id":
        row.pop("id")
    else:
        documents["EP-1234567-A1"].append(dict(row, context="Different source text"))
    with pytest.raises(ValueError, match="source document|document identity"):
        build(name, documents)
