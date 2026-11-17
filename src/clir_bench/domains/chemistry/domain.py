"""
The chemistry-patents domain: one wiring point.

Everything the core needs is assembled here from the modules alongside. Reading
this file should be enough to understand what the domain is; the details live in
``vocabulary.py`` (what chemistry means), ``schema.py`` (what a document is),
``attribution.py`` (whose data it is), ``sources/`` (where it comes from) and
``qac/`` (how questions are made).
"""

from __future__ import annotations

from clir_bench.core.domain import (
    AnalysisVocab,
    DomainSpec,
    LanguageSpec,
    SourceSpec,
)
from clir_bench.domains.chemistry import vocabulary as vocab
from clir_bench.domains.chemistry.attribution import ATTRIBUTIONS
from clir_bench.domains.chemistry.qac.candidates import parse_candidates
from clir_bench.domains.chemistry.qac.decider import route_plan
from clir_bench.domains.chemistry.qac.plans import PLANS
from clir_bench.domains.chemistry.schema import SCHEMA
from clir_bench.domains.chemistry.sources.abstract_pairs import DATASET_REPO, validate_corpus

LANGUAGES = LanguageSpec(
    inventory=vocab.WORKING_LANGUAGES,
    working=vocab.WORKING_LANGUAGES,
    names=vocab.LANGUAGE_NAMES,
    priority=vocab.LANGUAGE_PRIORITY,
    answer_order=vocab.ANSWER_LANGUAGE_ORDER,
)

SOURCES = (
    SourceSpec(
        name="gp",
        description="BASF-AI AI4Chem accepted within-document abstract pairs",
        languages=vocab.WORKING_LANGUAGES,
        corpus_relpath="ai4chem_abstract_pairs/multilingual_corpus.csv",
        qac_dir_relpath="ai4chem_abstract_pairs/qac",
        attribution_key="ai4chem_abstract_pairs",
        source_value="ai4chem_abstract_pairs",
    ),
)

ANALYSIS = AnalysisVocab(
    modes=vocab.MODES,
    strategies=vocab.STRATEGY_ORDER,
    strategy_numbers=vocab.STRATEGY_NUMBERS,
    diagnostic_languages=vocab.DIAGNOSTIC_LANGUAGES,
)

# The gp command key is retained, but its active corpus has an isolated location.
# Historical ingestion locations remain addressable without loading their data.
DATA_LAYOUT = {
    "gp_raw": "google_patents/chemistry_patents.ndjson",
    "gp_preprocessed": "google_patents/preprocessed",
    "gp_corpus": "ai4chem_abstract_pairs/multilingual_corpus.csv",
    "gp_qac": "ai4chem_abstract_pairs/qac",
    "gp_pairs": "ai4chem_abstract_pairs/accepted_pairs.csv",
    "gp_manifest": "ai4chem_abstract_pairs/manifest.json",
    "epo_manifest": "EPO/manifest.json",
    "epo_corpus": "EPO/multilingual_corpus.csv",
    "epo_qac": "EPO/qac",
    "chebi": "chebi",
    "alias_graph": "alias_graph",
    "code_switched": "code_switched",
    "progressive": "progressive_cs",
    "human_eval": "human_eval",
    "baselines": "baselines",
}

# Defaults a user can override in clir.toml under [domains.chemistry].
DEFAULTS = {
    "generation_model": "auto",
    # Input is distinct from the writable derived-corpus publishing destination.
    "input_dataset_repo": DATASET_REPO,
    "corpus_repo": "MehdiAstaraki/multilingual_GP",
    "benchmark_repo": "MehdiAstaraki/multi-lingual-qac-chem-patents",
    "gp_benchmark_repo": "MehdiAstaraki/multi-lingual-qac-chem-patents",
    "alias_graph_repo": "MehdiAstaraki/multi-lingual-qac-alias-graph",
    "progressive_repo": "MehdiAstaraki/progressive-code-switch",
    "chebi_variant": "full",
    # Read by the evaluation harness when it builds the MTEB task: the language
    # tags it validates eval_langs against, and the subject tags of the task
    # itself. Supplied as domain data so the harness never names a subject.
    "mteb_language_codes": vocab.MTEB_LANGUAGE_CODES,
    "mteb_task_domains": ("Chemistry", "Engineering"),
    # Column names in this domain's already-published datasets. The analysis
    # layer is ontology-agnostic and asks for candidates rather than assuming
    # ChEBI, so the historical names are declared here instead.
    "concept_columns": ("concept_id", "chebi_id"),
    "negative_name_columns": ("neighbor_name", "neighbor_concept_id", "neighbor_chebi_id"),
    "eval_models": (
        "Qwen/Qwen3-Embedding-0.6B",
        "intfloat/multilingual-e5-large-instruct",
        "BAAI/bge-m3",
        "jinaai/jina-embeddings-v3",
        "google/embeddinggemma-300m",
        "ibm-granite/granite-embedding-278m-multilingual",
        "jinaai/jina-colbert-v2",
        "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        "sentence-transformers/LaBSE",
        "nomic-ai/nomic-embed-text-v2-moe",
    ),
}


SPEC = DomainSpec(
    name="chemistry",
    title="Chemistry patents",
    description=(
        "Multilingual chemistry-patent retrieval built only from BASF-AI's "
        "AI4Chem within-document Google Patents abstract pairs. Accepted exact "
        "and high-equivalence pairs in English, German, French, Spanish, "
        "Chinese, Japanese and Russian provide the explicit cross-language relevance links."
    ),
    schema=SCHEMA,
    languages=LANGUAGES,
    sources=SOURCES,
    prompts_package="clir_bench.domains.chemistry.qac.prompts",
    attributions=ATTRIBUTIONS,
    analysis=ANALYSIS,
    data_layout=DATA_LAYOUT,
    defaults=DEFAULTS,
    qac_plans=PLANS,
    qac_candidate_parser=parse_candidates,
    qac_route_plan=route_plan,
    qac_validate_corpus=validate_corpus,
)


__all__ = ["SPEC"]
