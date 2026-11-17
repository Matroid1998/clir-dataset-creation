"""Legal documents: EUR-Lex articles and United Nations document blocks.

Question generation uses the existing structured indexes and source-specific
prompt packs. The optional flat corpus paths below are reserved for exports;
they never point at the structured source files.
"""

from clir_bench.core.domain import AnalysisVocab, CorpusSchema, DomainSpec, LanguageSpec, SourceSpec

SPEC = DomainSpec(
    name="legal",
    title="Legal documents",
    description="Multilingual questions grounded in EUR-Lex articles and United Nations blocks.",
    schema=CorpusSchema(
        fields=("target_id", "document_id", "question_language", "document_text", "corpus"),
        family_field="target_id",
        id_field="target_id",
        language_field="question_language",
        text_fields=("document_text",),
    ),
    languages=LanguageSpec(
        inventory=("ar", "de", "en", "es", "fr", "ru", "zh"),
        working=("en", "fr", "de", "es", "zh", "ar", "ru"),
        names={
            "ar": "Arabic",
            "de": "German",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "ru": "Russian",
            "zh": "Chinese",
        },
        priority=("en", "fr", "de", "es", "zh", "ar", "ru"),
    ),
    sources=(
        SourceSpec(
            name="eurlex",
            languages=("en", "fr", "de", "es"),
            corpus_relpath="eurlex/multilingual_corpus.csv",
            qac_dir_relpath="eurlex/qac",
            attribution_key="eurlex",
            description="EUR-Lex articles and resolved references",
        ),
        SourceSpec(
            name="un",
            languages=("ar", "en", "es", "fr", "ru", "zh"),
            corpus_relpath="un_parallel/multilingual_corpus.csv",
            qac_dir_relpath="un_parallel/qac",
            attribution_key="un",
            description="United Nations Parallel Corpus document blocks",
        ),
    ),
    # The run pipeline selects prompts_eurlex or prompts_un per source. This
    # anchor identifies their common package without pretending they are one pack.
    prompts_package="clir_bench.domains.legal.qac",
    attributions={
        "eurlex": (
            "Source: EUR-Lex, European Union (https://eur-lex.europa.eu), and "
            "MultiEURLEX: Chalkidis, Fergadiotis, and Androutsopoulos (2021), EMNLP."
        ),
        "un": (
            "Source: United Nations. Ziemski, Junczys-Dowmunt, and Pouliquen (2016), "
            "The United Nations Parallel Corpus, LREC. Corpus terms: "
            "https://conferences.unite.un.org/UNCorpus."
        ),
    },
    analysis=AnalysisVocab(
        modes=("lookup", "fact_pattern", "technical", "conceptual", "semantic", "descriptive",
               "practitioners", "comparison", "claim_verification", "source_finding")
    ),
    data_layout={
        "eurlex_structure": "eurlex/structure",
        "un_blocks": "un_parallel/blocks",
        "un_parallel": "un_parallel/6way",
    },
    defaults={
        "qac_runs": True,
        "generation_model": "gpt-5.6-luna",
        "verifier_model": "google/gemini-3.8-flash",
    },
)


def qac_command(command, args, context) -> int:
    """Dispatch through the legal pipeline without loading its indexes at startup."""
    from clir_bench.domains.legal import qac

    return getattr(qac, command)(args, context)


__all__ = ["SPEC", "qac_command"]
