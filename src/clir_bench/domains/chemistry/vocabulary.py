"""
Chemistry and patent vocabulary.

Everything the pipeline knows about *chemistry* as a subject and *patents* as a
document type lives here, so the core never has to. Constants that were spread
across a dozen modules -- four inconsistent language lists, three different
chemistry classifier definitions, two conflicting ``LANG_NAMES`` maps -- are
collected in one file where their differences are visible and explained.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #

# Languages requested from BigQuery. Extraction casts a wide net; only a few of
# these end up with enough parallel coverage to be usable.
EXTRACTION_LANGUAGES = (
    "en", "de", "fr", "es", "ja", "ko", "zh",
    "ru", "pt", "it", "nl",
    "ar", "fa", "tr", "pl", "hi",
)

# Languages the benchmark actually generates questions in: those with genuine
# human-translated parallel patent text and a full prompt set.
WORKING_LANGUAGES = ("en", "de", "fr", "es", "zh", "ja", "ru")

# The EPO publishes in its three official languages only.
EPO_LANGUAGES = ("en", "fr", "de")

LANGUAGE_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "ar": "Arabic", "fa": "Persian",
    "tr": "Turkish", "pl": "Polish", "hi": "Hindi",
}

# Order used when picking "the best available language version" of a document.
# Distinct from WORKING_LANGUAGES because the ordering is load-bearing: it
# decides which passage a generator reads when several are available.
LANGUAGE_PRIORITY = ("en", "de", "fr", "es", "zh", "ja", "ru")

# Order used when choosing the surface form of a concept in an answer. The
# ``chebi`` entry is not a language: it is the ontology's own canonical name,
# used when no Wikipedia title exists for the query language.
ANSWER_LANGUAGE_ORDER = ("en", "de", "fr", "es", "zh", "ja", "ru", "chebi")

# ISO code -> the language tags MTEB expects.
MTEB_LANGUAGE_CODES = {
    "ar": "arb-Arab", "bg": "bul-Cyrl", "cs": "ces-Latn", "da": "dan-Latn",
    "de": "deu-Latn", "el": "ell-Grek", "en": "eng-Latn", "es": "spa-Latn",
    "et": "est-Latn", "fa": "pes-Arab", "fi": "fin-Latn", "fr": "fra-Latn",
    "hi": "hin-Deva", "hu": "hun-Latn", "it": "ita-Latn", "ja": "jpn-Jpan",
    "ko": "kor-Hang", "lt": "lit-Latn", "lv": "lav-Latn", "mt": "mlt-Latn",
    "nl": "nld-Latn", "pl": "pol-Latn", "pt": "por-Latn", "ro": "ron-Latn",
    "ru": "rus-Cyrl", "sk": "slk-Latn", "sl": "slv-Latn", "sv": "swe-Latn",
    "tr": "tur-Latn", "zh": "zho-Hans",
}

# --------------------------------------------------------------------------- #
# What counts as chemistry
# --------------------------------------------------------------------------- #
# The two sources filter for chemistry differently, and the difference is real
# rather than accidental. Google Patents filters in SQL, where a narrow prefix
# set plus the SureChemBL chemical-annotation join is both cheap and precise.
# The EPO archives carry no such annotation, so that loader scores documents in
# Python from a wider classification list plus title keywords. Their shared core
# is CORE_CHEMISTRY_PREFIXES; anything beyond that is source-specific and should
# stay that way unless the corpora are deliberately re-harmonized.

# C = chemistry/metallurgy, A61K = medicinal preparations, A61P = therapeutic activity.
CORE_CHEMISTRY_PREFIXES = ("C", "A61K", "A61P")

# Google Patents: CPC/IPC prefixes used in the extraction SQL.
GP_CPC_PREFIXES = CORE_CHEMISTRY_PREFIXES
GP_IPC_PREFIXES = CORE_CHEMISTRY_PREFIXES

# EPO: broader classification list, since no chemical annotation is available.
EPO_CLASSIFICATION_PREFIXES = (
    "C", "A01N", "A23L", "A61K", "A61P",
    "B01D", "B01F", "B01J", "B01L",
    "C25", "G01N", "H01M",
)

# EPO: title keywords that promote a document to "chemistry related" when its
# classification alone is inconclusive. English-only, which biases recall toward
# English-titled documents -- a known limitation of the EPO ingest.
EPO_CHEMISTRY_KEYWORDS = (
    "catalyst", "catalytic", "polymer", "polymeric", "pharmaceutical", "compound",
    "chemical", "synthesis", "synthesise", "synthesize", "reaction", "reagent",
    "solvent", "molecule", "molecular", "crystalline", "crystallization",
    "formulation", "composition", "derivative", "acid", "alkaline", "oxidation",
    "reduction", "electrolyte", "monomer", "resin", "adhesive", "coating",
    "pigment", "enzyme", "protein", "peptide", "salt", "ester",
)

# Chemistry relevance labels produced by the EPO classifier, weakest first.
CHEMISTRY_LABELS = ("not_chemistry", "chemistry_related", "chemistry_core")

# IPC subclasses reported in corpus composition analyses.
IPC_CLASS_NAMES = {
    "C01": "Inorganic chemistry",
    "C02": "Water/waste treatment",
    "C03": "Glass and mineral wool",
    "C04": "Cements and ceramics",
    "C05": "Fertilisers",
    "C06": "Explosives and matches",
    "C07": "Organic chemistry",
    "C08": "Organic macromolecular compounds",
    "C09": "Dyes, paints, adhesives",
    "C10": "Petroleum, gas, fuels",
    "C11": "Animal/vegetable oils, detergents",
    "C12": "Biochemistry, microbiology",
    "C13": "Sugar industry",
    "C14": "Skins, hides, leather",
    "C21": "Metallurgy of iron",
    "C22": "Metallurgy, ferrous alloys",
    "C23": "Coating metallic material",
    "C25": "Electrolytic processes",
    "C30": "Crystal growth",
    "A61": "Medical and veterinary science",
    "A23": "Foods and foodstuffs",
    "A01": "Agriculture",
    "B01": "Physical/chemical processes",
}

# Core chemistry subclasses, used to split "core chemistry" from pharma and other.
CORE_CHEMISTRY_IPC_RANGE = tuple(f"C{n:02d}" for n in range(1, 31))

# --------------------------------------------------------------------------- #
# Question generation vocabulary
# --------------------------------------------------------------------------- #

MODE_TECHNICAL = "technical"
MODE_SEMANTIC = "semantic"
MODES = (
    "lookup",
    "process_retrieval",
    "conceptual",
    "constraint_search",
    "comparison",
    "assertion_verification",
    "source_finding",
)

# Analysis labels. ``forced_zh`` marks questions from the run that added Chinese
# coverage by asking in Chinese regardless of the document's own languages.
STRATEGY_ORDER = ("random_any", "random_missing", "random_existing", "all", "forced_zh")
STRATEGY_NUMBERS = {
    0: "forced_zh",
    1: "random_any",
    2: "random_missing",
    3: "random_existing",
    4: "all",
}

# Languages shown in the same-language retrieval-bias diagnostic.
DIAGNOSTIC_LANGUAGES = ("de", "en", "es", "fr", "ja", "pt", "ru", "zh")

# --------------------------------------------------------------------------- #
# Text limits (quality gates applied at corpus build time)
# --------------------------------------------------------------------------- #

MIN_ABSTRACT_WORDS = 50
MIN_ABSTRACT_CHARS = 300
# EPO only: a language counts as present when it has a real abstract OR a first
# claim of at least this many words. Deliberately stricter than the Google
# Patents gate -- the EPO publishes titles in all three official languages for
# nearly every document, so a title-aware gate would make the multilingual
# filter trivially true and fill the corpus with untranslated documents.
EPO_MIN_FIRST_CLAIM_WORDS = 10
MIN_DESCRIPTION_CHARS = 200
FIRST_CLAIM_MAX_CHARS = 1500
# Only the EPO ingest stores a description: BigQuery's localized descriptions are
# overwhelmingly English-only, so filling that column from Google Patents would
# make the corpus look multilingual where it is not. The EPO parser keeps 2000
# characters because its full text is the only place some of that content exists.
EPO_DESCRIPTION_MAX_CHARS = 2000


__all__ = [name for name in dir() if name.isupper()]
