"""Create chemistry equivalents of the legal corpus and language count slides.

Run with:
    .venv/bin/python reports/qa_generation_model_comparison/plot_chemistry_questions_by_mode_and_corpus.py

Every canonical candidate CSV row is one output. Best-question copies,
retry files, failures, and archived/completion files are not added.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/chemistry/ai4chem_abstract_pairs/qac/chemistry_google_patents_model_comparison_generator"
OUT = Path(__file__).resolve().parent
CORPUS = "Google Patents"
MODES = {
    "assertion_verification": "Assertion verification",
    "comparison": "Comparison",
    "conceptual": "Conceptual",
    "constraint_search": "Constraint search",
    "lookup": "Lookup",
    "process_retrieval": "Process retrieval",
    "source_finding": "Source finding",
}
LANGUAGES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German", "zh": "Chinese",
}
WRAPPED_MODES = [
    "Assertion\nverification", "Comparison", "Conceptual", "Constraint\nsearch",
    "Lookup", "Process\nretrieval", "Source\nfinding",
]
BACKGROUND = "#fbf9f6"
SURFACE = "#fbfcfe"
INK = "#1e3449"
LABEL = "#314e69"
MUTED = "#6e839a"
BLUE = LinearSegmentedColormap.from_list("chemistry_blue", ["#eff6fc", "#8eafc6"])
TOTAL_BLUE = LinearSegmentedColormap.from_list("chemistry_total", ["#dce9f3", "#1e5b87"])


def load_counts():
    by_mode = Counter()
    by_language_mode = Counter()
    sources = []
    paths = sorted(DATA.glob("generator_compare_200_*_gemini38.csv"))
    if not paths:
        raise FileNotFoundError(f"No canonical candidate CSVs in {DATA}")
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for line_number, row in enumerate(rows, start=2):
            mode, language = row["mode"], row["question_language"]
            if mode not in MODES or language not in LANGUAGES:
                raise ValueError(f"{path.name}:{line_number}: unexpected mode/language {mode}/{language}")
            if not row["question"].strip():
                raise ValueError(f"{path.name}:{line_number}: empty question")
            by_mode[mode] += 1
            by_language_mode[language, mode] += 1
        sources.append({"file": str(path.relative_to(ROOT)), "candidate_rows": len(rows)})
    for mode in MODES:
        if by_mode[mode] != sum(by_language_mode[lang, mode] for lang in LANGUAGES):
            raise ValueError(f"Counts do not reconcile for {mode}")
    return by_mode, by_language_mode, sources


def slide(title, n_models):
    fig = plt.figure(figsize=(12, 6.75), dpi=100, facecolor=BACKGROUND)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1200)
    ax.set_ylim(675, 0)
    ax.axis("off")
    ax.plot([0, 1200], [11, 11], color="#c9ced1", linewidth=0.5)
    ax.add_patch(Rectangle((0, 12), 12, 18, facecolor="#218d8f", edgecolor="none"))
    ax.text(28, 74, "Generated chemistry questions", fontsize=28,
            family="DejaVu Serif", weight="bold", color=INK, va="top")
    ax.add_patch(Rectangle((24, 140), 1152, 509, facecolor=SURFACE, edgecolor="none"))
    ax.text(44, 167, title, fontsize=17, weight="bold", color=INK, va="top")
    ax.text(44, 203, f"Chemistry · Google Patents · all candidate outputs from {n_models} models",
            fontsize=10, color=MUTED, va="top")
    ax.text(44, 627,
            "Counts include every candidate row; identical question text from different models is counted separately.",
            fontsize=8.1, color=MUTED, va="center")
    return fig, ax


def table(ax, row_labels, column_labels, values, *, x, y, width, row_height):
    """Draw rounded heatmap cells, with the final row representing totals."""
    cell_width = width / len(column_labels)
    body_max = max(max(row) for row in values[:-1]) or 1
    total_max = max(values[-1]) or 1
    for j, label in enumerate(column_labels):
        ax.text(x + (j + 0.5) * cell_width, y - 21, label,
                ha="center", va="center", fontsize=10.2 if len(column_labels) < 4 else 9.2,
                weight="bold", color=LABEL, linespacing=1.2)
    for i, (label, row) in enumerate(zip(row_labels, values)):
        is_total = i == len(values) - 1
        center_y = y + (i + 0.5) * row_height
        ax.text(x - 18, center_y, label, ha="right", va="center",
                fontsize=10.3, weight="bold" if is_total else "normal", color=LABEL)
        for j, value in enumerate(row):
            shade = value / (total_max if is_total else body_max)
            color = TOTAL_BLUE(shade) if is_total else BLUE(shade)
            ax.add_patch(FancyBboxPatch(
                (x + j * cell_width + 1.5, y + i * row_height + 1.5),
                cell_width - 3, row_height - 3,
                boxstyle="round,pad=0,rounding_size=4",
                facecolor=color, edgecolor="#45657f" if is_total else "none",
                linewidth=1.1 if is_total else 0,
            ))
            ax.text(x + (j + 0.5) * cell_width, center_y, f"{value:,}",
                    ha="center", va="center", fontsize=12, weight="bold",
                    color="white" if is_total and shade > 0.55 else INK)


def write_csv(stem, headers, rows):
    with (OUT / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def save_figure(fig, stem, combined_pdf):
    fig.savefig(OUT / f"{stem}.png", dpi=200, facecolor=BACKGROUND)
    fig.savefig(OUT / f"{stem}.pdf", facecolor=BACKGROUND)
    combined_pdf.savefig(fig, facecolor=BACKGROUND)
    plt.close(fig)


def main():
    by_mode, by_language_mode, sources = load_counts()
    grand_total = sum(by_mode.values())
    n_models = len(sources)
    corpus_stem = "chemistry_questions_by_mode_and_corpus"
    language_stem = "chemistry_questions_by_language_and_mode"
    corpus_values = [[by_mode[mode], by_mode[mode]] for mode in MODES]
    corpus_values.append([grand_total, grand_total])
    corpus_labels = [*MODES.values(), "Total"]
    language_values = [
        [by_language_mode[language, mode] for mode in MODES] for language in LANGUAGES
    ]
    language_values.append([by_mode[mode] for mode in MODES])
    language_labels = [*LANGUAGES.values(), "Total"]

    write_csv(corpus_stem, ["Mode", CORPUS, "Total"],
              [[label, *row] for label, row in zip(corpus_labels, corpus_values)])
    write_csv(language_stem, ["Language", *MODES.values()],
              [[label, *row] for label, row in zip(language_labels, language_values)])

    with PdfPages(OUT / "chemistry_generated_questions.pdf") as combined_pdf:
        fig, ax = slide("Generated questions by mode and corpus", n_models)
        table(ax, corpus_labels, [CORPUS, "Total"], corpus_values,
              x=420, y=278, width=620, row_height=40)
        save_figure(fig, corpus_stem, combined_pdf)
        fig, ax = slide("Generated questions by language and mode", n_models)
        table(ax, language_labels, WRAPPED_MODES, language_values,
              x=246, y=286, width=890, row_height=52)
        save_figure(fig, language_stem, combined_pdf)

    manifest = {
        "domain": "chemistry", "corpus": CORPUS,
        "source_directory": str(DATA.relative_to(ROOT)),
        "input_pattern": "generator_compare_200_*_gemini38.csv (top level only)",
        "counting_unit": "Every candidate row; question text is not deduplicated across models or candidates.",
        "model_count": n_models, "total_candidate_rows": grand_total,
        "counts_by_mode": {mode: by_mode[mode] for mode in MODES},
        "counts_by_language": {
            lang: sum(by_language_mode[lang, mode] for mode in MODES) for lang in LANGUAGES
        },
        "sources": sources,
    }
    (OUT / "chemistry_question_counts_sources.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created both chemistry count slides: {grand_total:,} candidate outputs, {n_models} models.")
    print(f"Source: {DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
