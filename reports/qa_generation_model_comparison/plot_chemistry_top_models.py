"""Plot the chemistry generator with the highest mean grade per language/mode.

Uses the same canonical candidate files as the chemistry question-count slides.
Rankings and ties use exact, unrounded means of total_score (out of 40).
"""

from __future__ import annotations

import csv
import json
import textwrap
from collections import defaultdict
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from plot_chemistry_questions_by_mode_and_corpus import (
    DATA, INK, LABEL, LANGUAGES, MODES, MUTED, OUT, ROOT, WRAPPED_MODES,
)

MODELS = {
    "anthropic_claude-haiku-5.5": "Claude Haiku",
    "deepseek_deepseek-v4.1-flash": "DeepSeek",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "minimax_minimax-m2.7": "MiniMax M2.7",
    "mistralai_mistral-small-2603": "Mistral Small",
    "nvidia_nemotron-3.5-lightning": "Nemotron 3.5",
    "prism-ml_ternary-bonsai-2-27b": "Ternary Bonsai",
    "qwen_qwen3.8-27b": "Qwen3.8-27B",
    "upstage_solar-pro4": "Solar Pro4",
    "z-ai_glm-5.3-flashx": "GLM-5.3",
}
STEM = "chemistry_top_generator_by_language_and_mode"
SINGLE = "#e5f1f8"
TIE = "#fff1d8"
EMPTY = "#eff1f4"
TIE_INK = "#6e4a1a"


def load_grades():
    groups = defaultdict(list)
    sources = []
    seen_models = set()
    rubric_versions = set()
    ungraded = 0
    for path in sorted(DATA.glob("generator_compare_200_*_gemini38.csv")):
        model = path.name.removeprefix("generator_compare_200_").removesuffix("_gemini38.csv")
        if model not in MODELS:
            raise ValueError(f"Add a display label for model {model}")
        seen_models.add(model)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        n_graded = 0
        for line, row in enumerate(rows, start=2):
            language, mode = row["question_language"], row["mode"]
            if language not in LANGUAGES or mode not in MODES:
                raise ValueError(f"{path.name}:{line}: unknown language/mode")
            if not row["total_score"].strip():
                ungraded += 1
                continue
            score = Fraction(row["total_score"])
            if not 0 <= score <= 40:
                raise ValueError(f"{path.name}:{line}: score outside [0, 40]")
            if score != Fraction(row["faith_overall"]) + Fraction(row["qual_overall"]):
                raise ValueError(f"{path.name}:{line}: total_score disagrees with its components")
            rubric_versions.add((row["faithfulness_rubric_version"], row["quality_rubric_version"]))
            groups[language, mode, model].append(score)
            n_graded += 1
        sources.append({"file": str(path.relative_to(ROOT)), "rows": len(rows), "graded_rows": n_graded})
    if seen_models != set(MODELS):
        raise ValueError(f"Missing model CSVs: {sorted(set(MODELS) - seen_models)}")
    if len(rubric_versions) != 1:
        raise ValueError(f"Expected comparable rubric versions, found {rubric_versions}")
    means = {key: sum(scores) / len(scores) for key, scores in groups.items()}
    winners = {}
    for language in LANGUAGES:
        for mode in MODES:
            available = [model for model in MODELS if (language, mode, model) in means]
            if not available:
                winners[language, mode] = []
                continue
            highest = max(means[language, mode, model] for model in available)
            winners[language, mode] = [model for model in available if means[language, mode, model] == highest]
    return groups, means, winners, sources, rubric_versions, ungraded


def write_data(groups, means, winners, sources, rubric_versions, ungraded):
    with (OUT / f"{STEM}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", "mode", "status", "winner_models", "mean_total_score",
                         "score_out_of", "winner_sample_counts", "models_with_grades"])
        for (language, mode), models in winners.items():
            score = float(means[language, mode, models[0]]) if models else ""
            status = "tie" if len(models) > 1 else "winner" if models else "no_graded_samples"
            writer.writerow([
                language, mode, status, " | ".join(models), score, 40,
                " | ".join(str(len(groups[language, mode, model])) for model in models),
                sum((language, mode, model) in means for model in MODELS),
            ])
    with (OUT / f"{STEM}_all_model_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", "mode", "model", "graded_outputs", "sum_total_score",
                         "mean_total_score", "exact_mean", "is_winner"])
        for language in LANGUAGES:
            for mode in MODES:
                for model in MODELS:
                    key = language, mode, model
                    mean = means.get(key)
                    scores = groups.get(key, [])
                    writer.writerow([
                        language, mode, model, len(scores), float(sum(scores)),
                        float(mean) if mean is not None else "", str(mean) if mean is not None else "",
                        model in winners[language, mode],
                    ])
    manifest = {
        "domain": "chemistry", "corpus": "Google Patents",
        "source_directory": str(DATA.relative_to(ROOT)),
        "metric": "Arithmetic mean of total_score over all graded candidate rows per model/language/mode",
        "score_out_of": 40, "ties": "Exact unrounded means, compared as rational numbers",
        "selection": "Top-level generator_compare_200_*_gemini38.csv; excludes best, retry, failure, and archive copies",
        "graded_outputs": sum(map(len, groups.values())), "ungraded_outputs": ungraded,
        "rubric_versions": sorted(rubric_versions), "model_labels": MODELS, "sources": sources,
    }
    (OUT / f"{STEM}_sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def plot(groups, means, winners):
    fig = plt.figure(figsize=(18, 10.3), dpi=100, facecolor="#fbfcfe")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1800)
    ax.set_ylim(1030, 0)
    ax.axis("off")
    ax.plot([0, 1800], [11, 11], color="#c9ced1", linewidth=0.5)
    ax.add_patch(Rectangle((0, 12), 16, 20, facecolor="#218d8f", edgecolor="none"))
    ax.text(40, 48, "Top chemistry generator model by language and mode",
            fontsize=23, weight="bold", color=INK, va="top")
    ax.text(40, 96,
            f"Google Patents · highest average full grade (total_score, out of 40) · {len(MODELS)} generator models · Gemini 3.8 Flash grading",
            fontsize=11.5, color=MUTED, va="top")

    x0, y0, width, row_height = 154, 200, 1608, 140
    cell_width = width / len(MODES)
    for j, label in enumerate(WRAPPED_MODES):
        ax.text(x0 + (j + 0.5) * cell_width, y0 - 31, label, ha="center", va="center",
                fontsize=11.3, weight="bold", color=LABEL, linespacing=1.15)
    for i, (language, language_label) in enumerate(LANGUAGES.items()):
        y = y0 + i * row_height
        ax.text(x0 - 20, y + row_height / 2, language_label, ha="right", va="center",
                fontsize=12, weight="bold", color=LABEL)
        for j, mode in enumerate(MODES):
            x = x0 + j * cell_width
            models = winners[language, mode]
            tied = len(models) > 1
            fill = TIE if tied else SINGLE if models else EMPTY
            ax.add_patch(FancyBboxPatch(
                (x + 1.5, y + 1.5), cell_width - 3, row_height - 3,
                boxstyle="round,pad=0,rounding_size=4", facecolor=fill, edgecolor="none",
            ))
            if not models:
                ax.text(x + cell_width / 2, y + row_height / 2, "No graded samples",
                        ha="center", va="center", fontsize=10.5, color=MUTED)
                continue
            average = float(means[language, mode, models[0]])
            counts = "/".join(str(len(groups[language, mode, model])) for model in models)
            if tied:
                ax.text(x + 12, y + 13, f"{len(models)}-model tie", va="top",
                        fontsize=11.5, weight="bold", color=TIE_INK)
                names = textwrap.fill(" / ".join(MODELS[model] for model in models),
                                      width=30, break_long_words=False, break_on_hyphens=False)
                ax.text(x + 12, y + 38, names, va="top", fontsize=9.8,
                        color=TIE_INK, linespacing=1.2)
                ax.text(x + 12, y + row_height - 14, f"Avg {average:.2f} / 40 · n={counts}",
                        va="bottom", fontsize=9.3, color=TIE_INK)
            else:
                ax.text(x + 12, y + 48, MODELS[models[0]], va="top",
                        fontsize=11.5, weight="bold", color=INK)
                ax.text(x + 12, y + 77, f"Avg {average:.2f} / 40 · n={counts}",
                        va="top", fontsize=10.1, color=LABEL)

    notes = [
        "Blue = one winner · amber = exact tie · gray = no graded samples. Ties are determined before rounding scores.",
        "n = graded candidate rows per winner; tie counts follow the displayed model order. "
        f"All {sum(map(len, groups.values())):,} graded outputs are included.",
        "Some winning averages use only 1–3 outputs; sample sizes differ across models and language–mode pairs.",
        "Short labels: Claude Haiku 5.5 · DeepSeek v4.1 Flash · GLM-5.3 FlashX · Ternary Bonsai 2 27B. Full model IDs are in the CSV.",
    ]
    for k, note in enumerate(notes):
        ax.text(40, 936 + k * 22, note, fontsize=9.8, color=MUTED, va="top")
    fig.savefig(OUT / f"{STEM}.png", dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(OUT / f"{STEM}.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    groups, means, winners, sources, rubrics, ungraded = load_grades()
    write_data(groups, means, winners, sources, rubrics, ungraded)
    plot(groups, means, winners)
    print(f"Wrote {OUT / (STEM + '.png')}")
    print(f"{sum(map(len, groups.values())):,} graded outputs; "
          f"{sum(len(v) == 1 for v in winners.values())} sole winners; "
          f"{sum(len(v) > 1 for v in winners.values())} tied cells; "
          f"{sum(not v for v in winners.values())} cells without grades.")


if __name__ == "__main__":
    main()
