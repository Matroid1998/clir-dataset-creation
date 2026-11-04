"""Plot all generated chemistry candidates by persona and generator."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/chemistry/ai4chem_abstract_pairs/qac/chemistry_google_patents_model_comparison_generator"
OUT = Path(__file__).resolve().parent

LABELS = {
    "anthropic_claude-haiku-5.5": "Claude Haiku\n5.5",
    "deepseek_deepseek-v4.1-flash": "DeepSeek\nv4.1 Flash",
    "gpt-5.6-luna": "GPT-5.6\nLuna",
    "minimax_minimax-m2.7": "MiniMax\nM2.7",
    "mistralai_mistral-small-2603": "Mistral Small\n2603",
    "nvidia_nemotron-3.5-lightning": "Nemotron 3.5\nLightning",
    "prism-ml_ternary-bonsai-2-27b": "Ternary Bonsai\n2 27B",
    "qwen_qwen3.8-27b": "Qwen3.8\n27B",
    "upstage_solar-pro4": "Solar Pro4",
    "z-ai_glm-5.3-flashx": "GLM-5.3\nFlashX",
}
MODES = [
    "assertion_verification", "comparison", "conceptual", "constraint_search",
    "lookup", "process_retrieval", "source_finding",
]
MODE_LABELS = {
    "assertion_verification": "Assertion verification",
    "comparison": "Comparison",
    "conceptual": "Conceptual",
    "constraint_search": "Constraint search",
    "lookup": "Lookup",
    "process_retrieval": "Process retrieval",
    "source_finding": "Source finding",
}
LANGUAGES = ["en", "fr", "es", "de", "zh"]
LANGUAGE_LABELS = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German", "zh": "Chinese",
}
SURFACE = "#fcfcfb"
INK = "#14283b"
MUTED = "#667b8d"
BLUE = LinearSegmentedColormap.from_list("table_blue", ["#f3f8fc", "#bfd3e2", "#739cb9"])
TOTAL_BLUE = LinearSegmentedColormap.from_list("total_blue", ["#e1edf5", "#1e5e88"])


def read_counts():
    counts: dict[str, dict[str, int]] = {}
    models: dict[str, int] = {}
    for path in DATA.glob("generator_compare_200_*_gemini38.csv"):
        label = path.name.removeprefix("generator_compare_200_").removesuffix("_gemini38.csv")
        if label not in LABELS:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        counts[label] = {mode: 0 for mode in MODES}
        for row in rows:
            mode = row.get("mode", "")
            if mode in counts[label]:
                counts[label][mode] += 1
        models[label] = len(rows)
    missing = set(LABELS) - set(counts)
    if missing:
        raise FileNotFoundError(f"Missing best-question CSVs for: {', '.join(sorted(missing))}")
    order = sorted(LABELS, key=lambda model: (-models[model], model))
    return counts, models, order


def read_language_mode_counts():
    counts = {language: {mode: 0 for mode in MODES} for language in LANGUAGES}
    for path in DATA.glob("generator_compare_200_*_gemini38.csv"):
        label = path.name.removeprefix("generator_compare_200_").removesuffix("_gemini38.csv")
        if label not in LABELS:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                language, mode = row.get("question_language", ""), row.get("mode", "")
                if language in counts and mode in counts[language]:
                    counts[language][mode] += 1
    return counts


def plot_language_mode():
    counts = read_language_mode_counts()
    row_totals = {language: sum(counts[language].values()) for language in LANGUAGES}
    col_totals = {mode: sum(counts[language][mode] for language in LANGUAGES)
                  for mode in MODES}
    grand_total = sum(row_totals.values())

    csv_path = OUT / "generated_questions_by_language_and_mode.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Language", *[MODE_LABELS[m] for m in MODES], "Total"])
        for language in LANGUAGES:
            writer.writerow([LANGUAGE_LABELS[language],
                             *[counts[language][mode] for mode in MODES], row_totals[language]])
        writer.writerow(["Total", *[col_totals[mode] for mode in MODES], grand_total])

    fig, ax = plt.subplots(figsize=(14.5, 6.7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x0, cell_w, row_h, header_y = 2.35, 1.06, 0.68, 1.65
    total_x = x0 + len(MODES) * cell_w
    ax.set_xlim(-0.02, total_x + cell_w + 0.05)
    ax.set_ylim(-0.12, len(LANGUAGES) + 1.85)
    ax.invert_yaxis()
    ax.axis("off")
    ax.text(0, 0.02, "Generated questions by language and mode", fontsize=19,
            weight="bold", color=INK, va="top")
    ax.text(0, 0.63, "All candidate outputs · Google Patents model comparison · 10 models",
            fontsize=10.5, color=MUTED, va="top")

    for j, mode in enumerate(MODES):
        ax.text(x0 + j * cell_w + (cell_w - 0.025) / 2, header_y,
                MODE_LABELS[mode], ha="center", va="bottom", fontsize=8.8,
                weight="bold", color=INK, wrap=True)
    ax.text(total_x + (cell_w - 0.025) / 2, header_y, "Total", ha="center", va="bottom",
            fontsize=9.3, weight="bold", color=INK)

    max_cell = max(counts[lang][mode] for lang in LANGUAGES for mode in MODES) or 1
    min_cell = min(counts[lang][mode] for lang in LANGUAGES for mode in MODES)
    span = max(max_cell - min_cell, 1)
    max_row = max(row_totals.values()) or 1
    max_col = max(col_totals.values()) or 1

    def draw_cell(x, y, value, color, text_color=INK, edge=SURFACE, width=1.0):
        ax.add_patch(plt.Rectangle((x, y), cell_w - 0.025, row_h - 0.035,
                     facecolor=color, edgecolor=edge, linewidth=width))
        ax.text(x + (cell_w - 0.025) / 2, y + (row_h - 0.035) / 2, f"{value:,}",
                ha="center", va="center", fontsize=9.5, weight="bold", color=text_color)

    for i, language in enumerate(LANGUAGES):
        y = header_y + 0.24 + i * row_h
        ax.text(x0 - 0.15, y + (row_h - 0.035) / 2, LANGUAGE_LABELS[language],
                ha="right", va="center", fontsize=10, color=INK)
        for j, mode in enumerate(MODES):
            value = counts[language][mode]
            shade = 0.08 + 0.9 * (value - min_cell) / span
            draw_cell(x0 + j * cell_w, y, value, BLUE(shade))
        draw_cell(total_x, y, row_totals[language],
                  TOTAL_BLUE(0.2 + 0.72 * row_totals[language] / max_row), "white")

    total_y = header_y + 0.24 + len(LANGUAGES) * row_h
    ax.text(x0 - 0.15, total_y + (row_h - 0.035) / 2, "Total", ha="right", va="center",
            fontsize=10.5, weight="bold", color=INK)
    for j, mode in enumerate(MODES):
        draw_cell(x0 + j * cell_w, total_y, col_totals[mode],
                  TOTAL_BLUE(0.3 + 0.68 * col_totals[mode] / max_col), "white",
                  edge="#52758f", width=1.1)
    draw_cell(total_x, total_y, grand_total, "#1e5e88", "white", edge="#315c79", width=1.2)

    fig.text(0.055, 0.025,
             "Every generated candidate is counted. Repeated question text across models "
             "is counted once per model output.", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.97, bottom=0.075)
    fig.savefig(OUT / "generated_questions_by_language_and_mode.png", dpi=220,
                facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(OUT / "generated_questions_by_language_and_mode.pdf",
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote language-mode plot and CSV; {grand_total:,} candidates")


def main():
    counts, models, order = read_counts()
    column_totals = {model: models[model] for model in order}
    row_totals = {mode: sum(counts[model][mode] for model in order) for mode in MODES}
    grand_total = sum(column_totals.values())

    output = OUT / "generated_questions_by_mode_and_model.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Mode", *[LABELS[m].replace("\n", " ") for m in order], "Total"])
        for mode in MODES:
            writer.writerow([MODE_LABELS[mode], *[counts[m][mode] for m in order], row_totals[mode]])
        writer.writerow(["Total", *[column_totals[m] for m in order], grand_total])

    fig, ax = plt.subplots(figsize=(18, 7.3), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.02, 14.05)
    ax.set_ylim(-0.12, len(MODES) + 1.55)
    ax.invert_yaxis()
    ax.axis("off")

    x0, cell_w, row_h = 2.75, 1.0, 0.72
    header_y = 1.65
    ax.text(0, 0.02, "Generated questions by mode and model", fontsize=20,
            weight="bold", color=INK, va="top")
    ax.text(0, 0.63, "All candidate outputs · Google Patents model comparison · Gemini 3.8 Flash grading",
            fontsize=11, color=MUTED, va="top")

    for j, model in enumerate(order):
        x = x0 + j * cell_w
        ax.text(x + cell_w / 2, header_y, LABELS[model], ha="center", va="bottom",
                fontsize=9.4, weight="bold", color=INK, linespacing=1.1)
    total_x = x0 + len(order) * cell_w
    ax.text(total_x + cell_w / 2, header_y, "Total", ha="center", va="bottom",
            fontsize=10, weight="bold", color=INK)

    max_cell = max(counts[m][mode] for m in order for mode in MODES) or 1
    min_cell = min(counts[m][mode] for m in order for mode in MODES)
    cell_span = max(max_cell - min_cell, 1)
    max_row_total = max(row_totals.values()) or 1
    max_model_total = max(column_totals.values()) or 1
    for i, mode in enumerate(MODES):
        y = header_y + 0.24 + i * row_h
        ax.text(x0 - 0.15, y + row_h / 2, MODE_LABELS[mode], ha="right", va="center",
                fontsize=10.3, color=INK)
        for j, model in enumerate(order):
            x = x0 + j * cell_w
            value = counts[model][mode]
            shade = 0.08 + 0.9 * (value - min_cell) / cell_span
            ax.add_patch(plt.Rectangle((x, y), cell_w - 0.025, row_h - 0.035,
                         facecolor=BLUE(shade),
                         edgecolor=SURFACE, linewidth=1.1))
            ax.text(x + (cell_w - 0.025) / 2, y + (row_h - 0.035) / 2, f"{value:,}",
                    ha="center", va="center", fontsize=10, weight="bold", color=INK)
        value = row_totals[mode]
        ax.add_patch(plt.Rectangle((total_x, y), cell_w - 0.025, row_h - 0.035,
                     facecolor=TOTAL_BLUE(0.2 + 0.72 * value / max_row_total),
                     edgecolor=SURFACE, linewidth=1.1))
        ax.text(total_x + (cell_w - 0.025) / 2, y + (row_h - 0.035) / 2, f"{value:,}",
                ha="center", va="center", fontsize=10, weight="bold", color="white")

    total_y = header_y + 0.24 + len(MODES) * row_h
    ax.text(x0 - 0.15, total_y + row_h / 2, "Total", ha="right", va="center",
            fontsize=10.5, weight="bold", color=INK)
    for j, model in enumerate(order):
        x = x0 + j * cell_w
        value = column_totals[model]
        ax.add_patch(plt.Rectangle((x, total_y), cell_w - 0.025, row_h - 0.035,
                     facecolor=TOTAL_BLUE(0.3 + 0.68 * value / max_model_total),
                     edgecolor="#52758f", linewidth=1.15))
        ax.text(x + (cell_w - 0.025) / 2, total_y + (row_h - 0.035) / 2, f"{value:,}",
                ha="center", va="center", fontsize=10, weight="bold", color="white")
    ax.add_patch(plt.Rectangle((total_x, total_y), cell_w - 0.025, row_h - 0.035,
                 facecolor="#1e5e88", edgecolor="#315c79", linewidth=1.2))
    ax.text(total_x + (cell_w - 0.025) / 2, total_y + (row_h - 0.035) / 2, f"{grand_total:,}",
            ha="center", va="center", fontsize=10.5, weight="bold", color="white")

    fig.text(0.06, 0.025,
             "Every generated candidate is counted, including multiple candidates per case. "
             "Repeated question text across models is counted once for each model.",
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.97, bottom=0.075)
    fig.savefig(OUT / "generated_questions_by_mode_and_model.png", dpi=220,
                facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(OUT / "generated_questions_by_mode_and_model.pdf",
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'generated_questions_by_mode_and_model.png'}")
    print(f"wrote {output}; {grand_total:,} candidate rows across {len(order)} models")
    plot_language_mode()


if __name__ == "__main__":
    main()
