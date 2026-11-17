"""Refresh the existing legal count and best-generator grids, plus grade tables."""
from __future__ import annotations

import csv
import json
import textwrap
from collections import Counter, defaultdict
from fractions import Fraction

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plot_chemistry_questions_by_mode_and_corpus import slide, table
from plot_chemistry_top_models import MODELS
from plot_data import LANGUAGE_NAMES, OUT, ROOT, snapshot_note, snapshot_rows

LEGAL = ROOT / 'data/legal/qac/generation_600_20261008_model_comparison'
MODES = ['claim_verification', 'comparison', 'conceptual', 'fact_pattern_practitioner',
         'lookup', 'source_finding']
MODE_LABELS = ['Claim\nverification', 'Comparison', 'Conceptual', 'Fact pattern /\npractitioner',
               'Lookup', 'Source\nfinding']
LANGUAGES = ['en', 'fr', 'es', 'de', 'zh', 'ar', 'ru']


def mode_of(row):
    return ('fact_pattern_practitioner' if row['mode'] in ('fact_pattern', 'practitioner', 'practitioners')
            else row['mode'])


def score_of(row):
    if row.get('grading_status') == 'failed' or not row.get('total_score', '').strip():
        return None
    score = Fraction(row['total_score'])
    if not 0 <= score <= 40:
        raise ValueError('Grade outside [0,40]')
    return score


def save(fig, stem, pdf):
    for destination in (OUT, LEGAL):
        for suffix in ('svg', 'png', 'pdf'):
            fig.savefig(destination / f'{stem}.{suffix}', dpi=200, bbox_inches='tight')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def write_csv(stem, headers, rows):
    with (OUT / f'legal_{stem}.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def counts_plot(title, labels, columns, values, stem, pdf):
    fig, ax = slide(title, 10)
    # Reuse the established report style with legal labels and an honest snapshot footer.
    ax.texts[0].set_text('Generated legal questions')
    ax.texts[2].set_text('EUR-Lex and UN · all candidate outputs from 10 models')
    table(ax, labels, columns, values, x=270, y=270, width=865,
          row_height=min(43, 325 / len(values)))
    save(fig, stem, pdf)


def main():
    plt.rcParams['svg.fonttype'] = 'none'
    rows = [row for row in snapshot_rows() if row['corpus'] in ('eurlex', 'un')]
    counts = Counter((row['corpus'], mode_of(row)) for row in rows)
    language_counts = Counter((row['question_language'], mode_of(row)) for row in rows)
    corpus_values = [[counts['eurlex', mode], counts['un', mode],
                      counts['eurlex', mode] + counts['un', mode]] for mode in MODES]
    corpus_values.append([sum(counts[source, mode] for mode in MODES)
                          for source in ('eurlex', 'un')] + [len(rows)])
    language_values = [[language_counts[language, mode] for mode in MODES] for language in LANGUAGES]
    language_values.append([sum(language_counts[language, mode] for language in LANGUAGES) for mode in MODES])
    write_csv('questions_by_mode_and_corpus', ['Mode', 'EUR-Lex', 'UN', 'Total'],
              [[label.replace('\n', ' '), *value] for label, value in
               zip([*MODE_LABELS, 'Total'], corpus_values)])
    write_csv('questions_by_language_and_mode', ['Language', *[m.replace('\n', ' ') for m in MODE_LABELS]],
              [[LANGUAGE_NAMES.get(language, 'Total'), *value] for language, value in
               zip([*LANGUAGES, 'total'], language_values)])
    groups = defaultdict(list)
    for row in rows:
        score = score_of(row)
        if score is not None:
            groups[row['question_language'], mode_of(row), row['generator_model_id'].replace('/', '_')].append(score)
    means = {key: sum(values) / len(values) for key, values in groups.items()}
    winners = {}
    with (OUT / 'legal_all_model_scores.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(['language', 'mode', 'model', 'graded_outputs', 'mean_total_score'])
        for (language, mode, model), values in sorted(groups.items()):
            writer.writerow([language, mode, model, len(values), float(means[language, mode, model])])
    for language in LANGUAGES:
        for mode in MODES:
            available = [model for model in MODELS if (language, mode, model) in means]
            highest = max((means[language, mode, model] for model in available), default=None)
            winners[language, mode] = [model for model in available if means[language, mode, model] == highest]
    with PdfPages(OUT / 'legal_generated_questions.pdf') as pdf:
        counts_plot('Generated questions by mode and corpus', [*MODE_LABELS, 'Total'],
                    ['EUR-Lex', 'UN', 'Total'], corpus_values, 'generated_questions_grid', pdf)
        counts_plot('Generated questions by language and mode',
                    [*[LANGUAGE_NAMES[language] for language in LANGUAGES], 'Total'],
                    MODE_LABELS, language_values, 'generated_questions_language_mode_grid', pdf)
        height = 2.5 + 1.15 * len(LANGUAGES)
        fig, ax = plt.subplots(figsize=(17, height), facecolor='#fbfcfe')
        ax.set_xlim(-1.25, len(MODES))
        ax.set_ylim(len(LANGUAGES) + 1, -1.3)
        ax.axis('off')
        ax.text(-1.2, -1.1, 'Top generator model by language and mode', fontsize=20, weight='bold')
        ax.text(-1.2, -.7, 'Highest mean total_score / 40 · fact pattern and practitioner combined', fontsize=10)
        for j, label in enumerate(MODE_LABELS):
            ax.text(j + .5, -.12, label, ha='center', va='bottom', fontsize=10, weight='bold')
        winner_rows = []
        for i, language in enumerate(LANGUAGES):
            ax.text(-.1, i + .5, LANGUAGE_NAMES[language], ha='right', va='center', fontsize=11)
            for j, mode in enumerate(MODES):
                models = winners[language, mode]
                ax.add_patch(plt.Rectangle((j + .01, i + .01), .98, .98,
                             color='#fff1d8' if len(models) > 1 else '#e5f1f8' if models else '#eff1f4'))
                if models:
                    labels = textwrap.fill(' / '.join(MODELS[model] for model in models),
                                           width=29, break_long_words=False, break_on_hyphens=False)
                    mean = float(means[language, mode, models[0]])
                    sizes = '/'.join(str(len(groups[language, mode, model])) for model in models)
                    text = f'{labels}\n{mean:.2f} / 40 · n={sizes}'
                else:
                    mean, sizes, text = '', '', 'No graded samples'
                ax.text(j + .5, i + .5, text, ha='center', va='center',
                        fontsize=7.8 if len(models) > 3 else 8.4, linespacing=1.12)
                winner_rows.append([language, mode, ' | '.join(models), mean, sizes])
        ax.text(-1.2, len(LANGUAGES) + .25, 'Exact unrounded means determine ties; n = graded candidate outputs. Small samples can win.', fontsize=9)
        ax.text(-1.2, len(LANGUAGES) + .55, snapshot_note(), fontsize=8)
        save(fig, 'best_model_language_mode_grid', pdf)
    write_csv('top_generator_by_language_and_mode',
              ['language', 'mode', 'winner_models', 'mean_total_score', 'winner_sample_counts'], winner_rows)
    (OUT / 'legal_plot_summary.json').write_text(json.dumps({
        'generated_questions': len(rows), 'graded_questions': sum(map(len, groups.values())),
        'languages': {language: sum(row['question_language'] == language for row in rows) for language in LANGUAGES},
        'snapshot': snapshot_note(),
    }, indent=2) + '\n')
    print(f'Refreshed three legal grids: {len(rows):,} candidates.')


if __name__ == '__main__':
    main()
