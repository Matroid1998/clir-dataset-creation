"""Separate UN and EUR-Lex question-count and best-generator figures."""
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
CORPORA = {
    'eurlex': {'title': 'EUR-Lex', 'languages': ['en', 'fr', 'es', 'de'], 'directory': 'eurlex',
               'modes': {'claim_verification': 'Claim\nverification', 'comparison': 'Comparison',
                         'conceptual': 'Conceptual', 'fact_pattern': 'Fact pattern',
                         'lookup': 'Lookup', 'source_finding': 'Source\nfinding'}},
    'un': {'title': 'UN', 'languages': ['en', 'fr', 'es', 'zh', 'ar', 'ru'], 'directory': 'un_parallel',
           'modes': {'claim_verification': 'Claim\nverification', 'comparison': 'Comparison',
                     'conceptual': 'Conceptual', 'practitioners': 'Practitioner',
                     'lookup': 'Lookup', 'source_finding': 'Source\nfinding'}},
}


def mode_of(row):
    return 'practitioners' if row['mode'] == 'practitioner' else row['mode']


def score_of(row):
    if row.get('grading_status') == 'failed' or not row.get('total_score', '').strip():
        return None
    score = Fraction(row['total_score'])
    if not 0 <= score <= 40:
        raise ValueError('Grade outside [0,40]')
    return score


def analyze(rows, corpus):
    """Keep all counts and grade comparisons within the selected corpus."""
    definition = CORPORA[corpus]
    selected = [row for row in rows if row['corpus'] == corpus]
    counts, groups = Counter(), defaultdict(list)
    for row in selected:
        language, mode = row['question_language'], mode_of(row)
        if language not in definition['languages'] or mode not in definition['modes']:
            raise ValueError(f'Unsupported {corpus} language/mode: {language}/{mode}')
        counts[language, mode] += 1
        score = score_of(row)
        if score is not None:
            groups[language, mode, row['generator_model_id'].replace('/', '_')].append(score)
    means = {key: sum(values) / len(values) for key, values in groups.items()}
    winners = {}
    for language in definition['languages']:
        for mode in definition['modes']:
            available = [model for model in MODELS if (language, mode, model) in means]
            highest = max((means[language, mode, model] for model in available), default=None)
            winners[language, mode] = [model for model in available if means[language, mode, model] == highest]
    return selected, counts, groups, means, winners


def save(fig, corpus, stem, pdfs):
    directory = LEGAL / CORPORA[corpus]['directory']
    directory.mkdir(parents=True, exist_ok=True)
    for destination in (OUT, directory):
        for suffix in ('svg', 'png', 'pdf'):
            fig.savefig(destination / f'{corpus}_{stem}.{suffix}', dpi=200, bbox_inches='tight')
    for pdf in pdfs:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def write_csv(corpus, stem, headers, rows):
    with (OUT / f'{corpus}_{stem}.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(headers)
        writer.writerows(rows)


def counts_plot(corpus, title, labels, columns, values, stem, pdfs):
    fig, ax = slide(title, 10)
    name = CORPORA[corpus]['title']
    ax.texts[0].set_text(f'Generated {name} questions')
    ax.texts[2].set_text(f'{name} · all candidate outputs from 10 models')
    table(ax, labels, columns, values, x=270, y=270, width=865,
          row_height=min(43, 325 / len(values)))
    save(fig, corpus, stem, pdfs)


def plot_corpus(corpus, rows, combined_pdf):
    definition = CORPORA[corpus]
    title, languages, modes = definition['title'], definition['languages'], definition['modes']
    selected, counts, groups, means, winners = analyze(rows, corpus)
    mode_totals = {mode: sum(counts[language, mode] for language in languages) for mode in modes}
    mode_values = [[mode_totals[mode]] for mode in modes] + [[len(selected)]]
    language_values = [[*[counts[language, mode] for mode in modes],
                         sum(counts[language, mode] for mode in modes)] for language in languages]
    language_values.append([*[mode_totals[mode] for mode in modes], len(selected)])
    write_csv(corpus, 'questions_by_mode', ['Mode', 'Questions'],
              [[label.replace('\n', ' '), *value] for label, value in zip([*modes.values(), 'Total'], mode_values)])
    write_csv(corpus, 'questions_by_language_and_mode',
              ['Language', *[label.replace('\n', ' ') for label in modes.values()], 'Total'],
              [[LANGUAGE_NAMES.get(language, 'Total'), *value] for language, value in
               zip([*languages, 'total'], language_values)])
    write_csv(corpus, 'all_model_scores',
              ['language', 'mode', 'model', 'graded_outputs', 'mean_total_score'],
              [[language, mode, model, len(values), float(means[language, mode, model])]
               for (language, mode, model), values in sorted(groups.items())])
    with PdfPages(OUT / f'{corpus}_generated_questions.pdf') as corpus_pdf:
        pdfs = (corpus_pdf, combined_pdf)
        counts_plot(corpus, f'{title}: generated questions by mode', [*modes.values(), 'Total'],
                    ['Questions'], mode_values, 'generated_questions_by_mode', pdfs)
        counts_plot(corpus, f'{title}: generated questions by language and mode',
                    [*[LANGUAGE_NAMES[language] for language in languages], 'Total'],
                    [*modes.values(), 'Total'], language_values, 'generated_questions_by_language_and_mode', pdfs)
        fig, ax = plt.subplots(figsize=(17, 2.5 + 1.15 * len(languages)), facecolor='#fbfcfe')
        ax.set_xlim(-1.25, len(modes))
        ax.set_ylim(len(languages) + 1, -1.3)
        ax.axis('off')
        ax.text(-1.2, -1.1, f'{title}: top generator model by language and mode', fontsize=19, weight='bold')
        ax.text(-1.2, -.7, f'Highest mean total_score / 40 · {title} candidates only', fontsize=10)
        for j, label in enumerate(modes.values()):
            ax.text(j + .5, -.12, label, ha='center', va='bottom', fontsize=10, weight='bold')
        winner_rows = []
        for i, language in enumerate(languages):
            ax.text(-.1, i + .5, LANGUAGE_NAMES[language], ha='right', va='center', fontsize=11)
            for j, mode in enumerate(modes):
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
        ax.text(-1.2, len(languages) + .25, 'Exact unrounded means determine ties; n = graded candidate outputs. Small samples can win.', fontsize=9)
        ax.text(-1.2, len(languages) + .55, snapshot_note(), fontsize=8)
        save(fig, corpus, 'best_model_language_mode_grid', pdfs)
    write_csv(corpus, 'top_generator_by_language_and_mode',
              ['language', 'mode', 'winner_models', 'mean_total_score', 'winner_sample_counts'], winner_rows)
    summary = {'corpus': corpus, 'generated_questions': len(selected),
               'graded_questions': sum(map(len, groups.values())),
               'languages': {language: sum(counts[language, mode] for mode in modes) for language in languages},
               'snapshot': snapshot_note()}
    (OUT / f'{corpus}_plot_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    plt.rcParams['svg.fonttype'] = 'none'
    summaries = {}
    rows = snapshot_rows()
    with PdfPages(OUT / 'legal_generated_questions.pdf') as combined_pdf:
        for corpus in CORPORA:
            summaries[corpus] = plot_corpus(corpus, rows, combined_pdf)
    (OUT / 'legal_plot_summary.json').write_text(json.dumps({'corpora': summaries}, indent=2) + '\n')
    print('Refreshed separate legal grids:', {corpus: value['generated_questions']
                                             for corpus, value in summaries.items()})


if __name__ == '__main__':
    main()
