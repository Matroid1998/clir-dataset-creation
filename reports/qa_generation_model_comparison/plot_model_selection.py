"""Render the reproducible model-selection analysis; never apply generation routes."""
from __future__ import annotations

import csv
import json
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from model_selection_analysis import PRACTICAL_MARGIN, PREFERRED, RESULTS
from plot_chemistry_questions_by_mode_and_corpus import LANGUAGES, MODES
from plot_chemistry_top_models import MODELS
from plot_legal_questions import CORPORA

SPECS = {**CORPORA, 'chemistry': {'title': 'Google Patents',
                                'languages': list(LANGUAGES), 'modes': MODES}}
NAMES = {**LANGUAGES, 'ar': 'Arabic'}
INK = '#20374b'


def name(model):
    return MODELS.get(model.replace('/', '_'), model)


def number(value):
    return float(value) if value not in ('', None) else None


def page(title, subtitle):
    fig = plt.figure(figsize=(12, 8), facecolor='#fbfcfe')
    fig.text(.06, .94, title, fontsize=23, weight='bold', color=INK)
    fig.text(.06, .90, subtitle, fontsize=11, color='#506477')
    fig.text(.06, .025, 'Generated-question model selection | frozen local data | 17 November 2026',
             fontsize=8, color='#506477')
    return fig


def save(fig, stem, pdf):
    fig.savefig(RESULTS / f'{stem}.png', dpi=180)
    pdf.savefig(fig)
    plt.close(fig)


def overview(rows, summary, pdf):
    margin = summary['practical_margin']
    fig = page('Use Haiku or Qwen as the preferred generators',
               'A preference policy supported by current scores, with explicit uncertainty flags')
    text = [
        f"{summary['within_raw_margin'][str(margin)]} of {len(rows)} populated cells have Haiku or Qwen within {margin:g} points of the observed leader.",
        'No alternative meets the paired-evidence rule for switching at 0.5, 1 or 2 points.',
        'EUR-Lex has the strongest coverage. All Google Patents choices remain provisional.',
        f"Choosing the best candidate instead of the average changes {summary['best_candidate_route_changes']} Haiku/Qwen routes; no alternative qualifies.",
    ]
    for y, line in zip((.81, .75, .69, .63), text):
        fig.text(.07, y, line, fontsize=12, color=INK)
    ax = fig.add_axes((.08, .29, .84, .25))
    ax.axis('off')
    values = []
    for corpus, spec in SPECS.items():
        selected = [r for r in rows if r['corpus'] == corpus]
        values.append([spec['title'], len(selected),
                       sum(r['recommended_model'] == PREFERRED[0] for r in selected),
                       sum(r['recommended_model'] == PREFERRED[1] for r in selected),
                       sum(r['evidence'] == 'provisional' for r in selected),
                       len(spec['languages']) * len(spec['modes']) - len(selected)])
    table = ax.table(cellText=values, colLabels=['Corpus', 'Scored cells', 'Haiku', 'Qwen',
                                                'Provisional', 'No data'],
                     cellLoc='center', loc='center', colWidths=[.27, .15, .12, .12, .17, .17])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    fig.text(.08, .19, f'The margin is {margin:g} points on the 0–40 combined grade ({100 * margin / 40:g}% of the scale).', fontsize=11)
    fig.text(.08, .14, 'This is an explicit practical threshold, not a finding that the models are equivalent.', fontsize=11)
    fig.text(.08, .09, 'No active generation settings were changed. Review the suggested routes before using them.', fontsize=11)
    save(fig, 'selection_overview', pdf)


def methodology(pdf):
    fig = page('How to read the margins', 'Compare within a corpus, mode and language; UN and EUR-Lex are separate')
    paragraphs = [
        ('Score unit', 'Average the successfully graded candidates for each input target/run, then average those input scores. A model that emits more candidates does not gain more weight. UN targets can be blocks; independent uncertainty units are their parent documents.'),
        ('Shared-input comparison', 'Compare models on the exact same target and experiment. Bootstrap 10,000 resamples of independent UN documents, EUR-Lex acts or patent publications, keeping their blocks/runs together. Suppress intervals below five independent documents.'),
        ('Choosing Haiku versus Qwen', 'Use the higher paired mean when at least five parent documents overlap; otherwise use the higher observed input-balanced mean. Mark the route provisional when fewer than ten independent documents support the choice or the paired Haiku/Qwen comparison.'),
        ('When another model can replace both', f'Require at least ten paired independent documents against each available preferred model, with a one-sided, within-cell multiplicity-adjusted bootstrap lower bound above the practical margin. The default margin is {PRACTICAL_MARGIN:g} points; sensitivity checks also use 1 and 2.'),
        ('Limits', 'Scores are conditional on successfully graded outputs. Failures, abstentions, latency and cost are not quality scores. The automated grades and bootstrap intervals are exploratory; a failure to show an advantage does not establish equivalence. Rubrics differ by corpus, so do not rank corpora against one another.'),
    ]
    y = .81
    for title, body in paragraphs:
        fig.text(.07, y, title, fontsize=12, weight='bold', color=INK)
        wrapped = textwrap.fill(body, width=112)
        fig.text(.07, y - .03, wrapped, fontsize=10.5, va='top', linespacing=1.5)
        y -= .15
    save(fig, 'selection_methodology', pdf)


def grid(rows, corpus, kind, pdf):
    spec = SPECS[corpus]
    modes, languages = list(spec['modes']), spec['languages']
    lookup = {(r['language'], r['mode']): r for r in rows if r['corpus'] == corpus}
    route = kind == 'routes'
    fig = page(f"{spec['title']} | {'recommended generators' if route else 'Haiku versus Qwen'}",
               'Each cell is one mode and language; grades are out of 40')
    ax = fig.add_axes((.13, .23, .76, .56))
    values = np.full((len(languages), len(modes)), np.nan)
    for i, language in enumerate(languages):
        for j, mode in enumerate(modes):
            row = lookup.get((language, mode))
            if row:
                value = number(row['recommended_raw_gap'] if route else row['haiku_minus_qwen_paired'])
                if value is not None:
                    values[i, j] = value
    cmap = plt.get_cmap('YlOrBr' if route else 'coolwarm').copy()
    cmap.set_bad('#e8edf1')
    chart = ax.imshow(values, cmap=cmap, vmin=0 if route else -4, vmax=3 if route else 4,
                      aspect='auto')
    ax.set_xticks(range(len(modes)), [spec['modes'][m].replace('_', ' ') for m in modes], fontsize=10)
    # Wrap chemistry's long labels; legal labels already include line breaks.
    ax.set_xticklabels([textwrap.fill(label.replace('\n', ' '), 15)
                        for label in [spec['modes'][m] for m in modes]], fontsize=10)
    ax.set_yticks(range(len(languages)), [NAMES[l] for l in languages], fontsize=11)
    ax.tick_params(axis='both', length=0, pad=10)
    ax.set_xticks(np.arange(-.5, len(modes), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(languages), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, language in enumerate(languages):
        for j, mode in enumerate(modes):
            row = lookup.get((language, mode))
            if not row or not np.isfinite(values[i, j]):
                label = '—'
            elif route:
                label = 'Haiku' if row['recommended_model'] == PREFERRED[0] else 'Qwen'
                label += '*' if row['evidence'] == 'provisional' else ''
                label += '†' if row['changes_with_best_candidate'] == 'True' else ''
                label += f"\ngap {values[i, j]:.2f}"
            else:
                n = int(row['haiku_qwen_paired_documents'])
                label = f"{values[i, j]:+.2f}\nn={n}" + ('*' if n < 10 else '')
            dark = np.isfinite(values[i, j]) and (values[i, j] > 2.1 if route else abs(values[i, j]) > 2.9)
            ax.text(j, i, label, ha='center', va='center', fontsize=11,
                    color='white' if dark else INK, linespacing=1.6)
    colorbar = fig.colorbar(chart, ax=ax, fraction=.035, pad=.025,
                           extend='neither' if route else 'both')
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label('Leader minus choice (raw mean)' if route else 'Paired mean: Haiku minus Qwen', fontsize=9)
    if route:
        footer = ['Haiku = Claude Haiku 5.5; Qwen = Qwen3.8-27B. * Provisional; † changes when selecting the best candidate.',
                  'Gap = observed leader mean minus chosen model mean. Unequal input coverage can affect these raw gaps.',
                  '— No successfully graded data. Use the next page and paired_margins.csv to assess shared-input evidence.']
    else:
        footer = ['Positive values favor Haiku; negative values favor Qwen. n = independent documents shared by both models.',
                  '* Fewer than ten paired documents: the direction is provisional. Values alone do not establish a reliable advantage.',
                  'The full CSV includes paired 95% intervals where n ≥ 5. Multiple questions from one document are not independent.']
    for y, line in zip((.125, .092, .059), footer):
        fig.text(.065, y, line, fontsize=9, color=INK)
    save(fig, f'{corpus}_{kind}', pdf)


def exceptions_page(selected, page_index, total, pdf):
    fig = page(f'Score gaps above {PRACTICAL_MARGIN:g} points | page {page_index}',
               f'{total} cells deserve follow-up; current evidence does not meet the switching rule')
    content = []
    for row in selected:
        low, high = number(row['leader_ci95_low']), number(row['leader_ci95_high'])
        interval = f'[{low:+.2f}, {high:+.2f}]' if low is not None else 'Insufficient n'
        content.append([f"{SPECS[row['corpus']]['title']} / {NAMES[row['language']]}\n{row['mode'].replace('_', ' ')}",
                        name(row['observed_leader']), name(row['preferred_choice']),
                        f"+{float(row['raw_margin_over_best_preferred']):.2f}",
                        f"{float(row['leader_paired_margin']):+.2f}",
                        row['leader_paired_parent_documents'], interval])
    ax = fig.add_axes((.035, .29, .93, .5))
    ax.axis('off')
    table = ax.table(cellText=content, colLabels=['Cell', 'Leader', 'Preferred', 'Raw gap',
                                                'Paired gap', 'Paired n', 'Paired 95% CI'],
                     colWidths=[.29, .15, .13, .09, .09, .08, .17], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3.3)
    for y, text in zip((.19, .145, .10), [
        'Raw gap compares each model on its observed inputs; paired gap uses only inputs shared with the listed preferred model.',
        'Intervals shown here are unadjusted exploratory intervals. The replacement rule uses stricter within-cell adjusted lower bounds.',
        f'Spanish lookup is the strongest follow-up candidate; its paired interval still includes improvements below {PRACTICAL_MARGIN:g} points.',
    ]):
        fig.text(.06, y, text, fontsize=10)
    save(fig, 'follow_up_cases' if page_index == 1 else f'follow_up_cases_{page_index}', pdf)


def exceptions(rows, pdf):
    selected = sorted((r for r in rows
                       if float(r['raw_margin_over_best_preferred']) > PRACTICAL_MARGIN + 1e-9),
                      key=lambda r: float(r['raw_margin_over_best_preferred']), reverse=True)
    for start in range(0, len(selected), 6):
        exceptions_page(selected[start:start + 6], start // 6 + 1, len(selected), pdf)


def suggested_routes(rows):
    lookup = {(r['corpus'], r['language'], r['mode']): r for r in rows}
    routes = []
    for corpus, spec in SPECS.items():
        for language in spec['languages']:
            for mode in spec['modes']:
                row = lookup.get((corpus, language, mode))
                routes.append({'corpus': corpus, 'language': language, 'mode': mode,
                               'model': row['recommended_model'] if row else None,
                               'evidence': row['evidence'] if row else 'no_data',
                               'raw_gap_to_leader': number(row['recommended_raw_gap']) if row else None,
                               'haiku_qwen_paired_documents': int(row['haiku_qwen_paired_documents']) if row else 0,
                               'changes_with_best_candidate': row['changes_with_best_candidate'] == 'True' if row else None})
    (RESULTS / 'recommended_routes.json').write_text(json.dumps({
        'status': 'suggestion_only_not_applied', 'practical_margin_out_of_40': PRACTICAL_MARGIN,
        'routes': routes}, indent=2) + '\n')


def main():
    rows = list(csv.DictReader((RESULTS / 'recommendations.csv').open()))
    summary = json.loads((RESULTS / 'summary.json').read_text())
    with PdfPages(RESULTS / 'model_selection_margins.pdf') as pdf:
        overview(rows, summary, pdf)
        methodology(pdf)
        for corpus in SPECS:
            grid(rows, corpus, 'routes', pdf)
            grid(rows, corpus, 'paired_haiku_qwen', pdf)
        exceptions(rows, pdf)
    suggested_routes(rows)


if __name__ == '__main__':
    main()
