"""Refresh question/grade plots now, optionally refreshing once recovery finishes."""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from fractions import Fraction

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_chemistry_top_models import MODELS
from plot_data import LANGUAGE_NAMES, NEW, OUT, refresh_snapshot, snapshot_note, snapshot_rows


def plot_all_grades():
    rows = snapshot_rows()
    groups = defaultdict(list)
    documents = defaultdict(set)
    for row in rows:
        source, language, model = row['corpus'], row['question_language'], row['generator_model_id'].replace('/', '_')
        document = (row['document_id'] if source in ('eurlex', 'un') else row['source_document_id'])
        documents[source, language].add(document or row['target_id'])
        if row['grading_status'] == 'failed' or not row['total_score'].strip():
            continue
        value = Fraction(row['total_score'])
        if not 0 <= value <= 40:
            raise ValueError('Grade outside [0,40]')
        groups[source, language, model].append(value)
    with (OUT / 'generated_documents_by_corpus_and_language.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(['corpus', 'language', 'distinct_documents_with_questions'])
        writer.writerows([source, language, len(values)] for (source, language), values in sorted(documents.items()))
    with (OUT / 'generated_question_grades_by_model_and_language.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(['corpus', 'language', 'model', 'graded_candidates', 'mean_total_score'])
        writer.writerows([source, language, model, len(values), float(sum(values) / len(values))]
                         for (source, language, model), values in sorted(groups.items()))
    fig, axes = plt.subplots(1, 3, figsize=(22, 9), facecolor='#fbfcfe',
                             gridspec_kw={'width_ratios': [4, 6, 7]})
    definitions = [('eurlex', 'EUR-Lex', ['en', 'fr', 'es', 'de']),
                   ('un', 'UN', ['en', 'fr', 'es', 'zh', 'ar', 'ru']),
                   ('chemistry', 'BASF Google Patents', ['en', 'fr', 'es', 'de', 'zh', 'ja', 'ru'])]
    for ax, (source, title, languages) in zip(axes, definitions):
        values = np.full((len(MODELS), len(languages)), np.nan)
        for i, model in enumerate(MODELS):
            for j, language in enumerate(languages):
                scores = groups[source, language, model]
                if scores:
                    values[i, j] = float(sum(scores) / len(scores))
        im = ax.imshow(np.ma.masked_invalid(values), cmap='Blues', vmin=0, vmax=40, aspect='auto')
        ax.set_title(title, fontsize=15, weight='bold', pad=20)
        ax.set_xticks(range(len(languages)), [LANGUAGE_NAMES[lg] for lg in languages], rotation=45, ha='right')
        ax.set_yticks(range(len(MODELS)), [MODELS[model] for model in MODELS])
        ax.tick_params(length=0, labelsize=9)
        for i, model in enumerate(MODELS):
            for j, language in enumerate(languages):
                scores = groups[source, language, model]
                text = f'{values[i, j]:.1f}\nn={len(scores)}' if scores else '—'
                ax.text(j, i, text, ha='center', va='center', fontsize=8.3,
                        color='white' if scores and values[i, j] > 25 else '#14283b')
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle('Generated-question grades by model and language', fontsize=22, weight='bold', y=.97)
    fig.text(.02, .055, 'Mean total_score / 40; n = graded candidate outputs. Descriptive averages; samples and prompt versions differ across runs.', fontsize=10)
    fig.text(.02, .025, snapshot_note(), fontsize=9)
    fig.subplots_adjust(top=.87, bottom=.17, left=.07, right=.97, wspace=.75)
    fig.colorbar(im, ax=list(axes), shrink=.65, pad=.02, label='Mean total score / 40')
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'generated_question_grades_by_model_and_language.{suffix}', dpi=200)
    plt.close(fig)


def refresh():
    info = refresh_snapshot()
    for script in ('plot_generated_questions.py', 'plot_chemistry_questions_by_mode_and_corpus.py',
                   'plot_chemistry_top_models.py', 'plot_legal_questions.py'):
        subprocess.run([sys.executable, str(OUT / script)], check=True)
    plot_all_grades()
    if merge := shutil.which('pdfunite'):
        inputs = [OUT / filename for filename in (
            'legal_generated_questions.pdf', 'chemistry_generated_questions.pdf',
            'generated_questions_by_mode_and_model.pdf',
            'chemistry_top_generator_by_language_and_mode.pdf',
            'generated_question_grades_by_model_and_language.pdf')]
        temporary = OUT / 'generated_questions_and_grades.tmp.pdf'
        subprocess.run([merge, *map(str, inputs), str(temporary)], check=True)
        temporary.replace(OUT / 'generated_questions_and_grades.pdf')
    print('Snapshot counts:', info['counts'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-completion', action='store_true',
                        help='After refreshing now, refresh once more when generation summary.json appears')
    args = parser.parse_args()
    refresh()
    if args.wait_for_completion and not (NEW / 'summary.json').exists():
        print('Waiting to refresh the plots once generation recovery completes.', flush=True)
        while not (NEW / 'summary.json').exists():
            time.sleep(30)
        refresh()


if __name__ == '__main__':
    main()
