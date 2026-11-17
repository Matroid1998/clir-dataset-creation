"""One frozen, deduplicated plotting snapshot of original and supplemental runs."""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NEW = ROOT / 'data/new_language_generation_20261117'
SNAPSHOT = OUT / 'generated_question_plot_inputs.csv'
MANIFEST = OUT / 'generated_question_plot_sources.json'
FIELDS = ('corpus', 'generator_model_id', 'question_language', 'mode', 'question',
          'total_score', 'faith_overall', 'qual_overall', 'source_document_id',
          'document_id', 'target_id', 'candidate_id', 'candidate_rank', 'grading_status',
          'faithfulness_rubric_version', 'quality_rubric_version', 'plot_source')
LANGUAGE_NAMES = {'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German',
                  'zh': 'Chinese', 'ja': 'Japanese', 'ar': 'Arabic', 'ru': 'Russian'}


def legal_outcomes(paths, *, include_failed=False):
    """Overlay task outcomes, keeping a validated result over a failed retry."""
    outcomes = {}
    for path in paths:
        if not path.exists():
            continue
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as connection:
            connection.execute('BEGIN')
            records = connection.execute('SELECT task,status,rows FROM outcomes').fetchall()
        for task, status, serialized in records:
            if status == 'failed' and task in outcomes and outcomes[task][0] != 'failed':
                continue
            outcomes[task] = (status, json.loads(serialized), str(path.relative_to(ROOT))
                              if path.is_relative_to(ROOT) else str(path))
    return [(row, source) for status, rows, source in outcomes.values()
            if include_failed or status in ('completed', 'no_candidates') for row in rows]


def normalize(row, corpus, source, model=None):
    if not row.get('question', '').strip():
        return None
    result = {key: row.get(key, '') for key in FIELDS}
    result.update(corpus=corpus, plot_source=source,
                  generator_model_id=model or row.get('generator_model_id') or row.get('generator_model_name'))
    if not result['generator_model_id']:
        raise ValueError(f'Missing generator identity: {source}')
    return result


def deduplicate(rows):
    """Preserve separate candidates/models; eliminate repeated artifact copies."""
    selected = {}
    for row in rows:
        identity = (row['corpus'], row['generator_model_id'], row['question_language'],
                    row['candidate_id'] or (row['source_document_id'] or row['target_id']
                    or row['document_id'], row['mode'], str(row['candidate_rank']), row['question']))
        selected[identity] = row
    return list(selected.values())


def refresh_snapshot():
    csv.field_size_limit(100_000_000)
    export = json.loads((ROOT / 'docs/generation_export.json').read_text())
    rows, inputs, pending = [], [], {}
    for corpus in ('eurlex', 'un'):
        for filename in export['corpora'][corpus]['generation_sqlite']:
            path = ROOT / filename
            inputs.append(filename)
            rows.extend(normalize(row, corpus, source) for row, source in
                        legal_outcomes([path], include_failed=True))
    for entry in export['corpora']['chemistry']['generation_csv']:
        filename = entry['path']
        inputs.append(filename)
        with (ROOT / filename).open(newline='', encoding='utf-8-sig') as handle:
            rows.extend(normalize(row, 'chemistry', filename, entry['model'])
                        for row in csv.DictReader(handle))
    for language in ('ja', 'ru'):
        folder = NEW / 'chemistry' / language / 'round_01'
        statuses = Counter()
        for path in sorted((folder / 'targets').glob('*.json')):
            result = json.loads(path.read_text())
            statuses[result['status']] += 1
            if result['status'] != 'completed':
                continue
            source = str(path.relative_to(ROOT))
            inputs.append(source)
            rows.extend(normalize(row, 'chemistry', source, result['model']) for row in result['rows'])
        pending[f'chemistry/{language}'] = dict(statuses)
    for language in ('ar', 'ru'):
        folder = NEW / 'un' / language / 'round_01'
        paths = [folder / 'run/run.sqlite', *sorted(folder.glob('repair_*.sqlite'))]
        inputs.extend(str(path.relative_to(ROOT)) for path in paths if path.exists())
        rows.extend(normalize(row, 'un', source) for row, source in legal_outcomes(paths))
    rows = deduplicate([row for row in rows if row is not None])
    rows.sort(key=lambda row: (row['corpus'], row['question_language'], row['generator_model_id'],
                               row['mode'], row['question']))
    temporary = SNAPSHOT.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(SNAPSHOT)
    info = {
        'snapshot_time': datetime.now(ZoneInfo('America/Toronto')).isoformat(timespec='seconds'),
        'supplemental_recovery_complete': (NEW / 'summary.json').exists(),
        'counting_unit': 'Generated candidate rows; separate models/candidates count separately.',
        'grade_policy': 'Complete numeric total scores only; failed/ungraded totals are excluded.',
        'supplemental_policy': 'Latest successful task outcomes; failed retries and artifact copies are excluded.',
        'original_policy': 'Original question candidates retained, including candidates with failed grading.',
        'counts': {corpus: dict(Counter(row['question_language'] for row in rows if row['corpus'] == corpus))
                   for corpus in ('eurlex', 'un', 'chemistry')},
        'supplemental_chemistry_cases': pending,
        'sources': sorted(set(inputs)),
    }
    if (NEW / 'summary.json').exists():
        final = json.loads((NEW / 'summary.json').read_text())
        info['remaining_supplemental_failures'] = {
            f'{corpus}/{language}': len(value['remaining_failures'])
            for corpus, languages in final.items() for language, value in languages.items()}
    MANIFEST.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    snapshot_rows.cache_clear()
    return info


@cache
def snapshot_rows():
    csv.field_size_limit(100_000_000)
    if not SNAPSHOT.exists():
        refresh_snapshot()
    with SNAPSHOT.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def chemistry_by_model():
    groups = {}
    for row in snapshot_rows():
        if row['corpus'] == 'chemistry':
            groups.setdefault(row['generator_model_id'].replace('/', '_'), []).append(row)
    return groups


def snapshot_note():
    info = json.loads(MANIFEST.read_text())
    status = ('Recovery pass finished; failed cases excluded' if info['supplemental_recovery_complete']
              else 'UN recovery snapshot; failed cases excluded')
    return f"{status} · {info['snapshot_time'][:16].replace('T', ' ')} Toronto"
