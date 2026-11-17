"""Explicitly refresh the packaged generation table from the reviewed analysis."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / 'model_selection' / 'recommendations.csv'
DESTINATION = ROOT / 'src/clir_bench/core/generation_model_table.json'


def number(value):
    return float(value) if value else None


def main():
    rows = list(csv.DictReader(SOURCE.open()))
    if any(float(row['practical_margin']) != .5 for row in rows):
        raise ValueError('Generate recommendations at a 0.5-point margin before exporting')
    summary = json.loads((SOURCE.parent / 'summary.json').read_text())
    table = {
        'version': '2026-11-17',
        'source': str(SOURCE.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'input_sha256': summary['input_sha256'],
        'score_scale': 40,
        'rows': [
            {'corpus': 'gp' if row['corpus'] == 'chemistry' else row['corpus'],
             'language': row['language'], 'mode': row['mode'],
             'recommended_model': row['recommended_model'],
             'haiku_mean': number(row['haiku_mean']), 'qwen_mean': number(row['qwen_mean']),
             'haiku_minus_qwen_paired': number(row['haiku_minus_qwen_paired']),
             'paired_documents': int(row['haiku_qwen_paired_documents'])}
            for row in rows
        ],
    }
    DESTINATION.write_text(json.dumps(table, indent=2) + '\n')
    print(f'Exported {len(rows)} scored cells to {DESTINATION.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
