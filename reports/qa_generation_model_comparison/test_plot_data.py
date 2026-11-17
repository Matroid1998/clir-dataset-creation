"""A repaired task must appear once, while separate model outputs remain distinct."""
import json
import sqlite3

from plot_data import deduplicate, legal_outcomes, normalize


def journal(path, values):
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE outcomes (task TEXT,status TEXT,rows TEXT)')
        connection.executemany('INSERT INTO outcomes VALUES (?,?,?)',
                               [(task, status, json.dumps(rows)) for task, status, rows in values])


def row(model='a', **values):
    return {'question': 'Which document?', 'question_language': 'ru', 'mode': 'lookup',
            'generator_model_id': model, 'source_document_id': 'doc', 'candidate_id': 'candidate',
            'candidate_rank': 1, **values}


def test_repair_replaces_failed_task_without_duplicating_original_candidates(tmp_path):
    base, repair, retry = (tmp_path / f'{name}.sqlite' for name in ('base', 'repair', 'retry'))
    journal(base, [('one', 'completed', [row()]), ('two', 'failed', [])])
    journal(repair, [('two', 'completed', [row(candidate_id='repaired')])])
    journal(retry, [('two', 'failed', [])])
    values = legal_outcomes([base, repair, retry])
    assert {value['candidate_id'] for value, _ in values} == {'candidate', 'repaired'}
    assert len(values) == 2


def test_candidate_deduplication_preserves_models_and_ranks():
    one = normalize(row(), 'un', 'source')
    other_model = normalize(row(model='b'), 'un', 'source')
    assert len(deduplicate([one, dict(one), other_model])) == 2
    without_id = normalize(row(candidate_id=''), 'chemistry', 'source')
    second = dict(without_id, candidate_rank=2)
    assert len(deduplicate([without_id, dict(without_id), second])) == 2


def test_empty_question_is_excluded_and_declared_model_is_preserved():
    assert normalize(row(question='  '), 'un', 'source') is None
    assert normalize(row(), 'chemistry', 'source', model='full/model')['generator_model_id'] == 'full/model'
