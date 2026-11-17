"""Recovery must reuse validated generation and leave original journals intact."""
import json
import sqlite3
from dataclasses import asdict
from types import SimpleNamespace

import finish

from clir_bench.domains.legal.qac.batch_recording import RunState


def test_failed_grading_reuses_valid_generation_in_a_separate_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, 'RUN', tmp_path)
    models = [f'test/model{position}' for position in range(10)]
    entries, prepared = [], []
    for position in range(50):
        target = finish.un_batch.Target(f'd{position}', f'd{position}#0', 0, 1,
                                        'resolution', 'auto', 'ar')
        entry = {'corpus': 'un', 'target': asdict(target), 'payload_sha256': 'source-hash'}
        entries.append(entry)
        prepared.append((entry, target, SimpleNamespace()))
    original = tmp_path / 'un/ar/round_01/run/run.sqlite'
    with RunState(original) as state:
        state.put('targets', entries)
        state.put('config', {'generation_model': models, 'context_chars': 30000})
        for entry, target, _ in prepared:
            route = finish.persona_routing.task_id(entry)
            with state.transaction() as connection:
                connection.execute('INSERT INTO stages VALUES (?,?,?,?,?,?)', (
                    route, 'persona_selection', 1, 'completed',
                    json.dumps({'mode': 'lookup', 'generation_mode': 'lookup'}), None))
            for model in models:
                task = finish.qac._digest(['un', asdict(target), model, []])
                state.outcome(task, [])
        target = prepared[0][1]
        failed_task = finish.qac._digest(['un', asdict(target), models[0], []])
        state.outcome(failed_task, [], 'quality failed after three attempts')
        with state.transaction() as connection:
            connection.execute('INSERT INTO stages VALUES (?,?,?,?,?,?)', (
                failed_task, 'generation', 1, 'completed', '[{"preserved":true}]', None))
    monkeypatch.setattr(finish.un_context, 'BlockIndex', lambda: SimpleNamespace())
    monkeypatch.setattr(finish.qac, '_prepare', lambda *args: (prepared, {}))
    monkeypatch.setattr(finish.decider, 'row_metadata', lambda value: {'decider_mode': value['mode']})
    generated = []

    def work(target, index, *, checkpoint, gen_model, **kwargs):
        def forbidden_regeneration():
            generated.append(True)
            raise AssertionError('Validated generation must be reused')
        assert checkpoint.run('generation', forbidden_regeneration) == [{'preserved': True}]
        checkpoint.run('quality', lambda: {'score': 4})
        return [{'corpus': 'un', 'target_id': target.block_id, 'document_id': target.doc_id,
                 'question_language': 'ar', 'mode': 'lookup', 'generator_model_id': gen_model,
                 'candidate_id': 'repaired', 'question': 'سؤال؟', 'answer': 'جواب',
                 'grading_status': 'completed', 'total_score': 32, 'faith_grounding': 4}]

    monkeypatch.setattr(finish.un_batch, 'run_one', work)
    finish.repair_un('ar', models, workers=2)
    assert not generated
    with sqlite3.connect(original) as connection:
        assert connection.execute('SELECT status FROM outcomes WHERE task=?',
                                  (failed_task,)).fetchone()[0] == 'failed'
    folder = original.parent.parent
    assert json.loads((folder / 'remaining_failures.json').read_text()) == []
    rows = finish.corpus.read_rows(folder / 'results.csv')
    assert len(rows) == 1 and rows[0]['candidate_id'] == 'repaired'
    assert rows[0]['source_payload_sha256'] == 'source-hash'
    assert rows[0]['source_generation_run'] == str(original)
