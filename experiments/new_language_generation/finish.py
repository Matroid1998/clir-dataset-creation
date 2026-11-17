"""Finish checkpointed new-language runs, retry failed UN stages, and export results.

Retries use separate SQLite journals. Valid original stages are reused; original
run databases and raw responses remain intact. The optional wait runs locally.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from dotenv import load_dotenv
from run import ROOT, RUN, VERIFIER, RecordedClient, save

from clir_bench.core import corpus, llm
from clir_bench.domains.legal import qac
from clir_bench.domains.legal.qac import decider, persona_routing, un_batch, un_context
from clir_bench.domains.legal.qac.batch_recording import RunState

LOCAL = threading.local()


def running():
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            arguments = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'experiments/new_language_generation/run.py' in arguments:
            return True
    return False


def repair_un(language, models, workers):
    folder = RUN / 'un' / language / 'round_01'
    original = folder / 'run/run.sqlite'
    with sqlite3.connect(original.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        metadata = {k: json.loads(v) for k, v in connection.execute('SELECT key,value FROM metadata')}
        outcomes = {task: {'task': task, 'status': status, 'rows': json.loads(rows), 'error': error}
                    for task, status, rows, error in connection.execute('SELECT * FROM outcomes')}
        completed_stages = {(task, stage): (task, stage, attempts, status, value, error)
                            for task, stage, attempts, status, value, error in connection.execute(
                                "SELECT * FROM stages WHERE status='completed'")}
    index = un_context.BlockIndex()
    prepared, _ = qac._prepare(metadata['targets'], {'un': index}, metadata['config'], 'generate')
    cases = {}
    for entry, target, payload in prepared:
        decision = json.loads(completed_stages[(persona_routing.task_id(entry), 'persona_selection')][4])
        for model in models:
            task = qac._digest(['un', asdict(target), model, []])
            cases[task] = (entry, replace(target, mode=decision['generation_mode']), payload, model, decision)
    assert set(outcomes) == set(cases) and len(cases) == 500
    for attempt in range(1, 4):
        failed = [task for task, result in outcomes.items() if result['status'] == 'failed']
        if not failed:
            break
        journal = folder / f'repair_{attempt:02d}.sqlite'
        with RunState(journal) as state:
            state.put('source_run', str(original))
            state.put('config', metadata['config'])
            state.put('repair_attempt', attempt)
            with state.transaction() as connection:
                for (task, _), row in completed_stages.items():
                    if task in failed:
                        connection.execute('INSERT OR IGNORE INTO stages VALUES (?,?,?,?,?,?)', row)

            def work(task, journal=journal):
                entry, target, payload, model, decision = cases[task]
                LOCAL.feedback = outcomes[task].get('error') or ''
                # A valid original generation is reused for verifier-only repairs.
                if (task, 'generation') in completed_stages:
                    LOCAL.feedback = ''
                try:
                    rows = un_batch.run_one(
                        target, index, gen_model=model, grade_model=VERIFIER,
                        context_chars=metadata['config']['context_chars'], keep=3,
                        checkpoint=state.checkpoint(task, retries=3), payload=payload,
                        deduplicate_exact_questions=True)
                    for row in rows:
                        row.update(decider.row_metadata(decision), repair_run=str(journal),
                                   source_generation_run=str(original))
                        row['source_payload_sha256'] = entry['payload_sha256']
                    errors = '; '.join(row.get('grading_error', '') for row in rows
                                       if row.get('grading_status') == 'failed')
                    state.outcome(task, rows, errors or None)
                except Exception as error:  # noqa: BLE001 -- retain failures for the next journal
                    state.outcome(task, [], f'{type(error).__name__}: {error}')
                finally:
                    LOCAL.feedback = ''

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for future in as_completed([pool.submit(work, task) for task in failed]):
                    future.result()
            outcomes.update({result['task']: result for result in state.outcomes()})
            with state.transaction() as connection:
                for row in connection.execute("SELECT * FROM stages WHERE status='completed'"):
                    completed_stages[(row[0], row[1])] = tuple(row)
        print('UN repair', language, attempt, Counter(v['status'] for v in outcomes.values()), flush=True)
    rows = [row for value in outcomes.values() for row in value['rows']]
    qac._rank(rows)
    qac._write_csv(folder / 'results.csv', rows)
    failures = [value for value in outcomes.values() if value['status'] == 'failed']
    save(folder / 'remaining_failures.json', failures)
    save(folder / 'done.json', {'completed': not failures, 'cases': len(outcomes), 'failures': len(failures)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-running', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.wait_running:
        while running():
            time.sleep(20)
    load_dotenv(ROOT / '.env')
    manifest = json.loads((RUN / 'manifest.json').read_text())
    os.environ.pop('CLIR_PROMPT_SOURCE', None)
    os.environ['CLIR_PROMPT_MANIFEST'] = str(RUN / 'legal_prompt_manifest.json')
    original_client = llm.client_for
    clients = {model: RecordedClient(original_client(model).with_options(timeout=180, max_retries=0))
               for model in [*manifest['models'], VERIFIER]}
    llm.client_for = lambda model: clients[model]
    original_messages = un_batch.gen.build_messages

    def messages(payload, mode, language):
        result = original_messages(payload, mode, language)
        feedback = getattr(LOCAL, 'feedback', '')
        if feedback:
            result[-1] = dict(result[-1], content=result[-1]['content'] + '\n\n'
                'OUTPUT REPAIR: The preceding attempt failed validation: ' + feedback[:500]
                + '\nReturn a JSON array of one to three candidates, or one skip_reason object. '
                'Recopy each evidence span character for character from the supplied TARGET BLOCK '
                'in its original language. Prefer a shorter sufficient exact excerpt; never '
                'normalize its whitespace, translate evidence or invent missing support.')
        return result

    un_batch.gen.build_messages = messages
    # Chemistry checkpoints already preserve completed/abstained targets.
    for _ in range(3):
        pending = []
        for language in ('ja', 'ru'):
            for path in (RUN / 'chemistry' / language / 'round_01/targets').glob('*.json'):
                if json.loads(path.read_text())['status'] == 'failed':
                    pending.append(path)
        if not pending:
            break
        subprocess.run([sys.executable, str(ROOT / 'experiments/new_language_generation/run.py'),
                        '--corpora', 'chemistry', '--workers', str(args.workers)], check=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(repair_un, language, manifest['models'], args.workers)
                                    for language in ('ar', 'ru')]):
            future.result()
    summary = {}
    for source, languages in manifest['targets'].items():
        summary[source] = {}
        for language in languages:
            folder = RUN / source / language / 'round_01'
            rows = corpus.read_rows(folder / 'results.csv')
            fields = tuple(dict.fromkeys(key for row in rows for key in row))
            corpus.write_rows(RUN / source / f'{language}_results.csv', rows, fields)
            for model in manifest['models']:
                subset = [row for row in rows if row['generator_model_id'] == model]
                corpus.write_rows(RUN / source / f'{language}_{model.replace("/", "_")}.csv', subset, fields)
            documents = {row.get('source_document_id') or row.get('document_id') for row in rows}
            summary[source][language] = {
                'selected_documents': 50, 'documents_with_questions': len(documents),
                'generated_questions': len(rows),
                'models': dict(Counter(row['generator_model_id'] for row in rows)),
                'remaining_failures': json.loads((folder / 'remaining_failures.json').read_text())
                if source == 'un' else json.loads((folder / 'failures.json').read_text()),
            }
    save(RUN / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
