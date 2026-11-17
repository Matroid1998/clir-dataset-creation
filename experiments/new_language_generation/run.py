"""Extend the ten-model comparison to new languages, with resumable local outputs.

Uses the existing generation, eligibility, evidence and grading implementations.
Selects 50 source documents per new language and applies all ten generators.
No existing generation runs or published datasets are overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from clir_bench import domains
from clir_bench.cli import build_parser
from clir_bench.core import corpus, llm, qagen
from clir_bench.core.config import load_settings
from clir_bench.core.context import AppContext
from clir_bench.core.grading import GraderConfig
from clir_bench.core.prompts import PromptPack
from clir_bench.domains.chemistry.qac import decider as chemistry_decider
from clir_bench.domains.legal import qac as legal_qac
from clir_bench.domains.legal.qac import un_batch, un_context

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'data/new_language_generation_20261117'
VERIFIER = 'google/gemini-3.8-flash'
LOCAL = threading.local()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def label(model):
    return model.replace('/', '_')


class RecordedClient:
    def __init__(self, client):
        self.client = client
        self.chat = SimpleNamespace(completions=self)

    def __getattr__(self, name):
        return getattr(self.client, name)

    def with_options(self, **kwargs):
        return RecordedClient(self.client.with_options(**kwargs))

    def create(self, **kwargs):
        response = self.client.chat.completions.create(**kwargs)
        save(RUN / 'calls' / getattr(LOCAL, 'task', 'un') / (uuid.uuid4().hex + '.json'),
             {'request': kwargs, 'response': response.model_dump()})
        return response


def read_rows(path):
    return corpus.read_rows(path) if path.exists() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--corpora', nargs='+', choices=['un', 'chemistry'],
                        default=['chemistry', 'un'])
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    export = json.loads((ROOT / 'docs/generation_export.json').read_text())
    models = [item['model'] for item in export['corpora']['chemistry']['generation_csv']]
    expected = {'un': {'ar': 50, 'ru': 50}, 'chemistry': {'ja': 50, 'ru': 50}}
    configuration = {'models': models, 'verifier': VERIFIER, 'targets': expected,
                     'target_unit': 'source_documents_per_language', 'seed': 20261117,
                     'generation_reasoning_effort': 'medium', 'grading_reasoning_effort': 'low'}
    if args.dry_run:
        print(json.dumps(configuration, indent=2))
        return
    RUN.mkdir(parents=True, exist_ok=True)
    manifest = RUN / 'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == configuration, 'Run configuration changed'
    else:
        save(manifest, configuration)
    legal_prompts = RUN / 'legal_prompt_manifest.json'
    if not legal_prompts.exists():
        save(legal_prompts, json.loads((ROOT / '.clir/active_prompt_manifest.json').read_text()))
    os.environ.pop('CLIR_PROMPT_SOURCE', None)
    os.environ['CLIR_PROMPT_MANIFEST'] = str(legal_prompts)
    original_client = llm.client_for
    clients = {model: RecordedClient(original_client(model).with_options(
        timeout=180, max_retries=0)) for model in [*models, VERIFIER]}
    llm.client_for = lambda model: clients[model]
    settings = load_settings(ROOT / 'clir.toml')
    contexts = {name: AppContext.build(settings, domains.load(name))
                for name in ('chemistry', 'legal')}
    chemistry = contexts['chemistry']
    grouped = corpus.load_grouped(chemistry.workspace.corpus_csv('gp'), chemistry.schema)
    chemistry.domain.qac_validate_corpus(grouped, chemistry.schema)
    prompts = PromptPack(chemistry.domain.prompts_package)
    fields = (*qagen.output_fieldnames(chemistry.schema, prompts=prompts,
                                      modes=chemistry.domain.analysis.modes),
              'generator_model_id', 'generator_model_name')
    prompt_texts = {f'{mode}/{lang}': prompts.generation(mode, lang)
                    for mode in chemistry.domain.analysis.modes for lang in ('ja', 'ru')}
    prompt_texts.update({f'verifier/{mode}': prompts.quality(mode)
                        for mode in chemistry.domain.analysis.modes})
    prompt_texts['faithfulness'] = prompts.faithfulness()
    prompt_texts['eligibility'] = chemistry_decider.eligibility_prompt_text()
    snapshot = RUN / 'chemistry_prompts.json'
    if snapshot.exists():
        assert json.loads(snapshot.read_text()) == prompt_texts, 'Chemistry prompts changed'
    else:
        save(snapshot, prompt_texts)
    index = None

    def chemistry_batch(language, round_id, size, used):
        folder = RUN / 'chemistry' / language / f'round_{round_id:02d}'
        plan_path = folder / 'plan.json'
        if plan_path.exists():
            plan = [qagen.PlanItem(**item) for item in json.loads(plan_path.read_text())]
        else:
            candidates = [row for rows in grouped.values() for row in rows
                          if row['language'] == language and row['id'] not in used]
            candidates.sort(key=lambda row: row['id'])
            random.Random(20261117 + round_id).shuffle(candidates)
            plan = [qagen.PlanItem(row['publication_number'], language, 'lookup',
                                  document_id=row['id'], strategy=3)
                    for row in candidates[:size]]
            assert len(plan) == size, 'Not enough fresh source documents'
            options = SimpleNamespace(decider_model='jev', modes=None, append=False,
                                      workers=args.workers, seed=20261117 + round_id)
            # Retain the sampled identities and screen replacements if no persona is eligible.
            sampled = [asdict(item) for item in plan]
            plan = chemistry_decider.route_plan(
                context=chemistry, source=chemistry.domain.source('gp'), args=options,
                grouped=grouped, plan=plan, output=folder / 'results.csv')
            cursor = size
            while len(plan) < size:
                needed = size - len(plan)
                extra = [qagen.PlanItem(row['publication_number'], language, 'lookup',
                                       document_id=row['id'], strategy=3)
                         for row in candidates[cursor:cursor + needed]]
                assert len(extra) == needed, 'Not enough eligible source documents'
                cursor += needed
                sampled.extend(asdict(item) for item in extra)
                options.append = True
                plan.extend(chemistry_decider.route_plan(
                    context=chemistry, source=chemistry.domain.source('gp'), args=options,
                    grouped=grouped, plan=extra, output=folder / 'results.csv'))
            save(folder / 'sampled.json', sampled)
            save(plan_path, [asdict(item) for item in plan])
        tasks = [(model, item) for item in plan for model in models]

        def work(task):
            model, item = task
            key = hashlib.sha256(json.dumps([model, asdict(item)], sort_keys=True).encode()).hexdigest()
            checkpoint = folder / 'targets' / (key + '.json')
            if checkpoint.exists() and json.loads(checkpoint.read_text())['status'] != 'failed':
                return
            LOCAL.task = 'chemistry/' + language + '/' + label(model)
            config = qagen.GenerationConfig(
                generation_model=model, generation_reasoning_effort='medium',
                grader=GraderConfig(model=VERIFIER, reasoning_effort='low',
                                   thinking_budget_tokens=8000, thinking_max_tokens=12000),
                retries=3, candidate_parser=chemistry.domain.qac_candidate_parser)
            result = {'model': model, 'item': asdict(item), 'rows': []}
            try:
                rows = qagen.generate_for_document(
                    item, grouped[item.family], schema=chemistry.schema, prompts=prompts,
                    config=config, generation_client=clients[model], grading_client=clients[VERIFIER],
                    language_order=chemistry.languages.priority)
                for row in rows:
                    row.update(generator_model_id=model, generator_model_name=model.split('/')[-1])
                result.update(status='completed' if rows else 'abstained', rows=rows)
            except Exception as error:  # noqa: BLE001 -- checkpoint all API failures
                chain = []
                while error:
                    chain.append(f'{type(error).__name__}: {error}')
                    error = error.__cause__
                result.update(status='failed', error=' <- '.join(chain))
            save(checkpoint, result)
            print('chemistry', language, model, result['status'], len(result['rows']), flush=True)

        # A later invocation retries failed targets and never repeats completed ones.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(work, task) for task in tasks]):
                future.result()
        rows, failures = [], []
        for checkpoint in sorted((folder / 'targets').glob('*.json')):
            result = json.loads(checkpoint.read_text())
            rows.extend(result['rows'])
            if result['status'] == 'failed':
                failures.append(result)
        corpus.write_rows(folder / 'results.csv',
                          [qagen.normalize_row(row, fields) for row in rows], fields)
        save(folder / 'failures.json', failures)
        if failures:
            raise RuntimeError(f'{len(failures)} failed chemistry tasks; rerun to resume: {folder}')
        return {item.document_id for item in plan}

    def run_phase(name, language, target):
        used = set()
        for round_id in (1,):
            folder = RUN / name / language / f'round_{round_id:02d}'
            if folder.exists() and (folder / 'done.json').exists():
                continue
            size = target
            print(f'{name}/{language}: {size} source documents × {len(models)} models', flush=True)
            if name == 'chemistry':
                used.update(chemistry_batch(language, round_id, size, used))
            else:
                plan_path = folder / 'plan.json'
                if plan_path.exists():
                    selections = json.loads(plan_path.read_text())
                else:
                    shared = RUN / 'un' / 'shared_targets.json'
                    if not shared.exists():
                        save(shared, [asdict(item) for item in un_batch.select(
                            index, n=size, seed=20261117, languages=['ar'],
                            modes=['lookup'], max_per_doc=1)])
                    selections = [{'corpus': 'un', 'target': dict(item, language=language)}
                                  for item in json.loads(shared.read_text())]
                    assert len({item['target']['doc_id'] for item in selections}) == size
                    save(plan_path, selections)
                run_dir = folder / 'run'
                options = ['qac', 'generate', '--source', 'un', '--langs', language,
                           '--questions', str(len(selections)), '--run-dir', str(run_dir),
                           '--decider-model', 'jev', '--verifier-model', VERIFIER,
                           '--workers', str(args.workers), '--keep', '3']
                for model in models:
                    options += ['--generation-model', model]
                if (run_dir / 'run.sqlite').exists():
                    options += ['--resume']
                parsed = build_parser(contexts['legal'], domains.load_module('legal')).parse_args(options)
                code = legal_qac.generate(parsed, contexts['legal'], selections=selections,
                                          supplied_indexes={'un': index})
                if code:
                    raise RuntimeError(f'UN run has failures; rerun to resume: {run_dir}')
                original = run_dir / 'results.csv'
                (folder / 'results.csv').write_bytes(original.read_bytes())
            save(folder / 'done.json', {'completed': True})

    if "un" in args.corpora:
        index = un_context.BlockIndex()
    phases = [(name, language, target) for name in args.corpora
              for language, target in expected[name].items()]
    with ThreadPoolExecutor(max_workers=len(phases)) as pool:
        for future in as_completed([pool.submit(run_phase, *phase) for phase in phases]):
            future.result()
    summary = {}
    for name, languages in expected.items():
        summary[name] = {}
        for language, target in languages.items():
            rows = [row for path in sorted((RUN / name / language).glob('round_*/results.csv'))
                    for row in read_rows(path)]
            counts = Counter(row.get('generator_model_id') or row.get('generator_model_name') for row in rows)
            if rows:
                combined = RUN / name / f'{language}_results.csv'
                export_fields = tuple(dict.fromkeys(key for row in rows for key in row))
                corpus.write_rows(combined, rows, export_fields)
            documents = {row.get('source_document_id') or row.get('document_id') for row in rows}
            summary[name][language] = {'selected_documents': target,
                                      'documents_with_questions': len(documents),
                                      'generated_questions': len(rows),
                                      'models': dict(counts)}
            if name in args.corpora:
                assert len(counts) == 10, (name, language, counts)
    save(RUN / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
