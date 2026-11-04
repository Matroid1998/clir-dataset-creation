"""Resume the saved 200-document chemistry comparison with per-target checkpoints.

Uses the existing pipeline, prompts, strict parsers and verifier. Missing-model
routing reuses Nemotron's saved Jev decisions; Mistral retains its own decisions.
Run with --smoke for one pending target per model, then without it to finish.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace
import uuid

from dotenv import load_dotenv
from clir_bench import domains
from clir_bench.core import corpus, llm, qagen
from clir_bench.core.config import load_settings
from clir_bench.core.context import AppContext
from clir_bench.core.grading import GraderConfig
from clir_bench.core.prompts import PromptPack
from clir_bench.domains.chemistry.qac.decider import row_metadata

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'data/chemistry/ai4chem_abstract_pairs/qac/chemistry_google_patents_model_comparison_generator'
STATE = OUTPUT / 'completion_20261103'
MODELS = ['mistralai/mistral-small-2603', 'nvidia/nemotron-3.5-lightning',
          'minimax/minimax-m2.7', 'upstage/solar-pro4']
VERIFIER = 'google/gemini-3.8-flash'
LOCAL = threading.local()


def label(model):
    return model.replace('/', '_')


def stem(model):
    return f'generator_compare_200_{label(model)}_gemini38'


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


class RecordedClient:
    def __init__(self, client):
        self.client = client.with_options(timeout=180, max_retries=0)
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        response = self.client.chat.completions.create(**kwargs)
        save_json(STATE / 'calls' / LOCAL.model / LOCAL.document / f'{uuid.uuid4().hex}.json',
                  {'request': kwargs, 'response': response.model_dump(),
                   'at': datetime.now(timezone.utc).isoformat()})
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    ctx = AppContext.build(load_settings(ROOT / 'clir.toml'), domains.load('chemistry'))
    grouped = corpus.load_grouped(ctx.workspace.corpus_csv(ctx.domain.source('gp')), ctx.schema)
    prompts = PromptPack(ctx.domain.prompts_package)
    fields = qagen.output_fieldnames(ctx.schema, prompts=prompts, modes=ctx.domain.analysis.modes)
    order = ctx.languages.priority or ctx.languages.working
    prompt_texts = {f'generation/{mode}/{lang}': prompts.generation(mode, lang)
                    for mode in ctx.domain.analysis.modes for lang in ctx.languages.working}
    prompt_texts.update({f'verifiers/{mode}': prompts.quality(mode) for mode in ctx.domain.analysis.modes})
    prompt_texts['verifiers/faithfulness'] = prompts.faithfulness()
    hashes = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in prompt_texts.items()}
    manifest_path = STATE / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())['prompt_hashes'] == hashes, 'Prompts changed since checkpoint'
    elif not args.dry_run:
        save_json(manifest_path, {'models': MODELS, 'verifier': VERIFIER,
                  'generation_reasoning_effort': ctx.settings.llm.generation_reasoning_effort,
                  'grading_reasoning_effort': ctx.settings.llm.grading_reasoning_effort,
                  'prompt_hashes': hashes, 'prompt_texts': prompt_texts,
                  'new_model_routing_from': stem(MODELS[1]) + '_eligibility.json'})

    task_groups, existing, plans = [], {}, {}
    for model in args.models:
        path = OUTPUT / (stem(model) + '.csv')
        original = STATE / 'originals' / path.name
        if not args.dry_run and not original.exists():
            original.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ['.csv', '_best.csv', '_failures.csv', '_eligibility.json']:
                source = OUTPUT / (stem(model) + suffix)
                if source.exists():
                    shutil.copy2(source, original.parent / source.name)
            if not path.exists():
                corpus.write_rows(original, [], fields)
        existing[model] = corpus.read_rows(original if original.exists() else path) if path.exists() or original.exists() else []
        done = {(row['source_document_id'], row['mode']) for row in existing[model]}
        audit_path = OUTPUT / (stem(model) + '_eligibility.json')
        routing_path = audit_path if audit_path.exists() else OUTPUT / (stem(MODELS[1]) + '_eligibility.json')
        audit = json.loads(routing_path.read_text())
        if not audit_path.exists() and not args.dry_run:
            audit['reused_from'] = str(routing_path.relative_to(ROOT))
            save_json(audit_path, audit)
        run = audit['runs'][0]
        plans[model] = run['assignments']
        pending = []
        for assignment in run['assignments']:
            if assignment['mode'] == 'skip':
                continue
            item = qagen.PlanItem(family=assignment['family'], language=assignment['language'],
                                 mode=assignment['mode'], document_id=assignment['document_id'],
                                 strategy=qagen.STRATEGY_RANDOM_EXISTING,
                                 metadata=row_metadata(assignment, run['prompt_sha256']))
            selected = qagen.select_plan_rows(item, grouped[item.family], ctx.schema)
            text = corpus.build_passages_text(selected, ctx.schema, order=order, include_identifiers=True)
            assert hashlib.sha256(text.encode()).hexdigest() == assignment['source_sha256'], item.document_id
            checkpoint = STATE / 'targets' / label(model) / (item.document_id + '.json')
            if (item.document_id, item.mode) in done:
                continue
            if checkpoint.exists() and json.loads(checkpoint.read_text())['status'] in ('success', 'abstained'):
                continue
            pending.append((model, item, checkpoint))
        print(f'{model}: {len(existing[model])} original rows, {len(pending)} pending targets', flush=True)
        task_groups.append(pending[:1] if args.smoke else pending)
    # Round-robin the selected models so the concurrent worker pool stays
    # balanced across providers instead of exhausting one model first.
    tasks = [task for group in zip_longest(*task_groups) for task in group if task is not None]
    if args.dry_run:
        return
    client = RecordedClient(llm.client_for(VERIFIER))

    def work(task):
        model, item, checkpoint = task
        LOCAL.model, LOCAL.document = label(model), item.document_id
        config = qagen.GenerationConfig(
            generation_model=model,
            generation_reasoning_effort=ctx.settings.llm.generation_reasoning_effort,
            grader=GraderConfig(model=VERIFIER, reasoning_effort=ctx.settings.llm.grading_reasoning_effort,
                               thinking_budget_tokens=ctx.settings.llm.thinking_budget_tokens,
                               thinking_max_tokens=ctx.settings.llm.thinking_max_tokens),
            retries=ctx.settings.llm.retries, candidate_parser=ctx.domain.qac_candidate_parser)
        result = {'model': model, 'item': asdict(item), 'rows': [],
                  'at': datetime.now(timezone.utc).isoformat()}
        try:
            rows = qagen.generate_for_document(item, grouped[item.family], schema=ctx.schema,
                    prompts=prompts, config=config, generation_client=client,
                    grading_client=client, language_order=order)
            result.update(status='success' if rows else 'abstained', rows=rows)
        except Exception as error:
            chain = []
            while error:
                chain.append(f'{type(error).__name__}: {error}')
                error = error.__cause__
            result.update(status='failed', error=' <- '.join(chain))
        save_json(checkpoint, result)
        return result

    def export():
        summary = {}
        for model in args.models:
            rows = list(existing[model])
            failures, abstained, completed = [], 0, 0
            for checkpoint in sorted((STATE / 'targets' / label(model)).glob('*.json')):
                result = json.loads(checkpoint.read_text())
                if result['status'] == 'failed':
                    item = result['item']
                    failures.append({'source_document_id': item['document_id'],
                        'publication_number': item['family'], 'question_language': item['language'],
                        'mode': item['mode'], 'error': result['error']})
                else:
                    rows.extend(result['rows'])
                    completed += 1
                    abstained += result['status'] == 'abstained'
            rows.sort(key=lambda r: (r['source_document_id'], r['mode'], int(r['candidate_rank'])))
            normalized = [qagen.normalize_row(row, fields) for row in rows]
            path = OUTPUT / (stem(model) + '.csv')
            for destination, output_rows in [(path, normalized),
                     (OUTPUT / (stem(model) + '_best.csv'), qagen.select_best(normalized, ctx.schema))]:
                temp = destination.with_suffix('.tmp')
                corpus.write_rows(temp, output_rows, fields)
                temp.replace(destination)
            corpus.write_rows(OUTPUT / (stem(model) + '_failures.csv'), failures,
                ('source_document_id', 'publication_number', 'question_language', 'mode', 'error'))
            covered = len({r['source_document_id'] for r in rows})
            summary[model] = {'candidate_rows': len(rows), 'documents_with_questions': covered,
                'abstained': abstained, 'failures': len(failures),
                'pending': sum(a['mode'] != 'skip' for a in plans[model]) - covered - abstained - len(failures)}
        save_json(STATE / 'summary.json', summary)
        return summary

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, future in enumerate(as_completed([pool.submit(work, task) for task in tasks]), 1):
            result = future.result()
            print(f"[{index}/{len(tasks)}] {result['model']} {result['item']['document_id']}: "
                  f"{result['status']} {len(result['rows'])} rows {result.get('error', '')}", flush=True)
            if index % 20 == 0:
                export()
    print(json.dumps(export(), indent=2), flush=True)


if __name__ == '__main__':
    main()
