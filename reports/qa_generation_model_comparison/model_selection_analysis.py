"""Document-balanced, paired margin analysis for choosing generation models."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from plot_data import OUT, snapshot_rows

RESULTS = OUT / 'model_selection'
PREFERRED = ('anthropic/claude-haiku-5.5', 'qwen/qwen3.8-27b')
PRACTICAL_MARGIN = .5
MIN_SWITCH_DOCUMENTS = 10
REPLICATES = 10000


def experiment_of(source):
    """Keep repeated historical experiments separate, but join repair journals."""
    if 'data/new_language_generation_' in source:
        return source.split('/round_01')[0] + '/round_01'
    return str(Path(source).parent)


def build_scores(rows, aggregate='mean'):
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    parents = {}
    candidate_counts = defaultdict(int)
    for row in rows:
        if row['grading_status'] == 'failed' or not row['total_score'].strip():
            continue
        score = float(row['total_score'])
        if not np.isfinite(score) or not 0 <= score <= 40:
            raise ValueError('Invalid total score')
        cell = (row['corpus'], row['question_language'], row['mode'])
        model = row['generator_model_id']
        target = row['target_id'] or row['source_document_id'] or row['document_id']
        unit = (experiment_of(row['plot_source']), target)
        parent = row['document_id'] if row['corpus'] in ('eurlex', 'un') else row['source_document_id']
        parents[cell, unit] = parent or target
        values[cell][model][unit].append(score)
        candidate_counts[cell, model] += 1
    scores = {cell: {model: {unit: float(np.mean(v) if aggregate == 'mean' else max(v))
                            for unit, v in units.items()}
                     for model, units in models.items()} for cell, models in values.items()}
    return scores, parents, candidate_counts


def paired(models, parents, cell, model, reference, comparisons=1, replicates=REPLICATES):
    common = sorted(set(models.get(model, {})) & set(models.get(reference, {})))
    clustered = defaultdict(list)
    for unit in common:
        clustered[parents[cell, unit]].append(models[model][unit] - models[reference][unit])
    result = {'paired_targets': len(common), 'paired_parent_documents': len(clustered),
              'paired_margin': None, 'ci95_low': None, 'ci95_high': None,
              'adjusted_lower_bound': None}
    if not common:
        return result
    result['paired_margin'] = float(np.mean([v for values in clustered.values() for v in values]))
    if len(clustered) < 5:
        return result
    totals = np.array([sum(v) for v in clustered.values()])
    sizes = np.array([len(v) for v in clustered.values()])
    seed = int(hashlib.sha256(repr((cell, model, reference)).encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(totals), size=(replicates, len(totals)))
    differences = totals[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    result.update(ci95_low=float(np.quantile(differences, .025)),
                  ci95_high=float(np.quantile(differences, .975)),
                  adjusted_lower_bound=float(np.quantile(differences, .05 / max(1, comparisons))))
    return result


def analyze(rows, aggregate='mean'):
    scores, parents, candidate_counts = build_scores(rows, aggregate)
    cells, pair_rows, model_rows = [], [], []
    for cell, models in sorted(scores.items()):
        corpus, language, mode = cell
        means = {model: float(np.mean(list(units.values()))) for model, units in models.items()}
        preferences = [model for model in PREFERRED if model in models]
        if not preferences:
            raise ValueError(f'No preferred generator has observations for {cell}')
        hq = paired(models, parents, cell, *PREFERRED) if len(preferences) == 2 else None
        if hq and hq['paired_parent_documents'] >= 5:
            preferred = (PREFERRED[0] if hq['paired_margin'] > 0 else PREFERRED[1]
                         if hq['paired_margin'] < 0 else max(preferences, key=lambda m: (means[m], len(models[m]))))
        else:
            preferred = max(preferences, key=lambda m: (means[m], len(models[m])))
        leader = max(means, key=lambda m: (means[m], len(models[m]), m in PREFERRED))
        raw_preferred = max(preferences, key=means.get)
        alternatives = [model for model in models if model not in PREFERRED]
        multiplicity = max(1, len(alternatives) * len(preferences))
        comparisons = {}
        for model in models:
            for reference in preferences:
                if model == reference:
                    continue
                comparison = paired(models, parents, cell, model, reference, multiplicity)
                comparisons[model, reference] = comparison
                pair_rows.append({'corpus': corpus, 'language': language, 'mode': mode,
                                  'model': model, 'reference': reference, **comparison})
        margins = {}
        for model in alternatives:
            values = [comparisons[model, reference] for reference in preferences]
            if all(v['paired_parent_documents'] >= MIN_SWITCH_DOCUMENTS for v in values):
                margins[model] = min(v['adjusted_lower_bound'] for v in values)
        observed_units = set().union(*(set(v) for v in models.values()))
        for model, units in models.items():
            model_rows.append({'corpus': corpus, 'language': language, 'mode': mode,
                               'model': model, 'graded_candidates': candidate_counts[cell, model],
                               'target_runs': len(units),
                               'parent_documents': len({parents[cell, unit] for unit in units}),
                               'observed_target_union': len(observed_units),
                               'observed_target_coverage': len(units) / len(observed_units),
                               'target_balanced_mean': means[model]})
        preferred_n = len({parents[cell, unit] for unit in models[preferred]})
        leader_pair = comparisons.get((leader, preferred), {}) if leader != preferred else {
            'paired_targets': len(models[preferred]), 'paired_parent_documents': preferred_n,
            'paired_margin': 0.0, 'ci95_low': 0.0, 'ci95_high': 0.0,
            'adjusted_lower_bound': 0.0}
        base = {'corpus': corpus, 'language': language, 'mode': mode,
                'observed_leader': leader, 'leader_mean': means[leader],
                'leader_parent_documents': len({parents[cell, unit] for unit in models[leader]}),
                'best_preferred_by_raw_mean': raw_preferred,
                'best_preferred_mean': means[raw_preferred],
                'raw_margin_over_best_preferred': means[leader] - means[raw_preferred],
                'haiku_raw_gap': means[leader] - means[PREFERRED[0]] if PREFERRED[0] in means else None,
                'qwen_raw_gap': means[leader] - means[PREFERRED[1]] if PREFERRED[1] in means else None,
                'preferred_choice': preferred, 'preferred_mean': means[preferred],
                'preferred_parent_documents': preferred_n,
                'haiku_mean': means.get(PREFERRED[0]), 'qwen_mean': means.get(PREFERRED[1]),
                'haiku_minus_qwen_paired': hq['paired_margin'] if hq else None,
                'haiku_qwen_paired_documents': hq['paired_parent_documents'] if hq else 0,
                'haiku_minus_qwen_ci95_low': hq['ci95_low'] if hq else None,
                'haiku_minus_qwen_ci95_high': hq['ci95_high'] if hq else None,
                **{f'leader_{key}': value for key, value in leader_pair.items()}}
        for delta in (.5, 1., 2.):
            eligible = [model for model, lower in margins.items() if lower > delta]
            selected = max(eligible, key=lambda m: (margins[m], means[m])) if eligible else preferred
            cells.append({**base, 'practical_margin': delta, 'recommended_model': selected,
                          'recommended_raw_gap': means[leader] - means[selected],
                          'decision': 'clear_alternative_advantage' if eligible else 'prefer_haiku_or_qwen',
                          'evidence': ('provisional' if preferred_n < 10 or not hq
                                       or hq['paired_parent_documents'] < 10 else 'document_supported'),
                          'selection_adjusted_lower_bound': margins.get(selected) if eligible else None,
                          'aggregate': aggregate})
    return cells, pair_rows, model_rows


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = snapshot_rows()
    primary, pairs, models = analyze(rows)
    best, _, _ = analyze(rows, aggregate='best')
    chosen = [dict(row) for row in primary if row['practical_margin'] == PRACTICAL_MARGIN]
    best_lookup = {(r['corpus'], r['language'], r['mode']): r for r in best
                   if r['practical_margin'] == PRACTICAL_MARGIN}
    for row in chosen:
        sensitivity = best_lookup[row['corpus'], row['language'], row['mode']]
        row['best_candidate_recommended_model'] = sensitivity['recommended_model']
        row['changes_with_best_candidate'] = sensitivity['recommended_model'] != row['recommended_model']
    write_csv(RESULTS / 'recommendations.csv', chosen)
    write_csv(RESULTS / 'all_model_document_scores.csv', models)
    write_csv(RESULTS / 'paired_margins.csv', pairs)
    write_csv(RESULTS / 'sensitivity.csv', [*primary, *best])
    summary = {'practical_margin': PRACTICAL_MARGIN, 'score_scale': 40,
               'preferred_models': PREFERRED, 'bootstrap_replicates': REPLICATES,
               'minimum_parent_documents_for_alternative': MIN_SWITCH_DOCUMENTS,
               'switch_rule': 'Alternative must beat each available preferred model by more than delta on the multiplicity-adjusted paired bootstrap lower bound.',
               'multiplicity': 'Bonferroni one-sided 95% lower bounds across alternative-versus-preferred comparisons within each cell; no across-cell claim.',
               'primary_aggregate': 'Average candidate grade per target/run; equal weight per target/run.',
               'sensitivity_aggregate': 'Best candidate per target/run.',
               'bootstrap_unit': 'Parent UN document, EUR-Lex act or patent publication; paired targets retained within each parent.',
               'confidence_intervals': 'Percentile cluster bootstrap; suppressed below five independent parent documents. Exploratory intervals, not equivalence tests.',
               'provisional_definition': 'Fewer than ten independent documents in the chosen preferred model or in the paired Haiku/Qwen comparison.',
               'scope': 'Conditional on successfully graded outputs; failures and abstentions are not assigned invented scores.',
               'coverage_definition': 'Graded target/run coverage relative to the observed union within a cell; not an attempted-generation success rate.',
               'input_sha256': hashlib.sha256((OUT / 'generated_question_plot_inputs.csv').read_bytes()).hexdigest(),
               'cells_with_data': len(chosen),
               'within_raw_margin': {str(delta): sum(r['raw_margin_over_best_preferred'] <= delta + 1e-9
                                                     for r in chosen) for delta in (.5, 1., 2.)},
               'best_candidate_route_changes': sum(r['changes_with_best_candidate'] for r in chosen),
               'corpora': {}}
    for corpus in ('eurlex', 'un', 'chemistry'):
        selected = [r for r in chosen if r['corpus'] == corpus]
        summary['corpora'][corpus] = {
            'cells_with_data': len(selected),
            'recommended_models': {model: sum(r['recommended_model'] == model for r in selected)
                                   for model in sorted({r['recommended_model'] for r in selected})},
            'provisional_cells': sum(r['evidence'] == 'provisional' for r in selected),
            'largest_raw_gaps': sorted(selected, key=lambda r: r['raw_margin_over_best_preferred'], reverse=True)[:5]}
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: {key: value for key, value in v.items() if key != 'largest_raw_gaps'}
                      for k, v in summary['corpora'].items()}, indent=2))


if __name__ == '__main__':
    main()
