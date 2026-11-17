"""Check safeguards that matter for a decision based on sparse generation data."""
import pytest
from model_selection_analysis import PRACTICAL_MARGIN, PREFERRED, analyze, build_scores, paired


def candidate(model, target, score, *, parent=None, source='baseline/model.csv'):
    return {'corpus': 'un', 'question_language': 'en', 'mode': 'lookup',
            'generator_model_id': model, 'target_id': target,
            'document_id': parent or target, 'source_document_id': target,
            'total_score': str(score), 'grading_status': 'completed', 'plot_source': source}


def recommendation(rows, margin=PRACTICAL_MARGIN):
    return next(r for r in analyze(rows)[0] if r['practical_margin'] == margin)


def test_multiple_candidates_do_not_outweigh_other_input_documents():
    rows = [candidate(PREFERRED[0], 'one', 40)] * 8
    rows += [candidate(PREFERRED[0], 'two', 20)]
    scores, _, _ = build_scores(rows)
    assert next(iter(scores.values()))[PREFERRED[0]] == {
        ('baseline', 'one'): 40, ('baseline', 'two'): 20}
    assert recommendation(rows)['haiku_mean'] == 30


def test_high_score_on_one_document_does_not_replace_preferred_models():
    rows = [candidate(m, str(i), 30) for m in PREFERRED for i in range(20)]
    rows.append(candidate('other', '0', 40))
    row = recommendation(rows)
    assert row['observed_leader'] == 'other'
    assert row['recommended_model'] in PREFERRED
    assert row['leader_paired_parent_documents'] == 1
    assert row['leader_ci95_low'] is None


def test_consistent_material_advantage_can_replace_both_preferred_models():
    rows = [candidate(m, str(i), 30) for m in PREFERRED for i in range(20)]
    rows += [candidate('other', str(i), 30.75) for i in range(20)]
    assert recommendation(rows)['recommended_model'] == 'other'
    assert recommendation(rows, margin=1)['recommended_model'] in PREFERRED


def test_many_blocks_of_one_document_are_not_independent_documents():
    rows = [candidate(m, str(i), score, parent='one-document')
            for m, score in [(PREFERRED[0], 30), (PREFERRED[1], 30), ('other', 40)]
            for i in range(20)]
    row = recommendation(rows)
    assert row['leader_paired_targets'] == 20
    assert row['leader_paired_parent_documents'] == 1
    assert row['recommended_model'] in PREFERRED


def test_pairing_requires_same_experiment_but_recovery_keeps_original_input():
    rows = [candidate(PREFERRED[0], 'one', 30, source='run_a/results.csv'),
            candidate(PREFERRED[1], 'one', 40, source='run_b/results.csv')]
    scores, parents, _ = build_scores(rows)
    cell = next(iter(scores))
    assert paired(scores[cell], parents, cell, *PREFERRED)['paired_targets'] == 0
    rows = [candidate(PREFERRED[0], 'one', 30,
                      source='data/new_language_generation_20261117/round_01/checkpoint.json'),
            candidate(PREFERRED[1], 'one', 32,
                      source='data/new_language_generation_20261117/round_01/repair/checkpoint.json')]
    scores, parents, _ = build_scores(rows)
    cell = next(iter(scores))
    assert paired(scores[cell], parents, cell, *PREFERRED)['paired_margin'] == pytest.approx(-2)
