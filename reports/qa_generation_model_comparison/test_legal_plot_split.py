"""Model winners must not pool question grades from different legal corpora."""
from plot_legal_questions import CORPORA, MODELS, analyze


def candidate(corpus, model, score, **fields):
    return {'corpus': corpus, 'generator_model_id': model, 'question_language': 'en',
            'mode': 'lookup', 'total_score': str(score), 'grading_status': 'completed', **fields}


def test_corpus_specific_winners_use_only_that_corpus_grades():
    model_a, model_b = list(MODELS)[:2]
    rows = [candidate('eurlex', model_a, 10), candidate('eurlex', model_b, 30),
            candidate('un', model_a, 40), candidate('un', model_b, 20)]
    eurlex = analyze(rows, 'eurlex')
    un = analyze(rows, 'un')
    assert len(eurlex[0]) == len(un[0]) == 2
    assert eurlex[-1]['en', 'lookup'] == [model_b]
    assert un[-1]['en', 'lookup'] == [model_a]
    assert eurlex[3]['en', 'lookup', model_b] == 30
    assert un[3]['en', 'lookup', model_a] == 40


def test_separate_personas_and_failed_grade_counting():
    model = next(iter(MODELS))
    rows = [candidate('eurlex', model, 30, mode='fact_pattern'),
            candidate('un', model, 20, mode='practitioner'),
            candidate('un', model, '', mode='practitioners', grading_status='failed')]
    eurlex = analyze(rows, 'eurlex')
    un = analyze(rows, 'un')
    assert eurlex[1]['en', 'fact_pattern'] == 1
    assert un[1]['en', 'practitioners'] == 2
    assert len(un[2]['en', 'practitioners', model]) == 1
    assert 'fact_pattern' not in CORPORA['un']['modes']
    assert 'practitioners' not in CORPORA['eurlex']['modes']
