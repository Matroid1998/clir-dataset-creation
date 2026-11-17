# Generated-question count and grade plots

Refresh all figures and their CSV/provenance files with:

```bash
MPLCONFIGDIR=/tmp/clir-matplotlib .venv/bin/python reports/qa_generation_model_comparison/refresh_plots.py
```

`generated_questions_and_grades.pdf` combines the legal and patent count grids,
the patent model counts, both top-generator grids, and the all-corpus grade
comparison. Individual PDF/PNG files remain available. The three existing legal
SVG files in `data/legal/qac/generation_600_20261008_model_comparison/` are also
refreshed.

Inputs come from `docs/generation_export.json` and the new-language runs in
`data/new_language_generation_20261117/`. `plot_data.py` freezes one consistent
CSV snapshot before rendering. Validated repair outcomes replace failed task
outcomes; they are never added as duplicate candidates. A failed later retry
does not replace an earlier validated result.

Question counts retain the original definition: every generated candidate row,
including separate outputs from different models. Original candidates whose
grading failed remain in generation counts. Failed supplemental cases without
validated outputs are excluded. Grade comparisons require a valid numeric
total score and exclude failed grading; low scores remain included. Winners
use exact unrounded means and show their sample sizes.

`generated_question_plot_sources.json` records source files, snapshot time,
per-language counts and remaining supplemental failures. The companion
`generated_documents_by_corpus_and_language.csv` counts each source document
once across models, separately from candidate counts. Average grades are
descriptive: persona mixes, sample sizes and prompt versions differ across runs.

If recovery is still running, `--wait-for-completion` refreshes now and once more
after its final summary is written.
