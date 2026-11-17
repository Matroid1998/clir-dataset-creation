# New-language ten-model generation

Select 50 documents per new language: Arabic and Russian for UN, Japanese and
Russian for the accepted BASF abstract-pair corpus. UN reuses the same 50 source
documents and blocks across both languages. BASF uses separate publications
because the accepted Japanese and Russian endpoints have no common publication.

All ten existing generators use the same selected inputs within each language.
Jev selects eligible personas; Gemini 3.8 Flash grades faithfulness and quality.
Unsupported needs may produce a valid abstention. Original generation runs and
published datasets are preserved.

```bash
.venv/bin/python experiments/new_language_generation/run.py --dry-run
.venv/bin/python -u experiments/new_language_generation/run.py --workers 10
```

Workers are per language set; the four sets run concurrently. Completed patent
targets have individual JSON checkpoints. UN stages and requests are journaled
in native `run.sqlite` databases. Inputs, prompts, decisions and raw responses
are saved under `data/new_language_generation_20261117/`.

The finishing process can wait for the initial run, retry failures, and create
combined and per-model CSV files:

```bash
.venv/bin/python -u experiments/new_language_generation/finish.py --wait-running
```

UN repairs use separate journals and reuse validated original stages. Failed
generation attempts receive explicit format/evidence feedback; actual repair
requests are recorded. Original journals retain their original attempts and
failures. Up to three repair passes are attempted. Remaining failures are
reported explicitly in `summary.json`; they are never treated as abstentions.

Final files include `summary.json`, `<corpus>/<language>_results.csv`, and
`<corpus>/<language>_<model>.csv`. These files are local; publication is separate.
