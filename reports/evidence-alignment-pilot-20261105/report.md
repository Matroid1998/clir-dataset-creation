# Evidence-alignment pilot

Run status: **completed**.
Quality gate: **failed**.

This is a 30-document feasibility pilot, not a dataset-wide accuracy guarantee.
Semantic decisions are qualitative Codex assistant reviews, not independent expert annotations.

| Corpus | Tasks | First exact | Final exact | Reviewed valid | Pending reviews | Coverage | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| chemistry | 44 | 35 | 39 | 28 | 0 | 63.6% | failed |
| eurlex | 228 | 224 | 226 | 216 | 0 | 94.7% | passed |
| un | 485 | 467 | 469 | 448 | 0 | 92.4% | passed |

A valid match is an exact quotation with a current semantic review confirming equivalent support, preserved qualifications, minimality, and language correctness.

## Outcomes by target language

| Corpus / language | Final outcomes | Reviewed precision | Valid coverage |
|---|---|---:|---:|
| chemistry / de | matched: 8 | 50.0% | 50.0% |
| chemistry / en | matched: 10, not_found: 2 | 90.0% | 75.0% |
| chemistry / fr | matched: 17, not_found: 2, uncertain: 1 | 70.6% | 60.0% |
| chemistry / zh | matched: 4 | 75.0% | 75.0% |
| eurlex / de | matched: 65 | 93.8% | 93.8% |
| eurlex / en | matched: 54 | 94.4% | 94.4% |
| eurlex / es | matched: 50 | 96.0% | 96.0% |
| eurlex / fr | invalid: 1, matched: 57, uncertain: 1 | 96.6% | 94.9% |
| un / ar | matched: 96, not_found: 1 | 92.7% | 91.8% |
| un / en | matched: 62, not_found: 1 | 91.9% | 90.5% |
| un / es | matched: 72, not_found: 4 | 94.4% | 89.5% |
| un / fr | invalid: 1, matched: 81, not_found: 3, uncertain: 4 | 98.8% | 91.0% |
| un / ru | matched: 96, not_found: 1 | 95.8% | 94.8% |
| un / zh | matched: 62, not_found: 1 | 98.4% | 96.8% |

## Cost

Recorded pilot cost: **$0.131542 USD**, from 121 attempts, 526,763 input tokens and 149,215 output tokens. Provider-reported usage.cost where present; token-rate calculation otherwise. Token-rate calculation alone: $0.131542. This is API usage accounting, not an invoice.

An interrupted run may have up to 3 additional requests without saved responses; final pilot billing is unknown.

Allowance for these requests: $0.055549; recorded spending plus allowance: $0.187092. At most the recorded incident request count, each using the largest observed/estimated input, all cache-write input, and the full 32,768 output-token limit. This is an allowance, not a verified billing bound.

The requested cost estimate is for budgeting only; the pilot has not passed all quality criteria.

| Corpus | Standard estimate | Conservative | Batch estimate |
|---|---:|---:|---:|
| eurlex | $0.4602 | $0.5908 | $0.2301 |
| un | $2.1743 | $2.8582 | $1.0872 |
| chemistry | $0.2978 | $0.3816 | $0.1489 |
| **Total** | **$2.9324** | **$3.8305** | **$1.4662** |

Full-request input counts (approximate where calibrated_local_tokenizer is recorded); pilot output tokens per anchor including protocol retries through the first valid response; pilot attempt counts per group. Strata: corpus/source/target/genre, then corpus/source/target, corpus/target, corpus. All input is priced at the cache-write rate ($0.125/M), with no assumed cache-read savings, matching first-call billing. Conservative estimate adds 25% to measured output and retry-adjusted input, plus 10% input allowance for calibrated counts (a heuristic, not a confidence interval). Duplicate calls after the first valid raw response are excluded from projection only; actual spending includes them.

Standard rates per million tokens: $0.10 input, $0.01 cached input, $0.125 cache writes, $0.50 output. Requests above 272,000 input tokens use 2× input/cache and 1.5× output rates. Batch estimates use half those rates. [Official pricing](https://developers.openai.com/api/docs/models/gpt-6-luna)

No retrieval chunks or Hugging Face updates are produced by this pilot.
