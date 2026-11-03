# Coordinator notes and example provenance

Two four-page documents (three pages of guidance and one qualification-test page) for annotating chemistry and legal questions and their supplied answers. At the user's request, annotators see one fixed set of eight criteria, with no question-type names, categories, selection rules or type-specific scoring instructions. Both retain three faithfulness scores and five quality scores, using the original domain-specific 1–5 scale: 4 can represent fully successful execution; 5 requires demonstrably exceptional work.

| Domain | Share with annotators | Editable Word copy | Markdown source |
|---|---|---|---|
| Chemistry | [PDF](../chemistry_annotation_guidelines.pdf) | [DOCX](../chemistry_annotation_guidelines.docx) | [Markdown](../chemistry_annotation_guidelines.md) |
| Legal, EUR-Lex and UN | [PDF](../legal_annotation_guidelines.pdf) | [DOCX](../legal_annotation_guidelines.docx) | [Markdown](../legal_annotation_guidelines.md) |

The PDFs have fixed four-page layouts. Word pagination can vary with the application and installed fonts. The guidelines are in English. Any non-English example must include a clearly labeled English translation.

The eight internal score fields remain Grounding, Precision, Numerical fidelity, Search realism, Standalone clarity, Information value, Focus and formulation, and Linguistic quality. Public headings express these as practical questions for domain experts, followed by the unified criterion name in parentheses. Every criterion paragraph defines the criterion first, then gives judging and scoring instructions. Faithfulness/quality group names are not shown. Each guide introduces one genuine source document before the criteria. Every criterion has a Good and Bad question–answer pair, now followed by a suggested 1–5 score and a short reason for that criterion only. Separate verdict and evidence fields are not shown. The private alignment record maps the plain-language headings to the unchanged internal fields. Numerical fidelity remains 4 with a separate not-applicable notation when no relevant numerical/citational content occurs. An overall total remains /40; inspect individual scores rather than averaging away unsupported content.

The chemistry quality criteria map directly into these five common families. The legal rubric is harmonized, not a lossless renaming: Informativeness and Consequence are both assessed under Information value; Focus, Lexical distance, and the wording aspects of Terminology and distance are assessed under Focus and formulation. Their checks remain explicit, but some original legal templates score two of these aspects separately or omit an aspect scored in the common rubric. The resulting weights and scores are not automatically interchangeable with the original verifier scores or totals. Judge the complete item against the unified definitions; do not calculate a new score by averaging old fields. The private alignment record documents each mapping and its limits.

The user requested saved generated questions where applicable and authorized newly written examples when needed. The paired-example records distinguish verbatim outputs from adapted or newly authored teaching examples; authored mistakes must not be represented as observed model errors. A pair may be reused for different checks. Good, Bad and each suggested score refer only to the criterion being illustrated, not an overall judgment. Suggested scores are teaching recommendations rather than expert annotations or copied machine grades. Each private pair records its teaching_score and score_reason. The earlier example-provenance files are retained as historical source records; the paired-example files describe the current public guides.

The final page is an open-book check of how to perform the annotation task. It has 10 single-choice questions about the task and scoring instructions, each with four options and one correct answer. The private key gives the correct choices and explanations and requires confirmed understanding before proceeding; corrected responses after clarification are accepted. Do not distribute the key with the test.

## Curator source register

The `curator/` files retain original record locations, IDs, source snapshots, criterion mappings, and verification results. They reveal question types and may reveal generator identities through file names or task metadata. Share only the public guidelines and blinded annotation packets, excluding these files and any type/mode labels from the packets.

- [Chemistry example provenance](chemistry_example_provenance.json)
- [Legal example provenance](legal_example_provenance.json)
- [Current chemistry pairs](chemistry_paired_examples.json)
- [Current legal pairs](legal_paired_examples.json)
- [Record verification](example_verification.json)
- [Rubric alignment and source hashes](rubric_alignment.json)
- [Qualification-test answer key and passing rule](qualification_test_answer_key.md)
- [Qualification-test questions and answer records](qualification_tests.json)
- [Document build script](build_guidelines.py)

To rebuild review copies, run `python3 docs/annotation/curator/build_guidelines.py docs/annotation/chemistry_annotation_guidelines.md docs/annotation/legal_annotation_guidelines.md`. The script uses the adjacent Word files as style templates and writes review copies to `/tmp/clir_guideline_build/`. Check their content and four-page PDF layouts before replacing the public files.

The saved legal examples come from `data/legal/qac/generation_600_20261008_model_comparison/`. Saved chemistry examples come from `data/chemistry/ai4chem_abstract_pairs/qac/chemistry_google_patents_model_comparison_generator/`, with sources verified in the corresponding multilingual corpus. Verbatim pairs retain their original question and answer; authored variants are recorded separately. The common definitions draw on the chemistry `chemistry-clir-exceptional-r2` and legal `legal-clir-exceptional-r3` verifier files listed in the alignment record. The legal run itself pins `legal-decider-score32-optimized-20261007`; saved prompt hashes remain in provenance so archived judgments are not silently treated as current human scores. Good/Bad labels illustrate the selected criterion and are not reproductions of archived machine grades. Generator and verifier prompts are unchanged.
