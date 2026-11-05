# Accuracy of the 30-document pilot

GPT-6 Luna via OpenRouter was tested on exactly 10 documents per corpus. The 757 alignment tasks are source passages crossed with available target-language versions, not additional sampled documents.

All 734 final exact quotations were semantically reviewed. There were also 15 not_found results, 6 uncertain results, and 2 non-verbatim quotations. These are qualitative assistant reviews, not independent expert gold annotations.

| Corpus | Attempts | Exact quotes reviewed | Meaning and qualifications preserved | All criteria, including minimality |
|---|---:|---:|---:|---:|
| eurlex | 228 | 226 | 216/226 (95.6%) | 216/226 (95.6%) |
| un | 485 | 469 | 449/469 (95.7%) | 448/469 (95.5%) |
| chemistry | 44 | 39 | 35/39 (89.7%) | 28/39 (71.8%) |

EUR-Lex and UN pass the agreed corpus-level feasibility criteria. Chemistry fails: four quotations lose or change required support, and seven additional quotations contain avoidable material. Performance varies across target languages; corpus averages do not imply that every language clears the threshold.

## Checked failure examples

- EUR-Lex: an English quotation began “gene therapy clinical trials may be carried out” after omitting the preceding “No,” reversing a prohibition.
- UN: a Spanish quotation described improved resource transparency but omitted the preceding support for the Anti-Corruption Commission, which answered the question about how that improvement was to happen.
- Chemistry: a French quotation omitted the complete anatomy-imaging paragraph required by the source passage. Three French-to-German process alignments changed “dissolve OR oligomerise” to “dissolve AND oligomerise” because the supplied language versions themselves differ.
- Correct abstention: a UN Spanish source says 60 treaty parties while its Arabic and Russian counterparts say 70. The model returned not_found for those two alignments.

Exact substring matching is necessary but does not detect missing negation, conditions, attribution, or overlong excerpts. The reviewed accepted subset is stored locally in accepted_alignments.jsonl; raw model outputs remain unchanged.

## Cost already requested

Estimated full eligible workload: **$2.93 USD**, or **$3.83** with the stated conservative allowances. This is a budget estimate, not quality approval.

Recorded pilot API charges: **$0.1315**. An interrupted parser-fix run may have left up to three unrecorded in-flight requests; their billing is unknown. The full report keeps that uncertainty separate.

Input counts for unobserved requests are calibrated local-tokenizer estimates because OpenRouter has no compatible token-count endpoint. Full-workload generation, retrieval chunking, and Hugging Face publication were not performed.
