"""Citation extraction and resolution for the UN corpus.

Positive cases are real corpus citation strings; negative cases are the
look-alikes the extractor must ignore (internal references, dates, non-UN
symbols). Resolution tests use a small in-memory symbol map.
"""

from __future__ import annotations

import pytest

from clir_bench.domains.legal.qac import un_references as refs
from clir_bench.domains.legal.qac.un_references import (
    KIND_DECISION,
    KIND_ES,
    KIND_ROMAN,
    KIND_SC,
    KIND_SLASH,
    KIND_SYMBOL,
    anchored_block_index,
    extract_citations,
    normalise_symbol,
    referenced_docs,
    resolve_citations,
)

MAP = {
    "S/RES/872(1993)": "1993/s/res/872_1993_",
    "S/RES/1265(1999)": "1999/s/res/1265_1999_",
    "S/RES/1296(2000)": "2000/s/res/1296_2000_",
    "S/RES/1244(1999)": "1999/s/res/1244_1999_",
    "A/RES/51/210": "1997/a/res/51/210",
    "A/RES/57/270": "2003/a/res/57/270",
    "A/HRC/RES/17/4": "2011/a/hrc/res/17/4",
    "S/PRST/1994/16": "1994/s/prst/1994/16",
    "S/1994/565": "1994/s/1994/565",
    "A/53/500/ADD.1": "1998/a/53/500/add_1",
}


def kinds(cites):
    return [c.kind for c in cites]


def by_kind(cites, kind):
    return [c for c in cites if c.kind == kind]


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def test_sc_resolution_single():
    text = ("Reaffirming its resolution 872 (1993) of 5 October 1993 by which "
            "it established UNAMIR,")
    (c,) = extract_citations(text)
    assert c.kind == KIND_SC and c.symbol == "S/RES/872(1993)"


def test_sc_resolution_plural_run():
    text = "Recalling its resolutions 1265 (1999) and 1296 (2000) on the protection of civilians,"
    cites = extract_citations(text)
    assert [c.symbol for c in cites] == ["S/RES/1265(1999)", "S/RES/1296(2000)"]


def test_slash_resolution_with_named_body():
    text = "the General Assembly, by its resolution 51/210 of 17 December 1996, established"
    (c,) = extract_citations(text)
    assert c.kind == KIND_SLASH
    assert c.candidates == ("A/RES/51/210",)
    assert c.organ_hint == "General Assembly"


def test_section_letter_captured():
    text = "pursuant to resolution 57/270 B of 23 June 2003 on integrated follow-up"
    (c,) = extract_citations(text)
    assert c.section_letter == "B"


def test_bare_symbols_and_parenthesised():
    text = ("statements by the President (S/PRST/1994/16) and the report of "
            "the Secretary-General dated 13 May 1994 (S/1994/565)")
    cites = by_kind(extract_citations(text), KIND_SYMBOL)
    assert [c.symbol for c in cites] == ["S/PRST/1994/16", "S/1994/565"]


def test_symbol_addendum_suffix_kept():
    (c,) = extract_citations("as noted in A/53/500/Add.1 to the report")
    assert c.symbol == "A/53/500/ADD.1"


def test_prose_form_symbol_with_year_tail():
    (c,) = extract_citations("acting under S/RES/918 (1994) the Council imposed")
    assert c.symbol == "S/RES/918(1994)"


def test_explicit_symbol_not_doubled_as_slash_item():
    cites = extract_citations("as set out in resolution A/RES/51/210 of the Assembly")
    assert len(by_kind(cites, KIND_SYMBOL)) == 1
    assert not by_kind(cites, KIND_SLASH)


def test_non_un_symbols_rejected():
    text = ("compare OEA/Ser.L/III.15 and MSC/72/13/2 as well as HIV/AIDS "
            "figures and a 24/7 presence")
    assert extract_citations(text) == []


def test_roman_is_unresolvable_and_es_gets_the_dotted_candidate():
    cites = extract_citations(
        "recalling resolution 2819 (XXVI) and resolution ES-10/2 of the Assembly")
    assert kinds(cites) == [KIND_ROMAN, KIND_ES]
    roman, es = cites
    assert roman.candidates == ()
    # Emergency special sessions are GA-only; corpus stores them dotted.
    assert es.candidates == ("A/RES/ES.10/2",)


def test_decision_kind():
    (c,) = extract_citations("Notwithstanding General Assembly decision 51/421, there was")
    assert c.kind == KIND_DECISION
    assert c.candidates == ("A/HRC/DEC/51/421",)


def test_paragraph_anchor_attached():
    text = ("as required by paragraph 11 (j) of Security Council resolution "
            "1244 (1999) of 10 June 1999")
    (c,) = extract_citations(text)
    assert c.kind == KIND_SC and c.symbol == "S/RES/1244(1999)"
    assert c.paragraph == 11


def test_internal_references_are_classified_not_ignored():
    text = ("as noted in paragraph 5 above and in paragraph 3 of the present "
            "resolution, and again on 17 December 1996")
    cites = extract_citations(text)
    assert [c.kind for c in cites] == [refs.KIND_INTERNAL_PARA, refs.KIND_INTERNAL_PARA]
    assert all(c.scope == refs.SCOPE_INTERNAL for c in cites)
    assert [c.numbers for c in cites] == [(5,), (3,)]


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

def test_resolution_chain_and_normalisation():
    cites = resolve_citations(extract_citations(
        "by its resolution 51/210 and Human Rights Council resolution 17/4"), MAP)
    assert cites[0].doc_id == "1997/a/res/51/210"
    assert cites[1].doc_id == "2011/a/hrc/res/17/4"
    assert cites[1].symbol == "A/HRC/RES/17/4"


def test_normalise_symbol_variants():
    assert normalise_symbol("S/RES/1234 (2005)") == "S/RES/1234(2005)"
    assert normalise_symbol("s/1994/565.") == "S/1994/565"


def test_self_citation_dropped():
    cites = resolve_citations(extract_citations("RESOLUTION 872 (1993)"), MAP,
                              citing_doc_id="1993/s/res/872_1993_")
    assert cites[0].doc_id is None


def test_unresolved_citation_keeps_symbol():
    (c,) = resolve_citations(extract_citations("its resolution 48/141 of 1993"), MAP)
    assert c.doc_id is None and c.symbol == "A/RES/48/141"


def test_unresolved_symbol_follows_organ_hint():
    (c,) = resolve_citations(extract_citations(
        "pursuant to Human Rights Council resolution 10/27 of March 2009"), MAP)
    assert c.doc_id is None and c.symbol == "A/HRC/RES/10/27"


@pytest.mark.parametrize("text", [
    "Recalling Human Rights Council resolution 51/1,",
    "Recalling human rights council resolution 51/1,",
    "Recalling Human Rights\nCouncil resolution 51/1,",
    "The Human Rights Council, by its resolution 51/1, requested a report.",
    "Recalling the Human Rights Council's resolution 51/1,",
])
def test_named_body_never_falls_back_to_another_body(text):
    (c,) = resolve_citations(extract_citations(text), {"A/RES/51/1": "ga"})
    assert c.symbol == "A/HRC/RES/51/1"
    assert c.doc_id is None and c.reason == "out_of_corpus"
    assert refs.is_blocking(c)
    assert refs.citation_status([c])["complete"] is False


@pytest.mark.parametrize("organ, symbol", [
    ("General Assembly", "A/RES/51/1"),
    ("Human Rights Council", "A/HRC/RES/51/1"),
])
def test_named_body_selects_its_document_when_numbers_overlap(organ, symbol):
    mapping = {"A/RES/51/1": "ga", "A/HRC/RES/51/1": "hrc"}
    (c,) = resolve_citations(extract_citations(f"Recalling {organ} resolution 51/1,"),
                             mapping)
    assert c.symbol == symbol and c.doc_id == mapping[symbol]
    assert c.reason is None


def test_named_body_applies_to_every_resolution_in_a_run():
    mapping = {"A/RES/51/1": "ga", "A/HRC/RES/51/2": "hrc"}
    first, second = resolve_citations(extract_citations(
        "Recalling Human Rights Council resolutions 51/1 and 51/2,"), mapping)
    assert first.symbol == "A/HRC/RES/51/1"
    assert first.doc_id is None and first.reason == "out_of_corpus"
    assert second.doc_id == "hrc" and second.reason is None


def test_unsupported_named_body_never_falls_back_to_another_body():
    (c,) = resolve_citations(extract_citations(
        "Recalling Commission on Human Rights resolution 2001/1,"),
        {"E/RES/2001/1": "ecosoc"})
    assert c.doc_id is None and c.reason == "out_of_corpus"


def test_lettered_resolution_cannot_fall_back_to_a_different_body():
    (c,) = resolve_citations(extract_citations(
        "Recalling Human Rights Council resolution 51/153 B,"),
        {"A/RES/51/153B": "ga"})
    assert c.doc_id is None and c.reason == "out_of_corpus"


def test_unnamed_slash_resolution_with_overlapping_numbers_is_ambiguous():
    mapping = {"A/RES/51/1": "ga", "A/HRC/RES/51/1": "hrc"}
    (c,) = resolve_citations(extract_citations("Recalling resolution 51/1,"), mapping)
    assert c.doc_id is None and c.reason == "ambiguous_candidates"
    assert refs.is_blocking(c)
    assert referenced_docs([c]) == ([], [])


def test_unnamed_slash_resolution_with_one_match_still_resolves():
    (c,) = resolve_citations(extract_citations("Recalling resolution 51/1,"),
                             {"A/HRC/RES/51/1": "hrc"})
    assert c.doc_id == "hrc" and c.reason is None


def test_nearby_body_is_not_treated_as_the_cited_body():
    text = "The Human Rights Council met. Recalling resolution 51/1,"
    (c,) = resolve_citations(extract_citations(text), {"A/RES/51/1": "ga"})
    assert c.organ_hint == "Human Rights Council"
    assert c.doc_id == "ga" and c.reason is None


@pytest.mark.parametrize("text, symbol", [
    ("The General Assembly established the Human Rights Council by its resolution 60/251,",
     "A/RES/60/251"),
    (("The Assembly established the Peacebuilding Commission and the Human Rights Council "
      "by its resolutions 60/180 and 60/251, respectively."), "A/RES/60/180"),
    (("The Assembly supported the work of the Human Rights Council in its resolutions "
      "65/281 and 66/169,"), "A/RES/65/281"),
    (("Mandates given by the General Assembly and Security Council resolutions "
      "60/180 and 1645 (2005), respectively."), "A/RES/60/180"),
    (("Operations approved by the Assembly or the Security Council, by its resolution "
      "56/274 B of 27 June 2002, the Assembly appropriated funds."), "A/RES/56/274B"),
])
def test_object_or_coordinated_body_does_not_override_the_resolution(text, symbol):
    citations = resolve_citations(extract_citations(text), {symbol: "ga"})
    (citation,) = [c for c in citations if c.kind == KIND_SLASH and c.symbol == symbol]
    assert citation.doc_id == "ga" and citation.reason is None


def test_explicit_symbol_takes_precedence_over_a_nearby_body():
    text = "The Human Rights Council referred to A/RES/51/1."
    mapping = {"A/RES/51/1": "ga", "A/HRC/RES/51/1": "hrc"}
    (c,) = resolve_citations(extract_citations(text), mapping)
    assert c.symbol == "A/RES/51/1" and c.doc_id == "ga"
    assert c.reason is None


def test_referenced_docs_cap_and_order():
    text = ("Recalling its resolutions 1265 (1999) and 1296 (2000), resolution "
            "51/210, resolution 57/270, Human Rights Council resolution 17/4 "
            "and the statement S/PRST/1994/16")
    cites = resolve_citations(extract_citations(text), MAP)
    kept, dropped = referenced_docs(cites, max_references=4)
    assert [c.symbol for c in kept] == [
        "S/RES/1265(1999)", "S/RES/1296(2000)", "A/RES/51/210", "A/RES/57/270"]
    assert dropped == ["A/HRC/RES/17/4", "S/PRST/1994/16"]


def test_referenced_docs_deduplicates():
    text = "resolution 872 (1993) ... recalled resolution 872 (1993) again"
    cites = resolve_citations(extract_citations(text), MAP)
    kept, dropped = referenced_docs(cites)
    assert len(kept) == 1 and not dropped


# --------------------------------------------------------------------------- #
# Paragraph anchoring
# --------------------------------------------------------------------------- #

def test_anchored_block_first_match_wins():
    texts = [
        "Preamble text with no numbering at all here.",
        "1. Decides to establish the mission;\n2. Requests the Secretary-General to report;",
        "Annex\n1. The committee shall meet twice a year.",   # numbering restart
    ]
    assert anchored_block_index(texts, 2) == 1
    assert anchored_block_index(texts, 1) == 1                # first match, not the annex
    assert anchored_block_index(texts, 9) is None


# --------------------------------------------------------------------------- #
# Safe resolution: lettered sections, supplements, special sessions, reasons
# --------------------------------------------------------------------------- #

LETTERED_MAP = dict(MAP)
LETTERED_MAP.update({
    "A/RES/51/153": "1997/a/res/51/153_",
    "A/RES/51/153B": "1997/a/res/51/153b_",
    "A/RES/52/12": "1998/a/res/52/12_",          # plain twin only -- no 52/12B
    "A/52/13(SUPP)": "1998/a/52/13_supp_",
    "A/RES/ES.10/2": "1997/a/res/es.10/2_",
    "A/RES/S.20/2": "2002/a/res/s.20/2_",
    "A/HRC/RES/S.20/1": "2013/a/hrc/res/s.20/1_",
    "A/HRC/RES/S.15/1": "2011/a/hrc/res/s.15/1_",
})


def test_lettered_citation_resolves_only_to_the_lettered_document():
    (c,) = resolve_citations(extract_citations("recalling its resolution 51/153 B,"),
                             LETTERED_MAP)
    assert c.candidates == ("A/RES/51/153B",)
    assert c.doc_id == "1997/a/res/51/153b_"


def test_lettered_citation_never_falls_back_to_the_plain_twin():
    (c,) = resolve_citations(extract_citations("recalling its resolution 52/12 B,"),
                             LETTERED_MAP)
    assert c.doc_id is None and c.reason == "out_of_corpus"


def test_supp_twin_is_a_second_candidate_but_nothing_else_is():
    (c,) = resolve_citations(extract_citations("as shown in document A/52/13, the"),
                             LETTERED_MAP)
    assert c.doc_id == "1998/a/52/13_supp_"
    (c2,) = resolve_citations(extract_citations("as shown in document A/52/676, the"),
                              LETTERED_MAP)
    assert c2.doc_id is None and c2.reason == "out_of_corpus"


def test_emergency_special_session_resolves_dotted():
    (c,) = resolve_citations(extract_citations("recalling resolution ES-10/2 of"),
                             LETTERED_MAP)
    assert c.doc_id == "1997/a/res/es.10/2_"


def test_s_session_organ_surface_pins_the_space():
    (c,) = resolve_citations(extract_citations(
        "recalling Human Rights Council resolution S-15/1 on Libya"), LETTERED_MAP)
    assert c.doc_id == "2011/a/hrc/res/s.15/1_"


def test_s_session_in_both_spaces_is_ambiguous():
    # Session 20 exists as both A/RES/S.20/* and A/HRC/RES/S.20/*.
    (c,) = resolve_citations(extract_citations("recalling resolution S-20/2 of"),
                             LETTERED_MAP)
    assert c.doc_id is None and c.reason == "ambiguous_candidates"


def test_s_session_in_one_space_resolves():
    (c,) = resolve_citations(extract_citations("recalling resolution S-15/1 of"),
                             LETTERED_MAP)
    assert c.doc_id == "2011/a/hrc/res/s.15/1_"


def test_dirty_stored_symbol_is_reachable():
    index = refs.SymbolIndex.from_pairs([("1998/s/res/1173__1998_", "S/RES/1173(.1998)")])
    assert index.n_dirty_keys == 1
    (c,) = resolve_citations(extract_citations("recalling resolution 1173 (1998),"), index)
    assert c.doc_id == "1998/s/res/1173__1998_"


def test_symbol_map_collisions_are_dropped_not_last_write_wins():
    index = refs.SymbolIndex.from_pairs([("doc1", "A/50/1"), ("doc2", "A/50/1")])
    assert index.n_collisions == 1 and "A/50/1" not in index.map


def test_reason_taxonomy_on_misses():
    cites = resolve_citations(extract_citations(
        "recalling resolution 2819 (XXVI) and decision 51/421 and resolution 9999 (2099)"),
        MAP, citing_doc_id="x")
    by_kind = {c.kind: c for c in cites if c.scope == refs.SCOPE_EXTERNAL}
    assert by_kind[KIND_ROMAN].reason == "pre_corpus"
    assert by_kind[KIND_DECISION].reason == "out_of_corpus"
    assert by_kind[KIND_SC].reason == "out_of_corpus"


def test_self_citation_gets_a_benign_reason():
    cites = resolve_citations(extract_citations("Reaffirming its resolution 1244 (1999),"),
                              MAP, citing_doc_id="1999/s/res/1244_1999_")
    (c,) = cites
    assert c.doc_id is None and c.reason == "self_reference"
    assert not refs.is_blocking(c)
    kept, _ = referenced_docs(cites)
    assert kept == []


def test_resolved_external_invariant():
    cites = resolve_citations(extract_citations(
        "recalling resolutions 1265 (1999) and 9999 (2099),"), MAP)
    for c in cites:
        assert (c.reason is None) == (c.doc_id is not None)


# --------------------------------------------------------------------------- #
# Unmodelled families
# --------------------------------------------------------------------------- #

def test_treaty_article_blocks():
    (c,) = extract_citations("in accordance with Article 19 of the Covenant, states")
    assert c.kind == refs.KIND_TREATY and refs.is_blocking(c)


def test_treaty_span_never_swallows_a_resolution_citation():
    cites = extract_citations(
        "under paragraph 11 of resolution 1244 (1999) and the Charter of the United Nations")
    assert any(c.kind == KIND_SC and c.paragraph == 11 for c in cites)
    assert not any(c.kind == refs.KIND_TREATY for c in cites)


def test_rules_of_procedure_blocks():
    (c,) = extract_citations("under rule 120 of the provisional rules of procedure of")
    assert c.kind == refs.KIND_RULE and refs.is_blocking(c)


def test_general_comment_blocks():
    (c,) = extract_citations("as stated in general comment No. 24 on reservations")
    assert c.kind == refs.KIND_GC and refs.is_blocking(c)


def test_agenda_item_is_recorded_but_never_blocks():
    (c,) = extract_citations("under agenda item 106, the Committee")
    assert c.kind == refs.KIND_AGENDA and not refs.is_blocking(c)


def test_structural_part_is_recorded_but_never_blocks():
    (c,) = extract_citations("as set out in section III above")
    assert c.kind == refs.KIND_STRUCT and not refs.is_blocking(c)


# --------------------------------------------------------------------------- #
# Internal paragraph and annex citations
# --------------------------------------------------------------------------- #

def test_paragraph_of_resolution_is_not_double_counted():
    cites = extract_citations(
        "as required by paragraph 11 (j) of Security Council resolution 1244 (1999)")
    assert [c.kind for c in cites] == [KIND_SC]


def test_paragraph_of_literal_symbol_attaches():
    cites = extract_citations("pursuant to paragraph 6 of document S/2004/650, the")
    (c,) = [c for c in cites if c.kind == KIND_SYMBOL]
    assert c.paragraph == 6
    assert not any(c.kind == refs.KIND_PARA_OTHER for c in cites)


def test_bare_paragraph_is_internal_only_in_resolutions():
    res = extract_citations("as set out in paragraph 5, the Council",
                            doc_id="1999/s/res/1244_1999_")
    (c,) = [c for c in res if c.kind == refs.KIND_INTERNAL_PARA]
    assert c.scope == refs.SCOPE_INTERNAL

    pv = extract_citations("as set out in paragraph 5, the Council",
                           doc_id="1999/s/pv_4011_")
    (c2,) = [c for c in pv if c.kind == refs.KIND_INTERNAL_PARA]
    assert c2.reason == "bare_paragraph_nonres_recorded" and not refs.is_blocking(c2)


def test_paragraph_of_other_report_is_recorded_not_blocking():
    (c,) = extract_citations("as described in paragraph 12 of that report, the")
    assert c.kind == refs.KIND_PARA_OTHER and not refs.is_blocking(c)


def test_annex_to_present_is_internal():
    (c,) = extract_citations("the measures in the annex to the present resolution shall")
    assert c.kind == refs.KIND_INTERNAL_ANNEX and c.scope == refs.SCOPE_INTERNAL
    assert c.annex_label == ""


def test_numbered_annex_with_no_tail_is_internal():
    (c,) = extract_citations("the list set out in annex II shall be updated")
    assert c.kind == refs.KIND_INTERNAL_ANNEX and c.annex_label == "II"


def test_annex_to_resolution_annotates_the_document_citation():
    cites = extract_citations("the plan in the annex to resolution 1244 (1999) shall")
    (c,) = [c for c in cites if c.kind == KIND_SC]
    assert c.annex_label == "" and c.part_kind == "annex"
    assert not any(c.kind == refs.KIND_INTERNAL_ANNEX for c in cites)


def test_annex_to_unnamed_document_is_recorded():
    (c,) = extract_citations("the annex to the report of the Secretary-General shows")
    assert c.reason == "annex_of_other_recorded" and not refs.is_blocking(c)


def test_bare_annex_requires_a_determiner():
    assert extract_citations("decided to annex further material later") == []


def test_own_annex_heading_line_is_not_a_citation():
    assert extract_citations("Annex II\nTECHNICAL SPECIFICATIONS for the network") == []


def test_annexed_hereto_is_internal():
    (c,) = extract_citations("the declaration annexed hereto, which")
    assert c.kind == refs.KIND_INTERNAL_ANNEX and c.scope == refs.SCOPE_INTERNAL


# --------------------------------------------------------------------------- #
# Internal resolution against the document's own structure
# --------------------------------------------------------------------------- #

BLOCK_TEXTS = [
    "PREAMBLE text with no numbering",
    "1. Decides to establish the mission;\n2. Requests the Secretary-General",
    "3. Calls upon all States to cooperate;\n4. Decides to remain seized",
    "Annex I\n1. The mission shall consist of",
    "2. The mission shall report annually",
]
BLOCK_PARTS = ["", "", "", "anx_1", "anx_1"]
ANNEXES = [{"part_id": "anx_1", "kind": "annex", "label": "I",
            "block_start": 3, "block_end": 4}]


def _internal(text, **kw):
    cites = extract_citations(text, doc_id="1999/s/res/1244_1999_")
    refs.resolve_internal(cites, block_index=kw.pop("block_index", 0),
                          block_texts=BLOCK_TEXTS, block_parts=BLOCK_PARTS,
                          annexes=ANNEXES)
    return cites


def test_internal_paragraph_unique_anchor():
    (c,) = _internal("as set out in paragraph 3 above, all States")
    assert c.reason is None and c.anchor_blocks == (2,)


def test_internal_paragraph_ambiguous_across_parts_is_scoped_out():
    # "2." exists in the body (block 1) and in the annex (block 4); a body
    # citation anchors only in the body.
    (c,) = _internal("as set out in paragraph 2 above, the Secretary-General")
    assert c.reason is None and c.anchor_blocks == (1,)
    (c2,) = _internal("pursuant to paragraph 2 above", block_index=4)
    assert c2.anchor_blocks == ()   # its own part: block 4 itself has "2."


def test_internal_paragraph_not_found():
    (c,) = _internal("as set out in paragraph 99 above, the")
    assert c.reason == "internal_paragraph_not_found" and refs.is_blocking(c)


def test_internal_annex_unique():
    (c,) = _internal("set out in the annex to the present resolution")
    assert c.reason is None and c.part_id == "anx_1" and c.anchor_blocks == (3,)


def test_internal_annex_label_mismatch_blocks():
    (c,) = _internal("set out in annex III to the present resolution")
    assert c.reason == "internal_annex_not_found"


def test_paragraph_of_the_annex_anchors_inside_the_part():
    (c,) = _internal("as required by paragraph 2 of the annex, the mission")
    assert c.kind == refs.KIND_INTERNAL_ANNEX
    assert c.reason is None and set(c.anchor_blocks) == {3, 4}


def test_internal_annex_ambiguous_with_duplicate_labels():
    cites = extract_citations("set out in annex I to this resolution",
                              doc_id="1999/s/res/1244_1999_")
    refs.resolve_internal(cites, block_index=0, block_texts=BLOCK_TEXTS,
                          block_parts=BLOCK_PARTS,
                          annexes=ANNEXES + [{"part_id": "anx_1b", "kind": "annex",
                                              "label": "1", "block_start": 5,
                                              "block_end": 5}])
    (c,) = cites
    assert c.reason == "internal_annex_ambiguous"


def test_external_annex_resolution():
    cites = resolve_citations(extract_citations(
        "the plan in annex I to resolution 1244 (1999)"), MAP)
    refs.resolve_external_annexes(
        cites, lambda d: [{"part_id": "anx_1", "kind": "annex", "label": "I",
                           "block_start": 3, "block_end": 4}])
    (c,) = [c for c in cites if c.kind == KIND_SC]
    assert c.part_id == "anx_1" and c.reason is None


def test_external_annex_missing_blocks():
    cites = resolve_citations(extract_citations(
        "the plan in annex IV to resolution 1244 (1999)"), MAP)
    refs.resolve_external_annexes(cites, lambda d: [])
    (c,) = [c for c in cites if c.kind == KIND_SC]
    assert c.reason == "external_annex_not_found" and refs.is_blocking(c)


def test_citation_status_verdicts():
    cites = resolve_citations(extract_citations(
        "recalling resolution 1265 (1999) and Article 19 of the Covenant"), MAP)
    status = refs.citation_status(cites)
    assert status["n_external_resolved"] == 1
    assert status["n_unmodelled_blocking"] == 1
    assert status["complete"] is False

    fine = resolve_citations(extract_citations("recalling resolution 1265 (1999),"), MAP)
    assert refs.citation_status(fine)["complete"] is True

    assert refs.citation_status([])["complete"] is True


def test_comma_form_treaty_citation_is_treaty_not_bare_paragraph():
    cites = extract_citations(
        "acting under article 5, paragraph 4, of the Optional Protocol to the "
        "International Covenant on Civil and Political Rights,",
        doc_id="1999/s/res/1244_1999_")
    assert [c.kind for c in cites if c.scope != refs.SCOPE_UNMODELLED or
            refs.is_blocking(c)] == [refs.KIND_TREATY]
    assert not any(c.kind == refs.KIND_INTERNAL_PARA and c.scope == refs.SCOPE_INTERNAL
                   for c in cites)


def test_comma_form_paragraph_of_resolution_still_attaches():
    cites = extract_citations(
        "as set out in paragraph 2, of its resolution 1244 (1999),")
    (c,) = [c for c in cites if c.kind == KIND_SC]
    assert c.paragraph == 2


def test_annex_failed_external_is_not_counted_resolved():
    cites = resolve_citations(extract_citations(
        "the plan in annex IV to resolution 1244 (1999)"), MAP)
    refs.resolve_external_annexes(cites, lambda d: [])
    status = refs.citation_status(cites)
    assert status["n_external_resolved"] == 0
    assert status["complete"] is False
