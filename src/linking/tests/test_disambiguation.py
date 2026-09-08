"""
Tests for the Tier A disambiguation policy.

These lock in a correctness fix, not a preference. The first version of the
matcher took the first row of a lookup, which silently converted every ambiguity
into a confident answer. Measured on the human gold that made `diptera` — the
single most frequent BIOTIC ENTITY link, 16 mentions — resolve to *Saxifraga*, a
plant genus carrying *Diptera* as an internal name, and `fish` resolve to
*Gadus morhua*.

Four properties are pinned:

  1. Rank preference settles cross-rank homonyms (diptera, bats, bacteria).
  2. A generic mention is never resolved to one particular organism, while a
     specific vernacular still resolves (fish vs annamite striped rabbit — both
     species-rank vernacular matches, so nothing about the CANDIDATE separates
     them; the distinguishing fact is about the mention).
  3. Genuine ambiguity is reported as such and never guessed at.
  4. UNROUTED, NIL and AMBIGUOUS stay distinct: they call for a reranker, a
     bigger vocabulary and nothing respectively, so collapsing them into one
     "unlinked" rate hides all three.

Requires the compiled artifacts; the taxonomic tests skip without
mobiko_gbif.sqlite rather than failing, since it is an optional 3 GB download.

Run with:
    pytest src/linking/tests/test_disambiguation.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.linking.authorities.registry import artifact_path
from src.linking.dictionary import Dictionary
from src.linking.match import Outcome, resolve
from src.linking import wordnet_guard

pytestmark = pytest.mark.skipif(
    not artifact_path("ontologies").exists(),
    reason="compiled dictionary absent; run compile_db")

_HAS_GBIF = artifact_path("gbif").exists()
_HAS_GEONAMES = artifact_path("geonames").exists()
needs_gbif = pytest.mark.skipif(not _HAS_GBIF, reason="mobiko_gbif.sqlite absent")
needs_geonames = pytest.mark.skipif(not _HAS_GEONAMES, reason="mobiko_geonames.sqlite absent")


@pytest.fixture(scope="module")
def d():
    dic = Dictionary.open()
    yield dic
    dic.close()


# ── 1. rank preference ───────────────────────────────────────────

@needs_gbif
@pytest.mark.parametrize("mention,expect_label,expect_rank", [
    # Saxifraga (genus) vs Diptera (order) — the regression that motivated this.
    ("diptera", "Diptera", "order"),
    ("trichoptera", "Trichoptera", "order"),      # vs Diastata (genus)
    ("bats", "Chiroptera", "order"),              # vs Vespertilionidae (family)
    ("bacteria", "Bacteria", "kingdom"),          # vs Bacteria (genus)
    ("beetle", "Coleoptera", "order"),            # vs an arbitrary species
])
def test_rank_preference_picks_the_broader_taxon(d, mention, expect_label, expect_rank):
    r = resolve(d, mention, "BIOTIC ENTITY")
    assert r.outcome is Outcome.LINKED, f"{mention}: {r.reason}"
    assert r.match.label == expect_label
    assert r.match.rank == expect_rank


@needs_gbif
def test_saxifraga_is_never_the_answer_for_diptera(d):
    """The specific regression, pinned by URI."""
    r = resolve(d, "diptera", "BIOTIC ENTITY")
    assert r.uri != "https://www.gbif.org/species/2987833"
    assert r.uri == "https://www.gbif.org/species/811"


# ── 2. genericness guard ─────────────────────────────────────────

@needs_gbif
@pytest.mark.skipif(not wordnet_guard.available(), reason="WordNet corpus absent")
def test_generic_mention_is_not_resolved_to_one_species(d):
    r = resolve(d, "fish", "BIOTIC ENTITY")
    assert r.outcome is Outcome.AMBIGUOUS
    assert "generic" in r.reason


@needs_gbif
@pytest.mark.parametrize("mention,expect", [
    ("annamite striped rabbit", "Nesolagus timminsi"),
    ("pinus kesiya", "Pinus kesiya"),
])
def test_specific_vernacular_still_resolves_to_its_species(d, mention, expect):
    """The guard must not fire here.

    Structurally identical to the `fish` case — single, species-rank, reached by
    a vernacular alias — so only a fact about the MENTION can separate them.
    """
    r = resolve(d, mention, "BIOTIC ENTITY")
    assert r.outcome is Outcome.LINKED, f"{mention}: {r.reason}"
    assert r.match.label == expect


@needs_gbif
@pytest.mark.parametrize("mention,label", [("insects", "Insecta"),
                                           ("mammals", "Mammalia")])
def test_generic_mention_may_link_to_a_generic_taxon(d, mention, label):
    """Genericness alone is not disqualifying — only genericness vs a species."""
    r = resolve(d, mention, "BIOTIC ENTITY")
    assert r.outcome is Outcome.LINKED, f"{mention}: {r.reason}"
    assert r.match.label == label


def test_guard_is_inert_when_wordnet_is_missing(monkeypatch):
    """A missing corpus must degrade to the old behaviour, not reject everything."""
    monkeypatch.setattr(wordnet_guard, "_available", False)
    assert wordnet_guard.blocks_link("fish", "species", "vernacular") is False


# ── 3. ambiguity is reported, not guessed ────────────────────────

@needs_gbif
@pytest.mark.parametrize("mention", ["trout", "dung beetle"])
def test_many_same_rank_candidates_are_ambiguous(d, mention):
    r = resolve(d, mention, "BIOTIC ENTITY")
    assert r.outcome is Outcome.AMBIGUOUS
    assert len(r.candidates) > 1


@needs_geonames
def test_places_sharing_a_name_are_ambiguous(d):
    """There really are several Sierra Nevadas; context is needed, not a guess."""
    r = resolve(d, "sierra nevada", "SPATIAL ENTITY")
    assert r.outcome is Outcome.AMBIGUOUS


@needs_geonames
def test_unambiguous_place_resolves(d):
    """The ID used by MoBiKo_rdf_schema_v1's own skos:exactMatch example.

    Passed as it appears in text, not pre-lowercased: casefolding destroys the
    capitalisation the proper-name rule reads. See resolve()'s contract.
    """
    r = resolve(d, "Carpathian Mountains", "SPATIAL ENTITY")
    assert r.outcome is Outcome.LINKED
    assert r.uri == "https://sws.geonames.org/682674/"


# ── 4. the outcomes stay distinct ────────────────────────────────

def test_every_type_routes_while_the_ontologies_are_universal(d):
    """UNROUTED is unreachable with this registry, and that is the decision.

    ENVO and PATO hold concepts rather than names, so they answer for every
    entity type (decision, 8 Sep, confirming the measurement in
    `registry.Authority`: 51 links gained, and `unrouted` went from 647
    mentions to 0). Nothing is therefore "not attempted" any more — CONCEPT
    reaches the ontologies, finds nothing, and comes back NIL.

    The UNROUTED outcome is kept in the enum and in the coverage report because
    gating the ontologies again would immediately make it reachable, and
    "not attempted" must never be reported as "attempted and missed". This test
    pins which of the two the current registry produces, so that a future
    gating change fails loudly here instead of quietly moving 647 mentions
    between columns.
    """
    assert d.sources_for("CONCEPT")
    assert d.sources_for("A TYPE NOBODY HAS DECLARED")
    assert resolve(d, "sdg", "CONCEPT").outcome is Outcome.NIL


def test_nil_is_not_unrouted(d):
    """Routed and searched, but absent from every vocabulary."""
    r = resolve(d, "species richness", "BIOTIC PROPERTY")
    assert r.outcome is Outcome.NIL


def test_every_non_linked_outcome_explains_itself(d):
    """The unlinked set feeds NIL clustering, which needs to know why."""
    for mention, etype in [("sdg", "CONCEPT"), ("species richness", "BIOTIC PROPERTY"),
                           ("qqzzxx", "ABIOTIC ENTITY")]:
        r = resolve(d, mention, etype)
        assert r.outcome is not Outcome.LINKED
        assert r.reason, f"{mention} gave {r.outcome} with no reason"


@needs_geonames
@needs_gbif
def test_a_gated_authority_answers_only_for_the_types_it_serves(d):
    """Each mention routes by its OWN type, never the union of the types the
    string was given elsewhere. Unioning produced a leak on 135 mentions.

    Probed on the two GATED authorities, because they are where the rule still
    bites: the ontologies are universal by decision (8 Sep), so reaching ENVO
    from any type is correct now and cannot demonstrate a leak. GBIF and
    GeoNames hold millions of proper names that collide with ordinary English,
    which is exactly why they stayed gated.
    """
    # GeoNames is not consulted for a living thing...
    assert resolve(d, "Vietnam", "BIOTIC ENTITY").outcome is not Outcome.LINKED
    # ...nor GBIF for a place.
    assert resolve(d, "diptera", "SPATIAL ENTITY").outcome is not Outcome.LINKED
    # Under the types they do serve, both link.
    assert resolve(d, "Vietnam", "SPATIAL ENTITY").match.source == "geonames"
    assert resolve(d, "diptera", "BIOTIC ENTITY").match.source == "gbif"


def test_same_string_under_two_types_is_two_entities(d):
    """`biodiversity` is BIOTIC PROPERTY 11x and BIOTIC ENTITY 9x in the gold.
    They resolve independently; neither borrows the other's route."""
    a = resolve(d, "biodiversity", "BIOTIC PROPERTY")
    b = resolve(d, "biodiversity", "CONCEPT")
    assert a.entity_type != b.entity_type
    # CONCEPT reaches ENVO and PATO like every other type; neither carries
    # `biodiversity`, so this is a real miss rather than an unrouted type.
    assert b.outcome is Outcome.NIL


# ── 5. the genericness claim itself ──────────────────────────────
#
# An earlier version of this module claimed generic and specific terms occupy
# disjoint hyponym ranges. They do not, and the threshold was set on that
# mistaken belief. These two tests pin what is actually true instead.

@pytest.mark.skipif(not wordnet_guard.available(), reason="WordNet corpus absent")
def test_generic_terms_do_not_have_a_hyponym_floor():
    """Documents the counterexamples, so the wrong claim cannot come back.

    `tetrapods` and `leafhoppers` are plainly generic and score 1. The guard is
    safe on them for a different reason — neither reaches a species-rank
    concept — not because the ranges separate.
    """
    assert wordnet_guard.hyponym_count("tetrapods") < 3
    assert wordnet_guard.hyponym_count("leafhoppers") < 3


@needs_gbif
@pytest.mark.skipif(not wordnet_guard.available(), reason="WordNet corpus absent")
def test_no_species_link_in_the_gold_comes_from_a_hyponym_bearing_mention(d):
    """The property the guard actually relies on.

    Over every BIOTIC ENTITY mention in the human gold that resolves to a
    species or subspecies, the mention's hyponym count must be 0. That is the
    whole population where the guard can fail open, so it is the population
    worth asserting over rather than a hand-picked list.
    """
    import json
    from src.linking.coverage_report import iter_mentions
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "multi_agent_annotation"))
    from label_norm import canonicalize_entity_type

    paths = [Path("data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"),
             Path("data/aug_runs/combined_M_D_Davnah_merged_postprocessed.jsonl")]
    if not all(p.exists() for p in paths):
        pytest.skip("human gold not present")

    checked = 0
    for path in paths:
        for text, raw_type in iter_mentions(path):
            if canonicalize_entity_type(raw_type) != "BIOTIC ENTITY":
                continue
            r = resolve(d, text, "BIOTIC ENTITY")
            if r.outcome is Outcome.LINKED and (r.match.rank or "") in ("species", "subspecies"):
                checked += 1
                assert wordnet_guard.hyponym_count(text) == 0, (
                    f"{text!r} carries hyponyms yet linked to species "
                    f"{r.match.label!r}; the guard would have missed it")
    assert checked > 20, f"expected the gold's species links, found {checked}"


# ── 6. regressions found by auditing, not by the original design ─

@needs_gbif
def test_rank_preference_never_ranks_across_authorities(d):
    """`biota` matches ENVO:06105012 and GBIF's Platycladus (formerly *Biota*).

    Rank is defined only inside GBIF, so an ENVO concept sorts as "unknown".
    Comparing the two made every taxon outrank every ontology concept, which
    silently turned this mention from the correct ENVO sense into a conifer
    genus. Two authorities answering is two kinds of claim with no principled
    tiebreak, so it must be AMBIGUOUS.
    """
    r = resolve(d, "biota", "BIOTIC ENTITY")
    assert r.outcome is Outcome.AMBIGUOUS
    assert "authorities" in r.reason
    assert {m.source for m in r.candidates} == {"envo", "gbif"}


@needs_gbif
def test_a_truncated_candidate_set_is_never_decided_on(d):
    """The lookup SQL has no ORDER BY, so a capped set is an arbitrary subset.

    Applying rank preference to an arbitrary subset is a coin flip, not a
    policy — `beetle` has 65 candidates and an earlier 50-row cap returned the
    right answer only because Coleoptera happened to land inside the window.
    """
    capped = resolve(d, "beetle", "BIOTIC ENTITY", limit=10)
    assert capped.outcome is Outcome.AMBIGUOUS
    assert "cap" in capped.reason
    # With room to enumerate, the same mention resolves.
    full = resolve(d, "beetle", "BIOTIC ENTITY")
    assert full.outcome is Outcome.LINKED
    assert full.match.label == "Coleoptera"


def test_resolution_is_deterministic(d):
    """Same input, same artifacts, same answer — the property that makes a
    collaborator's run comparable to ours."""
    for mention, etype in [("beetle", "BIOTIC ENTITY"), ("biota", "BIOTIC ENTITY"),
                           ("sierra nevada", "SPATIAL ENTITY"), ("lake", "ABIOTIC ENTITY")]:
        runs = [resolve(d, mention, etype) for _ in range(5)]
        assert len({(r.outcome, r.uri) for r in runs}) == 1, f"{mention} varied"


def test_coverage_report_is_byte_reproducible():
    """Same input, same artifacts, byte-identical output — across PROCESSES.

    The in-process determinism test above passed while the report still varied,
    because Python randomises str hashing per process: sorting a set of entity
    types by count alone left the five one-mention types to reshuffle. Every
    number was stable; only the row order moved. This runs the real CLI twice
    so that class of bug cannot hide again.
    """
    import subprocess
    if not artifact_path("ontologies").exists():
        pytest.skip("dictionary absent")
    cmd = [sys.executable, "-m", "src.linking.coverage_report", "--top", "5"]
    root = Path(__file__).resolve().parents[3]
    outs = []
    for _ in range(2):
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        outs.append(r.stdout)
    assert outs[0] == outs[1]


# ── 7. common nouns must not bind to named individuals ───────────

@needs_geonames
@pytest.mark.parametrize("mention", ["landscape", "mountain range", "center"])
def test_common_noun_does_not_resolve_to_a_named_place(d, mention):
    """"the landscape" denotes something, but not the settlement called
    Landscape. None of these may be bound to a place by spelling alone."""
    assert resolve(d, mention, "SPATIAL ENTITY").outcome is Outcome.AMBIGUOUS


@needs_geonames
@pytest.mark.parametrize("mention", ["landscape", "center"])
def test_a_common_noun_is_declined_as_a_coreference_question(d, mention):
    """Not merely unresolvable — UNDER-SPECIFIED. "the landscape" denotes a
    landscape named earlier in the text, which is a coreference question this
    module cannot answer, so the reason says so.

    `mountain range` is excluded: it is still declined, but now by the
    cross-authority rule, because since the ontologies became universal ENVO
    answers for SPATIAL ENTITY alongside GeoNames. Same refusal, earlier gate.
    """
    r = resolve(d, mention, "SPATIAL ENTITY")
    assert r.outcome is Outcome.AMBIGUOUS
    assert "coreference" in r.reason


@needs_geonames
@pytest.mark.parametrize("mention,uri", [
    ("southern Africa", "https://sws.geonames.org/9406051/"),
    ("Carpathian Mountains", "https://sws.geonames.org/682674/"),
])
def test_a_capitalised_token_anywhere_is_enough(d, mention, uri):
    """`southern Africa` passes on *Africa*, so the test is "any token",
    not "the first token"."""
    r = resolve(d, mention, "SPATIAL ENTITY")
    assert r.outcome is Outcome.LINKED
    assert r.uri == uri


@needs_geonames
def test_scope_preference_is_not_applied_between_places(d):
    """`lake` returns a dozen "Lake County" rows plus one obscure feature whose
    official name is exactly "Lake". Preferring label-over-synonym picked the
    obscure one — a tiebreak that means something for a term and its synonym and
    nothing for two different places."""
    r = resolve(d, "lake", "SPATIAL ENTITY")
    assert r.outcome is Outcome.AMBIGUOUS
    assert len(r.candidates) > 5


# ── candidate generation ─────────────────────────────────────────
#
# `Dictionary.search` had no caller and had never returned a row: it built an
# implicit-AND FTS5 query, so any mention carrying one token the index lacks
# came back empty — which is every mention a candidate generator exists for.

def test_the_search_query_is_ored_not_anded(d):
    """AND is the bug. One absent token must not empty the whole result."""
    q = d._fts_query("Salvelina fontinalis")
    assert " OR " in q and '" "' not in q


@pytest.mark.parametrize("text", ["3,4-dichlorophenol", "species' range", 'a "q" b'])
def test_punctuation_in_a_mention_is_not_an_fts_syntax_error(d, text):
    """`3,4-dichlorophenol` and `species'` are legitimate mentions, and both are
    a syntax error unquoted. Interior punctuation is load-bearing elsewhere in
    this module, so it reaches here intact and has to survive."""
    d.search(text, ["envo", "pato"], limit=5)          # must not raise


@needs_gbif
def test_a_misspelling_reaches_its_concept_as_a_candidate(d):
    """The case Phase 7 asks for. Before the fix this returned zero rows while
    *Salvelinus fontinalis* sat in the index."""
    cands = d.search("Salvelina fontinalis", ["gbif"], limit=300)
    assert cands
    assert any(m.label == "Salvelinus fontinalis" for m in cands)


@needs_geonames
def test_a_prefix_reaches_a_shorter_variant(d):
    """`annamites` must find *Annamite Mountains*, i.e. the prefix has to drop
    the mention's own tail — `"annamites"*` matches only terms starting with
    all of it, and the index holds the shorter `annamite`."""
    assert any(m.label == "Annamite Mountains"
               for m in d.search("Annamites", ["geonames"], limit=50))


@pytest.mark.parametrize("mention", ["mountains", "Mountains", "mountain"])
def test_mountains_is_not_an_undersea_mountain(d, mention):
    """ENVO ships this pair with their scopes swapped: `mountains` is an EXACT
    synonym of *seamount* and only a RELATED synonym of *mountain*. Related is
    not an identity scope, so the correct concept was excluded and the undersea
    one won all 14 mentions of it in a corpus about mountain biodiversity.

    Corrected in the artifact at build time
    (`compile_db.OBO_SCOPE_CORRECTIONS`), because no matching rule can tell a
    mis-scoped synonym from a correct one — they are string-identical. This
    test fails if the artifact predates the correction.
    """
    assert resolve(d, mention, "ABIOTIC ENTITY").match.curie == "ENVO:00000081"


def test_the_scope_correction_does_not_disturb_seamount(d):
    """Demoting the bad alias must not cost *seamount* its own name."""
    assert resolve(d, "seamount", "ABIOTIC ENTITY").match.curie == "ENVO:00000264"


def test_proper_name_rule_does_not_apply_to_class_naming_sources(d):
    """Ontology and taxon concepts name CLASSES, so a lowercase common noun is
    exactly what should match them."""
    r = resolve(d, "lake", "ABIOTIC ENTITY")
    assert r.outcome is Outcome.LINKED
    assert r.match.curie == "ENVO:00000020"
