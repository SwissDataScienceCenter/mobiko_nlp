"""
Tests for NIL clustering and the resolution table.

The properties pinned here are the ones the deliverable's meaning rests on:

  1. Clustering merges LEXICAL variants only. Modifier-preserving granularity
     was chosen because over-merging is unrecoverable and under-merging is not,
     so `bird species`, `bat species` and `species` must stay three entities.
  2. Entity ids are content-derived, hence stable across runs and corpora. A
     sequential counter would renumber everything when a document is added and
     silently break the relation join that attaches later by `uid`.
  3. Two mentions denote one entity IFF they share an entity_id — the table's
     entire contract.
  4. Ambiguous mentions get no id. Minting one would assert that all eight uses
     of "the lake" are the same lake, which is the guess the matcher refused.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.linking.authorities.registry import artifact_path
from src.linking.cluster import (build_clusters, content_key, entity_id,
                                 rank_by_document_frequency)

ROOT = Path(__file__).resolve().parents[3]


# ── 1. lexical variants merge, modifiers do not ──────────────────

@pytest.mark.parametrize("a,b", [
    ("species", "Species"),                                   # case
    ("populations", "population"),                            # plural
    ("Annamites", "Annamite"),                                # plural
    ("Annamites:", "Annamites"),                              # trailing punctuation
    ("Bidoup—Nui Ba National Park", "Bidoup Nui Ba National Park"),   # em dash
    ("broadleaf evergreen forest", "evergreen broadleaf forests"),    # word order
    ("Doñana", "Donana"),                                     # diacritics
])
def test_lexical_variants_share_a_key(a, b):
    assert content_key(a, "BIOTIC ENTITY") == content_key(b, "BIOTIC ENTITY")


@pytest.mark.parametrize("a,b", [
    ("bird species", "bat species"),
    ("species", "bird species"),
    ("temperature", "high temperatures"),
    ("temperature", "peak water temperature"),
    ("forest", "coniferous forest"),
])
def test_a_modifier_difference_is_a_different_entity(a, b):
    """Modifier-preserving granularity, which is the whole safety argument."""
    assert content_key(a, "BIOTIC ENTITY") != content_key(b, "BIOTIC ENTITY")


def test_every_label_gets_its_own_node():
    """Identity is the full label, role and domain together (decision, 7 Sep).

    `habitat` under three labels is three nodes; what they share is stated as a
    common `authority_uri` rather than by merging them, so both directions stay
    recoverable. An earlier key used the role alone and merged them, which was
    not recoverable afterwards.
    """
    assert content_key("biodiversity", "BIOTIC PROPERTY") != \
           content_key("biodiversity", "BIOTIC ENTITY")
    keys = {content_key("habitat", t) for t in
            ("ABIOTIC ENTITY", "BIOTIC ENTITY", "SPATIAL ENTITY")}
    assert len(keys) == 3


def test_surface_key_groups_a_phrase_across_its_labels():
    """The handle for entities no authority covers: `authority_uri` groups the
    grounded ones, `surface_key` groups the rest — without merging either."""
    from src.linking.cluster import surface_key
    assert (surface_key("habitat") == surface_key("habitats")
            == surface_key("the habitat"))


def test_identity_joins_on_either_words_or_authority():
    """Union of two signatures, because each alone loses cases the other keeps.

    Both signatures carry the FULL label. Joining on the role alone would pull
    the three habitat nodes back together through their shared ENVO id.
    """
    from src.linking.cluster import link_signatures
    k_m1 = ("key", content_key("mountains", "ABIOTIC ENTITY"))
    k_m2 = ("key", content_key("mountain", "ABIOTIC ENTITY"))
    envo = ("uri", "ABIOTIC ENTITY", "ENVO_00000081")
    k_vn = ("key", content_key("Vietnam", "SPATIAL ENTITY"))
    k_vn2 = ("key", content_key("Viet Nam", "SPATIAL ENTITY"))
    geo = ("uri", "SPATIAL ENTITY", "geonames:1562822")

    comp = link_signatures([(k_m1, envo), (k_m2, None), (k_vn, geo), (k_vn2, geo)])
    # same words and label, only one grounded -> still one entity
    assert comp[k_m1] == comp[k_m2]
    # different words, same label and authority -> one entity
    assert comp[k_vn] == comp[k_vn2]
    assert comp[k_m1] != comp[k_vn]


def test_a_shared_authority_does_not_merge_across_labels():
    """The meeting's decision, as a property: three habitat nodes, one ENVO id."""
    from src.linking.cluster import link_signatures
    sigs = []
    for label in ("ABIOTIC ENTITY", "BIOTIC ENTITY", "SPATIAL ENTITY"):
        sigs.append((("key", content_key("habitat", label)),
                     ("uri", label, "ENVO_01000739")))
    comp = link_signatures(sigs)
    assert len({comp[s[0]] for s in sigs}) == 3


@pytest.mark.parametrize("a,b", [
    ("gradient of temperature", "temperature gradient"),
    ("rates of diversification", "diversification rates"),
    ("the number of", "number"),
    ("to conserve", "conserve"),
    ("length from head to abdomen", "head-to-abdomen length"),
    ("small patches of", "small patches"),
])
def test_function_words_do_not_split_an_entity(a, b):
    """The key is an unordered multiset, so a preposition or an infinitive
    marker carries nothing that survives into it. All 15 merges this produces
    on the gold were hand-checked and every one is a genuine variant."""
    assert content_key(a, "BIOTIC PROPERTY") == content_key(b, "BIOTIC PROPERTY")


def test_a_phrase_of_only_function_words_keeps_them():
    """Otherwise every such span collapses onto one empty key."""
    assert content_key("of the", "CONCEPT")[1]
    assert content_key("of the", "CONCEPT") != content_key("in on", "CONCEPT")


def test_determiners_are_stripped_but_quantifiers_are_not():
    """"the lake" is "lake"; "all species" is not "species" — the schema types
    quantifiers as QUANTITATIVE PROPERTY in their own right."""
    assert content_key("the lake", "ABIOTIC ENTITY") == content_key("lake", "ABIOTIC ENTITY")
    assert content_key("all species", "BIOTIC ENTITY") != content_key("species", "BIOTIC ENTITY")


# ── 2. ids are stable ────────────────────────────────────────────

def test_entity_id_is_content_derived_and_stable():
    k = content_key("species richness", "BIOTIC PROPERTY")
    assert entity_id(k) == entity_id(k)
    assert entity_id(k).startswith("mobiko:e")
    # Independent of corpus composition: same phrase, same id, different corpus.
    assert entity_id(content_key("Species Richness", "BIOTIC PROPERTY")) == entity_id(k)


def test_cluster_label_is_the_commonest_form_deterministically():
    c = build_clusters([("u1", "populations", "BIOTIC ENTITY", "d1"),
                        ("u2", "populations", "BIOTIC ENTITY", "d1"),
                        ("u3", "population", "BIOTIC ENTITY", "d2")])
    (only,) = c.values()
    assert only.label == "populations"
    assert only.n_mentions == 3
    assert only.doc_frequency == 2


def test_gap_ranking_uses_documents_not_mentions():
    """A phrase used many times in one abstract is that author's habit; one used
    once in many abstracts is domain vocabulary the ontologies lack."""
    c = build_clusters(
        [("a%d" % i, "lakes", "ABIOTIC ENTITY", "d1") for i in range(20)] +
        [("b%d" % i, "endemic", "BIOTIC PROPERTY", "d%d" % i) for i in range(5)])
    ranked = rank_by_document_frequency(c)
    assert ranked[0].label == "endemic"      # 5 docs beats 20 mentions in 1


# ── 3-4. the table's contract ────────────────────────────────────

@pytest.mark.skipif(not artifact_path("ontologies").exists(),
                    reason="compiled dictionary absent")
def test_resolution_table_contract(tmp_path):
    inputs = [ROOT / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"]
    if not inputs[0].exists():
        pytest.skip("human gold not present")
    out = tmp_path / "resolution.jsonl"
    r = subprocess.run(
        [sys.executable, "-m", "src.linking.resolve_corpus",
         "--input", str(inputs[0]), "--out", str(out), "--no-split"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]

    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert rows

    for row in rows:
        assert row["uid"], "every mention must be addressable by uid"
        # Two kinds of row carry no entity, both deliberately: an AMBIGUOUS
        # mention, and the container row of a coordinated span whose parts hold
        # the entities ("eastern and western slopes" is not one thing).
        declines = row["outcome"] == "ambiguous" or row.get("split_into")
        if declines:
            assert row["entity_id"] is None
        else:
            assert row["entity_id"], f"{row['outcome']} row carries no entity_id"

    # Identity is the ontological ROLE, not the full label: one entity may carry
    # several domain labels (habitat is ABIOTIC/BIOTIC/SPATIAL ENTITY) but never
    # two roles.
    # Identity is the full label, so an entity carries exactly one.
    by_types = {}
    for row in rows:
        if row["entity_id"]:
            by_types.setdefault(row["entity_id"], set()).update(row["entity_types"] or [])
    assert all(len(v) == 1 for v in by_types.values()), "an entity spans two labels"

    # …and separated labels of one concept are joined by a shared authority_uri.
    by_uri = {}
    for row in rows:
        if row["authority_uri"]:
            by_uri.setdefault(row["authority_uri"], set()).add(row["entity_id"])
    assert any(len(v) > 1 for v in by_uri.values()), \
        "no ontological entity is shared across labels; the link is missing"

    # And the provenance sidecar names the vocabulary that produced it.
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["normalizer_version"]
    assert meta["dictionary"]


@pytest.mark.skipif(not artifact_path("ontologies").exists(),
                    reason="compiled dictionary absent")
def test_documents_are_split_at_abstract_boundaries(tmp_path):
    """The input concatenates 15 abstracts under one doc_id, so taking that
    field at face value makes document frequency identically 1."""
    src = ROOT / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"
    if not src.exists():
        pytest.skip("human gold not present")
    out = tmp_path / "r.jsonl"
    subprocess.run([sys.executable, "-m", "src.linking.resolve_corpus",
                    "--input", str(src), "--out", str(out), "--no-split"],
                   cwd=ROOT, capture_output=True, text=True, check=True)
    docs = {json.loads(l)["doc_id"] for l in out.open(encoding="utf-8")}
    assert len(docs) > 10, f"expected one doc per abstract, got {docs}"


@pytest.mark.skipif(not artifact_path("ontologies").exists(),
                    reason="compiled dictionary absent")
def test_gap_report_lists_the_entities_that_stayed_unresolved(tmp_path):
    """The gap report is the artefact a domain expert reads, so it has to be
    written at all — it had been raising after the header line, leaving a
    60-byte file that still looked plausible — and it must not claim an entity
    is missing from the ontologies when grounding actually reached it."""
    src = ROOT / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"
    if not src.exists():
        pytest.skip("human gold not present")
    out = tmp_path / "r.jsonl"
    gap = tmp_path / "gap.tsv"
    subprocess.run([sys.executable, "-m", "src.linking.resolve_corpus",
                    "--input", str(src), "--out", str(out), "--no-split",
                    "--gap-report", str(gap)],
                   cwd=ROOT, capture_output=True, text=True, check=True)

    lines = gap.read_text(encoding="utf-8").splitlines()
    header, body = lines[0].split("\t"), [l.split("\t") for l in lines[1:]]
    assert body, "the gap report is header-only"
    col = {name: i for i, name in enumerate(header)}
    assert all(row[col["entity_type"]] for row in body), \
        "entity_type column is empty"

    # Every listed entity must be a project-local entity of the same run, i.e.
    # one no authority covered. A chained-into-grounded key must not appear.
    local = {r["entity_id"] for r in map(json.loads, out.open(encoding="utf-8"))
             if r["entity_id"] and not r["authority_uri"]}
    listed = {row[col["entity_id"]] for row in body}
    assert listed <= local, \
        f"{len(listed - local)} listed entities are not unresolved in the table"

    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert len(listed) == meta["n_entities_project_local"]
    assert meta["n_entities_from_authorities"] + meta["n_entities_project_local"] \
        == meta["n_entities"]


@pytest.mark.skipif(not artifact_path("ontologies").exists(),
                    reason="compiled dictionary absent")
def test_the_two_entity_reports_partition_the_entities(tmp_path):
    """--linked-report and --gap-report are a pair: what an authority caught
    and what it missed. Between them they must account for every entity
    exactly once, or one of the two is quietly lying about coverage."""
    src = ROOT / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"
    if not src.exists():
        pytest.skip("human gold not present")
    out, gap, linked = (tmp_path / "r.jsonl", tmp_path / "gap.tsv",
                        tmp_path / "linked.tsv")
    subprocess.run([sys.executable, "-m", "src.linking.resolve_corpus",
                    "--input", str(src), "--out", str(out), "--no-split",
                    "--gap-report", str(gap), "--linked-report", str(linked)],
                   cwd=ROOT, capture_output=True, text=True, check=True)

    def ids(path, col):
        lines = path.read_text(encoding="utf-8").splitlines()
        i = lines[0].split("\t").index(col)
        return {l.split("\t")[i] for l in lines[1:]}

    gap_ids, linked_ids = ids(gap, "entity_id"), ids(linked, "entity_id")
    assert linked_ids, "no grounded entities reported"
    assert not (gap_ids & linked_ids), "an entity is in both reports"

    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert linked_ids == {r["entity_id"] for r in rows if r["authority_uri"]}
    assert gap_ids | linked_ids == {r["entity_id"] for r in rows if r["entity_id"]}

    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert len(linked_ids) == meta["n_entities_from_authorities"]


# ── the singular fallback ────────────────────────────────────────

from src.linking.cluster import singular_form  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("glaciations", "glaciation"),
    ("oligotrophic lakes", "oligotrophic lake"),
    ("Annamites:", "Annamite"),          # edge punctuation trimmed off the head
    ("Lakes", "Lake"),                   # capitalisation SURVIVES — see below
    ("glaciation", None),                # already singular
    ("", None),
])
def test_singular_form(text, expected):
    assert singular_form(text) == expected


def test_the_singular_keeps_its_capital_so_it_can_still_be_refused():
    """Folding case here would resurrect the failure morphology was demoted for.

    `landscapes` must reach the matcher as lowercase `landscape`, so the
    proper-name rule still refuses it against the settlement named Landscape;
    upper-casing it would sail straight past that guard.
    """
    assert singular_form("landscapes") == "landscape"
    assert singular_form("Forests") == "Forest"


@pytest.mark.skipif(not artifact_path("ontologies").exists(),
                    reason="compiled dictionary absent")
def test_the_singular_fallback_only_ever_adds_grounding(tmp_path):
    """It runs only on mentions that matched nothing, and takes only a LINK, so
    it can ground a mention but must never move one to a different entity.

    The same shape as `--no-split`: turning the feature off yields a strict
    weakening of the same table, not a different one.
    """
    src = ROOT / "data/aug_runs/combined_M_D_Mark_postprocessed.jsonl"
    if not src.exists():
        pytest.skip("human gold not present")

    def run(*extra):
        out = tmp_path / ("r" + "".join(extra).replace("-", "") + ".jsonl")
        subprocess.run([sys.executable, "-m", "src.linking.resolve_corpus",
                        "--input", str(src), "--out", str(out), "--no-split", *extra],
                       cwd=ROOT, capture_output=True, text=True, check=True)
        return {r["mention_id"]: r
                for r in map(json.loads, out.open(encoding="utf-8"))}

    off, on = run("--no-singular"), run()
    assert off.keys() == on.keys(), "the fallback must not add or drop rows"

    moved = [k for k, n in on.items()
             if off[k]["entity_id"] != n["entity_id"] and off[k]["authority_uri"]]
    assert not moved, f"{len(moved)} mentions were moved to a different entity"

    grounded_off = {k for k, r in off.items() if r["authority_uri"]}
    grounded_on = {k for k, r in on.items() if r["authority_uri"]}
    assert grounded_off < grounded_on, "the fallback grounded nothing new"
    # Whatever it grounds, it grounds through the recorded canonical form.
    for k in grounded_on - grounded_off:
        if on[k]["concept_source"] == "singular":
            assert on[k]["canonical_text"] != on[k]["text"]


# ── concept_text is optional input, validated not trusted ────────

from src.linking.resolve_corpus import validate_concept_text  # noqa: E402


def test_concept_text_accepted_when_the_sentence_supports_it():
    """The distributed-modifier case: `high` scopes over a conjunction, so the
    concept is not contiguous in the sentence but every token is present. A
    substring test would wrongly reject this."""
    sent = ("We find that centres of species richness correlate with areas of "
            "high temperatures, annual rainfall and topographic relief.")
    assert validate_concept_text("high annual rainfall", sent) == "sentence"
    assert validate_concept_text("high topographic relief", sent) == "sentence"


def test_tokens_are_trimmed_of_punctuation():
    """A sentence ending "...topographic relief." must still support the token
    `relief`. Trimming only the whole string leaves `relief.` and rejects it —
    the bug that failed most valid values on the first attempt."""
    assert validate_concept_text("topographic relief",
                                 "…and topographic relief.") == "sentence"


def test_document_scope_rescues_abbreviated_binomials():
    """The most valuable corrections reach past the sentence: annotators expand
    a bare epithet into a binomial using a genus named earlier in the abstract,
    and a full binomial is what GBIF can match."""
    sent = "We also recorded M. frater at higher elevations."
    doc = "Bats included Myotis muricola and Myotis frater. " + sent
    assert validate_concept_text("Myotis frater", sent) is None
    assert validate_concept_text("Myotis frater", sent, doc) == "document"


@pytest.mark.parametrize("concept", ["x three", "x phylogenetic", "Helarctos Malayans"])
def test_junk_and_corruption_are_rejected_at_both_scopes(concept):
    """190 of 195 rejections in the gold are the tool placeholder "x" prefixed
    to the span; the rest are genuine corruptions."""
    sent = "Three phylogenetic groups of Helarctos malayanus were recorded."
    assert validate_concept_text(concept, sent, sent) is None
