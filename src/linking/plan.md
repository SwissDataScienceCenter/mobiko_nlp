# Entity de-duplication and ontology canonicalisation

## Context

The pipeline emits triplets. Before a graph can be built from them, a thing
mentioned in twenty sentences has to become one entity, and that entity should
carry an external ID wherever an authority has one.

**Scope: entities only.** Relations are out — their mapping is handled later.
Relations stay untouched in the input files.

The existing `src/linking/`, `src/discovery/` and the `link` / `nil` Makefile
targets are dead code — never used. This is a fresh build.

Live and worth building on: `label_norm.py` (single source of truth for label
normalisation), `dependency_relations.py` (spaCy loader, span→head mapping,
coordination detection), `add_coref_annotations.py` (52 within-document coref
links in the current gold), the span `uid` field, and the provenance machinery
in `multi_agent_annotation_ag2.py` (`_git_commit`, `_file_sha256`,
`_build_run_meta`).

## The deliverable

**A mention → canonical-entity resolution table, keyed by span `uid`.**

Every mention keeps its own node with its `uid` and provenance; a merge is an
edge to a canonical entity, never a deletion, so a bad merge stays reversible.
Relations are not touched and not rewritten — when their mapping is built later
it is a join on `uid`, with no rework on either side.

## What the data says

Over the human gold (`data/aug_runs/combined_M_D_*_postprocessed.jsonl`) —
3,269 mentions, 1,692 unique lowercased forms, 154 sentences:

- About ten Latin binomials. Under 1% of unique forms. One, *Salvelina
  fontinalis*, is misspelled.
- The mass is generic: `species`(65), `lakes`(36), `biodiversity`(25),
  `species richness`(19), `higher`(18), `presence`(17), `diversity`(15),
  `endemic`(15), `mountains`(14), `temperature`(14).
- The best-linkable class present is places: *Bidoup Nui Ba National Park*,
  *Langbian Plateau*, *Sierra Nevada Massif*, *Baetic-Rifian Range*.

So "wolf = *Canis lupus*" is real but rare, and GBIF alone will look like a
failure. The mass needs ENVO and PATO.

## Three rules the build follows

**1. Ground first, cluster last.** Two mentions resolving to the same external
ID are merged by that ID — no threshold, no embedding, no error. Clustering only
touches what failed to ground.

**2. Retrieval finds candidates; it never decides.** Cosine similarity does not
separate synonymy from antonymy (`eastern slopes` / `western slopes` — both
SPATIAL ENTITY in the kimi run), co-hyponymy (`coniferous forest`(13) /
`broadleaf evergreen forest`(10)) or hypernymy (`lake` / `Sierran lakes`). A
similarity threshold merges all of them.

**3. No component of the required path needs an API key or a GPU.** The code
runs on collaborators' machines. Neural models and LLMs are optional tiers that
add decisions; they never change the shape of the output.

## Decisions taken

- All linkable types get grounded, ENVO and PATO included.
- Build now; pin authority versions and stamp provenance from day one so an
  evaluation can be added later without re-running anything.
- Node granularity is **modifier-preserving**: `high temperatures` and
  `temperature` stay separate entities joined by a broader/narrower link.
  Rationale is asymmetry of risk — over-merging is unrecoverable, under-merging
  is not.
- `concept_text` is **strictly optional input**, assumed present on some spans
  and absent on most.

## Architecture: three tiers

Every decision is stamped `decided_by`, so a Tier A run produces a strict subset
of a Tier C run — the same output with fewer decisions made, never a different
one. The tier boundary doubles as a free ablation later.

| Tier | Needs | Does |
|---|---|---|
| **A — required** | `sqlite3` (stdlib), `rapidfuzz`, `nltk`+wordnet, `spacy`+`en_core_web_sm` | normalise, unroll spans, type-route, exact/alias match, fuzzy match, WordNet veto, block-and-cluster residue |
| **B — optional** | `sentence-transformers`, `en_core_web_trf` (torch already present) | candidate generation over Tier A's NIL set; better parses for unrolling |
| **C — optional** | API key or GPU | LLM adjudication of the hard residue, verified by lookup |

Exact and alias matching against a good dictionary is where the volume is, so
Tier A is expected to carry most decisions on its own.

Two more signals that need no LLM:

- **Cross-run agreement.** 5 models × 3 replicates already type each mention
  several times independently. Free merge-confidence from calls already paid
  for; reuses paper 1's machinery.
- **Corpus frequency and document spread**, which distinguish a real concept
  from a one-off phrasing without any model at all.

## Engineering shape

- **SQLite + FTS5 is the whole retrieval layer.** Verified here (sqlite 3.45.1,
  `bm25()` available). FTS5 ships BM25; `sqlite3` is stdlib. Removes faiss, a
  separate BM25 library, and any server from the required path.
- **Build-time and run-time are separate.** Raw GBIF backbone and GeoNames are
  each well over a gigabyte. One machine, online, compiles everything into a
  single versioned dictionary file; collaborators download that one artifact and
  run offline. ENVO (~6.5k terms) and PATO (~2.5k) ship whole.
- **One command per stage, JSONL between stages** — matches the existing
  `scripts/xrun/` style, so each stage is independently re-runnable and
  inspectable.
- **Determinism is an assertable test.** Same input + same dictionary version →
  byte-identical output. This is what makes a collaborator's run comparable.
- Extend the existing pytest suite under `src/multi_agent_annotation/tests/`.

## Span unrolling (dependency parser)

A first-class step, not a detail. Four jobs:

1. **Distribute modifiers across conjuncts.** In *"areas of high temperatures,
   annual rainfall and topographic relief"*, `high` scopes over the conjuncts,
   so the span `annual rainfall` unrolls to `high annual rainfall`.
2. **Split coordinated spans.** `eastern and western slopes` → two concepts,
   `eastern slopes` and `western slopes`. One span, two entities.
3. **Recover elided heads.** *"at least 46 ground-dwelling mammal and bird
   species"* → `mammal species` + `bird species`. This case is in the label
   guidance itself (§4.3).
4. **Resolve `X of Y`.** `centres of species richness` — head is `centres`, the
   PP qualifies it. Determines the blocking key.

Build it on `dependency_relations.py`: reuse `_head_token()` (which handles
spans both with and without char offsets — agent entities have none), the cached
`_parse()`, and the coordination-edge set. Factor the shared loader out rather
than standing up a second one; `label_norm.py`'s docstring is explicit about
what three copies of one normaliser cost.

**Record whether it actually ran.** `DependencyRelationHinter` defaults to
`en_core_web_trf` with an `en_core_web_sm` fallback and degrades silently when
neither is present — and neither is installed today, so dependency hints have
been quietly disabled in every run so far. The unroller stamps
`unroll_available` and the loaded model name into `run_meta`, following the same
effective-vs-requested convention already used for `hints_effective`.

Validate against Mark's 225 `concept_text` pairs, held out from the input path
so the evaluation is not circular.

## `concept_text` — optional input

Assume it will be present on some spans and absent on most. Today it appears on
no agent output at all (497 entities carry only `text, entity_type,
guideline_rule, confidence, reasoning`) and on neither postprocessed human file;
it lives only in `Mark_relation_input.json` / `Davnah_relation_input.json`.

**Precedence for the normalisation target**, recorded per mention as
`concept_source`:

1. validated `concept_text`, when present
2. parser-derived unrolling
3. the surface text

**Validate, do not trust.** The existing human values are inconsistent — Mark
uses it 225 times to state the intended concept, Davnah 35 times mostly to
repair off-by-one span boundaries (`ichness` → `richness`, `tochastic events` →
`stochastic events`) and to reduce qualitative spans to a bare adjective
(`novel habitats` → `novel`). One entry corrupts its span outright
(`Helarctos malayanus` → `Helarctos Malayans`).

The check: **every token of `concept_text` must appear in the sentence.** A
verbatim-substring test would wrongly reject the valid cases — `high annual
rainfall` is not contiguous in its sentence — while token containment accepts
those and catches `Helarctos Malayans`. Failures are flagged and counted, never
silently used or silently dropped.

## WordNet

Useful for a narrow job. Its biodiversity coverage is thin and it was frozen in
2006 — `species richness`, `defaunation` and `permafrost thaw` are absent. So:

**Negative evidence and morphology only. Never a positive link. Absence from
WordNet is never evidence of anything.**

- **Must-not-link oracle**, deterministic and offline: antonyms (`eastern` /
  `western`), co-hyponyms (`bird` / `mammal` under `vertebrate`), hypernym
  chains (`wolf` / `canid` is hierarchy, not identity), siblings (`lake` /
  `pond`). Every one is a pair embedding similarity merges happily.
- **Morphy** for irregular plurals a naive `-s` rule breaks on: `taxa` →
  `taxon`, `fungi` → `fungus`, `larvae` → `larva`, `genera` → `genus`, `algae` →
  `alga`. The data contains `taxa`.

`nltk` is installed; the wordnet data is not downloaded yet (~10MB).

## Where the code goes

`src/linking/` — replacing the dead `build_faiss.py` and `link_mentions.py`
there. (`src/discovery/aggregate_nil.py` is dead too; NIL ranking moves into
`cluster.py` and that directory can be removed separately.)

```
src/linking/
  spacy_util.py      shared loader, _parse, _head_token — factored out of
                     dependency_relations.py so there is one copy, not two
  mention_norm.py    normalisation + Morphy
  unroll.py          dependency-parser span unrolling
  wordnet_veto.py    must-not-link oracle
  route.py           entity type -> authority routing
  match.py           exact / alias / fuzzy  (Tier A)
  cluster.py         NIL clustering + document-frequency ranking
  resolve.py         orchestrator; emits the resolution table
  cli.py             one command per stage
  authorities/
    registry.py      snapshot registry: versions, sha256, provenance
    fetch.py         download (build machine only)
    compile_db.py    -> the SQLite + FTS5 artifact
```

**Environment: `.mobiko-venv`**, not `.venv`. It is the pipeline environment —
it carries autogen and every permission entry in `.claude/settings.local.json`
points at it. FTS5 confirmed working there.

## TODO

### Phase 0 — Environment

1. `.mobiko-venv`: install `rapidfuzz` and `pronto`; `python -m spacy download
   en_core_web_sm`; `python -m nltk.downloader wordnet omw-1.4`. Already
   present there: `spacy`, `nltk`, `torch`, `sentence-transformers`, `pandas`,
   `requests`. Neither venv currently has any spaCy model or the WordNet data.
2. Add `rapidfuzz` to `requirements.txt` and record the required-vs-optional
   dependency split in the README, so a collaborator can run Tier A without
   torch.

### Phase 1 — Authorities and the dictionary artifact

Build the registry/fetch/compile machinery against the **small ontologies
first** — ENVO, PATO and GO are tens of megabytes and prove the whole path end
to end in minutes. GBIF backbone and GeoNames are each well over a gigabyte;
pull them only once the compile path is known to work, and confirm before
starting a download that size.

3. `registry.py`: authority records with source URL, version, sha256, licence,
   and local snapshot path. No live API calls anywhere in the pipeline —
   numbers must not move when GBIF updates.
4. `fetch.py` + `compile_db.py`, then fetch and compile the small ontologies:
   ENVO (`envo.obo`), PATO (`pato.obo`), GO (`go-basic.obo`, filtered to
   `biological_process`). `pronto` reads all three.
5. Verify the compile path on those: exact lookup, alias/synonym lookup, and an
   FTS5 BM25 query all return sane rows.
6. Then the large ones: GBIF backbone (`Taxon.tsv` + `VernacularName.tsv`) and
   GeoNames, filtered to keep the artifact downloadable. Skip NCBITaxon for
   lookup — the RDF schema uses it as a class and GBIF covers the biodiversity
   names. GMBA Mountain Inventory v2 and the Wikidata region subset after that.
7. Schema: `name, type, uri, source` plus synonyms and vernacular names, with an
   FTS5 index over names and synonyms.
8. Version and hash the finished artifact. This single file is what
   collaborators download.

### Phase 2 — Normalisation and unrolling

7. Factor the spaCy loader, `_parse()` and `_head_token()` out of
   `dependency_relations.py` into a shared module.
8. Write the unroller: modifier distribution, coordination splitting, elided
   head recovery, `X of Y` head resolution. Stamp `unroll_available` and the
   loaded model into `run_meta`.
9. Write `mention_norm.py` in the style of `label_norm.py`: case, NFKC,
   whitespace, Morphy for irregular plurals, strip taxonomic authorship
   (`(Audebert, 1799)`), expand acronyms (`sdg`). Always keep the surface form
   as an alias — destroy nothing.
10. Consume `concept_text` when present: token-containment validation,
    precedence over the parser, `concept_source` recorded per mention, failures
    counted.
11. Test the unroller against Mark's 225 `concept_text` pairs, held out.
12. Run the normaliser over the 1,692 unique gold forms and read the full
    collapse list by hand. Confirm `lakes`+`lake` collapse, `high`/`higher`
    do not.

### Phase 3 — Grounding (Tier A)

13. Route by entity type. Canonicalise the type first with
    `canonicalize_entity_type`. Soft-gate on the union of types assigned across
    runs, never one run's type — the runs type `sierra nevada massif` as
    `ABIOTIC_ENTITY` where SPATIAL ENTITY is expected.

    | Entity type | Authority |
    |---|---|
    | BIOTIC ENTITY | GBIF; ENVO/ECOCORE for non-taxa (`forest`, `communities`) |
    | BIOTIC PROCESS | GO biological process + ENVO |
    | ABIOTIC ENTITY / PROCESS | ENVO |
    | ANTHROPOGENIC ENTITY / PROCESS | ENVO anthropogenic branch + Wikidata |
    | any PROPERTY | PATO, OBA, ENVO qualities |
    | SPATIAL ENTITY | GeoNames, Wikidata, GMBA inventory, GADM |
    | CONCEPT | SWEET, else NIL by design |
    | TEMPORAL / QUANTITATIVE / QUALITATIVE | none — normalise only |

14. Exact and alias match via the SQLite index.
15. Fuzzy match with `rapidfuzz`, type-gated, high threshold — handles
    *Salvelina* → *Salvelinus*.
16. WordNet must-not-link veto on every candidate link or merge.
17. Report linked vs NIL **per entity type**. A low ENVO hit rate is a finding
    and it arrives early enough to change the plan.

### Phase 4 — Optional tiers

18. Tier B: FTS5 BM25 ∪ sentence-transformer retrieval over the NIL set, top-k
    candidates only. `en_core_web_trf` as the better parser for unrolling.
19. Tier C: LLM decides from mention + sentence + candidate *definitions*, may
    answer NIL. Verify every proposed ID by lookup in the snapshot; drop and
    count anything unverifiable — the same discipline as
    `add_coref_annotations.py`. LLMs emit plausible-looking GBIF IDs readily.
20. Stamp `decided_by` on every decision. Assert Tier A output is a subset of
    Tier C output.

### Phase 5 — Clustering the residue

21. Cluster only unlinked mentions. Block on (canonical type, head lemma) so
    nothing merges across types; high threshold within blocks; WordNet veto.
    Feed in the existing coref links from `add_coref_annotations.py` as
    pre-existing same-entity edges.
22. Rank NIL clusters by document frequency. The top of this list is the
    measured coverage gap in ENVO/GBIF for mountain biodiversity — the most
    interesting output of the exercise, and worth taking back to GMBA.

### Phase 6 — Output

23. Emit the resolution table: mention `uid` → canonical entity, with
    `decided_by`, `concept_source`, authority URI, and doc/sentence provenance.
    Mention rows are never deleted or rewritten.
24. Stamp provenance using the existing `_git_commit`, `_file_sha256`,
    `_build_run_meta` helpers, plus the version and sha256 of the dictionary
    artifact, the tier that ran, and `unroll_available`.

### Phase 7 — Checks

25. Determinism: same input + same artifact version → byte-identical output.
26. Adversarial pairs stay unmerged: `eastern slopes` / `western slopes`,
    `coniferous forest` / `broadleaf evergreen forest`, `high` / `higher`,
    `lake` / `Sierran lakes`.
27. Known merges happen: `Galeopterus variegatus` / `Bornean Colugo`,
    `Salvelina fontinalis` → *Salvelinus fontinalis*.
28. Count Tier C proposals rejected at verification, and `concept_text` values
    rejected by token containment. Both must be visible in the run record.
29. Cluster-size distribution — catch a bare adjective absorbing a huge cluster.
30. Hand-read the largest clusters and the top of the NIL list. That reading, by
    a domain expert, is the acceptance test.

## Deferred

- **Relations**: inverse-direction canonicalisation, invented predicates, the
  HAS_PROPERTY question (51 of 102 edges in the sampled runs), edge weights,
  shared-neighbour merge evidence, and RDF export against
  `MoBiKo_rdf_schema_v1` — which is relation-shaped, so it waits with them.
  All join to this work on span `uid`.
- **Are bare adjectives entities at all?** `higher`(18), `endemic`(15),
  `high`(13), `local`(9), `threatened`(9). Suggestion: keep as mention rows for
  provenance, do not promote to canonical entities. Confirm once step 29 shows
  the cluster-size distribution.
- **Is `biodiversity`(25) the same as `diversity`(15) and `species
  richness`(19)?** ENVO and ECOCORE distinguish them; a domain expert may not
  want them distinguished. A human call, not a threshold.

## Evaluation, if this later becomes a paper

Not built now, but the design admits it. Sample mention pairs *stratified by
difficulty* (same type, high similarity), never uniformly — uniform sampling is
~99% trivial negatives and yields a meaningless score. Reuse
`annotator_agreement_spans.py` and `human_iaa_report.py`. Metrics: B-cubed and
CEAF for clustering, accuracy@1 for linking, with NIL precision/recall and
coverage reported separately from precision — otherwise a system that links
nothing scores perfectly. The Tier A/B/C boundary is a ready-made ablation.

---

# Build log

## Phase 0 — done (7 Sep 2026)

Installed into `.mobiko-venv`: `rapidfuzz`, `pronto`, `en_core_web_sm`, NLTK
`wordnet`+`omw-1.4`. `rapidfuzz` added to `requirements.txt`; the Tier A/B/C
dependency split is now documented in the README.

**Two plan claims corrected by verification:**

- **`en_core_web_trf` was already installed** in `.mobiko-venv`. The plan said
  no spaCy model was present and therefore that dependency hints had been
  silently disabled in every run — that check was run against `.venv`, the wrong
  environment. The hints machinery had `trf` available all along.
- **Morphy is weaker than claimed.** It handles `taxa`→`taxon`, `larvae`→`larva`,
  `genera`→`genus`, `phyla`→`phylum`, `strata`→`stratum`, `phenomena`→`phenomenon`.
  It does **not** handle `fungi`, `algae`, `bacteria`, `flora`, `fauna` — all
  returned unchanged, with or without an explicit POS. Phase 2 needs an explicit
  supplementary plural table, which is the `label_norm.py` house style anyway.

**And one finding that moves the parser between tiers.** On the plan's flagship
sentence, `en_core_web_sm` mis-parses the coordination that the whole unroller
depends on: it attaches `rainfall` as a conjunct of `centres`, and reads
`correlate` as a `pobj`. `en_core_web_trf` gets both right (`rainfall` conj
`temperatures`). Since `high` distributes onto `rainfall` only via that
`temperatures` link, unrolling on `sm` yields nothing or the wrong modifier.

So span unrolling needs `trf`, which needs torch, which was supposed to be a
Tier B dependency. **This also breaks the "Tier A output is a subset of Tier C
output" property**, because unrolling changes the node key itself rather than
just adding decisions: Tier A would key a node `annual rainfall` where Tier B
keys it `high annual rainfall`, and under modifier-preserving granularity those
are different entities, not a subset. Unresolved; must be settled before the
unroller is written in Phase 2. The likely shape is that the unrolled form
becomes an additional *alias* on the mention rather than a replacement of its
key, so tiers add matching power without moving node identity.

## Phase 1 — done for the ontologies (7 Sep 2026)

Built `src/linking/authorities/{registry,fetch,compile_db}.py`,
`src/linking/mention_norm.py` and `src/linking/coverage_report.py`. Removed the
dead `build_faiss.py` and `link_mentions.py`.

**Source format changed from `.obo` to pinned OBO Graphs JSON.** Three reasons,
found by trying the planned route first: the OBO PURLs redirect to each
project's **master branch** rather than a release, so they are a moving target;
the current ENVO master `.obo` **fails to parse at all** through fastobo, on an
escaped comma inside an xref URL; and the JSON carries **synonym scope**
(`hasExactSynonym` vs `hasBroad`/`Narrow`/`RelatedSynonym`), which is the exact
signal needed to avoid hypernym merges and which the plan had assumed would
have to come from WordNet. Versions are now pinned in `registry.py` and read
back out of the payload, so a mis-pin shows up as a disagreement in the manifest
instead of being asserted away. GO already exhibits one: its `2026-08-05`
release directory ships a payload whose own data-version is `2026-07-26`.

Compiled artifact: **42,633 concepts, 150,422 aliases, 72.3 MB** from ENVO,
PATO, GO (biological_process) and ECOCORE. Alias scopes: 72,393 exact,
42,219 label, 16,362 narrow, 15,026 related, 2,723 broad.

### The coverage number — this is the finding

`python -m src.linking.coverage_report`, over the 3,269 human-gold mentions:

**6.4% of mentions link on label/exact, 8.0% including loose scopes, 4.4% of
unique forms.**

It decomposes, and the decomposition matters more than the headline:

- **~41% of mentions are routable but not yet attempted.** SPATIAL ENTITY (340)
  and CONCEPT (405) have no authority registered yet; BIOTIC ENTITY (589) scores
  6.5% because GBIF is not fetched, and its top misses are `diptera`(16),
  `insects`(12), `populations`(11), `fish`(10) — all things GBIF matches.
- **Morphology is worth real points.** `lakes`(36 mentions) is unlinked while
  `lake` links to ENVO:00000020. Same for `insects`/`insect`,
  `populations`/`population`. That is Phase 2 work, not a coverage gap.
- **Type instability is now measured, not predicted.** `lakes` is typed ABIOTIC
  ENTITY 22 times and SPATIAL ENTITY 14 times *in the human gold*. Soft-gating on
  the union of assigned types is load-bearing; hard gating would lose half.
- **The irreducible gap is the ecology-abstraction vocabulary.** `species
  richness`(19), `endemic`(15), `presence`(15), `diversity`(13),
  `biodiversity`(11+9), `body size`(11), `abundance`(10), `speciation` are absent
  from ENVO, PATO, GO, ECOCORE **and OBA** — five ontologies, ~140k concepts
  checked. Plus the bare adjectives `higher`(16), `high`(12), `local`(9).

ECOCORE was added on the strength of this probe (it is where `predator`, 11
occurrences, lives). OBA was tested and rejected: 84k nodes, 82 MB, and zero
hits on any probed term.

`forest` is a good illustration of the scope machinery working: ENVO has it, but
only as a **related** synonym of `forested area`, so strict matching correctly
declines to claim identity. That is the population a reranker exists to
adjudicate — the `exact`-to-`+loose` gap in the report.

Side note: `canonicalize_entity_type` surfaced six off-schema types in the human
gold — COMPARATIVE PROPERTY, NEGATION, QUANTITATIVE CHANGE, CHANGE, PROCESS,
BIOTIC SUB-ENTITY — 7 mentions in total. Kept and counted, not dropped.

### Still open in Phase 1

GBIF backbone and GeoNames (each >1 GB) are registered and gated behind
`--yes-large`, not yet fetched. They are the single biggest available lift:
together they address the ~41% that is currently routable-but-unattempted.

## Phase 1 — GBIF and GeoNames (7 Sep 2026)

Both fetched: GBIF backbone 971 MB (`Taxon.tsv` alone is 2.25 GB unpacked,
7,746,724 rows), GeoNames 421 MB (13,464,057 rows, 1.79 GB unpacked).

### The artifact had to be split

One file holding all three groups lands near 4 GB, which defeats the point of
"collaborators download one file". Since each entity type routes to exactly one
group, the artifact is now **one SQLite per group**:

| artifact | size | required? |
|---|---|---|
| `mobiko_ontologies.sqlite` | 73 MB | yes |
| `mobiko_geonames.sqlite` | 988 MB | only for SPATIAL ENTITY |
| `mobiko_gbif.sqlite` | ~2 GB | only for BIOTIC ENTITY |

`src/linking/dictionary.py` hides the split: it ATTACHes whichever artifacts are
present and answers over a generated UNION, and it exposes `present` /
`missing_groups` so a report can distinguish **not attempted** from **attempted
and missed**. The compiler was rewritten to stream into SQLite in batches
rather than accumulate — GBIF alone would otherwise hold ~4M concepts in memory.

### Filters, and why they are precision decisions rather than size ones

**GBIF.** Dropped `unranked` (1,275,874 rows — these are not names, they are
UNITE fungal sequence identifiers like `SH1076374.09FU`), `doubtful` (300,247,
which GBIF itself does not vouch for), and variety/form (510k, below the
resolution any ecology abstract writes at). The 3.29M **synonym** rows become
aliases on their accepted taxon rather than concepts of their own — that single
rule is the free de-duplication the whole design rests on.

**GeoNames.** Kept a feature-code set (ranges, plateaus, parks, reserves,
regions, countries) and dropped the long tail: 504k hills, 179k localities, 225k
fourth-order admin units. Not mainly for size — that tail is unusually rich in
collisions with ordinary English words, so against a 154-sentence corpus it is a
false-positive generator.

### Diacritic folding — a normaliser bug caught by real data

`normalize_key` v1 did NFKC, which does **not** strip accents. So `Doñana` ≠
`Donana`, `Białowieża` ≠ `Bialowieza`, `Phước Bình` ≠ `Phuoc Binh` — precisely
the names European and Southeast Asian biodiversity papers use, and precisely
the join that has to work for an English abstract to reach an accented authority
record. GBIF's own matcher folds; GeoNames ships an `asciiname` column for this
exact reason.

Fixed: **NORMALIZER_VERSION bumped to 2**, NFKD plus combining-mark removal,
with an explicit table for the letters NFKD does not decompose (`ł ø đ ß æ œ þ ð`).
Interior punctuation is still preserved (`baetic-rifian range`,
`3,4-dichlorophenol`). The display form is never touched. All three artifacts
were rebuilt; the in-flight GBIF build was stopped rather than finish a 2 GB
file that would be discarded on the next lookup.

The version guard did its job here — this is exactly the silent under-firing it
exists to make loud.

### GeoNames coverage is weaker than expected for this corpus

Confirmed working, including the case the RDF schema itself uses:

- `Carpathian Mountains` → **geonames:682674** — the exact ID in
  `MoBiKo_rdf_schema_v1`'s `skos:exactMatch` example
- `Sierra Nevada`, `Alps`, `Vietnam` all resolve

But checked against the raw dump, several corpus places are **absent from
GeoNames entirely**, not filtered out by me:

- `Bidoup` — no row of any kind
- `Phuoc Binh` — exists only as a populated place (`P.PPL`), not as the
  national park
- `Annamites` — no mountain-range row
- `Langbian Plateau` — present as `Núi Lang Bian` (`T.MT`), so this one is a
  name-variant problem for fuzzy matching, not a coverage gap

So GeoNames does ranges and admin regions well and **protected areas badly**.
Two authorities should be added for SPATIAL ENTITY before concluding anything
about its link rate:

- **WDPA** (World Database on Protected Areas) — the canonical source for
  national parks and reserves; this is where Bidoup Nui Ba and Phuoc Binh live.
- **GMBA Mountain Inventory v2** — named mountain ranges, and it is the
  project's own consortium's dataset. Small.

### Coverage after GBIF + GeoNames

| | exact | +loose | unique forms |
|---|---|---|---|
| ontologies only | 6.4% | 8.0% | 4.4% |
| **+ GBIF + GeoNames** | **15.2%** | **16.7%** | **10.0%** |

BIOTIC ENTITY 6.5% → **41.4%**, SPATIAL ENTITY 0% → **23.5%**.

The taxonomic machinery works, including the case that motivated the whole
request. `annamite striped rabbit` → *Nesolagus timminsi*, `large-antlered
muntjac` → *Muntiacus vuquangensis*, `owston's civet` → *Chrotogale owstoni*,
`collared laughingthrush` → *Trochalopteron yersini*, `insects` → Insecta,
`bats` → Chiroptera — vernacular and scientific names landing on one URI with no
threshold anywhere. That is the de-duplication, obtained for free from grounding.

### But 41.4% is inflated, and the reason is systematic

An audit of every BIOTIC ENTITY link found real false positives:

- `diptera`(16 mentions) → **gbif:2987833 *Saxifraga*** — a plant genus carrying
  *Diptera* as an internal name. A cross-kingdom homonym, and it is the single
  most frequent link of the whole type.
- `fish`(10) → *Gadus morhua*; `beetle`(3) → *Biocrypta curtipennis*;
  `dung beetle`(3) → *Platytomus parvulus*; `trout`(5) → *Salvelinus fontinalis*.

Two distinct failures, both traceable to `LIMIT 1` silently resolving an
ambiguity — precisely the "confident wrong answer" the type gate was meant to
stop:

1. **Homonyms across ranks.** 13 of 84 linked forms are ambiguous, affecting 54
   mentions. `diptera` = Saxifraga(genus) vs Diptera(**order**); `bats` =
   Chiroptera(**order**) vs Vespertilionidae(family); `trichoptera` =
   Trichoptera(**order**) vs Diastata(genus); `bacteria` = Bacteria(genus) vs
   Bacteria(**kingdom**).
2. **Generic English words carrying a species-level vernacular.** `trout` has
   10 candidate species and no higher-rank option; `fish` and `beetle` have one
   arbitrary species each.

Estimated precision on BIOTIC ENTITY links: roughly 85%.

### The disambiguation policy this implies — all Tier A, no LLM

1. **Prefer the highest taxonomic rank** among candidates. Fixes every case in
   group (1) on its own; the rank is already stored per GBIF concept.
2. **Never let `LIMIT 1` decide.** If candidates remain tied after rank
   preference, mark the mention *ambiguous* and hand it up a tier. Ambiguous is
   a third outcome alongside linked and NIL, and must be reported separately.
3. **WordNet as a homonym guard.** A mention whose only candidates are
   species/subspecies rank, reached by a *vernacular* alias, and which is a
   common English noun in WordNet, is suspect — `fish`, `beetle`, `trout` all
   are; `chironomidae` is not. This is the negative-evidence role the plan
   reserved for WordNet, and it lands exactly where predicted.

Consequence for the report: coverage must be quoted as linked / ambiguous / NIL,
never as a single rate, and precision must be sampled rather than assumed.

## Disambiguation policy — implemented (7 Sep 2026)

`src/linking/match.py`, `src/linking/wordnet_guard.py`, tests in
`src/linking/tests/test_disambiguation.py` (20 pass).

**Four outcomes replace a single link/no-link rate.** `LINKED`, `AMBIGUOUS`,
`NIL`, `UNROUTED`. NIL and UNROUTED were already distinguished in the report;
AMBIGUOUS is new and is the one that was previously being silently converted
into a confident answer by `LIMIT 1`.

**The policy, in order** — all deterministic, offline, no LLM:

1. Route by the **union** of entity types assigned across runs.
2. Look up *every* candidate on identity scopes (label, exact, vernacular).
3. **Prefer the highest taxonomic rank.** Settles the homonyms outright.
4. Then prefer the stronger scope: label > exact > vernacular.
5. **Refuse a generic mention resolved to one species** (WordNet guard).
6. Anything still tied is AMBIGUOUS, handed up a tier rather than guessed.

**The WordNet guard, and why plain membership would not have worked.** The
failing and succeeding cases are structurally identical — both single,
species-rank, reached by a vernacular alias:

    fish                    -> Gadus morhua        species   WRONG
    annamite striped rabbit -> Nesolagus timminsi  species   RIGHT

So nothing about the *candidate* separates them; the distinguishing fact is
about the *mention*. WordNet membership does not capture it either —
`chironomidae`, `culicidae` and `ephemeroptera` are all WordNet nouns. The
hyponym **closure size** does, and on this corpus it separates without overlap:

    mammals 1169  birds 871  fish 617  insects 454  bacteria 87
    beetle    62  bats  40   trout  5  dung beetle 3
    ----------------------------------------------------------- threshold 3
    chironomidae, culicidae, ephemeroptera, diptera, rhinolophidae,
    pinus kesiya, annamite striped rabbit, large-antlered muntjac,
    collared laughingthrush, owston's civet        all exactly 0

Crucially the guard fires on the **conjunction**, not on genericness alone:
`insects` -> Insecta and `mammals` -> Mammalia are maximally generic mentions
and correct links, because their candidates are class-rank. And the guard is
inert when the corpus is missing rather than rejecting everything — pinned by a
test.

### Results

Every previously-wrong case is fixed, and two are better than merely blocked —
rank preference *finds the right answer* rather than declining:

| mention | before | after |
|---|---|---|
| `diptera` (16) | *Saxifraga* (genus) | **Diptera (order)** |
| `beetle` (3) | *Biocrypta curtipennis* | **Coleoptera (order)** |
| `bats` (6) | Vespertilionidae or Chiroptera, arbitrary | **Chiroptera (order)** |
| `bacteria` (4) | Bacteria (genus) or (kingdom), arbitrary | **Bacteria (kingdom)** |
| `fish` (10) | *Gadus morhua* | **ambiguous, declined** |
| `trout` (5) | *Salvelinus fontinalis* | **ambiguous, 10 candidates** |
| `dung beetle` (3) | *Platytomus parvulus* | **ambiguous, 7 candidates** |
| `annamite striped rabbit` | *Nesolagus timminsi* | unchanged — still linked |

Corpus totals, now honest:

```
TOTAL   3269 mentions   linked 441 (13.5%)   ambiguous 55 (1.7%)
                        nil   2126 (65.0%)   unrouted 647 (19.8%)
of the 2622 routed mentions: 16.8% linked, 2.1% ambiguous, 81.1% NIL
```

The headline fell from 15.2% to 13.5% because the false positives are no longer
counted as links. **SPATIAL ENTITY is the biggest correction**: 23.5% linked
became 13.2% linked + 10.3% ambiguous, i.e. nearly half its previous "links"
were arbitrary picks among places sharing a name (7 Sierra Nevadas, 5
Californias). Those are exactly the population a Tier C reranker with sentence
context exists for — and they are now labelled as such instead of being wrong
quietly. BIOTIC ENTITY went 41.4% -> 38.0% linked + 3.4% ambiguous.

### Note on the existing test suite

`src/multi_agent_annotation/tests/test_multi_agent_annotation.py` fails to
import (`cannot import name '_get_embedded_guideline'`). Pre-existing and
unrelated to this work — that function does not exist in
`multi_agent_annotation_ag2.py` and neither file was touched here. The other 73
tests pass.

## Audit — three bugs found after the fact (7 Sep 2026)

Prompted by "are you sure everything is correct?". All three were live; one was
producing a wrong answer, and it was a regression introduced by the
disambiguation policy itself.

**1. Cross-authority rank comparison — WRONG ANSWER, self-inflicted.**
Taxonomic rank is defined only inside GBIF, so an ENVO or GeoNames concept
sorted as "unknown" (99) while a GBIF genus sorted as 5. Every taxon therefore
outranked every ontology concept. `biota` (3 mentions) had resolved correctly to
**ENVO:06105012** before the policy; the policy silently changed it to
**gbif:2684854 *Platycladus***, a conifer genus formerly named *Biota*. Fixed:
candidates spanning two authorities are AMBIGUOUS — two kinds of claim with no
principled tiebreak. This one matters beyond the fix, because the earlier
report's "every case correct" was checked only against the cases already under
discussion, not re-checked across the corpus.

**2. Truncated candidate sets were being decided on.** `resolve` asked for 50
rows and the lookup SQL carries no ORDER BY, so a saturated set is an
*arbitrary* subset rather than a best-50. `beetle` has 65 candidates: the right
answer (Coleoptera) came back only because it happened to fall inside the
window. Preferring the highest rank out of an arbitrary subset is a coin flip
wearing a policy's clothes. Fixed: cap raised to 1000 (widest real fan-out is
65) and saturation reported as AMBIGUOUS instead of decided.

**3. The report was not byte-reproducible**, contradicting the determinism
property claimed in the engineering section. `sorted(set, key=-count)` left
ties to set iteration order, which moves with `PYTHONHASHSEED`, so the five
one-mention entity types reshuffled between runs. Every *number* was stable —
only row order moved — but the claim was that output is byte-identical. Fixed
with a total ordering. Note the in-process determinism test passed throughout;
it took running the actual CLI twice in separate processes to see it, and there
is now a test that does exactly that.

Also verified in the audit, and clean: all three artifacts' `build_meta` counts
match their real row counts, FTS row counts match `alias`, zero orphan aliases,
zero duplicate `(uri, norm, scope)` triples, all built at normaliser v2.

Corpus totals after the fixes — linked 438 (13.4%), ambiguous 58 (1.8%), NIL
2126 (65.0%), unrouted 647 (19.8%). Tests: 26 in `src/linking/tests`, 99
overall.

**What this says about the earlier numbers.** Every coverage figure reported
before this audit was produced by code carrying bugs 1 and 2. The totals barely
moved (441 -> 438 linked) because both bugs were rare, but "barely moved" was
not knowable in advance and is not the point: the `biota` case shows the policy
could make a correct link worse, and it was found by auditing rather than by any
test that existed at the time.

## Where to go next — measured, 7 Sep 2026

Three things were measured before choosing, and two of them overturned the
assumption in the plan.

**1. Morphology is worth far less than assumed.** The plan treated singular/
plural as an obvious win ("`lakes`(36) misses while `lake` hits"). Measured:
singularising the head noun recovers **83 mentions, 3.9% of the NIL set**, across
40 forms — `coniferous forests`, `habitats`, `glaciations`, `landslides`,
`canids`. Real, but not the lever the plan implied. Worse, it currently *adds*
false positives: `landscapes` -> `landscape` -> GeoNames "Landscape",
`mountain ranges` -> "Mountain Home Range" in Idaho.

**2. SPATIAL ENTITY has the GeoNames analogue of the `fish` problem, and the
existing guard does not cover it.** `wordnet_guard.SPECIFIC_RANKS` is
species/subspecies only, so nothing fires for places. Split by whether the
mention carries a capital letter:

    Capitalised   24 linked   15 ambiguous   108 NIL   (n=147)
    lowercase     21 linked   20 ambiguous   152 NIL   (n=193)

Of those 21 lowercase links, roughly 15 are wrong — `lake`(8) -> geonames:6048236
"Lake", `landscape`(2) -> "Landscape", `mountain range`(2) -> "Mountain Home
Range", `center`(1) -> "Zou Department", `highland`, `highlands`. The one good
lowercase-initial case is `southern Africa`(6), which contains a capitalised
token. So **a proper-name test — any capitalised token in the surface form —
separates them**, and it is free. Note the hyponym heuristic would NOT work
here: `landscape` scores 0.

This also means SPATIAL ENTITY's 13.2% linked is itself inflated by about a
third.

**3. Scope preference is doing unjustified work inside GeoNames.** `lake` as
SPATIAL ENTITY returns many `Lake County` entries plus one obscure place whose
primary label is exactly "Lake". All share rank (feature codes are not in
RANK_ORDER, so all sort as unknown), so the label-over-exact tiebreak picks the
obscure one. Meaningful for an ontology term versus its synonym; arbitrary for
two places.

### The pattern, and what it implies

Every failure found by auditing has been the same shape: **a generic word
matching a specific record.** Not recall. The fixes are all deterministic, cheap
and offline.

And the strategic consequence of 65% NIL plus an ecology vocabulary that no
ontology carries: **grounding is the minority path for this corpus.** Most
entity resolution here will be done by NIL clustering into project-local
entities, not by authority linking. That inverts the emphasis the plan opened
with, and it argues for building the resolution table and clustering before
chasing more recall.

### Recommended order

1. **Finish the precision work** (~1 day): proper-name test for GeoNames,
   restrict scope preference to within-ontology, re-audit. Makes every number
   trustworthy before anything is built on top of them.
2. **Build the deliverable** — the resolution table plus NIL clustering
   (Phases 5-6). Nothing writes output today; all of this is measurement
   infrastructure. With 65% NIL this is where the actual de-duplication happens.
3. **Then Phase 2** (morphology, unrolling) — demoted on the measurement above,
   and still blocked on the unresolved tier/node-identity question.
4. **Deferred**: WDPA and GMBA for SPATIAL ENTITY recall.

## Step 1 done — precision fixes (7 Sep 2026)

Two changes to `match.py`, 33 tests in `src/linking/tests` (106 overall).

**A. A common noun does not bind to a named individual.** `INDIVIDUAL_SOURCES =
{geonames}`: concepts there are one particular place, not a class, so a mention
must carry a capitalised token to resolve to one. Framing owed to Anisia and it
is the better one — such a mention is not unresolvable but **under-specified**:
"the lake" denotes a lake named earlier in the text, which is a coreference
question, not a lookup question. The outcome says so, and
`add_coref_annotations.py` already resolves anaphora over these documents, so
feeding those antecedents in is the principled fix this outcome is asking for.

The test is "any capitalised token", not "the first" — `southern Africa` passes
on *Africa*.

**B. Scope preference no longer applies between places.** label-over-synonym
says a concept's own name beats its synonym. True of an ontology term; two
places are not a name and a synonym. It had been picking an obscure feature
officially named "Lake" over every "Lake County".

**A contract this exposed, now documented:** `resolve()` must be given the
mention's **surface form**, never a normalised key — capitalisation is evidence
and casefolding destroys it irreversibly. One earlier test was passing a
pre-lowercased string and started failing, correctly.

### Result

SPATIAL ENTITY 45 linked -> **26 linked**, 35 ambiguous -> 54. Every one of the
26 survivors was hand-checked and is a real place: Vietnam, Laos, Cambodia,
Argentina, Ecuador, Nepal, Himachal Pradesh, Uttarakhand, southern Africa,
Sierra National Forest, World -> Earth. The 19 that moved were `lake`(8) -> a
hamlet named "Lake", `landscape`(2), `mountain range`(2) -> "Mountain Home
Range", `center` -> "Zou Department", `highland`, `highlands`.

`Viet Nam` and `Vietnam` both resolve to geonames:1562822 — the de-duplication
working as designed, two surface forms on one entity.

Corpus totals: linked 419 (12.8%), ambiguous 77 (2.4%), NIL 2126 (65.0%),
unrouted 647 (19.8%).

**The headline has now fallen three times — 15.2% -> 13.5% -> 13.4% -> 12.8% —
and every drop removed a wrong answer.** The recall figures were never real;
they were false positives being counted as links.

### Left deliberately alone

`Alps` has two candidates, an A.ADM2 unit named Alps and T.MTS the mountain
range, and is now ambiguous. A feature-class preference (physical geography over
administrative units) would resolve it and would probably be right for this
corpus — but it is a domain heuristic, not a principle, and heuristics of
exactly that shape produced the *Saxifraga* bug. Two candidates handed to a
reranker is the cheaper mistake. Noted, not implemented.

Verified not a bug: `Sierra Nevada` matching *Sierra Tacatrán* and *Sierra Piré
Mahuida* is correct — GeoNames really does record "Sierra Nevada" as an
alternate name for both Argentine ranges. 14 genuine candidates; declining is right.

## Decision: entity type is part of entity identity (7 Sep 2026)

`biodiversity` as BIOTIC PROPERTY and `biodiversity` as BIOTIC ENTITY are **two
entities**, not one entity two annotators disagreed about. Confirmed by Anisia.

Consequences:

**Blocking key for NIL clustering is (entity type, head lemma)** — already what
was planned, now justified rather than assumed. Type conflicts propagate into
the graph as separate nodes; that is the intended behaviour, not a defect to
repair upstream.

**Union-type routing was wrong and is retired.** `resolve()` had documented
multi-type routing as the *normal* case, justified by `lakes` being ABIOTIC
ENTITY 22x and SPATIAL ENTITY 14x in the gold. That reasoning was backwards:
those are 36 mentions of two entities, not 36 mentions of one entity with a
disputed type. Measured, unioning changes 135 mentions and the changes are wrong
in kind — it leaks across ontological roles:

    mountains     as SPATIAL ENTITY   nil      -> ENVO:00000264
    habitat       as SPATIAL ENTITY   nil      -> ENVO:01000739
    location      as SPATIAL ENTITY   nil      -> PATO:0000140  (a property)
    distribution  as CONCEPT          unrouted -> PATO:0000060

**No reported number changes**: the coverage report already routed per-type, so
the wrong principle lived only in the API default and its docstring. Totals
stand at linked 419 (12.8%), ambiguous 77 (2.4%), NIL 2126 (65.0%), unrouted
647 (19.8%). 34 tests; two new ones pin that routing cannot leak across roles
and that one string under two types stays two entities.

## What Step 2 actually is

**The resolution table** — the deliverable, which nothing writes yet. One row
per mention, keyed by the span `uid` already in the annotation format:

    uid       surface form        doc/sent  entity_id           outcome    decided_by
    def3357d  "Vietnam"           d1/s3     geonames:1562822    linked     tier_a
    a1b2c3d4  "Viet Nam"          d2/s7     geonames:1562822    linked     tier_a
    7f9e2a1b  "species richness"  d1/s2     mobiko:e0042        clustered  tier_a
    b8a9c0d1  "the lake"          d1/s9     -                   ambiguous  -

Two mentions are the same entity iff they share `entity_id`. Relations join back
on `uid`, so they need no rework.

**NIL clustering** gives project-local ids to the 65% that ground to nothing.
Blocking on (type, head lemma) collapses 2126 NIL mentions to **920 candidate
blocks**; the largest are:

    [BIOTIC ENTITY | species]        91 mentions, 18 forms
    [BIOTIC ENTITY | forest]         32 mentions, 10 forms
    [BIOTIC PROPERTY | richness]     29 mentions,  5 forms
    [SPATIAL ENTITY | area]          27 mentions, 15 forms
    [ABIOTIC PROPERTY | temperature] 26 mentions, 16 forms

Blocking is not clustering — it only narrows the candidate set cheaply. Under
modifier-preserving granularity most members of a block stay separate:
`temperatures` = `temperature`, but `peak water temperature` and `gradient of
temperature` are distinct, and `bird species` != `bat species` != `species`. So
the work is mostly **lexical variants** (plural, case, word order), which is the
safe version — over-merging is unrecoverable, under-merging is not.

Side output, possibly the most valuable one: ranking those clusters by document
frequency yields a measured list of concepts this domain uses constantly that no
ontology carries.

## Step 2 done — resolution table and NIL clustering (7 Sep 2026)

`src/linking/cluster.py`, `src/linking/resolve_corpus.py`, tests in
`test_cluster_and_table.py`. 53 tests in `src/linking`, 126 overall.

Output: `output/resolution.jsonl` (3,269 rows) + `.meta.json` sidecar, and
`output/ontology_gap.tsv`.

### Clustering is an exact-key problem, not a similarity problem

This falls straight out of the granularity decision rather than any cleverness.
Modifier-preserving means `high temperatures` and `temperature` are *meant* to
be different entities, so all clustering has to do is recognise lexical variants
of one phrase. That is an exact key:

    key = (entity type, sorted multiset of lemmatised content words)

No threshold, no embeddings, no WordNet veto for this step. 2126 unresolved
mentions -> 1614 entities; 73 clusters merge more than one surface form and
every one is a genuine variant (word order, plural, case, em dash, trailing
punctuation, diacritics). Nothing merges across a modifier difference.

Ids are content-derived (`mobiko:e<sha256[:12]>`), so the same phrase gets the
same identity in any run or corpus. A sequential counter would renumber
everything when a document was added and break the relation join that attaches
later by `uid`.

### Four problems found while building it

**1. `doc_id` was useless for document frequency.** The files concatenate 15
abstracts under one `doc_id: "abstracts"`, so every gap-report row scored
`doc_freq=1` and the ranking degenerated to mention count — defeating the whole
reason for ranking by documents. Fixed by splitting at `[abstract]` markers,
reusing the boundary regex and the rationale from
`scripts/add_coref_annotations.py`. The fix visibly works: `lakes` (22 mentions,
all in one abstract) correctly dropped out of the top, while `endemic` (16
mentions across 6 abstracts) rose.

**2. `uid` is not unique — not across files, and not within one.** 15 uids are
reused inside Mark's file alone, mostly a span recorded twice identically, and
once (`359f7971e0`, "species" at sentence 159) the *same span* annotated BIOTIC
ENTITY **and** TEMPORAL PROPERTY — an unresolved type conflict sitting in the
gold. Keying the table on `uid` silently merged those. Rows are now keyed by
`mention_id` = `<file>:<uid>#<occurrence>`; `uid` is retained because that is
what a relation file joins against within its own annotator's data. 27 reuses
are counted into the sidecar rather than repaired — they are the annotators'
files, and collapsing duplicates would change span counts other analyses rely on.

**3. Grounding was quietly undoing type-as-identity.** `habitat` is ABIOTIC
ENTITY in one sentence and BIOTIC ENTITY in another, and both link to
ENVO:01000739 — one entity by the URI, two by the rule. Ten pairs do this
(`habitat`, `size`, `climate`, `grassland`, `length`, `rhizosphere`,
`orientation`, `temperate`, `sensitive`, `variability`), every one an annotator
disagreement about domain or role rather than about which concept is meant.
Resolved by separating the two notions: `entity_id` is type-scoped
(`BIOTIC ENTITY@https://www.gbif.org/species/734`) while `authority_uri` carries
the bare URI. Two entities may share an authority_uri, which RDF renders as two
nodes with `skos:exactMatch` to one term — visible to a human who may want to
merge them. Splitting is the reversible error.

**4. Two placeholder expressions shipped into the first draft** of the writer
(`n_entities` summing over an empty list). Caught before running, but they were
written, and they would have produced a plausible-looking wrong count.

### Result

    3,269 mentions -> 1,763 distinct entities
      149 grounded in an authority
    1,614 project-local
    3,192 carry an entity_id; 77 deliberately unresolved (ambiguous)

De-duplication working end to end, including the case that prompted the request:
`Diptera`/`flies`/`Diptera.` -> one GBIF entity; `insects`/`insect`/`Insect`/
`Insects` -> gbif:216; `Vietnam`/`Viet Nam` -> geonames:1562822;
`elevation`/`Elevation` -> PATO:0001687. 123 entities merge more than one
surface form.

### The gap report

`output/ontology_gap.tsv`, 1,614 entities ranked by how many abstracts use them.
The head of it is the measured coverage gap for this domain:

    11 docs  species              6 docs  populations, endemic, high
     5 docs  biodiversity         4 docs  higher, diversity, local, data, future
     3 docs  species richness, abundance

`biodiversity` appears twice, as BIOTIC PROPERTY (5 docs) and BIOTIC ENTITY
(4 docs) — two entities, exactly as decided.

## The identity model, settled (7 Sep 2026)

Two decisions from Anisia, reconciled by the schema's own two-layer structure
(`MoBiKo_label_guidance_v4`, Step 2: an ontological **role** — entity, process,
property, concept — crossed with a **domain** — abiotic, biotic, spatial, …).

**Identity is the role. The domain is a label.**

  * `biodiversity` as ENTITY / PROPERTY / CONCEPT — three claims about the
    world, so **three entities**.
  * `habitat` as ABIOTIC / BIOTIC / SPATIAL ENTITY — two annotators disagreeing
    about one thing, and it grounds to ENVO:01000739 every time, so **one
    entity carrying three labels**.

**Two mentions are one entity if they share a content key OR an authority URI,
within a role.** Each signal alone loses cases the other keeps:

    Vietnam / Viet Nam    different words, same GeoNames id   -> authority joins
    Diptera / flies       different words, same GBIF id       -> authority joins
    mountains x2          same words, only one grounded       -> words join

The union is taken as connected components (`cluster.link_signatures`), so
joining chains: `mountains` typed ABIOTIC ENTITY found ENVO_00000081 while
`mountains` typed SPATIAL ENTITY found nothing, and because the words join them
the whole group inherits the ENVO id.

Effect on the split that prompted this:

    keyed on the full label      164 forms / 667 mentions / 20.4%
    keyed on role only           106 forms / 454 mentions / 13.9%
    plus authority-or-words      428 mentions, ALL of them genuine role
                                 differences; same-role artefacts 26 -> 0

Checked for the risk this creates: **0 entities span more than one authority
URI**, so nothing was over-merged by chaining.

Final: 3,269 mentions -> **1,663 entities** (121 grounded, 1,542 project-local),
3,192 carrying an id and 77 deliberately unresolved. 54 tests in `src/linking`.

## concept_text honoured (7 Sep 2026)

Optional input, as decided: where a span carries a `concept_text` and the text
supports it, that becomes the mention's **canonical form** and is what both
matching and identity use. The surface form is kept on every row, and
`concept_source` records which was used, so a run is readable without guessing.

**Two bugs found in my own first attempt, both by measuring rather than
reasoning.**

*Tokenisation.* `normalize_key` strips punctuation only from the ends of the
whole string, so a sentence ending "...topographic relief." yields the token
`relief.` and containment reported `relief` as missing from a sentence plainly
containing it. That single bug rejected most valid values. Tokens are now
trimmed individually.

*Scope.* Validating against the sentence alone was too narrow, and it failed on
exactly the highest-value corrections: the annotators expand abbreviated
binomials — span `frater` -> `Myotis frater`, `javanicus` -> `Pipistrellus
javanicus` — using a genus named earlier in the abstract. A full binomial is
what GBIF can match and a bare epithet is not, so validation now falls back to
document scope and records which scope accepted it.

**A data-quality finding that matters for the plan.** Of the 281 `concept_text`
values in the relation-annotation inputs, **190 are the annotation tool's
placeholder** — the literal token `x` prefixed to the span text ("x three",
"x phylogenetic", "x Bacillota"), all of them in Mark's file, none in Davnah's.
86 are real corrections; 5 are genuine corruptions ("Helarctos malayanus" ->
"Helarctos Malayans", "predator presence" -> "predator changed").

So the field as it stands is roughly two-thirds noise. Worth knowing before
relying on it: the plan assumed concept_text would be sparse but sound, and it
is sparse and mostly junk. The validator handles it — the placeholders are
rejected because `x` occurs nowhere in the text — but the annotation tool
should be fixed at source.

On the default corpus nothing changes (the postprocessed files carry no
concept_text): 1,663 entities, unchanged. 60 tests in `src/linking`, 133 overall.

## Parser analysis — the unroller should not be built (7 Sep 2026)

The plan made span unrolling a first-class step, on the strength of one example:
*"areas of high temperatures, annual rainfall and topographic relief"*, where
`high` distributes across the conjunction. Measured against real ground truth,
that case is rare and the approach does not work.

### Speed is irrelevant; accuracy is the only question

`en_core_web_sm` parses the 377 corpus sentences in ~1s, `en_core_web_trf` in
~15s on CPU. Neither is a constraint.

### The two models disagree a lot

Over 1,641 annotated spans with offsets:

    identical parse                     78.4%
    same head, different attachment     13.7%   <- what unrolling reads
    same head, different modifiers       4.6%
    different head token                 3.3%

**21.6% of spans parse differently.** Running a coordination-distribution rule
under both, the models disagree on the OUTPUT for 4.3% of spans — and trf
produces *more* garbage, because it finds more coordination structure to
distribute across:

    Annamite crested argus -> "rabbit Nesolagus Annamite crested argus"
    large-antlered muntjac -> "rabbit Nesolagus large-antlered muntjac"
    wildlife exploitation  -> "timber wildlife exploitation"

In a species list the first conjunct's own name gets distributed onto the rest.
Restricting to adjectives, and to conjuncts with no adjective of their own, and
forbidding proper nouns, each removed the garbage — and also removed every
correct case.

### Evaluated against ground truth: recall 1/38

Mark's and Davnah's validated `concept_text` values give 65 real corrections, of
which 38 prepend a modifier — exactly what unrolling should reproduce.

    coordination rule, en_core_web_sm     recall  1/38
    coordination rule, en_core_web_trf    recall  2/38

Because the premise was wrong. What the annotators are actually doing:

| correction | count | needs a parser? |
|---|---:|---|
| span drawn too narrow — extend left over adjacent words | 15 | no, POS tagging suffices |
| abbreviated binomial — `frater` -> *Myotis frater* | ~6 | no, document abbreviation lookup |
| **coordination modifier distribution** | **~3** | yes |
| other context supply, no reliable signal | ~14 | — |

The flagship case is 3 of 38. Building a parser-driven unroller to catch it,
on a parse the two models disagree about 21.6% of the time, is not worth it.

### What to build instead

**1. Span extension (15/38).** The concept_text is a contiguous substring of the
sentence ending where the span ends — `responses` -> `positive responses`,
`Annamites` -> `southern Annamites`, `evergreen forest` -> `broadleaf evergreen
forest`. Finding these needs no parser at all; only the stopping rule does, and
a POS tagger gives that far more reliably than sm's parser gives attachment.

**2. Abbreviated binomial expansion.** Measured on the corpus: **7 spans** are a
bare epithet or abbreviated genus that GBIF cannot match, and that WOULD match
once expanded from a genus named in the same abstract:

    N. siki    -> Nomascus siki          frater     -> Myotis frater
    O. mykiss  -> Oncorhynchus mykiss    insignis   -> Tadarida insignis
    javanicus  -> Pipistrellus javanicus huttonii   -> Murina huttonii

Small in absolute terms, but these are BIOTIC ENTITY mentions converting
straight from NIL to grounded, which is the type where grounding actually works.
No parser, no model — a regex for `X. epithet`, the genus names already in the
document, and a lookup that must succeed in GBIF before the expansion is
accepted.

**Decision: drop the coordination unroller. Build span extension and binomial
expansion.** Both are deterministic, both stay in Tier A with no GPU, and
together they address 21 of the 38 corrections against the unroller's 1.

## Coordination splitting — built; distribution dropped (7 Sep 2026)

`src/linking/split_spans.py`, 12 tests. The two halves of "unrolling" turned out
to have completely different value, and only measurement separated them.

**Distribution — dropped.** Pushing a shared modifier onto a neighbouring span
(`annual rainfall` -> `high annual rainfall`) reproduced **1 of 38** real
corrections with sm, 2 with trf. Almost none of what the annotators actually
record in `concept_text` is coordination: 15 of 38 are spans drawn too narrow,
~6 are abbreviated binomials, and only ~3 are coordination.

**Splitting — built.** Of 17 distinct coordinated spans in the gold it splits 8
and declines 9, and the declines are the traps:

    eastern and western slopes  -> eastern slopes | western slopes   (matches gold)
    high and transient relief   -> high relief | transient relief
    broadleaf evergreen and coniferous habitat mosaic
                                -> evergreen broadleaf habitat mosaic
                                 | coniferous broadleaf habitat mosaic
    between 1100 and 1700m      -> declined (a range is one quantity)
    Monitoring and Assessment Programme -> declined (AMAP, a proper name)

`eastern and western slopes` is the one case the annotators split themselves, as
two `concept_text` values on one span, and the splitter reproduces it exactly.

**Splitting requires `en_core_web_trf`, and that is a safety decision rather
than an accuracy preference.** sm parses the corpus in ~1s against trf's ~15s,
so speed is irrelevant; but the two disagree on the attachment of 13.7% of
spans, and on THIS operation sm converts disagreement into invented entities
while trf declines:

    Da Nhim and Dran Protection Forests   sm: "Da Nhim" + "Da Forests"
    Monitoring and Assessment Programme   sm: "Monitoring" + "Programme"

An entity minted from a mis-parse is not recoverable downstream, so without trf
the splitter is OFF and says so, rather than degrading. Falling back to sm is
opt-in.

**Two bugs found while building, both by looking at output.** Taking the
modifier's subtree whole made the first part "eastern western slopes", because
spaCy hangs later conjuncts under the first one. And taking only the head token
dropped shared words, turning "…coniferous habitat mosaic" into "coniferous
mosaic" — a silent meaning change.

**Representation.** A coordinated span is emitted as a container row with **no**
`entity_id` — "eastern and western slopes" is not an entity — plus one row per
part carrying `split_from` and `mention_id` suffixed `.1`, `.2`. The original
span keeps its offsets, so nothing is lost and `--no-split` yields a strict
subset of the rows.

Corpus: 12 spans split into 24 parts; 3,293 rows, **1,671 entities**. 72 tests
in `src/linking`, 145 overall.

## Decision reversed: full-label identity, shared ontology link (7 Sep 2026, after meeting)

`habitat` under ABIOTIC / BIOTIC / SPATIAL ENTITY is **three nodes**, all
carrying the same `authority_uri` — ENVO:01000739 — so the ontological entity
they share is stated rather than achieved by merging them.

This supersedes the role-only key tried earlier the same day, which made habitat
one node with three labels. Reason: a merge is not recoverable afterwards,
whereas separate nodes sharing an `authority_uri` are recoverable in both
directions. In RDF it is three nodes with `skos:exactMatch` to one ENVO term.

**Implementation.** `content_key` and both union-find signatures carry the full
label again; scoping the URI signature by role alone would have pulled the three
habitat nodes back together through their shared id. `entity_role` and
`entity_domain` are still computed and exposed — the two-layer distinction is
real — they just no longer decide identity.

**Two grouping handles, so nothing is lost by keeping the nodes apart:**

- `authority_uri` — 10 ontological entities in the gold are shared across labels
  (`ENVO_01000739` habitat, `PATO_0000117` size, `ENVO_01001082` climate…)
- `surface_key` — the content words with the label stripped, for concepts no
  authority covers. 175 groups, e.g. `habitat` → 4 nodes, `community` → 4.

**Result:** 1,671 → **1,754 entities** (122 grounded, 1,632 project-local) over
3,293 rows. 74 tests in `src/linking`. `UNIFICATION_RULES.md` §3, §4, §6 and §7
updated, with the superseded rule recorded rather than deleted.

## Universal ontology routing confirmed; the gap report repaired (8 Sep 2026)

**Decision (Anisia): ENVO and PATO may link to any entity type.** This confirms
the `universal=True` setting already in `registry.py` and supersedes the reading
in "Decision: entity type is part of entity identity", which listed `mountains
as SPATIAL ENTITY -> ENVO:00000264` among the union-routing leaks. Reaching an
ontology from any type is now correct by decision; the gated authorities (GBIF,
GeoNames) are where per-type routing still bites, and that is where the test
now probes it.

Four tests were pinning the pre-universal policy and had been failing since the
setting changed. Rewritten to pin the same properties under the new one:
UNROUTED is now unreachable and the test says so and says why; the routing test
uses GBIF and GeoNames (`Vietnam` as BIOTIC ENTITY, `diptera` as SPATIAL
ENTITY); `mountain range` is still declined but by the cross-authority rule
rather than the coreference one, so the reason assertion moved to `landscape`
and `center`. `UNIFICATION_RULES.md` §5.1 and §8 updated. **77 tests pass, 0
fail.**

### Three bugs in the resolution table and the gap report

**1. `--gap-report` had been crashing.** It read `c.entity_type` where `Cluster`
exposes `entity_types` and `role`, so it raised after writing the header and
left a 60-byte file that still looked like output. `output/ontology_gap.tsv` had
been empty since the full-label identity rewrite, and no test exercised the
flag. `Cluster.entity_type` added; a test now covers the flag end to end.

**2. The sidecar's grounded/project-local split was wrong.** It was derived as
`len(distinct_entities) - len(clusters)`, but a content key that chains into a
grounded component is named by its URI and is no longer project-local. Reported
148/1,606; the table itself says **176/1,578**. Now counted directly, with
`n_unresolved_clusters` recording the cluster count so the difference is
visible. The "(122 grounded, 1,632 project-local)" figure in the entry above
came from the same formula and is low by the same mechanism.

**3. The gap report was overstating the gap** by 28 entities — the chained ones,
whose minted ids appear nowhere in the table. Since this file is what a domain
expert reads, it now lists only what stayed unresolved.

### Phase 7 item 29 — cluster-size distribution, run

No over-merging, and no bare adjective absorbing a cluster:

    1 mention  1085 entities      max 63 mentions, mean 1.69
    2          313                merging >1 surface form: 94/1578 (6.0%)
    3           70                max surface forms in one entity: 3

The largest merges are all genuine variants (`endemic/Endemic/endemics`,
`broadleaf evergreen forest` x3 word orders, `Annamites/Annamite/Annamites:`).
`higher`(16) and `high`(12) are SINGLE-form clusters — frequent mentions of one
phrase, not merges — and they split correctly across labels: `higher` is three
entities, QUALITATIVE / BIOTIC / QUANTITATIVE PROPERTY. Gap report verified
byte-identical across two runs.

Item 30, the domain-expert hand-read, is the remaining acceptance test and needs
Davnah or Mark. The artefact is ready.

### Open: `mountains` links to *seamount*, and it is not a routing problem

ENVO carries `mountains` as an **exact** synonym of `seamount` (ENVO:00000264)
and as a merely **related** synonym of `mountain` (ENVO:00000081). Related is
not an identity scope, so the correct concept is excluded and the undersea one
wins. **All 14 mentions of `mountains`/`Mountains` in the corpus resolve to
seamount**, in a corpus about mountain biodiversity. Singular `mountain` is
correct.

Note this is independent of the routing decision: 8 of the 14 are ABIOTIC
ENTITY, which ENVO serves by declaration. It also means the recorded claim that
opening the ontologies "gained 51 links, lost zero" is not right — this is a
loss, and `variable` -> PATO `variant` looks like another. Unfixed.

## `mountains` -> *seamount* fixed at the source (8 Sep 2026)

Confirmed against the raw release: ENVO ships this pair with their scopes
swapped, so our compiler was faithful and the error is upstream.

    ENVO:00000081  mountain   hasRelatedSynonym  "mountains"
    ENVO:00000264  seamount   hasExactSynonym    "mountains"

`related` is not an identity scope, so the correct concept never entered the
candidate set and the undersea one won unopposed. Singular `mountain` was
always right, which is what identified it as data rather than policy.

**Three fixes were measured before choosing.**

*Rejected — decline when exact and related conflict.* "Another concept carries
this string at a loose scope, so the identity claim is unsafe." Fires on 6
forms, of which **3 are correct answers**: `coniferous forest` -> needleleaf
forest, `grassland` x2 -> grassland ecosystem. Loose synonyms on neighbouring
concepts are ordinary ontology structure and carry no signal. It would destroy
three right answers to fix three wrong ones.

*Not taken — let a concept's own LABEL beat another's synonym*, allowing a
singular variant, restricted to mentions that already matched so it adds no
links. Touches only 4 forms corpus-wide, but fixes just 8 of the 14 mentions
and declines the other 6 (`mountains` as SPATIAL ENTITY would go ambiguous
across ENVO and GeoNames), and it turns `carnivores` from GBIF *Carnivora* into
an ambiguity against ECOCORE `carnivore`. A defensible policy change on its own
merits; not the right way to route around bad data.

*Taken — correct the two rows at build time.* `OBO_SCOPE_CORRECTIONS` in
`compile_db.py`, keyed by (URI, normalised alias). Same category as the
existing `GBIF_RANKS` / `GEONAMES_KEEP` curation of upstream data, and the only
option that gets all 14 right. `build_meta` records the declared corrections
and any that no longer fire, so an ENVO upgrade surfaces a stale entry instead
of silently keeping it. Worth filing with ENVO.

**Result.** All 14 mentions now resolve to ENVO:00000081 *mountain* under both
ABIOTIC and SPATIAL ENTITY, sharing one `authority_uri` as the identity model
intends; zero mentions land on seamount; *seamount* itself still resolves to
its own concept. Corpus totals unchanged at 470 linked / 77 ambiguous / 2,746
NIL — a re-pointing, not a gain. Artifact rebuilt (sha256 99f5036496ad…).
**81 tests pass.**

**Correction to an earlier entry.** "Opening ENVO and PATO to every type gained
51 links, lost zero" was not right: this was a loss, and `variable` -> PATO
`variant` looks like another. The universal decision still stands (Anisia,
8 Sep); the "lost zero" claim was never re-checked across the corpus, the same
gap the 7 Sep audit found in "every case correct".

### Still open, measured but not decided

**Morphology deserves re-measuring.** It was demoted on the grounds that it
adds false positives — `landscapes` -> GeoNames "Landscape", `mountain ranges`
-> "Mountain Home Range". Both are lowercase mentions binding to GeoNames
individuals, which the proper-name rule added AFTERWARDS already refuses. That
measurement predates its own fix. **43 currently-NIL forms** have a singular
that is some concept's label: `Lakes` -> lake, `glaciers` -> glacier,
`habitats` -> habitat, `ecosystems` -> ecosystem, `glaciations` -> glaciation.

## Morphology re-measured, and the singular fallback built (8 Sep 2026)

Morphology was demoted on 7 Sep for adding false positives: `landscapes` ->
GeoNames "Landscape", `mountain ranges` -> "Mountain Home Range". **Both are
lowercase mentions binding to GeoNames individuals, which the proper-name rule
added later that same day already refuses.** The measurement predated its own
fix. Re-run against the current guards, both cases now DECLINE:

    landscapes      -> landscape       "common-noun mention matched a named individual place"
    mountain ranges -> mountain range  "candidates in 2 authorities (envo, geonames)"

All 8 forms the fallback would decline are SPATIAL ENTITY, every one caught by
the proper-name rule, the cross-authority rule or a rank/scope tie — all three
added after the original measurement. **The failure mode morphology was
rejected for is now structurally unreachable.** Zero GeoNames links among the
gains.

### Why the gain is 28 and not the 83 first reported

`content_key` already lemmatises, so a plural whose singular appears elsewhere
in the corpus is ALREADY one entity by chaining. The old figure counted work
the clustering was doing anyway. The genuine gain is the residue where no
singular form occurs anywhere: **28 mentions across 24 forms.**

Roughly 24 of the 24 forms are clean — `glaciations` -> glaciation,
`landslides` -> landslide, `boulders` -> boulder, `thunderstorms` ->
thunderstorm, `protected areas` -> protected area, `littoral zones` -> littoral
zone, `mutualisms` -> mutualism, `canids` -> Canidae via the vernacular "Canid".
Four are wrong or dubious and all are low-frequency: `genera`(2) -> a GBIF junk
row literally named *Genus* at genus rank, `variables`(1) -> PATO *variant*,
`splits`(1) -> PATO *split*.

### Built as a third canonical form, not as a change to the normaliser

Two ways to finish the job the `mention_norm` docstring left open:

*A — put Morphy in `normalize_key`.* The eventual right answer, and expensive:
NORMALIZER_VERSION 2->3 and a rebuild of all three artifacts, with everyone
holding the current ones re-downloading. Measured the collision risk on the
ontologies: lemmatising 148k stored norms creates only **+25** new collisions,
but every one is a CHEBI pair where singular and plural are genuinely different
concepts (`sulfate`/`sulfates`, `nitrate`/`nitrates`) — real distinctions that
would be destroyed. GBIF's 8.7M and GeoNames' 2.7M aliases are unmeasured, and
that is where names, not words, would be lemmatised.

*B — taken.* The singular is one more entry in the canonical-form precedence
the code already has for `concept_text`, reached only when the surface form
matched nothing, recorded as `concept_source: "singular"`, disabled with
`--no-singular`. No rebuild, no version bump, per-row provenance, and it does
not block A: once the normaliser lemmatises, step 3 stops finding anything and
can be deleted.

**Only a LINK is accepted, never an AMBIGUOUS.** The singular is a hypothesis
about what the plural denotes; one that grounds cleanly is evidence, one that
comes back ambiguous says only that the singular is ambiguous — a fact about a
form the annotator did not write. Adopting it would strip 15 mentions of an
entity_id they currently hold (`landscapes` inheriting `landscape`'s refusal)
and gain nothing.

### Result, diffed against `--no-singular`

    rows                                3,293
      entity_id unchanged               3,265
      ungrounded -> grounded               28
      MOVED between two entities            0

    linked   470 (14.3%) -> 561 (17.0%)     nil 2,746 -> 2,655
    entities 1,754 unchanged; grounded 176 -> 200, project-local 1,578 -> 1,554

91 mentions changed nil->linked, of which 63 already carried an `authority_uri`
through chaining and kept the same one. Those 63 are a reporting fix rather
than new grounding: they had been recorded as `outcome: "nil"` while carrying
an authority URI, which reads as a coverage gap and was not one.

Morphy's known blind spots do not bite on this corpus: of `fungi`/`algae`/
`bacteria`/`flora`/`fauna`, only `bacteria`(4) and `phyla`(2) occur and both
were already linked; `taxa`(7) and `genera`(2) lemmatise correctly. No
supplementary plural table needed here.

**89 tests pass.** The fallback is pinned by the same invariant as `--no-split`:
turning it off yields a strict weakening of the same table, never a different
one.

## Fuzzy matching measured and rejected; `Dictionary.search` repaired (8 Sep 2026)

**Rejected for general use.** Over 1,665 unresolved forms, identity scopes only,
exact matches excluded, `fuzz.ratio` over BM25 candidates:

    thr 95 -> 11 forms / 14 mentions      thr 90 -> 37 / 44
    thr 92 -> 24 forms / 28 mentions      thr 88 -> 58 / 79

Hand-checked precision is roughly 45% and **does not improve as the threshold
rises**, which is the finding. The top-scoring errors are one-character
taxonomic differences, and in Latin nomenclature the suffix IS the rank:

    Arachnidae     -> Arachnoidae   95.2     different family
    Scarabaeinae   -> Scarabaeidae  91.7     subfamily vs family
    Actinomycetota -> Actinomycetia 88.9     phylum vs class

So string distance is anti-correlated with semantic distance for exactly the
strings this corpus is about. It also resurrects `Dipteran` -> *Saxifraga*, and
brings back `landscapes` -> "Landscape" and `mountain ranges` -> "Mountain Home
Range" — capitalised, so the proper-name rule waves them through. Worst by
volume: `human wellbeing` -> **human dwelling**, 7 mentions at 89.7.
`predator behavior` -> *antipredation* behavior inverts the meaning.

WRatio is worse still: its partial-ratio component gives 90 to any candidate
CONTAINED IN the mention, so `Bidoup Nui Ba National Park` — the one case whose
right answer is NIL, since Bidoup is genuinely absent — draws a confident 90.0
to "Núi Ba".

**Every true positive has a better deterministic fix**, which is the useful
half of the result: morphology Morphy missed (`interglacials`, `biomarkers`,
`nutrients`), en-US/en-GB spelling (`behavior`/`behaviour`), and PATO's
systematic comparative pattern (`higher elevations` -> increased elevation,
6+ mentions). The only irreducible fuzzy case is the typo `Salvelina
fontinalis`, 2 mentions. A narrow rule — GBIF only, >=2 tokens, species rank,
ratio 90-100 — yields exactly that one form with zero false positives, and
would close Phase 7 item 27. Not built; noted.

### `Dictionary.search` was broken, not merely unused

It built `"Salvelina" "fontinalis"`, and FTS5 puts an implicit **AND** between
bare terms. That inverts what a candidate generator is for: a correctly spelled
mention was already found by `lookup`, and a misspelled one has a token the
index does not contain, so requiring every token returns nothing exactly when
candidates are needed. Measured: 0 candidates while *Salvelinus fontinalis* sat
in the index on 4 rows; the same tokens OR-ed return 567. With no caller and no
test, it had never once produced a row — and would have handed its first caller
a plausible-looking empty result, the "not attempted vs attempted and missed"
confusion the rest of the module is careful about.

Fixed: terms OR-ed; long terms matched by a prefix of themselves with up to 3
trailing characters dropped (`"annamites"*` matches only terms starting with all
of it and so misses the shorter `annamite` — dropping the tail finds both);
every term quoted, because `3,4-dichlorophenol` and `species'` are legitimate
mentions and a syntax error unquoted. Known targets now reachable at rank 1
where none was reachable before: *Salvelinus fontinalis*, *Annamite Mountains*,
ecocore `species range`. `Langbian Plateau` is still not reached — the index
holds "Plateau du Lang Bian", so `Langbian` vs `Lang Bian` is a tokenisation
gap that needs a trigram index, not a prefix. 5 tests. **96 tests pass.**

## Clustering the unlinked mentions — reviewed and tightened (8 Sep 2026)

State before: 2,641 unresolved mentions -> 1,554 project-local entities, 1,065
of them singletons, 94 merging more than one surface form. The exact-key design
holds up; three candidate relaxations were measured against it.

**Taken — function words are dropped from the key.** The key is an unordered
multiset of lemmatised words, so a preposition, conjunction or infinitive
marker carries nothing that survives into it, yet it was still splitting
entities. Dropping `of in on for with to from by at and or` from anywhere in
the phrase (determiners were already stripped from the front) gives **15
merges, and all 15 were checked by hand and are genuine**:

    gradient of temperature / temperature gradient
    rates of diversification / diversification rates
    length from head to abdomen / head-to-abdomen length
    the number of / number      to conserve / conserve      in space / space
    a decrease in / decrease    centres of / centres        2070, / by 2070
    implementation of / implementation   importance of / importance
    majority of / majority      small patches of / small patches
    dates of ice-off / ice-off date      identify / to identify

Quantifiers still stay, for the reason already recorded. A span consisting of
nothing but function words keeps them, or every such span collapses onto one
empty key.

Result: 1,754 -> **1,738 entities** (200 grounded, 1,538 project-local).
Diffed against the previous run: 118 mentions changed `entity_id`, no outcome
changed, **nothing lost its grounding**, and one mention gained it — `east of`
now joins the grounded `east` at PATO:0045092, which is right. Note this
renumbers the affected ids; `entity_id` is content-derived, so changing the key
function is a one-time migration. Cheap now, expensive once anything consumes
the table.

**Rejected — an irregular-plural table merges nothing.** Morphy's blind spots
turn out not to bite clustering at all: `interglacials`, `biomarkers` have no
singular counterpart anywhere in the corpus, so there is nothing to merge them
with. They matter for LINKING, where the singular fallback already covers the
words Morphy does handle. (An earlier measurement appeared to show the entity
count going UP; that was a confounded replication of `content_key`, not a
finding.)

**Blocked — coref links, Phase 5 item 21.** The postprocessed gold carries only
`text`, `spans` and `relations`; there are no coref links in it. The only coref
available is model-generated under `cache/coref/`, and Apertus-70B invents
about 40% of its mentions. This is blocked on data, not on code.

### Noted, not fixed

A mention that reaches a grounded entity by CHAINING carries `entity_label:
null`, because the label is looked up in `clusters` and a chained row's
`entity_id` is the URI-named one. Cosmetic, pre-existing, and visible on the
`east of` row above.

**103 tests pass.**

## Coref measured — it belongs to the relation layer, not to linking (8 Sep 2026)

Correction to the entry above: the coref links DO exist, in
`data/aug_runs/{Mark,Davnah}_relation_input.json` (not in the postprocessed
gold, which carries only text/spans/relations). 52 per file, 104 total, from
`rcp-kimi-2.7` — not the Apertus model whose mentions are ~40% invented. Each
carries the anaphor with offsets, the antecedent text, its sentence and offset,
a kind (pronoun / demonstrative / definite_np) and a confidence.

**Alignment.** Only 6 of 104 anaphors match an annotated span exactly; allowing
a leading determiner ("the coniferous forest" vs the span "coniferous forest")
raises it to **43**. The other 61 are pronouns and demonstratives the
annotators did not mark as spans at all, which is correct for their schema.

**As a replacement text for matching: zero gain, six losses.** Resolving each
of the 43 through its antecedent instead of its surface form:

    nil -> nil 33 | linked -> nil 5 | linked -> linked 2
    nil -> ambiguous 2 | ambiguous -> nil 1        became linked: 0

Structural, not incidental: antecedents are long descriptive noun phrases
("areas of high species richness", "The Sierra Nevada of California",
"a violation of independence between data points") and long phrases are not in
ontologies. The antecedent is LESS matchable than the anaphor, which is the
same modifier-preserving problem from the other direction.

**As same-entity edges for clustering (Phase 5 item 21): 5 merges, 4 of them
forbidden.** Grouping the anaphors that share an antecedent:

    mountain decay      This process ABIOTIC PROCESS + process CONCEPT
    coniferous forest   habitat SPATIAL ENTITY + habitat BIOTIC ENTITY
    Sierra Nevada …     mountain range ABIOTIC + SPATIAL ENTITY
    the Great Escarp…   Escarpment + eastern Escarpment   (same label)
    a novel conservation index (CI)   CI + index           (same label)

The first three cross a LABEL, which full-label identity forbids — and the
second is the `habitat` case that decision was settled on by name. The fourth
crosses a modifier, which modifier-preserving granularity forbids. Only
`CI`/`index` is a legitimate merge, and it is an acronym pair, i.e. exactly the
thing a content key cannot catch and coref is not needed for either.

**Why this is systematic rather than unlucky.** Coref tracks DISCOURSE
REFERENTS; the identity model tracks CONCEPTS. "the habitat" and "coniferous
forest" are one referent and two concepts, and both statements are true at
their own layer. So feeding coref in as same-entity edges would have quietly
undone both identity decisions. Phase 5 item 21 should be considered answered
in the negative and closed.

**Where it does belong:** the relation layer. A relation whose argument is "the
habitat" needs to know which habitat, and that is a per-mention question about
a discourse referent — which is what coref actually provides. `match.py`'s
existing AMBIGUOUS reason ("its referent depends on coreference, which is not
resolved here") stays accurate as a statement about the mention; what it should
NOT imply is that resolving the coref would produce a link here. Measured: it
would not.
