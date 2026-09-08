# Entity unification rules

Reference for `src/linking`. Every rule below is enforced in code and pinned by a
test in `src/linking/tests/`.

**Scope: entities only.** Relations are not touched, not rewritten, and not
consulted. They join back later on `uid`.

---

## 0. Vocabulary

| term | meaning |
|---|---|
| **mention** | one annotated span in one sentence |
| **entity** | the thing a mention denotes; identified by `entity_id` |
| **grounded** | the entity carries an external authority ID (GBIF / ENVO / PATO / GO / ECOCORE / GeoNames) |
| **project-local** | no authority has this concept, so an ID is minted: `mobiko:e<hash>` |
| **unresolved** | candidates existed and the pipeline deliberately declined to choose; no `entity_id` |

**The contract:** two mentions denote the same entity **if and only if** they
carry the same `entity_id`. It is an equality test, never a similarity score.

---

## 1. Normalisation

Applied identically to authority names at build time and to mentions at match
time. Divergence between the two is silent, so `NORMALIZER_VERSION` (currently
**2**) is stamped into every artifact and checked at load; a mismatch is a loud
failure, not a lower link rate.

1. NFKC; unify dash and quote variants (`Baetic–Rifian` = `Baetic-Rifian`).
2. **Fold diacritics** — `Doñana` = `Donana`, `Białowieża` = `Bialowieza`,
   `Phước Bình` = `Phuoc Binh`. Letters NFKD does not decompose are mapped
   explicitly: `ł ø đ ð þ ß æ œ ı`.
3. Casefold.
4. Strip punctuation at the **edges only**. Interior punctuation is
   load-bearing: `baetic-rifian range`, `3,4-dichlorophenol`.
5. Collapse whitespace.

The unfolded form is never destroyed — it is kept as the display label and as
the stored alias.

---

## 2. Canonical form of a mention

Precedence, recorded per mention in `concept_source`:

1. **`concept_text`**, when present *and* validated
2. the **surface form**
3. the surface form with its **head noun singularised**, reached only when 2
   matched nothing (`--no-singular` turns it off)

`concept_text` is optional and validated, never trusted:

- **Token containment**, not substring. `annual rainfall` → `high annual
  rainfall` is valid because `high` distributes across a conjunction; the phrase
  is not contiguous in the sentence but every token is present.
- Tokens are trimmed of punctuation **individually**. Otherwise a sentence
  ending `…topographic relief.` yields `relief.` and rejects a valid value.
- **Sentence scope first, then document scope.** Document scope exists for the
  highest-value corrections: annotators expand abbreviated binomials, `frater` →
  *Myotis frater*, using a genus named earlier in the abstract. A full binomial
  is matchable in GBIF; a bare epithet is not.
- Values supported by neither scope are **rejected and counted**, never silently
  used. In the current gold that rejects 190 tool placeholders (`x three`,
  `x phylogenetic`) and 5 genuine corruptions (`Helarctos Malayans`).

---

## 3. Identity is the FULL label; the ontology link is what is shared

The label schema has two layers (`MoBiKo_label_guidance_v4`, Step 2):

- **role** — ENTITY, PROCESS, PROPERTY, CONCEPT
- **domain** — ABIOTIC, BIOTIC, ANTHROPOGENIC, SPATIAL, TEMPORAL, QUANTITATIVE, QUALITATIVE

`BIOTIC ENTITY` is the pair. The role is the **last** word; everything before it
is the domain. A label whose last word is not a known role is kept whole as the
role rather than guessed at (`NEGATION`, `QUANTITATIVE CHANGE`).

**Identity keys on the pair.** Every distinct label gets its own node, and what
those nodes share is stated explicitly rather than by merging them:

```
entity_id       ABIOTIC ENTITY@…/ENVO_01000739
                BIOTIC  ENTITY@…/ENVO_01000739
                SPATIAL ENTITY@…/ENVO_01000739
authority_uri   http://purl.obolibrary.org/obo/ENVO_01000739   ← all three
```

In RDF: three nodes, each `skos:exactMatch` to one ENVO term. Both directions
stay recoverable — a consumer wanting the labels apart has them, one wanting
them together groups on `authority_uri`.

`biodiversity` as ENTITY / PROPERTY / CONCEPT is likewise three nodes, and those
share no authority because none covers the concept.

**No entity ever spans two labels.** The role and domain are still computed and
exposed as `entity_role` and `entity_domain` — the distinction is real and
analyses want it — they just do not decide identity.

> Superseded: an earlier key used the **role alone**, merging the three habitat
> nodes into one carrying three labels. It merged 213 mentions the domain
> distinguishes, and a merge is not recoverable afterwards, whereas separate
> nodes sharing an `authority_uri` are recoverable in both directions.

**Grouping handles.** `authority_uri` groups the labels of one grounded concept
(10 such groups in the gold). `surface_key` — the content words with the label
stripped — does the same for concepts no authority covers (175 groups),
so `species` under three labels can be gathered without being merged.

---

## 4. Content key

```
key = ( full entity label , sorted multiset of lemmatised content words )
```

- Sorting handles word order: `broadleaf evergreen forest` =
  `evergreen broadleaf forests`.
- Lemmatising handles plurals (WordNet Morphy): `populations` = `population`.
- Normalisation handles case, dashes, diacritics, punctuation.
- **Determiners are stripped** (`the a an this that these those`): `the lake` =
  `lake`.
- **Quantifiers are NOT stripped** (`all many several`): the schema types them
  as QUANTITATIVE PROPERTY in their own right, so `all species` ≠ `species`.

**Modifier-preserving.** A modifier difference is a different entity:

```
bird species  ≠  bat species  ≠  species
temperature   ≠  high temperatures  ≠  peak water temperature
forest        ≠  coniferous forest
```

This is why clustering needs no threshold, no embeddings and no similarity
measure: the only merges left are lexical variants of one phrase, which is an
exact-key problem. Over-merging is unrecoverable; under-merging is not.

---

## 5. Grounding

### 5.1 Routing

Route on the mention's **own** entity type. Never a union of the types a surface
form was given elsewhere — unioning leaks across ontological roles
(`location` as SPATIAL ENTITY reaching a PATO *property*), affecting 135
mentions of the gold.

**Two kinds of authority, and only one is gated.**

| | gated to `serves` | answers for every type |
|---|---|---|
| GBIF, GeoNames | **yes** | — |
| ENVO, PATO | — | **yes** (`universal=True`) |

Name authorities stay gated because they hold millions of proper names that
collide with ordinary English: ungated, `data` and `areas` reach genera,
`extinct` reaches the Labrador Duck. Concept ontologies are opened to every
type because a spurious match is rare and the ambiguity rules catch it — it
gained 51 links, and took `unrouted` from 647 mentions to 0.

A type no authority serves is `UNROUTED` — **not attempted**, not a miss.
While ENVO and PATO are universal this outcome is **unreachable**, since every
type reaches them; it is kept because gating them again would make it reachable
at once, and the two must never be reported as the same thing.
(Decision 8 Sep, superseding an earlier reading that treated any ontology hit
outside `serves` as a leak.)

### 5.2 Which alias scopes assert identity

| scope | identity? |
|---|---|
| `label`, `exact`, `vernacular` | **yes** |
| `narrow`, `broad`, `related` | **no** — candidates only |

A BROAD synonym names a concept the term is *subsumed by*. Treating it as equal
is the hypernym merge this design exists to prevent. Obsolete concepts are
excluded from matching but retained in the dictionary.

Because everything above rests on the scope being right, a scope that is wrong
*upstream* cannot be caught by any matching rule — a mis-scoped synonym is
string-identical to a correct one. Known errors are therefore repaired in the
artifact at build time, by an explicit table
(`compile_db.OBO_SCOPE_CORRECTIONS`) that names the exact rows it changes and
is recorded in `build_meta`; corrections that stop firing are reported as stale
on the next build. One entry today: ENVO ships `mountains` as an EXACT synonym
of *seamount* and only a RELATED synonym of *mountain*, which sent all 14
mentions of it to an undersea mountain.

### 5.3 Decision order

1. Fetch **all** candidates, up to `CANDIDATE_CAP` (1000). The SQL has no
   `ORDER BY`, so a saturated set is an *arbitrary* subset — hitting the cap is
   reported as AMBIGUOUS, never decided on.
2. Collapse candidates by URI, keeping the strongest scope.
3. **Two authorities answering → AMBIGUOUS.** No principled tiebreak exists;
   rank is defined only inside GBIF, so comparing across sources ranks every
   taxon above every ontology concept. (`biota` → *Platycladus* was this bug.)
4. **Prefer the highest taxonomic rank**, within one source:
   `kingdom < phylum < class < order < family < genus < species < subspecies`.
   Settles homonyms: `diptera` = *Saxifraga*(genus) vs **Diptera**(order);
   `bacteria` = genus vs **kingdom**.
5. **Prefer the stronger scope** — `label > exact > vernacular` — but **not for
   sources naming individuals** (GeoNames). Two places are not a name and its
   synonym; that tiebreak had been picking an obscure feature named "Lake" over
   every "Lake County".
6. **A common noun must not bind to a named individual.** For GeoNames the
   mention must carry a capitalised token — any token, not the first, so
   `southern Africa` passes on *Africa* while `landscape`, `lake` and
   `mountain range` do not. Such a mention is not unresolvable but
   *under-specified*: its referent is a coreference question.
7. **A generic mention must not resolve to one organism.** If the mention has
   any WordNet hyponyms *and* the candidate is species/subspecies rank, refuse.
   Blocks `fish` → *Gadus morhua*; permits `insects` → Insecta (class rank) and
   `annamite striped rabbit` → *Nesolagus timminsi* (no hyponyms).
8. Anything still tied → **AMBIGUOUS**.

WordNet is used for **negative evidence and morphology only**. Never a positive
link; absence from WordNet is never evidence of anything.

---

## 6. Unification

> **Two mentions are one entity if they share a content key OR an authority URI,
> under the same label.**

Each signal alone loses what the other keeps:

| | content key | authority URI |
|---|---|---|
| `Vietnam` / `Viet Nam` | ✗ different words | ✓ same GeoNames ID |
| `Diptera` / `flies` | ✗ different words | ✓ same GBIF ID |
| `mountains` ×2 same label, one grounded | ✓ same words | ✗ only one has an ID |

Both signatures carry the **full label**, so joining never crosses one — that is
what keeps the three habitat nodes apart despite their shared ENVO id. Within a
label the join chains: two mentions of `mountains` as ABIOTIC ENTITY, only one
of which grounded, are one entity and the ungrounded one inherits the ENVO id.

**Naming a component:** its authority URI where it has one
(`ENTITY@https://www.gbif.org/species/811`), otherwise the content-key hash
(`mobiko:e<sha256[:12]>`).

**Unresolved mentions never participate.** Minting an entity for the eight uses
of "the lake" would assert they are one lake — the guess the matcher refused.

---

## 7. Invariants

1. No entity spans two labels.
2. Labels of one ontological concept are joined by a shared `authority_uri`;
   ungrounded ones by a shared `surface_key`. Neither merges the nodes.
3. Two mentions are the same entity **iff** they share an `entity_id`.
4. Mention rows are never deleted or rewritten. A merge is two rows sharing an
   `entity_id`, so it is reversible.
5. Ambiguous mentions carry no `entity_id`, and always carry a `reason`.
6. `entity_id` is content-derived, hence **stable across runs and corpora**.
   Adding a document never renumbers anything.
7. **Determinism**: same input + same artifact version → byte-identical output.
8. No entity spans more than one authority URI.
9. Rows are keyed by `mention_id`, not `uid` — `uid` is not unique, neither
   across annotator files nor within one.

---

## 8. Outcomes

| outcome | meaning | gets an `entity_id`? |
|---|---|---|
| `linked` | one authority concept survived every rule | yes, grounded |
| `nil` | routed, searched, nothing in any authority | yes, project-local |
| `unrouted` | no authority serves this entity type — unreachable while the ontologies are universal, see §5.1 | yes, project-local |
| `ambiguous` | candidates found, deliberately declined | **no** |

`nil` and `unrouted` are kept apart because they call for different fixes: a
bigger vocabulary versus a routing decision. `ambiguous` calls for neither — it
needs sentence context or resolved coreference.
