# How a mention gets linked

The rules `src/linking` follows, in the order it applies them. Every one is
enforced in code and pinned by a test in `src/linking/tests/`.

**Scope: entities only.** Relations are not touched, not rewritten and not
consulted. They join back later on `uid`.

---

## The path a mention takes

```
one annotated span
  1. pick the name to look up      concept_text > surface form > singular
  2. normalise it                  
  3. pick the authorities          by the mention's own entity type
  4. look it up                    identity-bearing aliases only
  5. choose, or refuse             rank, then scope, then two safety rules
  6. give it an identity           by authority URI, or by its words
```

Steps 1–5 decide what a mention *is*. Step 6 decides which mentions are the
*same thing*. All of it is deterministic and offline.
---

## 1. Which name we look up

Recorded per mention in `concept_source`:

1. the **span as written**
2. the annotator's **`concept_text`**, if they gave one — only if 1 matched
   nothing
3. the span with its **head noun in the singular form** — only if 2 matched
   nothing too (`--no-singular` turns it off)

Each step is reached only when the one before it found nothing, and each accepts
**only a link**. So a later step can add grounding and never take an answer
away.

### Why the span goes first

`concept_text` is sound — of the 281 values in the human files, 5 are corrupt,
under 2%, and the 190 that look like junk are a single tool artefact (a bare
`x ` written in front of the value, which is stripped and the rest used). So
this ordering is not about distrusting the annotator.

It is about what the values mostly **are**: extensions of the span rather than
corrections of it. An extension names a *different entity* under §6, where
identity is modifier-preserving. Measured over both human files, preferring
`concept_text` grounded 5 mentions and cost 7, and four of the seven gave up a
link that is correct for the span the annotator actually marked:

```
biomass            -> trout biomass            gave up ENVO:01000155
water temperature  -> Peak water temperature    gave up ENVO:09200014
grassland          -> grassland remnants        gave up ENVO:01001206
insect             -> insect groups             gave up gbif:216
```

`trout biomass` is not ENVO's `biomass` and should not be linked to it, so there
is nothing to recover there — the span is simply the better thing to look up.
Going span-first keeps all five gains and none of the losses:

```
insignis            -> Tadarida insignis        gbif:4266607
frater              -> Myotis frater            gbif:2432428
javanicus           -> Pipistrellus javanicus   gbif:5218488
huttonii            -> Murina huttonii          gbif:8307118
introduced predator -> predator                 ECOCORE:00000089
```

**Nothing tests the value against the text.** Until 11 Sep a `concept_text` was
used only if every one of its tokens appeared in the sentence or the abstract.
Measured against the authorities, that requirement was anti-correlated with
usefulness: it accepted `Pi. javanicus`, `My. frater` and `Mu. huttonii` —
abbreviated genera that are in the text and ground to *nothing*, because GBIF
holds full binomials — and rejected `Tadarida insignis`, which appears nowhere
in the abstract and is `gbif:4266607`. A field that exists to supply what the
span does not say cannot be gated on the span's own words. Requiring a link
instead does the same job without that flaw: the corrupt values ground nowhere
either, so `Helarctos malayanus` keeps `gbif:2433403` and is never replaced by
`Helarctos Malayans`.

`concept_text_scope` is still recorded — `sentence`, `document`, or `external`
for a value the text does not contain — but only as provenance. It decides
nothing.

## 2. Normalisation

The same function runs over authority names when the dictionary is built and
over mentions when they are matched. If the two ever diverge, matching silently
under-fires.

1. Unify dash and quote variants — `Baetic–Rifian` = `Baetic-Rifian`
2. **Fold diacritics** — `Doñana` = `Donana`, `Phước Bình` = `Phuoc Binh`
3. Casefold
4. Strip punctuation at the **edges only** — `3,4-dichlorophenol` keeps its comma
5. Collapse whitespace


## 3. Which authorities may answer


| authority | answers for |
|---|---|
| GBIF | BIOTIC ENTITY only |
| GeoNames | SPATIAL ENTITY only |
| GO (biological process), ECOCORE | the types they declare |
| ENVO, PATO | **every type** |

Name authorities stay gated because they hold millions of proper names that
collide with ordinary English.
ENVO and PATO are open to every type because they hold concepts rather than
names, so a spurious match is rare and the rules below catch it.

A type no authority serves is `UNROUTED` — *not attempted*, which is not the
same as missed.

## 4. Which aliases count

Every name in the dictionary carries a scope, which says how that name relates
to the concept. Only some scopes mean "this name *is* the concept".

| scope | can a mention link to it? |
|---|---|
| `label`, `exact`, `vernacular` | yes |
| `narrow`, `broad`, `related` | no — they can only suggest candidates |

A `label` and its `exact` and `vernacular` synonyms all name the same thing, so
a mention that matches one can be linked to it. The other three scopes name
something *related* to the concept, not equal to it, which is why they are
excluded. Two real cases from ENVO:

- `dam` is a **broad** synonym of *beaver dam* (ENVO:00000500) — a dam is a
  wider thing than a beaver dam. Accept it and a mention of "dam" can come back
  as a beaver dam, which the author never said.
- `run` is a **narrow** synonym of *watercourse* (ENVO:00000029) — a run is one
  kind of watercourse. Accept it and a mention of "run" comes back as a
  watercourse, which is vaguer than what was written.

Either way the mention stops meaning what it said, and starts meaning either a
category it belongs to or one member of itself. That is the error this design
exists to avoid.

`dam` also shows the rule is safe and not merely strict. ENVO carries *dam* a
second time, as the label of ENVO:00000074, the general concept. With broad
synonyms excluded, `dam` matches only that label and lands on the general
concept, which is the right answer. `run` has no such label anywhere in ENVO, so
it stays NIL — correctly, because the ontology holds no concept for it.

This only works if the scopes themselves are right, and we cannot check that
here: a synonym filed under the wrong scope looks exactly like one filed
correctly, so there is nothing for a matching rule to test. Known mistakes are
therefore fixed when the dictionary is built, from a list in
`compile_db.OBO_SCOPE_CORRECTIONS`. The list is written into `build_meta`, and
any entry that stops changing anything is reported as stale. There is one entry
today: ENVO lists `mountains` as an exact synonym of *seamount*.

## 5. Choosing between candidates

1. Fetch **all** of them, up to 1000. The query has no ordering, so a full set
   is an arbitrary subset — hitting the cap is reported as AMBIGUOUS.
2. Collapse by URI, keeping the strongest scope.
3. **Two authorities answering → AMBIGUOUS.** There is no principled tiebreak
   between them.
4. **Prefer the highest taxonomic rank**, within one source:
   `kingdom < phylum < class < order < family < genus < species`. This settles
   the homonyms — `diptera` is *Saxifraga* (genus) or **Diptera** (order).
5. **Prefer the stronger scope**, `label > exact > vernacular` — but **not**
   between places. Two places are not a name and its synonym.
6. **A common noun must not bind to a named individual.** GeoNames holds
   millions of places whose names are ordinary English words, so a mention with
   no capitalised word anywhere in it is refused every GeoNames candidate.
   `landscape` is the real case: its one candidate is a settlement called
   *Landscape*, and the mention is refused. Such a mention is not unresolvable
   but *under-specified* — which landscape is a coreference question, not a
   lookup one.
7. **A generic mention must not resolve to one organism.** If the mention has
   any WordNet hyponyms *and* the candidate is species rank, refuse. Blocks
   `fish` → *Gadus morhua*; permits `insects` → Insecta and
   `annamite striped rabbit` → *Nesolagus timminsi*.
8. Anything still tied → **AMBIGUOUS**, handed up a tier rather than guessed.

### Rules 6 and 7 are vetoes, not routes

Candidates come from an **exact match on the whole mention**: some alias has to
equal the entire mention once normalised (`Dictionary.lookup` is a
`WHERE norm = ?`). There is no head-word or substring fallback, so a mention
never reaches a concept named by only part of it. Rules 6 and 7 throw candidates
away.

That is what separates these outcomes in the corpus, all of them SPATIAL ENTITY:

| mention | outcome | why |
|---|---|---|
| `southern Africa` | linked, `geonames:9406051` | GeoNames has a region actually called *Southern Africa*, and the whole string matched its label. It is **not** matching *Africa*. |
| `northern Brazil` | nil | GeoNames holds no name `northern Brazil`. Zero candidates, so rule 6 never runs. |
| `landscape` | ambiguous | one candidate, the settlement *Landscape*, refused by rule 6. |
| `lake` | ambiguous | refused by **rule 3**, not rule 6 — candidates in both ENVO and GeoNames. |

So passing rule 6 is necessary for a GeoNames link and never sufficient. Most
place mentions that fail do so for the ordinary reason: the name is not in the
gazetteer under that spelling. `Southeast Asia`, `Hindu Kush Himalaya`,
`northern South America` and `western coast of North America` are all NIL for
that reason, and `North America` and `Andes` are AMBIGUOUS because several
GeoNames entries tie.

### How WordNet is used

**1. Refusing to resolve a category to one organism**
GBIF mentions ordinary English words as vernacular (ordinary) names of particular species:
`fish` is listed against *Gadus morhua*, the Atlantic cod. The string match (fish) is
correct but the meaning is wrong, and no string evidence can tell the two apart.
`fish` → *Gadus morhua* and `annamite striped rabbit` → *Nesolagus timminsi* are
both single species-rank matches created through an ordinary name.

WordNet is asked how many kinds of the thing there are — the transitive hyponym count,
taking the largest noun sense:

```
mammals 1169   fish 617   insects 454   beetle 62   bats 40   trout 5
dung beetle 3   |   annamite striped rabbit 0   Nesolagus timminsi 0
```

The link is refused if both hold:
- the mention has >= 1 hyponyms, and
- the winner's rank is species, subspecies, variety, or form.
Outcome becomes AMBIGUOUS, and the hyponym count is written into the `reason`.


**2. Singularising nouns.** `morphy` gives the noun lemma — for every token
of the content key (§6), and for the head noun in the third lookup attempt (§1).
It handles irregular plurals a rule would miss (`taxa`, `larvae`, `genera`) and
misses others (`fungi` and `algae`). That is left alone: a
wrong lemma merges two entities, whereas an unmerged variant is recoverable.

Why it can't help, only refuse. A word WordNet doesn't know gets a count of 0, which means "not generic", which means no veto. So WordNet's coverage gaps — species richness, defaunation — can never cause a wrong answer; they just make the guard inapplicable. It's never consulted to find or confirm a concept.

## 6. When two mentions are one entity

> **Two mentions are one entity if they share a content key OR an authority
> URI, under the same label.** 

The **content key** is `(entity label, sorted multiset of lemmatised words)`.
Sorting handles word order, lemmatising handles plurals, normalisation handles
case and punctuation. Determiners and function words are dropped; quantifiers
are not, because `all species` is its own thing.

It is **modifier-preserving** — a modifier difference is a different entity:

```
bird species ≠ bat species ≠ species        temperature ≠ high temperatures
```

That is what makes clustering an exact-key problem rather than a similarity
one: the only merges left are lexical variants of a single phrase. Over-merging
is unrecoverable; under-merging is not.

Neither signal is enough alone, so the two are unioned as connected components:

| | content key | authority URI |
|---|---|---|
| `Vietnam` / `Viet Nam` | ✗ different words | ✓ same GeoNames ID |
| `Diptera` / `flies` | ✗ different words | ✓ same GBIF ID |
| `mountains` ×2, one grounded | ✓ same words | ✗ only one has an ID |

**Identity is the whole label.** `BIOTIC ENTITY` is one label, not a role plus a
domain — so `habitat` under ABIOTIC / BIOTIC / SPATIAL ENTITY is three entities,
and all three carry the same `authority_uri`, ENVO:01000739. 



**Unresolved mentions never participate.** Creating an entity for the eight uses
of "the lake" would assert they are one lake, which is the guess the matcher
refused.

## 7. Outcomes

| outcome | meaning | gets an `entity_id`? |
|---|---|---|
| `linked` | one concept survived every rule | yes, grounded |
| `nil` | routed, searched, nothing in any authority | yes, project-local |
| `unrouted` | no authority serves this type (unreachable today, §3) | yes, project-local |
| `ambiguous` | candidates found, deliberately declined | **no** |

`nil` and `unrouted` are kept apart because they call for different fixes — a
bigger vocabulary versus a routing decision. `ambiguous` calls for neither: it
needs sentence context or resolved coreference.

## 8. Invariants

1. Two mentions are the same entity **iff** they share an `entity_id`.
2. No entity spans two labels, and none spans two authority URIs.
3. Mention rows are never deleted or rewritten. A merge is two rows sharing an
   `entity_id`, so a bad merge is reversible.
4. Ambiguous mentions carry no `entity_id`, and always carry a `reason`.
5. `entity_id` is derived from content, so it is stable across runs and
   corpora. Adding a document renumbers nothing.
6. Same input + same artifacts → byte-identical output.
7. Rows are keyed by `mention_id`, not `uid`; `uid` is not unique, neither
   across annotator files nor within one.
