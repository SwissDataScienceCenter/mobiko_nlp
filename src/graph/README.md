# Knowledge graph from annotation runs

Turns the output of the annotation pipeline into a graph of linked entities and
loads it into Neo4j. Single-pass, review-pipeline and blind-pair runs all work,
because every one of them writes `final_entities` and `final_relations`, which
is all this reads.

```
run JSONL  ──►  annotations.json  ──►  resolution.jsonl  ──►  nodes/edges.jsonl  ──►  Neo4j
               spans + relations      mention -> entity       triplets, aggregated     graph.html
               by uid                 linked to ENVO, PATO,
                                      GO, GBIF, GeoNames
 scripts/agent_jsonl_to_      src.linking.resolve_corpus   src.graph.build_graph   src.graph.load_neo4j
 relation_input.py                                                                 src.graph.render_html
```

Each stage is one command with files between stages, so any stage can be rerun or
inspected on its own. Entity normalisation and linking are entirely
`src/linking` (see `plan.md` and `UNIFICATION_RULES.md` there); this package
only joins relations onto its output.

## Run it

```bash
make neo4j-up                       # once; Docker, data in data/neo4j, creds in .env
make kg RUNS=data/auto_annotated/second_iteration SOURCE_DIR=data/second_iteration_data
make kg-load RUNS=data/auto_annotated/second_iteration

# or start from an existing annotation file, e.g. one with coreference added
# by scripts/add_coref_annotations.py (this is how second_iteration is built):
make kg ANNOTATIONS=data/aug_runs/second_iteration_relation_input_coref.json KG_NAME=second_iteration
make kg-load KG_NAME=second_iteration
```

Output lands in `output/kg/<runs dir name>/`:

| file | what |
|---|---|
| `annotations.json` | spans and relations with uids, from the run |
| `resolution.jsonl`, `entities.tsv` | one row per mention / per entity, with authority links |
| `nodes.jsonl`, `edges.jsonl` | the graph; edges list the statements behind them, by sentence id |
| `sentences.sqlite` | the text, once: every sentence with its section, each node's mentions, each edge's evidence, each paper; keyword search (`text_store.py`) |
| `triplets.tsv` | one readable row per triplet, most-attested first |
| `graph.html` | offline viewer (no Neo4j needed): open it in a browser; ● node in one paper, ■ shared node across papers (any type, not the CONCEPT label), ◆ ontology term |
| `*.meta.json` | provenance and counts for each stage |

**Several runs in one graph** (for example, several models over the same
documents): convert each run to its own annotation file with a **distinct file
name**, then pass all of them to both later stages:

```bash
python -m src.linking.resolve_corpus --input out/kimi.json --input out/qwen.json --out out/resolution.jsonl
python -m src.graph.build_graph --annotations out/kimi.json out/qwen.json \
    --resolution out/resolution.jsonl --out-dir out
```

Entity ids are content-derived, so the same entity from two runs is one node,
and each edge lists the `annotators` that stated it. Two runs over the same
paper share its instances, because the instance key is the paper, not the run.
No node id contains the input file name: measurement and unresolved nodes are
`value:<doc_id>:<uid>#<n>` and `unresolved:<doc_id>:<uid>#<n>`, so renaming
or re-splitting the annotation files changes nothing in the graph.

**Ids as the corpus grows.** An entity is named by its smallest authority URI,
else by its words, and the name does not depend on the order of the inputs.
It still changes in two cases: a mention in a new paper links an entity that
had no authority before, or chains it to a second URI. Every id the entity
could have had is kept in `aliases` (`entity_aliases` in the resolution
table), so an id saved earlier can still be found:

```cypher
MATCH (e:Entity {graph:'second_iteration'}) WHERE $old_id IN e.aliases RETURN e
```

## The graph

```
(:KGNode:Entity:<Type>)       a concept, one per entity_id, e.g. :BioticEntity, :SpatialProperty
(:KGNode:Instance:<Type>)     that concept in one paper, and of one bearer where known
(:KGNode:Unresolved:<Type>)   an ambiguous mention used by a relation, one node per mention
(:KGNode:AmbiguousPlace:<Type>) a place name with several GeoNames referents, shared
(:KGNode:OntologyTerm)        one per authority URI (ENVO, PATO, GBIF, GeoNames, ...)
(a)-[:<RELATION> {n_evidence, docs, annotators, confidence_mean, sentence_ids, distributed}]->(b)
(:Instance)-[:INSTANCE_OF]->(:Entity)
(:Entity)-[:EXACT_MATCH]->(:OntologyTerm)
(:AmbiguousPlace)-[:MAY_BE {feature, docs_naming_candidate}]->(:Entity)
(:OntologyTerm)-[:WITHIN]->(:OntologyTerm)    a GMBA range inside its parent range
```

**Mountain ranges carry their GMBA hierarchy** (GMBA Mountain Inventory v2,
decided 3 Nov 2026). A mountain linked to GMBA, or a GeoNames mountain whose
name finds exactly one GMBA range, has an `EXACT_MATCH` to that range's term,
and the term is `WITHIN` its parent ranges up to the continent's mountain
region ("Africa (GMBA mountain region)", not the continent). Place authorities
answer in order GeoNames, then GMBA, then WDPA protected areas.

```cypher
MATCH (r:OntologyTerm {graph:'second_iteration', name:'Eastern Arc Mountains'})
      <-[:WITHIN*0..]-(t)<-[:EXACT_MATCH]-(e)
RETURN e.name                       // Udzungwa, Uluguru, Nguru, Taita, Usambara
```

**Ambiguous place names are meta-nodes** (decided 3 Nov 2026). Bare
"Kilimanjaro" ties between the mountain and the region, so it becomes one
shared node `place:kilimanjaro` with `MAY_BE` to each candidate; the mention's
relations attach to it. Nothing is claimed about which one a paper meant, and
a question still reaches every Kilimanjaro. A candidate no mention links to
gets an Entity of its own (`candidate_only`). Only proper names qualify:
lower-case or WordNet-class phrases ("lowlands", "National Park") stay
Unresolved. `resolve_corpus` writes the candidates (`candidates`, at most 20)
into each ambiguous row; `--no-ambiguous-places` turns this off.

```cypher
MATCH (p:AmbiguousPlace {graph:'second_iteration', name:'Kilimanjaro'})-[:MAY_BE]->(e)
OPTIONAL MATCH (e)-[r]-(x) WHERE NOT type(r) IN ['EXACT_MATCH','MAY_BE','INSTANCE_OF']
RETURN e.name, type(r), x.name
```

**Relations attach to instances, not concepts.** `elevation` in two papers is
the same concept (PATO:0001687) but not the same fact. With one shared node, a
correlation from one paper and a value from another would be joined into a path
that no paper states. So:

* Each concept gets **one instance per paper** (`elevation · PMC6920850`), with
  `INSTANCE_OF` to the shared concept. Cross-paper questions go through that
  edge.
* A property with **one bearer** in its sentence (`Udzungwa mountains HAS_PROPERTY
  altitude`) is further scoped by that bearer: `altitude of Udzungwa mountains ·
  PMC6920850`. With two or more bearers on one mention it stays at paper scope,
  and the count is reported. The relations that mark a bearer are set by
  `--bearer-relations` (default `HAS_PROPERTY`), since the relation guide is
  still changing.
* **Named places and taxa** (GeoNames, WDPA protected areas, GBIF) are not split: Kilimanjaro is the
  same mountain in every paper, so papers meet at that node.
* **Coreference** (when the annotation file has `coref` links): an anaphor that
  is an annotated span takes its antecedent's node, so "the mountain" attaches
  to *Mt Kenya* in that paper. Links are used only when both ends are annotated
  spans with the same role, the confidence is high, and the link is not
  contrastive ("other soil samples"), coordinated ("shrews and rodents") or a
  counted set ("two assemblages"). On second_iteration: 28 of 215 links, 64
  relation ends. Each redirect shows on the edge (`coref` in the evidence) and
  in the node's sentences. `--no-coref` and `--coref-min-confidence` control it.
* `build_graph --scope document` turns off bearer scoping. `--scope concept`
  gives the old concept-only graph.

**Values are measurements.** A property mention whose text is a value ("2743 m",
"6–10 µm", "25%", "eight", "p < .001", a coordinate) is a `:Measurement` node with
the number parsed (`src/graph/values.py`): `value` or `low`/`high`, `unit`,
`comparator`, `uncertainty`, `raw`. One per mention, never `INSTANCE_OF` a
shared "25%", so values do not bridge papers and can be compared:

```cypher
MATCH (x {graph:'second_iteration'})-[r]->(m:Measurement)
WHERE m.unit STARTS WITH 'm' AND coalesce(m.value, m.high) > 2000
RETURN x.name, m.raw, m.doc_id
```

Statements carry no p-values (agreed 3 Nov 2026): significance is in the
relation itself, since a result reported as not significant is annotated
`Negation` and becomes a `NOT_…` edge. The values stay in the evidence
sentences. `--no-measurements` restores values as ordinary nodes.

**Qualities are shared nodes.** A QUALITATIVE PROPERTY mention ("high",
"significant", "relatively constant") has no per-paper node: relations point
straight at the shared concept, which is labelled `:Quality`. Each edge keeps
its papers and sentences, so clicking it shows where the triplet appeared.
A quality that is itself a relation's subject keeps a per-paper node, so a
`:Quality` node only receives edges and never links two papers' facts.
`--no-shared-qualities` turns this off.

```cypher
MATCH (x {graph:'second_iteration'})-[r]->(q:Quality {name:'high'})
RETURN x.name, type(r), r.docs, r.sentence_ids
```

**Relation properties** (the annotation page's `property`: `Negation`,
`Negative_polarity`, `Exclusive`, `Dominance`, or custom) are part of an edge's
identity, so a statement and its denial are never one edge. A negated relation
gets its own type, `NOT_AFFECTS`, so `MATCH ()-[:AFFECTS]->()` cannot return a
finding the paper denies; the other properties are an edge attribute
(`r.property`). A negated `HAS_PROPERTY` is not used for bearer scoping. In
`graph.html` negated edges are red and dashed, the others are marked `(−)`,
`(only)` or `(mainly)`. second_iteration is built from
`data/aug_runs/second_iteration_relation_input_coref_props.json`.

**Text is outside the graph, in `sentences.sqlite`** (decided 3 Nov 2026).
Neo4j is slow with long strings, and the graph used to copy each sentence onto
every node and edge it touched (13 times over on second_iteration), with
shared concepts collecting every paper's sentences in one property. Now each
sentence is stored once, keyed `<doc_id>/<sent_index>` (`PMC6920850#00/37`);
edges carry `sentence_ids`, and the store says which node each mention is
drawn as. Clicking a node in `graph.html` still shows
`PMC6920850 s37: … found at moderate «elevations» in …`, a concept every
paper's mentions; a mention the annotator inferred rather than read is flagged
`[implied: elevation]`.

```python
from src.graph.text_store import TextStore
st = TextStore("output/kg/second_iteration/sentences.sqlite")
st.contexts("<node id>")                 # a node's sentences, marked
st.contexts("<entity id>", concept=True) # a concept's, in every paper
st.evidence("<edge id>")                 # the statements behind an edge
st.search("pollination")                 # keyword search (SQLite FTS5)
```

For RAG the sentence id is the join: a Cypher result gives edge
`sentence_ids` and node ids to look up here, and a text hit gives a sentence
id that `evidence` and `mention` lead back into the graph.

Unchanged from linking (`UNIFICATION_RULES.md`): entity identity is label plus
concept, so `habitat` as ABIOTIC ENTITY and as BIOTIC ENTITY are two concepts
linked to one ENVO term. An ambiguous mention (`the lake`) keeps its triplet
without a shared identity. A relation on a coordination ("eastern and western
slopes") is applied to each part (`distributed: true`). Every node has a `graph`
property, and a load touches only that graph.

**Every statement knows its paper, journal, year and section** (decided 9 Nov
2026). `build_graph --papers <SOURCE_DIR>` (the Makefile passes `SOURCE_DIR`)
reads the fetch script's `manifest.csv` and BioC records in `raw/`
(`papers.py`). Each paper is a `Paper` node (citation, title, journal, year,
DOI, PMID). Each statement in an edge's `evidence` has `section` (Abstract or
Results) and `subsection` (the Results heading, e.g. "Microbial communities").
Each edge has `papers`, `years`, `journals` and `sections`; in Neo4j also
`evidence_sections`, aligned with `evidence_docs`. A sentence whose text no
longer matches the fetched file gets no section rather than a wrong one. In
`sentences.sqlite`, `sentence` has `section` and `subsection`, and a `paper`
table joins on the pmcid; `TextStore.evidence()` returns all of it.

```cypher
// AFFECTS statements from Results sections of papers published since 2020
MATCH (a:KGNode {graph:'second_iteration'})-[r:AFFECTS]->(b)
UNWIND r.papers AS pid MATCH (p:Paper {graph:'second_iteration', pmcid: pid})
WHERE p.year >= 2020 AND 'Results' IN r.sections
RETURN a.name, b.name, p.citation, p.journal, r.evidence_sections
```

**`make kg-load` syncs rather than reloads** (3 Nov 2026). It compares content
hashes (`sync_hash`) with what Neo4j holds, writes only new and changed nodes
and edges, then deletes what is gone. The graph stays whole while it runs, an
unchanged graph writes nothing (0.7 s here, 9 s for an 85k-node test), and a
load that fails half way is finished by running it again. `--full` empties
the graph first, as before.

Useful queries in the Neo4j Browser (http://localhost:7474):

```cypher
// best-attested triplets
MATCH p=(a:KGNode {graph:'second_iteration'})-[r]->(b)
WHERE NOT type(r) IN ['INSTANCE_OF','EXACT_MATCH'] AND r.n_evidence >= 2 RETURN p

// one paper
MATCH p=(a:KGNode {graph:'second_iteration'})-[r]->(b)
WHERE NOT type(r) IN ['INSTANCE_OF','EXACT_MATCH'] AND 'PMC6920850#00' IN r.docs RETURN p

// a concept across papers: every paper's elevation and what it relates to
MATCH (c:Entity {graph:'second_iteration', name:'elevation'})<-[:INSTANCE_OF]-(i)
MATCH p=(c)<-[:INSTANCE_OF]-(i)-[r]-(b) WHERE type(r) <> 'INSTANCE_OF' RETURN p

// concepts that share an ontology term across labels
MATCH (e:Entity)-[:EXACT_MATCH]->(t:OntologyTerm)
WITH t, collect(e.name + ' [' + e.entity_type + ']') AS ents WHERE size(ents) > 1
RETURN t.curie, t.name, ents ORDER BY size(ents) DESC
```

## Not done here

Relations are used as annotated. The relation-level work deferred in
`src/linking/plan.md` is still open: merging inverse directions, off-schema
labels (`OCCURS_IN`, `RELATED_TO`, …), and RDF export against
`MoBiKo_rdf_schema_v1`.
