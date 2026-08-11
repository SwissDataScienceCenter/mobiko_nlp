"""Entity schema for CoNLL-2003 NER (the MTurk re-annotation by Rodrigues et al.).

Four types, mirroring the shape of entity_schema.SCHEMA_BIODIV_SHORT / _LIST so the
same MultiAgentAnnotator constructor arguments work unchanged:

    from src.resources_updated.entity_schema_conll import (
        SCHEMA_CONLL_SHORT, SCHEMA_CONLL_LIST)

DEFINITIONS FOLLOW THE CORPUS, NOT A GENERIC NER STANDARD. They were checked
against all 9,907 gold spans in ground_truth.txt, because the obvious external
reference (the Universal NER guidelines, universalner.org) disagrees with
CoNLL-2003 on points that matter:

  * UNER's fourth type is OTH and covers nationalities, languages and product
    brands. CoNLL's MISC covers nationality ADJECTIVES (German x39, Russian x39,
    British x31 in gold) and also named events and competitions (World Cup,
    World War II, African Nations Cup) which OTH does not.
  * UNER includes the possessive marker in the span. CoNLL excludes it — exactly
    2 of 9,907 gold spans end in one.
  * UNER says not to tag adjectives derived from names. CoNLL tags precisely
    those as MISC.

Encoding a guideline that contradicts the corpus would depress agreement for
reasons unrelated to model ability, so these definitions are corpus-first.
"""

SCHEMA_CONLL_SHORT = """
PER (DEFINITION: A named person, real or fictional — first names, surnames, full names, initials. Includes people referred to by a single name (Clinton, Arafat) and groups named as a family (Brothers Grimm). EXCLUDES pronouns, job titles standing alone (the governor, the president), and honorifics attached to a name — tag "President Lincoln" as Lincoln only.)

ORG (DEFINITION: A named collection of people acting as a unit: companies, agencies, institutions, political parties, sports clubs and national teams, news agencies. NOTE that a place name is ORG, not LOC, when it stands for a team or an administrative body acting as an organisation — in sports reports and standings, city and country names such as BALTIMORE or CHICAGO are ORG.)

LOC (DEFINITION: A named geographic or physical place: countries, cities, regions, rivers, mountains, buildings, addresses, planets. Report datelines naming the city of origin (LONDON, PARIS) are LOC. Use LOC when the name denotes the place itself rather than an organisation based there.)

MISC (DEFINITION: A named entity that is none of the above. In this corpus that is chiefly (a) adjectives and demonyms derived from place names — German, Russian, British, Iraqi, Israeli — and (b) named events, competitions and eras — World Cup, World War II, Italian Cup, African Nations Cup. Also covers named products, laws and languages.)
"""

SCHEMA_CONLL_LIST = [
    "PER",
    "ORG",
    "LOC",
    "MISC",
]

# No relations are annotated in this corpus. Passed where the pipeline expects a
# relation schema, so that schema_lookup answers "nothing is valid" rather than
# offering MoBiKo's biodiversity relations on newswire text.
SCHEMA_CONLL_RELATIONS: dict = {}
