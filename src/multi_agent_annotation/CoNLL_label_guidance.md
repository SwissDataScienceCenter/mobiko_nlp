# CoNLL-2003 Named Entity Labelling Guidance

This is a working guideline for annotating named entities in English newswire under
the four CoNLL-2003 types: PER, ORG, LOC, MISC.

It exists because the crowd workers who produced this corpus had no published
guideline, and comparing a guideline-driven pipeline against them with no guidance
at all would test the wrong thing. Treat it as a minimal, public-knowledge
guideline: the conventions any competent annotator of this corpus would need,
written down.

Sources. The structure and the treatment of boundaries, metonymy and ambiguity
follow the publicly documented Universal NER conventions (universalner.org). The
type definitions and several boundary rules were then corrected against the
CoNLL-2003 gold annotations themselves, which differ from Universal NER on the
fourth type, on possessives, and on adjectives derived from names. Where the two
disagree, this document follows the corpus.

## How to use this guideline

Annotate only *named* entities — expressions that name a specific individual thing.
Common nouns describing a category are not entities, however important they seem.

Decide the type from the local sentence, not from the name in isolation. The same
string is often a different type in a different sentence, and that is expected
rather than an error.

If a span is genuinely ambiguous after reading the sentence, prefer the literal,
most concrete reading; if that does not settle it, prefer the reading that is more
common for that name in news text.

## PER — people

Tag names of people, real or fictional: given names, surnames, full names, and
initials used as names. A person referred to by surname alone is still PER.

Include the whole name as one span. Do not include a title or honorific even when
it sits directly before the name — in *President Lincoln*, tag only *Lincoln*; in
*Mr Grinch*, tag only *Grinch*.

A family or duo named as a unit is PER, such as *Brothers Grimm*.

Do not tag pronouns. Do not tag a role or occupation standing on its own —
*the governor*, *the striker*, *the spokesman* are not entities. Do not tag an
animal's name.

## ORG — organisations

Tag named groups of people acting as a single body: companies, government
departments and agencies, international bodies, political parties, universities,
news agencies, sports clubs, and national teams.

Include a corporate designator when it is part of the name, such as *Co* or *Ltd*.

Note that a company name used as the subject of corporate action is ORG. A brand
name used to refer to a product rather than the company is not ORG.

## LOC — places

Tag named geographic and physical places: countries, cities, states, regions,
rivers, seas, mountains, buildings, streets and addresses, and planets.

A dateline naming the city a report was filed from is LOC — for example the leading
*LONDON* or *PARIS* in a wire story.

Use LOC when the name refers to the place itself. Use ORG when it refers to a team
or a governing body — see the section on place names standing for organisations.

Do not extend the span to include a direction or modifier that is not part of the
proper name: in *northern France*, tag *France* only.

## MISC — everything else that is named

MISC is the residual type for named entities that are not a person, organisation or
place. In this corpus it is dominated by two groups.

First, adjectives and demonyms derived from place names: *German*, *Russian*,
*British*, *Iraqi*, *Israeli*, *Dutch*, *American*. These are tagged MISC even
though they are adjectives, and this is one of the most frequent MISC decisions in
the corpus.

Second, named events, competitions and historical periods: *World Cup*,
*World War II*, *Italian Cup*, *African Nations Cup*, *American League*.

MISC also covers named products, named laws, and languages.

## Span boundaries — how much to include

Tag the longest span that forms the complete name, and do not separately tag parts
inside it. For a name such as *University of Washington St. Louis*, the whole thing
is one ORG, not several nested entities.

Include numbers when they belong to the name, as in *10 Downing St*.

## Possessives

Do not include the possessive marker in the span. In *Chaplin's office*, the entity
is *Chaplin*; the *'s* is outside it. This corpus is consistent on this point —
possessive markers are essentially never inside a gold span.

When a possessor and the thing possessed are both named entities, tag them
separately.

## Punctuation, quotation marks and hyphens

Leave surrounding punctuation outside the span, including quotation marks around a
title and a dash separating two names. In a construction like *Oslo-Bergen*, tag the
two place names as two separate LOC spans and leave the dash out.

When a hyphen falls inside a single name, keep the name intact.

## Conjunctions and lists

Names joined by *and*, *or*, or commas are separate entities, one span each. Do not
merge a list of names into a single span, and do not include the conjunction.

## Place names standing for organisations

A place name is ORG, not LOC, when it stands for a team or an administrative body
rather than the territory.

In sports reporting this is the common case: in a match report or a league table,
city and country names refer to the competing sides and are ORG. The same applies
when a city name is the actor in a governmental decision.

Read the verb. A place that plays, wins, loses, signs, announces or rules is acting
as an organisation. A place that is travelled to, located in, or bordered by is
acting as a location.

## Resolving ambiguity

Work through these in order.

1. Read the rest of the sentence. Local context usually decides the type.
2. Prefer the literal meaning of the name over a figurative one.
3. If the sentence does not decide it, choose the most common use of that name in
   news text.
4. As a last check, consider what kind of thing a reference work would say this
   name denotes.

Ambiguity between a place and the organisation based there is the single most
common hard case, and it is genuinely uncertain in some sentences. Choose the
better-supported reading rather than leaving the span untagged.

## What not to tag

Do not tag pronouns, bare job titles, bare role descriptions, or generic common
nouns.

Do not tag a descriptive phrase merely because it contains a name, unless the name
itself is the entity — tag the name, not the surrounding phrase.

Do not invent spans to be thorough. An unnamed reference is not a named entity.

## Newswire conventions in this corpus

Text is tokenised, so punctuation stands as separate tokens. Judge spans by the
words, and do not let tokenisation spacing change what you consider part of a name.

Headlines and datelines are frequently in capitals. Capitalisation carries no extra
weight here — an all-caps city name is treated exactly as the same name in mixed
case would be.

Sports results, standings and scorecards are common. In these, competitor names are
ORG, and the surrounding numbers, scores and match notation are not entities.
