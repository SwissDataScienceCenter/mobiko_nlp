# MoBiKo – Guidance for Span Labelling

The workflow consists of two steps: **(1) identifying a span** and **(2) labelling it**.

---

## Preliminary Comments

- Before identifying and labeling spans, make sure the sentence contains relevant information about mountain biodiversity, i.e. contents that tell something about mountain species biodiversity and ecosystems, their status, trends, and drivers, about their conservation, protection, restoration, and management as well as about the ecosystem services or the nature’s contributions to people they support or deliver.  
    
- Ignore sentences consisting of:  
    
  - Information about **methods and analyses** (e.g. *The analysis was run using the R packages igraph (Csardi and Nepusz 2006\) and gplots (Warnes et al 2019); PCR amplification of the DNA (MrDNA, Inc., Houston, Texas) used the 16S rRNA gene V4 variable region PCR primers 515 and 806*), including about sampling designs and protocols (*Eighty 1 m  1 m quadrats were placed and photographed along the transects on striped periglacial patterned ground*), unless this information specifies the location of the study.  
  - **General statements**, not specific for a particular study and not bringing new/important information (e.g. *Assessing biodiversity loss and species extinction is necessary to warn society and raise awareness of the impacts of ongoing climate change*).  
  - Statements that bring no new information and only refer to already published studies  
  - **Hypotheses**.


- Ignore spans that are only methods-related in any sentence (e.g. linear regression, binomial distribution, PCR analysis, etc.).  
- If the sentence contains facts/results of the analysis AND the method parts (i.e., which tools were used or details of the analysis process, methods, etc) \--- label the factual/result part and ignore the method.

---

## Step 1: Identifying Spans


Entity corresponds to a noun phrase (\[mountain biodiversity\], \[habitat quality\], \[species richness\], \[relict species\], \[biome affiliation\]) without any function words (articles: “**the** province”, prepositions: “province **of** Parinacota”, conjunctions: “trees **and** mountains”). Any noun/adjective that is related to the main noun of the span can be connected with “HAS\_PROPERTY” relation (“species richness” → “species HAS\_PROPERTY richness”).

Here we don’t need to unroll:  
“**Green and tall** trees” → **tree** HAS\_PROPERTY green   
			    **tree** HAS\_PROPERTY tall  
“antibacterial and antifungal properties” 

Where to unroll:   
“Disappearing birds and insects” → **birds** HAS\_PROPERTY disappearing  
				          **insects** HAS\_PROPRETY disappearing  
\---\> We have 2 core nouns (subjects) with the same property. We unroll them:  
“Disappearing birds”  
“Disappearing insects”

---

## Step 2: Labelling Spans

Labeling follows a classification consisting of two logical layers:

- **Ontological role:** entity; process; property; concept  
- **Domain:** abiotic; biotic; anthropogenic; spatial; temporal; quantitative; qualitative

Except for concepts, which are not associated with a domain, all other ontological roles are systematically associated with a domain.

### Definitions

#### Ontological Roles

| Label | Definition |
| :---- | :---- |
| **Entity** | Aggregation or assemblage of entities functioning as a unit, OR something that exists as itself — an individual object or component of the environment. |
| **Process** | Change in the environment; a noun or verb referring to a change, event, or action unfolding over time. *Q: Is this something that happens?* |
| **Property** | State, measurement, or characteristic. *Q: Is this something that something has?* |
| **Concept** | Abstract or theoretical construct used in analysis or discourse (e.g. climate, conservation status, resilience, scenario, vulnerability, link, trend, ecosystem service). |

#### Domains

| Label | Definition |
| :---- | :---- |
| **Abiotic** | Refers to any ecosystem factor, attribute, component, or constituent that is devoid of life; without life. |
| **Biotic** | Refers to any living factor, attribute, component, or constituent; with life. |
| **Anthropogenic** | Caused or related to humans or their activities. |
| **Spatial** | Related to space. |
| **Temporal** | Related to time. |
| **Quantitative** | Something you can count: how much, how many, few, many, numerous, first, last, second, etc. |
| **Qualitative** | The kind (e.g. big, bright, etc.) — involving quality or kind or feature (not related to time, space or counting or humans or purely biotic/abiotic properties, e.g., “antibacterial”, “parasitic”. |

---

## Handling Difficult or Ambiguous Cases

### 1\. Properties — Look at what the property belongs to

Use the **modified noun** to make a decision.

**Example: *density***

- **population** density → `BIOTIC_PROPERTY`  
- **soil** density → `ABIOTIC_PROPERTY`

**Example: *distribution***

- **species** distribution → `BIOTIC_PROPERTY`  
- **temperature** distribution → `ABIOTIC_PROPERTY`  
- **normal** distribution → `CONCEPT`

**Example: *quality***

- **habitat** quality → `SPATIAL_PROPERTY or BIOTIC_PROPERTY`

**Common ecological measurement terms and system-level ecological** are almost always properties (property type depends on the entity):

- abundance  
- biomass  
- richness  
- productivity  
- stability  
- structure

### 2\. Conjunctions and Disjunctions (AND/OR)

When multiple properties, entities, or processes are connected with AND/OR, unfold them into separate spans.

**Example: *Antibacterial and antifungal properties***

- Antibacterial properties → `BIOTIC_PROPERTY`  
- Antifungal properties → `BIOTIC_PROPERTY`

---

### 3\. Ontological Role Differs by Context

**Example: *rainfall***

- *Heavy **rainfall** caused erosion* → `ABIOTIC_PROCESS`  
- *…receives 100 mm of **rainfall*** → `ABIOTIC_ENTITY`

---

### 4\. Species and Taxonomic Groups

#### 4.1 Taxonomic groups used generically → biotic collective entities

*We failed to detect large and medium size **canids** and **felids***

- canids → `BIOTIC_ENTITY`  
- felids → `BIOTIC_ENTITY`

*Mountains are key features of the Earth's surface and host a substantial proportion of the world's **species***.

- species → `BIOTIC_ENTITY`

*Specifically, we analyse how erosion, relief, soil and climate relate to the geographical distribution of terrestrial **tetrapods**, which include **amphibians**, **birds** and **mammals**.*

- tetrapods → `BIOTIC_ENTITY`  
- amphibians → `BIOTIC_ENTITY`  
- birds → `BIOTIC_ENTITY`  
- mammals → `BIOTIC_ENTITY`

*Over millions of years, these processes generally lead to a concentration of **species** at low to middle elevations…*

- species → `BIOTIC_ENTITY` (generic use of the term)

*Many **species** go extinct, in particular those that are ecological specialists or confined to particular montane habitats.*

- species → `BIOTIC_ENTITY` (generic use of the term)

*While our work confirms prior findings that **predator presence** drives strong reductions in **insect emergence**…*

- predator → `BIOTIC_ENTITY` (generic use of the term)  
- insect → `BIOTIC_ENTITY` (generic use of the term)

*The vast majority (96.6%) of **insects** collected from emergence traps…*

- insects → `BIOTIC_ENTITY` (generic use of the term)

#### 4.2 Taxonomic names → biotic entities

- Vulpes vulpes → `BIOTIC_ENTITY`

#### 4.3 Taxonomic groups that can be enumerated individually → biotic entities

*We also recorded a number of large **mammals** that are rare in the region.*

- mammals → `BIOTIC_ENTITY` ("a number of" helps decide)

*We recorded several threatened and endemic **species**.*

- species → `BIOTIC_ENTITY` ("several" helps decide)

*We recorded at least 46 ground-dwelling **mammal** and **bird species**.*

- mammal → `BIOTIC_ENTITY` (bound number of individually identifiable)  
- bird species → `BIOTIC_ENTITY` (same reasoning)

*The formation of mountains drastically transforms previously homogenous landscapes, often characterized by having mature soils, low erosion rates, old **relict species** and low speciation rates.*

- relict species → `BIOTIC_ENTITY` (limited, enumerable set)

*…facilitate the establishment of immigrant lineages from surrounding lowlands (for example, **flying birds** or **bats** and the seeds they carry)…*

- flying birds → `BIOTIC_ENTITY` (subset of birds that could be listed)

*The vast majority (96.6%) of insects collected from emergence traps were Diptera (**flies**).*

- flies → `BIOTIC_ENTITY` (clarified by Diptera)

*Following the IPBES framework, each abstract was tagged for information on biodiversity (5 species groups)*

- species → `BIOTIC_ENTITY` (enumeration: 5 groups of given species)

*We developed a novel conservation index (CI) to prioritize areas and populations of an endangered mountain **tree species** that need protection…*

- species → `BIOTIC_ENTITY` (one unique species)

*Prioritizing protected areas… to conserve **species at risk** of extinction.*

- species → `BIOTIC_ENTITY` (specific set of species)

---

### 5\. Spatial vs. Abiotic Entity

The key question: **can you point to this on a map?**

- **Mountain glaciers** → `ABIOTIC_ENTITY` (cannot be located on a map as a distinct place)  
- **Aletsch glaciers** → `SPATIAL_ENTITY` (can be identified and located on a map)

SPATIAL\_ENTITY always gets ABIOTIC\_ENTITY type by default (as a subset of ABIOTIC). SPATIAL is labeled only if the entity can be located on a map.

---

### 9\. Polysemic Terms (context-dependent categories)

#### *Biodiversity*

| Context | Label |
| :---- | :---- |
| *…safeguard their inhabitants, their ecosystems, their **biodiversity**, and the livelihoods they support* | `BIOTIC_ENTITY` |
| *…tepuis are table-top mountains with elevations above 1000 m and high **biodiversity** and endemism levels* | `BIOTIC_PROPERTY` |

#### *Human wellbeing*

| Context | Label |
| :---- | :---- |
| *…to assess and compare the contents of 631 abstracts on the interactions among biodiversity, ecosystem services, **human wellbeing**, and drivers of change…* | `ANTHROPOGENIC_PROPERTY` |

#### *Mountain / montane / alpine*

| Term | Context | Label |
| :---- | :---- | :---- |
| mountain | *…using the find function and the keywords "mountain," "montane," and "alpine"…* | `ABIOTIC_ENTITY` |
| mountains | *Mountains are facing growing environmental, social, and economic challenges* | `ABIOTIC_ENTITY` |
| montane, alpine | *…using the find function and the keywords "mountain," "montane," and "alpine"…* | `ABIOTIC_PROPERTY` |
| mountain | *…prioritize areas and populations of an endangered **mountain** tree species…* | `ABIOTIC_PROPERTY (not labeled separately if not split)` |
| alpine | Biotic modifier | `BIOTIC_PROPERTY` |

#### *Elevation*

| Context | Label |
| :---- | :---- |
| *…we assessed the effect of **elevation** on dung beetle assemblage structure…* | `SPATIAL_PROPERTY` |

#### *Area*

| Context | Label |
| :---- | :---- |
| *…populations located in rugged **areas*** | `ABIOTIC_ENTITY` |
| Used as a measure | `SPATIAL_PROPERTY` |

#### *Geology*

| Context | Label |
| :---- | :---- |
| *The **geology** of the Alps has changed over the centuries.* | `ABIOTIC_PROPERTY` |
| *In **geology**, bird breeding behaviors are not studied systematically* | `CONCEPT` |

#### *Grazing*

| Context | Label |
| :---- | :---- |
| *Grazing by livestock is causing land degradation* | `BIOTIC_PROCESS` |
| *Wildlife grazing has increased over the last decades, causing land degradation* | `BIOTIC_PROCESS` |

---

### 10\. Typical Difficult Cases

#### Case A

*We investigated patterns of speciation and micro-endemism from modeled past, present, and future distributions in six clades of **southern African bats** from three families (**Rhinolophidae, Cistugidae, and Vespertilionidae**) having different **crown ages** (Pleistocene to Miocene) and **biome affiliations** (temperate to arid).*

| Span | Label |
| :---- | :---- |
| southern African bats | `BIOTIC_ENTITY` (specific bats that can be listed) |
| Rhinolophidae, Cistugidae, Vespertilionidae | `BIOTIC_ENTITY` |
| crown ages | `BIOTIC_PROPERTY` |
| biome affiliations | `BIOTIC_PROPERTY` |

**Note if we split into sub-nodes:** in the context of “southern African bats"- → “southern African" is `SPATIAL_PROPERTY` of the BIOTIC\_ENTITY “bats”.   
“Europear bee-eaters” \---\> european is not a SPATIAL\_PROPERTY of the “bee-eater” but a part the name (check by taxonomy verification). 

#### Case B

*In **horseshoe bats** (Rhinolophidae), both the **western** and **eastern "arms" of the Escarpment** have facilitated **dispersals** from the **Afrotropics** into **southern Africa**.*

| Span | Label |
| :---- | :---- |
| horseshoe bats | `BIOTIC_ENTITY` |
| western and eastern "arms" of the Escarpment | `SPATIAL_ENTITY` |
| dispersals | `BIOTIC_PROCESS` |
| Afrotropics | `SPATIAL_ENTITY` |
| southern Africa | `SPATIAL_ENTITY` |

#### Case C

*While our work confirms prior findings that **predator presence** drives strong reductions in **insect emergence**…*

| Span | Label |
| :---- | :---- |
| predator | `BIOTIC_ENTITY` (generic use) |
| insect | `BIOTIC_ENTITY` (generic use) |
| presence | `BIOTIC_PROPERTY` |
| emergence | `BIOTIC_PROCESS` |

---

## General Tiebreaker Rule

**When unsure, choose the category describing the primary referent of the term in the sentence.**

