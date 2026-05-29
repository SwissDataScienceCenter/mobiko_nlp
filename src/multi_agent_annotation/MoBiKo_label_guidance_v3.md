# MoBiKo – Guidance for Span Labelling

The workflow consists of two steps: **(1) identifying a span** and **(2) labelling it**.

---

## Preliminary Comments

- Before identifying and labeling spans, make sure the sentence contains relevant information, i.e. contents that tell something about mountain biodiversity and ecosystems, their status, trends, and drivers.

- Ignore sentences consisting of:
  - Information about **methods** (e.g. *The analysis was run using the R packages igraph (Csardi and Nepusz 2006) and gplots (Warnes et al 2019)*).
  - **General statements**, not specific for a particular study and not bringing new/important information (e.g. *Assessing biodiversity loss and species extinction is necessary to warn society and raise awareness of the impacts of ongoing climate change*).
  - **Hypotheses**.

- Ignore spans that are only methods-related in any sentence (e.g. linear regression, binomial distribution, etc.).

---

## Step 1: Identifying Spans

**Rule:** spans should consist in the minimum number of words. When possible, entities are therefore split. However, when entities taken together express a commonly used concept and very frequently appear together, they should not be split.

Good examples: [mountain biodiversity], [habitat quality], [species richness], [relict species], [biome affiliation]. A longer span has richer meaning than its separate components (1 + 1 > 2).

---

## Step 2: Labelling Spans

Labeling follows a classification consisting of two logical layers:

- **Ontological role:** entity; process; property; concept
- **Domain:** abiotic; biotic; anthropogenic; spatial; temporal; quantitative; qualitative

Except for concepts, which are not associated with a domain, all other ontological roles are systematically associated with a domain.

### Definitions

#### Ontological Roles

| Label | Definition |
|---|---|
| **Entity** | Aggregation or assemblage of entities functioning as a unit, OR something that exists as itself — an individual object or component of the environment. |
| **Process** | Change in the environment; a noun or verb referring to a change, event, or action unfolding over time. *Q: Is this something that happens?* |
| **Property** | State, measurement, or characteristic. *Q: Is this something that something has?* |
| **Concept** | Abstract or theoretical construct used in analysis or discourse (e.g. climate, conservation status, resilience, scenario, vulnerability, link, trend). |

#### Domains

| Label | Definition |
|---|---|
| **Abiotic** | Refers to any ecosystem factor, attribute, component, or constituent that is devoid of life; without life. |
| **Biotic** | Refers to any living factor, attribute, component, or constituent; with life. |
| **Anthropogenic** | Caused by humans or their activities. |
| **Spatial** | Related to space. |
| **Temporal** | Related to time. |
| **Quantitative** | Something you can measure: how much, how many. |
| **Qualitative** | The kind (e.g. big, bright, etc.) — involving quality or kind. |


---

## Handling Difficult or Ambiguous Cases

### 1. Properties — Look at what the property belongs to

Use the **modified noun** to make a decision.

**Example: *density***

- **population** density → `BIOTIC_PROPERTY`
- **soil** density → `ABIOTIC_PROPERTY`

**Example: *distribution***

- **species** distribution → `BIOTIC_PROPERTY`
- **temperature** distribution → `ABIOTIC_PROPERTY`
- **normal** distribution → `CONCEPT`

**Example: *quality***

- **habitat** quality → `SPATIAL_PROPERTY`

> **Note:** Depending on the context, "habitat" could be `BIOTIC_ENTITY` — label accordingly.

---

### 2. Conjunctions and Disjunctions (AND/OR)

When multiple properties, entities, or processes are connected with AND/OR, unfold them into separate spans.

**Example: *Antibacterial and antifungal properties***

- Antibacterial properties → `BIOTIC_PROPERTY`
- Antifungal properties → `BIOTIC_PROPERTY`

---

### 3. Ontological Role Differs by Context

**Example: *rainfall***

- *Heavy rainfall caused erosion* → `ABIOTIC_PROCESS`
- *…receives 100 mm of rainfall* → `ABIOTIC_PROPERTY`

---

### 4. Species and Taxonomic Groups

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

### 5. Ecological Attributes

Common ecological measurement terms are almost always properties.

| Term | Label |
|---|---|
| abundance | `BIOTIC_PROPERTY` |
| biomass | `BIOTIC_PROPERTY` |
| richness | `BIOTIC_PROPERTY` |
| productivity | `BIOTIC_PROPERTY` |

---

### 6. Human Research Activities

Research activities are anthropogenic processes.

| Term | Label |
|---|---|
| analysis | `ANTHROPOGENIC_PROCESS` |
| sampling | `ANTHROPOGENIC_PROCESS` |
| measurement | `ANTHROPOGENIC_PROCESS` |

> **Note:** Human research activity is usually part of the method section and is generally not labeled.

---

### 7. System-Level Ecological Terms

Some terms describe whole-system states.

| Term | Label |
|---|---|
| biodiversity | `BIOTIC_PROPERTY` |
| ecosystem stability | `BIOTIC_PROPERTY` |
| community structure | `BIOTIC_PROPERTY` |

---

### 8. Spatial vs. Abiotic Entity

The key question: **can you point to this on a map?**

- **Mountain glaciers** → `ABIOTIC_ENTITY` (cannot be located on a map as a distinct place)
- **Aletsch glaciers** → `SPATIAL_ENTITY` (can be identified and located on a map)

---

### 9. Polysemic Terms (context-dependent categories)

#### *Biodiversity*

| Context | Label |
|---|---|
| *…tagged with life on land (SDG 15), even if the SDG itself was not explicitly mentioned* | `BIOTIC_ENTITY` |
| *…safeguard their inhabitants, their ecosystems, their **biodiversity**, and the livelihoods they support* | `BIOTIC_ENTITY` |
| *…tepuis are table-top mountains with elevations above 1000 m and high **biodiversity** and endemism levels* | `BIOTIC_PROPERTY` |
| Component of the IPBES framework | `CONCEPT` |

#### *Human wellbeing*

| Context | Label |
|---|---|
| *Elements of the IPBES framework… biodiversity and ecosystems, ecosystem goods and services, **human wellbeing**, direct drivers…* | `CONCEPT` |
| *…to assess and compare the contents of 631 abstracts on the interactions among biodiversity, ecosystem services, **human wellbeing**, and drivers of change…* | `ANTHROPOGENIC_PROPERTY` |

#### *Mountain / montane / alpine*

| Term | Context | Label |
|---|---|---|
| mountain | *…using the find function and the keywords "mountain," "montane," and "alpine"…* | `SPATIAL_ENTITY` |
| mountains | *Mountains are facing growing environmental, social, and economic challenges* | `ABIOTIC_ENTITY` |
| montane, alpine | *…using the find function and the keywords "mountain," "montane," and "alpine"…* | `SPATIAL_PROPERTY` |
| mountain | *…prioritize areas and populations of an endangered **mountain** tree species…* | *(modifier, not labeled separately)* |
| alpine | Biotic modifier | `BIOTIC_PROPERTY` |

#### *Elevation*

| Context | Label |
|---|---|
| *…we assessed the effect of **elevation** on dung beetle assemblage structure…* | `SPATIAL_PROPERTY` |
| *…tepuis are table-top mountains with **elevations** above 1000 m…* | `ABIOTIC_PROPERTY` |

#### *Area*

| Context | Label |
|---|---|
| *…populations located in rugged **areas***  | `SPATIAL_ENTITY` |
| Used as a measure | `SPATIAL_PROPERTY` |

#### *Geology*

| Context | Label |
|---|---|
| *The **geology** of the Alps has changed over the centuries.* | `ABIOTIC_PROPERTY` |
| *In **geology**, bird breeding behaviors are not studied systematically* | `CONCEPT` |

#### *Grazing*

| Context | Label |
|---|---|
| *Grazing by livestock is causing land degradation* | `ANTHROPOGENIC_PROCESS` |
| *Wildlife grazing has increased over the last decades, causing land degradation* | `BIOTIC_PROCESS` |

---

### 10. Typical Difficult Cases

#### Case A

*We investigated patterns of speciation and micro-endemism from **modeled past, present, and future distributions** in six clades of **southern African bats** from three families (**Rhinolophidae, Cistugidae, and Vespertilionidae**) having different **crown ages** (Pleistocene to Miocene) and **biome affiliations** (temperate to arid).*

| Span | Label |
|---|---|
| southern African bats | `BIOTIC_ENTITY` (specific bats that can be listed) |
| Rhinolophidae, Cistugidae, Vespertilionidae | `BIOTIC_ENTITY` |
| crown ages | `BIOTIC_PROPERTY` |
| biome affiliations | `BIOTIC_PROPERTY` |
| modeled past, present, and future distributions | `CONCEPT` |

> **Note:** "southern African bats" can also be read as `SPATIAL_PROPERTY` depending on context.

#### Case B

*In **horseshoe bats** (Rhinolophidae), both the **western** and **eastern "arms" of the Escarpment** have facilitated **dispersals** from the **Afrotropics** into **southern Africa**.*

| Span | Label |
|---|---|
| horseshoe bats | `BIOTIC_ENTITY` |
| western and eastern "arms" of the Escarpment | `SPATIAL_ENTITY` |
| dispersals | `BIOTIC_PROCESS` |
| Afrotropics | `SPATIAL_ENTITY` |
| southern Africa | `SPATIAL_ENTITY` |

#### Case C

*While our work confirms prior findings that **predator presence** drives strong reductions in **insect emergence**…*

| Span | Label |
|---|---|
| predator | `BIOTIC_ENTITY` (generic use) |
| insect | `BIOTIC_ENTITY` (generic use) |
| presence | `BIOTIC_PROPERTY` |
| emergence | `BIOTIC_PROCESS` |

---

## General Tiebreaker Rule

> **When unsure, choose the category describing the primary referent of the term in the sentence.**
