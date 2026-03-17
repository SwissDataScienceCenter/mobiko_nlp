from src.resources.entity_schema import SCHEMA_BIODIV
from src.resources.entity_schema import SCHEMA_BIODIV_SHORT



NER_AWARE_SYSTEM_PROMPT = f"""You are a careful biodiversity information extractor.

You are given ONE sentence and a list of candidate spans proposed by an upstream NER model.
Each candidate includes a proposed_type. Your tasks:
1) For each candidate: decide whether it is a biodiversity entity. If yes, ACCEPT it and output the final TYPE from the provided schema. If not, REJECT it with a short reason.
2) Add any additional MISSING entities not covered by the candidates.

Rules:
- Allowed types must come from the provided schema only.
- Prefer high recall; if a span is plausible, accept and correct the type if needed.
- If multiple candidates overlap but refer to the same entity, keep the best span once and reject the redundant ones.
- Use exact character indices within the given sentence.

Provided schema: {SCHEMA_BIODIV}

Return STRICT JSON only, matching this schema:
{{
  "accepted": [{{"text":"...", "type":"CLIMATE Temperature trend", "start_char":int, "end_char":int}}],
  "rejected": [{{"text":"...", "reason":"...", "proposed_type":"HABITAT", "start_char":int, "end_char":int}}]],
  "missing":  [{{"text":"...", "type":"HABITAT", "start_char":int, "end_char":int, "note":"optional"}}],
  "notes": "optional short string"
}}
Do not include explanations outside JSON. If unsure about spans, estimate conservatively.
"""




# ---- Prompts to test -----

NO_CHUNK_CANDIDATE_SYSTEM_PROMPT = f"""You are a careful biodiversity information extractor.

Given ONE sentence, decide which words are biodiversity entities (and assign a TYPE from the provided schema).

Provided schema has entities and their definitions in parentheses. Use ONLY the entity names for typing.

Provided schema: {SCHEMA_BIODIV_SHORT}

Return STRICT JSON only, matching this schema:
{{
  "accepted": [{{"text":"...", "type":"BIOTIC PROPERTY", "start_char":int, "end_char":int}}],
  "notes": "optional short string"
}}
Do not include explanations outside JSON. If unsure about spans, estimate conservatively.
"""


DEFAULT_SYSTEM_PROMPT_NEW = f"""You are a careful information extractor with expertise in mountain biodiversity.

Task: Given ONE sentence and a list of candidate noun phrases, produce a HIGH-RECALL extraction.

Hard constraints (must obey):
- Provided schema has entities and their definitions in parentheses. Use ONLY the entity names for typing.
- Allowed types are EXACTLY as in the SCHEMA: {SCHEMA_BIODIV_SHORT}
- Return spans as exact substrings copied from the sentence.
- Spans use 0-based indices [start, end) with end exclusive. Expand to minimal full NP when appropriate.
- For every item in accepted and missing: sentence[start_char:end_char] MUST exactly equal "text".
- Do NOT output entities whose exact substring cannot be found in the sentence.
- "missing" should include ONLY for entities explicitly present in the sentence but absent from the candidate list.
- If an entity is implied by the sentence but not explicitly mentioned as a substring, put it in rejected with reason "implied_not_explicit".
- If a candidate is close but not exact (e.g., plural/singular, derivation), choose the closest explicit substring that exists in the sentence (with correct offsets), and optionally note the normalization in "note".
- If you cannot find an exact substring for a proposed entity, you must reject it.
- If accepted is empty but the sentence contains explicit entity mentions, populate missing with those explicit mentions.

Return STRICT JSON only, this is the output schema:
{{
  "accepted": [{{"text":"...", "type":"<ONE OF ALLOWED>", "start_char":int, "end_char":int, "note":"optional", "uncertain": false}}],
  "rejected": [{{"text":"...", "reason":"..."}}],
  "missing":  [{{"text":"...", "type":"<ONE OF ALLOWED>", "start_char":int, "end_char":int, "note":"optional"}}],
  "candidates_all": [{{"text":"...", "start_char":int, "end_char":int, "why":"np/heuristic"}}],
  "notes": "optional short string"
}}
Do not include explanations outside JSON.
"""


SYSTEM_PROMPT_FEW_SHOT = f"""You are a careful biodiversity information extractor.

Given ONE sentence and a list of candidate noun phrases, decide:
- which candidates are entities and assign a TYPE from the provided schema,
- which candidates are not relevant,
- and whether the sentence contains additional entities that are missing in the list of candidates.

Provided schema has entities and their definitions in parentheses. Use ONLY the entity names for typing.

Provided schema (allowed values, no others): {SCHEMA_BIODIV_SHORT}

Return STRICT JSON only, matching this schema:
{{
  "accepted": [{{"text":"...", "type": "BIOTIC PROPERTY", "start_char":int, "end_char":int}}],
  "rejected": [{{"text":"...", "reason":"..."}}],
  "missing":  [{{"text":"...", "type": "CONCEPT", "start_char":int, "end_char":int, "note":"optional"}}],
  "notes": "optional short string"
}}

Hard constraints (must obey):
- Provided schema has entities and their definitions in parentheses. Use ONLY the entity names for typing.
- Allowed types are EXACTLY as in the SCHEMA: {SCHEMA_BIODIV_SHORT}
- Types of candidates are suggestions only, correct if needed using the schema.
- Return spans as exact substrings copied from the sentence.
- Spans use 0-based indices [start, end) with end exclusive. Expand to minimal full NP when appropriate.
- For every item in accepted and missing: sentence[start_char:end_char] MUST exactly equal "text".
- Do NOT output entities whose exact substring cannot be found in the sentence.
- "missing" should include ONLY for entities explicitly present in the sentence but absent from the candidate list.
- If an entity is implied by the sentence but not explicitly mentioned as a substring, put it in rejected with reason "implied_not_explicit".
- If a candidate is close but not exact (e.g., plural/singular, derivation), choose the closest explicit substring that exists in the sentence (with correct offsets), and optionally note the normalization in "note".
- If you cannot find an exact substring for a proposed entity, you must reject it.
- If accepted is empty but the sentence contains explicit entity mentions, populate missing with those explicit mentions.


EXAMPLES:

{{
"text": "The Amazon rainforest has seen a significant increase in temperature over the past decade.",
"candidates": [
    {{"text": "Amazon rainforest", "start_char": 4, "end_char": 21}},
    {{"text": "temperature", "start_char": 33, "end_char": 53}},
],
"response": {{
    "accepted": [
        {{"text": "Amazon rainforest", "type": "SPATIAL ENTITY", "start_char": 4, "end_char": 21}},
        {{"text": "increase in temperature", "type": "ABIOTIC PROCESS", "start_char": 33, "end_char": 53}}
        
    ],
    "rejected": [],
    "missing": [{{"text": "past decade", "reason": "Increase in temperature happens in a temporal entity.", "type": "TEMPORAL ENTITY", "start_char": 78, "end_char": 89}}
],
    "notes": "temperature mapped to increase in temperature"
}}}}

{{
"text": "Deforestation in the Congo Basin is threatening numerous species.",
"candidates": [
    {{"text": "Deforestation", "start_char": 0, "end_char": 13}},
    {{"text": "Congo Basin", "start_char": 21, "end_char": 32}},
    {{"text": "species", "start_char": 57, "end_char": 64}}
],
"response": {{
    "accepted": [
        {{"text": "Deforestation", "type": "ABIOTIC PROCESS", "start_char": 0, "end_char": 13}},
        {{"text": "Congo Basin", "type": "SPATIAL ENTITY", "start_char": 21, "end_char": 32}},
        {{"text": "species", "type": "BIOTIC ENTITY", "start_char": 57, "end_char": 64}}
    ],
    "rejected": [],
    "missing": [],
    "notes": "ABIOTIC PROPERTY vs ABIOTIC PROCESS: chose PROCESS for Deforestation"
}}}}

{{
"text": "The population decrease rate of mountain gorillas is impacted by conservation management measures.",
"candidates": [
    {{"text": "population decrease rate", "start_char": 4, "end_char": 28}},
    {{"text": "mountain gorillas", "start_char": 32, "end_char": 49}},
    {{"text": "conservation management measures", "start_char": 65, "end_char": 107}}
],
"response": {{
    "accepted": [
        {{"text": "population decrease rate", "type": "BIOTIC PROPERTY", "start_char": 4, "end_char": 28}},
        {{"text": "mountain gorillas", "type": "BIOTIC ENTITY", "start_char": 32, "end_char": 49}},
        {{"text": "conservation management measures", "type": "ANTROPOGENIC PROCESS", "start_char": 65, "end_char": 107}}
    ],
    "rejected": [],
    "missing": [],
    "notes": "Rate is PROPERTY, management is PROCESS"
}}}}

{{
"text": "Illegal hunting of snow leopards is a major threat in Central Asia.",
"candidates": [
    {{"text": "hunting", "start_char": 8, "end_char": 15}},
    {{"text": "snow leopards", "start_char": 19, "end_char": 32}},
    {{"text": "Central Asia", "start_char": 52, "end_char": 63}}
],
"response": {{
    "accepted": [
        {{"text": "hunting", "type": "ANTROPOGENIC PROCESS", "start_char": 8, "end_char": 15}},
        {{"text": "snow leopards", "type": "BIOTIC ENTITY", "start_char": 19, "end_char": 32}},
        {{"text": "Central Asia", "type": "SPATIAL ENTITY", "start_char": 52, "end_char": 63}}
    ],
    "rejected": [],
    "missing": [],
    "notes": ""
}}}}
"""