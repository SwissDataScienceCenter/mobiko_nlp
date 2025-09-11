import json
import csv
import re
import argparse
import os
import sys
import logging
from typing import List, Dict, Optional, Tuple, NamedTuple, Set, Any
from collections import namedtuple
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from collections import Counter

#

# Add the src directory to Python path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(sys.path)

from ner.labels import EntityLabel


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# -------- Rules (schema) handling --------

# Constants
DEFAULT_REGEX = ".*"
TAXONOMY_SOURCES = ("ncbitaxon", "ott", "mdd")
HABITAT_KEYWORDS = r"\b(habitat|ecosystem(s)?|biome|biotope)\b"
THREAT_KEYWORDS = r"\b(anthropogenic|disturbance|barrier|road|settlement|land\s*use|mobility)\b"
POPULATION_KEYWORDS = r"\b(population|metapopulation)\b"
LOCATION_NAMES = r"(alps|alpine|andes|pyrenees|himalaya(s)?)"
HUMAN_KEYWORDS = r"(homo\s*sapiens|humans?)"

# BIO tags
O_TAG = "O"
B_PREFIX = "B-"
I_PREFIX = "I-"

# Processing constants
SPACY_BATCH_SIZE = 5000

@dataclass
class ProcessingConfig:
    """Configuration for processing pipeline."""
    schema_path: Optional[str] = None
    gazetteer_path: Optional[str] = None
    prefer_spacy: bool = False


class Rule(NamedTuple):
    """Configuration rule for entity labeling."""
    source: str
    type_regex: str
    term_regex: str
    label: str
    include: bool
    priority: int


class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass


# -------- Utility Functions --------

def normalize_text_for_matching(text: str) -> str:
    """Normalize text for case-insensitive matching."""
    return text.lower().replace("_", "").replace("-", "")


def safe_regex_match(pattern: str, text: str, flags: int = 0) -> bool:
    """Safely apply regex matching with error handling."""
    try:
        return bool(re.search(pattern, text, flags=flags))
    except re.error as e:
        logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        return False


def safe_regex_fullmatch(pattern: str, text: str, flags: int = 0) -> bool:
    """Safely apply regex fullmatch with error handling."""
    try:
        return bool(re.fullmatch(pattern, text, flags=flags))
    except re.error as e:
        logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        return False


def should_exclude_rule(include_value: str) -> bool:
    """Check if rule should be excluded based on include value."""
    exclude_values = {"0", "false", "False", "NO", "no"}
    return str(include_value).strip() in exclude_values


# -------- Rules (schema) handling --------


def parse_rule_row(row: Dict[str, str], priority: int) -> Rule:
    """Parse a single rule from CSV row."""
    return Rule(
        source=row.get("concept_source", "").strip() or DEFAULT_REGEX,
        type_regex=row.get("type", "").strip() or DEFAULT_REGEX,
        term_regex=row.get("preferred_term_regex", "").strip() or DEFAULT_REGEX,
        label=(row.get("label", "") or "*").strip(),
        include=not should_exclude_rule(row.get("include", "1")),
        priority=int(row.get("priority", priority))
    )


def load_rules(csv_path: Optional[str]) -> List[Rule]:
    """Load labeling rules from CSV file.

    Args:
        csv_path: Path to CSV file with rules, or None

    Returns:
        List of Rule objects sorted by priority
    """
    if not csv_path:
        return []

    rules = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for priority, row in enumerate(reader):
                try:
                    rule = parse_rule_row(row, priority)
                    rules.append(rule)
                except Exception as e:
                    logger.warning(f"Skipping rule row due to error: {e} -- {row}")

        rules.sort(key=lambda r: r.priority)
        logger.info(f"Loaded {len(rules)} rules from {csv_path}")
        return rules

    except FileNotFoundError:
        logger.error(f"Rules file not found: {csv_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading rules from {csv_path}: {e}")
        return []


def apply_rules(infons: Dict[str, Any], rules: List[Rule]) -> Optional[str]:
    """Apply labeling rules to entity annotations."""
    if not rules:
        return None

    src = infons.get("concept_source") or ""
    typ = infons.get("type") or ""
    pref = infons.get("preferred_term") or ""

    for rule in rules:
        if not safe_regex_fullmatch(rule.source, src, flags=re.I):
            continue
        if not safe_regex_fullmatch(rule.type_regex, typ, flags=re.I):
            continue
        if not safe_regex_match(rule.term_regex, pref, flags=re.I):
            continue

        if not rule.include:
            return None
        return None if rule.label in {"", "*"} else rule.label

    return None


# -------- Heuristic default labeling --------


def is_human_taxon(preferred_term: str) -> bool:
    """Check if taxon refers to humans."""
    return safe_regex_match(HUMAN_KEYWORDS, preferred_term, flags=re.I)


def classify_taxonomy_term(src: str, preferred_term: str) -> Optional[str]:
    """Classify taxonomy terms."""
    if src.startswith(TAXONOMY_SOURCES):
        return None if is_human_taxon(preferred_term) else EntityLabel.TAXON.value
    return None


def classify_envo_term(preferred_term: str) -> Optional[str]:
    """Classify ENVO terms as HABITAT or ENV_FEATURE."""
    if safe_regex_match(HABITAT_KEYWORDS, preferred_term, flags=re.I):
        return EntityLabel.HABITAT.value
    return EntityLabel.ENV_FEATURE.value


def classify_mesh_term(preferred_term: str) -> Optional[str]:
    """Classify MeSH terms."""
    if safe_regex_match(r"\becosystem(s)?\b", preferred_term, flags=re.I):
        return EntityLabel.HABITAT.value
    return None


def classify_threat_term(preferred_term: str) -> Optional[str]:
    """Classify threat-related terms."""
    if safe_regex_match(THREAT_KEYWORDS, preferred_term, flags=re.I):
        return EntityLabel.THREAT.value
    return None


def classify_population_term(preferred_term: str) -> Optional[str]:
    """Classify population-related terms."""
    if safe_regex_match(POPULATION_KEYWORDS, preferred_term, flags=re.I):
        return EntityLabel.POPULATION.value
    return None


def classify_location_term(preferred_term: str) -> Optional[str]:
    """Classify location terms."""
    if safe_regex_fullmatch(LOCATION_NAMES, preferred_term, flags=re.I):
        return EntityLabel.LOCATION.value
    return None



def default_label(infons: Dict[str, Any]) -> Optional[str]:
    """Apply heuristic labeling rules to entity annotations."""
    src = normalize_text_for_matching(infons.get("concept_source") or "")
    pref = normalize_text_for_matching(infons.get("preferred_term") or "")

    # Try each classification strategy
    classifiers = [
        lambda: classify_taxonomy_term(src, pref),
        lambda: classify_envo_term(pref) if src == "envo" else None,
        lambda: classify_mesh_term(pref) if src == "mesh" else None,
        lambda: classify_threat_term(pref),
        lambda: classify_population_term(pref),
        lambda: classify_location_term(pref),
    ]

    for classifier in classifiers:
        result = classifier()
        if result is not None:
            return result

    return None  # default: drop



# -------- Gazetteer (ENVO habitats) --------


def split_synonyms(synonym_string: str) -> List[str]:
    """Split synonym string on semicolons or pipes."""
    if not synonym_string:
        return []
    parts = re.split(r"[;|]\s*", synonym_string)
    return [p.strip() for p in parts if p.strip()]


def normalize_gazetteer_term(term: str) -> str:
    """Normalize gazetteer terms for consistent matching."""
    return re.sub(r"\s+", " ", term.replace(""", "\"").replace(""", "\"").strip())


def load_habitat_gazetteer(csv_path: str) -> List[Dict[str, str]]:
    """Load habitat gazetteer from CSV file."""
    rows = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                envo_id = row.get("envo_id", "")
                label = (row.get("label", "") or "").strip()

                # Collect all synonyms
                synonyms = []
                for syn_type in ["exact_synonyms", "broad_synonyms", "narrow_synonyms", "related_synonyms"]:
                    synonyms.extend(split_synonyms(row.get(syn_type, "")))

                # Process all terms (label + synonyms)
                terms = [label] + synonyms
                for term in terms:
                    normalized_term = normalize_gazetteer_term(term)
                    if normalized_term:
                        rows.append({
                            "surface": normalized_term,
                            "envo_id": envo_id,
                            "label": EntityLabel.HABITAT.value
                        })

        # Remove case-insensitive duplicates
        dedup_dict = {}
        for row in rows:
            key = row["surface"].lower()
            dedup_dict[key] = row

        logger.info(f"Loaded {len(dedup_dict)} unique habitat terms from gazetteer")
        return list(dedup_dict.values())

    except Exception as e:
        logger.error(f"Error loading gazetteer from {csv_path}: {e}")
        return []



def build_spacy_matcher(gazetteer: List[Dict[str, str]]):
    """Build spaCy-based phrase matcher."""
    try:
        import spacy
        from spacy.matcher import PhraseMatcher

        nlp = spacy.blank("en")
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

        # Add patterns in batches to avoid memory issues
        patterns = [nlp.make_doc(item["surface"]) for item in gazetteer]
        for i in range(0, len(patterns), SPACY_BATCH_SIZE):
            batch = patterns[i:i + SPACY_BATCH_SIZE]
            matcher.add("HABITAT", batch)

        def match_function(text: str) -> List[Dict[str, Any]]:
            doc = nlp(text)
            spans = []

            for match_id, start, end in matcher(doc):
                char_start = doc[start].idx
                char_end = doc[end - 1].idx + len(doc[end - 1])
                spans.append({
                    "start": char_start,
                    "end": char_end,
                    "text": text[char_start:char_end],
                    "label": EntityLabel.HABITAT.value,
                    "source": "ENVO_GAZ",
                    "concept_id": None,
                    "preferred_term": doc[start:end].text
                })

            return filter_overlapping_spans(spans)

        return nlp, match_function

    except Exception as e:
        logger.warning(f"spaCy not available: {e}")
        raise


def build_fallback_matcher(gazetteer: List[Dict[str, str]]):
    """Build fallback substring matcher."""
    # Sort terms by length (longest first) for better matching
    terms = sorted({item["surface"] for item in gazetteer}, key=len, reverse=True)
    lower_terms = [term.lower() for term in terms]

    def dummy_tokenizer(text: str) -> str:
        return text  # Not used in fallback matcher

    def match_function(text: str) -> List[Dict[str, Any]]:
        text_lower = text.lower()
        spans = []
        occupied = [False] * len(text)

        for term, term_lower in zip(terms, lower_terms):
            start_pos = 0
            while True:
                idx = text_lower.find(term_lower, start_pos)
                if idx == -1:
                    break

                end_idx = idx + len(term)

                # Skip if overlaps with existing span
                if any(occupied[idx:end_idx]):
                    start_pos = idx + 1
                    continue

                spans.append({
                    "start": idx,
                    "end": end_idx,
                    "text": text[idx:end_idx],
                    "label": EntityLabel.HABITAT.value,
                    "source": "ENVO_GAZ",
                    "concept_id": None,
                    "preferred_term": term
                })

                # Mark positions as occupied
                for k in range(idx, end_idx):
                    occupied[k] = True

                start_pos = end_idx

        return sorted(spans, key=lambda x: (x["start"], -(x["end"] - x["start"])))

    return dummy_tokenizer, match_function


def filter_overlapping_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter overlapping spans, preferring longer ones."""
    spans.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    filtered = []
    taken = [False] * len(spans)

    for i, span_a in enumerate(spans):
        if taken[i]:
            continue

        # Mark contained spans as taken
        for j, span_b in enumerate(spans[i + 1:], i + 1):
            if (span_a["start"] <= span_b["start"] and
                span_a["end"] >= span_b["end"]):
                taken[j] = True

        filtered.append(span_a)

    return filtered


def build_matcher_from_gazetteer(gazetteer: List[Dict[str, str]], prefer_spacy: bool = True):
    """Build matcher from gazetteer, with spaCy preference and fallback."""
    if prefer_spacy:
        try:
            return build_spacy_matcher(gazetteer)
        except Exception as e:
            logger.info(f"Falling back to substring matcher: {e}")

    return build_fallback_matcher(gazetteer)


# -------- Tokenization & BIO tagging --------


def simple_whitespace_tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Simple whitespace tokenization with character spans."""
    tokens, spans = [], []
    for match in re.finditer(r"\S+", text):
        start, end = match.start(), match.end()
        tokens.append(text[start:end])
        spans.append((start, end))
    return tokens, spans


def assign_bio_tags(tags: List[str], start: int, end: int, label: str,
                   token_spans: List[Tuple[int, int]]) -> None:
    """Assign BIO tags for a single character span over token spans."""
    started = False
    for i, (token_start, token_end) in enumerate(token_spans):
        # Skip non-overlapping tokens
        if token_end <= start or token_start >= end:
            continue

        # Don't overwrite different labels
        if tags[i] != O_TAG and not tags[i].endswith(label):
            continue

        # Assign B- for first token, I- for subsequent
        tags[i] = (B_PREFIX if not started else I_PREFIX) + label
        started = True



def repair_bio_tags(tags: List[str]) -> None:
    """Convert illegal I- tags into B- tags (strict IOB2 format)."""
    prev_tag = O_TAG
    for i, tag in enumerate(tags):
        if tag == O_TAG:
            prev_tag = O_TAG
            continue

        if tag.startswith(I_PREFIX):
            tag_type = tag[2:]
            # Convert I- to B- if previous tag doesn't match
            if (prev_tag == O_TAG or
                (prev_tag.startswith((B_PREFIX, I_PREFIX)) and prev_tag[2:] != tag_type)):
                tags[i] = B_PREFIX + tag_type

        prev_tag = tags[i]


# -------- Core processing --------

def extract_base_spans_from_bioc_passage(passage: Dict[str, Any],
                                        rules: List[Rule]) -> List[Dict[str, Any]]:
    """Extract entity spans from BioC passage using rules and heuristics."""
    text = passage.get("text", "")
    spans = []

    for annotation in passage.get("annotations", []):
        infons = annotation.get("infons", {}) or {}

        # Try schema rules first, then default heuristics
        label = apply_rules(infons, rules)
        if label is None:
            label = default_label(infons)
        if not label:
            continue

        # Process all locations for this annotation
        for location in annotation.get("locations", []):
            start = int(location.get("offset", 0))
            length = int(location.get("length", 0))
            end = start + length

            # Validate span boundaries
            if end <= start or start < 0 or end > len(text):
                continue

            spans.append({
                "start": start,
                "end": end,
                "text": text[start:end],
                "label": label,
                "source": infons.get("concept_source"),
                "concept_id": infons.get("concept_id"),
                "preferred_term": infons.get("preferred_term")
            })

    # Sort by start position and length (longest first)
    spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))

    # Remove exact duplicates
    filtered_spans = []
    seen_keys = set()
    for span in spans:
        key = (span["start"], span["end"], span["label"])
        if key not in seen_keys:
            seen_keys.add(key)
            filtered_spans.append(span)

    return filtered_spans


def merge_spans(base_spans: List[Dict[str, Any]],
                added_spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge additional spans with base spans, avoiding overlaps."""
    merged = base_spans[:]

    for span in added_spans:
        # Check for overlaps with existing spans
        has_overlap = any(
            not (span["end"] <= existing["start"] or span["start"] >= existing["end"])
            for existing in merged
        )

        if not has_overlap:
            merged.append(span)

    # Sort by position and length
    merged.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    return merged


def extract_records_from_sentence_format(article: Dict[str, Any],
                                         rules: List[Rule],
                                         gaz_matcher=None) -> List[Dict[str, Any]]:
    """Handle BioC JSON with top-level 'sentences' and flat 'annotations'."""
    records = []

    # Build sentence index: (field, sentence_number) -> sentence text
    sent_by_key: Dict[Tuple[str, int], str] = {}
    sent_objs: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for s in article.get("sentences", []):
        fld = s.get("field") or ""
        num = int(s.get("sentence_number", 0))
        txt = s.get("sentence") or ""
        key = (fld, num)
        sent_by_key[key] = txt
        sent_objs[key] = s  # keep original if needed

    # Group annotations by (field, sentence_number)
    ann_by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    for a in article.get("annotations", []):
        fld = a.get("field") or ""
        num = int(a.get("sentence_number", 0))
        key = (fld, num)
        ann_by_key.setdefault(key, []).append(a)

    # For each sentence, convert annotations → spans → BIO
    for key, text in sent_by_key.items():
        fld, num = key
        anns = ann_by_key.get(key, [])
        spans: List[Dict[str, Any]] = []

        # Convert flat annotations into our internal span dicts
        for a in anns:
            # Build infons-like dict so rules/default_label still work
            infons = {
                "concept_source": a.get("concept_source", ""),
                "type": a.get("type", ""),
                "preferred_term": a.get("preferred_term", ""),
                "concept_id": a.get("concept_id", ""),
            }
            label = apply_rules(infons, rules)
            if label is None:
                label = default_label(infons)
            if not label:
                continue


            start = int(a.get("start_index", 0))
            end   = int(a.get("end_index", 0))   # end is exclusive in your example
            # Validate boundaries against this sentence text
            if start < 0 or end <= start or end > len(text):
                continue

            spans.append({
                "start": start,
                "end": end,
                "text": text[start:end],
                "label": label,

                # --- added metadata ---
                "concept_source": a.get("concept_source"),
                "concept_id": a.get("concept_id"),
                "concept_form": a.get("concept_form"),
                "evidence_code": a.get("evidence_code"),
                "preferred_term": a.get("preferred_term"),
                "score": a.get("score"),
                "provider": a.get("provider"),
                "provenance": a.get("provenance"),
                "field": a.get("field"),
            })

        # Optional gazetteer augmentation
        if gaz_matcher is not None and text:
            gaz_spans = gaz_matcher(text)
            spans = merge_spans(spans, gaz_spans)
        else:
            spans.sort(key=lambda s: (s["start"], -(s["end"]-s["start"])))

        # Tokenize + BIO
        tokens, token_spans = simple_whitespace_tokenize(text)
        tags = [O_TAG] * len(tokens)
        for sp in spans:
            assign_bio_tags(tags, sp["start"], sp["end"], sp["label"], token_spans)
        repair_bio_tags(tags)

        # Validate consistency
        assert len(tags) == len(tokens), f"Tag/token length mismatch: {len(tags)} vs {len(tokens)}"
        # #
        # print(tags)
        # print(tokens)
        # print(spans)
        # print('___________')


        records.append({
            "doc_id": article.get("id") or article.get("doc_id") or "",
            "sentence_id": num,
            "field": fld,
            "text": text,
            "tokens": tokens,
            "spans": spans,
            "tags": tags,
        })

    return records


def extract_records_from_manual_format(article, rules, gaz_matcher=None):
    """ Hanadle manual annotation format with 'sentences' """
    records = []

    for s in article.get("sentences", []):
        text = s.get("text") or ""
        spans: List[Dict[str, Any]] = []

        for a in s.get('spans'):
            sources = a.get("sources", []) # manually annoated file has more than 1 source per identical span (why?)
            best_label = None
            best_infons = None
            label = None

            for source in sources:
                infons = {
                    "concept_source": source.get("source", ""),
                    "concept_id": source.get("source_id", "")
                }
                label = apply_rules(infons, rules)

                if label is None:
                    label = default_label(infons)

                if label is not None:
                    best_label = label
                    best_infons = infons
                    break

            label = best_label
            infons = best_infons

            if not label:
                continue

            start = int(a.get("start_char", 0))
            end = int(a.get("end_char", 0))

            # Validate boundaries against this sentence text
            if start < 0 or end <= start or end > len(text):
                continue

            spans.append({
                "start": start,
                "end": end,
                "text": text[start:end],
                "label": label,

                # --- added metadata ---
                "concept_source": infons.get("concept_source"),
                "concept_id": infons.get("concept_id"),
            })

        # Optional gazetteer augmentation
        if gaz_matcher is not None and text:
            gaz_spans = gaz_matcher(text)
            spans = merge_spans(spans, gaz_spans)
        else:
            spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))

        # Tokenize + BIO
        tokens, token_spans = simple_whitespace_tokenize(text)
        tags = [O_TAG] * len(tokens)
        for sp in spans:
            assign_bio_tags(tags, sp["start"], sp["end"], sp["label"], token_spans)
        repair_bio_tags(tags)

        # Validate consistency
        assert len(tags) == len(tokens), f"Tag/token length mismatch: {len(tags)} vs {len(tokens)}"

        records.append({
            "doc_id": article.get("id") or article.get("doc_id") or "",
            "text": text,
            "tokens": tokens,
            "spans": spans,
            "tags": tags,
        })

    return records


# --- LEGACY FORMAT: 'passages' with embedded annotations/locations ---
def extract_records_from_legacy_passages(article, rules, gaz_matcher=None):
    """Handle BioC JSON with 'passages' (legacy structure)."""

    records = []
    doc_id = article.get("id") or article.get("doc_id") or ""
    for passage in article.get("passages", []):
        text = passage.get("text", "") or ""
        sent_id = passage.get("infons", {}).get("sentence_number")

        base_spans = extract_base_spans_from_bioc_passage(passage, rules)
        spans = merge_spans(base_spans, gaz_matcher(text)) if (gaz_matcher and text) else base_spans

        tokens, token_spans = simple_whitespace_tokenize(text)
        tags = [O_TAG] * len(tokens)
        for sp in spans:
            assign_bio_tags(tags, sp["start"], sp["end"], sp["label"], token_spans)

        repair_bio_tags(tags)

        records.append({
            "doc_id": doc_id,
            "sentence_id": sent_id,
            "text": text,
            "tokens": tokens,
            "spans": spans,
            "tags": tags
        })
    return records


def process_json_file(json_path: Path, rules: List[Rule], manual=False,
                     gaz_matcher=None, tokenizer=None) -> List[Dict[str, Any]]:
    """Process a single JSON file and return records for either 'passages' or 'sentences' format."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ProcessingError(f"Error reading {json_path}: {e}")

    out_records: List[Dict[str, Any]] = []

    # Support multiple top-level shapes
    articles = (data.get("sibils_article_set") or data.get("articles") or [data])

    for article in articles:
        doc_id = article.get("id") or article.get("doc_id") or json_path.stem

        # New format: sentences + flat annotations
        if article.get("sentences") and isinstance(article.get("sentences"), list):
            if manual:
                recs = extract_records_from_manual_format(article, rules, gaz_matcher=gaz_matcher)
            else:
                recs = extract_records_from_sentence_format(article, rules, gaz_matcher=gaz_matcher)
                # Ensure doc_id present
            for r in recs:
                if not r.get("doc_id"):
                    r["doc_id"] = doc_id
            out_records.extend(recs)
            continue

        # Legacy format
        if isinstance(article.get("passages"), list):
            out_records.extend(extract_records_from_legacy_passages(article, rules, gaz_matcher=gaz_matcher))
            continue

        continue

    return out_records


# -------- CLI --------

def setup_gazetteer(config: ProcessingConfig):
    """Set up gazetteer matcher if configured."""
    if not config.gazetteer_path or not os.path.exists(config.gazetteer_path):
        return None, None

    try:
        gazetteer = load_habitat_gazetteer(config.gazetteer_path)
        tokenizer, matcher = build_matcher_from_gazetteer(
            gazetteer, prefer_spacy=config.prefer_spacy
        )
        return tokenizer, matcher
    except Exception as e:
        logger.error(f"Failed to load gazetteer: {e}")
        return None, None


def find_json_files(input_path: Path) -> List[Path]:
    """Find all JSON files in the input path."""
    if input_path.is_file():
        return [input_path]

    json_files = sorted(input_path.rglob("*.json"))
    if not json_files:
        raise ProcessingError(f"No JSON files found under {input_path}")

    return json_files


def compute_stats(records, by_bio=False):
    """
    Returns a dict with:
      - total_sentences
      - percent_no_tags
      - avg_tags_per_tagged_sentence
      - label_distribution (entity type, e.g., TAXON/HABITAT/…)
      - bio_distribution (optional: B-*/I-*/O)
    """

    n = len(records)
    if n == 0:
        return {
            "total_sentences": 0,
            "percent_no_tags": 0.0,
            "avg_tags_per_tagged_sentence": 0.0,
            "label_distribution": {},
            "bio_distribution": {} if by_bio else None,
        }

    num_no_tags = 0
    tags_per_tagged = []
    label_ctr = Counter()
    bio_ctr = Counter()

    for r in records:
        # count spans (entity instances) per sentence
        spans = r.get("spans", []) or []
        if len(spans) == 0:
            num_no_tags += 1
        else:
            tags_per_tagged.append(len(spans))

        # distribution over entity *types* (labels)
        for sp in spans:
            lab = sp.get("label")
            if lab:
                label_ctr[lab] += 1

        if by_bio:
            for t in r.get("tags", []) or []:
                bio_ctr[t] += 1

    percent_no = 100.0 * num_no_tags / max(1, n)
    avg_tags = (sum(tags_per_tagged) / len(tags_per_tagged)) if tags_per_tagged else 0.0

    out = {
        "total_sentences": n,
        "percent_no_tags": round(percent_no, 2),
        "avg_tags_per_tagged_sentence": round(avg_tags, 3),
        "label_distribution": dict(sorted(label_ctr.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    if by_bio:
        out["bio_distribution"] = dict(sorted(bio_ctr.items(), key=lambda kv: (-kv[1], kv[0])))
    return out




def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Process BioC JSON files to JSONL with NER tags")
    parser.add_argument("--in_path", required=True,
                        help="JSON file or directory with BioC-like JSONs")
    parser.add_argument("--out_jsonl", required=True,
                        help="Output JSONL with tokens/spans/tags")
    parser.add_argument("--schema", default=None,
                        help="CSV rules file (optional)")
    parser.add_argument("--envo_gazetteer_csv", default=None,
                        help="CSV built from ENVO term table (optional)")
    parser.add_argument("--prefer_spacy", action="store_true",
                        help="Use spaCy PhraseMatcher if available")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--stats_out", default=None,
                    help="Write a JSON file with split-level stats (coverage, distributions)")
    parser.add_argument("--stats_bio", action="store_true", help="Also include BIO tag distribution")
    parser.add_argument("--manual", action="store_true", help="Input uses manual annotation format")
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        config = ProcessingConfig(
            schema_path=args.schema,
            gazetteer_path=args.envo_gazetteer_csv,
            prefer_spacy=args.prefer_spacy
        )

        # Load rules and gazetteer
        rules = load_rules(config.schema_path)
        tokenizer, gaz_matcher = setup_gazetteer(config)

        # Find input files
        input_path = Path(args.in_path)
        json_files = find_json_files(input_path)
        logger.info(f"Found {len(json_files)} JSON files to process")

        # Create output directory
        Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

        # Process files
        total_records = 0
        with open(args.out_jsonl, "w", encoding="utf-8") as output_file:
            for json_path in json_files:
                logger.info(f"Processing {json_path.name}...")

                try:
                    records = process_json_file(
                        json_path, rules,
                        manual=args.manual,
                        gaz_matcher=gaz_matcher,
                        tokenizer=tokenizer
                    )

                    for record in records:
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                    total_records += len(records)
                    logger.info(f"Generated {len(records)} records from {json_path.name}")

                except ProcessingError as e:
                    logger.error(f"Error processing {json_path}: {e}")
                    continue

        # Accumulate all records to compute stats
        all_records = []
        with open(args.out_jsonl, "r", encoding="utf-8") as fin:
            for ln in fin:
                all_records.append(json.loads(ln))

        stats = compute_stats(all_records, by_bio=args.stats_bio)

        print("\n=== Split statistics ===")
        print(f"Total sentences: {stats['total_sentences']}")
        print(f"Sentences with NO tags: {stats['percent_no_tags']}%")
        print(f"Avg #tags per tagged sentence: {stats['avg_tags_per_tagged_sentence']}")
        print("Label distribution (entity types):")
        for lab, cnt in stats["label_distribution"].items():
            print(f"  {lab:>12s}: {cnt}")

        if args.stats_bio and stats.get("bio_distribution"):
            print("BIO tag distribution:")
            for tag, cnt in stats["bio_distribution"].items():
                print(f"  {tag:>12s}: {cnt}")

        if args.stats_out:
            with open(args.stats_out, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"Saved stats → {args.stats_out}")

        logger.info(f"Completed! Processed {len(json_files)} files, "
                    f"generated {total_records} total records")
        logger.info(f"Output written to: {args.out_jsonl}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
