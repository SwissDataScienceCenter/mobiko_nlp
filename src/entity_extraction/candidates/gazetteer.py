# entity_extraction/candidates/gazetteer.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from src.preprocess.gazetteer_matcher import (
    Rule as GazetteerRule,
    load_gaz_rules_from_dir,
    GazetteerMatcher,
    load_general_rules_from_csv,
)
from tables.thesaurus.general_cols import general_columns


def load_gazetteer_matcher(
    gaz_dir: Optional[str],
    general_table_dir: Optional[str],
) -> Optional[GazetteerMatcher]:
    if not gaz_dir:
        return None

    gaz_rules = load_gaz_rules_from_dir(dir_path=gaz_dir)
    general_gaz_rules = []
    if general_table_dir:
        general_gaz_rules = load_general_rules_from_csv(
            path=general_table_dir, keep_meta_cols=general_columns
        )
    matcher = GazetteerMatcher(gaz_rules + general_gaz_rules)
    print(
        f"[gazetteer] Loaded {len(gaz_rules)} rules from {gaz_dir} "
        f"(phrase={len(matcher.phrase_rules)}, regex={len(matcher.regex_rules)})"
    )
    return matcher


def gazetteer_candidates(sentence: str, gazetteer: Optional[GazetteerMatcher]) -> List[Dict[str, Any]]:
    if gazetteer is None:
        return []
    hits = gazetteer.match(sentence)
    out: List[Dict[str, Any]] = []
    for h in hits:
        out.append(
            {
                "start_char": int(h["start_char"]),
                "end_char": int(h["end_char"]),
                "text": h["text"],
                "type": h.get("type"),
                "source": "gazetteer",
                "meta": {"rule_id": h.get("rule_id"), "backend": h.get("source")},
            }
        )
    return out
