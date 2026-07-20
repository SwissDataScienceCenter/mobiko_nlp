"""
relation_annotation_page.py
----------------------------
Drop-in relation annotation page for the BioNER Streamlit tool.

Integration into streamlit_app_annotate.py
-------------------------------------------
1. Import at the top:
       from relation_annotation_page import render_relation_page

2. Wrap the existing app body in a mode selector.  After
   `st.set_page_config(...)` and the file-upload block, add:

       mode = st.sidebar.radio("Annotation mode", ["NER", "Relations"],
                               horizontal=True)
       if mode == "Relations":
           render_relation_page(model)   # pass the loaded model dict
           st.stop()
       # ... existing NER rendering continues below ...

   `model` is the dict already built by `_load_sets()` in the NER page
   (keyed by (doc_id, sent_idx), values are lists of sentence dicts).
   If you call _load_sets() before the mode branch both pages share the
   same uploaded file and save directory.
"""

from __future__ import annotations

import html as html_mod
import io
import json
import os
import re as re_mod
import tempfile
import datetime
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple

import streamlit as st

# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

RELATION_TYPES = [
    "HAS_PROPERTY",
    "IS_PART_OF",
    "LOCATED_IN",
    "AFFECTS",
    "HAS_PROCESS",
    "COMPARES_TO",
    "RELATED_TO",
    "CAUSES",
    "DURING",
]

RE_SAVE_DIR = "/mydata/mobiko/anisia/data/aug_runs"
RE_MAX_SNAPSHOTS = 50

_CUSTOM_RELATION_SENTINEL = "＋ Custom relation…"

# One background colour per entity type family.
# Chosen to be distinct but soft; same palette as the NER mockup.
_TYPE_COLORS: Dict[str, Tuple[str, str]] = {
    # (background, text-on-background)
    "BIOTIC ENTITY":              ("#a5d6a7", "#1b5e20"),
    "BIOTIC COLLECTIVE ENTITY":   ("#69b06b", "#143d15"),
    "BIOTIC PROCESS":             ("#b2dfdb", "#004d40"),
    "BIOTIC PROPERTY":            ("#80cbc4", "#003d33"),
    "ABIOTIC ENTITY":             ("#b2ebf2", "#006064"),
    "ABIOTIC COLLECTIVE ENTITY":  ("#4dd0e1", "#004d55"),
    "ABIOTIC PROCESS":            ("#b3e5fc", "#01579b"),
    "ABIOTIC PROPERTY":           ("#81d4fa", "#014e89"),
    "SPATIAL ENTITY":             ("#ffe0b2", "#bf360c"),
    "SPATIAL PROPERTY":           ("#ffcc80", "#a33000"),
    "TEMPORAL ENTITY":            ("#e1bee7", "#4a148c"),
    "TEMPORAL PROPERTY":          ("#ce93d8", "#38006b"),
    "ANTHROPOGENIC ENTITY":       ("#f8bbd0", "#880e4f"),
    "ANTHROPOGENIC PROCESS":      ("#f48fb1", "#6d0f40"),
    "ANTHROPOGENIC PROPERTY":     ("#f06292", "#560a30"),
    "QUALITATIVE PROPERTY":       ("#f8bbd0", "#880e4f"),
    "QUANTITATIVE PROPERTY":      ("#b2ebf2", "#006064"),
    "CONCEPT":                    ("#c8e6c9", "#1b5e20"),
}
_DEFAULT_COLOR = ("#e0e0e0", "#333333")


def _span_color(entity_type: str) -> Tuple[str, str]:
    return _TYPE_COLORS.get(entity_type, _DEFAULT_COLOR)


# ------------------------------------------------------------------ #
#  UID helpers                                                        #
# ------------------------------------------------------------------ #

def _uid() -> str:
    return uuid.uuid4().hex[:10]


# ------------------------------------------------------------------ #
#  HTML rendering                                                     #
# ------------------------------------------------------------------ #

def _build_entity_html(text: str, spans: List[Dict], hints: List[Dict] | None = None) -> str:
    """
    Return an HTML string with all spans highlighted in their type colour.
    Overlapping spans: inner-most wins (last boundary wins in tie).
    Mark's relation hints (if given) are underlined on top of that.
    Spans/hints with missing char offsets are skipped rather than crashing.
    """
    n = len(text)
    # Build a character-level label array (entity type or None)
    labels: List[str | None] = [None] * n
    valid_spans = [
        sp for sp in spans
        if sp.get("start_char") is not None and sp.get("end_char") is not None
    ]
    # Sort by span length descending so wider spans paint first and
    # narrower (more specific) spans overwrite them
    for sp in sorted(valid_spans, key=lambda s: s["end_char"] - s["start_char"], reverse=True):
        a = max(0, int(sp["start_char"]))
        b = min(n, int(sp["end_char"]))
        for i in range(a, b):
            labels[i] = sp.get("type", "")

    hint_labels: List[str | None] = [None] * n
    for h in (hints or []):
        if h.get("start_char") is None or h.get("end_char") is None:
            continue
        a = max(0, int(h["start_char"]))
        b = min(n, int(h["end_char"]))
        for i in range(a, b):
            hint_labels[i] = h.get("label", "")

    # Build fragments: consecutive chars with same (label, hint_label) pair
    fragments: List[Tuple[str, str | None, str | None]] = []
    i = 0
    while i < n:
        lab = labels[i]
        hlab = hint_labels[i]
        j = i + 1
        while j < n and labels[j] == lab and hint_labels[j] == hlab:
            j += 1
        fragments.append((text[i:j], lab, hlab))
        i = j

    parts = []
    for seg, lab, hlab in fragments:
        esc = html_mod.escape(seg)
        styles = []
        titles = []
        if lab is not None:
            bg, fg = _span_color(lab)
            styles.append(f"background:{bg};color:{fg};border-radius:4px;padding:1px 4px;")
            titles.append(lab)
        if hlab is not None:
            styles.append("border-bottom:3px dotted #d84315;")
            titles.append(f"Mark's relation hint: {hlab}")
        if styles:
            parts.append(
                f'<mark style="{"".join(styles)}" '
                f'title="{html_mod.escape(" / ".join(titles))}">{esc}</mark>'
            )
        else:
            parts.append(esc)

    css = """
    <style>
      .re-sent {
        white-space: pre-wrap;
        word-break: break-word;
        font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Arial;
        font-size: 14px;
        line-height: 2;
        background: #f7f9ff;
        border: 1px solid #dbe1ff;
        border-radius: 10px;
        padding: 10px 14px;
        color: #0c1222;
      }
      mark { cursor: default; }
    </style>
    """
    return css + f'<div class="re-sent">{"".join(parts)}</div>'


# ------------------------------------------------------------------ #
#  Session-state helpers                                              #
# ------------------------------------------------------------------ #
#  Save / autosave utilities (mirrors NER page behaviour)             #
# ------------------------------------------------------------------ #

def _atomic_write_re(text: str, dst_path: str):
    """Write atomically: temp file → fsync → rename."""
    dirpath = os.path.dirname(dst_path)
    fd, tmppath = tempfile.mkstemp(prefix=".tmp_re_", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmppath, dst_path)
        dir_fd = os.open(dirpath, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(tmppath):
            try:
                os.remove(tmppath)
            except Exception:
                pass


def _safe_dirname_re(name: str) -> str:
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    safe = re_mod.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return safe or "re_upload"


def _normalize_relation_label(text: str) -> str:
    """Normalize a free-typed relation label to the UPPER_SNAKE_CASE
    convention used by the fixed RELATION_TYPES, e.g. "migrates to" ->
    "MIGRATES_TO". Returns "" if nothing usable remains."""
    safe = re_mod.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_")
    return safe.upper()


def _prune_re_snapshots(save_dir: str):
    if not save_dir or not os.path.isdir(save_dir):
        return
    files = [f for f in os.listdir(save_dir)
             if f.startswith("re_") and f.endswith(".jsonl")]
    for f in sorted(files, reverse=True)[RE_MAX_SNAPSHOTS:]:
        try:
            os.remove(os.path.join(save_dir, f))
        except Exception:
            pass


def _re_current_paths():
    save_dir = st.session_state.get("re_upload_dir")
    if not save_dir:
        return None, None
    run_id = st.session_state.get("re_run_id", "unknown")
    run_path    = os.path.join(save_dir, f"re_{run_id}.jsonl")
    latest_path = os.path.join(save_dir, "re_latest.jsonl")
    return run_path, latest_path


def _maybe_autosave_re(model, keys):
    run_path, latest_path = _re_current_paths()
    if not run_path or not latest_path:
        return
    payload = export_relations_jsonl(model, keys)
    _atomic_write_re(payload, run_path)
    _atomic_write_re(payload, latest_path)
    _prune_re_snapshots(os.path.dirname(run_path))


# ------------------------------------------------------------------ #
#  JSON loader (RE files are JSON, not JSONL)                         #
# ------------------------------------------------------------------ #

def load_json_grouped(file_buffer) -> Dict:
    """
    Parse a JSON file into the same (doc_id, sent_idx) → [sent_dict] structure
    used throughout the app.  Accepts three shapes:
      - A single record:  {"doc_id": "...", "sentences": [...]}
      - A list of records: [{"doc_id": "...", "sentences": [...]}, ...]
      - A plain list of sentences (no doc_id wrapper): [{"text":..., "spans":...}, ...]
    """
    raw = file_buffer.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)

    spans_by_key: Dict = defaultdict(list)

    # Normalise to a list of doc-level records
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        # Check whether items are doc records or bare sentences
        if data and isinstance(data[0], dict) and "sentences" in data[0]:
            records = data
        else:
            # Bare sentence list — wrap in a single synthetic record
            records = [{"doc_id": "doc_0", "sentences": data}]
    else:
        raise ValueError(f"Unexpected JSON shape: {type(data)}")

    for rec in records:
        doc_id = rec.get("doc_id", "unknown")
        for i, sent in enumerate(rec.get("sentences", [])):
            spans_by_key[(doc_id, i)].append(sent)

    return spans_by_key


# ------------------------------------------------------------------ #

def _init_rel_state(keys: List[Tuple]):
    if "rel_gold" not in st.session_state:
        st.session_state.rel_gold = {k: [] for k in keys}
    else:
        # Ensure all keys exist (new file loaded)
        for k in keys:
            st.session_state.rel_gold.setdefault(k, [])


def _get_spans(model: Dict, key: Tuple) -> List[Dict]:
    """Return the spans list for a (doc_id, sent_idx) key."""
    for sent_dict in model.get(key, []):
        spans = sent_dict.get("spans", [])
        if spans:
            return spans
    return []


def _get_hints(model: Dict, key: Tuple) -> List[Dict]:
    """Return Mark's relation_hints list (if any) for a (doc_id, sent_idx) key."""
    for sent_dict in model.get(key, []):
        hints = sent_dict.get("relation_hints", [])
        if hints:
            return hints
    return []


def _get_text(model: Dict, key: Tuple) -> str:
    for sent_dict in model.get(key, []):
        t = sent_dict.get("text", "")
        if t:
            return t
    return ""


# ------------------------------------------------------------------ #
#  Export                                                             #
# ------------------------------------------------------------------ #

def export_relations_jsonl(model: Dict, keys: List[Tuple]) -> str:
    """
    Produce a JSONL where every sentence has both its original 'spans'
    and the annotated 'relations' list.
    Output format per sentence:
        {
          "text": "...",
          "spans": [...],
          "relations": [
            {"uid": "...", "relation": "HAS_PROPERTY",
             "e1": {span dict}, "e2": {span dict},
             "note": "..."}
          ]
        }
    """
    buf = io.StringIO()
    docs: Dict[str, List[Tuple[int, Tuple]]] = defaultdict(list)
    for key in keys:
        doc_id, sent_idx = key
        docs[doc_id].append((sent_idx, key))

    for doc_id, items in docs.items():
        items.sort(key=lambda t: t[0])
        sentences = []
        for sent_idx, key in items:
            text = _get_text(model, key)
            spans = _get_spans(model, key)
            rels = st.session_state.rel_gold.get(key, [])
            sentences.append({
                "text": text,
                "spans": spans,
                "relations": rels,
            })
        buf.write(json.dumps({"doc_id": doc_id, "sentences": sentences},
                              ensure_ascii=False) + "\n")
    return buf.getvalue()


# ------------------------------------------------------------------ #
#  Main page renderer                                                 #
# ------------------------------------------------------------------ #

def render_relation_page(_unused_model: Dict = None):
    """
    Self-contained relation annotation page with its own file uploader.
    Accepts an optional _unused_model for backwards-compatibility with the
    integration call site, but ignores it — RE files are JSON, not JSONL.
    """
    st.markdown("### Annotator")
    username = st.text_input(
        "Your username (keeps your saved annotations separate from other "
        "annotators working on the same file)",
        value=st.session_state.get("re_username", ""),
        key="re_username_input",
    ).strip()
    if not username:
        st.info("Enter your username above to start annotating.")
        return
    st.session_state["re_username"] = username

    st.markdown("### Input (relation annotation)")
    re_file = st.file_uploader(
        "Upload JSON file with pre-labelled entities", type=["json"]
    )

    if re_file is None:
        st.info("Upload a JSON file to start annotating relations.")
        return

    # Re-parse only when a new file is uploaded
    if st.session_state.get("re_upload_name") != re_file.name:
        try:
            model = load_json_grouped(re_file)
        except Exception as exc:
            st.error(f"Could not parse JSON: {exc}")
            return
        st.session_state.re_model = model
        st.session_state.re_upload_name = re_file.name
        # Reset relation annotations when file changes
        st.session_state.rel_gold = {}
        # Fresh run ID for this file
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        st.session_state.re_run_id = f"{ts}-{uuid.uuid4().hex[:6]}"
        st.session_state.re_upload_dir_username = None  # force dir (re)computation below

    # (Re)compute the save directory whenever the file or the username
    # changes — nested under the annotator's username so two people
    # annotating the SAME uploaded file get separate autosave/latest files
    # instead of overwriting each other's "re_latest.jsonl". A username
    # change alone does NOT reset the in-progress annotations, only where
    # they're saved from now on.
    if st.session_state.get("re_upload_dir_username") != username:
        upload_dir = os.path.join(
            RE_SAVE_DIR, _safe_dirname_re(re_file.name) + "_relation",
            _safe_dirname_re(username),
        )
        os.makedirs(upload_dir, exist_ok=True)
        st.session_state.re_upload_dir = upload_dir
        st.session_state.re_upload_dir_username = username

    model = st.session_state.get("re_model")
    if not model:
        st.error("File loaded but no sentences found.")
        return

    keys = sorted(model.keys())
    _init_rel_state(keys)

    # ---- sentence navigation ----------------------------------------
    if "re_sentence_idx" not in st.session_state:
        st.session_state.re_sentence_idx = 0
    st.session_state.re_sentence_idx = max(
        0, min(st.session_state.re_sentence_idx, len(keys) - 1)
    )

    nav1, nav2, nav3, nav4 = st.columns([1, 4, 2, 1])
    with nav1:
        if st.button("◀ Prev", key="re_prev",
                     disabled=st.session_state.re_sentence_idx == 0):
            st.session_state.re_sentence_idx -= 1
            st.rerun()
    with nav2:
        idx = st.session_state.re_sentence_idx
        rel_count = len(st.session_state.rel_gold.get(keys[idx], []))
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>"
            f"Sentence {idx + 1} / {len(keys)} — "
            f"<b>{rel_count}</b> relation(s) annotated</div>",
            unsafe_allow_html=True,
        )
    with nav3:
        jump = st.number_input(
            "Jump to", min_value=1, max_value=len(keys),
            value=st.session_state.re_sentence_idx + 1,
            step=1, label_visibility="collapsed", key="re_jump"
        )
        if jump - 1 != st.session_state.re_sentence_idx:
            st.session_state.re_sentence_idx = jump - 1
            st.rerun()
    with nav4:
        if st.button("Next ▶", key="re_next",
                     disabled=st.session_state.re_sentence_idx == len(keys) - 1):
            st.session_state.re_sentence_idx += 1
            st.rerun()

    idx = st.session_state.re_sentence_idx
    cur_key = keys[idx]
    text = _get_text(model, cur_key)
    spans = _get_spans(model, cur_key)
    hints = _get_hints(model, cur_key)

    st.subheader(f"{cur_key[0]} · sentence {cur_key[1]}")

    # ---- two-column layout ------------------------------------------
    col_left, col_right = st.columns([3, 2], gap="large")

    # ---- LEFT: sentence + entity index ------------------------------
    with col_left:
        st.markdown("#### Sentence")
        if hasattr(st, "html"):
            st.html(_build_entity_html(text, spans, hints))
        else:
            st.components.v1.html(_build_entity_html(text, spans, hints), height=200, scrolling=True)

        st.markdown("#### Entities")
        if not spans:
            st.info("No entities on this sentence.")
        else:
            for i, sp in enumerate(spans):
                bg, fg = _span_color(sp.get("type", ""))
                label_html = (
                    f'<span style="background:{bg};color:{fg};'
                    f'border-radius:4px;padding:2px 7px;font-size:12px;'
                    f'font-weight:500;margin-right:6px">{html_mod.escape(sp["text"])}</span>'
                    f'<span style="font-size:11px;color:#888">{html_mod.escape(sp.get("type",""))}</span>'
                )
                concept_text = sp.get("concept_text")
                if concept_text and concept_text != sp.get("text"):
                    label_html += (
                        f'<span style="font-size:11px;color:#666;margin-left:8px">'
                        f'→ concept: <i>{html_mod.escape(concept_text)}</i></span>'
                    )
                st.markdown(
                    f'<div style="padding:2px 0"><span style="color:#aaa;font-size:10px;'
                    f'margin-right:6px">E{i+1}</span>{label_html}</div>',
                    unsafe_allow_html=True,
                )

        if hints:
            st.markdown("#### Mark's relation hints")
            st.caption("Preliminary relation phrases from Mark's annotation — "
                       "pick one below to suggest its label.")
            for i, h in enumerate(hints):
                st.markdown(
                    f'<div style="padding:2px 0"><span style="color:#aaa;font-size:10px;'
                    f'margin-right:6px">R{i+1}</span>'
                    f'<span style="border-bottom:3px dotted #d84315;font-size:12px;'
                    f'margin-right:6px">{html_mod.escape(h.get("text",""))}</span>'
                    f'<span style="font-size:11px;color:#888">→ {html_mod.escape(h.get("label",""))}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ---- RIGHT: add form + current relations ------------------------
    with col_right:
        st.markdown("#### Add relation")

        if not spans:
            st.warning("No entities — cannot add relations to this sentence.")
        else:
            def _entity_label(i: int, sp: Dict) -> str:
                label = f"E{i+1} — {sp['text']} ({sp.get('type','')})"
                concept_text = sp.get("concept_text")
                if concept_text and concept_text != sp.get("text"):
                    label += f" → concept: {concept_text}"
                return label

            entity_labels = [_entity_label(i, sp) for i, sp in enumerate(spans)]

            with st.container(border=True):
                e1_idx = st.selectbox(
                    "Source entity (E1)",
                    options=range(len(spans)),
                    format_func=lambda i: entity_labels[i],
                    key=f"re_e1_{cur_key}",
                )

                hint_options = ["(none — pick manually)"] + [
                    f'R{i+1} — "{h.get("text","")}" → {h.get("label","")}'
                    for i, h in enumerate(hints)
                ]
                hint_sel = st.selectbox(
                    "Suggest from Mark's relation hint (optional)",
                    options=range(len(hint_options)),
                    format_func=lambda i: hint_options[i],
                    key=f"re_hint_{cur_key}",
                ) if hints else 0
                suggested_label = hints[hint_sel - 1]["label"] if hint_sel else None

                # Custom relation types typed in earlier this session are
                # remembered so they reappear as regular options afterwards.
                custom_types = st.session_state.setdefault("re_custom_relation_types", [])
                rel_options = list(RELATION_TYPES)
                for t in custom_types:
                    if t not in rel_options:
                        rel_options.append(t)
                if suggested_label and suggested_label not in rel_options:
                    rel_options.append(suggested_label)
                default_index = rel_options.index(suggested_label) if suggested_label else 0
                rel_type_sel = st.selectbox(
                    "Relation type",
                    options=rel_options + [_CUSTOM_RELATION_SENTINEL],
                    index=default_index,
                    # Key varies with the chosen hint so a new hint pick
                    # refreshes the default, while the user can still
                    # freely override it without the selection resetting.
                    key=f"re_rel_{cur_key}_{hint_sel}",
                )
                if rel_type_sel == _CUSTOM_RELATION_SENTINEL:
                    custom_input = st.text_input(
                        "New relation type (letters/numbers only, e.g. MIGRATES_TO)",
                        key=f"re_rel_custom_{cur_key}_{hint_sel}",
                    )
                    rel_type = _normalize_relation_label(custom_input)
                    if custom_input and not rel_type:
                        st.warning("Enter at least one letter or number for the new relation type.")
                else:
                    rel_type = rel_type_sel
                e2_idx = st.selectbox(
                    "Target entity (E2)",
                    options=range(len(spans)),
                    format_func=lambda i: entity_labels[i],
                    key=f"re_e2_{cur_key}",
                )
                note = st.text_input(
                    "Note / reasoning (optional)",
                    key=f"re_note_{cur_key}",
                )

                same_entity = (e1_idx == e2_idx)
                if same_entity:
                    st.warning("Source and target must be different entities.")
                missing_custom = rel_type_sel == _CUSTOM_RELATION_SENTINEL and not rel_type
                if missing_custom:
                    st.warning("Type a name for the new relation, or pick one from the list.")

                if st.button(
                    "＋ Add relation",
                    type="primary",
                    disabled=(same_entity or missing_custom),
                    key=f"re_add_{cur_key}",
                ):
                    if rel_type not in custom_types and rel_type not in RELATION_TYPES:
                        custom_types.append(rel_type)
                    new_rel = {
                        "uid": _uid(),
                        "relation": rel_type,
                        "e1": spans[e1_idx],
                        "e2": spans[e2_idx],
                        "note": note.strip(),
                        "annotator": username,
                    }
                    st.session_state.rel_gold[cur_key].append(new_rel)
                    _maybe_autosave_re(model, keys)
                    st.rerun()

        # ---- current relations list ---------------------------------
        st.markdown("#### Current relations")
        rels = st.session_state.rel_gold.get(cur_key, [])

        if not rels:
            st.caption("No relations yet for this sentence.")
        else:
            to_delete = []
            for j, rel in enumerate(rels):
                e1 = rel["e1"]
                e2 = rel["e2"]
                uid = rel.get("uid", str(j))
                bg1, fg1 = _span_color(e1.get("type", ""))
                bg2, fg2 = _span_color(e2.get("type", ""))

                with st.container(border=True):
                    # concept_text (if any) shown as a hover tooltip on the chip
                    e1_title = html_mod.escape(e1.get("concept_text") or "")
                    e2_title = html_mod.escape(e2.get("concept_text") or "")
                    e1_title_attr = f' title="concept: {e1_title}"' if e1_title else ""
                    e2_title_attr = f' title="concept: {e2_title}"' if e2_title else ""
                    # Display row: E1 chip → relation badge → E2 chip
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:6px;'
                        f'flex-wrap:wrap;margin-bottom:4px">'
                        f'<span style="background:{bg1};color:{fg1};border-radius:4px;'
                        f'padding:2px 7px;font-size:12px;font-weight:500"{e1_title_attr}>'
                        f'{html_mod.escape(e1["text"])}</span>'
                        f'<span style="font-size:12px;color:#888">→</span>'
                        f'<span style="background:#f0f0f0;color:#444;border-radius:4px;'
                        f'padding:2px 7px;font-size:11px;font-weight:500;border:0.5px solid #ccc">'
                        f'{html_mod.escape(rel["relation"])}</span>'
                        f'<span style="font-size:12px;color:#888">→</span>'
                        f'<span style="background:{bg2};color:{fg2};border-radius:4px;'
                        f'padding:2px 7px;font-size:12px;font-weight:500"{e2_title_attr}>'
                        f'{html_mod.escape(e2["text"])}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Editable relation type + delete
                    c_type, c_saved, c_del = st.columns([3, 2, 1])
                    with c_type:
                        # Include the saved value even if it's a custom label
                        # (e.g. one of Mark's hints) not in the fixed list —
                        # otherwise the selectbox would fall back to the first
                        # option and silently overwrite it on the next edit.
                        type_options = list(RELATION_TYPES)
                        if rel["relation"] not in type_options:
                            type_options.append(rel["relation"])
                        new_type = st.selectbox(
                            "Relation",
                            options=type_options,
                            index=type_options.index(rel["relation"]),
                            key=f"re_type_{uid}",
                            label_visibility="collapsed",
                        )
                    with c_saved:
                        # Show "✓ saved" for exactly one render cycle after a change
                        if st.session_state.get("re_just_saved") == uid:
                            st.markdown(
                                '<span style="color:#1D9E75;font-size:12px">✓ saved</span>',
                                unsafe_allow_html=True,
                            )
                            del st.session_state["re_just_saved"]
                    with c_del:
                        st.write("")
                        if st.button("Delete", key=f"re_del_{uid}"):
                            to_delete.append(j)

                    # Detect change → persist + flag → rerun to show indicator
                    if new_type != rel["relation"]:
                        st.session_state.rel_gold[cur_key][j]["relation"] = new_type
                        _maybe_autosave_re(model, keys)
                        st.session_state["re_just_saved"] = uid
                        st.rerun()

                    if rel.get("note"):
                        st.caption(f'📝 {rel["note"]}')

            if to_delete:
                st.session_state.rel_gold[cur_key] = [
                    r for k, r in enumerate(rels) if k not in to_delete
                ]
                _maybe_autosave_re(model, keys)
                st.rerun()

    # ---- Export -----------------------------------------------------
    st.divider()
    st.subheader("Export")

    total_rels = sum(len(v) for v in st.session_state.rel_gold.values())
    sents_with_rels = sum(1 for v in st.session_state.rel_gold.values() if v)
    st.caption(
        f"{total_rels} relation(s) across {sents_with_rels} / {len(keys)} sentences."
    )

    run_path, latest_path = _re_current_paths()
    if run_path and latest_path:
        st.caption(f"Autosave: session file → {run_path}")
        st.caption(f"Latest snapshot → {latest_path}")
    else:
        st.caption("Autosave: waiting for an uploaded JSON file.")

    jsonl_out = export_relations_jsonl(model, keys)
    st.download_button(
        "Download relations JSONL",
        data=jsonl_out.encode("utf-8"),
        file_name=f"relations_annotated_{_safe_dirname_re(username)}.jsonl",
        mime="application/jsonl",
    )
