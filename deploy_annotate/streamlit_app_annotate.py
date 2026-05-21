import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import os, datetime, tempfile, re
import io
import streamlit.components.v1 as components
import streamlit as st
from relation_annotation_page import render_relation_page

# ---------- Persist options defaults ----------

SAVE_DIR = "/mydata/mobiko/anisia/data/aug_runs"
os.makedirs(SAVE_DIR, exist_ok=True)

CUSTOM = "(custom)"


SCHEMA_TYPES_SHORT_TO_TEST = [
    "ABIOTIC ENTITY",
    "ABIOTIC PROCESS",
    "ABIOTIC PROPERTY",
    "ANTHROPOGENIC ENTITY",
    "ANTHROPOGENIC PROCESS",
    "ANTHROPOGENIC PROPERTY",
    "BIOTIC ENTITY",
    "BIOTIC PROCESS",
    "BIOTIC PROPERTY",
    "CONCEPT",
    "SPATIAL ENTITY",
    "SPATIAL PROPERTY",
    "TEMPORAL ENTITY",
    "TEMPORAL PROPERTY",
    "QUALITATIVE PROPERTY",
    "QUANTITATIVE PROPERTY",
]



if "autosave" not in st.session_state:
    st.session_state.autosave = True

import uuid

import zlib

def wkey(*parts: str) -> str:
    # Namespaced by current sentence so keys don't collide across sentences
    doc, sent = cur_key
    return "w|" + str(doc) + "|" + str(sent) + "|" + "|".join(str(p) for p in parts)


def prop_sig(span: dict, text: str) -> str:
    # Stable signature for a proposal row
    a = int(span.get("start_char", -1))
    b = int(span.get("end_char", -1))
    t = (span.get("type") or span.get("label") or span.get("entity") or span.get("category") or "")
    # include a short hash of the covered text to avoid collisions when two props share a,b,t
    snippet = text[max(0, a):max(0, min(len(text), b))]
    h = zlib.crc32(snippet.encode("utf-8")) & 0xFFFFFFFF
    return f"{a}-{b}-{t}-{h:08x}"


def _uid() -> str:
    return uuid.uuid4().hex[:10]

def ensure_uids(spans: list[dict]) -> list[dict]:
    """Ensure each span has a stable 'uid'. Mutates in place and returns the list."""
    for s in spans:
        if "uid" not in s or not s["uid"]:
            s["uid"] = _uid()
    return spans


def _apply_hotadd_from_query(cur_key, text, proposals, default_label_options):
    try:
        q = st.query_params
    except Exception:
        return False
    token = (q.get("hotadd") or [None])[0]
    if not token:
        return False

    try:
        doc, idx_s, a_s, b_s = token.split("|")
        idx_i = int(idx_s); a = int(a_s); b = int(b_s)
    except Exception:
        q.clear()  # to clear
        return False

    if not (cur_key[0] == doc and cur_key[1] == idx_i):
        q.clear()  # to clear
        return False

    # label heuristic
    label = None
    for s in proposals:
        if int(s.get("start_char",-1)) == a and int(s.get("end_char",-1)) == b and s.get("type"):
            label = s["type"]; break
    if label is None:
        overlaps = [s for s in proposals if not (b <= int(s.get("start_char",0)) or a >= int(s.get("end_char",0)))]
        if len(overlaps) == 1 and overlaps[0].get("type"):
            label = overlaps[0]["type"]
    if label is None:
        label = st.session_state.get("hot_lbl") or next((x for x in default_label_options if x != CUSTOM), "MISC")

    new_span = {"start_char": a, "end_char": b, "type": label}
    st.session_state.aug_gold[cur_key] = merge_span_into_gold(
        st.session_state.aug_gold[cur_key], new_span, metric="dice", thr=0.8, do_merge=True, text=text
    )
    _maybe_autosave()

    q.clear()  # to clear
    force_rerun()
    return True


# ---------- Rerun compatibility ----------

def force_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        try:
            from streamlit.runtime.scriptrunner import RerunException, RerunData
            raise RerunException(RerunData(None))
        except Exception:
            pass

# ---------- Export (define early so autosave can call it) ----------

def _atomic_write(text: str, dst_path: str):
    """Write text to a temp file and atomically replace dst_path."""
    dirpath = os.path.dirname(dst_path)
    fd, tmppath = tempfile.mkstemp(prefix=".tmp_", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmppath, dst_path)
        # fsync the directory to persist the rename on crash/power loss
        dir_fd = os.open(dirpath, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        # If anything exploded before replace, clean the tmp
        if os.path.exists(tmppath):
            try: os.remove(tmppath)
            except Exception: pass


def _safe_dirname(name: str) -> str:
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return safe or "upload"


def _current_paths():
    save_dir = st.session_state.get("upload_dir")
    if not save_dir:
        return None, None
    run_path = os.path.join(save_dir, f"aug_{st.session_state.run_id}.jsonl")
    latest_path = os.path.join(save_dir, "aug_latest.jsonl")
    return run_path, latest_path


def _prune_snapshots(save_dir: str):
    """Keep only the newest MAX_SNAPSHOTS aug_*.jsonl files in save_dir."""
    if not save_dir or not os.path.isdir(save_dir):
        return
    files = [f for f in os.listdir(save_dir) if f.startswith("aug_") and f.endswith(".jsonl")]
    files = sorted(files, reverse=True)
    for f in files[MAX_SNAPSHOTS:]:
        try: os.remove(os.path.join(save_dir, f))
        except Exception:
            pass


def export_augmented_jsonl():
    buf = io.StringIO()
    docs = defaultdict(list)
    for (doc, idx2), spans in st.session_state.aug_gold.items():
        docs[doc].append((idx2, spans))
    for doc_id, items in docs.items():
        items.sort(key=lambda t: t[0])
        sentences=[]
        for (i, _spans) in items:
            text_here = extract_text(gold.get((doc_id,i), []) or model.get((doc_id,i), []))
            fixed_spans, _ = enforce_text_consistency(_spans, text_here, repair=True)
            sentences.append({"text": text_here, "spans": fixed_spans})
        rec = {"doc_id": doc_id, "sentences": sentences}
        buf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return buf.getvalue()


def _maybe_autosave():
    if not st.session_state.get("autosave", True):
        return

    # Validate & repair all sentences before export
    any_mismatch = False
    repaired = {}
    for (doc, idx2), spans in st.session_state.aug_gold.items():
        text_here = extract_text(gold.get((doc,idx2), []) or model.get((doc,idx2), []))
        fixed, mism = enforce_text_consistency(spans, text_here, repair=True)
        if mism:
            any_mismatch = True
        repaired[(doc, idx2)] = fixed
    if any_mismatch:
        st.info("Some spans had mismatched 'text' vs indices; corrected before save.")
    st.session_state.aug_gold.update(repaired)

    payload = export_augmented_jsonl()

    run_path, latest_path = _current_paths()
    if not run_path or not latest_path:
        return

    # 1) Write to this session's RUN_PATH
    _atomic_write(payload, run_path)
    # 2) Update the rolling latest pointer
    _atomic_write(payload, latest_path)
    # 3) Prune old snapshots
    _prune_snapshots(os.path.dirname(run_path))


# -------------------- Span math --------------------

def span_len(a, b): return max(0, int(b) - int(a))

def overlap(a0, a1, b0, b1):
    s = max(int(a0), int(b0)); e = min(int(a1), int(b1))
    return max(0, e - s)

def dice(a0, a1, b0, b1):
    inter = overlap(a0,a1,b0,b1); denom = span_len(a0,a1) + span_len(b0,b1)
    return (2*inter/denom) if denom > 0 else 0.0

def iou(a0, a1, b0, b1):
    inter = overlap(a0,a1,b0,b1); uni = span_len(a0,a1) + span_len(b0,b1) - inter
    return (inter/uni) if uni > 0 else 0.0

def overlap_min(a0,a1,b0,b1):
    inter = overlap(a0,a1,b0,b1); denom = min(span_len(a0,a1), span_len(b0,b1))
    return (inter/denom) if denom > 0 else 0.0

def score_pair(p, g, metric):
    a0,a1 = int(p["start_char"]), int(p["end_char"])
    b0,b1 = int(g["start_char"]), int(g["end_char"])
    if metric == "iou": return iou(a0,a1,b0,b1)
    if metric == "min": return overlap_min(a0,a1,b0,b1)
    return dice(a0,a1,b0,b1)

# -------------------- Normalization --------------------

def normalize_span(s: Dict[str, Any]) -> Dict[str, Any]:
    s["start_char"] = int(s.get("start_char", 0))
    s["end_char"]   = int(s.get("end_char", 0))
    if "type" not in s or not s.get("type"):
        for k in ("label", "ent_type", "entity", "category"):
            if k in s and s[k]:
                s["type"] = s[k]; break
    return s

# -------------------- I/O --------------------

def load_jsonl_grouped(path_or_buffer):
    """(doc_id, sent_idx) -> [sentence dicts]"""
    spans_by_key = defaultdict(list)
    if hasattr(path_or_buffer, "read"):
        text = path_or_buffer.read()
        if isinstance(text, bytes): text = text.decode("utf-8")
        lines = text.splitlines()
    else:
        with open(path_or_buffer, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    for line in lines:
        if not line.strip(): continue
        r = json.loads(line)
        doc_id = r.get("doc_id")
        for i, sent in enumerate(r.get("sentences", [])):
            spans_by_key[(doc_id, i)].append(sent)
    return spans_by_key


def extract_text(value_list: List[Dict]) -> str:
    for inst in value_list:
        t = inst.get("text") or ""
        if t: return t
    return ""


def get_gold_spans(value_list: List[Dict]) -> List[Dict]:
    out = []
    for inst in value_list:
        out.extend([normalize_span(x) for x in (inst.get("spans", []) or [])])
    return out

def get_pred_spans(value_list: List[Dict], source="accepted_and_missing") -> List[Dict]:
    out = []
    for inst in value_list:
        llm = inst.get("llm", {}) or {}
        if source == "accepted":
            out.extend([normalize_span(x) for x in (llm.get("accepted", []) or [])])
        else:
            out.extend([normalize_span(x) for x in (llm.get("accepted", []) or [])])
            out.extend([normalize_span(x) for x in (llm.get("missing", []) or [])])
    return out

# -------------------- Matching & categorization --------------------

def categorize_spans(gold_spans, pred_spans, thr, metric="dice", require_type=False,
                     allow_many_to_one=True, suppress_nested_fp=True):
    pairs = []
    for pi, p in enumerate(pred_spans):
        pt = p.get("type")
        for gi, g in enumerate(gold_spans):
            if require_type and pt != g.get("type"):
                continue
            s = score_pair(p, g, metric)
            if s >= thr:
                pairs.append((s, pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    matched_g, matched_p = {}, {}
    for s, pi, gi in pairs:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g[gi] = pi
        matched_p[pi] = gi
    # containment credit
    if allow_many_to_one:
        for gi, g in enumerate(gold_spans):
            if gi in matched_g:
                continue
            for pi, p in enumerate(pred_spans):
                if pi not in matched_p:
                    continue
                if require_type and p.get("type") != g.get("type"):
                    continue
                a0,a1 = int(p["start_char"]), int(p["end_char"])
                b0,b1 = int(g["start_char"]), int(g["end_char"])
                if b0 >= a0 and b1 <= a1:
                    matched_g[gi] = pi; break
    tp_idx = sorted(set(matched_p.keys()))
    fp_idx = [i for i in range(len(pred_spans)) if i not in matched_p]
    fn_idx = [i for i in range(len(gold_spans)) if i not in matched_g]
    if suppress_nested_fp and tp_idx:
        tp_regions = [(int(pred_spans[i]["start_char"]), int(pred_spans[i]["end_char"])) for i in tp_idx]
        keep = []
        for i in fp_idx:
            a0,a1 = int(pred_spans[i]["start_char"]), int(pred_spans[i]["end_char"])
            if any(a0 >= t0 and a1 <= t1 for (t0,t1) in tp_regions):
                continue
            keep.append(i)
        fp_idx = keep
    tp = [pred_spans[i] for i in tp_idx]
    fp = [pred_spans[i] for i in fp_idx]
    fn = [gold_spans[i] for i in fn_idx]
    return tp, fp, fn


# -------------------- Rendering (single layer + hover labels) --------------------

def _norm_spans_with_labels(text, spans):
    n=len(text); out=[]
    for s in spans:
        a=max(0,int(s["start_char"])); b=min(n,int(s["end_char"]))
        if a<b: out.append((a,b,s.get("type","")))
    return out


def merge_to_fragments(text, tp, fp, fn):
    tpN=_norm_spans_with_labels(text,tp); fpN=_norm_spans_with_labels(text,fp); fnN=_norm_spans_with_labels(text,fn)
    bounds={0,len(text)}
    for a,b,_ in tpN+fpN+fnN:
        bounds.add(a); bounds.add(b)
    bounds=sorted(bounds)
    def labels_overlapping(a,b, spans3):
        labs=set()
        for x0,x1,lab in spans3:
            if x0<b and x1>a and lab: labs.add(lab)
        return labs
    frags=[]
    for i in range(len(bounds)-1):
        a,b=bounds[i],bounds[i+1]
        seg=text[a:b]
        if not seg: continue
        cats=set()
        if any(x0<b and x1>a for x0,x1,_ in tpN): cats.add("tp")
        if any(x0<b and x1>a for x0,x1,_ in fpN): cats.add("fp")
        if any(x0<b and x1>a for x0,x1,_ in fnN): cats.add("fn")
        p_labels = labels_overlapping(a,b,tpN+fpN)
        g_labels = labels_overlapping(a,b,fnN)
        frags.append((seg,cats,p_labels,g_labels))
    merged=[]
    for seg,cats,pl,gl in frags:
        if merged and merged[-1][1]==cats and merged[-1][2]==pl and merged[-1][3]==gl:
            merged[-1]=(merged[-1][0]+seg,cats,pl,gl)
        else:
            merged.append((seg,cats,pl,gl))
    return merged


def build_single_layer_html(text, tp, fp, fn, show_tp, show_fp, show_fn,
                            word_spacing_em=0.06, light_bg=True, doc_id=None, sent_idx=None):
    import html as html_mod, json as _json

    fragments = merge_to_fragments(text, tp, fp, fn)
    enabled = {"tp": show_tp, "fp": show_fp, "fn": show_fn}
    mix = 0.35 if light_bg else 0.25

    def style_for_cats(cats):
        on = [c for c in cats if enabled.get(c, False)]
        if not on: return ("transparent", "none")
        if len(on) == 1:
            c = on[0]
            return (f"color-mix(in srgb, var(--{c}) {int(mix*100)}%, transparent)",
                    f"inset 0 0 0 1px color-mix(in srgb, var(--{c}) 60%, {'white' if light_bg else 'black'})")
        step = 6
        seq = ", ".join([f"var(--{c}) {i*step}px {(i+1)*step}px" for i,c in enumerate(on)])
        return (f"repeating-linear-gradient(135deg, {seq})", "inset 0 0 0 1px #0003")

    parts = []
    for seg, cats, p_labels, g_labels in fragments:
        esc = html_mod.escape(seg)
        if not cats:
            parts.append(esc); continue
        bg, box = style_for_cats(cats)
        pred_lbl = ", ".join(sorted(p_labels)) if p_labels else "—"
        gold_lbl = ", ".join(sorted(g_labels)) if g_labels else "—"
        title = f"Pred: {pred_lbl}\\nGold: {gold_lbl}"
        parts.append(f'<span class="mark" style="background:{bg}; box-shadow:{box};" title="{html_mod.escape(title)}">{esc}</span>')

    css = """
    <style>
      :root { --tp:#2e7d32; --fp:#c62828; --fn:#6a1b9a; }
      .sent-wrap { outline: none; }
      .sent-card { background:#f7f9ff; border:1px solid #dbe1ff; border-radius:12px; padding:10px; }
      .toolbar { display:flex; gap:8px; align-items:center; font-family:ui-sans-serif,-apple-system,Segoe UI,Roboto,Arial; margin-bottom:6px; }
      .btn { border:1px solid #c9d3ff; border-radius:8px; padding:4px 8px; background:white; cursor:pointer; }
      .btn:active { transform: translateY(1px); }
      .hint { font-size:12px; opacity:0.7; }
      .single-layer { white-space:pre-wrap; word-break:break-word; font-family:ui-sans-serif,-apple-system,Segoe UI,Roboto,Arial; color:#0c1222; }
      .mark { border-radius:4px; padding:0; }
      .mark:hover { outline:2px solid #0002; cursor:help; }
    </style>
    """ if light_bg else """
    <style>
      :root { --tp:#2e7d32; --fp:#c62828; --fn:#6a1b9a; }
      .sent-wrap { outline: none; }
      .sent-card { background:#0e1730; border:1px dashed #2a355e; border-radius:10px; padding:10px; }
      .toolbar { display:flex; gap:8px; align-items:center; font-family:ui-sans-serif,-apple-system,Segoe UI,Roboto,Arial; margin-bottom:6px; color:#e6ecff; }
      .btn { border:1px solid #344078; border-radius:8px; padding:4px 8px; background:#101a39; color:#e6ecff; cursor:pointer; }
      .btn:active { transform: translateY(1px); }
      .hint { font-size:12px; opacity:0.8; }
      .single-layer { white-space:pre-wrap; word-break:break-word; font-family:ui-sans-serif,-apple-system,Segoe UI,Roboto,Arial; color:#e6ecff; }
      .mark { border-radius:4px; padding:0; }
      .mark:hover { outline:2px solid #fff3; cursor:help; }
    </style>
    """

    html_core = (
        f"{css}"
        f"<div id='sent-wrap' class='sent-wrap' tabindex='0' data-doc='{html_mod.escape(str(doc_id) if doc_id is not None else '')}' "
        f"     data-idx='{html_mod.escape(str(sent_idx) if sent_idx is not None else '')}'>"
        f"  <div class='sent-card'>"
        f"    <div class='single-layer' style='word-spacing:{word_spacing_em}em;'>"
        f"      {''.join(parts)}"
        f"    </div>"
        f"  </div>"
        f"</div>"
    )
    return html_core


# -------------------- Annotation helpers --------------------

def set_span_text(s: dict, text: str) -> dict:
    """Return span with 'text' field set from [start_char:end_char] on provided text."""
    a = int(s.get("start_char", 0))
    b = int(s.get("end_char", 0))
    a = max(0, min(len(text), a))
    b = max(0, min(len(text), b))
    if b < a:  # normalize bad order if any
        a, b = b, a
    s["start_char"] = a
    s["end_char"] = b
    s["text"] = text[a:b]
    return s


def enforce_text_consistency(spans: list[dict], text: str, *, repair: bool = True) -> tuple[list[dict], list[dict]]:
    """
    Ensure span['text'] == text[start:end].
    If repair=True, overwrite 'text' to match indices and clamp indices to text bounds.
    Returns (fixed_spans, mismatches_list) where each mismatch is {'uid', 'was', 'now', 'a', 'b'}.
    Virtual spans (virtual=True) are passed through unchanged.
    """
    fixed = []
    mismatches = []
    n = len(text)
    for s in spans:
        if s.get("virtual"):
            fixed.append(s)
            continue
        a = max(0, min(n, int(s.get("start_char", 0))))
        b = max(0, min(n, int(s.get("end_char", 0))))
        if b < a:
            a, b = b, a
        expected = text[a:b]
        was = s.get("text", None)
        if was != expected:
            mismatches.append({"uid": s.get("uid"), "was": was, "now": expected, "a": a, "b": b})
        if repair:
            s = dict(s)  # avoid mutating caller unexpectedly
            s["start_char"], s["end_char"], s["text"] = a, b, expected
        fixed.append(s)
    return fixed, mismatches


def dedup_exact(spans: list[dict]) -> list[dict]:
    seen = set()
    out  = []
    for s in spans:
        if s.get("virtual"):
            key = (-1, -1, s.get("type",""), s.get("text",""))
        else:
            key = (int(s["start_char"]), int(s["end_char"]), s.get("type",""))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)  # keep original dict with its uid
    return out


def dedup_proposals(proposals, text):
    seen = set(); out = []
    for s in proposals:
        sig = prop_sig(s, text)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(s)
    return out


def merge_span_into_gold(gold_spans: list[dict], new_span: dict,
                         metric="dice", thr=0.8, do_merge=True, text: str | None = None) -> list[dict]:
    """Merge new_span; preserve UIDs on existing spans; if we merge INTO an existing span, keep that span's UID."""
    ns = {
        "start_char": int(new_span.get("start_char", -1)),
        "end_char":   int(new_span.get("end_char", -1)),
        "type":       new_span.get("type",""),
        "uid":        new_span.get("uid") or _uid(),  # new uid if truly new
    }
    if new_span.get("virtual"):
        ns["virtual"] = True
        ns["text"] = new_span.get("text", "")
    if not do_merge or new_span.get("virtual"):
        return dedup_exact(ensure_uids(gold_spans + [ns]))

    out = []
    merged = False

    for g in gold_spans:
        same_type = (g.get("type","") == ns.get("type",""))
        if same_type and score_pair(ns, g, metric) >= thr:
            # Merge ranges, but KEEP the existing span's UID
            a0 = min(int(ns["start_char"]), int(g["start_char"]))
            a1 = max(int(ns["end_char"]),   int(g["end_char"]))
            out.append({
                "start_char": a0,
                "end_char":   a1,
                "type":       g.get("type",""),
                "uid":        g.get("uid") or _uid(),
            })
            merged = True
        else:
            out.append(g)
    if not merged:
        out.append(ns)

    out = dedup_exact(out)
    out = ensure_uids(out)

    # If sentence text known, enforce text fields for all spans
    if text is not None:
        out, _ = enforce_text_consistency(out, text, repair=True)
    return out

def render_html(html: str):
    # Use st.html (no sandbox) if available; otherwise fallback to components.html (sandboxed)
    if hasattr(st, "html"):  # Streamlit >= 1.36
        st.html(html)
    else:
        st.warning("Ctrl+Enter hotkey requires Streamlit ≥ 1.36 (using st.html). "
                   "Current build uses sandboxed iframe; hotkey will not work.")
        # st.components.v1.html(html, height=height, scrolling=scrolling)


# -------------------- UI --------------------

if "run_id" not in st.session_state:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    st.session_state.run_id = f"{ts}-{uuid.uuid4().hex[:6]}"


MAX_SNAPSHOTS = 50

st.set_page_config(page_title="BioNER Annotation", layout="wide")
st.title("BioNER Annotation")

mode = st.sidebar.radio("Annotation mode", ["NER", "Relations"], index=0)

if mode == "Relations":
    render_relation_page()
    st.stop()

# --- NER: Minimal input (single JSONL uploader) ---
st.markdown("### Input")

pred_up = st.file_uploader("Upload JSONL for annotation (required)", type=["jsonl"])
gold_up = None  # Not used in this simplified flow
pred_path = ""
gold_path = ""
    # Default matching/view parameters (no UI controls)
thr = 0.5
metric = "dice"
require_type = False
allow_many_to_one = True
suppress_nested_fp = True
show_tp, show_fp, show_fn = True, True, True
ws = 0.06
light_bg = True
merge_same_label = True
merge_thr = 0.8

if pred_up is not None:
    if st.session_state.get("upload_name") != pred_up.name:
        st.session_state.upload_name = pred_up.name
        upload_dir = os.path.join(SAVE_DIR, _safe_dirname(pred_up.name))
        st.session_state.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)


# --- Loader: gold optional, model required ---
def _load_sets():
    model = None
    if pred_up is not None:
        model = load_jsonl_grouped(pred_up)
    gold = {}
    return gold, model

gold, model = _load_sets()
if not model:
    st.info("Upload a JSONL file to start annotating.")
    st.stop()

# Keys come from union (or just model if no gold)
keys = sorted(set((gold or {}).keys()) | set(model.keys()))

# Detect predictions-only (no gold spans anywhere)
gold_total_spans = 0
for k in gold.keys():
    gold_total_spans += len(get_gold_spans(gold.get(k, [])))
predictions_only = (gold_total_spans == 0)

# Init working gold per sentence
if "aug_gold" not in st.session_state:
    working = {}
    for k in keys:
        base = [] if predictions_only else get_gold_spans(gold.get(k, []))
        working[k] = ensure_uids(base)  # << add this
    st.session_state.aug_gold = working


def _label_options(cur_key, proposals):
    """Return sorted list of candidate labels for the hotkey UI."""
    opts = set()
    # Current accepted gold for this sentence
    for s in st.session_state.aug_gold.get(cur_key, []) or []:
        t = (s.get("type") or "").strip()
        if t and t != CUSTOM:
            opts.add(t)
    # Current proposals shown to the annotator (FPs)
    for s in (proposals or []):
        t = (s.get("type") or s.get("label") or s.get("entity") or s.get("category") or "").strip()
        if t and t != CUSTOM:
            opts.add(t)
    model_labels = sorted(opts)
    result_labels = []
    for lab in model_labels:
        if lab in SCHEMA_TYPES_SHORT_TO_TEST:
            result_labels.append(lab)
    for lab in SCHEMA_TYPES_SHORT_TO_TEST:
        if lab not in model_labels:
            result_labels.append(lab)

    seen = set()
    cleaned = []
    for lab in result_labels:
        if not lab:
            continue
        if lab.lower() in {"custom", "(custom)"}:
            continue
        if lab not in seen:
            seen.add(lab)
            cleaned.append(lab)

    cleaned.append(CUSTOM)  # ensure present and last
    return cleaned


# Sentence navigation
if "sentence_idx" not in st.session_state:
    st.session_state.sentence_idx = 0
st.session_state.sentence_idx = max(0, min(st.session_state.sentence_idx, len(keys) - 1))

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 4, 2, 1])
with nav_col1:
    if st.button("◀ Prev", disabled=st.session_state.sentence_idx == 0):
        st.session_state.sentence_idx -= 1
        st.rerun()
with nav_col2:
    st.markdown(f"<div style='text-align:center;padding-top:6px'>Sentence {st.session_state.sentence_idx + 1} / {len(keys)}</div>", unsafe_allow_html=True)
with nav_col3:
    jump = st.number_input("Jump to", min_value=1, max_value=len(keys), value=st.session_state.sentence_idx + 1, step=1, label_visibility="collapsed")
    if jump - 1 != st.session_state.sentence_idx:
        st.session_state.sentence_idx = jump - 1
        st.rerun()
with nav_col4:
    if st.button("Next ▶", disabled=st.session_state.sentence_idx == len(keys) - 1):
        st.session_state.sentence_idx += 1
        st.rerun()

idx = st.session_state.sentence_idx
cur_key = keys[idx]

text = extract_text(gold.get(cur_key, []) or model.get(cur_key, []))
gold_spans = st.session_state.aug_gold[cur_key]
pred_spans = get_pred_spans(model.get(cur_key, []), "accepted_and_missing")

# Compute what to render + proposals list
if predictions_only:
    # Even without external gold, treat aug_gold as ground truth-in-progress
    tp, fp, fn = categorize_spans(gold_spans, pred_spans, thr, metric=metric,
                                  require_type=require_type,
                                  allow_many_to_one=allow_many_to_one,
                                  suppress_nested_fp=suppress_nested_fp)
    tp_view, fp_view, fn_view = tp, fp, fn
    proposals = fp
    label_options = _label_options(cur_key, proposals)
else:
    tp, fp, fn = categorize_spans(gold_spans, pred_spans, thr, metric=metric,
                                  require_type=require_type,
                                  allow_many_to_one=allow_many_to_one,
                                  suppress_nested_fp=suppress_nested_fp)
    tp_view, fp_view, fn_view = tp, fp, fn
    proposals = fp
    label_options = _label_options(cur_key, proposals)

proposals = dedup_proposals(proposals, text)


# --- Global hotkey defaults for this page (rendered once) ---
if "hot_lbl" not in st.session_state:
    st.session_state.hot_lbl = next((x for x in label_options if x != CUSTOM), "MISC")



# Handle any hot-add request from the selection hotkey
_apply_hotadd_from_query(cur_key, text, proposals, label_options)

# Header + viewer
if predictions_only:
    st.subheader(f"{cur_key[0]}:{cur_key[1]}  —  Candidates:{len(pred_spans)}  Gold accepted:{len(gold_spans)}")


if not predictions_only:
    st.subheader(f"{cur_key[0]}:{cur_key[1]}  —  TP:{len(tp_view)}  FP:{len(fp_view)}  FN:{len(fn_view)}")



html_block = build_single_layer_html(
    text, tp_view, fp_view, fn_view,
    show_tp=show_tp, show_fp=show_fp, show_fn=show_fn,
    word_spacing_em=ws, light_bg=light_bg,
    doc_id=cur_key[0], sent_idx=cur_key[1]
)


colS, colA = st.columns([3, 2], gap="large")

with colS:
    # components.html(html_block)
    render_html(html_block)


st.divider()

colG, colP = st.columns(2)


# --- Helper: find all occurrences of a substring with context previews ---
def _find_occurrences(text: str, query: str, window: int = 40):
    out = []
    if not query:
        return out
    start = 0
    while True:
        i = text.find(query, start)
        if i == -1:
            break
        j = i + len(query)
        left = max(0, i - window)
        right = min(len(text), j + window)
        preview = f"{text[left:i]}[ {text[i:j]} ]{text[j:right]}"
        out.append({"start": i, "end": j, "preview": preview})
        start = i + 1
    return out

with colA:
    st.markdown("### ➕ Add manual span (not proposed by the model)")
    keyroot = f"{cur_key[0]}_{cur_key[1]}"

    # --- versioned keys to force remount/clear ---
    ver_key = f"{keyroot}_ver"
    if ver_key not in st.session_state:
        st.session_state[ver_key] = 0
    ver = st.session_state[ver_key]

    # --- reset gate (must run BEFORE widgets are instantiated) ---
    if st.session_state.get(f"{keyroot}_clear", False):
        st.session_state.pop(f"{keyroot}_add_q", None)
        st.session_state.pop(f"{keyroot}_add_pick", None)
        st.session_state.pop(f"{keyroot}_add_lab", None)
        st.session_state[ver_key] = ver + 1
        st.session_state[f"{keyroot}_clear"] = False

    with st.container(border=True):
        # Label chooser (reuse discovered labels, allow custom)
        lab = st.selectbox("Label", options=label_options, key=f"{keyroot}_add_lab")
        if lab == CUSTOM:
            lab = st.text_input("Custom label", value="", key=f"{keyroot}_add_lab_custom")

        is_virtual = st.checkbox(
            "Virtual span (text not literally in sentence)",
            key=f"{keyroot}_add_virtual_{ver}",
            help="Use when the entity text must be reconstructed, e.g. 'antibacterial and antifungal drugs' → add 'antibacterial drugs' as a virtual span",
        )

        if is_virtual:
            vtext = st.text_input(
                "Span text (virtual)",
                key=f"{keyroot}_add_vtext_{ver}",
                help="Type the entity text as it should be recorded; it will be stored with virtual=true and start_char/end_char=-1",
            )
            can_add = bool(lab and vtext)
            btn = st.button("Add virtual span", type="primary", disabled=not can_add, key=f"{keyroot}_add_vbtn")
            if btn and can_add:
                new_span = {"start_char": -1, "end_char": -1, "type": lab, "text": vtext, "virtual": True, "uid": _uid()}
                current = st.session_state.aug_gold[cur_key]
                st.session_state.aug_gold[cur_key] = dedup_exact(ensure_uids(current + [new_span]))
                _maybe_autosave()
                st.session_state[f"{keyroot}_clear"] = True
                force_rerun()
        else:
            q_key = f"{keyroot}_add_q_{st.session_state[ver_key]}"
            q = st.text_input("Exact text to add (case-sensitive substring)", key=q_key)
            matches = _find_occurrences(text, q) if q else []
            if matches:
                pick = st.selectbox(
                    "Choose occurrence (shows local context)",
                    options=list(range(len(matches))),
                    format_func=lambda k: matches[k]["preview"],
                    key=f"{keyroot}_add_pick"
                )
                chosen = matches[pick]
                a, b = chosen["start"], chosen["end"]
                st.caption(f"Will add: `[{a},{b})` → `{text[a:b]}`")
            else:
                chosen = None
                if q:
                    st.warning("No matches found in this sentence.")

            can_add = (lab is not None and lab != "" and chosen is not None)
            btn = st.button("Add manual span", type="primary", disabled=not can_add, key=f"{keyroot}_add_btn")
            if btn and can_add:
                new_span = {"start_char": int(chosen["start"]), "end_char": int(chosen["end"]), "type": lab}
                st.session_state.aug_gold[cur_key] = merge_span_into_gold(
                    st.session_state.aug_gold[cur_key],
                    new_span,
                    metric=metric,
                    thr=merge_thr,
                    do_merge=merge_same_label,
                    text=text
                )
                st.session_state.aug_gold[cur_key] = ensure_uids(st.session_state.aug_gold[cur_key])
                st.session_state.aug_gold[cur_key] = dedup_exact(st.session_state.aug_gold[cur_key])
                _maybe_autosave()
                st.session_state[f"{keyroot}_clear"] = True
                force_rerun()


with colP:
    st.markdown("### Proposed additions")
    if not proposals:
        st.info("No proposals on this sentence.")
    else:
        for s in proposals:
            a0 = int(s.get("start_char", 0))
            b0 = int(s.get("end_char", 0))
            default_label = (s.get("type") or s.get("label") or s.get("entity") or s.get("category") or "")
            sig = prop_sig(s, text)  # <<< stable identity for widget keys

            with st.container(border=True):
                st.write(f"`{text[a0:b0]}`  [{a0},{b0})")
                c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

                # Label selector (keyed by sig, not index)
                with c1:
                    # compute a stable default index
                    all_opts = label_options
                    idx_default = all_opts.index(default_label) if default_label in label_options else 0

                    new_lbl = st.selectbox(
                        "Label",
                        options=all_opts,
                        index=idx_default,
                        key=wkey("fp_lab", sig),
                    )
                    if new_lbl == CUSTOM:
                        new_lbl = st.text_input("Custom", value=default_label, key=wkey("fp_custom", sig))

                # Start/end inputs (keyed by sig)
                with c2:
                    new_a = st.number_input("Start", value=a0, step=1, key=wkey("fp_a", sig))
                with c3:
                    new_b = st.number_input("End", value=b0, step=1, key=wkey("fp_b", sig))

                # Accept button (keyed by sig)
                with c4:
                    st.write("")
                    if st.button("Accept", key=wkey("fp_accept", sig)):
                        # write gold by merging; proposals list will recompute on rerun
                        new = {"start_char": int(new_a), "end_char": int(new_b), "type": new_lbl}
                        st.session_state.aug_gold[cur_key] = merge_span_into_gold(
                            st.session_state.aug_gold[cur_key],
                            new,
                            metric=metric,
                            thr=merge_thr,
                            do_merge=merge_same_label,
                            text=text
                        )
                        st.session_state.aug_gold[cur_key] = ensure_uids(st.session_state.aug_gold[cur_key])
                        _maybe_autosave()
                        force_rerun()


with colG:
    st.markdown("### Current gold (editable)")
    gold_spans = ensure_uids(st.session_state.aug_gold[cur_key])  # make sure they have uids
    if not gold_spans:
        st.info("No gold spans yet. Approve candidates on the left to add some.")
    to_delete = []
    for j, g in enumerate(gold_spans):
        a, b = int(g["start_char"]), int(g["end_char"])
        lbl   = g.get("type","")
        uid   = g.get("uid")  # stable identity
        is_virtual = g.get("virtual", False)

        with st.container(border=True):
            if is_virtual:
                st.write(f"`{g.get('text','')}` *(virtual)*")
            else:
                st.write(f"`{text[a:b]}`  [{a},{b})")
            c1, c2, c3, c4 = st.columns([2,2,2,2])
            with c1:
                opts = label_options  # CUSTOM guaranteed at end
                idx = opts.index(lbl) if lbl in opts else 0

                new_lbl = st.selectbox(
                    f"Label #{j}",
                    options=opts,
                    index=idx,
                    key=f"g_lab_{uid}",
                )
                if new_lbl == CUSTOM:
                    new_lbl = st.text_input(f"Custom g#{j}", value=lbl, key=f"g_custom_{uid}")
            with c2:
                if is_virtual:
                    new_vtext = st.text_input(f"Text g#{j}", value=g.get("text",""), key=f"g_vtext_{uid}")
                else:
                    new_a = st.number_input(f"Start g#{j}", value=a, step=1, key=f"g_a_{uid}")
            with c3:
                if not is_virtual:
                    new_b = st.number_input(f"End g#{j}", value=b, step=1, key=f"g_b_{uid}")
                else:
                    st.caption("virtual")
            with c4:
                st.write("")
                b_update = st.button("Update", key=f"g_upd_{uid}")
                b_delete = st.button("Delete", key=f"g_del_{uid}")

            if b_update:
                if is_virtual:
                    updated = {"start_char": -1, "end_char": -1, "type": new_lbl, "text": new_vtext, "virtual": True, "uid": uid}
                else:
                    updated = {"start_char": int(new_a), "end_char": int(new_b), "type": new_lbl, "uid": uid}
                    updated = set_span_text(updated, text)
                st.session_state.aug_gold[cur_key][j] = updated
                st.session_state.aug_gold[cur_key] = dedup_exact(st.session_state.aug_gold[cur_key])
                st.session_state.aug_gold[cur_key] = ensure_uids(st.session_state.aug_gold[cur_key])
                _maybe_autosave()
                force_rerun()

            if b_delete:
                to_delete.append(j)

    if to_delete:
        st.session_state.aug_gold[cur_key] = [s for k,s in enumerate(st.session_state.aug_gold[cur_key]) if k not in to_delete]
        st.session_state.aug_gold[cur_key] = ensure_uids(st.session_state.aug_gold[cur_key])
        _maybe_autosave()
        force_rerun()


# -------------------- Export --------------------
st.divider()
st.subheader("Export augmented gold")

run_path, latest_path = _current_paths()
if run_path and latest_path:
    st.caption(f"Autosave: session file → {run_path}")
    st.caption(f"Latest snapshot → {latest_path}")
else:
    st.caption("Autosave: waiting for an uploaded JSONL file.")

aug_jsonl = export_augmented_jsonl()

st.download_button(
    f"Download augmented_gold ({st.session_state.run_id}).jsonl",
    data=aug_jsonl.encode("utf-8"),
    file_name=f"augmented_gold_{st.session_state.run_id}.jsonl",
    mime="application/jsonl",
)