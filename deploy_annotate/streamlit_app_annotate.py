import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import io
import streamlit.components.v1 as components

import streamlit as st

# ---------- Persist options defaults ----------
SAVE_PATH = "/s3/mobiko/mobiko-data/augmented_gold.jsonl"

# Autosave is always on; change SAVE_PATH above if needed.
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
        label = st.session_state.get("hot_lbl") or (default_label_options[0] if default_label_options else "MISC")

    new_span = {"start_char": a, "end_char": b, "type": label}
    st.session_state.aug_gold[cur_key] = merge_span_into_gold(
        st.session_state.aug_gold[cur_key], new_span, metric="dice", thr=0.8, do_merge=True
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
            sentences.append({"text": text_here, "spans": _spans})
        rec = {"doc_id": doc_id, "sentences": sentences}
        buf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return buf.getvalue()

def _maybe_autosave():
    if st.session_state.get("autosave", True):
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            f.write(export_augmented_jsonl())

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
            if require_type and pt != g.get("type"): continue
            s = score_pair(p, g, metric)
            if s >= thr: pairs.append((s, pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    matched_g, matched_p = {}, {}
    for s, pi, gi in pairs:
        if gi in matched_g or pi in matched_p: continue
        matched_g[gi] = pi; matched_p[pi] = gi
    # containment credit
    if allow_many_to_one:
        for gi, g in enumerate(gold_spans):
            if gi in matched_g: continue
            for pi, p in enumerate(pred_spans):
                if pi not in matched_p: continue
                if require_type and p.get("type") != g.get("type"): continue
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

    # Use % formatting to avoid f-string brace escaping
    js = """
<script>
(function(){
  function ready(fn){ if (document.readyState !== 'loading') fn(); else document.addEventListener('DOMContentLoaded', fn); }
  ready(function(){
    try{
      console.log("[hotkey] init start");
      const wrap = document.getElementById('sent-wrap');
      const card = document.querySelector('.single-layer');
      const btn  = document.getElementById('acceptBtn');
      const DOC = %(doc)s;
      const IDX = %(idx)s;

      if (!wrap || !card) { console.warn("[hotkey] wrap/card missing"); return; }

      // Make sure the container can receive key events:
      try { wrap.focus({preventScroll:true}); } catch(_) {}
      console.log("[hotkey] focused wrapper");

      function selectionOffsetsWithin(el){
        const sel = window.getSelection();
        if (!sel || sel.rangeCount === 0) return null;
        const rng = sel.getRangeAt(0);
        if (!el.contains(rng.startContainer) || !el.contains(rng.endContainer)) return null;

        const preA = document.createRange();
        preA.selectNodeContents(el);
        preA.setEnd(rng.startContainer, rng.startOffset);
        const a = (preA.toString() || "").length;

        const preB = document.createRange();
        preB.selectNodeContents(el);
        preB.setEnd(rng.endContainer, rng.endOffset);
        const b = (preB.toString() || "").length;

        return (b > a) ? {a:a, b:b} : null;
      }

      function hotAdd(){
        const offs = selectionOffsetsWithin(card);
        if (!offs) { console.warn("[hotkey] no selection or outside card"); return; }
        console.log("[hotkey] hotAdd", offs);

        // Update URL and reload (requires st.html / non-sandbox)
        try {
          const qp = new URLSearchParams(window.location.search || "");
          qp.set('hotadd', String(DOC) + '|' + String(IDX) + '|' + String(offs.a) + '|' + String(offs.b));
          const url = window.location.pathname + '?' + qp.toString();
          console.log("[hotkey] navigate", url);
          window.location.href = url;
        } catch (e) {
          console.error("[hotkey] navigation failed", e);
        }
      }

      // Multiple listeners to dodge framework event traps
      window.addEventListener('keydown', function(e){
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); hotAdd(); }
      }, true);
      document.addEventListener('keydown', function(e){
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); hotAdd(); }
      }, true);
      if (btn) btn.addEventListener('click', hotAdd);

      console.log("[hotkey] listeners attached");
    } catch (err) {
      console.error("[hotkey] init error", err);
    }
  });
})();
</script>
""" % {"doc": _json.dumps(doc_id), "idx": _json.dumps(sent_idx)}

    return html_core + js



# -------------------- Annotation helpers --------------------
def dedup_exact(spans: list[dict]) -> list[dict]:
    seen = set()
    out  = []
    for s in spans:
        key = (int(s["start_char"]), int(s["end_char"]), s.get("type",""))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)  # keep original dict with its uid
    return out


def merge_span_into_gold(gold_spans: list[dict], new_span: dict,
                         metric="dice", thr=0.8, do_merge=True) -> list[dict]:
    """Merge new_span; preserve UIDs on existing spans; if we merge INTO an existing span, keep that span's UID."""
    ns = {
        "start_char": int(new_span["start_char"]),
        "end_char":   int(new_span["end_char"]),
        "type":       new_span.get("type",""),
        "uid":        new_span.get("uid") or _uid(),  # new uid if truly new
    }
    if not do_merge:
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
    return ensure_uids(out)


def _label_options(cur_key, proposals):
    """Return sorted list of candidate labels for the hotkey UI."""
    opts = set()
    # Current accepted gold for this sentence
    for s in st.session_state.aug_gold.get(cur_key, []) or []:
        t = (s.get("type") or "").strip()
        if t:
            opts.add(t)
    # Current proposals shown to the annotator (FPs)
    for s in (proposals or []):
        t = (s.get("type") or s.get("label") or s.get("entity") or s.get("category") or "").strip()
        if t:
            opts.add(t)
    return sorted(opts) if opts else ["MISC"]


def render_html(html: str):
    # Use st.html (no sandbox) if available; otherwise fallback to components.html (sandboxed)
    if hasattr(st, "html"):  # Streamlit >= 1.36
        st.html(html)
    else:
        st.warning("Ctrl+Enter hotkey requires Streamlit ≥ 1.36 (using st.html). "
                   "Current build uses sandboxed iframe; hotkey will not work.")
        # st.components.v1.html(html, height=height, scrolling=scrolling)


# -------------------- UI --------------------


st.set_page_config(page_title="NER Annotation — Minimal", layout="wide")


st.title("NER Annotation — Minimal")

# --- Minimal input (single JSONL uploader) ---
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

# Label options from existing gold + model
def collect_labels():
    labs=set()
    for k in keys:
        for s in st.session_state.aug_gold.get(k, []):
            if s.get("type"): labs.add(s["type"])
        for inst in model.get(k, []):
            llm = inst.get("llm", {}) or {}
            for seq in (llm.get("accepted", []) or []) + (llm.get("missing", []) or []):
                t=(seq.get("type") or seq.get("label") or seq.get("ent_type") or seq.get("entity") or seq.get("category"))
                if t: labs.add(t)
    return sorted(labs)

label_options = collect_labels() or ["ORG","PER","LOC","MISC"]

# Sentence navigation
idx = st.slider("Sentence index", 0, len(keys)-1, 0, 1)
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

# --- Global hotkey defaults for this page (rendered once) ---
if "hot_lbl" not in st.session_state:
    st.session_state.hot_lbl = (label_options[0] if label_options else "MISC")



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

render_html(html_block)

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

st.divider()
st.markdown("### ➕ Add manual span (not proposed by the model)")

with st.container(border=True):
    mode = st.radio("Add by…", ["Text match", "Offsets"], horizontal=True, key=f"add_mode_{idx}")

    # Label chooser (reuse discovered labels, allow custom)
    lab = st.selectbox("Label", options=label_options + ["(custom)"], key=f"add_lab_{idx}")
    if lab == "(custom)":
        lab = st.text_input("Custom label", value="", key=f"add_lab_custom_{idx}")

    if mode == "Text match":
        q = st.text_input("Exact text to add (case-sensitive substring)", key=f"add_q_{idx}")
        matches = _find_occurrences(text, q) if q else []
        if matches:
            pick = st.selectbox(
                "Choose occurrence (shows local context)",
                options=list(range(len(matches))),
                format_func=lambda k: matches[k]["preview"],
                key=f"add_pick_{idx}"
            )
            chosen = matches[pick]
            a, b = chosen["start"], chosen["end"]
            st.caption(f"Will add: `[{a},{b})` → `{text[a:b]}`")
        else:
            chosen = None
            if q:
                st.warning("No matches found in this sentence.")

    else:  # Offsets
        a = st.number_input("Start char", min_value=0, max_value=len(text), value=0, step=1, key=f"add_a_{idx}")
        b = st.number_input("End char", min_value=0, max_value=len(text), value=min(len(text), 1), step=1, key=f"add_b_{idx}")
        chosen = {"start": int(a), "end": int(b)} if int(a) < int(b) else None
        if chosen:
            aa, bb = chosen["start"], chosen["end"]
            st.caption(f"Preview: `[{aa},{bb})` → `{text[aa:bb]}`")

    c1, c2 = st.columns([1,1])
    with c1:
        snap = st.checkbox("Auto-merge if overlaps same label", value=merge_same_label, key=f"add_merge_{idx}")
    with c2:
        go_next = st.checkbox("After adding, jump to next sentence", value=False, key=f"add_next_{idx}")

    can_add = (lab is not None and lab != "" and chosen is not None)
    btn = st.button("Add manual span", type="primary", disabled=not can_add, key=f"add_btn_{idx}")
    if btn and can_add:
        new_span = {"start_char": int(chosen["start"]), "end_char": int(chosen["end"]), "type": lab}
        st.session_state.aug_gold[cur_key] = merge_span_into_gold(
            st.session_state.aug_gold[cur_key],
            new_span,
            metric=metric,
            thr=merge_thr,
            do_merge=snap
        )
        st.session_state.aug_gold[cur_key] = ensure_uids(st.session_state.aug_gold[cur_key])
        st.session_state.aug_gold[cur_key] = dedup_exact(st.session_state.aug_gold[cur_key])
        _maybe_autosave()
        if go_next and idx < len(keys) - 1:
            # move the slider forward programmatically
            st.experimental_set_query_params(sent=str(idx+1))
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
                    all_opts = label_options + ["(custom)"]
                    idx_default = all_opts.index(default_label) if default_label in all_opts else len(label_options)
                    new_lbl = st.selectbox(
                        "Label",
                        options=all_opts,
                        index=idx_default,
                        key=wkey("fp_lab", sig),
                    )
                    if new_lbl == "(custom)":
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

        with st.container(border=True):
            st.write(f"`{text[a:b]}`  [{a},{b})")
            c1, c2, c3, c4 = st.columns([2,2,2,2])
            with c1:
                new_lbl = st.selectbox(f"Label #{j}",
                    options=label_options + ["(custom)"],
                    index=(label_options+["(custom)"]).index(lbl) if lbl in label_options else len(label_options),
                    key=f"g_lab_{uid}")  # << uid here
                if new_lbl == "(custom)":
                    new_lbl = st.text_input(f"Custom g#{j}", value=lbl, key=f"g_custom_{uid}")  # << uid
            with c2:
                new_a = st.number_input(f"Start g#{j}", value=a, step=1, key=f"g_a_{uid}")  # << uid
            with c3:
                new_b = st.number_input(f"End g#{j}", value=b, step=1, key=f"g_b_{uid}")    # << uid
            with c4:
                st.write("")
                b_update = st.button("Update", key=f"g_upd_{uid}")   # << uid
                b_delete = st.button("Delete", key=f"g_del_{uid}")   # << uid

            if b_update:
                st.session_state.aug_gold[cur_key][j] = {
                    "start_char": int(new_a), "end_char": int(new_b),
                    "type": new_lbl, "uid": uid  # preserve uid
                }
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

aug_jsonl = export_augmented_jsonl()

# Autosave writes to SAVE_PATH; no manual server-path UI
st.caption(f"Autosaving to {SAVE_PATH} on every change.")

st.download_button("Download augmented_gold.jsonl", data=aug_jsonl.encode("utf-8"),
                   file_name="augmented_gold.jsonl", mime="application/jsonl")

st.caption("Approve **candidates** (left) to build gold, edit/delete in **Current gold** (right), then **Download** or **Save**. In predictions-only mode, all model spans are candidates; in gold+model mode, candidates are FPs against the current gold.")
