import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import io

import streamlit as st

# ---------- Persist options defaults ----------
if "save_path" not in st.session_state:
    st.session_state.save_path = ""
if "autosave" not in st.session_state:
    st.session_state.autosave = False

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
    sp = st.session_state.get("save_path", "")
    if st.session_state.get("autosave", False) and sp:
        with open(sp, "w", encoding="utf-8") as f:
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

def build_single_layer_html(text, tp, fp, fn, show_tp, show_fp, show_fn, word_spacing_em=0.06, light_bg=True):
    import html as html_mod
    fragments=merge_to_fragments(text,tp,fp,fn)
    enabled={"tp":show_tp,"fp":show_fp,"fn":show_fn}
    mix = 0.35 if light_bg else 0.25
    def style_for_cats(cats):
        on=[c for c in cats if enabled.get(c, False)]
        if len(on)==0: return ("transparent", "none")
        if len(on)==1:
            c=on[0]
            return (f"color-mix(in srgb, var(--{c}) {int(mix*100)}%, transparent)",
                    f"inset 0 0 0 1px color-mix(in srgb, var(--{c}) 60%, {'white' if light_bg else 'black'})")
        step=6
        seq=", ".join([f"var(--{c}) {i*step}px {(i+1)*step}px" for i,c in enumerate(on)])
        return (f"repeating-linear-gradient(135deg, {seq})", "inset 0 0 0 1px #0003")
    parts=[]
    for seg,cats,p_labels,g_labels in fragments:
        esc=html_mod.escape(seg)
        if not cats: parts.append(esc); continue
        bg, box = style_for_cats(cats)
        pred_lbl = ", ".join(sorted(p_labels)) if p_labels else "—"
        gold_lbl = ", ".join(sorted(g_labels)) if g_labels else "—"
        title = f"Pred: {pred_lbl}\\nGold: {gold_lbl}"
        parts.append(f'<span class="mark" style="background:{bg}; box-shadow:{box};" title="{html_mod.escape(title)}">{esc}</span>')
    if light_bg:
        css = f"""
        <style>
        :root {{ --tp:#2e7d32; --fp:#c62828; --fn:#6a1b9a; }}
        .sent-card {{ background: #f7f9ff; border: 1px solid #dbe1ff; border-radius: 12px; padding: 12px; }}
        .single-layer {{ white-space: pre-wrap; word-break: break-word; font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Arial; color: #0c1222; word-spacing: {word_spacing_em}em; }}
        .mark {{ border-radius: 4px; padding: 0; }}
        .mark:hover {{ outline: 2px solid #0002; cursor: help; }}
        </style>
        """
    else:
        css = f"""
        <style>
        :root {{ --tp:#2e7d32; --fp:#c62828; --fn:#6a1b9a; }}
        .sent-card {{ background: #0e1730; border: 1px dashed #2a355e; border-radius: 10px; padding: 10px; }}
        .single-layer {{ white-space: pre-wrap; word-break: break-word; font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Arial; color: #e6ecff; word-spacing: {word_spacing_em}em; }}
        .mark {{ border-radius: 4px; padding: 0; }}
        .mark:hover {{ outline: 2px solid #fff3; cursor: help; }}
        </style>
        """
    return f"{css}<div class='sent-card'><div class='single-layer'>{''.join(parts)}</div></div>"

# -------------------- Annotation helpers --------------------
def dedup_exact(spans: List[Dict]) -> List[Dict]:
    seen=set(); out=[]
    for s in spans:
        key=(int(s["start_char"]), int(s["end_char"]), s.get("type",""))
        if key in seen: continue
        seen.add(key); out.append(s)
    return out

def merge_span_into_gold(gold_spans: List[Dict], new_span: Dict, metric="dice", thr=0.8, do_merge=True) -> List[Dict]:
    ns = {"start_char": int(new_span["start_char"]), "end_char": int(new_span["end_char"]), "type": new_span.get("type","")}
    if not do_merge: return dedup_exact(gold_spans + [ns])
    merged = False
    out = []
    for g in gold_spans:
        if g.get("type","") == ns.get("type",""):
            sc = score_pair(ns, g, metric)
            if sc >= thr:
                a0,a1 = int(min(ns["start_char"], g["start_char"])), int(max(ns["end_char"], g["end_char"]))
                out.append({"start_char": a0, "end_char": a1, "type": ns.get("type","")})
                merged = True
            else:
                out.append(g)
        else:
            out.append(g)
    if not merged:
        out.append(ns)
    return dedup_exact(out)

# -------------------- UI --------------------
st.set_page_config(page_title="NER Annotation (Promote Predictions)", layout="wide")
st.title("NER Annotation — Approve Model Spans into Gold")

with st.sidebar:
    st.header("Persist")
    st.session_state.save_path = st.text_input(
        "Save augmented gold to (server path)",
        value=st.session_state.save_path
    )
    st.session_state.autosave = st.checkbox(
        "Autosave on change",
        value=st.session_state.autosave
    )

    st.header("Data")
    tab1, tab2 = st.tabs(["Upload files", "Use remote paths"])
    with tab1:
        gold_up = st.file_uploader("Gold JSONL (optional)", type=["jsonl"])
        pred_up = st.file_uploader("Model JSONL (required)", type=["jsonl"])
    with tab2:
        gold_path = st.text_input("Gold JSONL path (optional)")
        pred_path = st.text_input("Model JSONL path (required)")

    st.header("Matching / View")
    thr = st.slider("Match score threshold", 0.0, 1.0, 0.5, 0.05)
    metric = st.selectbox("Match metric", ["dice","iou","min"], index=0)
    require_type = st.checkbox("Require type match", value=False)
    allow_many_to_one = st.checkbox("Containment credit (recall-friendly)", value=True)
    suppress_nested_fp = st.checkbox("Suppress nested FP shards", value=True)

    st.header("Display")
    show_tp = st.checkbox("Highlight TP (or candidates in pred-only)", value=True)
    show_fp = st.checkbox("Highlight FP", value=True)
    show_fn = st.checkbox("Highlight FN", value=True)
    ws = st.slider("Word spacing (em)", 0.0, 0.3, 0.06, 0.01)
    light_bg = st.checkbox("Use light sentence background", value=True)

    st.header("Accept settings")
    merge_same_label = st.checkbox("Auto-merge with overlapping same-label gold", value=True)
    merge_thr = st.slider("Merge threshold (same label)", 0.0, 1.0, 0.8, 0.05)

# --- Loader: gold optional, model required ---
def _load_sets():
    # model
    model = None
    if pred_up is not None:
        model = load_jsonl_grouped(pred_up)
    elif pred_path:
        model = load_jsonl_grouped(pred_path)

    # gold (optional)
    gold = {}
    if gold_up is not None:
        gold = load_jsonl_grouped(gold_up)
    elif gold_path:
        gold = load_jsonl_grouped(gold_path)

    return gold, model

gold, model = _load_sets()
if not model:
    st.info("Load a Model JSONL (gold is optional).")
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
        working[k] = [] if predictions_only else get_gold_spans(gold.get(k, []))
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
    tp_view, fp_view, fn_view = pred_spans, [], []
    # Propose only spans not already accepted into gold (exact match)
    accepted = {(int(s["start_char"]), int(s["end_char"]), s.get("type","")) for s in gold_spans}
    proposals = [s for s in pred_spans
                 if (int(s["start_char"]), int(s["end_char"]), s.get("type","")) not in accepted]
else:
    tp, fp, fn = categorize_spans(gold_spans, pred_spans, thr, metric=metric,
                                  require_type=require_type,
                                  allow_many_to_one=allow_many_to_one,
                                  suppress_nested_fp=suppress_nested_fp)
    tp_view, fp_view, fn_view = tp, fp, fn
    proposals = fp

# Header + viewer
if predictions_only:
    st.subheader(f"{cur_key[0]}:{cur_key[1]}  —  Candidates:{len(pred_spans)}  Gold accepted:{len(gold_spans)}")
else:
    st.subheader(f"{cur_key[0]}:{cur_key[1]}  —  TP:{len(tp_view)}  FP:{len(fp_view)}  FN:{len(fn_view)}")

html_block = build_single_layer_html(
    text, tp_view, fp_view, fn_view,
    show_tp=show_tp, show_fp=show_fp, show_fn=show_fn,
    word_spacing_em=ws, light_bg=light_bg
)
st.components.v1.html(html_block, height=260, scrolling=True)

colG, colP = st.columns(2)

with colP:
    st.markdown("### Proposed additions (" + ("model spans" if predictions_only else "model FPs") + ")")
    if not proposals:
        st.info("No proposals on this sentence.")
    for i, s in enumerate(proposals):
        a,b = int(s["start_char"]), int(s["end_char"])
        default_label = s.get("type","")
        # If your Streamlit is older and `border=True` errors, change to: with st.container():
        with st.container(border=True):
            st.write(f"`{text[a:b]}`  [{a},{b})")
            c1,c2,c3,c4 = st.columns([2,2,2,2])
            with c1:
                la = st.selectbox(f"Label #{i}", options=label_options + ["(custom)"],
                                  index=(label_options+["(custom)"]).index(default_label) if default_label in label_options else len(label_options),
                                  key=f"fp_lab_{idx}_{i}")
                if la == "(custom)":
                    la = st.text_input(f"Custom label #{i}", value=default_label, key=f"fp_custom_{idx}_{i}")
            with c2:
                na = st.number_input(f"Start #{i}", value=a, step=1, key=f"fp_a_{idx}_{i}")
            with c3:
                nb = st.number_input(f"End #{i}", value=b, step=1, key=f"fp_b_{idx}_{i}")
            with c4:
                if st.button("Accept", key=f"fp_accept_{idx}_{i}"):
                    new = {"start_char": int(na), "end_char": int(nb), "type": la}
                    st.session_state.aug_gold[cur_key] = merge_span_into_gold(
                        st.session_state.aug_gold[cur_key], new,
                        metric=metric, thr=merge_thr, do_merge=merge_same_label
                    )
                    _maybe_autosave()
                    force_rerun()

with colG:
    st.markdown("### Current gold (editable)")
    if not gold_spans:
        st.info("No gold spans yet. Approve candidates on the left to add some.")
    to_delete=[]
    for j, g in enumerate(gold_spans):
        a,b = int(g["start_char"]), int(g["end_char"])
        lbl = g.get("type","")
        with st.container(border=True):
            st.write(f"`{text[a:b]}`  [{a},{b})")
            c1,c2,c3,c4 = st.columns([2,2,2,2])
            with c1:
                new_lbl = st.selectbox(f"Label g#{j}", options=label_options + ["(custom)"],
                                       index=(label_options+["(custom)"]).index(lbl) if lbl in label_options else len(label_options),
                                       key=f"g_lab_{idx}_{j}")
                if new_lbl == "(custom)":
                    new_lbl = st.text_input(f"Custom g#{j}", value=lbl, key=f"g_custom_{idx}_{j}")
            with c2:
                new_a = st.number_input(f"Start g#{j}", value=a, step=1, key=f"g_a_{idx}_{j}")
            with c3:
                new_b = st.number_input(f"End g#{j}", value=b, step=1, key=f"g_b_{idx}_{j}")
            with c4:
                st.write("")
                b_update = st.button("Update", key=f"g_upd_{idx}_{j}")
                b_delete = st.button("Delete", key=f"g_del_{idx}_{j}")
            if b_update:
                st.session_state.aug_gold[cur_key][j] = {
                    "start_char": int(new_a), "end_char": int(new_b), "type": new_lbl
                }
                st.session_state.aug_gold[cur_key] = dedup_exact(st.session_state.aug_gold[cur_key])
                _maybe_autosave()
                force_rerun()
            if b_delete:
                to_delete.append(j)
    if to_delete:
        st.session_state.aug_gold[cur_key] = [s for k,s in enumerate(st.session_state.aug_gold[cur_key]) if k not in to_delete]
        _maybe_autosave()
        force_rerun()

# -------------------- Export --------------------
st.divider()
st.subheader("Export augmented gold")

aug_jsonl = export_augmented_jsonl()

if st.sidebar.button("Save now"):
    sp = st.session_state.get("save_path", "")
    if sp:
        with open(sp, "w", encoding="utf-8") as f:
            f.write(aug_jsonl)
        st.sidebar.success(f"Saved to {sp}")
    else:
        st.sidebar.warning("Provide a server path to save.")

st.download_button("Download augmented_gold.jsonl", data=aug_jsonl.encode("utf-8"),
                   file_name="augmented_gold.jsonl", mime="application/jsonl")

st.caption("Approve **candidates** (left) to build gold, edit/delete in **Current gold** (right), then **Download** or **Save**. In predictions-only mode, all model spans are candidates; in gold+model mode, candidates are FPs against the current gold.")
