import json, argparse, html
from collections import defaultdict

# ---------- Span math ----------
def span_len(a, b): 
    return max(0, b - a)

def overlap(a0, a1, b0, b1):
    s = max(a0, b0)
    e = min(a1, b1)
    return max(0, e - s)

def dice(a0, a1, b0, b1):
    inter = overlap(a0,a1,b0,b1)
    denom = span_len(a0,a1) + span_len(b0,b1)
    return (2*inter/denom) if denom > 0 else 0.0

def iou(a0, a1, b0, b1):
    inter = overlap(a0,a1,b0,b1)
    uni = span_len(a0,a1) + span_len(b0,b1) - inter
    return (inter/uni) if uni > 0 else 0.0

def overlap_min(a0,a1,b0,b1):
    inter = overlap(a0,a1,b0,b1)
    denom = min(span_len(a0,a1), span_len(b0,b1))
    return (inter/denom) if denom > 0 else 0.0

def score_pair(p, g, metric):
    a0,a1 = int(p["start_char"]), int(p["end_char"])
    b0,b1 = int(g["start_char"]), int(g["end_char"])
    if metric == "iou": 
        return iou(a0,a1,b0,b1)
    if metric == "min": 
        return overlap_min(a0,a1,b0,b1)
    return dice(a0,a1,b0,b1)  # default: dice

# ---------- I/O ----------
def load_spans(path):
    spans_by_key = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            r = json.loads(line)
            for i, sentence in enumerate(r["sentences"]):
                key = (r["doc_id"], i)
                spans_by_key[key].append(sentence)
    return spans_by_key

def extract_sentence_text(value_list):
    for inst in value_list:
        txt = inst.get("text") or ""
        if txt: 
            return txt
    return ""

def extract_gold_spans(value_list):
    out = []
    for inst in value_list:
        out.extend([normalize_span(x) for x in (inst.get("spans", []) or [])])
    return out

def normalize_span(s):
    # Normalize label key under 'type'
    if "type" not in s or s.get("type") in (None, "", "UNK"):
        for k in ("label", "ent_type", "entity", "category"):
            if k in s and s[k]:
                s["type"] = s[k]
                break
    return s

def extract_model_spans(value_list, source):
    out = []
    for inst in value_list:
        llm = inst.get("llm", {}) or {}
        if source == "accepted":
            out.extend([normalize_span(x) for x in (llm.get("accepted", []) or [])])
        else:  # accepted_and_missing
            out.extend([normalize_span(x) for x in (llm.get("accepted", []) or [])])
            out.extend([normalize_span(x) for x in (llm.get("missing", []) or [])])
    return out

# ---------- Matching ----------
def categorize_spans(gold_spans, pred_spans, thr, metric="dice", require_type=True, allow_many_to_one=True, suppress_nested_fp=True):
    # Build all candidate pairs with scores
    pairs = []
    for pi, p in enumerate(pred_spans):
        pt = p.get("type")
        for gi, g in enumerate(gold_spans):
            if require_type and pt != g.get("type"): 
                continue
            s = score_pair(p, g, metric)
            if s >= thr:
                pairs.append((s, pi, gi))

    # Greedy by descending score, 1-1
    pairs.sort(reverse=True, key=lambda x: x[0])
    matched_g = {}   # gi -> pi
    matched_p = {}   # pi -> gi
    for s, pi, gi in pairs:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g[gi] = pi
        matched_p[pi] = gi

    # Optional: many-to-one containment — if a gold is fully contained in a matched pred, count it matched.
    if allow_many_to_one:
        for gi, g in enumerate(gold_spans):
            if gi in matched_g: 
                continue
            for pi, p in enumerate(pred_spans):
                if pi not in matched_p: 
                    continue  # only extend from already matched preds
                if require_type and p.get("type") != g.get("type"): 
                    continue
                a0,a1 = int(p["start_char"]), int(p["end_char"])
                b0,b1 = int(g["start_char"]), int(g["end_char"])
                if b0 >= a0 and b1 <= a1:  # containment
                    matched_g[gi] = pi
                    break

    tp_idx = sorted(set(matched_p.keys()))
    tp = [pred_spans[i] for i in tp_idx]
    fp_idx = [i for i in range(len(pred_spans)) if i not in matched_p]
    fn_idx = [i for i in range(len(gold_spans)) if i not in matched_g]

    # Suppress FP shards fully inside any TP region (readability)
    if suppress_nested_fp and tp:
        tp_regions = [(int(p["start_char"]), int(p["end_char"])) for p in tp]
        keep_fp = []
        for i in fp_idx:
            p = pred_spans[i]
            a0,a1 = int(p["start_char"]), int(p["end_char"])
            contained = any(a0 >= t0 and a1 <= t1 for (t0,t1) in tp_regions)
            if not contained:
                keep_fp.append(i)
        fp_idx = keep_fp

    fp = [pred_spans[i] for i in fp_idx]
    fn = [gold_spans[i] for i in fn_idx]
    return tp, fp, fn

# ---------- Single-layer segmentation ----------
def _norm_spans(text, spans):
    n = len(text); out = []
    for s in spans:
        a = max(0, int(s["start_char"])); b = min(n, int(s["end_char"]))
        if a < b: 
            out.append((a,b))
    return out

def merge_to_fragments(text, tp, fp, fn):
    n = len(text)
    tpN = _norm_spans(text, tp)
    fpN = _norm_spans(text, fp)
    fnN = _norm_spans(text, fn)

    bounds = {0, n}
    for a,b in tpN+fpN+fnN:
        bounds.add(a); bounds.add(b)
    bounds = sorted(bounds)

    frags = []
    for i in range(len(bounds)-1):
        a, b = bounds[i], bounds[i+1]
        seg = text[a:b]
        if not seg: 
            continue
        cats = set()
        if any(x0<b and x1>a for x0,x1 in tpN): cats.add("tp")
        if any(x0<b and x1>a for x0,x1 in fpN): cats.add("fp")
        if any(x0<b and x1>a for x0,x1 in fnN): cats.add("fn")
        frags.append((seg, cats))
    # merge adjacent with same cats
    merged = []
    for seg, cats in frags:
        if merged and merged[-1][1] == cats:
            merged[-1] = (merged[-1][0] + seg, cats)
        else:
            merged.append((seg, cats))
    return merged

def build_sentence_block(sent_id, text, tp, fp, fn):
    frags = merge_to_fragments(text, tp, fp, fn)
    html_parts = []
    for seg, cats in frags:
        esc = html.escape(seg)
        if not cats:
            html_parts.append(esc)
        else:
            data = " ".join(sorted(cats))
            html_parts.append(f'<span class="mark" data-cats="{data}">{esc}</span>')
    body = "".join(html_parts)
    return f"""
    <div class="sentence" data-sentid="{sent_id}">
      <div class="sent-header">
        <span class="badge">#{sent_id}</span>
      </div>
      <div class="sent-body"><div class="single-layer">{body}</div></div>
    </div>
    """

# ---------- Theming ----------
def build_css(theme: str) -> str:
    if theme == "light":
        return r"""
        <style>
        :root {
          --bg:#f7f9ff; --fg:#0c1222; --muted:#44506b;
          --tp:#2e7d32; --fp:#c62828; --fn:#6a1b9a;
          --card:#ffffff; --accent:#3b5bfd;
          --edge:#ffffff;          /* used for color-mix borders on light */
          --chip-bg:#eef2ff; --chip-border:#dbe1ff;
          --panel-bg:#ffffff; --panel-border:#e6eaff;
          --ws: 0.06em;
        }
        *{box-sizing:border-box}
        body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;background:var(--bg);color:var(--fg)}
        header{position:sticky;top:0;z-index:10;background:linear-gradient(180deg,rgba(247,249,255,.98),rgba(247,249,255,.85));backdrop-filter:blur(6px)}
        .wrap{max-width:1200px;margin:0 auto;padding:16px 20px}
        h1{font-size:20px;margin:0 0 6px 0;font-weight:700}
        .meta{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);font-size:13px}

        .controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:10px}
        .toggle{display:flex;align-items:center;gap:6px;background:var(--chip-bg);border:1px solid var(--chip-border);border-radius:999px;padding:6px 10px;cursor:pointer;user-select:none}
        .toggle input{accent-color:var(--accent)}
        .search{flex:1;min-width:260px}
        .search input{width:100%;padding:8px 10px;border-radius:10px;border:1px solid var(--panel-border);background:#fff;color:var(--fg)}
        .slider{display:flex;align-items:center;gap:10px}
        .slider input[type=range]{width:220px}

        .grid{display:grid;grid-template-columns:1fr;gap:10px;margin-top:14px}
        .sentence{background:var(--card);border:1px solid var(--panel-border);border-radius:14px;padding:10px 12px}
        .sent-header{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:12px;margin-bottom:6px}
        .badge{background:#f3f6ff;border:1px solid var(--chip-border);border-radius:999px;padding:3px 8px}
        .sent-body{padding:8px;border-radius:10px;background:#fbfcff;border:1px dashed var(--chip-border)}
        .single-layer{white-space:pre-wrap;word-break:break-word; word-spacing: var(--ws);}
        .mark{border-radius:4px;padding:0;}
        footer{color:var(--muted);font-size:12px;padding:16px 20px}
        .pill{background:#f3f6ff;border:1px solid var(--chip-border);border-radius:999px;padding:2px 8px;font-size:12px;color:var(--muted)}
        </style>
        """
    # dark (original palette)
    return r"""
    <style>
    :root {
      --bg:#0b1020; --fg:#e6ecff; --muted:#9badc9;
      --tp:#2e7d32; --fp:#c62828; --fn:#6a1b9a;
      --card:#121a33; --accent:#4c6fff;
      --edge:black;           /* used for color-mix borders on dark */
      --chip-bg:#1a2447; --chip-border:#2a355e;
      --panel-bg:#0e1730; --panel-border:#2a355e;
      --ws: 0.06em;
    }
    *{box-sizing:border-box}
    body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;background:var(--bg);color:var(--fg)}
    header{position:sticky;top:0;z-index:10;background:linear-gradient(180deg,rgba(11,16,32,.98),rgba(11,16,32,.85));backdrop-filter:blur(6px)}
    .wrap{max-width:1200px;margin:0 auto;padding:16px 20px}
    h1{font-size:20px;margin:0 0 6px 0;font-weight:700}
    .meta{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);font-size:13px}

    .controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:10px}
    .toggle{display:flex;align-items:center;gap:6px;background:var(--chip-bg);border:1px solid var(--chip-border);border-radius:999px;padding:6px 10px;cursor:pointer;user-select:none}
    .toggle input{accent-color:var(--accent)}
    .search{flex:1;min-width:260px}
    .search input{width:100%;padding:8px 10px;border-radius:10px;border:1px solid var(--panel-border);background:var(--panel-bg);color:var(--fg)}
    .slider{display:flex;align-items:center;gap:10px}
    .slider input[type=range]{width:220px}

    .grid{display:grid;grid-template-columns:1fr;gap:10px;margin-top:14px}
    .sentence{background:var(--card);border:1px solid #1b2447;border-radius:14px;padding:10px 12px}
    .sent-header{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:12px;margin-bottom:6px}
    .badge{background:var(--panel-bg);border:1px solid var(--panel-border);border-radius:999px;padding:3px 8px}
    .sent-body{padding:8px;border-radius:10px;background:var(--panel-bg);border:1px dashed var(--panel-border)}
    .single-layer{white-space:pre-wrap;word-break:break-word; word-spacing: var(--ws);}
    .mark{border-radius:4px;padding:0;}
    footer{color:var(--muted);font-size:12px;padding:16px 20px}
    .pill{background:var(--panel-bg);border:1px solid var(--panel-border);border-radius:999px;padding:2px 8px;font-size:12px;color:var(--muted)}
    </style>
    """

def html_report_old(items, overall_counts, output_path, predicted_only=False):
    css = r"""
    <style>
    :root {
      --bg:#0b1020; --fg:#e6ecff; --muted:#9badc9;
      --tp:#2e7d32; --fp:#c62828; --fn:#6a1b9a;
      --card:#121a33; --accent:#4c6fff;
      --ws: 0.06em; /* default word spacing */
    }
    *{box-sizing:border-box}
    body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;background:var(--bg);color:var(--fg)}
    header{position:sticky;top:0;z-index:10;background:linear-gradient(180deg,rgba(11,16,32,.98),rgba(11,16,32,.85));backdrop-filter:blur(6px)}
    .wrap{max-width:1200px;margin:0 auto;padding:16px 20px}
    h1{font-size:20px;margin:0 0 6px 0;font-weight:700}
    .meta{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);font-size:13px}

    .controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:10px}
    .toggle{display:flex;align-items:center;gap:6px;background:#1a2447;border-radius:999px;padding:6px 10px;cursor:pointer;user-select:none}
    .toggle input{accent-color:var(--accent)}
    .search{flex:1;min-width:260px}
    .search input{width:100%;padding:8px 10px;border-radius:10px;border:1px solid #2a355e;background:#0e1730;color:var(--fg)}
    .slider{display:flex;align-items:center;gap:10px}
    .slider input[type=range]{width:220px}

    .grid{display:grid;grid-template-columns:1fr;gap:10px;margin-top:14px}
    .sentence{background:var(--card);border:1px solid #1b2447;border-radius:14px;padding:10px 12px}
    .sent-header{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:12px;margin-bottom:6px}
    .badge{background:#0e1730;border:1px solid #2a355e;border-radius:999px;padding:3px 8px}
    .sent-body{padding:8px;border-radius:10px;background:#0e1730;border:1px dashed #2a355e}
    .single-layer{white-space:pre-wrap;word-break:break-word; word-spacing: var(--ws);}
    .mark{border-radius:4px;padding:0;}
    footer{color:var(--muted);font-size:12px;padding:16px 20px}
    .pill{background:#0e1730;border:1px solid #2a355e;border-radius:999px;padding:2px 8px;font-size:12px;color:var(--muted)}
    </style>
    """
    js = r"""
    <script>
    (()=>{
      const $ = (q,el=document)=>el.querySelector(q);
      const $$ = (q,el=document)=>Array.from(el.querySelectorAll(q));

      const search=$("#search");
      const tpT=$("#toggle-tp");
      const fpT=$("#toggle-fp");
      const fnT=$("#toggle-fn");
      const thr=$("#thr"); const thrVal=$("#thr-val");
      const ws=$("#ws"); const wsVal=$("#ws-val");

      function styleForCats(cats, enabled){
        const on = cats.filter(c => enabled[c]);
        if(on.length===0) return {bg:"", box:""};
        if(on.length===1){
          const c = on[0];
          return {
            bg: `color-mix(in srgb, var(--${c}) 25%, transparent)`,
            box: `inset 0 0 0 1px color-mix(in srgb, var(--${c}) 60%, black)`
          };
        }
        const step=6;
        const seq = on.map((c,i)=>`var(--${c}) ${i*step}px ${(i+1)*step}px`).join(", ");
        return { bg: `repeating-linear-gradient(135deg, ${seq})`, box: `inset 0 0 0 1px #0003` };
      }

      function updateVisibility(){
        const enabled = { tp: tpT?.checked ?? true, fp: fpT?.checked ?? false, fn: fnT?.checked ?? false };
        const q=(search?.value||"").toLowerCase();
        $$(".sentence").forEach(card=>{
          const text=card.querySelector(".single-layer").textContent.toLowerCase();
          const show = text.includes(q);
          card.style.display = show ? "" : "none";
          if(!show) return;
          card.querySelectorAll(".mark").forEach(span=>{
            const cats=(span.dataset.cats||"").split(/\s+/).filter(Boolean);
            const s = styleForCats(cats, enabled);
            span.style.background = s.bg || "transparent";
            span.style.boxShadow = s.box || "none";
          });
        });
      }

      thr?.addEventListener("input", ()=>{ if(thrVal) thrVal.textContent = thr.value; });
      function applyWs(){ if(ws) { document.documentElement.style.setProperty("--ws", ws.value + "em"); if(wsVal) wsVal.textContent = ws.value; } }
      ws?.addEventListener("input", applyWs); applyWs();

      [search,tpT,fpT,fnT].forEach(el=> el?.addEventListener("input", updateVisibility));
      updateVisibility();
    })();
    </script>
    """
    # Controls differ slightly in predictions-only mode
    if predicted_only:
        controls = f"""
        <div class="controls">
          <label class="toggle"><input id="toggle-tp" type="checkbox" checked> Show predictions</label>
          <div class="slider">
            <span>Word spacing</span>
            <input id="ws" type="range" min="0" max="0.2" step="0.01" value="0.06">
            <span id="ws-val">0.06</span><span>em</span>
          </div>
          <div class="search"><input id="search" placeholder="Search sentence text..."></div>
        </div>
        """
    else:
        controls = f"""
        <div class="controls">
          <label class="toggle"><input id="toggle-tp" type="checkbox" checked> Show TP</label>
          <label class="toggle"><input id="toggle-fp" type="checkbox" checked> Show FP</label>
          <label class="toggle"><input id="toggle-fn" type="checkbox" checked> Show FN</label>
          <div class="slider">
            <span>Score ≥</span>
            <input id="thr" type="range" min="0" max="1" step="0.05" value="{overall_counts['overlap_threshold']}">
            <span id="thr-val">{overall_counts['overlap_threshold']}</span>
          </div>
          <div class="slider">
            <span>Word spacing</span>
            <input id="ws" type="range" min="0" max="0.2" step="0.01" value="0.06">
            <span id="ws-val">0.06</span><span>em</span>
          </div>
          <div class="search"><input id="search" placeholder="Search sentence text..."></div>
        </div>
        """

    mode_pill = '<span class="pill">Mode: predictions-only</span>' if predicted_only else ''
    head = f"""
    <!doctype html>
    <html lang="en">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>NER Overlap Report</title>
    {css}
    <body>
    <header>
      <div class="wrap">
        <h1>NER Overlap Report</h1>
        <div class="meta">
          <div>Total sentences: {len(items)}</div>
          <div>TP: {overall_counts['tp']} &nbsp; FP: {overall_counts['fp']} &nbsp; FN: {overall_counts['fn']}</div>
          {mode_pill}
        </div>
        {controls}
      </div>
    </header>
    <main class="wrap"><div class="grid">
    """
    body = []
    for it in items:
        body.append(build_sentence_block(it["sent_id"], it["text"], it["tp"], it["fp"], it["fn"]))
    footer_note = (
        "Predictions-only view: showing model spans as a single layer. FP/FN disabled."
        if predicted_only else
        "Greedy global matching (Dice score by default). Many-to-one containment on; nested FP shards suppressed."
    )
    tail = f"""
    </div></main>
    <footer class="wrap">{footer_note}</footer>
    {js}
    </body></html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(body) + tail)


def html_report(items, overall_counts, output_path, predicted_only=False, theme="light"):
    css = build_css(theme)
    js = r"""
    <script>
    (()=>{
      const $ = (q,el=document)=>el.querySelector(q);
      const $$ = (q,el=document)=>Array.from(el.querySelectorAll(q));

      const search=$("#search");
      const tpT=$("#toggle-tp");
      const fpT=$("#toggle-fp");
      const fnT=$("#toggle-fn");
      const thr=$("#thr"); const thrVal=$("#thr-val");
      const ws=$("#ws"); const wsVal=$("#ws-val");

      function styleForCats(cats, enabled){
        const on = cats.filter(c => enabled[c]);
        if(on.length===0) return {bg:"", box:""};
        if(on.length===1){
          const c = on[0];
          return {
            bg: `color-mix(in srgb, var(--${c}) 25%, transparent)`,
            box: `inset 0 0 0 1px color-mix(in srgb, var(--${c}) 60%, var(--edge))`
          };
        }
        const step=6;
        const seq = on.map((c,i)=>`var(--${c}) ${i*step}px ${(i+1)*step}px`).join(", ");
        return { bg: `repeating-linear-gradient(135deg, ${seq})`, box: `inset 0 0 0 1px #0003` };
      }

      function updateVisibility(){
        const enabled = { tp: tpT?.checked ?? true, fp: fpT?.checked ?? false, fn: fnT?.checked ?? false };
        const q=(search?.value||"").toLowerCase();
        $$(".sentence").forEach(card=>{
          const text=card.querySelector(".single-layer").textContent.toLowerCase();
          const show = text.includes(q);
          card.style.display = show ? "" : "none";
          if(!show) return;
          card.querySelectorAll(".mark").forEach(span=>{
            const cats=(span.dataset.cats||"").split(/\s+/).filter(Boolean);
            const s = styleForCats(cats, enabled);
            span.style.background = s.bg || "transparent";
            span.style.boxShadow = s.box || "none";
          });
        });
      }

      thr?.addEventListener("input", ()=>{ if(thrVal) thrVal.textContent = thr.value; });
      function applyWs(){ if(ws) { document.documentElement.style.setProperty("--ws", ws.value + "em"); if(wsVal) wsVal.textContent = ws.value; } }
      ws?.addEventListener("input", applyWs); applyWs();

      [search,tpT,fpT,fnT].forEach(el=> el?.addEventListener("input", updateVisibility));
      updateVisibility();
    })();
    </script>
    """
    # Controls differ slightly in predictions-only mode
    if predicted_only:
        controls = f"""
        <div class="controls">
          <span class="pill">Mode: predictions-only</span>
          <label class="toggle"><input id="toggle-tp" type="checkbox" checked> Show predictions</label>
          <div class="slider">
            <span>Word spacing</span>
            <input id="ws" type="range" min="0" max="0.2" step="0.01" value="0.06">
            <span id="ws-val">0.06</span><span>em</span>
          </div>
          <div class="search"><input id="search" placeholder="Search sentence text..."></div>
        </div>
        """
    else:
        controls = f"""
        <div class="controls">
          <label class="toggle"><input id="toggle-tp" type="checkbox" checked> Show TP</label>
          <label class="toggle"><input id="toggle-fp" type="checkbox" checked> Show FP</label>
          <label class="toggle"><input id="toggle-fn" type="checkbox" checked> Show FN</label>
          <div class="slider">
            <span>Score ≥</span>
            <input id="thr" type="range" min="0" max="1" step="0.05" value="{overall_counts['overlap_threshold']}">
            <span id="thr-val">{overall_counts['overlap_threshold']}</span>
          </div>
          <div class="slider">
            <span>Word spacing</span>
            <input id="ws" type="range" min="0" max="0.2" step="0.01" value="0.06">
            <span id="ws-val">0.06</span><span>em</span>
          </div>
          <div class="search"><input id="search" placeholder="Search sentence text..."></div>
        </div>
        """

    head = f"""
    <!doctype html>
    <html lang="en" data-theme="{theme}">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>NER Overlap Report</title>
    {css}
    <body>
    <header>
      <div class="wrap">
        <h1>NER Overlap Report</h1>
        <div class="meta">
          <div>Total sentences: {len(items)}</div>
          <div>TP: {overall_counts['tp']} &nbsp; FP: {overall_counts['fp']} &nbsp; FN: {overall_counts['fn']}</div>
          <span class="pill">Theme: {theme}</span>
        </div>
        {controls}
      </div>
    </header>
    <main class="wrap"><div class="grid">
    """
    body = []
    for it in items:
        body.append(build_sentence_block(it["sent_id"], it["text"], it["tp"], it["fp"], it["fn"]))
    footer_note = (
        "Predictions-only view: showing model spans as a single layer. FP/FN disabled."
        if predicted_only else
        "Greedy global matching (Dice score by default). Many-to-one containment on; nested FP shards suppressed."
    )
    tail = f"""
    </div></main>
    <footer class="wrap">{footer_note}</footer>
    {js}
    </body></html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(body) + tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_file", help="Optional gold JSONL; omit for predictions-only mode", required=False)
    ap.add_argument("--model_file", required=True)
    ap.add_argument("--output_html", required=True)
    ap.add_argument("--overlap_threshold", type=float, default=0.5)
    ap.add_argument("--metric", choices=["dice","iou","min"], default="dice")
    ap.add_argument("--require_type", action="store_true", default=False)
    ap.add_argument("--no_require_type", dest="require_type", action="store_false")
    ap.add_argument("--pred_source", choices=["accepted","accepted_and_missing"], default="accepted_and_missing")
    ap.add_argument("--allow_many_to_one", action="store_true", default=True)
    ap.add_argument("--no_many_to_one", dest="allow_many_to_one", action="store_false")
    ap.add_argument("--suppress_nested_fp", action="store_true", default=True)
    ap.add_argument("--no_suppress_nested_fp", dest="suppress_nested_fp", action="store_false")
    ap.add_argument("--theme", choices=["light","dark"], default="light")
    args = ap.parse_args()

    gold = load_spans(args.gold_file) if args.gold_file else {}
    model = load_spans(args.model_file)

    keys = sorted(set(gold.keys()) | set(model.keys())) if gold else sorted(set(model.keys()))
    items = []
    counts = {"tp":0, "fp":0, "fn":0, "overlap_threshold": args.overlap_threshold}

    # Detect predictions-only mode if no gold provided OR there are zero gold spans overall
    gold_total_spans = 0
    if gold:
        for k in gold.keys():
            gold_total_spans += len(extract_gold_spans(gold.get(k, [])))
    predictions_only = (not gold) or (gold_total_spans == 0)


    for k in keys:
        g_list = gold.get(k, []) if gold else []
        m_list = model.get(k, [])
        text = extract_sentence_text(g_list or m_list)
        g_spans = extract_gold_spans(g_list) if not predictions_only else []
        p_spans = extract_model_spans(m_list, args.pred_source)

        if predictions_only:
            # Just show predictions as a single layer (use TP channel for color)
            tp, fp, fn = p_spans, [], []
        else:
            tp, fp, fn = categorize_spans(
                g_spans, p_spans, args.overlap_threshold,
                metric=args.metric, require_type=args.require_type,
                allow_many_to_one=args.allow_many_to_one,
                suppress_nested_fp=args.suppress_nested_fp
            )
        counts["tp"] += len(tp); counts["fp"] += len(fp); counts["fn"] += len(fn)

        items.append({"sent_id": f"{k[0]}:{k[1]}", "text": text or "", "tp": tp, "fp": fp, "fn": fn})

    html_report(items, counts, args.output_html, predicted_only=predictions_only, theme=args.theme)

if __name__ == "__main__":
    main()
