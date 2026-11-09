"""
Render a built graph as one self-contained HTML file — no database needed.

Neo4j is the place to query the graph; this is the place to look at it without
running anything. The file embeds ``nodes.jsonl`` and ``edges.jsonl`` and draws
them with vis-network (loaded from a pinned CDN version), so it can be mailed
to a collaborator and opened in a browser.

Filters: document, relation type, minimum evidence, and a search box that
narrows the view to one entity's neighbourhood. Clicking a node shows its
surface forms and authority link; clicking an edge shows the sentences that
state it. The sentences come from ``sentences.sqlite`` next to the graph
(``text_store``) and are embedded once each; without it the page draws the
graph with no sentences.

Usage
-----
    python -m src.graph.render_html --in-dir output/kg/second_iteration
    # -> output/kg/second_iteration/graph.html
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from src.graph.text_store import TextStore, mention_parts, short_doc

VIS_NETWORK = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"

_URL_OF_CURIE = {
    "gbif": "https://www.gbif.org/species/{}",
    "geonames": "https://www.geonames.org/{}",
    "wdpa": "https://www.protectedplanet.net/{}",
    "gmba": "https://www.earthenv.org/mountains#{}",
}


def _utf16(text: str, i: int) -> int:
    """A Python (code point) offset as a JavaScript (UTF-16) one."""
    return len(text[:i].encode("utf-16-le")) // 2


def _compact(nodes: List[dict], edges: List[dict], store: Optional[TextStore] = None) -> dict:
    """Only what the page draws or shows; full evidence stays in edges.jsonl.

    Text is embedded once: ``sentences`` is [label, text] per sentence, and
    ``mentions`` is [sentence index, mark start, mark end, note] per mention
    (start -1: nothing to mark). A node lists its mention indices (``s``), a
    concept those of every paper; an edge lists sentence indices.
    """
    sent_ix: dict = {}
    sentences: list = []
    mention_ix: dict = {}
    mentions: list = []
    by_node, by_concept = defaultdict(list), defaultdict(list)
    if store is not None:
        for m in store.con.execute(
                "SELECT m.*, s.text AS sentence FROM mention m "
                "JOIN sentence s ON s.id = m.sentence_id"):
            m = dict(m)
            if m["node_id"]:
                by_node[m["node_id"]].append(m)
            if m["concept_id"]:
                by_concept[m["concept_id"]].append(m)

    section: dict = {}
    if store is not None:
        section = {r[0]: " / ".join(v for v in r[1:] if v) for r in store.con.execute(
            "SELECT id, section, subsection FROM sentence")}

    def sentence(sid: str, doc_id: str, sent_index: int, text: str) -> int:
        if sid not in sent_ix:
            sent_ix[sid] = len(sentences)
            where = f" ({section[sid]})" if section.get(sid) else ""
            sentences.append([f"{short_doc(doc_id)} s{sent_index}{where}", text])
        return sent_ix[sid]

    def mention(m: dict) -> int:
        """Marked as text_store.render_mention does, so both read the same."""
        text = m["sentence"]
        marked, note = mention_parts(m, text)
        si = sentence(m["sentence_id"], m["doc_id"], m["sent_index"], text)
        rec = (si, _utf16(text, m["start_char"]) if marked else -1,
               _utf16(text, m["end_char"]) if marked else -1, note)
        if rec not in mention_ix:
            mention_ix[rec] = len(mentions)
            mentions.append(list(rec))
        return mention_ix[rec]

    def order(ms: List[dict]) -> List[dict]:
        seen, out = set(), []
        for m in sorted(ms, key=lambda r: (r["doc_id"], r["sent_index"],
                                            r["start_char"] if r["start_char"] is not None else -1,
                                            r["key"])):
            if m["key"] not in seen:
                seen.add(m["key"])
                out.append(m)
        return out

    out_nodes, papers = [], {}
    for n in nodes:
        if n["kind"] == "Paper":
            # shown with the edges and the document filter, not drawn
            papers[n["pmcid"]] = {k: n.get(k) for k in (
                "citation", "title", "journal", "year", "doi") if n.get(k)}
            continue
        row = {k: n.get(k) for k in (
            "id", "kind", "name", "base_name", "entity_type", "role", "grounded",
            "global", "curie", "n_instances", "value_concept", "value_kind", "value", "low", "high",
            "unit", "comparator", "uncertainty", "stat", "lat", "lon", "raw",
            "authority_uri", "authority_label", "surface_forms", "docs", "doc_id",
            "n_mentions", "degree", "reason", "n_candidates", "candidate_only", "wikidata")
            if n.get(k) not in (None, [], "")}
        ms = by_node.get(n["id"], []) + (by_concept.get(n["id"], [])
                                          if n["kind"] == "Entity" else [])
        if ms:
            row["s"] = list(dict.fromkeys(mention(m) for m in order(ms)))
        out_nodes.append(row)
    out_edges = []
    for e in edges:
        row = {"source": e["source"], "target": e["target"], "relation": e["relation"],
               "property": e.get("property") or "",
               "n": e.get("n_evidence", 1), "docs": e.get("docs") or []}
        if e.get("feature"):
            row["feature"] = e["feature"]
        if e.get("docs_naming_candidate"):
            row["docs_naming_candidate"] = e["docs_naming_candidate"]
        ev = e.get("evidence") or []
        if ev:
            if store is not None:
                first = {}
                for x in ev:
                    first.setdefault(x["sentence_id"], x)
                row["s"] = [sentence(sid, x["doc_id"], x["sent_index"],
                                     store.sentence(sid) or "")
                            for sid, x in sorted(first.items())]
            row["confidence"] = e.get("confidence_mean")
            row["distributed"] = e.get("distributed")
        out_edges.append(row)
    return {"nodes": out_nodes, "edges": out_edges, "sentences": sentences,
            "mentions": mentions, "papers": papers}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__VIS__"></script>
<style>
:root {
  --bg: #fbfbf9; --panel: #ffffff; --ink: #1f2328; --muted: #646b73;
  --line: #d9dcdf; --accent: #2f6fb0;
  --c-entity: #3b7dd8; --c-process: #2e9e6b; --c-property: #d98a1f;
  --c-concept: #8a5cc7; --c-unresolved: #9aa0a6; --c-term: #c94f6d;
  --c-negated: #d1453b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #15171a; --panel: #1d2024; --ink: #e6e8ea; --muted: #9aa1a9;
    --line: #33383e; --accent: #6aa5e6;
  }
}
:root[data-theme="dark"] {
  --bg: #15171a; --panel: #1d2024; --ink: #e6e8ea; --muted: #9aa1a9;
  --line: #33383e; --accent: #6aa5e6;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  display: grid; grid-template-columns: 300px 1fr; height: 100vh; }
aside { border-right: 1px solid var(--line); background: var(--panel);
  padding: 16px; overflow-y: auto; }
h1 { font-size: 16px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 12px; margin-bottom: 16px; }
label { display: block; font-size: 12px; color: var(--muted); margin: 12px 0 4px; }
input[type=search], select { width: 100%; padding: 6px 8px; font: inherit;
  color: var(--ink); background: var(--bg); border: 1px solid var(--line);
  border-radius: 6px; }
.row { display: flex; gap: 8px; align-items: center; font-size: 13px; margin: 4px 0; }
.rels { max-height: 180px; overflow-y: auto; border: 1px solid var(--line);
  border-radius: 6px; padding: 4px 8px; }
.legend span { display: inline-flex; align-items: center; gap: 4px;
  margin: 0 10px 4px 0; font-size: 12px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
#net { height: 100vh; }
#info { margin-top: 16px; border-top: 1px solid var(--line); padding-top: 12px;
  font-size: 13px; word-wrap: break-word; }
#info .k { color: var(--muted); }
#info ul { padding-left: 18px; margin: 4px 0; }
#stat { font-size: 12px; color: var(--muted); margin-top: 8px; }
.sents { max-height: 45vh; overflow-y: auto; }
.sents li { margin-bottom: 6px; }
mark { background: color-mix(in srgb, var(--accent) 25%, transparent); color: inherit;
  border-radius: 3px; padding: 0 2px; }
a { color: var(--accent); }
button { font: inherit; padding: 4px 10px; border: 1px solid var(--line);
  background: var(--bg); color: var(--ink); border-radius: 6px; cursor: pointer; }
@media (max-width: 720px) {
  body { grid-template-columns: 1fr; grid-template-rows: auto 70vh; height: auto; }
  aside { border-right: 0; border-bottom: 1px solid var(--line); }
  #net { height: 70vh; }
}
</style>
</head>
<body>
<aside>
  <h1>__TITLE__</h1>
  <div class="sub">Entities linked to ontologies, joined by annotated relations.</div>
  <div class="legend">
    <span><i class="dot" style="background:var(--c-entity)"></i>entity</span>
    <span><i class="dot" style="background:var(--c-process)"></i>process</span>
    <span><i class="dot" style="background:var(--c-property)"></i>property</span>
    <span><i class="dot" style="background:var(--c-concept)"></i>concept</span>
    <span><i class="dot" style="background:var(--c-unresolved)"></i>unresolved</span>
    <span>▲ ambiguous place (may be …)</span>
    <span><i class="dot" style="background:var(--c-term)"></i>ontology term</span>
  </div>
  <div class="sub">Colour is the entity type's role. A box is a measured value
    (2456 m, 25%, p &lt; 0.001). Otherwise shape is the layer:
    ● a node in one paper (<i>elevation · PMC6920850</i>) &nbsp;■ the shared node all
    papers' versions point to, of any type &nbsp;◆ an ontology term. Named places and
    taxa (thick border) are shared directly.</div>
  <label for="q">Focus on entity</label>
  <input id="q" type="search" list="names" placeholder="e.g. elevation">
  <datalist id="names"></datalist>
  <div class="row"><span>Hops</span>
    <select id="hops" style="width:auto"><option>1</option><option selected>2</option></select>
    <button id="clear">Clear</button></div>
  <label>Documents <span id="docn"></span></label>
  <div class="row"><button id="docall">All</button><button id="docnone">None</button></div>
  <div class="rels" id="docs"></div>
  <label for="minev">Minimum evidence: <b id="minevv">1</b></label>
  <input id="minev" type="range" min="1" max="5" value="1" style="width:100%">
  <div class="row"><input type="checkbox" id="concepts"><label for="concepts" style="margin:0">Show shared nodes (link papers)</label></div>
  <div class="row"><input type="checkbox" id="terms"><label for="terms" style="margin:0">Show ontology terms</label></div>
  <div class="row"><input type="checkbox" id="unres" checked><label for="unres" style="margin:0">Show unresolved mentions</label></div>
  <div class="row"><input type="checkbox" id="negated" checked><label for="negated" style="margin:0">Show negated statements <span style="color:var(--c-negated)">(red, NOT …)</span></label></div>
  <label>Relation types</label>
  <div class="rels" id="rels"></div>
  <div id="stat"></div>
  <div id="info">Click a node or an edge.</div>
</aside>
<div id="net"></div>
<script>
const DATA = __DATA__;
const URLS = __URLS__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ROLE_VAR = {ENTITY: "--c-entity", PROCESS: "--c-process", PROPERTY: "--c-property",
                  CONCEPT: "--c-concept"};
function colorOf(n) {
  if (n.kind === "OntologyTerm") return css("--c-term");
  if (n.kind === "Unresolved") return css("--c-unresolved");
  return css(ROLE_VAR[n.role] || "--c-concept");
}
const byId = new Map(DATA.nodes.map(n => [n.id, n]));
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
// One mention in its sentence: "PMC… s37: … «elevations» …", marked by offsets.
const mentionHtml = i => {
  const [si, a, b, note] = DATA.mentions[i], [label, t] = DATA.sentences[si];
  const body = a >= 0 ? esc(t.slice(0, a)) + "<mark>" + esc(t.slice(a, b)) + "</mark>" + esc(t.slice(b))
                      : esc(t);
  return `<li>${esc(label)}: ${body}${esc(note)}</li>`;
};

// "PMC6920850#00" -> its paper: "Temu et al. 2019 · Microorganisms (PMC6920850)"
const paperOf = d => (DATA.papers || {})[d.split("#")[0]] || {};
const paperName = d => { const p = paperOf(d), id = d.replace(/#00$/, "");
  return p.citation ? `${p.citation} · ${p.journal || ""} (${id})` : id; };
const paperHtml = d => { const p = paperOf(d);
  return `<li><b>${esc(p.citation || d.replace(/#00$/, ""))}</b>${p.title ? ": " + esc(p.title) : ""}`
    + `${p.journal ? ". <i>" + esc(p.journal) + "</i>" : ""}${p.year ? ", " + p.year : ""}`
    + `${p.doi ? `. <a href="https://doi.org/${esc(p.doi)}" target="_blank" rel="noopener">doi:${esc(p.doi)}</a>` : ""}</li>`; };

const STRUCTURAL = new Set(["EXACT_MATCH", "INSTANCE_OF", "MAY_BE", "WITHIN"]);
const relTypes = [...new Set(DATA.edges.map(e => e.relation))].filter(r => !STRUCTURAL.has(r)).sort();
const relBox = document.getElementById("rels");
relTypes.forEach(r => {
  const n = DATA.edges.filter(e => e.relation === r).length;
  relBox.insertAdjacentHTML("beforeend",
    `<div class="row"><input type="checkbox" class="rel" value="${esc(r)}" checked>${esc(r)} <span class="k" style="color:var(--muted)">${n}</span></div>`);
});
// Several documents at once, not all-or-one: the whole corpus is too dense to
// read. Opens on the first document only.
const docs = [...new Set(DATA.edges.flatMap(e => e.docs))].sort();
const docBox = document.getElementById("docs");
docs.forEach((d, i) => {
  const n = DATA.edges.filter(e => e.docs.includes(d)).length;
  docBox.insertAdjacentHTML("beforeend",
    `<div class="row"><input type="checkbox" class="docc" value="${esc(d)}" ${i === 0 ? "checked" : ""}><span title="${esc(paperOf(d).title || "")}">${esc(paperName(d))}</span> <span class="k" style="color:var(--muted)">${n}</span></div>`);
});
const setDocs = on => { document.querySelectorAll(".docc").forEach(x => x.checked = on); render(); };
document.getElementById("docall").addEventListener("click", () => setDocs(true));
document.getElementById("docnone").addEventListener("click", () => setDocs(false));
// Search by the shared name, so "elevation" finds every paper's elevation.
const names = [...new Set(DATA.nodes.filter(n => n.kind !== "OntologyTerm" && n.degree)
                                   .map(n => n.base_name || n.name))].sort();
document.getElementById("names").innerHTML = names.map(n => `<option value="${esc(n)}">`).join("");
const maxN = Math.max(1, ...DATA.edges.map(e => e.n));
document.getElementById("minev").max = Math.min(maxN, 10);

const nodesDS = new vis.DataSet(), edgesDS = new vis.DataSet();
const net = new vis.Network(document.getElementById("net"), {nodes: nodesDS, edges: edgesDS}, {
  layout: {improvedLayout: false},
  physics: {solver: "forceAtlas2Based", stabilization: {iterations: 250},
            forceAtlas2Based: {gravitationalConstant: -40, springLength: 90}},
  interaction: {hover: true, tooltipDelay: 150},
  nodes: {shape: "dot", font: {size: 12, color: css("--ink")}},
  edges: {arrows: {to: {enabled: true, scaleFactor: 0.5}}, smooth: false,
          font: {size: 9, color: css("--muted"), strokeWidth: 0, align: "middle"},
          color: {color: css("--line"), highlight: css("--accent"), hover: css("--accent")}},
});

function render() {
  const relOn = new Set([...document.querySelectorAll(".rel:checked")].map(x => x.value));
  const showConcepts = document.getElementById("concepts").checked;
  const showTerms = document.getElementById("terms").checked;
  const showUnres = document.getElementById("unres").checked;
  const showNegated = document.getElementById("negated").checked;
  const docOn = new Set([...document.querySelectorAll(".docc:checked")].map(x => x.value));
  document.getElementById("docn").textContent = `(${docOn.size} of ${docs.length})`;
  const minEv = +document.getElementById("minev").value;
  document.getElementById("minevv").textContent = minEv;
  let edges = DATA.edges.filter(e => relOn.has(e.relation) && e.n >= minEv
    && e.docs.some(d => docOn.has(d))
    && (showNegated || e.property !== "Negation")
    && (showUnres || (byId.get(e.source).kind !== "Unresolved"
                      && byId.get(e.target).kind !== "Unresolved")));
  const instOf = showConcepts ? DATA.edges.filter(e => e.relation === "INSTANCE_OF") : [];
  const q = document.getElementById("q").value.trim().toLowerCase();
  if (q) {
    let keep = new Set(DATA.nodes.filter(n => n.kind !== "OntologyTerm"
      && (n.base_name || n.name || "").toLowerCase() === q).map(n => n.id));
    const walk = edges.concat(instOf);
    for (let h = 0; h < +document.getElementById("hops").value; h++) {
      const next = new Set(keep);
      walk.forEach(e => { if (keep.has(e.source)) next.add(e.target);
                          if (keep.has(e.target)) next.add(e.source); });
      keep = next;
    }
    edges = edges.filter(e => keep.has(e.source) && keep.has(e.target));
  }
  // Structural edges hang off what is drawn: an instance's concept, then the
  // ontology term of any drawn concept or shared entity.
  const ids = new Set(edges.flatMap(e => [e.source, e.target]));
  const extra = instOf.filter(e => ids.has(e.source));
  extra.forEach(e => ids.add(e.target));
  // An ambiguous place name always shows the places it may be.
  DATA.edges.filter(e => e.relation === "MAY_BE" && ids.has(e.source))
            .forEach(e => { extra.push(e); ids.add(e.target); });
  if (showTerms) {
    DATA.edges.filter(e => e.relation === "EXACT_MATCH" && ids.has(e.source))
              .forEach(e => { extra.push(e); ids.add(e.target); });
    // A mountain range's parents (GMBA): Udzungwa within Eastern Arc Mountains, ...
    for (let grew = true; grew;) {
      grew = false;
      DATA.edges.filter(e => e.relation === "WITHIN" && ids.has(e.source) && !ids.has(e.target))
                .forEach(e => { extra.push(e); ids.add(e.target); grew = true; });
    }
    DATA.edges.filter(e => e.relation === "WITHIN" && ids.has(e.source) && ids.has(e.target)
                      && !extra.includes(e)).forEach(e => extra.push(e));
  }
  edges = edges.concat(extra);
  nodesDS.clear(); edgesDS.clear();
  nodesDS.add([...ids].map(id => {
    const n = byId.get(id);
    return {id, label: n.name, color: {background: colorOf(n), border: n.grounded ? css("--ink") : colorOf(n)},
            borderWidth: n.grounded ? 3 : 1, size: 6 + Math.min(18, Math.sqrt(n.degree || 1) * 3),
            shape: n.kind === "OntologyTerm" ? "diamond"
                 : n.kind === "Measurement" ? "box"
                 : n.kind === "AmbiguousPlace" ? "triangle"
                 : (n.kind === "Entity" && !n.global ? "square" : "dot"),
            title: `${n.name}\n${n.entity_type || n.curie || ""}`};
  }));
  edgesDS.add(edges.map((e, i) => {
    const neg = e.property === "Negation";
    return {id: i, from: e.source, to: e.target,
      label: STRUCTURAL.has(e.relation) ? "" : relLabel(e), width: Math.min(6, e.n),
      dashes: STRUCTURAL.has(e.relation) || neg || !!e.distributed, _e: e,
      ...(neg ? {color: {color: css("--c-negated"), highlight: css("--c-negated")},
                 font: {color: css("--c-negated")}} : {}),
      ...(e.relation === "MAY_BE" ? {label: "may be", font: {size: 9}} : {}),
      ...(e.relation === "WITHIN" ? {label: "within", font: {size: 9}} : {}),
      title: STRUCTURAL.has(e.relation) ? e.relation.toLowerCase().replace("_", " ")
                                        : `${relLabel(e)} — ${e.n} statement(s)`};
  }));
  document.getElementById("stat").textContent = `${ids.size} nodes, ${edges.length} edges shown`;
  // Changing the data does not always restart stabilisation, so start it
  // explicitly; fit once the layout has settled, on every redraw.
  net.once("stabilizationIterationsDone", () => net.fit());
  net.stabilize(250);
}

// A negated relation reads as its denial; the other properties qualify a
// relation that holds, so they are shown beside its name.
const PROP_MARK = {Negative_polarity: "(−)", Exclusive: "(only)", Dominance: "(mainly)"};
function relLabel(e) {
  if (e.property === "Negation") return `NOT ${e.relation}`;
  if (!e.property) return e.relation;
  return `${e.relation} ${PROP_MARK[e.property] || "(" + e.property + ")"}`;
}

// A parsed value as a person would write it: "2456 m", "6–10 µm", "p < 0.001".
function valueText(n) {
  if (n.value_kind === "coordinate") return `${n.lat ?? ""}, ${n.lon ?? ""}`;
  if (n.value_kind === "statistic") return `${n.stat} ${n.comparator} ${n.value}`;
  const cmp = n.comparator && n.comparator !== "=" ? n.comparator + " " : "";
  const num = n.low != null ? `${n.low}–${n.high}` : `${n.value}${n.uncertainty != null ? " ± " + n.uncertainty : ""}`;
  return `${cmp}${num}${n.unit ? (n.unit === "%" ? "%" : " " + n.unit) : ""}`;
}

function linkFor(n) {
  if (!n.curie) return "";
  const [p, local] = n.curie.split(":");
  const tpl = URLS[p];
  const url = n.wikidata || (tpl ? tpl.replace("{}", local) : n.authority_uri || n.id);
  return `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(n.curie)}</a>`;
}
net.on("click", p => {
  const info = document.getElementById("info");
  if (p.nodes.length) {
    const n = byId.get(p.nodes[0]);
    info.innerHTML = `<b>${esc(n.name)}</b><br><span class="k">${esc(n.kind)}${n.entity_type ? " · " + esc(n.entity_type) : ""}</span>`
      + (n.curie ? `<br>${linkFor(n)}${n.authority_label ? " — " + esc(n.authority_label) : ""}` : "")
      + (n.surface_forms ? `<div class="k">Surface forms</div><ul>${n.surface_forms.map(f => `<li>${esc(f)}</li>`).join("")}</ul>` : "")
      + (n.docs ? `<div class="k">Documents: ${n.docs.map(esc).join(", ")}</div>` : "")
      + (n.kind === "Measurement" ? `<div class="k">Value: <b>${esc(valueText(n))}</b> (${esc(n.value_kind)})</div>` : "")
      + (n.n_instances ? `<div class="k">${n.n_instances} instance(s) across papers</div>` : "")
      + (n.value_concept ? `<div class="k">Shared quality: ${DATA.edges.filter(e => e.target === n.id && !STRUCTURAL.has(e.relation)).length} relation(s) from ${new Set(DATA.edges.filter(e => e.target === n.id).flatMap(e => e.docs)).size} paper(s); click an edge for its sentences</div>` : "")
      + (n.kind === "AmbiguousPlace" ? `<div class="k">Ambiguous place name; it may be:</div><ul>${DATA.edges.filter(e => e.source === n.id && e.relation === "MAY_BE").map(e => { const t = byId.get(e.target);
          return `<li>${esc(t.name)} ${linkFor(t)}${e.feature ? " · " + esc(e.feature) : ""}${e.docs_naming_candidate ? " — also named in " + e.docs_naming_candidate.map(esc).join(", ") : ""}</li>`; }).join("")}</ul>` : "")
      + (n.candidate_only ? `<div class="k">Candidate only: no mention links here directly</div>` : "")
      + (n.reason && n.kind === "Unresolved" ? `<div class="k">Unresolved: ${esc(n.reason)}</div>` : "")
      + (n.s ? `<div class="k">Sentences (${n.s.length})</div><ul class="sents">${n.s.map(mentionHtml).join("")}</ul>` : "");
  } else if (p.edges.length) {
    const e = edgesDS.get(p.edges[0])._e;
    const s = byId.get(e.source), t = byId.get(e.target);
    info.innerHTML = `<b>${esc(s.name)}</b> —${esc(relLabel(e))}→ <b>${esc(t.name)}</b>`
      + (e.property ? `<div class="k">Relation property: <b>${esc(e.property)}</b>${
          e.property === "Negation" ? " — the sentence says this does NOT hold" : ""}</div>` : "")
      + `<div class="k">${e.n} statement(s)${e.confidence != null ? ", mean confidence " + e.confidence : ""}${e.distributed ? ", distributed over a coordination" : ""}</div>`
      + (e.docs.length ? `<div class="k">Paper(s)</div><ul class="sents">${e.docs.map(paperHtml).join("")}</ul>` : "")
      + (e.s ? `<ul class="sents">${e.s.map(i => `<li>${esc(DATA.sentences[i][0])}: ${esc(DATA.sentences[i][1])}</li>`).join("")}</ul>` : "");
  }
});
net.on("doubleClick", p => { if (p.nodes.length) {
  document.getElementById("q").value = byId.get(p.nodes[0]).name; render(); } });
["minev", "concepts", "terms", "unres", "negated", "hops"].forEach(id => document.getElementById(id).addEventListener("input", render));
document.getElementById("q").addEventListener("change", render);
relBox.addEventListener("change", render);
docBox.addEventListener("change", render);
document.getElementById("clear").addEventListener("click", () => { document.getElementById("q").value = ""; render(); });
render();
</script>
</body>
</html>
"""


def render(nodes: List[dict], edges: List[dict], title: str,
           vis_src: str = VIS_NETWORK, store: Optional[TextStore] = None) -> str:
    """The page. ``vis_src`` is where the browser loads vis-network from: the
    pinned CDN by default, or a local copy for a deployment that should not
    depend on a third party (deploy_graph/)."""
    data = json.dumps(_compact(nodes, edges, store), ensure_ascii=False, sort_keys=True)
    # A sentence containing "</script>" would otherwise end the script block.
    data = data.replace("</", "<\\/")
    return (PAGE.replace("__TITLE__", title)
                .replace("__VIS__", vis_src)
                .replace("__URLS__", json.dumps(_URL_OF_CURIE))
                .replace("__DATA__", data))


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <in-dir>/graph.html")
    ap.add_argument("--title", default=None)
    ap.add_argument("--vis-src", default=VIS_NETWORK,
                    help="URL or relative path of vis-network.min.js (default: pinned CDN)")
    args = ap.parse_args(argv)

    def read(name):
        with (args.in_dir / name).open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    out = args.out or args.in_dir / "graph.html"
    title = args.title or f"MoBiKo graph · {args.in_dir.resolve().name}"
    db = args.in_dir / "sentences.sqlite"
    store = TextStore(db) if db.exists() else None
    if store is None:
        print(f"  note: no {db}; the page will show no sentences")
    out.write_text(render(read("nodes.jsonl"), read("edges.jsonl"), title, args.vis_src,
                          store=store), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB) — open it in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
