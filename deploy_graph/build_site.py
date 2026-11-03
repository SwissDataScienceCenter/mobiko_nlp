"""
Assemble the static site the graph-viewer image serves.

    site/
      index.html                  one card per graph
      vendor/vis-network.min.js   pinned and hash-checked, so the page does not
                                  depend on a CDN being reachable
      <graph>/index.html          src/graph/render_html.py output for that graph

A graph is any directory holding ``nodes.jsonl`` and ``edges.jsonl`` from
``src.graph.build_graph`` — by default every ``output/kg/*/``.

Usage (from the repo root; deploy_graph/build.sh calls this):
    python deploy_graph/build_site.py
    python deploy_graph/build_site.py --graph output/kg/second_iteration --out deploy_graph/site
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.graph.build_graph import STRUCTURAL  # noqa: E402
from src.graph.render_html import VIS_NETWORK, render  # noqa: E402
from src.graph.text_store import TextStore  # noqa: E402

VIS_SHA256 = "f53f833ddb9bf97efe856bb0637d4fe88f39e39999c7e94a4b8afc8de8a1a2e5"


def fetch_vis(dest: Path, cache: Path) -> None:
    """The pinned vis-network build, verified against its hash before use."""
    if not cache.exists() or hashlib.sha256(cache.read_bytes()).hexdigest() != VIS_SHA256:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(VIS_NETWORK, timeout=60) as r:
            data = r.read()
        got = hashlib.sha256(data).hexdigest()
        if got != VIS_SHA256:
            raise SystemExit(f"vis-network hash mismatch: got {got}, pinned {VIS_SHA256}")
        cache.write_bytes(data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache, dest)


def _read(path: Path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summary(graph_dir: Path, nodes, edges) -> dict:
    meta_path = graph_dir / "graph.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    semantic = [e for e in edges if e["relation"] not in STRUCTURAL]
    return {
        "name": graph_dir.name,
        "docs": len({d for e in semantic for d in e.get("docs", [])}),
        "edges": len(semantic),
        "nodes": sum(1 for n in nodes if n["kind"] != "OntologyTerm"),
        "grounded": sum(1 for n in nodes if n["kind"] == "Entity" and n.get("grounded")),
        "annotators": sorted({a for e in semantic for a in e.get("annotators", [])}),
        "generated": (meta.get("generated_at") or "")[:10],
        "scope": meta.get("scope"),
    }


INDEX = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MoBiKo graphs</title>
<style>
:root { --bg: #fbfbf9; --panel: #ffffff; --ink: #1f2328; --muted: #646b73;
  --line: #d9dcdf; --accent: #2f6fb0; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg: #15171a; --panel: #1d2024; --ink: #e6e8ea;
    --muted: #9aa1a9; --line: #33383e; --accent: #6aa5e6; }
}
:root[data-theme="dark"] { --bg: #15171a; --panel: #1d2024; --ink: #e6e8ea;
  --muted: #9aa1a9; --line: #33383e; --accent: #6aa5e6; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 32px 16px; }
h1 { font-size: 22px; margin: 0 0 4px; }
p.sub { color: var(--muted); margin: 0 0 24px; }
a.card { display: block; background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; color: inherit;
  text-decoration: none; }
a.card:hover { border-color: var(--accent); }
.card h2 { font-size: 17px; margin: 0 0 6px; color: var(--accent); }
.stats { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--muted); }
.stats b { color: var(--ink); font-weight: 600; }
</style>
</head>
<body>
<main>
<h1>MoBiKo knowledge graphs</h1>
<p class="sub">Entities from the annotation runs, linked to ontologies and joined by
the annotated relations. Pick a graph to explore it.</p>
__CARDS__
</main>
</body>
</html>
"""


def card(s: dict) -> str:
    ann = ", ".join(a.removeprefix("agent:") for a in s["annotators"]) or "—"
    return (f'<a class="card" href="{html.escape(s["name"])}/">'
            f'<h2>{html.escape(s["name"])}</h2><div class="stats">'
            f'<span><b>{s["docs"]}</b> documents</span>'
            f'<span><b>{s["edges"]:,}</b> relations</span>'
            f'<span><b>{s["nodes"]:,}</b> nodes</span>'
            f'<span><b>{s["grounded"]:,}</b> entities linked to ontologies</span>'
            f'<span>annotated by <b>{html.escape(ann)}</b></span>'
            f'<span>built {html.escape(s["generated"])}</span></div></a>')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", type=Path, action="append", dest="graphs",
                    help="graph directory (repeatable; default: every output/kg/*/)")
    ap.add_argument("--out", type=Path, default=REPO / "deploy_graph" / "site")
    args = ap.parse_args(argv)

    graphs = args.graphs or sorted(p for p in (REPO / "output" / "kg").glob("*/")
                                   if (p / "nodes.jsonl").exists())
    missing = [g for g in graphs if not (g / "nodes.jsonl").exists()]
    if not graphs or missing:
        print(f"no graph in: {missing or 'output/kg/*/'} — run `make kg` first",
              file=sys.stderr)
        return 1
    names = Counter(g.resolve().name for g in graphs)
    if any(n > 1 for n in names.values()):
        print(f"two graphs share a directory name: {sorted(k for k, n in names.items() if n > 1)}",
              file=sys.stderr)
        return 1

    if args.out.exists():
        shutil.rmtree(args.out)
    fetch_vis(args.out / "vendor" / "vis-network.min.js",
              REPO / "deploy_graph" / ".cache" / "vis-network-9.1.9.min.js")
    cards = []
    for g in graphs:
        g = g.resolve()
        nodes, edges = _read(g / "nodes.jsonl"), _read(g / "edges.jsonl")
        page = args.out / g.name / "index.html"
        page.parent.mkdir(parents=True)
        db = g / "sentences.sqlite"
        store = TextStore(db) if db.exists() else None
        page.write_text(render(nodes, edges, f"MoBiKo graph · {g.name}",
                               vis_src="../vendor/vis-network.min.js", store=store),
                        encoding="utf-8")
        s = summary(g, nodes, edges)
        cards.append(card(s))
        print(f"  {g.name}: {s['docs']} docs, {s['edges']:,} relations, "
              f"{page.stat().st_size / 1e6:.1f} MB")
    (args.out / "index.html").write_text(INDEX.replace("__CARDS__", "\n".join(cards)),
                                         encoding="utf-8")
    print(f"site in {args.out} ({len(graphs)} graph(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
