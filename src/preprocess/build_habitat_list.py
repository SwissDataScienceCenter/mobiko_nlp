import argparse, csv, networkx as nx, obonet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo", help="Path to OntoBiotope .obo file", default='data/OntoBiotope-oct_2021.obo')
    ap.add_argument("--out", default="data/gazetteers/habitats.csv", help="Output CSV")
    ap.add_argument("--min_tokens", type=int, default=1, help="Keep names with at least this many tokens")
    args = ap.parse_args()

    print(f"Reading OBO: {args.obo}")
    G = obonet.read_obo(args.obo)  # NetworkX MultiDiGraph

    rows = []
    seen = set()

    for node_id, data in G.nodes(data=True):
        if data.get("obsolete") == "true":
            continue

        iri = f"http://purl.obolibrary.org/obo/{node_id.replace(':','_')}"
        # 1) primary label
        name = (data.get("name") or "").strip()
        if name and len(name.split()) >= args.min_tokens:
            key = (name.lower(), iri)
            if key not in seen:
                seen.add(key)
                rows.append([name, "HABITAT", iri, "OntoBiotope"])

        # 2) synonyms (obo-format: list of strings like: "syn text" EXACT [] {source:...})
        for syn in data.get("synonym", []) or []:
            # syn lines look like:  '"alpine meadow" EXACT []'
            s = syn.strip().strip('"')
            # If a scope is present (EXACT, RELATED, etc.), just take the leading quoted part
            if '"' in syn:
                try:
                    s = syn.split('"', 2)[1].strip()
                except Exception:
                    pass
            if s and len(s.split()) >= args.min_tokens:
                key = (s.lower(), iri)
                if key not in seen:
                    seen.add(key)
                    rows.append([s, "HABITAT", iri, "OntoBiotope"])

    rows.sort(key=lambda r: (r[0].lower(), r[2]))

    # write CSV
    out = args.out
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "type", "uri", "source"])
        w.writerows(rows)

    print(f"✔ Wrote {len(rows)} habitat names/synonyms to {out}")

if __name__ == "__main__":
    main()
