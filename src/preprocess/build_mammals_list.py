import requests, time, csv, sys
from pathlib import Path
from tqdm import tqdm

OUT = Path("data/gazetteers/mammals.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

"""
Pages through GBIF’s species API for accepted species under Mammalia (classKey=359).
"""

BASE = "https://api.gbif.org/v1"
HEADERS = {"User-Agent": "biodiv-ner/0.1"}


def fetch_species(offset=0, limit=300):
    params = {
        "highertaxon_key": 359,   # Mammalia
        "rank": "SPECIES",
        "status": "ACCEPTED",
        "limit": limit,
        "offset": offset
    }
    r = requests.get(f"{BASE}/species/search", params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_vernaculars(key):
    r = requests.get(f"{BASE}/species/{key}/vernacularNames", headers=HEADERS, timeout=60)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    data = r.json()
    # return only names marked as 'vernacularName'
    return [v.get("vernacularName","").strip() for v in data.get('results', []) if v.get("vernacularName")]


def main(max_records=50000, sleep=0.2):
    seen = set()
    rows = 0
    offset = 0
    pbar = tqdm(total=max_records, desc="GBIF mammals", unit="sp")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name","type","uri","source"])

        while True:
            try:
                page = fetch_species(offset=offset)
                results = page.get("results", [])
                if not results:
                    break
                for sp in results:
                    key = sp.get("key")
                    if not key:
                        continue
                    sci = (sp.get("scientificName") or "").strip()
                    if not sci:
                        continue
                    uri = f"https://www.gbif.org/species/{key}"

                    # scientific name
                    if sci.lower() not in seen:
                        rows += 1
                        w.writerow([sci, "MAMMAL", uri, "GBIF"])
                        seen.add(sci.lower())

                    # vernaculars (optional, can be many)
                    for name in fetch_vernaculars(key):
                        n = name.strip()
                        if n and n.lower() not in seen:
                            rows += 1
                            w.writerow([n, "MAMMAL", uri, "GBIF"])
                            seen.add(n.lower())

                    pbar.update(1)
                    if pbar.n >= max_records:
                        break

                if pbar.n >= max_records:
                    break
                offset += page.get("limit", 300)
                time.sleep(sleep)
            except Exception as ex:
                print(f'Error {ex}')
        pbar.close()


    print(f"✔ Wrote {len(rows)} names to {OUT}")


if __name__ == "__main__":
    main(max_records=60000, sleep=0.15)
