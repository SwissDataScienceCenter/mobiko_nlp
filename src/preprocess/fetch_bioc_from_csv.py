import argparse, asyncio, csv, hashlib, re, sys
from pathlib import Path

import pandas as pd
import httpx
from tqdm import tqdm

API_BASE = "https://biodiversitypmc.sibils.org/api/fetch"
DEFAULT_PARAMS = {"col": "pmc", "format": "bioc"}
PMC_RE = re.compile(r"PMC\d+")


def build_url(val: str) -> str:
    """
    Normalizer to make sure that URL is valid
    """
    val = (val or "").strip()
    if not val:
        return ""
    if val.startswith("http://") or val.startswith("https://"):
        return val
    m = PMC_RE.search(val)
    if m:
        pmc = m.group(0)
        return f"{API_BASE}?ids={pmc}&col={DEFAULT_PARAMS['col']}&format={DEFAULT_PARAMS['format']}"
    return f"{API_BASE}?ids={val}&col={DEFAULT_PARAMS['col']}&format={DEFAULT_PARAMS['format']}"


def safe_name(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    m = PMC_RE.search(url)
    prefix = m.group(0) if m else "doc"
    return f"{prefix}_{h}.xml"


async def fetch_one(client, url: str, outdir: Path, sem: asyncio.Semaphore, timeout=30.0):
    async with sem:
        try:
            r = await client.get(url, timeout=timeout)
            if r.status_code == 404:
                return ("404", url, None, "not found")
            if r.status_code == 429:
                await asyncio.sleep(3)
                r = await client.get(url, timeout=timeout)
            r.raise_for_status()
            content_type = r.headers.get("content-type", "").lower()
            ext = ".json" if "json" in content_type else ".xml"
            name = safe_name(url)
            if ext == ".json" and name.endswith(".xml"):
                name = name[:-4] + ".json"
            path = outdir / name
            path.write_bytes(r.content)
            return ("OK", url, str(path), content_type)
        except Exception as e:
            return ("ERR", url, None, repr(e))


async def main_async(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    if args.column not in df.columns:
        print(f"Column '{args.column}' not in CSV", file=sys.stderr)
        sys.exit(2)
    urls = []
    for v in df[args.column].astype(str).tolist():
        url = build_url(v)
        if url:
            urls.append(url)
    if not urls:
        print("No URLs built from the provided column.", file=sys.stderr)
        sys.exit(1)

    sem = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(60.0, connect=30.0)
    limits = httpx.Limits(max_keepalive_connections=args.concurrency, max_connections=args.concurrency)
    headers = {"User-Agent": "biodiv-ner-downloader/0.1 (+research use)"}
    results = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits, headers=headers) as client:
        tasks = [fetch_one(client, u, outdir, sem) for u in urls]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading"):
            results.append(await coro)

    manifest = outdir / "manifest.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["status", "url", "path", "content_type"])
        w.writerows(results)
    print(f"Wrote manifest with {len(results)} rows to {manifest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to .csv file containing a 'BioC' column")
    ap.add_argument("--column", default="BioC")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8, help="Number of parallel requests")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
