#!/usr/bin/env python3
"""
SIBiLS search + fetch utility for biodiversity PMC (BioC JSON).

This script wraps the SIBiLS API so you can:
  1) run an Elastic-style query (from a JSON file like query.json),
  2) collect document IDs, and
  3) fetch the corresponding BioC JSON documents in batches to a folder.

Designed to plug into your pipeline: point your converter at the output dir.

Usage examples
--------------
# Basic: run query.json, download up to 1,000 docs in batches of 20
python sibils_fetch.py \
  --query /mnt/data/query.json \
  --out_dir /mnt/data/sibils_raw \
  --limit 1000 --batch 20

# Restrict to a specific SIBiLS collection (default: pmc)
python sibils_fetch.py --query query.json --col pmc --out_dir data/sibils

# Safe resume (won't re-fetch ids already present in out_dir)
python sibils_fetch.py --query query.json --out_dir data/sibils --resume

Outputs
-------
- out_dir/manifest.json         : summary (timestamp, query hash, ids, counts)
- out_dir/search_response.json  : raw search response from SIBiLS
- out_dir/fetch_00000.json ...  : chunked BioC JSON responses (as returned)
- out_dir/ids.txt               : one id per line (final set used for fetch)

Notes
-----
- Requires: requests (pip install requests)
- API endpoints used:
  * search: https://biodiversitypmc.sibils.org/api/search
  * fetch : https://biodiversitypmc.sibils.org/api/fetch
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

SEARCH_URL = "https://biodiversitypmc.sibils.org/api/search"
FETCH_URL  = "https://biodiversitypmc.sibils.org/api/fetch"


# Default values
DEFAULT_COLLECTION = "pmc"
DEFAULT_LIMIT = 1000
DEFAULT_BATCH_SIZE = 20
DEFAULT_TIMEOUT = 120
DEFAULT_SEARCH_TIMEOUT = 60
DEFAULT_RETRIES = 3


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def save_json(path: Path, data: Any) -> None:
    """Save data to JSON file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def do_search(query_path: Path, collection: str, max_hits: int,
              timeout: int = DEFAULT_SEARCH_TIMEOUT) -> Dict[str, Any]:
    """
    Execute search query against SIBiLS API.

    Args:
        query_path: Path to JSON file containing Elastic query
        collection: SIBiLS collection name (e.g., 'pmc')
        max_hits: Maximum number of search results to return
        timeout: Request timeout in seconds

    Returns:
        Search response dictionary from SIBiLS API

    Raises:
        RuntimeError: If API returns success=false
        requests.RequestException: If HTTP request fails
    """
    query_str = query_path.read_text(encoding="utf-8")
    params = {
        "jq": query_str,
        "col": collection,
        "n": max_hits
    }

    response = requests.post(SEARCH_URL, data=params, timeout=timeout)
    response.raise_for_status()

    result = response.json()
    if not result.get("success"):
        raise RuntimeError(f"SIBiLS search returned success=false: {result}")

    return result


def extract_document_ids(search_response: Dict[str, Any], limit: int) -> List[str]:
    """
    Extract document IDs from search response.

    Args:
        search_response: Response dictionary from SIBiLS search API
        limit: Maximum number of IDs to return (0 for no limit)

    Returns:
        List of document IDs
    """
    hits = (search_response
            .get("elastic_output", {})
            .get("hits", {})
            .get("hits", []))

    document_ids = [hit.get("_id") for hit in hits if hit.get("_id")]

    if limit and limit > 0:
        document_ids = document_ids[:limit]

    return document_ids


def get_existing_ids(out_dir: Path) -> set[str]:
    """
    Read existing document IDs from ids.txt file.

    Args:
        out_dir: Output directory containing ids.txt

    Returns:
        Set of existing document IDs
    """
    ids_path = out_dir / "ids.txt"
    if not ids_path.exists():
        return set()
    content = ids_path.read_text(encoding="utf-8")
    return {line.strip() for line in content.splitlines() if line.strip()}


def write_document_ids(out_dir: Path, document_ids: List[str]) -> None:
    """Write document IDs to ids.txt file."""
    ids_content = "\n".join(document_ids) + "\n"
    (out_dir / "ids.txt").write_text(ids_content, encoding="utf-8")


def fetch_document_chunks(document_ids: List[str], collection: str, out_dir: Path,
                          batch_size: int, start_index: int = 0,
                          timeout: int = DEFAULT_TIMEOUT, sleep_seconds: float = 0.0,
                          max_retries: int = DEFAULT_RETRIES) -> List[Path]:
    """
    Fetch documents in batches from SIBiLS API.

    Args:
        document_ids: List of document IDs to fetch
        collection: SIBiLS collection name
        out_dir: Output directory for JSON files
        batch_size: Number of documents per batch
        start_index: Starting index for resume functionality
        timeout: Request timeout in seconds
        sleep_seconds: Sleep time between requests
        max_retries: Maximum retry attempts for failed requests

    Returns:
        List of paths to written JSON files

    Raises:
        requests.RequestException: If requests fail after all retries
    """
    written_files = []

    for i in range(start_index, len(document_ids), batch_size):
        chunk = document_ids[i:i + batch_size]
        output_path = out_dir / f"fetch_{i:05d}.json"

        # Skip if file already exists
        if output_path.exists():
            written_files.append(output_path)
            continue

        payload = {
            "ids": ",".join(chunk),
            "col": collection
        }

        # Retry logic
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Fetching {len(chunk)} ids [{i}-{i + len(chunk) - 1}] → {output_path.name}")
                response = requests.post(FETCH_URL, data=payload, timeout=timeout)
                response.raise_for_status()

                # Ensure output directory exists
                out_dir.mkdir(parents=True, exist_ok=True)

                # Write response to file
                output_path.write_text(response.text, encoding="utf-8")
                written_files.append(output_path)
                break

            except requests.RequestException as e:
                if attempt >= max_retries:
                    error_msg = f"Fetch failed after {max_retries} retries for batch starting {i}: {e}"
                    print(f"[ERROR] {error_msg}", file=sys.stderr)
                    raise

                backoff_time = min(2 ** attempt, 30)
                print(f"[WARN] Fetch error (attempt {attempt}/{max_retries}): {e}. "
                      f"Retrying in {backoff_time}s…")
                time.sleep(backoff_time)

        # Optional sleep between successful requests
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return written_files


def determine_resume_start_index(out_dir: Path, batch_size: int) -> int:
    """
    Determine starting index for resume based on existing fetch files.

    Args:
        out_dir: Output directory
        batch_size: Batch size used for fetching

    Returns:
        Starting index for resume
    """
    existing_files = sorted(out_dir.glob("fetch_*.json"))
    if not existing_files:
        return 0

    try:
        # Extract index from filename pattern fetch_00000.json
        last_file = existing_files[-1]
        last_index = int(last_file.stem.split("_")[-1])
        return last_index + batch_size
    except (ValueError, IndexError):
        return 0


def create_manifest(query_path: Path, query_hash: str, collection: str,
                    search_response: Dict[str, Any], document_ids: List[str],
                    batch_size: int, written_files: List[Path]) -> Dict[str, Any]:
    """Create manifest dictionary with run metadata."""
    search_hits = (search_response
                   .get("elastic_output", {})
                   .get("hits", {})
                   .get("hits", []))

    return {
        "query_file": str(query_path),
        "query_sha1": query_hash,
        "collection": collection,
        "search_hits": len(search_hits),
        "ids": len(document_ids),
        "batch_size": batch_size,
        "files": [str(path.name) for path in written_files],
    }




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Path to Elastic query JSON (as provided)")
    ap.add_argument("--out_dir", required=True, help="Output directory for raw BioC JSONs")
    ap.add_argument("--col", default=DEFAULT_COLLECTION, help="SIBiLS collection (default: pmc)")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max number of doc ids to fetch")
    ap.add_argument("--n", type=int, default=DEFAULT_LIMIT, help="Search API 'n' parameter (max hits returned)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Fetch batch size (ids per request)")
    ap.add_argument("--resume", action="store_true", help="Skip batches already on disk and reuse ids.txt if present")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep between fetch calls (seconds)")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load / run search
    qpath = Path(args.query)
    qtxt = qpath.read_text(encoding="utf-8")
    qhash = sha1_text(qtxt)

    # Handle search (with resume support)
    search_response_path = out_dir / "search_response.json"
    ids_path = out_dir / "ids.txt"

    if args.resume and search_response_path.exists() and ids_path.exists():
        print("[resume] Using existing search_response.json and ids.txt")
        search_resp = read_json(search_response_path)
        ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        print("Searching SIBiLS…")
        search_resp = do_search(qpath, collection=args.col, max_hits=args.n, timeout=args.timeout)
        save_json(search_response_path, search_resp)
        ids = extract_document_ids(search_resp, args.limit)
        write_document_ids(out_dir, ids)

    print(f"Found {len(ids)} ids to fetch (query sha1={qhash})")

    # Determine starting batch for resume
    start_index = 0
    if args.resume:
        start_index = determine_resume_start_index(out_dir, args.batch)
        if start_index > 0:
            print(f"[resume] Resuming fetch from index {start_index} (skipping existing files)")


    written = fetch_document_chunks(
        document_ids=ids,
        collection=args.col,
        out_dir=out_dir,
        batch_size=args.batch,
        start_index=start_index,
        timeout=args.timeout,
        sleep_seconds=args.sleep)


    manifest = create_manifest(
        query_path=qpath,
        query_hash=qhash,
        collection=args.col,
        search_response=search_resp,
        document_ids=ids,
        batch_size=args.batch,
        written_files=written
    )

    save_json(out_dir / "manifest.json", manifest)
    print(f"Done. Saved {len(written)} fetch files → {out_dir}")


if __name__ == "__main__":
    main()
