"""
Download authority snapshots. Build-machine only — never called at match time.

Two things this is careful about
--------------------------------
**Atomicity.** Downloads land on a ``.part`` file and are renamed only once the
stream completes. A half-written ENVO that looks like a finished one is worse
than no ENVO: it parses, it yields a smaller vocabulary, and the drop in link
rate looks like a finding rather than a truncated file.

**Size.** GBIF's backbone and GeoNames' allCountries are each over a gigabyte.
Those are gated behind an explicit ``--yes-large`` so nobody starts one by
running the obvious command on a laptop tether.

Usage
-----
    python -m src.linking.authorities.fetch --small          # envo, pato, go_bp
    python -m src.linking.authorities.fetch --key gbif --yes-large
    python -m src.linking.authorities.fetch --list
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import requests

from src.linking.authorities.registry import (
    AUTHORITIES, LARGE_KEYS, SMALL_KEYS, SNAPSHOT_DIR, Authority,
    file_sha256, load_manifest, record_snapshot, update_manifest,
)

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20      # 1 MiB
_TIMEOUT = (30, 300)  # (connect, read); OBO PURLs redirect and can be slow to start


def _download(auth: Authority, dest: Path) -> None:
    """Stream one authority to ``dest`` atomically."""
    part = dest.with_suffix(dest.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(auth.url, stream=True, timeout=_TIMEOUT,
                      headers={"User-Agent": "mobiko-nlp/linking"}) as resp:
        resp.raise_for_status()
        # Content-Length describes the bytes ON THE WIRE. When the server gzips
        # (the OBO PURLs do), iter_content hands back DECOMPRESSED bytes, so
        # comparing the two reports percentages over 100. Only show a percentage
        # when the response was not transfer-encoded; otherwise just count MB.
        encoded = bool(resp.headers.get("Content-Encoding"))
        total = 0 if encoded else int(resp.headers.get("Content-Length") or 0)
        seen = 0
        next_mark = 0
        with part.open("wb") as fh:
            for block in resp.iter_content(_CHUNK):
                if not block:
                    continue
                fh.write(block)
                seen += len(block)
                if total:
                    pct = int(100 * seen / total)
                    if pct >= next_mark:
                        next_mark = pct + 10
                        print(f"    {pct:3d}%  {seen/1e6:,.0f} / {total/1e6:,.0f} MB",
                              file=sys.stderr)
                elif seen >= next_mark:
                    next_mark = seen + (25 << 20)
                    print(f"    {seen/1e6:,.0f} MB", file=sys.stderr)
    part.replace(dest)


def fetch(auth: Authority, force: bool = False) -> bool:
    """Fetch one authority. Returns True if bytes were downloaded.

    An existing snapshot whose sha256 still matches the manifest is left alone —
    re-downloading GBIF because a caller listed it twice costs an hour and can
    only change the result for the worse.
    """
    recorded = (load_manifest().get(auth.key) or {}).get("sha256")
    if auth.path.exists() and not force:
        if recorded and file_sha256(auth.path) == recorded:
            print(f"  {auth.key:10s} already present and hash matches manifest — skipping")
            return False
        print(f"  {auth.key:10s} present but unverified — re-fetching")

    print(f"  {auth.key:10s} {auth.url}  ({auth.size_hint})")
    _download(auth, auth.path)

    rec = record_snapshot(auth)
    update_manifest(rec)
    print(f"  {auth.key:10s} done — {rec.bytes/1e6:,.1f} MB, "
          f"version={rec.payload_version or rec.pinned_version}, sha256={(rec.sha256 or '')[:12]}")
    return True


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", action="append", dest="keys",
                    help="authority key to fetch; repeatable")
    ap.add_argument("--small", action="store_true",
                    help=f"fetch the small ontologies ({', '.join(SMALL_KEYS)})")
    ap.add_argument("--all", action="store_true", help="fetch everything")
    ap.add_argument("--yes-large", action="store_true",
                    help="permit the >1GB downloads (gbif, geonames)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even when the local hash matches the manifest")
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list:
        manifest = load_manifest()
        print(f"{'key':10s} {'size':32s} {'local':7s} {'version':30s} serves")
        for a in AUTHORITIES.values():
            m = manifest.get(a.key) or {}
            print(f"{a.key:10s} {a.size_hint:32s} "
                  f"{'yes' if a.path.exists() else 'no':7s} "
                  f"{str(m.get('payload_version') or m.get('pinned_version') or '-'):30s} {len(a.serves)} types")
        return 0

    keys = list(args.keys or [])
    if args.small:
        keys += list(SMALL_KEYS)
    if args.all:
        keys += list(AUTHORITIES)
    if not keys:
        ap.error("nothing to do: pass --key, --small, --all or --list")

    seen, ordered = set(), []
    for k in keys:
        if k not in AUTHORITIES:
            ap.error(f"unknown authority {k!r}; known: {', '.join(AUTHORITIES)}")
        if k not in seen:
            seen.add(k)
            ordered.append(k)

    blocked = [k for k in ordered if k in LARGE_KEYS and not args.yes_large]
    if blocked:
        print(f"refusing to fetch {', '.join(blocked)} without --yes-large "
              f"(each is over a gigabyte)", file=sys.stderr)
        ordered = [k for k in ordered if k not in blocked]
        if not ordered:
            return 2

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for k in ordered:
        try:
            fetch(AUTHORITIES[k], force=args.force)
        except Exception as exc:
            print(f"  {k:10s} FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
