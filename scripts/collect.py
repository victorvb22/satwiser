"""Download external inputs into ``$SATWISER_DATA_DIR/raw``.

Examples::

    python scripts/collect.py esa
    python scripts/collect.py spaceweather
    python scripts/collect.py poeorb --satellite S1A --start 2023-03-01 --end 2023-04-01
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd
import requests

from satwiser.collect import esa, poeorb, spaceweather
from satwiser.config import raw_dir

try:  # Use the OS certificate store when available (corporate / antivirus TLS proxies).
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


def _poeorb(args: argparse.Namespace) -> None:
    index_path = raw_dir("poeorb") / f"index_{args.satellite}.csv"
    if index_path.exists() and not args.refresh_index:
        index = pd.read_csv(index_path, parse_dates=["created", "validity_start", "validity_stop"])
        refs = [poeorb.OrbitFileRef(**row) for row in index.to_dict("records")]
    else:
        with requests.Session() as session:
            refs = poeorb.list_all(args.satellite, session)
        pd.DataFrame([r.__dict__ for r in refs]).to_csv(index_path, index=False)
        print(f"Indexed {len(refs)} files for {args.satellite} -> {index_path}")
    selected = poeorb.select_window(refs, args.start, args.end)
    total_mb = sum(r.size for r in selected) / 1e6
    print(f"{len(selected)} files needed for {args.start}..{args.end} ({total_mb:.0f} MB)")
    poeorb.download(selected, raw_dir("poeorb", args.satellite))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)
    sub.add_parser("esa", help="Manoeuvre, mass and outage history (SentiWiki)")
    sub.add_parser("spaceweather", help="Daily F10.7 and Ap (GFZ Potsdam)")
    p = sub.add_parser("poeorb", help="Precise orbit files (AWS open data)")
    p.add_argument("--satellite", default="S1A", choices=["S1A", "S1B", "S1C"])
    p.add_argument("--start", type=date.fromisoformat, required=True, help="first UTC day")
    p.add_argument("--end", type=date.fromisoformat, required=True, help="exclusive end day")
    p.add_argument("--refresh-index", action="store_true", help="re-list the bucket")
    args = parser.parse_args()

    if args.source == "esa":
        for path in esa.download(raw_dir("esa")):
            print(path)
    elif args.source == "spaceweather":
        print(spaceweather.download(raw_dir("spaceweather")))
    else:
        _poeorb(args)


if __name__ == "__main__":
    main()
