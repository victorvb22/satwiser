"""Build the per-revolution series for a satellite and store it as Parquet.

Output: ``$SATWISER_DATA_DIR/processed/revolutions_<SAT>.parquet``, one row per complete
revolution with mean a, e, i, absolute orbit number, POD manoeuvre flag (label
cross-check only) and daily F10.7 / Ap.

    python scripts/build_series.py --satellite S1A
"""

from __future__ import annotations

import argparse
import time

from satwiser.collect import spaceweather
from satwiser.config import processed_dir, raw_dir
from satwiser.pipeline import series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--satellite", default="S1A", choices=["S1A", "S1B", "S1C"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    t0 = time.time()
    revs = series.build(raw_dir("poeorb", args.satellite), args.satellite, args.workers)
    sw = spaceweather.parse(raw_dir("spaceweather") / spaceweather.FILENAME)
    revs = series.add_space_weather(revs, sw)
    out = processed_dir() / f"revolutions_{args.satellite}.parquet"
    revs.to_parquet(out)
    print(f"{len(revs)} revolutions, {revs.index[0]} to {revs.index[-1]} "
          f"({time.time() - t0:.0f} s) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
