"""Download the Sentinel-1 manoeuvre, mass and outage histories published on SentiWiki.

Source page: https://sentiwiki.copernicus.eu/web/s1-processing (section on orbit
determination support files). The attachment URLs are content-addressed, so they change
whenever ESA republishes a file; update ``FILES`` when that happens.
"""

from __future__ import annotations

from pathlib import Path

import requests

SOURCE_PAGE = "https://sentiwiki.copernicus.eu/web/s1-processing"
_BASE = "https://sentiwiki.copernicus.eu/__attachments"

FILES: dict[str, str] = {
    "s1a.man": "a_d7549ab904a85c5ff63e89365addda4944068c44e2d04c87e64e540d6f836bae",
    "s1a.mhf": "a_9e96338013b073ceff8912fe6d368a6194b67c3578744540c6a8af776af1ddcb",
    "s1a.out": "a_fd07c8dad7248eeab4c61fd3a2f92470f30e2b724fc46bef194b7c5993271ab9",
    "s1c.man": "a_e7900cb2038da4270115f6d1fc5e6eb3b93c74d725303aa667e7ce9481eb3bd4",
    "s1c.mhf": "a_30ecd52a6795fc9509a5cd10f13a9bd58f2ff3117f3bec254597207ce8f285e9",
    "s1c.out": "a_9943372791fadd8fda353b29a157e0bd4c0e27c56c7d661004662d6a3a554a58",
}


def download(dest: Path, overwrite: bool = False) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    with requests.Session() as session:
        for name, attachment in FILES.items():
            path = dest / name
            if overwrite or not path.exists():
                response = session.get(f"{_BASE}/{attachment}/{name}", timeout=60)
                response.raise_for_status()
                path.write_bytes(response.content)
            paths.append(path)
    return paths
