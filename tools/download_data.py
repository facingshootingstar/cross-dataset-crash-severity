"""Downloads the public raw files into data/raw and records size and SHA-256 in data/raw/MANIFEST.tsv."""

import hashlib
import json
import os
import re
import subprocess
import urllib.request

from crashtransfer import paths

BAAC_API = "https://www.data.gouv.fr/api/1/datasets/53698f4ca3a729239d2036df/"
VIC = "https://opendata.transport.vic.gov.au/dataset/bb77800e-1857-4edc-bf9e-e188437a1c8e/resource/"
VIC_FILES = {
    "Accident": "20772c1a-8b19-424a-a733-eb84f725f611/download/accident.csv",
    "Vehicle": "6d0b21f7-583a-4991-a168-f15a70c13ec4/download/vehicle.csv",
    "Person": "60c8fc0c-2806-40f3-bb33-5c52691120e8/download/person.csv",
    "Node": "466fd3b5-201b-42b5-b10d-e926324fa215/download/node.csv",
    "Atmospheric_Condition": "27986d89-f1fd-4f35-880a-d5be0deaa741/download/atmospheric_cond.csv",
}
TABLES = [
    ("usagers", "usagers"),
    ("lieux", "lieux"),
    ("vehicules", "vehicules"),
    ("caract", "caract"),
    ("carcteristiques", "caract"),
]


def jobs():
    out = []
    for y in range(2016, 2025):
        out.append(
            (
                f"https://static.nhtsa.gov/nhtsa/downloads/CRSS/{y}/CRSS{y}CSV.zip",
                os.path.join("crss", f"CRSS{y}CSV.zip"),
            )
        )
    for t in ("collision", "vehicle", "casualty"):
        n = f"dft-road-casualty-statistics-{t}-1979-latest-published-year.csv"
        out.append((f"https://data.dft.gov.uk/road-accidents-safety-data/{n}", os.path.join("stats19", "bulk", n)))
    seen = set()
    for r in json.load(urllib.request.urlopen(BAAC_API))["resources"]:
        title = (r.get("title") or "").lower()
        y = re.search(r"20(1[6-9]|2[0-4])", title)
        if not y or not title.endswith(".csv") or "immatricul" in title:
            continue
        table = next(v for k, v in TABLES if title.startswith(k))
        assert (table, y.group(0)) not in seen, title
        seen.add((table, y.group(0)))
        out.append((r["url"], os.path.join("baac", f"{table}_{y.group(0)}.csv")))
    assert len(seen) == 36, len(seen)
    for name, res in VIC_FILES.items():
        out.append((VIC + res, os.path.join("vic", f"{name}.csv")))
    return out


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


rows = []
for url, rel in jobs():
    path = os.path.join(paths.RAW, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        subprocess.run(
            ["curl", "-sSL", "--fail", "--retry", "5", "--retry-delay", "10", "-o", path + ".part", url],
            check=True,
        )
        os.replace(path + ".part", path)
    rows.append((rel, os.path.getsize(path), sha256(path), url))
    print(rel, rows[-1][1], flush=True)
with open(os.path.join(paths.RAW, "MANIFEST.tsv"), "w", encoding="utf8") as f:
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
