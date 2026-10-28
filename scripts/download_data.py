"""Download the datasets used by the project from PINNED upstream sources and verify their SHA-256.

Run from the project root:   python scripts/download_data.py
Files go to data/raw/.  Sources: pygod-team/data at commit 13e75cd1 (weibo, reddit) and Planetoid Cora via PyG.

Why pinned: the upstream `main` branch changed weibo.pt on 2024-01-19 (commit 84b70591 "add mask"): the labels of every
node in the validation and test masks were set to 0, so only 347 of the 868 benchmark anomalies stay labelled.
That file is NOT valid for evaluation; this project uses the original 2022 file (868 anomalies, as in the BOND paper).
"""
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
COMMIT = "13e75cd16e5b674ac84fd6adfea8286ef0b69203"
BASE = f"https://github.com/pygod-team/data/raw/{COMMIT}/"
PINNED = {  # SHA-256 of the zip files at the pinned commit
    "weibo": "e3d72d92f01cc67440b5ae15fff56cd7bd7e2b6dcd25af7d7c7f86b710879b57",
    "reddit": "276432d36219349b7ad98e974649f73c94e124e11f5282f53b1ec41086c1e18a",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    ck = {}
    ok = True
    for n, expected in PINNED.items():
        z = RAW / f"{n}.pt.zip"
        pt = RAW / f"{n}.pt"
        if z.exists() and sha(z) != expected:
            print(f"{n}: existing zip does not match the pinned checksum; removing it and downloading again")
            z.unlink()
            pt.unlink(missing_ok=True)
        if not z.exists():
            print(f"downloading {n} from the pinned commit ...")
            try:
                urllib.request.urlretrieve(BASE + f"{n}.pt.zip", z)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED to download {n}: {e}\nManual fix: open {BASE}{n}.pt.zip in a browser, save it as {z} "
                      f"and re-run this script.")
                ok = False
                continue
        digest = sha(z)
        if digest != expected:
            print(f"{n}: CHECKSUM MISMATCH ({digest[:16]}... != pinned {expected[:16]}...). Deleting the file.")
            z.unlink()
            ok = False
            continue
        if not pt.exists():
            shutil.unpack_archive(z, RAW)
        ck[f"{n}.pt.zip"] = {"sha256": digest, "bytes": z.stat().st_size, "url": BASE + f"{n}.pt.zip",
                             "pinned_commit": COMMIT, "pt_sha256": sha(pt), "status": "matches pinned checksum"}
        print(f"{n}: sha256={digest[:16]}... [matches pinned checksum]")
    sys.path.insert(0, str(ROOT / "src"))
    from attngnn.data import load_cora_clean
    x, ei, y = load_cora_clean(RAW)
    print("cora (Planetoid):", x.shape, ei.shape)
    for p in (RAW / "planetoid" / "Cora" / "raw").glob("*"):
        ck[f"planetoid/Cora/raw/{p.name}"] = {"sha256": sha(p), "bytes": p.stat().st_size}
    (RAW / "checksums.json").write_text(json.dumps(ck, indent=2))
    print("checksums written to", RAW / "checksums.json")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
