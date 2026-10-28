"""Check that every bibliography entry's DOI/arXiv/URL resolves (HTTP status).  Writes docs/literature/citation_check.csv.
A 403/429 from a publisher usually means 'bot blocked', not 'broken'; those are marked 'blocked (bot)'."""
import csv
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docbuild as DB

bib = DB.parse_bib(ROOT / "docs" / "literature" / "references.bib")
rows = []
for key, e in bib.items():
    url = (f"https://doi.org/{e['doi']}" if e.get("doi") else
           f"https://arxiv.org/abs/{e['eprint']}" if e.get("eprint") else e.get("url"))
    status = "no url"
    if url:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (citation-check)"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                status = f"ok ({r.status})"
        except urllib.error.HTTPError as ex:
            status = "blocked (bot)" if ex.code in (401, 403, 429) else f"HTTP {ex.code}"
        except Exception as ex:  # noqa: BLE001
            status = f"error: {type(ex).__name__}"
    rows.append({"key": key, "url": url, "status": status})
    print(f"{key:28s} {status}")
out = ROOT / "docs" / "literature" / "citation_check.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["key", "url", "status"])
    w.writeheader()
    w.writerows(rows)
bad = [r for r in rows if r["status"].startswith(("HTTP", "error", "no url"))]
print(f"\n{len(rows)} entries, {len(bad)} unresolved -> {out}")
