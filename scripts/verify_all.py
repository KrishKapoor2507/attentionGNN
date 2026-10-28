"""Final verification: checkpoints reproduce inference, tables rebuild, PPTX/PDF/DOCX are valid, citations resolve.

    python scripts/verify_all.py            -> prints a checklist and writes artifacts/results/verification_log.txt
(The unit tests are run separately with: python -m pytest)
"""
import json
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

lines = []


def say(msg, ok=None):
    tag = "" if ok is None else ("[PASS] " if ok else "[FAIL] ")
    line = tag + msg
    print(line, flush=True)
    lines.append(line)
    return ok is not False


def main():
    from attngnn import RUNS_DIR
    from attngnn import inference as inf

    allok = True
    say(f"verification started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    runs = [d for d in inf.list_runs(RUNS_DIR) if (d / "checkpoint.pt").exists()]
    bad = [d.name for d in runs if not inf.verify_checkpoint_hash(d)]
    allok &= say(f"{len(runs)} checkpoints: SHA-256 equals the hash stored in metrics.json (mismatches: {bad})", not bad)

    bench = [d for d in runs if inf.load_metrics(d).get("tag") == "bench"]
    sample = []
    seen = set()
    for d in bench:                       # one run per dataset x variant (seed 0 where available)
        m = inf.load_metrics(d)
        key = (m["dataset"], m["variant"])
        if key not in seen and m["seed"] == 0:
            seen.add(key)
            sample.append(d)
    worst_all = 0.0
    fails = []
    for d in sample:
        r = inf.reproduce(d)
        worst = max(r["max_abs_diff"].values()) if r["max_abs_diff"] else 0.0
        worst_all = max(worst_all, worst)
        if not r["ok"]:
            fails.append(d.name)
    allok &= say(f"{len(sample)} saved checkpoints (every dataset x GNN variant, seed 0) recompute all scores; "
                 f"max |difference| to saved predictions = {worst_all:.2e}; failures: {fails}", not fails)

    # results rebuild consistency
    idx = json.loads((ROOT / "artifacts" / "results" / "results_index.json").read_text())
    missing = [(n, r) for n, i in {**idx["tables"], **idx["figures"]}.items() for r in i["source_runs"]
               if not (RUNS_DIR / r / "metrics.json").exists()]
    allok &= say(f"results index: {len(idx['tables'])} tables and {len(idx['figures'])} figures, all source runs exist "
                 f"(missing: {missing[:3]})", not missing)

    # dataset validation + local-only server
    vj = json.loads((ROOT / "artifacts" / "results" / "dataset_validation.json").read_text(encoding="utf-8"))
    nfail = sum(r["result"] == "FAIL" for r in vj["rows"])
    npass = sum(r["result"] == "PASS" for r in vj["rows"])
    allok &= say(f"dataset validation report: {npass} checks passed, {nfail} failed (built {vj['built']})", nfail == 0)
    launcher = (ROOT / "scripts" / "run_dashboard.ps1").read_text(encoding="utf-8")
    allok &= say("local launcher binds the dashboard to localhost only (--server.address localhost)", "--server.address localhost" in launcher)
    from scripts_check import wb_labels_ok
    allok &= say("Weibo data file is the original 868-anomaly version (pinned SHA-256)", wb_labels_ok())

    # PPTX
    from pptx import Presentation
    pp = ROOT / "reports" / "Attention_GNN_Final.pptx"
    prs = Presentation(pp)
    n = len(prs.slides)
    notes_ok = all(s.has_notes_slide and len(s.notes_slide.notes_text_frame.text) > 40 for s in prs.slides)
    charts = sum(1 for s in prs.slides for sh in s.shapes if sh.has_chart)
    tables = sum(1 for s in prs.slides for sh in s.shapes if sh.has_table)
    allok &= say(f"PPTX opens with python-pptx: {n} slides (required 14), speaker notes on every slide: {notes_ok}, "
                 f"{charts} native charts, {tables} native tables", n == 14 and notes_ok)
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        app = win32com.client.DispatchEx("PowerPoint.Application")
        pres = app.Presentations.Open(str(pp.resolve()), ReadOnly=True, Untitled=False, WithWindow=False)
        cnt = pres.Slides.Count
        pres.Close()
        app.Quit()
        allok &= say(f"PPTX opens in Microsoft PowerPoint with {cnt} slides", cnt == 14)
    except Exception as e:  # noqa: BLE001
        say(f"PowerPoint check skipped: {e}")
    text = " ".join(sh.text_frame.text for s in prs.slides for sh in s.shapes if sh.has_text_frame)
    ph = re.findall(r"lorem|ipsum|TODO|\[missing|MISSING|xxxx", text, flags=re.I)
    allok &= say(f"no placeholder text on slides (found: {ph})", not ph)

    # report
    import fitz
    pdf_path = next((p for p in (ROOT / "reports" / "Attention_GNN_Report.pdf", ROOT / "reports" / "final_report.pdf") if p.exists()))
    say(f"checking report PDF: {pdf_path.name}")
    pdf = fitz.open(pdf_path)
    ptxt = "\n".join(p.get_text() for p in pdf)
    heads = ["i. Abstract", "ii. Keywords", "1. Introduction", "1.1 Problem Formulation / Objectives", "2. Related Work",
             "3. Proposed Work: Attention-Driven Graph Neural Networks for Influence and Anomaly Detection in Complex Networks",
             "3.1 Graph Representation and Data Preparation",
             "3.2 Attention Encoder, Joint Objectives, Influence Ranking and Anomaly Scoring", "3.3 System Architecture",
             "3.4 Functional Block Diagram", "4. Experimental Analysis and observations", "5. Experimental Validations",
             "5.1 Test set (dataset) details and performance", "5.2 Baseline Comparison", "5.3 Ablation Study", "6. Conclusion",
             "7. Future References", "8. Reference"]
    flat = re.sub(r"\s+", "", ptxt)           # whitespace-insensitive: line breaks can split hyphenated words
    pos = [flat.find(re.sub(r"\s+", "", h)) for h in heads]
    in_order = all(p >= 0 for p in pos) and pos == sorted(pos)
    allok &= say(f"report PDF: {len(pdf)} pages; all 18 required headings present in the required order: {in_order}", in_order)
    md = (ROOT / "reports" / "Attention_GNN_Report.md").read_text(encoding="utf-8")
    miss = re.findall(r"MISSING (?:TABLE|FIGURE)[^*]*", md)
    allok &= say(f"report has no missing table/figure markers (found: {miss})", not miss)
    cited = sorted({int(x) for grp in re.findall(r"\[(\d+(?:, \d+)*)\]", md) for x in grp.split(", ")})
    refs = re.findall(r"^\[(\d+)\] ", md, flags=re.M)
    unresolved = [c for c in cited if str(c) not in refs]
    allok &= say(f"report citations: {len(cited)} distinct numbers cited, {len(refs)} references listed, unresolved: {unresolved}", not unresolved)
    cc = ROOT / "docs" / "literature" / "citation_check.csv"
    if cc.exists():
        import csv
        rows = list(csv.DictReader(open(cc, encoding="utf-8")))
        badc = [r["key"] for r in rows if r["status"].startswith(("HTTP", "error", "no url"))]
        allok &= say(f"bibliography URL check ({len(rows)} entries, {sum(r['status'].startswith('blocked') for r in rows)} publisher "
                     f"pages block bots): unresolved {badc}", not badc)
    emails = re.findall(r"EDIT_ME[^\s,]*", md)
    say(f"placeholders still to edit in metadata.yaml: {sorted(set(emails))}")
    out = ROOT / "artifacts" / "results" / "verification_log.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say("ALL AUTOMATED CHECKS PASSED" if allok else "SOME CHECKS FAILED (see above)")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
