"""Build the project report (Markdown + DOCX + PDF) from one metadata file, one bibliography and the
measured results.

    python scripts/build_report.py

Inputs : metadata.yaml, docs/literature/references.bib, docs/report_src/report.md,
         artifacts/results/{tables,figures,facts.json}, docs/figures/*.png, docs/screenshots/*.png
Outputs: reports/Attention_GNN_Report.{md,docx,pdf}
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import yaml
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

import docbuild as DB

RES = ROOT / "artifacts" / "results"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


def load_facts():
    p = RES / "facts.json"
    return json.loads(p.read_text()) if p.exists() else {}


def resolve(text, facts, cites):
    def inc(m):
        return (ROOT / "docs" / "report_src" / m.group(1)).read_text(encoding="utf-8").strip()

    text = re.sub(r"\{\{include:([^}]+)\}\}", inc, text)

    def tab(m):
        name, *opts = m.group(1).split("|")
        cap = opts[0] if opts else ""
        compact = "full" not in opts[1:]
        p = RES / "tables" / name
        if not p.exists():
            return f"**[MISSING TABLE {name}]**"
        rows = DB.csv_to_rows(p, compact=compact)
        return f"Table: {cap}\n{DB.rows_to_md(rows)}"

    text = re.sub(r"\{\{table:([^}]+)\}\}", tab, text)

    def fig(m):
        name, cap = (m.group(1).split("|") + [""])[:2]
        for base in (RES / "figures", ROOT / "docs" / "figures", ROOT / "docs" / "screenshots"):
            if (base / name).exists():
                return f"![{cap}]({(base / name).as_posix()})"
        return f"**[MISSING FIGURE {name}]**"

    text = re.sub(r"\{\{fig:([^}]+)\}\}", fig, text)

    def fact(m):
        k = m.group(1)
        if k not in facts:
            raise KeyError(f"fact {k!r} not in facts.json")
        return str(facts[k])

    text = re.sub(r"\{\{fact:([^}]+)\}\}", fact, text)
    return cites.sub(text)


def build():
    meta = yaml.safe_load((ROOT / "metadata.yaml").read_text(encoding="utf-8"))
    bib = DB.parse_bib(ROOT / "docs" / "literature" / "references.bib")
    cites = DB.Citations(bib)
    facts = load_facts()
    parts = [(ROOT / "docs" / "report_src" / n).read_text(encoding="utf-8") for n in ("part1.md", "part2.md", "part3.md")]
    src = "\n\n".join(parts)
    body = resolve(src, facts, cites)
    refs = cites.entries()

    # ---------------- markdown output
    md = [f"# {meta['title']}", ""]
    for a in meta["authors"]:
        md.append(f"**{a['name']}** ({a['reg_no']}), {a['email']}  ")
    md += [f"{meta['institution']}  ", meta["course"], ""]
    if facts.get("incomplete_banner"):
        md += [f"> **{facts['incomplete_banner']}**", ""]
    md.append(body)
    md += ["", "### References listed in order of first citation", ""]
    for n, k, f in refs:
        md.append(f"[{n}] {f}")
    md += ["", f"*AI-assistance disclosure:* {meta['ai_assistance_disclosure'].strip()}"]
    md_text = "\n".join(md)
    # build the numbered reference block inside Section 8
    (OUT / "Attention_GNN_Report.md").write_text(md_text, encoding="utf-8")

    # ---------------- docx
    doc = DB.new_document(meta["short_title"])
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(meta["title"])
    r.bold = True
    r.font.size = Pt(19)
    r.font.name = "Calibri"
    r.font.color.rgb = DB.NAVY
    for a in meta["authors"]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        DB.add_inline(p, f"**{a['name']}** ({a['reg_no']}), {a['email']}", size=10.5)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    DB.add_inline(p, f"{meta['institution']}. {meta['course']}", size=9.5, italic=True)
    if facts.get("incomplete_banner"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        DB.add_inline(p, f"**{facts['incomplete_banner']}**", size=9.5, color=DB.TEAL)

    fig_n = tab_n = eq_n = 0
    lines = body.splitlines()
    i = 0
    eqdir = ROOT / "artifacts" / "eq"
    eqdir.mkdir(parents=True, exist_ok=True)
    pending_caption = None
    para_buf = []

    def flush():
        nonlocal para_buf
        if para_buf:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            DB.add_inline(p, " ".join(para_buf))
            para_buf = []

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("<!-- pagebreak"):
            flush()
            doc.add_page_break()
        elif s.startswith("### "):
            flush()
            doc.add_heading(s[4:], level=3)
        elif s.startswith("## "):
            flush()
            doc.add_heading(s[3:], level=1)
        elif s.startswith("# "):
            flush()
            doc.add_heading(s[2:], level=1)
        elif s.startswith("Table:"):
            flush()
            pending_caption = s[6:].strip()
        elif s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            i -= 1
            tab_n += 1
            ncol = len(rows[0])
            font = 8 if ncol <= 5 else (7.2 if ncol <= 7 else 6.6)
            DB.add_table(doc, rows, pending_caption, font=font, number=tab_n)
            pending_caption = None
        elif s.startswith("!["):
            flush()
            m = re.match(r"!\[(.*)\]\((.*)\)", s)
            fig_n += 1
            DB.add_figure(doc, m.group(2), m.group(1), fig_n, width=6.3)
        elif s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            flush()
            eq_n += 1
            img = eqdir / f"eq{eq_n}.png"
            DB.render_equation(s[2:-2].strip(), img)
            DB.add_equation(doc, img, eq_n)
        elif re.match(r"^- ", s):
            flush()
            p = doc.add_paragraph(style="List Bullet")
            DB.add_inline(p, s[2:])
        elif re.match(r"^\d+\. ", s):
            flush()
            p = doc.add_paragraph(style="List Number")
            DB.add_inline(p, re.sub(r"^\d+\. ", "", s))
        elif s == "":
            flush()
        else:
            para_buf.append(s)
        i += 1
    flush()
    # references (numbered) after the Section 8 heading content in the source: appended here
    for n, k, f in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(24)
        p.paragraph_format.first_line_indent = Pt(-24)
        p.paragraph_format.space_after = Pt(3)
        DB.add_inline(p, f"[{n}]  {f}", size=9)
    p = doc.add_paragraph()
    DB.add_inline(p, f"*AI-assistance disclosure:* {meta['ai_assistance_disclosure'].strip()}", size=9)
    docx_path = OUT / "Attention_GNN_Report.docx"
    doc.save(docx_path)
    pages = DB.docx_to_pdf(docx_path, OUT / "Attention_GNN_Report.pdf")
    print(f"report built: {docx_path.name}, PDF {pages} pages, {len(refs)} references, "
          f"{fig_n} figures, {tab_n} tables, {eq_n} equations")
    # citation resolution check
    missing = [k for n, k, f in refs if not f.strip()]
    assert not missing, missing


if __name__ == "__main__":
    build()
