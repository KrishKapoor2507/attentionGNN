"""Render a PPTX (via PowerPoint) or a PDF (via PyMuPDF) to PNG pages for visual inspection.

    python scripts/render_office.py reports/Attention_GNN_Final.pptx artifacts/render/slides
    python scripts/render_office.py reports/Attention_GNN_Report.pdf artifacts/render/report
"""
import sys
from pathlib import Path


def render_pptx(pptx, outdir, width=1600):
    import pythoncom
    import win32com.client

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("*.png"):
        f.unlink()
    pythoncom.CoInitialize()
    app = win32com.client.DispatchEx("PowerPoint.Application")
    try:
        pres = app.Presentations.Open(str(Path(pptx).resolve()), ReadOnly=True, Untitled=False, WithWindow=False)
        n = pres.Slides.Count
        for i in range(1, n + 1):
            pres.Slides(i).Export(str((outdir / f"slide_{i:02d}.png").resolve()), "PNG", width, int(width * 9 / 16))
        pres.Close()
    finally:
        app.Quit()
    return n


def render_pdf(pdf, outdir, zoom=1.4):
    import fitz

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("*.png"):
        f.unlink()
    doc = fitz.open(pdf)
    for i, page in enumerate(doc, 1):
        page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).save(str(outdir / f"page_{i:02d}.png"))
    return len(doc)


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    n = render_pptx(src, out) if src.lower().endswith((".pptx", ".ppt")) else render_pdf(src, out)
    print(f"rendered {n} pages to {out}")
