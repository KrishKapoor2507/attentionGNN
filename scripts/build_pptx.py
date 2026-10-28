"""Generate the 14-slide final presentation (editable PPTX, native charts/tables) from measured results.

    python scripts/build_pptx.py          -> reports/Attention_GNN_Final.pptx

Every number on the slides comes from artifacts/results/facts.json or the results tables.
Speaker notes hold sources and the demo sequence.  Re-running regenerates the deck from saved runs.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import yaml
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

import docbuild as DB

RES = ROOT / "artifacts" / "results"
SHOT = ROOT / "docs" / "screenshots"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
FACTS = json.loads((RES / "facts.json").read_text())
META = yaml.safe_load((ROOT / "metadata.yaml").read_text(encoding="utf-8"))

NAVY, TEAL, ORANGE = RGBColor(0x0B, 0x25, 0x45), RGBColor(0x13, 0x80, 0x8B), RGBColor(0xD9, 0x82, 0x2B)
OFF, PALE, INK, MUTED, WHITE = (RGBColor(0xF7, 0xF9, 0xFB), RGBColor(0xE6, 0xEE, 0xF3), RGBColor(0x1B, 0x26, 0x34),
                                RGBColor(0x5B, 0x68, 0x78), RGBColor(255, 255, 255))
SAND = RGBColor(0xE9, 0xC4, 0x6A)
GREY = RGBColor(0x8D, 0x99, 0xAE)
HEAD, BODY = "Cambria", "Calibri"
W, H = 13.333, 7.5

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
slide_no = [0]


# ------------------------------------------------------------------------------------------ facts
def pm(key):
    v = FACTS.get(key + ".pm") or FACTS.get(key)
    if v is None:
        return None
    m = re.match(r"([-0-9.]+)(?: ± ([0-9.]+))?", str(v))
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else 0.0)


def fv(key, nd=3, sd=True, na="not run"):
    v = pm(key)
    if v is None:
        return na
    return f"{v[0]:.{nd}f}" + (f" ± {v[1]:.{nd}f}" if sd and v[1] else "")


def seeds(ds, var):
    return FACTS.get(f"{ds}.seeds.{var}", 0)


def P(ds):
    """Middle cascade probability key for a dataset."""
    ks = sorted({k.split(".")[2] + "." + k.split(".")[3] for k in FACTS if k.startswith(f"{ds}.inf.")})
    ks = sorted(ks, key=float)
    return ks[len(ks) // 2] if ks else "0"


# ------------------------------------------------------------------------------------------ helpers
def new_slide(dark=False, section=""):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = NAVY if dark else OFF
    slide_no[0] += 1
    if not dark and slide_no[0] > 1:
        tb(s, 0.6, 7.05, 9, 0.3, "Attention-driven GNN for influence and anomaly detection  |  Krish Kapoo"
           "r (23MIC0023), A.M. Karthikeshwaran (23MID0215)", 9, MUTED)
        tb(s, 12.2, 7.05, 0.6, 0.3, str(slide_no[0]), 9, MUTED, align=PP_ALIGN.RIGHT)
    return s


def _run(p, text, size, color, bold=False, font=BODY, italic=False):
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.bold = bold
    r.font.name = font
    r.font.italic = italic
    return r


def tb(slide, x, y, w, h, text, size=14, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
       anchor=MSO_ANCHOR.TOP, italic=False, name=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        _rich(p, ln, size, color, bold, font, italic)
    return box


def _rich(p, text, size, color, bold=False, font=BODY, italic=False):
    """**bold** segments inside text."""
    for k, seg in enumerate(re.split(r"(\*\*[^*]+\*\*)", text)):
        if not seg:
            continue
        if seg.startswith("**"):
            _run(p, seg[2:-2], size, color, True, font, italic)
        else:
            _run(p, seg, size, color, bold, font, italic)


def bullets(slide, x, y, w, h, items, size=15, color=INK, gap=6, name=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Inches(0.22))))
        pPr.set("indent", str(-int(Inches(0.22))))
        bu = etree.SubElement(pPr, qn("a:buChar"))
        bu.set("char", "•")
        p.space_after = Pt(gap)
        _rich(p, it, size, color)
    return box


def title(slide, text, sub=None, dark=False):
    tb(slide, 0.6, 0.4, 12.1, 0.8, text, 32, WHITE if dark else NAVY, True, HEAD, name="Title")
    if sub:
        tb(slide, 0.6, 1.12, 12.1, 0.4, sub, 15, RGBColor(0xC9, 0xD6, 0xE2) if dark else MUTED)


def card(slide, x, y, w, h, head, body, accent=TEAL, size=13, head_size=16):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.05
    sh.fill.solid()
    sh.fill.fore_color.rgb = WHITE
    sh.line.color.rgb = PALE
    sh.shadow.inherit = False
    tb(slide, x + 0.2, y + 0.15, w - 0.4, 0.4, head, head_size, accent, True, HEAD)
    if isinstance(body, list):
        bullets(slide, x + 0.2, y + 0.62, w - 0.4, h - 0.75, body, size, INK, gap=7)
    else:
        tb(slide, x + 0.2, y + 0.62, w - 0.4, h - 0.75, body, size, INK)
    return sh


def pill(slide, x, y, w, h, text, fill=TEAL, color=WHITE, size=12, bold=True):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.3
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _rich(p, text, size, color, bold)
    return sh


def flowbox(slide, x, y, w, h, text, fill=WHITE, line=NAVY, size=12, color=INK):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.08
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(1.25)
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        _rich(p, ln, size, color, bold=(i == 0))
    return sh


def arrow_shape(slide, x1, y1, x2, y2, color=NAVY):
    c = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def picture(slide, path, x, y, w=None, h=None, name=None):
    if not Path(path).exists():
        tb(slide, x, y, w or 4, 0.5, f"[missing image {Path(path).name}]", 12, ORANGE)
        return None
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)
    if name:
        pic.name = name
    return pic


def table(slide, rows, x, y, w, h, font=12, col_w=None, header_fill=NAVY):
    gs = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    t = gs.table
    tblPr = gs._element.graphic.graphicData.tbl.tblPr
    for a in ("bandRow", "firstRow"):
        tblPr.set(a, "0")
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(h / len(rows))
        for j, c in enumerate(r):
            cell = t.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if i == 0 else (WHITE if i % 2 else PALE)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            _rich(p, str(c), font, WHITE if i == 0 else INK, bold=(i == 0))
    return t


def bar_chart(slide, cats, series, x, y, w, h, colors=None, nd=2, title_=None, vmin=None, vmax=None,
              legend=False, horizontal=True, size=11, point_colors=None):
    cd = CategoryChartData()
    cd.categories = cats
    for nm, vals in series:
        cd.add_series(nm, vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.font.size = Pt(size)
    ch.font.name = BODY
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
    if title_:
        ch.has_title = True
        ch.chart_title.text_frame.text = title_
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(size + 1)
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
    else:
        ch.has_title = False
    plot = ch.plots[0]
    plot.gap_width = 45
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = f"0.{'0' * nd}"
    dl.number_format_is_linked = False
    dl.font.size = Pt(size)
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = PALE
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size)
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(size)
    ca.format.line.color.rgb = MUTED
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    if horizontal:
        ca.reverse_order = True
    for si, s in enumerate(plot.series):
        col = (colors or [TEAL, NAVY, ORANGE])[si % 3]
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = col
        s.invert_if_negative = False
        if point_colors and si == 0:
            for pi, pc in enumerate(point_colors):
                pt = s.points[pi]
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = pc
    for dpt in ch._chartSpace.xpath(".//c:dPt"):
        if not dpt.xpath("c:invertIfNegative"):
            inv = etree.SubElement(dpt, qn("c:invertIfNegative"))
            inv.set("val", "0")
            dpt.remove(inv)
            dpt.insert(1, inv)
    return ch


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def eq(latex, name, fs=17):
    d = ROOT / "artifacts" / "eq"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"slide_{name}.png"
    DB.render_equation(latex, p, fs=fs)
    return p


def eq_pic(slide, latex, name, x, y, maxw, fs=17):
    from PIL import Image
    p = eq(latex, name, fs)
    w, h = Image.open(p).size
    ww = min(maxw, w / 220 * 1.25)
    return picture(slide, p, x, y, w=ww)


# ------------------------------------------------------------------------------------------ slide 1
s = new_slide(dark=True)
tb(s, 0.8, 0.7, 11, 0.4, "FINAL PROJECT PRESENTATION  |  CSI1006  |  FIFTH SEMESTER", 13, RGBColor(0x8F, 0xD3, 0xDA), True)
tb(s, 0.8, 1.5, 11.7, 2.4, ["Attention-Driven Graph Neural Networks", "for Influence and Anomaly Detection",
                              "in Complex Networks"], 40, WHITE, True, HEAD, name="Title")
tb(s, 0.8, 4.35, 11.5, 0.5, "A shared graph-attention encoder, tested as a hypothesis on CPU-sized benchmarks", 18,
   RGBColor(0xC9, 0xD6, 0xE2), italic=True)
for i, a in enumerate(META["authors"]):
    tb(s, 0.8 + i * 5.2, 5.35, 5, 0.9, [f"**{a['name']}**", f"Registration No. {a['reg_no']}"], 16, WHITE)
tb(s, 0.8, 6.55, 11.5, 0.4, META["institution"], 14, RGBColor(0xC9, 0xD6, 0xE2))
pill(s, 10.2, 0.62, 2.4, 0.4, "measured, not promised", ORANGE, WHITE, 12)
notes(s, "Opening (30 s). Introduce the team and the question: can the attention weights of a graph attention network "
         "serve both to rank influential nodes and to flag anomalous ones? Stress at the start that we treat this as a "
         "hypothesis, that every number on later slides is measured from saved runs, and that influence labels are simulated "
         "while Cora anomalies are injected.\nSources: project proposal (Review 1); report Sections 1 and 5.")

# ------------------------------------------------------------------------------------------ slide 2
s = new_slide()
title(s, "Context: hubs and outliers are hard to explain", "Why influence and anomaly questions are usually answered separately")
card(s, 0.6, 1.8, 3.9, 3.2, "The challenge",
     "Social, communication and interaction networks hold many linked entities. Finding the ones that **spread** activity "
     "and the ones that **behave oddly** strains conventional pipelines.")
card(s, 4.72, 1.8, 3.9, 3.2, "The limitation",
     "Degree, PageRank and betweenness treat neighbours alike. Deep detectors return a score but rarely say **which "
     "neighbours** drove it.", ORANGE)
card(s, 8.84, 1.8, 3.9, 3.2, "The opportunity",
     "A graph attention network learns a **weight per neighbour**. Those weights might double as an influence signal and "
     "an anomaly signal.", NAVY)
sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(5.35), Inches(12.14), Inches(1.4))
sh.adjustments[0] = 0.08
sh.fill.solid(); sh.fill.fore_color.rgb = PALE; sh.line.fill.background(); sh.shadow.inherit = False
tb(s, 0.9, 5.5, 11.6, 1.2, ["**Problem.** Given an attributed network, can one learnable model both rank nodes by influence and "
                             "flag anomalous nodes, stay interpretable, and train on a laptop CPU?",
                             "We test this against classical baselines and non-attentive controls rather than assume it."], 16, INK)
notes(s, "Context (1 min). The three cards come from the proposal: challenge, limitation, opportunity. The problem "
         "statement is the proposal's, rephrased as a testable question.\nSources: proposal slides 2-3; Velickovic et al. 2018 (GAT); "
         "Akoglu et al. 2015 (survey).")

# ------------------------------------------------------------------------------------------ slide 3
s = new_slide()
title(s, "Objectives and research questions", "The proposal's six objectives, turned into measurable checks")
bullets(s, 0.6, 1.8, 6.3, 4.9, [
    "**1** Represent networks as attributed graphs; document data, labels and splits",
    "**2** Train a multi-head attention encoder under a CPU budget",
    "**3** Test attention as a shared signal for influence ranking and anomaly scoring",
    "**4** Benchmark against degree, PageRank, betweenness, Isolation Forest and a DOMINANT-style GCN",
    "**5** Evaluate with precision, recall, F1, ROC-AUC, AP, rank correlation, NDCG and held-out cascade spread",
    "**6** Validate the attention-based interpretation by perturbation, and record compute"], 15, INK, 9)
rq = [("RQ1", "Does attention-only scoring detect anomalies as well as reconstruction?"),
      ("RQ2", "Does received attention rank held-out spread better than degree and PageRank?"),
      ("RQ3", "Does joint training help either task versus task-only training?"),
      ("RQ4", "Do high-attention edges matter more to the output than random ones?")]
for i, (a, b) in enumerate(rq):
    y = 1.8 + i * 1.24
    pill(s, 7.3, y, 0.9, 0.5, a, TEAL, WHITE, 14)
    tb(s, 8.4, y - 0.02, 4.3, 1.1, b, 14, INK)
notes(s, "Objectives (1 min). Left: the six objectives from the proposal. Right: four research questions we can answer "
         "with the saved runs. Each is answered on slides 10-13 with the measured numbers, including negative outcomes.")

# ------------------------------------------------------------------------------------------ slide 4
s = new_slide()
title(s, "Theory: message passing and attention", "Who attends to whom, and what can be summed")
tb(s, 0.6, 1.75, 6.6, 0.35, "GAT coefficient for the arc j → i (softmax over arcs arriving at i)", 14, MUTED, True)
eq_pic(s, r"\alpha_{ij}^{h}=\frac{\exp\,\mathrm{LeakyReLU}\left(\mathbf{a}_{h}^{\top}[W^{h}z_{i}\Vert W^{h}z_{j}]\right)}"
          r"{\sum_{k\in N(i)}\exp\,\mathrm{LeakyReLU}\left(\mathbf{a}_{h}^{\top}[W^{h}z_{i}\Vert W^{h}z_{k}]\right)}",
       "att", 0.6, 2.15, 6.4, 16)
tb(s, 0.6, 3.55, 6.6, 0.35, "Layer update (heads concatenated)", 14, MUTED, True)
eq_pic(s, r"z_{i}^{(l+1)}=\mathrm{ELU}\left(\Vert_{h=1}^{H}\sum_{j\in N(i)}\alpha_{ij}^{h}W^{h}z_{j}^{(l)}\right)", "upd", 0.6, 3.95, 6.4, 17)
tb(s, 0.6, 5.0, 6.6, 0.35, "Received attention of a source j (our influence signal)", 14, MUTED, True)
eq_pic(s, r"R_{j}=\sum_{i:\,j\to i,\;i\neq j}\bar{\alpha}_{ij}", "recv", 0.6, 5.4, 4.0, 18)
bullets(s, 7.6, 1.8, 5.2, 4.9, [
    "**GCN** uses fixed degree-normalised weights: not uniform, but not learned per edge",
    "**GAT** learns the weights; **GATv2** makes the neighbour ranking depend on the query node",
    "Softmax makes the coefficients **arriving** at any node sum to 1, so summing there cannot rank nodes (unit-tested)",
    "Attention **leaving** a source, summed over its targets, can differ between nodes: that is what we rank by",
    "Attention is an **association** with the output, not automatically an explanation (Jain & Wallace 2019)"], 14, INK, 9)
notes(s, "Theory (1.5 min). Kipf & Welling 2017 for GCN, Velickovic et al. 2018 for GAT, Brody et al. 2022 for GATv2. Key "
         "point: because each target's coefficients sum to one, the useful quantity is attention a source RECEIVES from its "
         "targets. Isolated nodes get zero. Attention as explanation: Jain & Wallace 2019, Wiegreffe & Pinter 2019.")

# ------------------------------------------------------------------------------------------ slide 5
s = new_slide()
title(s, "Literature survey and the defensible gap", "Ingredients exist; our contribution is the controlled integration")
rows = [["Work", "What it establishes", "Implication for us"],
        ["GCN, GAT, GATv2 (2017-2022)", "Fixed vs learned vs dynamic neighbour weights", "Matched encoder comparison"],
        ["DOMINANT, AnomalyDAE, ResGCN", "Reconstruction (and attention) for anomalies", "DOMINANT-style GCN baseline"],
        ["BOND / PyGOD (2022)", "Organic vs injected anomalies, consistent protocols", "Weibo (organic) + Cora (injected)"],
        ["DeepLinker / GANR (2019-21)", "Received attention as a node centrality", "Not new: add degree control, held-out cascades"],
        ["Kempe et al. (2003)", "Independent-cascade spread model", "Simulated influence target"],
        ["GNNExplainer, Jain & Wallace", "Attention may not explain predictions", "Perturbation test of attention"]]
table(s, rows, 0.6, 1.75, 12.1, 3.55, 13, [3.5, 4.6, 4.0])
sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(5.6), Inches(12.1), Inches(1.25))
sh.adjustments[0] = 0.08
sh.fill.solid(); sh.fill.fore_color.rgb = PALE; sh.line.fill.background(); sh.shadow.inherit = False
tb(s, 0.9, 5.72, 11.5, 1.1, ["**Gap we can defend:** one shared encoder evaluated on both tasks under one protocol, with non-attentive "
                              "controls and a perturbation test. **Not claimed:** a first unified model or a new architecture.",
                              "The proposal's 'Zhang et al. (2022)' citation could not be verified and is not used."], 14, INK)
notes(s, "Literature (1 min). All entries were opened at primary sources; the matrix and claim ledger are in docs/literature. "
         "Zhang et al. 2022 as written in the proposal could not be found in arXiv, Crossref or OpenAlex; we excluded it and "
         "list real near-matches in docs/literature/zhang2022_verification.md. DeepLinker/GANR (Gu et al.) already rank by "
         "received attention.")

# ------------------------------------------------------------------------------------------ slide 6
s = new_slide()
title(s, "System architecture and functional flow", "What is actually implemented (editable shapes)")
tb(s, 0.6, 1.6, 6, 0.3, "ARCHITECTURE", 12, TEAL, True)
cols = [("Data", "Weibo (pinned, verified)\nCora + injection\nCSV import"), ("Preparation", "symmetrise, split\nnormalise (train)\ncascade targets"),
        ("Shared model", "2-layer GAT / GATv2 / GCN\nattribute, structure,\ninfluence heads"),
        ("Scoring", "anomaly scores\nreceived attention\nvalidation threshold"),
        ("Evidence + UI", "run records, tables\nStreamlit dashboard\nreport, slides")]
bw = 2.2
for i, (h_, t_) in enumerate(cols):
    x = 0.6 + i * (bw + 0.28)
    flowbox(s, x, 1.95, bw, 1.55, h_ + "\n" + t_, WHITE if i != 2 else RGBColor(0xDF, 0xF0, 0xF1), TEAL if i == 2 else NAVY, 12)
    if i:
        arrow_shape(s, x - 0.26, 2.72, x - 0.02, 2.72)
tb(s, 0.6, 3.85, 6, 0.3, "FUNCTIONAL FLOW", 12, TEAL, True)
flow = [("Train", "unsupervised\nlosses + train\ncascade targets", RGBColor(0xDF, 0xF0, 0xF1)),
        ("Select", "validation labels:\nalpha, threshold\n(then frozen)", RGBColor(0xFF, 0xF4, 0xE5)),
        ("Extract", "attention per\nlayer, head, arc", WHITE),
        ("Score", "rec | attention-only\n| combined; R_j raw\nand adjusted", WHITE),
        ("Evaluate", "same test nodes for\nall methods", WHITE),
        ("Perturb", "remove high / random\n/ low attention\nedges", WHITE)]
fw = 1.82
for i, (h_, t_, fc) in enumerate(flow):
    x = 0.6 + i * (fw + 0.24)
    flowbox(s, x, 4.2, fw, 1.7, h_ + "\n" + t_, fc, ORANGE if h_ == "Select" else NAVY, 12)
    if i:
        arrow_shape(s, x - 0.22, 5.05, x - 0.02, 5.05)
tb(s, 0.6, 6.2, 12.1, 0.6, "Labels enter only at 'Select' (validation) and 'Evaluate' (test). The encoder never sees anomaly labels; "
   "validation or test nodes' influence targets are never used for training.", 13, MUTED, italic=True)
notes(s, "Architecture (1 min). Top row mirrors report Figure 'System architecture'; bottom row mirrors 'Functional block "
         "diagram'. Emphasise the leakage boundary: unsupervised training, validation labels choose only alpha and the "
         "threshold, test labels are used once for evaluation. Transductive disclosure: the encoder sees all nodes' features "
         "and all edges, but no labels.")

# ------------------------------------------------------------------------------------------ slide 7
s = new_slide()
title(s, "Methodology: scoring equations and objectives", "Definitions fixed before the benchmark runs")
tb(s, 0.6, 1.7, 6.3, 0.3, "Training objective (shared encoder, three losses)", 13, MUTED, True)
eq_pic(s, r"\mathcal{L}=w_{rec}\left[a\,\mathcal{L}_{attr}+(1-a)\,\mathcal{L}_{struct}\right]+\lambda\,\mathcal{L}_{inf}", "loss", 0.6, 2.05, 6.3, 17)
tb(s, 0.6, 2.95, 6.3, 0.3, "Attention-only anomaly score (no decoder output)", 13, MUTED, True)
eq_pic(s, r"s_{att}(i)=\sum_{j\in N(i)\setminus\{i\}}\tilde{\alpha}_{ij}\,\frac{\Vert x_{i}-x_{j}\Vert_{2}}{\sqrt{d}}", "satt", 0.6, 3.3, 6.0, 17)
tb(s, 0.6, 4.55, 6.3, 0.3, "Reconstruction and combined scores", 13, MUTED, True)
eq_pic(s, r"s_{rec}=a_{sel}\,z(s_{attr})+(1-a_{sel})\,z(s_{struct}),\quad s_{comb}=\frac{1}{2}z(s_{rec})+\frac{1}{2}z(s_{att})", "srec", 0.6, 4.9, 6.3, 15)
bullets(s, 7.3, 1.8, 5.5, 4.9, [
    "Encoder: 2 layers, width 64; GAT 4 heads × 16; GATv2 and GCN as controls",
    "Structure loss on **sampled** edges and non-edges (1:1): no dense N×N matrix",
    "Influence head trained on **train-node** cascade targets only",
    "Each loss sends gradient into the shared encoder (unit-tested)",
    "Joint model = our extension; attention-only scores test the proposal's stronger claim",
    "Controls: uniform-neighbour deviation, untrained-GAT attention",
    "alpha and the F1 threshold are chosen on **validation** and frozen: model selection is label-informed"], 13.5, INK, 7)
notes(s, "Method (1.5 min). Equations are in report Section 3.2. a_sel in {0,.25,.5,.75,1} is chosen by validation ROC-AUC. "
         "The combined score uses fixed 0.5/0.5 weights on train-robust-z-scored components, not tuned. Adam, lr 0.005, "
         "weight decay 5e-4. On Weibo, 30% of arcs are kept per training epoch (DropEdge-style, Rong et al. 2020) for CPU "
         "feasibility; inference uses the full graph.")

# ------------------------------------------------------------------------------------------ slide 8
s = new_slide()
title(s, "Datasets, label provenance and evaluation protocol", "Organic and injected anomalies are kept apart")
ds_rows = [["", "Weibo (organic)", "Cora + injection (synthetic)"],
           ["Nodes / undirected edges", "8,405 / 377,271", "2,708 / 5,593"],
           ["Features", "400 (user attributes)", "1,433 (bag of words)"],
           ["Anomalies", "868 (10.3%), organic labels (original 2022 file)", "140 (5.2%): 70 structural + 70 contextual"],
           ["Source", "PyGOD / BOND file", "Planetoid Cora; seeded DOMINANT-style injection"],
           ["Caveat", "2024 upstream file blanks val/test labels: pinned + SHA-256 checked", "Never observed fraud"]]
table(s, ds_rows, 0.6, 1.75, 7.4, 3.6, 12.5, [2.1, 2.7, 2.6])
bullets(s, 8.3, 1.75, 4.5, 3.8, [
    "30/20/50 split, stratified, seeded; **transductive**",
    "Normalise with **train** statistics; thresholds frozen on validation",
    "Influence: independent-cascade spread, **separate random streams** for train, validation and test nodes",
    "All rankings scored on the **same held-out test nodes**",
    "≥3 seeds (Cora 5, Weibo 3)"], 13.5, INK, 8)
tb(s, 0.6, 5.6, 12.1, 0.3, "Metrics", 13, TEAL, True)
tb(s, 0.6, 5.95, 12.1, 0.9, ["Anomaly: precision, recall, F1 at the validation threshold, ROC-AUC, average precision, precision@k, confusion matrix.",
                              "Influence: Spearman, Kendall τ, NDCG@k, top-k overlap, spread of the selected 10/20-node seed set (held-out cascades)."], 13.5, INK)
notes(s, "Data and protocol (1.5 min). Cora topic classes are NOT anomaly labels. Weibo labels are organic (suspicious posting "
         "behaviour, BOND Appendix A.1.1). Our dataset validation found that the current upstream Weibo file blanks the validation and test labels (347 of 868 anomalies remain); "
         "we pin the original 2022 commit, verify its SHA-256 and re-ran every Weibo experiment. The comparison with the corrupted run is in report Sec. 5.1. Influence is a simulation proxy, never observed influence. Sources: Liu et al. 2022 (BOND); Ding et al. 2019 "
         "(injection); Kempe et al. 2003 (IC); Sen et al. 2008 and Yang et al. 2016 (Cora).")

# ------------------------------------------------------------------------------------------ slide 9
s = new_slide()
title(s, "Implemented features and real dashboard screenshots", "Streamlit app running locally; every view reads saved runs")
shots = [("01_overview.png", "Overview: animated hero and count-up tiles"), ("03_dataset_authenticity.png", "Dataset authenticity page"),
         ("06_anomaly_scores.png", "Anomaly scores + live distribution"), ("07_graph_explorer.png", "Sampled graph + node inspector")]
CROP_Y = {"01_overview.png": 0}
from PIL import Image
CROPS = ROOT / "artifacts" / "render" / "crops"
CROPS.mkdir(parents=True, exist_ok=True)
for i, (f, cap) in enumerate(shots):
    x = 0.6 + (i % 2) * 4.55
    y = 1.55 + (i // 2) * 2.62
    src_img = SHOT / f
    if src_img.exists():
        im = Image.open(src_img)
        y0 = CROP_Y.get(f, 120)
        crop = im.crop((300, y0, 1500, y0 + 675))      # main content area, 16:9
        crop.save(CROPS / f)
        picture(s, CROPS / f, x, y, w=4.1, name=cap)
    tb(s, x, y + 2.33, 4.1, 0.25, cap, 11, MUTED)
bullets(s, 9.85, 1.75, 3.0, 5.0, [
    "CSV graph import with validation",
    "Train / smoke jobs, status, checkpoint reload",
    "Influence table with degree, PageRank, simulated spread",
    "Adjustable anomaly threshold (tuned on validation)",
    "Layer / head attention view",
    "Saved-run comparisons, exports"], 13, INK, 5)
notes(s, "Features and demo sequence (2 min). All screenshots are from the running dashboard.\nDemo sequence (see reports/demo_script.md): "
         "1) Data & graph summary: pick Cora, show injection parameters. 2) Train / evaluate: start a smoke job, watch the log, then "
         "reload a checkpoint and verify that saved predictions are reproduced. 3) Anomaly scores: move the threshold, note that "
         "metrics at the slider are shown on validation only. 4) Influence ranking: pick p, sort by received attention and by "
         "degree. 5) Graph explorer: centre on the top test anomaly, switch layer and head. 6) Demos: known injected anomaly, "
         "high-spread node, explanation-vs-perturbation, model reload.")

# ------------------------------------------------------------------------------------------ slide 10
s = new_slide()
title(s, "Anomaly results and baseline comparison", "Test ROC-AUC, mean over seeds (axis starts at chance); controls included")
ROWS = [("Isolation Forest", "iforest_attr", "iforest"), ("GCN rec. (DOMINANT-style)", "gcn_anom", "rec"),
        ("GAT reconstruction", "gat_anom", "rec"), ("GAT attention-only", "gat_anom", "att_only"),
        ("GAT combined", "gat_anom", "combined"), ("Uniform-neighbour control", "gat_anom", "feat_dev_uniform")]
for k, (ds, lab, x) in enumerate([("weibo", "Weibo (organic labels)", 0.6), ("cora_injected", "Cora (injected labels)", 6.75)]):
    vals, cats = [], []
    for nm, var, sc in ROWS:
        v = pm(f"{ds}.anom.{var}.{sc}.roc_auc")
        cats.append(nm)
        vals.append(round(v[0], 3) if v else 0)
    n = seeds(ds, "gat_anom")
    bar_chart(s, cats, [("ROC-AUC", vals)], x, 1.7, 6.0, 3.9, nd=3, title_=f"{lab}, n={n} seeds", vmin=0.4, vmax=1.0, size=11,
              point_colors=[GREY, GREY, NAVY, TEAL, TEAL, SAND])
cora_u, cora_a = pm("cora_injected.anom.gat_anom.feat_dev_uniform.roc_auc"), pm("cora_injected.anom.gat_anom.att_only.roc_auc")
w_if, w_rec = pm("weibo.anom.iforest_attr.iforest.roc_auc"), pm("weibo.anom.gat_anom.rec.roc_auc")
bullets(s, 0.6, 5.75, 12.1, 1.3, [
    f"Cora: attention-only {fv('cora_injected.anom.gat_anom.att_only.roc_auc', sd=False)} vs uniform-neighbour control "
    f"{fv('cora_injected.anom.gat_anom.feat_dev_uniform.roc_auc', sd=False)} vs untrained attention {fv('cora_injected.anom.gat_untrained.att_only.roc_auc', sd=False)}: "
    "**learned attention adds at most +0.006 AUC** (paired, 5 of 5 seeds)",
    f"Weibo: GCN reconstruction {fv('weibo.anom.gcn_anom.rec.roc_auc', sd=False)}, GATv2 {fv('weibo.anom.gatv2_anom.rec.roc_auc', sd=False)}, GAT {fv('weibo.anom.gat_anom.rec.roc_auc', sd=False)}; "
    f"attention-only {fv('weibo.anom.gat_anom.att_only.roc_auc', sd=False)} is **below** its control {fv('weibo.anom.gat_anom.feat_dev_uniform.roc_auc', sd=False)}; "
    f"Isolation Forest {fv('weibo.anom.iforest_attr.iforest.roc_auc', sd=False)} (below chance)"], 13, INK, 4)
notes(s, "Anomaly results (1.5 min). Bars are means over seeds (standard deviations in report Table 5.2 and in artifacts/results/tables). "
         "Colours: grey baselines, navy reconstruction, teal attention-based scores, sand control. Message for RQ1: the "
         "attention-only score matches reconstruction on injected Cora but also matches an untrained-attention and a "
         "uniform-weight control, so the signal comes from the neighbour feature discrepancy, not from learned attention. "
         "Weibo: BOND reports that classical methods already do very well there, so a strong baseline is expected.\n"
         "Sources: results index artifacts/results/results_index.json; Liu et al. 2022.")

# ------------------------------------------------------------------------------------------ slide 11
s = new_slide()
title(s, "Influence results and baseline comparison", "Spearman vs held-out simulated spread (middle cascade probability), mean over seeds")
IR = [("Degree", "centrality", "degree"), ("PageRank", "centrality", "pagerank"), ("Betweenness", "centrality", "betweenness"),
      ("Uniform attention (control)", "centrality", "uniform_recv_att"), ("GAT received attention", "gat_joint", "recv_att"),
      ("  degree-adjusted", "gat_joint", "recv_att_adj"), ("GCN influence head", "gcn_joint", "model_pred"),
      ("GAT influence head", "gat_joint", "model_pred")]
for ds, lab, x in [("weibo", "Weibo (p=" + P("weibo") + ")", 0.6), ("cora_injected", "Cora (p=" + P("cora_injected") + ")", 6.75)]:
    p = P(ds)
    vals = []
    for nm, var, meth in IR:
        v = pm(f"{ds}.inf.{p}.{var}.{meth}.spearman")
        vals.append(round(v[0], 3) if v else 0)
    bar_chart(s, [r[0] for r in IR], [("Spearman", vals)], x, 1.7, 6.0, 3.9, nd=2, title_=f"{lab}, n={seeds(ds, 'gat_joint')} seeds", vmin=-0.7, vmax=1.0,
              size=11, point_colors=[GREY, GREY, GREY, SAND, TEAL, TEAL, NAVY, NAVY])
cd = P("cora_injected")
wd = P("weibo")
bullets(s, 0.6, 5.75, 12.1, 1.3, [
    f"Weibo: degree {fv('weibo.inf.' + wd + '.centrality.degree.spearman', sd=False)}, **GCN head {fv('weibo.inf.' + wd + '.gcn_joint.model_pred.spearman', sd=False)}**, "
    f"GAT received attention {fv('weibo.inf.' + wd + '.gat_joint.recv_att.spearman', sd=False)} (non-learned uniform control {fv('weibo.inf.' + wd + '.centrality.uniform_recv_att.spearman', sd=False)})",
    f"Cora: received attention {fv('cora_injected.inf.' + cd + '.gat_joint.recv_att.spearman', sd=False)} = uniform control {fv('cora_injected.inf.' + cd + '.centrality.uniform_recv_att.spearman', sd=False)}; "
    f"degree adjustment turns it negative ({fv('cora_injected.inf.' + cd + '.gat_joint.recv_att_adj.spearman', sd=False)})",
    f"Top-10 seed set (Cora): degree {fv('cora_injected.seedset.degree10', 0, False)} vs random {fv('cora_injected.seedset.random10', 0, False)} activated nodes"], 13, INK, 4)
notes(s, "Influence results (1.5 min). Ground truth = held-out independent-cascade spread on the same test nodes for every method. "
         "RQ2: received attention tracks the non-learned uniform-attention control closely because learned attention is nearly "
         "uniform (entropy ratio about 0.98-1.0 on Cora). Individual spread differs from seed-set spread: the oracle top-10 by "
         "individual spread is not the best seed set because chosen nodes overlap. Not observed influence: simulation only.\n"
         "Sources: Kempe et al. 2003; Gu et al. 2019/2021 for received attention; report Tables in Section 5.")

# ------------------------------------------------------------------------------------------ slide 12
s = new_slide()
title(s, "Ablation, explanation validation and measured compute", "RQ3 and RQ4, with measured cost")
abl = [["Cora, ROC-AUC / influence ρ", "Anomaly", "Influence head"]]
for nm, var in [("GCN joint", "gcn_joint"), ("GAT joint", "gat_joint"), ("GATv2 joint", "gatv2_joint"), ("GAT 1-head joint", "gat1h_joint"),
                ("GAT anomaly-only", "gat_anom"), ("GAT influence-only", "gat_inf")]:
    abl.append([nm, fv(f"cora_injected.anom.{var}.rec.roc_auc", 3, False) if var != "gat_inf" else "n/a (head untrained)",
                fv(f"cora_injected.inf.{cd}.{var}.model_pred.spearman", 3, False) if var != "gat_anom" else "n/a"])
table(s, abl, 0.6, 1.7, 4.6, 3.2, 12, [2.2, 1.1, 1.3])
import pandas as pd
exd = {}
for dsn in ("cora_injected", "weibo"):
    pth = RES / "tables" / f"explanation_{dsn}.csv"
    if pth.exists():
        exd[dsn] = {r["output"]: r for _, r in pd.read_csv(pth).iterrows()}
if exd:
    cats = ["high attention", "random", "low attention"]
    ser = []
    for dsn, nm in (("cora_injected", "Cora anomaly"), ("weibo", "Weibo anomaly"), ("weibo", "Weibo influence")):
        out = "influence" if nm.endswith("influence") else "anomaly"
        if dsn in exd and out in exd[dsn]:
            r = exd[dsn][out]
            ser.append((nm, [float(r["mean effect: remove high-attention"]), float(r["mean effect: remove random"]), float(r["mean effect: remove low-attention"])]))
    bar_chart(s, cats, ser, 5.4, 1.65, 3.7, 3.3, nd=2, title_="Output drop after edge removal", size=10, horizontal=False, legend=True, colors=[TEAL, NAVY, ORANGE])
    tb(s, 5.4, 4.95, 3.7, 0.4, "Weibo high vs random: anomaly p = 0.003, influence p = 0.0001; Cora anomaly p = 0.0095", 10, MUTED)
res = [["Cora train s", "Weibo train s"]]
rc = pd.read_csv(RES / "tables" / "resources_cora_injected.csv") if (RES / "tables" / "resources_cora_injected.csv").exists() else None
rw = pd.read_csv(RES / "tables" / "resources_weibo.csv") if (RES / "tables" / "resources_weibo.csv").exists() else None


def tsec(df, var):
    if df is None:
        return "not run"
    r = df[df.variant == var]
    return re.sub(r" \[n=\d+\]", "", r["train seconds"].iloc[0]).split(" ±")[0] if len(r) else "not run"


def prof(ds, var, col):
    p = RES / "tables" / f"resources_profile_{ds}.csv"
    if not p.exists():
        return "not run"
    d = pd.read_csv(p)
    r = d[d.variant == var]
    return f"{float(r[col].iloc[0]):.0f}" if len(r) else "not run"


MEM = "peak working set, MB (isolated process)"
comp = [["Isolated run", "Cora s", "Weibo s", "Weibo MB"]] + [
    [nm, prof("cora_injected", v, "train seconds"), prof("weibo", v, "train seconds"), prof("weibo", v, MEM)]
    for nm, v in [("GCN joint", "gcn_joint"), ("GAT joint", "gat_joint"), ("GATv2 joint", "gatv2_joint")]]
table(s, comp, 9.3, 1.7, 3.5, 1.9, 12, [1.3, 0.7, 0.8, 0.7])
tb(s, 9.3, 3.75, 3.5, 1.2, "Training seconds and memory, single runs, CPU only (i7-1165G7). On Weibo the benchmark averages agree with these; Cora benchmark averages were about 2x higher (CPU shared).", 11, MUTED)
bullets(s, 0.6, 5.35, 12.1, 1.5, [
    "Joint vs task-only: no reliable benefit (anomaly AUC +0.005 on Cora, +0.007 on Weibo; the influence-only head is as good or better)",
    "Edge removal: high-attention edges matter more than random ones for anomaly scores (Cora +0.019, Weibo +0.067) and for influence on Weibo (+0.298), not on Cora where attention is uniform",
    "Attention describes what the trained model uses; it is not a causal explanation and not a better predictor than the controls"], 13, INK, 4)
notes(s, "Ablations, explanation test and compute (1.5 min). Ablation numbers are means over seeds from artifacts/results/tables/ablation_*.csv. "
         "Perturbation test: remove m<=3 edges of a target chosen by highest attention, lowest attention or at random; recompute the "
         "frozen model's output identically; paired bootstrap CI over targets pooled across seeds. Parameter budgets: GAT and GCN "
         "have about 96k encoder parameters, GATv2 about 192k (two weight matrices); 1-head and 4-head GAT have identical "
         "parameter counts. Compute: wall-clock measured per run; memory in the report is a cumulative process peak (upper bound).")

# ------------------------------------------------------------------------------------------ slide 13
s = new_slide()
title(s, "Outcomes, failure cases, limitations and conclusion", "What the measurements support, and what they do not")
card(s, 0.6, 1.75, 3.95, 4.9, "What we found", [
    "Reconstruction and neighbour-discrepancy scores detect injected Cora anomalies (AUC about 0.77 to 0.80); Isolation Forest 0.68",
    "Attention-only scoring is matched by controls on Cora and weaker on Weibo",
    "Received attention equals the uniform control on Cora and is worse on Weibo; degree adjustment destroys it",
    "A plain GCN head is the best learned influence ranker (Weibo Spearman 0.89)",
    "Edge removal: attention marks edges that matter to the model where it is sharp (Weibo), not where it is uniform (Cora)"], TEAL, 13, 15)
card(s, 4.72, 1.75, 3.95, 4.9, "Failure cases and limits", [
    "Data trap: the upstream Weibo file blanks val/test labels; our first run used it. Found by validation, fixed, re-run",
    "GAT influence head overfits noisy targets",
    "Influence is simulated; Cora anomalies are injected",
    "Weibo structure error and Isolation Forest below chance; transductive; arc subsampling; 3 to 5 seeds"], ORANGE, 13, 15)
card(s, 8.84, 1.75, 3.95, 4.9, "Conclusion", [
    "As tested, attention-derived signals do not beat simple controls on either task",
    "The negative result is itself informative; the pipeline, controls and perturbation test are reusable",
    "Joint training: no reliable benefit at this scale",
    "Claims are limited to these datasets, budgets and proxies"], NAVY, 13, 15)
notes(s, "Outcomes (1.5 min). Be explicit that these are valid negative outcomes: they are results, not failures of the project. "
         "State the limits: simulation proxy for influence, injected labels on Cora, label-count discrepancy on Weibo, "
         "transductive evaluation, CPU compute budget, and few seeds. What is not run is listed in the report and PROGRESS.md.")

# ------------------------------------------------------------------------------------------ slide 14
s = new_slide()
title(s, "Future work and selected references")
bullets(s, 0.6, 1.5, 5.6, 5.3, [
    "Test attention on graphs with **dynamic** or sharper attention (GATv2 with entropy regularisation)",
    "Use **measured** cascades instead of simulated spread",
    "Run the full BOND suite and Reddit (near-chance for most detectors) under the same protocol",
    "Multi-scale neighbourhoods (Gutiérrez-Gómez et al. 2020)",
    "Stronger faithfulness tests and explainers (GNNExplainer-style)",
    "Inductive evaluation on unseen graphs"], 14, INK, 8)
refs = ["Kipf & Welling. Semi-supervised classification with GCNs. ICLR 2017.",
        "Veličković et al. Graph attention networks. ICLR 2018.",
        "Brody, Alon, Yahav. How attentive are GATs? ICLR 2022.",
        "Ding et al. Deep anomaly detection on attributed networks. SDM 2019.",
        "Liu et al. BOND: benchmarking unsupervised outlier node detection. NeurIPS D&B 2022.",
        "Kempe, Kleinberg, Tardos. Maximizing the spread of influence. KDD 2003.",
        "Ying et al. GNNExplainer. NeurIPS 2019.",
        "Gu et al. DeepLinker / GANR (attention rank). arXiv 2019; Sci. Rep. 2021.",
        "Akoglu, Tong, Koutra. Graph-based anomaly detection and description. DMKD 2015."]
tb(s, 6.6, 1.55, 6.2, 0.35, "Selected references (full list in report Section 8)", 13, TEAL, True)
bullets(s, 6.6, 1.95, 6.2, 4.9, refs, 12, INK, 5)
notes(s, "Close (1 min). Future work follows directly from the limits. Full numbered bibliography: report Section 8 and "
         "docs/literature/references.bib; every entry was opened at a primary source. Questions.")

out = OUT / "Attention_GNN_Final.pptx"
prs.save(out)
print("saved", out, "slides:", len(prs.slides))
