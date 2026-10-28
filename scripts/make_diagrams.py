"""Original architecture and functional block diagrams (matplotlib). Output: docs/figures/*.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, TEAL, ORANGE, PALE, GREY = "#0b2545", "#13808b", "#d9822b", "#eef3f7", "#8d99ae"
plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text, fc=PALE, ec=NAVY, tc=NAVY, fs=9, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, a, b, color=NAVY, ls="-", lw=1.3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw, ls=ls,
                                 shrinkA=2, shrinkB=2))


def mid_r(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def mid_l(b):
    return (b[0], b[1] + b[3] / 2)


def architecture():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 13.4)
    ax.axis("off")
    W = 4.6
    xs = [0.2, 5.5, 10.8, 16.1, 21.4]
    for x, t in zip(xs, ["1  Data", "2  Preparation", "3  Shared model", "4  Scoring and ranking",
                         "5  Evidence and interface"]):
        ax.text(x, 12.8, t, fontsize=11, fontweight="bold", color=TEAL)
    box(ax, xs[0], 9.9, W, 1.9, "Weibo (organic labels)\nPyGOD / BOND file\n8,405 nodes, 400 features")
    box(ax, xs[0], 7.2, W, 2.2, "Cora (Planetoid) plus seeded\nanomaly injection (synthetic):\ncliques and swapped attributes")
    box(ax, xs[0], 4.7, W, 2.0, "CSV import\nedges, features, labels\n(validated, size-limited)")
    box(ax, xs[1], 5.2, W, 6.6, "Preprocessing\n\nsymmetrise, de-duplicate,\ndrop self-loops\n\nstratified train / val / test\nsplit (seeded, hashed)\n\nnormalise with TRAIN\nstatistics only")
    box(ax, xs[1], 1.2, W, 3.0, "Independent-cascade\nsimulation\nseparate random streams for\ntrain, val and test nodes")
    arrow(ax, (xs[0] + W, 10.85), (xs[1], 10.85))
    arrow(ax, (xs[0] + W, 8.3), (xs[1], 8.3))
    arrow(ax, (xs[0] + W, 5.7), (xs[1], 5.7))
    arrow(ax, (xs[1] + W / 2, 5.2), (xs[1] + W / 2, 4.2))
    box(ax, xs[2], 8.0, W, 3.8, "Shared encoder\n2 layers, width 64\nGAT (4 heads x 16)\nGATv2 / GCN variants\nattention per layer, head, arc",
        fc="#dff0f1", ec=TEAL)
    box(ax, xs[2], 3.9, W, 3.4, "Task heads\nattribute decoder (conv + linear)\nstructure decoder sigmoid(zi.zj)\ninfluence head (MLP 64-32-|p|)", fs=8.6)
    box(ax, xs[2], 0.6, W, 2.4, "Training objective\nw_rec (a L_attr + (1-a) L_struct)\n+ lam L_inf  (all reach encoder)",
        fc="#fff4e5", ec=ORANGE, fs=8.6)
    arrow(ax, (xs[1] + W, 9.0), (xs[2], 9.0))
    arrow(ax, (xs[1] + W, 2.7), (xs[2], 4.6))
    arrow(ax, (xs[2] + W / 2, 8.0), (xs[2] + W / 2, 7.3))
    arrow(ax, (xs[2] + W / 2, 3.9), (xs[2] + W / 2, 3.0))
    box(ax, xs[3], 9.4, W, 2.4, "Anomaly scores\nreconstruction, attention-only,\ncombined, uniform control")
    box(ax, xs[3], 6.2, W, 2.6, "Validation-chosen threshold\n(frozen before test)\nROC-AUC, AP, F1, P@k,\nconfusion matrix")
    box(ax, xs[3], 3.1, W, 2.6, "Influence ranking\nreceived attention (raw, adjusted)\nhead output vs degree,\nPageRank, betweenness", fs=8.2)
    box(ax, xs[3], 0.4, W, 2.2, "Explanation test\nremove high / random / low\nattention edges, recompute")
    arrow(ax, (xs[2] + W, 10.0), (xs[3], 10.0))
    arrow(ax, (xs[3] + W / 2, 9.4), (xs[3] + W / 2, 8.8))
    arrow(ax, (xs[2] + W, 4.9), (xs[3], 4.4))
    arrow(ax, (xs[3] + W / 2, 3.1), (xs[3] + W / 2, 2.6), ls="--", color=GREY)
    box(ax, xs[4], 7.4, 4.4, 4.4, "Run record\nconfig, command, hashes,\nlogs, checkpoint,\npredictions, metrics", fc="#e7f4ec", ec=TEAL)
    box(ax, xs[4], 4.4, 4.4, 2.4, "Results builder\ntables and figures from runs\n+ results index", fc="#e7f4ec", ec=TEAL)
    box(ax, xs[4], 0.4, 4.4, 3.4, "Streamlit dashboard\nreport (DOCX / PDF / MD)\nslides (PPTX)", fc="#e7f4ec", ec=TEAL)
    arrow(ax, (xs[3] + W, 10.6), (xs[4], 10.6))
    arrow(ax, (xs[3] + W, 7.5), (xs[4], 7.9))
    arrow(ax, (xs[3] + W, 4.4), (xs[4], 5.4))
    arrow(ax, (xs[3] + W, 1.5), (xs[4], 1.5))
    arrow(ax, (xs[4] + 2.2, 7.4), (xs[4] + 2.2, 6.8))
    arrow(ax, (xs[4] + 2.2, 4.4), (xs[4] + 2.2, 3.8))
    fig.suptitle("System architecture of the implemented pipeline (CPU-first, local only)", fontsize=13,
                 color=NAVY, fontweight="bold", y=0.96)
    fig.savefig(OUT / "architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def block_diagram():
    fig, ax = plt.subplots(figsize=(15, 7.6))
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.text(0.2, 12.4, "A  Training and selection (labels are used only where marked)", fontsize=11,
            fontweight="bold", color=TEAL)
    w = 4.4
    xs = [0.2, 5.4, 10.6, 15.8, 21.0]
    labs = [("Attributed graph\nX (N x d), edges E", PALE, NAVY), ("Split and normalise\n(train statistics)", PALE, NAVY),
            ("Train shared encoder\nunsupervised losses,\nno anomaly labels", "#dff0f1", TEAL),
            ("Cascade targets of\nTRAIN nodes only\ntrain influence head", "#dff0f1", TEAL),
            ("VALIDATION labels set\nalpha_sel and the F1\nthreshold; then frozen", "#fff4e5", ORANGE)]
    bx = [box(ax, x, 9.6, w, 2.2, t, fc=fc, ec=ec) for x, (t, fc, ec) in zip(xs, labs)]
    for a, b in zip(bx[:-1], bx[1:]):
        arrow(ax, mid_r(a), mid_l(b))
    ax.text(0.2, 8.4, "B  Inference and evaluation on held-out test nodes", fontsize=11, fontweight="bold", color=TEAL)
    box(ax, 0.2, 3.4, 3.8, 2.6, "Frozen model\nforward pass\n(eval mode)", fc="#dff0f1", ec=TEAL)
    box(ax, 5.0, 6.0, 4.2, 1.9, "Reconstruction errors\nattr_err, struct_err")
    box(ax, 5.0, 0.6, 4.2, 4.2, "Attention per arc\n(layer x head)\nextracted in eval mode")
    box(ax, 10.4, 6.0, 4.2, 1.9, "rec score and\ncombined score")
    box(ax, 10.4, 3.4, 4.2, 1.9, "attention-only score:\nweighted feature discrepancy", fs=8.6)
    box(ax, 10.4, 0.7, 4.2, 1.9, "received attention R_j\nraw | degree-adjusted")
    box(ax, 15.8, 4.4, 4.0, 3.4, "Anomaly metrics\nROC-AUC, AP, F1, P@k,\nconfusion, curves")
    box(ax, 15.8, 0.4, 4.0, 3.2, "Influence metrics\nSpearman, Kendall, NDCG,\ntop-k overlap, seed-set\nspread (held-out cascades)", fs=8.6)
    box(ax, 20.8, 1.0, 4.9, 6.4, "Compared on the SAME\ntest nodes\n\nIsolation Forest, GCN\n(DOMINANT-style), GATv2,\ndegree, PageRank,\nbetweenness, uniform-\nattention and untrained-GAT\ncontrols\n\nExplanation test:\nedge removal", fc="#e7f4ec", ec=TEAL, fs=8.6)
    arrow(ax, (4.0, 5.2), (5.0, 6.9))
    arrow(ax, (4.0, 4.2), (5.0, 3.6))
    arrow(ax, (9.2, 6.95), (10.4, 6.95))
    arrow(ax, (9.2, 4.0), (10.4, 4.3))
    arrow(ax, (9.2, 1.6), (10.4, 1.65))
    arrow(ax, (14.6, 6.95), (15.8, 6.6))
    arrow(ax, (14.6, 4.35), (15.8, 5.2))
    arrow(ax, (14.6, 1.65), (15.8, 1.9))
    arrow(ax, (19.8, 6.0), (20.8, 5.6))
    arrow(ax, (19.8, 2.0), (20.8, 2.8))
    fig.suptitle("Functional block diagram", fontsize=13, color=NAVY, fontweight="bold", y=0.97)
    fig.savefig(OUT / "block_diagram.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    block_diagram()
    print("diagrams written to", OUT)
