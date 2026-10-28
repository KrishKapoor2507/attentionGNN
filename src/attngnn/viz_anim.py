"""Animated Plotly figures for the dashboard (pure functions; no Streamlit imports so they can be unit tested).

All colours follow the project palette: navy 0b2545, teal 13808b, orange d9822b, sand e9c46a.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

NAVY, TEAL, ORANGE, SAND, GREY = "#0b2545", "#13808b", "#d9822b", "#e9c46a", "#b8c4d0"


def _play_controls(frame_ms=450, transition_ms=250, label_prefix="step"):
    return dict(
        type="buttons", direction="left", x=0.02, y=-0.08, showactive=False,
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=True,
                                  transition=dict(duration=transition_ms, easing="cubic-in-out"))]),
            dict(label="❚❚ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]),
        ])


def _slider(names, prefix):
    return dict(active=0, x=0.2, len=0.78, y=-0.05, pad=dict(t=30, b=0),
                currentvalue=dict(prefix=prefix, font=dict(size=12)),
                steps=[dict(label=str(n), method="animate",
                            args=[[str(n)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                 transition=dict(duration=0))]) for n in names])


# ------------------------------------------------------------------------------------- cascade
def simulate_cascade(arcs_src, arcs_dst, nodes, seed_node, p, rng_seed=0, max_steps=12):
    """Independent-cascade steps restricted to the given arcs. Returns (activation_step per node (-1 = never),
    list of fired arcs per step). Purely illustrative: it is ONE random cascade on a small sampled subgraph."""
    rng = np.random.default_rng(rng_seed)
    out = {int(v): [] for v in nodes}
    for s, d in zip(arcs_src, arcs_dst):
        out[int(s)].append(int(d))
    step = {int(v): -1 for v in nodes}
    step[int(seed_node)] = 0
    frontier = [int(seed_node)]
    fired = [[]]
    t = 0
    while frontier and t < max_steps:
        t += 1
        nxt, fire = [], []
        for u in frontier:
            for v in out[u]:
                if step[v] == -1 and rng.random() < p:
                    step[v] = t
                    nxt.append(v)
                    fire.append((u, v))
        fired.append(fire)
        frontier = nxt
    return step, fired


def cascade_figure(pos, edges, step, fired, seed_node, title="Independent-cascade spread (one random run)"):
    """pos: node -> (x, y); edges: list of (u, v) undirected pairs; step/fired from simulate_cascade."""
    T = max(max(step.values()), 0)
    nodes = list(pos)
    xs, ys = [pos[v][0] for v in nodes], [pos[v][1] for v in nodes]
    ex, ey = [], []
    for u, v in edges:
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    base_edges = go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#c9d3dd", width=1), hoverinfo="skip", showlegend=False)

    def frame_data(t):
        colors, sizes, texts = [], [], []
        for v in nodes:
            s = step[v]
            if v == seed_node:
                colors.append(ORANGE); sizes.append(20)
            elif 0 <= s < t:
                colors.append(TEAL); sizes.append(12)
            elif s == t:
                colors.append(SAND); sizes.append(17)
            else:
                colors.append(GREY); sizes.append(9)
            texts.append(f"node {v}: " + (f"activated at step {s}" if s >= 0 else "not activated in this run"))
        fx, fy = [], []
        for u, v in (fired[t] if t < len(fired) else []):
            fx += [pos[u][0], pos[v][0], None]
            fy += [pos[u][1], pos[v][1], None]
        active = sum(1 for v in nodes if 0 <= step[v] <= t)
        return [go.Scatter(x=fx, y=fy, mode="lines", line=dict(color=ORANGE, width=3), hoverinfo="skip", showlegend=False),
                go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color=colors, size=sizes, line=dict(color=NAVY, width=0.7)),
                           text=texts, hoverinfo="text", showlegend=False)], active

    frames, names = [], list(range(T + 1))
    for t in names:
        data, active = frame_data(t)
        frames.append(go.Frame(data=data, traces=[1, 2], name=str(t),
                               layout=go.Layout(title_text=f"{title}: step {t}, {active} nodes active")))
    d0, a0 = frame_data(0)
    fig = go.Figure(data=[base_edges] + d0, frames=frames)
    fig.update_layout(title=f"{title}: step 0, {a0} nodes active", height=520, plot_bgcolor="white",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=50, b=70),
                      updatemenus=[_play_controls(650, 350)], sliders=[_slider(names, "step ")])
    return fig


# ------------------------------------------------------------------------------------- training replay
def loss_replay_figure(hist, n_frames=40):
    """hist: dict with lists loss/attr/struct/inf (one value per epoch). Curves are drawn progressively."""
    series = [("total", "loss", NAVY), ("attribute", "attr", TEAL), ("structure", "struct", ORANGE), ("influence", "inf", "#7b2cbf")]
    series = [(n, k, c) for n, k, c in series if hist.get(k) and np.ptp(hist[k]) > 0]
    E = len(hist[series[0][1]]) if series else 0
    if not series or E < 2:
        return None
    cuts = sorted({int(round(x)) for x in np.linspace(2, E, n_frames)})
    ymax = max(max(hist[k][:]) for _, k, _ in series)
    ymin = min(min(hist[k]) for _, k, _ in series)
    idx = list(range(len(series)))
    frames = [go.Frame(name=str(c), traces=idx, data=[go.Scatter(x=list(range(1, c + 1)), y=hist[k][:c]) for _, k, _ in series])
              for c in cuts]
    fig = go.Figure(data=[go.Scatter(x=list(range(1, E + 1)), y=hist[k], name=n, mode="lines", line=dict(color=c, width=2.5))
                          for n, k, c in series]
                    + [go.Scatter(x=list(range(1, E + 1)), y=hist[k], mode="lines", line=dict(color=c, width=1, dash="dot"), opacity=0.25,
                                  hoverinfo="skip", showlegend=False) for _, k, c in series], frames=frames)
    fig.update_layout(height=420, plot_bgcolor="white", margin=dict(l=10, r=10, t=40, b=70), yaxis_type="log",
                      xaxis=dict(title="epoch", range=[1, E]), yaxis=dict(title="loss (log scale)", range=[np.log10(max(ymin, 1e-4)), np.log10(ymax * 1.1)]),
                      title="Training replay: how each loss falls (all losses reach the shared encoder)",
                      updatemenus=[_play_controls(120, 60)], sliders=[dict(_slider(cuts, "epoch "), active=len(cuts) - 1)])
    return fig


# ------------------------------------------------------------------------------------- ROC drawing
def roc_draw_figure(curves, n_frames=24):
    """curves: dict name -> {'fpr': [...], 'tpr': [...]}; drawn progressively along the false-positive axis."""
    names = list(curves)
    palette = [NAVY, TEAL, ORANGE, "#7b2cbf", GREY, SAND]
    fr = np.linspace(0.04, 1.0, n_frames)

    def part(c, f):
        x = np.array(c["fpr"])
        y = np.array(c["tpr"])
        k = max(2, int(np.searchsorted(x, f)))
        return x[:k].tolist(), y[:k].tolist()

    idx = list(range(len(names)))
    frames = [go.Frame(name=f"{f:.2f}", traces=idx, data=[go.Scatter(x=part(curves[n], f)[0], y=part(curves[n], f)[1]) for n in names])
              for f in fr]
    fig = go.Figure(data=[go.Scatter(x=curves[n]["fpr"], y=curves[n]["tpr"], name=n, mode="lines",
                                     line=dict(color=palette[i % len(palette)], width=2.5)) for i, n in enumerate(names)]
                    + [go.Scatter(x=curves[n]["fpr"], y=curves[n]["tpr"], mode="lines", opacity=0.2, hoverinfo="skip", showlegend=False,
                                  line=dict(color=palette[i % len(palette)], width=1, dash="dot")) for i, n in enumerate(names)],
                    frames=frames)
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="#999", dash="dot"))
    fig.update_layout(height=430, plot_bgcolor="white", margin=dict(l=10, r=10, t=40, b=70),
                      xaxis=dict(title="false-positive rate", range=[0, 1]), yaxis=dict(title="true-positive rate", range=[0, 1.02]),
                      title="ROC curves drawing themselves (test nodes)",
                      updatemenus=[_play_controls(120, 60)], sliders=[dict(_slider([f"{f:.2f}" for f in fr], "FPR up to "), active=len(fr) - 1)])
    return fig


# ------------------------------------------------------------------------------------- growing bars
def bars_reveal_figure(labels, values, errors=None, colors=None, title="", xlabel="", vmin=0.0, n_frames=14, xmax=None):
    """Horizontal bars that grow from vmin to their value when played (also auto-plays on first Play click)."""
    labels = list(labels)
    values = np.array(values, float)
    colors = colors or [TEAL] * len(labels)
    steps = np.linspace(0, 1, n_frames)

    def bars(f):
        v = vmin + (values - vmin) * f
        return go.Bar(y=labels, x=v - vmin, base=vmin, orientation="h", marker=dict(color=colors),
                      error_x=dict(type="data", array=(np.array(errors) * f).tolist(), visible=True) if errors is not None else None,
                      text=[f"{x:.3f}" if f >= 0.999 else "" for x in v], textposition="outside", hoverinfo="skip")

    frames = [go.Frame(name=f"{f:.2f}", data=[bars(f)]) for f in steps]
    fig = go.Figure(data=[bars(1.0)], frames=frames)
    hi = xmax or float(max(values + (np.array(errors) if errors is not None else 0))) * 1.12
    fig.update_layout(title=title, height=max(300, 26 * len(labels) + 120), plot_bgcolor="white", showlegend=False,
                      xaxis=dict(title=xlabel, range=[vmin, hi]), yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=50, b=70),
                      updatemenus=[_play_controls(60, 40)], sliders=[dict(_slider([f"{f:.2f}" for f in steps], "reveal "), active=len(steps) - 1)])
    return fig


def score_histogram(scores, labels, threshold, title="Validation score distribution"):
    """Normal vs anomalous score histograms with a threshold line (labels = evaluation information, validation nodes only)."""
    fig = go.Figure()
    lo, hi = float(np.quantile(scores, 0.005)), float(np.quantile(scores, 0.995))
    for lab, name, col in ((0, "normal", TEAL), (1, "anomalous", ORANGE)):
        s = scores[labels == lab]
        fig.add_trace(go.Histogram(x=np.clip(s, lo, hi), name=name, marker_color=col, opacity=0.7, nbinsx=50,
                                   histnorm="probability density"))
    fig.add_vline(x=threshold, line=dict(color=NAVY, width=3, dash="dash"), annotation_text="threshold", annotation_position="top")
    fig.update_layout(barmode="overlay", height=300, plot_bgcolor="white", title=title, margin=dict(l=10, r=10, t=50, b=30),
                      transition=dict(duration=300), xaxis_title="score (clipped to the 0.5-99.5 percentile range)")
    return fig
