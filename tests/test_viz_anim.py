import numpy as np
import pytest

from attngnn import viz_anim as VA


def _path_graph():
    nodes = [0, 1, 2, 3]
    src = np.array([0, 1, 1, 2, 2, 3])
    dst = np.array([1, 0, 2, 1, 3, 2])
    return nodes, src, dst


def test_cascade_p_one_reaches_everything_in_order():
    nodes, src, dst = _path_graph()
    step, fired = VA.simulate_cascade(src, dst, nodes, 0, p=1.0, rng_seed=0)
    assert step == {0: 0, 1: 1, 2: 2, 3: 3}
    assert [len(f) for f in fired[:4]] == [0, 1, 1, 1]


def test_cascade_p_zero_activates_only_the_seed():
    nodes, src, dst = _path_graph()
    step, _ = VA.simulate_cascade(src, dst, nodes, 2, p=0.0)
    assert [v for v, s in step.items() if s >= 0] == [2]


def test_cascade_is_deterministic_per_seed():
    nodes, src, dst = _path_graph()
    a = VA.simulate_cascade(src, dst, nodes, 0, 0.5, 3)[0]
    b = VA.simulate_cascade(src, dst, nodes, 0, 0.5, 3)[0]
    assert a == b


def test_cascade_figure_has_one_frame_per_step_and_play_button():
    nodes, src, dst = _path_graph()
    step, fired = VA.simulate_cascade(src, dst, nodes, 0, 1.0)
    pos = {v: (v, 0) for v in nodes}
    fig = VA.cascade_figure(pos, [(0, 1), (1, 2), (2, 3)], step, fired, 0)
    assert len(fig.frames) == 4
    assert fig.layout.updatemenus[0].buttons[0].label.endswith("Play")


def test_loss_replay_and_roc_and_bars_frames():
    hist = {"loss": list(np.linspace(2, 0.5, 60)), "attr": list(np.linspace(1.5, 0.8, 60)), "struct": list(np.linspace(0.7, 0.4, 60)), "inf": [0] * 60}
    fig = VA.loss_replay_figure(hist, n_frames=10)
    assert fig is not None and len(fig.frames) >= 5 and len(fig.data) == 6     # 3 animated + 3 ghost curves; constant 'inf' series skipped
    assert list(fig.frames[0].traces) == [0, 1, 2]
    assert VA.loss_replay_figure({"loss": [1.0]}) is None
    curves = {"a": {"fpr": [0, .1, .5, 1], "tpr": [0, .6, .9, 1]}, "b": {"fpr": [0, .5, 1], "tpr": [0, .5, 1]}}
    r = VA.roc_draw_figure(curves, n_frames=6)
    assert len(r.frames) == 6 and len(r.data) == 4 and list(r.frames[0].traces) == [0, 1]
    b = VA.bars_reveal_figure(["x", "y"], [0.7, 0.8], [0.02, 0.03], title="t", n_frames=5)
    assert len(b.frames) == 5


def test_score_histogram_draws_threshold_line():
    s = np.r_[np.random.default_rng(0).normal(0, 1, 200), np.random.default_rng(1).normal(2, 1, 20)]
    y = np.r_[np.zeros(200), np.ones(20)]
    fig = VA.score_histogram(s, y, 1.0)
    assert len(fig.data) == 2 and len(fig.layout.shapes) == 1


def test_html_components_make_no_external_requests():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    import ui_theme as U
    for html in (U.hero_html("Title", "Subtitle"), U.kpi_html([("runs", 96, ""), ("auc", 0.79, "")]), U.CSS):
        low = html.lower()
        assert "http://" not in low and "https://" not in low and "<script src" not in low and "@import" not in low
    assert "prefers-reduced-motion" in U.hero_html("t", "s") and "prefers-reduced-motion" in U.CSS
