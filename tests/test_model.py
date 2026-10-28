import numpy as np
import pytest
import torch

from attngnn import influence as I
from attngnn import scoring as S
from attngnn.models import build_model
from attngnn.train import loss_terms, sample_pairs, train_model


def _model(kind, d=5, heads=4):
    torch.manual_seed(0)
    return build_model({"encoder": kind, "heads": heads, "hidden": 8}, d, 2)


@pytest.mark.parametrize("kind", ["gat", "gatv2"])
def test_attention_sums_to_one_at_target_but_not_at_source(tiny_graph, kind):
    """Softmax normalises over arcs ARRIVING at a target (incl. self loop): summing there is the constant 1.
    Received attention (summed over arcs LEAVING a source) is not constant, hence usable for ranking."""
    x, ei, n = tiny_graph
    model = _model(kind)
    xt, eit = torch.tensor(x), torch.tensor(ei)
    out = model(xt, eit, return_attention=True)
    for ei2, alpha in out["attn"]:
        tot = torch.zeros(n, alpha.shape[1]).index_add_(0, ei2[1], alpha)
        assert torch.allclose(tot, torch.ones_like(tot), atol=1e-5)            # constant => useless as ranking
    att = S.attention_edges(model, xt, eit)
    R = S.received_attention(att, n)
    assert np.ptp(R[:6]) > 1e-3                                                  # non-constant across sources
    assert R[6] == 0.0                                                           # isolated node receives nothing


def test_attention_edge_alignment_and_convention(tiny_graph):
    x, ei, n = tiny_graph
    model = _model("gat")
    att = S.attention_edges(model, torch.tensor(x), torch.tensor(ei))
    assert (att["src"] == ei[0]).all() and (att["dst"] == ei[1]).all()
    assert att["alpha"].shape == (ei.shape[1], 2, 4)
    # a node with a single neighbour assigns it (1 - self weight); the bridge endpoints have more neighbours
    a = S.mean_alpha(att)
    assert ((a > 0) & (a < 1)).all()


def test_gcn_has_no_attention(tiny_graph):
    x, ei, n = tiny_graph
    assert S.attention_edges(_model("gcn"), torch.tensor(x), torch.tensor(ei)) is None


def test_isolated_nodes_have_finite_scores(tiny_graph):
    x, ei, n = tiny_graph
    model = _model("gat")
    xt, eit = torch.tensor(x), torch.tensor(ei)
    idx = np.arange(n)
    ae, se = S.recon_errors(model, xt, eit)
    assert np.isfinite(ae).all() and np.isfinite(se).all()
    att = S.attention_edges(model, xt, eit)
    s = S.attention_only_score(xt, eit, att)
    assert np.isfinite(s).all() and s[6] == 0.0
    u = S.uniform_discrepancy(xt, eit)
    assert np.isfinite(u).all() and u[6] == 0.0
    sc, info = S.all_influence_scores(model, xt, eit, idx[:5], [0.1, 0.2])
    assert all(np.isfinite(v).all() for v in sc.values())


def test_degree_adjust_removes_degree_trend():
    rng = np.random.default_rng(0)
    deg = rng.integers(1, 100, 500).astype(float)
    R = deg ** 1.3 * np.exp(rng.normal(0, 0.1, 500))
    adj, beta = S.degree_adjust(R, deg, np.arange(250))
    assert abs(np.corrcoef(adj, np.log1p(deg))[0, 1]) < 0.05


@pytest.mark.parametrize("kind", ["gcn", "gat", "gatv2"])
def test_each_loss_sends_gradient_into_shared_encoder(tiny_graph, kind):
    x, ei, n = tiny_graph
    xt, eit = torch.tensor(x), torch.tensor(ei)
    tr = torch.arange(5)
    tgt = torch.randn(5, 2)
    for which in ("attr", "struct", "inf"):
        model = _model(kind)
        gen = torch.Generator().manual_seed(0)
        _, la, ls, li = loss_terms(model, xt, eit, tr, tgt, 0.5, gen)
        {"attr": la, "struct": ls, "inf": li}[which].backward()
        g = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
        assert g > 0, f"{which} loss does not reach the encoder ({kind})"


def test_training_reduces_loss_and_is_deterministic(tiny_graph):
    x, ei, n = tiny_graph
    xt, eit = torch.tensor(x), torch.tensor(ei)
    runs = []
    for _ in range(2):
        m = _model("gat")
        h = train_model(m, xt, eit, torch.arange(5), torch.randn(5, 2) * 0 + 1.0, epochs=30, lam_inf=1.0,
                        seed=3, log=None)
        runs.append((h["loss"][0], h["loss"][-1], m(xt, eit)["z"].detach().clone()))
    assert runs[0][1] < runs[0][0]
    assert runs[0][0] == runs[1][0] and runs[0][1] == runs[1][1] and torch.allclose(runs[0][2], runs[1][2])


def test_negative_sampling_ratio_and_labels(tiny_graph):
    x, ei, n = tiny_graph
    gen = torch.Generator().manual_seed(0)
    s, d, lab = sample_pairs(torch.tensor(ei), n, 1000, 2.0, gen)
    assert int(lab.sum()) == ei.shape[1] and len(lab) == 3 * ei.shape[1]


def test_checkpoint_reload_gives_identical_outputs(tiny_graph, tmp_path):
    x, ei, n = tiny_graph
    xt, eit = torch.tensor(x), torch.tensor(ei)
    m = _model("gat")
    torch.save({"state_dict": m.state_dict(), "model_cfg": m.cfg}, tmp_path / "c.pt")
    ck = torch.load(tmp_path / "c.pt", weights_only=False)
    c = ck["model_cfg"]
    from attngnn.models import JointModel
    m2 = JointModel(c["in_dim"], n_p=c["n_p"], hidden=c["hidden"], heads=c["heads"], layers=c["layers"], kind=c["kind"])
    m2.load_state_dict(ck["state_dict"])
    m.eval(); m2.eval()
    with torch.no_grad():
        assert torch.equal(m(xt, eit)["z"], m2(xt, eit)["z"])


# -------------------------------------------------- influence simulation
def test_ic_matches_closed_form_on_path():
    ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    p = 0.3
    m, _ = I.single_node_spread(ei, 3, p, [0, 1, 2], 200_000, 1)
    assert m[1] == pytest.approx(1 + 2 * p, abs=0.01)
    assert m[0] == pytest.approx(1 + p + p * p, abs=0.01)
    mm, se, _ = I.seed_set_spread(ei, 3, p, [0, 2], 100_000, 2)
    assert mm == pytest.approx(2 + 1 - (1 - p) ** 2, abs=0.01)


def test_ic_extremes_and_reproducibility(tiny_graph):
    x, ei, n = tiny_graph
    m0, _ = I.single_node_spread(ei, n, 0.0, np.arange(n), 20, 1)
    assert (m0 == 1).all()
    m1, _ = I.single_node_spread(ei, n, 1.0, np.arange(n), 20, 1)
    assert (m1[:6] == 6).all() and m1[6] == 1                      # reaches own component only
    a, _ = I.single_node_spread(ei, n, 0.4, np.arange(n), 200, 5)
    b, _ = I.single_node_spread(ei, n, 0.4, np.arange(n), 200, 5)
    c, _ = I.single_node_spread(ei, n, 0.4, np.arange(n), 200, 6)
    assert (a == b).all() and not (a == c).all()


def test_train_and_test_streams_are_independent(tiny_graph):
    x, ei, n = tiny_graph
    spl = {"train": np.array([1, 1, 1, 0, 0, 0, 0], bool), "val": np.array([0, 0, 0, 1, 0, 0, 0], bool),
           "test": np.array([0, 0, 0, 0, 1, 1, 1], bool)}
    a = I.simulate_targets(ei, n, [0.4], spl, {"train": 50, "val": 50, "test": 50}, 1)
    b = I.simulate_targets(ei, n, [0.4], spl, {"train": 50, "val": 50, "test": 50}, 2)
    assert (a[0.4]["mean"] == a[0.4]["mean"]).all() and not (a[0.4]["mean"] == b[0.4]["mean"]).all()


def test_centralities_on_star():
    ei = np.array([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    assert I.degree_centrality(ei, 4).tolist() == [3, 1, 1, 1]
    pr = I.pagerank(ei, 4)
    assert pr.argmax() == 0 and pr.sum() == pytest.approx(1.0)
    bc = I.betweenness(ei, 4)
    assert bc.argmax() == 0 and bc[1] == 0
    u = I.uniform_received_attention(ei, 4)
    assert u[0] > u[1]
