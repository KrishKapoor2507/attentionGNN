"""UI/backend integration: run the Streamlit script headlessly against a tiny saved run."""
import sys
from pathlib import Path

import pytest

from attngnn import experiments as E

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("apprun")
    cfg = E.load_config(smoke=True)
    E.run_one("cora_injected", "gat_joint", 0, cfg, tag="bench", out_root=out, force=True)
    return out


PAGES = ["Overview", "Data & graph summary", "Dataset authenticity", "Train / evaluate", "Influence ranking", "Anomaly scores",
         "Graph explorer & node inspector", "Results: comparison & ablation", "Demos", "Exports & documents"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(runs_dir, page, monkeypatch):
    monkeypatch.setenv("ATTNGNN_RUNS_DIR", str(runs_dir))
    import attngnn.app_backend as B
    monkeypatch.setattr(B, "run_table", lambda root=runs_dir: B.pd.DataFrame(
        [{"run_id": d.name, "dataset": "cora_injected", "variant": "gat_joint", "seed": 0, "tag": "bench",
          "train_s": 1.0, "has_checkpoint": True, "path": str(d)} for d in runs_dir.iterdir()]))
    monkeypatch.setattr(B.inf, "list_runs", lambda root=runs_dir, **k: [d for d in runs_dir.iterdir() if (d / "metrics.json").exists()])
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_csv_import_error_is_shown_not_raised(tmp_path):
    from attngnn import data as D
    p = tmp_path / "e.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(D.CSVSchemaError):
        D.load_csv_graph(p)


def test_job_launcher_creates_log(tmp_path, monkeypatch):
    import attngnn.app_backend as B
    monkeypatch.setattr(B, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(B.sys, "executable", sys.executable)
    jid, p = B.start_job("cora_injected", "iforest_attr", 0, smoke=True)
    p.wait(timeout=300)
    state, log = B.job_status(jid)
    assert state in ("finished", "stopped (no completion line)", "failed") and "run" in log


@pytest.mark.parametrize("mode", ["Known anomaly", "High-spread node", "Explanation test", "Saved model reload", "Training replay", "ROC drawing"])
def test_every_demo_mode_renders(runs_dir, mode, monkeypatch):
    import attngnn.app_backend as B
    monkeypatch.setattr(B, "run_table", lambda root=runs_dir: B.pd.DataFrame(
        [{"run_id": d.name, "dataset": "cora_injected", "variant": "gat_joint", "seed": 0, "tag": "bench",
          "train_s": 1.0, "has_checkpoint": True, "path": str(d)} for d in runs_dir.iterdir()]))
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("Demos").run()
    at.segmented_control(key="demo_mode").set_value(mode).run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_graph_explorer_includes_cascade_animation(runs_dir, monkeypatch):
    import attngnn.app_backend as B
    monkeypatch.setattr(B, "run_table", lambda root=runs_dir: B.pd.DataFrame(
        [{"run_id": d.name, "dataset": "cora_injected", "variant": "gat_joint", "seed": 0, "tag": "bench",
          "train_s": 1.0, "has_checkpoint": True, "path": str(d)} for d in runs_dir.iterdir()]))
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("Graph explorer & node inspector").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    figs = [e for e in at.get("plotly_chart")]
    assert len(figs) >= 2        # sampled graph + cascade animation
