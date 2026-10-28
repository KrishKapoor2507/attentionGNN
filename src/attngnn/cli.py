"""Command line entry points:  python -m attngnn.cli <run|smoke|benchmark> ..."""
from __future__ import annotations

import argparse
import sys
import time
import warnings

from . import ARTIFACTS, RUNS_DIR
from . import experiments as E

warnings.filterwarnings("ignore")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="attngnn")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one run: dataset x variant x seed")
    r.add_argument("--dataset", required=True)
    r.add_argument("--variant", required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tag", default="bench")
    r.add_argument("--config", default=None)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--force", action="store_true")
    r.add_argument("--data-arg", action="append", default=[], metavar="K=V",
                   help="override dataset generator argument, e.g. struct_cliques=3 (Cora injection)")

    se = sub.add_parser("sensitivity", help="Cora injection-rate sensitivity (low and high rate, anomaly-only models)")
    se.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])

    s = sub.add_parser("smoke", help="fast end-to-end check on injected Cora (NOT a benchmark)")
    s.add_argument("--config", default=None)

    b = sub.add_parser("benchmark", help="all datasets x variants x seeds (resumable)")
    b.add_argument("--config", default=None)
    b.add_argument("--datasets", nargs="*", default=None)
    b.add_argument("--variants", nargs="*", default=None)
    b.add_argument("--seeds", nargs="*", type=int, default=None)
    b.add_argument("--force", action="store_true")

    a = ap.parse_args(argv)
    cmdline = "python -m attngnn.cli " + " ".join(sys.argv[1:] if argv is None else argv)

    if a.cmd == "run":
        cfg = E.load_config(a.config, smoke=a.smoke)
        for kv in a.data_arg:
            k, v = kv.split("=")
            cfg["datasets"][a.dataset].setdefault("data_args", {})[k] = int(v)
        out = ARTIFACTS / "smoke" if a.smoke else RUNS_DIR
        E.run_one(a.dataset, a.variant, a.seed, cfg, tag="smoke" if a.smoke else a.tag, out_root=out,
                  command=cmdline, force=a.force)
    elif a.cmd == "sensitivity":
        settings = {"inj_low": {"struct_cliques": 3, "n_contextual": 30}, "inj_high": {"struct_cliques": 14, "n_contextual": 140}}
        t0 = time.time()
        for tag, da in settings.items():
            cfg = E.load_config(None)
            cfg["datasets"]["cora_injected"]["data_args"].update(da)
            for sd in a.seeds:
                for var in ("gat_anom", "gcn_anom"):
                    E.run_one("cora_injected", var, sd, cfg, tag=tag, out_root=RUNS_DIR, command=cmdline)
        print(f"sensitivity finished in {time.time() - t0:.0f}s")
    elif a.cmd == "smoke":
        cfg = E.load_config(a.config, smoke=True)
        out = ARTIFACTS / "smoke"
        t0 = time.time()
        for var in ("gat_joint", "gcn_anom", "iforest_attr", "centrality"):
            E.run_one("cora_injected", var, 0, cfg, tag="smoke", out_root=out, command=cmdline, force=True)
        print(f"\nSMOKE TEST PASSED in {time.time() - t0:.0f}s. Outputs: {out}")
        print("(smoke runs use tiny budgets; their metrics are NOT reportable results)")
    elif a.cmd == "benchmark":
        cfg = E.load_config(a.config)
        datasets = a.datasets or [d for d, c in cfg["datasets"].items() if not c.get("optional")]
        seeds = a.seeds if a.seeds is not None else cfg["seeds"]
        variants = (a.variants or (cfg["baselines"] + cfg["variants"]))
        total = len(datasets) * len(seeds) * len(variants)
        i = 0
        t0 = time.time()
        failures = []
        for ds in datasets:
            for sd in seeds:
                for var in variants:
                    i += 1
                    print(f"\n=== [{i}/{total}] {ds} / {var} / seed {sd}  (elapsed {time.time() - t0:.0f}s)")
                    try:
                        E.run_one(ds, var, sd, cfg, tag="bench", out_root=RUNS_DIR, command=cmdline, force=a.force)
                    except Exception as e:  # noqa: BLE001  keep going; failure recorded in FAILED.json
                        failures.append((ds, var, sd, repr(e)))
        print(f"\nbenchmark finished in {time.time() - t0:.0f}s; failures: {len(failures)}")
        for f in failures:
            print("  FAILED", f)


if __name__ == "__main__":
    main()
