"""Tiny helper for verify_all.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def wb_labels_ok():
    import download_data as DL
    from attngnn import data as D
    b = D.load_dataset("weibo")
    return DL.sha(ROOT / "data" / "raw" / "weibo.pt.zip") == DL.PINNED["weibo"] and int(b.y.sum()) == 868
