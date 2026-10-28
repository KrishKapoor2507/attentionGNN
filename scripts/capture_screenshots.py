"""Capture real screenshots of the running dashboard (Edge via Playwright; nothing is downloaded).

    python scripts/capture_screenshots.py [outdir]     -> docs/screenshots/NN_name.png (default)
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
PORT = 8599
SCROLL = "(()=>{const m=document.querySelector('[data-testid=stMain]')||document.querySelector('section.main');if(m)m.scrollTo(0,%d)})()"
# (sidebar label, file name, optional text to click after navigating, scroll offset)
SHOTS = [("Overview", "01_overview", None, 0), ("Data & graph summary", "02_data_summary", None, 0),
         ("Dataset authenticity", "03_dataset_authenticity", None, 0), ("Train / evaluate", "04_train_evaluate", None, 0),
         ("Influence ranking", "05_influence_ranking", None, 0), ("Anomaly scores", "06_anomaly_scores", None, 0),
         ("Graph explorer & node inspector", "07_graph_explorer", None, 0), ("Graph explorer & node inspector", "08_cascade_animation", None, 6000),
         ("Results: comparison & ablation", "09_results", None, 0), ("Demos", "10_demos_known_anomaly", None, 0),
         ("Demos", "11_demos_training_replay", "Training replay", 0), ("Demos", "12_demos_roc_drawing", "ROC drawing", 0),
         ("Exports & documents", "13_exports", None, 0)]


def main():
    from playwright.sync_api import sync_playwright

    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", str(PORT),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        else:
            raise RuntimeError("dashboard did not start")
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1500})
            page.goto(f"http://localhost:{PORT}")
            page.wait_for_selector(".st-key-nav", timeout=60000)
            for label, name, click, scroll in SHOTS:
                page.locator(".st-key-nav label", has_text=label).first.click()
                page.wait_for_timeout(2500)
                if click:
                    page.get_by_text(click, exact=True).first.click()
                    page.wait_for_timeout(2500)
                try:
                    page.wait_for_selector("[data-testid=stStatusWidget]", state="detached", timeout=90000)
                except Exception:  # noqa: BLE001
                    pass
                if scroll:
                    page.evaluate(SCROLL % scroll)
                page.wait_for_timeout(3200 if name in ("01_overview", "08_cascade_animation") else 2200)
                page.screenshot(path=str(OUT / f"{name}.png"), full_page=False)
                print("captured", name, flush=True)
            browser.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
