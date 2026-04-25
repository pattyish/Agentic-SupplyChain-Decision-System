"""Export Streamlit dashboard screenshots for README and demos.

Outputs:
- dashboard-full.png
- dashboard-overview.png
- dashboard-kpis.png

Usage:
1) Start Streamlit in a separate terminal:
   streamlit run dashboard/streamlit_app.py
2) Run this script:
   python scripts/export_dashboard_images.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


def _combined_clip_from_boxes(boxes: list[dict]) -> dict | None:
    valid = [b for b in boxes if b is not None]
    if not valid:
        return None

    x = min(b["x"] for b in valid)
    y = min(b["y"] for b in valid)
    right = max(b["x"] + b["width"] for b in valid)
    bottom = max(b["y"] + b["height"] for b in valid)

    return {
        "x": max(0, x),
        "y": max(0, y),
        "width": max(1, right - x),
        "height": max(1, bottom - y),
    }


def export_dashboard_images(base_url: str, output_dir: Path, timeout_ms: int = 30000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 2400})

        page.goto(base_url, wait_until="networkidle", timeout=timeout_ms)

        # Wait until the page title block is visible.
        try:
            page.locator("text=Dashboard (Real-World Impact)").first.wait_for(timeout=timeout_ms)
        except PlaywrightTimeoutError:
            # Fallback: wait for any Streamlit block container.
            page.locator("[data-testid='block-container']").first.wait_for(timeout=timeout_ms)

        # Wait for charts to paint if present.
        try:
            page.locator("div.js-plotly-plot").first.wait_for(timeout=5000)
        except PlaywrightTimeoutError:
            pass

        page.evaluate("window.scrollTo(0, 0)")

        # 1) Full page screenshot.
        page.screenshot(path=str(output_dir / "dashboard-full.png"), full_page=True)

        # 2) Overview screenshot (main content only if available).
        block_container = page.locator("[data-testid='block-container']").first
        if block_container.count() > 0:
            block_container.screenshot(path=str(output_dir / "dashboard-overview.png"))
        else:
            page.screenshot(path=str(output_dir / "dashboard-overview.png"), full_page=False)

        # 3) KPI row screenshot by clipping all KPI cards.
        kpi_cards = page.locator("div.kpi-card")
        card_count = kpi_cards.count()

        if card_count > 0:
            boxes = [kpi_cards.nth(i).bounding_box() for i in range(card_count)]
            clip = _combined_clip_from_boxes(boxes)
            if clip is not None:
                page.screenshot(path=str(output_dir / "dashboard-kpis.png"), clip=clip)
            else:
                page.screenshot(path=str(output_dir / "dashboard-kpis.png"), full_page=False)
        else:
            page.screenshot(path=str(output_dir / "dashboard-kpis.png"), full_page=False)

        browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dashboard screenshots using Playwright.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8501",
        help="URL where Streamlit dashboard is running.",
    )
    parser.add_argument(
        "--out",
        default="reports/figures",
        help="Output directory for PNG screenshots.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="Timeout in milliseconds for page load/waits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_dashboard_images(args.url, Path(args.out), timeout_ms=args.timeout_ms)
    print(f"Saved dashboard screenshots to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
