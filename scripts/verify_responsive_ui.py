"""Playwright responsive viewport and operational workflow verification script."""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = Path("docs/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

VIEWPORTS = [
    {"name": "desktop_1440", "width": 1440, "height": 900},
    {"name": "laptop_1024", "width": 1024, "height": 768},
    {"name": "tablet_768", "width": 768, "height": 1024},
    {"name": "mobile_390", "width": 390, "height": 844},
]


def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        print("Chrome browser launched successfully.")

        for vp in VIEWPORTS:
            context = browser.new_context(
                viewport={"width": vp["width"], "height": vp["height"]}
            )
            page = context.new_page()
            page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
            time.sleep(1.0)

            # Check horizontal overflow
            check_js = (
                "document.documentElement.scrollWidth > "
                "document.documentElement.clientWidth"
            )
            has_overflow = page.evaluate(check_js)
            print(f"Viewport {vp['width']}x{vp['height']}: overflow={has_overflow}")

            screenshot_path = SCREENSHOT_DIR / f"{vp['name']}.png"
            page.screenshot(path=str(screenshot_path), full_page=False)
            print(f"Saved: {screenshot_path}")

            context.close()

        # Operational Workflow Verification at 1440px
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
        time.sleep(1.0)

        # 1. Click preset
        print("Clicking 'ORD Winter Snow' scenario...")
        page.click("button[data-preset='ord_storm']")
        time.sleep(0.5)

        # 2. Click submit button
        print("Clicking 'Evaluate Flight Delay Risk' button...")
        page.click("#btn-submit-predict")

        # Wait for result container to be visible
        page.wait_for_selector("#eval-active-container", state="visible", timeout=10000)
        time.sleep(1.0)
        print("Inference completed and diagnostics displayed!")

        # Capture evaluated state
        eval_path = SCREENSHOT_DIR / "evaluated_1440.png"
        page.screenshot(path=str(eval_path), full_page=False)
        print(f"Saved evaluated state: {eval_path}")

        # 3. Test Tab 4 Model Governance
        print("Switching to Model Architecture & Governance tab...")
        page.click("button[data-tab='tab-model-governance']")
        time.sleep(0.8)

        gov_path = SCREENSHOT_DIR / "governance_1440.png"
        page.screenshot(path=str(gov_path), full_page=False)
        print(f"Saved governance tab: {gov_path}")

        browser.close()
        print("All visual and workflow tests completed successfully!")


if __name__ == "__main__":
    run_verification()
