"""Capture screenshots of the running app (streamlit run app.py) into screenshots/."""
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(exist_ok=True)
URL = "http://localhost:8501"


def pick_sample(page, name):
    page.get_by_role("combobox", name="Load sample applicant (real test-set rows)").click()
    page.get_by_role("option", name=name).click()
    page.wait_for_function(f"document.body.innerText.includes('Loaded sample: {name}')", timeout=90000)
    page.wait_for_timeout(3000)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 2100}, device_scale_factor=1)
    page.goto(URL); page.wait_for_selector("text=Applicant details", timeout=60000); page.wait_for_timeout(2000)
    page.screenshot(path=OUT / "0_home.png", full_page=True)

    for i, name in enumerate(["Approved applicant", "Rejected applicant", "Borderline applicant"], 1):
        pick_sample(page, name)
        page.get_by_text("Sources (").first.click() if name == "Rejected applicant" else None
        page.wait_for_timeout(1000)
        page.screenshot(path=OUT / f"{i}_{name.split()[0].lower()}_result.png", full_page=True)

    page.get_by_role("tab", name="Model performance").click(); page.wait_for_timeout(2500)
    page.screenshot(path=OUT / "4_model_performance.png", full_page=True)
    page.get_by_role("tab", name="About").click(); page.wait_for_timeout(3000)
    page.screenshot(path=OUT / "5_about.png", full_page=True)
    browser.close()
print("saved to", OUT)
