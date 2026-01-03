import traceback
from playwright.sync_api import sync_playwright
import time

URL = "http://localhost:8501"

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        resp = page.goto(URL, timeout=20000)
        with open('/tmp/playwright_resp.txt', 'w') as f:
            f.write(str(resp))
        try:
            page.wait_for_selector("text=Medley Swimming Stroke Recognition", timeout=25000)
            page.screenshot(path="/tmp/swimming_app_initial.png", full_page=True)
            page.content()
        except Exception as e:
            with open('/tmp/playwright_wait_error.txt', 'w') as f:
                f.write(str(e))
        context.close()
        browser.close()
except Exception as e:
    with open('/tmp/playwright_exception.txt', 'w') as f:
        f.write(traceback.format_exc())
