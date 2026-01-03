from playwright.sync_api import sync_playwright
import time

URL = "http://localhost:8501"

def element_exists(page, selector, timeout=5000):
    try:
        page.wait_for_selector(selector, timeout=timeout)
        return True
    except Exception:
        return False

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto(URL)

    # Wait for main title
    page.wait_for_selector("text=Medley Swimming Stroke Recognition", timeout=10000)

    results = {}

    # Check tabs exist
    tabs = [
        "📊 Acceleration & Gyroscope",
        "📐 Orientation",
        "⚡ Motion Magnitude",
        "🎨 All Signals"
    ]

    for t in tabs:
        results[f"tab_{t}"] = element_exists(page, f"text={t}")

    # Try to locate the Process button and confirmation checkbox
    process_button = None
    confirm_checkbox = None

    # The Process button label may vary; search by partial text
    possible_button_selectors = [
        "text=🔧 Process Data & Extract Features",
        "text=Process Data & Extract Features",
        "text=Process"
    ]

    for sel in possible_button_selectors:
        if element_exists(page, sel, timeout=2000):
            process_button = sel
            break

    # Confirmation checkbox key text we added is "Confirm" or similar; search generically
    checkbox_selectors = ["text=I confirm processing", "text=Confirm", "text=confirm"]
    for sel in checkbox_selectors:
        if element_exists(page, sel, timeout=2000):
            confirm_checkbox = sel
            break

    results['process_button_found'] = bool(process_button)
    results['confirm_checkbox_found'] = bool(confirm_checkbox)

    # Take screenshot of initial state
    page.screenshot(path="/tmp/swimming_app_initial.png", full_page=True)

    # If checkbox exists, click it and screenshot
    if confirm_checkbox:
        page.click(confirm_checkbox)
        time.sleep(0.5)
        page.screenshot(path="/tmp/swimming_app_after_confirm.png", full_page=True)

    # Click process if possible
    if process_button:
        try:
            page.click(process_button)
            time.sleep(1)
            page.screenshot(path="/tmp/swimming_app_after_process_click.png", full_page=True)
            results['process_click_attempted'] = True
        except Exception as e:
            results['process_click_error'] = str(e)
            results['process_click_attempted'] = False

    # Also capture tab-specific screenshots
    for i, t in enumerate(tabs):
        # Click tab
        try:
            page.click(f"text={t}")
            time.sleep(0.5)
            page.screenshot(path=f"/tmp/swimming_tab_{i}.png", full_page=True)
            results[f"tab_{i}_screenshot"] = f"/tmp/swimming_tab_{i}.png"
        except Exception as e:
            results[f"tab_{i}_error"] = str(e)

    print(results)

    context.close()
    browser.close()
