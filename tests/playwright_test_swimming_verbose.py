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
    print("Navigating to", URL)
    resp = page.goto(URL, timeout=15000)
    print("Goto response:", resp.status if resp else None)

    # Wait longer for main title
    try:
        page.wait_for_selector("text=Medley Swimming Stroke Recognition", timeout=20000)
        print('Main title found')
    except Exception as e:
        print('Main title not found:', e)

    results = {}

    # Save HTML snapshot
    html = page.content()
    with open('/tmp/swimming_page_snapshot.html', 'w') as f:
        f.write(html)
    print('Saved HTML snapshot to /tmp/swimming_page_snapshot.html')

    # Take a screenshot of the page
    page.screenshot(path="/tmp/swimming_app_initial.png", full_page=True)
    print('Saved screenshot /tmp/swimming_app_initial.png')

    # Check tabs exist
    tabs = [
        "📊 Acceleration & Gyroscope",
        "📐 Orientation",
        "⚡ Motion Magnitude",
        "🎨 All Signals"
    ]

    for t in tabs:
        present = element_exists(page, f"text={t}", timeout=2000)
        print(f"Tab '{t}' present: {present}")
        results[f"tab_{t}"] = present

    # Try to locate the Process button and confirmation checkbox
    process_button = None
    confirm_checkbox = None

    possible_button_selectors = [
        "text=🔧 Process Data & Extract Features",
        "text=Process Data & Extract Features",
        "text=Process"
    ]

    for sel in possible_button_selectors:
        if element_exists(page, sel, timeout=2000):
            process_button = sel
            print('Found process button selector:', sel)
            break

    checkbox_selectors = ["text=I confirm processing", "text=Confirm", "text=confirm", "text=I confirm"]
    for sel in checkbox_selectors:
        if element_exists(page, sel, timeout=2000):
            confirm_checkbox = sel
            print('Found checkbox selector:', sel)
            break

    results['process_button_found'] = bool(process_button)
    results['confirm_checkbox_found'] = bool(confirm_checkbox)

    # If checkbox exists, click it
    if confirm_checkbox:
        page.click(confirm_checkbox)
        time.sleep(0.5)
        page.screenshot(path="/tmp/swimming_app_after_confirm.png", full_page=True)
        print('Clicked checkbox and saved screenshot /tmp/swimming_app_after_confirm.png')

    # Click process if possible
    if process_button:
        try:
            page.click(process_button)
            time.sleep(1)
            page.screenshot(path="/tmp/swimming_app_after_process_click.png", full_page=True)
            results['process_click_attempted'] = True
            print('Clicked process button and saved screenshot /tmp/swimming_app_after_process_click.png')
        except Exception as e:
            results['process_click_error'] = str(e)
            results['process_click_attempted'] = False
            print('Error clicking process button:', e)

    # Capture each tab screenshot
    for i, t in enumerate(tabs):
        try:
            page.click(f"text={t}")
            time.sleep(0.5)
            page.screenshot(path=f"/tmp/swimming_tab_{i}.png", full_page=True)
            results[f"tab_{i}_screenshot"] = f"/tmp/swimming_tab_{i}.png"
            print(f'Saved screenshot for tab {i} -> /tmp/swimming_tab_{i}.png')
        except Exception as e:
            results[f"tab_{i}_error"] = str(e)
            print(f'Error capturing tab {i}:', e)

    print('Results:', results)

    context.close()
    browser.close()
