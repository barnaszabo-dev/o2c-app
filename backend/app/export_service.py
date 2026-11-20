"""HTML -> PDF with the Chrome / Chromium / Edge that is already installed on the machine (headless print-to-pdf). No extra Python packages needed.
If no browser is found the dashboard falls back to the browser's own 'Save as PDF' print dialog."""
import os, re, shutil, subprocess, tempfile
from pathlib import Path

class BrowserNotFound(RuntimeError):
    pass

_MAC = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"]
_WIN = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
_NAMES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "microsoft-edge", "msedge"]

def find_browser() -> str:
    env = os.environ.get("O2C_CHROME")                      # explicit override
    if env and Path(env).exists(): return env
    for p in _MAC + _WIN:
        if Path(p).exists(): return p
    for n in _NAMES:
        w = shutil.which(n)
        if w: return w
    raise BrowserNotFound("No Chrome / Chromium / Edge found (set O2C_CHROME to its path)")

def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "report")[:80] or "report"

def html_to_pdf(html: str, timeout: int = 120) -> bytes:
    exe = find_browser()
    html = re.sub(r"<script\b.*?</script>", "", html, flags=re.S | re.I)          # the report is static; never run scripts
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "report.html", Path(d) / "report.pdf"
        src.write_text(html, encoding="utf-8")
        cmd = [exe, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--no-pdf-header-footer", "--print-to-pdf-no-header",
               "--run-all-compositor-stages-before-draw", "--virtual-time-budget=4000", "--user-data-dir=" + str(Path(d) / "profile"), "--print-to-pdf=" + str(out), src.as_uri()]
        if hasattr(os, "geteuid") and os.geteuid() == 0: cmd.insert(1, "--no-sandbox")       # containers running as root
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if not out.exists() or out.stat().st_size < 500:
            raise RuntimeError("The browser produced no PDF: " + r.stderr.decode(errors="ignore")[-300:])
        return out.read_bytes()
