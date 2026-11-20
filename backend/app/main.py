from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.routes import router

ROOT = Path(__file__).resolve().parents[2]          # project folder (index.html, data/, assets/)

app = FastAPI(title="MOL O2C Lead Time Analytics - data & ML backend", version="0.1.0")
# Only needed when the dashboard is opened from another local server (e.g. python -m http.server 5500). Loopback origins only.
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1?\])(:\d+)?", allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

@app.get("/health")
def health(): return {"status": "ok"}

# The backend also serves the dashboard itself: open http://localhost:8000 - same origin, no browser cross-origin blocking, one terminal only.
@app.get("/", include_in_schema=False)
def index(): return FileResponse(ROOT / "index.html", headers={"Cache-Control": "no-store"})

for _d in ("data", "assets"):
    if (ROOT / _d).is_dir(): app.mount("/" + _d, StaticFiles(directory=ROOT / _d), name=_d)
