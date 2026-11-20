from fastapi import APIRouter, HTTPException
from app.core.dataset import load_df, load_schema
from app.core.filters import FilterSpec, apply_filters, filter_options, MIN_ROWS_FOR_MODELLING
from typing import Optional
from pydantic import BaseModel
from app.ml.features.leakage import allowed
from app.ml.stats import eda as eda_mod, correlation as corr_mod
from app.ml.stats.common import clean
from app.ml.features.catalog import catalog
from app.ml import regression_service as reg
from app.ml import forest_service as forest
from app.ml import interpretation_service as interp
from app.ml import prediction_service as predsvc
from app.ml import evaluation_service as evalsvc
from fastapi.responses import Response
from app import export_service as export

router = APIRouter(prefix="/api/v1")

# ---------- implemented now: data access & contracts (no modelling) ----------
@router.get("/meta/schema")
def schema(): return load_schema()

@router.get("/meta/filters")
def filters(): return filter_options(load_df())

@router.post("/data/summary")
def summary(spec: FilterSpec):
    """Parity endpoint with the dashboard KPIs + modelling-readiness warning."""
    R = apply_filters(load_df(), spec); h = R["h_po_to_oc"]
    if spec.exclude_negative_durations: h = h[h >= 0]
    n = len(R); ok = n >= MIN_ROWS_FOR_MODELLING
    return dict(n_rows=n, n_documents=int(R["d_sales_document_base"].nunique()),
                median_po_to_oc_hours=None if h.dropna().empty else round(float(h.median()), 2),
                within_48h_pct=None if n == 0 else round(float(R["d_within_48h_po_oc"].mean() * 100), 2),
                sufficient_for_modelling=ok,
                warning=None if ok else f"Only {n} observations: too few for reliable modelling (minimum {MIN_ROWS_FOR_MODELLING}).")

@router.get("/ml/prediction-points")
def points(): return load_schema()["prediction_points"]

@router.get("/ml/features/{point}")
def features(point: str, include_review: bool = False):
    if point not in load_schema()["prediction_points"]: raise HTTPException(404, "unknown prediction point")
    return allowed(point, include_review)

class _Req(BaseModel):
    filters: FilterSpec = FilterSpec()
    measure: str = "po_to_oc"
    lang: str = "en"

class EdaRequest(_Req):
    bins: int = 30

class CorrRequest(_Req):
    variables: Optional[list[str]] = None
    pair: Optional[list[str]] = None

def _prep(req):
    if req.measure not in {m["id"] for m in load_schema()["measures"]}: raise HTTPException(404, "unknown measure")
    R = apply_filters(load_df(), req.filters)
    if len(R) < 10: raise HTTPException(422, f"Only {len(R)} observations in the selection")
    return R, req.filters.exclude_negative_durations

@router.post("/ml/eda")
def eda(req: EdaRequest):
    R, neg = _prep(req); out = eda_mod.run(R, req.measure, neg, max(5, min(req.bins, 80)), req.lang)
    out["sufficient_for_modelling"] = len(R) >= MIN_ROWS_FOR_MODELLING; return out

@router.post("/ml/correlation")
def correlation(req: CorrRequest):
    R, neg = _prep(req); out = corr_mod.run(R, req.measure, neg, req.variables, req.pair, req.lang)
    out["sufficient_for_modelling"] = len(R) >= MIN_ROWS_FOR_MODELLING; return out

class RegRequest(_Req):
    prediction_point: str = "at_sales_order"
    features: Optional[list[str]] = None
    include_review: bool = False
    log_target: bool = False
    test_fraction: float = 0.2
    x: Optional[str] = None

@router.get("/ml/candidates/{point}")
def candidates(point: str, include_review: bool = False, target: Optional[str] = None):
    if point not in load_schema()["prediction_points"]: raise HTTPException(404, "unknown prediction point")
    return catalog(point, include_review, target)

class RFRequest(RegRequest):
    n_trees: int = 150
    max_depth: Optional[int] = 12
    min_samples_leaf: int = 20

@router.post("/ml/regression/random-forest")
def regression_forest(req: RFRequest):
    R, neg = _reg(req); out = forest.run_forest(R, req, neg); out["sufficient_for_modelling"] = len(R) >= MIN_ROWS_FOR_MODELLING; return out

class InterpRequest(RFRequest):
    order: Optional[str] = None

@router.post("/ml/interpretation")
def interpretation(req: InterpRequest):
    R, neg = _reg(req); out = interp.run_interpretation(R, req, neg); out["sufficient_for_modelling"] = len(R) >= MIN_ROWS_FOR_MODELLING; return out

class PredRequest(RegRequest):
    model: str = "random_forest"
    level: float = 0.8
    inputs: dict = {}

@router.post("/ml/prediction")
def prediction(req: PredRequest):
    R, neg = _reg(req); out = predsvc.run_prediction(R, req, neg); out["sufficient_for_modelling"] = len(R) >= MIN_ROWS_FOR_MODELLING; return out

@router.post("/ml/evaluation/compare")
def evaluation_compare(req: RegRequest):
    R, neg = _reg(req); out = evalsvc.run_comparison(R, req, neg); out["sufficient_for_modelling"] = len(R) >= MIN_ROWS_FOR_MODELLING; return out

class ExportRequest(BaseModel):
    html: str
    filename: str = "MOL_O2C_report"

@router.get("/export/status")
def export_status():
    try: return dict(direct_pdf=True, browser=export.find_browser())
    except export.BrowserNotFound: return dict(direct_pdf=False, browser=None)

@router.post("/export/pdf")
def export_pdf(req: ExportRequest):
    """Renders the report HTML (built by the dashboard from the current filters) to a PDF download."""
    try: pdf = export.html_to_pdf(req.html)
    except export.BrowserNotFound as e: raise HTTPException(501, detail="browser_not_found: " + str(e))
    except Exception as e: raise HTTPException(500, detail=str(e)[:400])
    return Response(content=pdf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="%s.pdf"' % export.safe_name(req.filename)})

def _reg(req):
    R, neg = _prep(req)
    if req.prediction_point not in load_schema()["prediction_points"]: raise HTTPException(404, "unknown prediction point")
    return R, neg

@router.post("/ml/regression/linear")
def regression_linear(req: RegRequest):
    R, neg = _reg(req); out = reg.run_linear(R, req, neg); out["sufficient_for_modelling"] = len(R) >= MIN_ROWS_FOR_MODELLING; return out

@router.post("/ml/regression/multiple")
def regression_multiple(req: RegRequest):
    R, neg = _reg(req); out = reg.run_multiple(R, req, neg); out["sufficient_for_modelling"] = len(R) >= MIN_ROWS_FOR_MODELLING; return out

# ---------- planned ML endpoints: contract only (HTTP 501) ----------
PLANNED = {}   # every planned module is implemented; future modules can be announced here (HTTP 501 until built)
def _make(p, d):
    def endpoint(filters: FilterSpec = FilterSpec()):
        raise HTTPException(501, detail=dict(status="not_implemented", endpoint=p, planned=d, request="FilterSpec + analysis parameters (docs/ML_ARCHITECTURE.md)"))
    return endpoint
for path, desc in PLANNED.items():
    router.add_api_route(f"/ml/{path}", _make(path, desc), methods=["POST"], summary=f"[planned] {desc}")
