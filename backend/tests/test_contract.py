import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app
c = TestClient(app)

def test_parity_with_dashboard():
    s = c.post("/api/v1/data/summary", json={}).json()
    assert s["n_rows"] == 14601 and abs(s["median_po_to_oc_hours"] - 80.1) < .06 and round(s["within_48h_pct"]) == 40
    i = c.post("/api/v1/data/summary", json={"allocation_group": ["Initial"]}).json()
    assert i["n_rows"] == 11929 and round(i["within_48h_pct"]) == 38
    t1 = c.post("/api/v1/data/summary", json={"allocation_group": ["Initial"], "customer_category": ["T1"]}).json()
    assert t1["n_rows"] == 4825 and round(t1["within_48h_pct"]) == 31

def test_small_selection_warns():
    s = c.post("/api/v1/data/summary", json={"region": ["R04"], "customer_category": ["T2"], "early": ["early"]}).json()
    assert s["sufficient_for_modelling"] or s["warning"]

def test_leakage_guard():
    f = c.get("/api/v1/ml/features/at_purchase_order").json()
    assert "Loaded volume" not in f["allowed_features"] and "Order Confirmation Date" not in f["allowed_features"]
    assert "Region" in f["allowed_features"] and "po_to_oc" in f["valid_targets"] and "po_to_so_sap" in f["valid_targets"]
    g = c.get("/api/v1/ml/features/at_scheduling_match").json()
    assert "h_po_to_so_sap" in g["allowed_features"] and "po_to_so_sap" not in g["valid_targets"]

def test_options_and_planned_endpoints():
    assert "R01" in c.get("/api/v1/meta/filters").json()["region"]
    assert not any("[planned]" in str(v) for v in c.get("/openapi.json").json()["paths"].values())      # every planned module is implemented now

def test_eda_and_correlation():
    e = c.post("/api/v1/ml/eda", json={}).json()
    assert e["n_rows"] == 14601 and abs(e["stats"]["median"] - 80.13) < .05 and e["quality"]["duplicate_sales_documents"] == 0
    g = {x["group"]: x for x in e["groups"]["allocation_group"]}
    assert abs(g["Initial"]["median"] - 81.6) < .1 and abs(g["Additional demand"]["median"] - 39.5) < .1 and len(e["insights"]) >= 3
    r = c.post("/api/v1/ml/correlation", json={"lang": "hu"}).json()
    assert r["cells"]["po_to_oc|po_to_oc"]["pearson"] == 1.0 and r["association"] and r["insights"][-1].startswith("Ezek")
    assert c.post("/api/v1/ml/eda", json={"measure": "nope"}).status_code == 404
    assert c.post("/api/v1/ml/eda", json={"filters": {"region": ["XX"]}}).status_code == 422

def test_regression_endpoints():
    m = c.post("/api/v1/ml/regression/multiple", json={}).json()
    assert m["split"]["n_train"] + m["split"]["n_test"] == 14601 and m["split"]["cutoff"] > "2026-04"
    assert max(u["vif"] for u in m["used_features"]) < 10 and any("collinear" in r["reason"] for r in m["rejected"])
    assert "Loaded volume (actual)" not in [u["label"] for u in m["used_features"]]          # leakage guard
    assert m["baseline_median"]["mae"] > 0 and m["test"]["rmse"] >= m["test"]["mae"] and len(m["coefficients"]) > 10
    bad = c.post("/api/v1/ml/regression/multiple", json={"features": ["loaded_volume", "region"]}).json()
    assert any(r["id"] == "loaded_volume" for r in bad["rejected"]) and "loaded_volume" not in [u["id"] for u in bad["used_features"]]
    l = c.post("/api/v1/ml/regression/linear", json={"x": "loaded_volume"}).json()
    assert any(i.startswith("Warning") for i in l["insights"]) and len(l["points"]) > 100
    assert c.post("/api/v1/ml/regression/linear", json={"x": "po_to_oc"}).status_code == 404
    cand = c.get("/api/v1/ml/candidates/at_purchase_order").json()
    assert next(x for x in cand if x["id"] == "doc_type")["allowed"] is False and next(x for x in cand if x["id"] == "region")["allowed"] is True

def test_random_forest():
    r = c.post("/api/v1/ml/regression/random-forest", json={"n_trees": 20, "max_depth": 6}).json()
    assert [x["model"] for x in r["comparison"]] == ["baseline_median", "multiple_linear", "random_forest"] and r["params"]["n_trees"] == 20
    assert r["importance"] and all(i["label"] != "Loaded volume (actual)" for i in r["importance"]) and r["split"]["n_train"] + r["split"]["n_test"] == 14601
    assert "mae" in r["test"] and len(r["cv"]["folds"]) == 5

def test_interpretation():
    r = c.post("/api/v1/ml/interpretation", json={"n_trees": 20, "max_depth": 6, "order": "401625043#20"}).json()
    assert r["importance"] and all("perm_mean" in i for i in r["importance"]) and r["coverage"]
    assert all(x["label"] != "Loaded volume (actual)" for x in r["importance"])                       # leakage guard
    assert any(x["id"] == "loaded_volume" and x["status"] == "excluded" for x in r["coverage"])
    assert (r["shap"]["available"] and r["local"]["document"] == "401625043#20" and r["categories"]) or not r["shap"]["available"]
    assert len(r["insights"]) >= 4 and c.post("/api/v1/ml/interpretation", json={"order": "nope"}).json()["local"]["error"] in ("not_found", "shap_unavailable")

def test_prediction():
    r = c.post("/api/v1/ml/prediction", json={"prediction_point": "at_purchase_order", "inputs": {"region": "R02"}}).json()
    p = r["prediction"]; assert p["lower"] <= p["value"] <= p["upper"] and r["form"]["values"]["region"] == "R02"
    ids = [f["id"] for f in r["form"]["fields"]]; assert "loaded_volume" not in ids and "doc_type" not in ids              # leakage guard at PO time
    cf = r["confidence"]; assert 0.5 < cf["coverage_test"] <= 1 and r["split"]["n_calibration"] > 100 and len(r["insights"]) >= 6
    assert c.post("/api/v1/ml/prediction", json={"measure": "po_to_so_sap", "prediction_point": "at_sales_order"}).status_code == 422   # target already over
    o = c.post("/api/v1/ml/prediction", json={"model": "ols", "prediction_point": "at_sales_order"}).json(); assert o["prediction"]["value"] > 0 and o["contribution_method"] == "occlusion"

def test_evaluation_consistent_with_other_pages():
    e = c.post("/api/v1/ml/evaluation/compare", json={}).json(); M = {m["id"]: m for m in e["models"]}
    assert {"median_baseline", "region_median", "linear_single", "multiple_linear", "random_forest", "gradient_boosting", "gradient_boosting_median"} <= set(M)
    assert sorted(m["rank_test"] for m in e["models"]) == list(range(1, len(M) + 1)) and len(e["rolling"]) >= 2 and e["tests_vs_baseline"] and e["by_region"]
    mr = c.post("/api/v1/ml/regression/multiple", json={}).json(); assert abs(M["multiple_linear"]["test"]["mae"] - mr["test"]["mae"]) < .01       # same numbers as the regression page
    rf = c.post("/api/v1/ml/regression/random-forest", json={}).json(); assert abs(M["random_forest"]["test"]["mae"] - rf["test"]["mae"]) < .01   # same numbers as the forest page


def test_export_pdf_pipeline(tmp_path, monkeypatch):
    """The Chrome call is replaced by a fake browser that writes a PDF to the --print-to-pdf path: tests the HTTP + temp-file plumbing."""
    fake = tmp_path / "fakechrome.sh"
    fake.write_text("#!/bin/sh\nfor a in \"$@\"; do case \"$a\" in --print-to-pdf=*) f=\"${a#--print-to-pdf=}\";; esac; done\nprintf '%%PDF-1.4\\n' > \"$f\"\nhead -c 800 /dev/zero >> \"$f\"\n")
    fake.chmod(0o755); monkeypatch.setenv("O2C_CHROME", str(fake))
    st = c.get("/api/v1/export/status").json(); assert st["direct_pdf"] is True
    r = c.post("/api/v1/export/pdf", json={"html": "<html><script>alert(1)</script><body>x</body></html>", "filename": "MOL riport/2026"})
    assert r.status_code == 200 and r.content.startswith(b"%PDF") and r.headers["content-type"] == "application/pdf"
    assert 'filename="MOL_riport_2026.pdf"' in r.headers["content-disposition"]

def test_export_without_browser(monkeypatch):
    from app import export_service
    def none(): raise export_service.BrowserNotFound("none")
    monkeypatch.setattr(export_service, "find_browser", none)
    assert c.get("/api/v1/export/status").json()["direct_pdf"] is False
    assert c.post("/api/v1/export/pdf", json={"html": "<p>x</p>"}).status_code == 501


def test_evaluation_with_filter_and_rolling_encoding():
    """Regression test: a filtered selection whose months have different category levels must not break the rolling validation."""
    r = c.post("/api/v1/ml/evaluation/compare", json={"filters": {"region": ["R02"]}})
    assert r.status_code == 200 and len(r.json()["rolling"]) >= 1


def test_backend_serves_the_dashboard_and_cors():
    r = c.get("/"); assert r.status_code == 200 and "MOL | O2C Lead Time Analytics" in r.text and r.headers["cache-control"] == "no-store"
    assert c.get("/data/orders.json").status_code == 200 and c.get("/assets/molgroup.png").status_code == 200
    for origin in ("http://localhost:5500", "http://127.0.0.1:5500", "http://[::1]:5500", "http://0.0.0.0:5500", "http://localhost:8000"):
        assert c.get("/api/v1/export/status", headers={"Origin": origin}).headers.get("access-control-allow-origin") == origin
    assert "access-control-allow-origin" not in c.get("/api/v1/export/status", headers={"Origin": "http://evil.example"}).headers   # not open to the world
