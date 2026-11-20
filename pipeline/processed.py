"""Raw Excel DataFrame -> processed analytical dataset (parquet) + schema.json (data dictionary + quality report).
Pure data preparation: no modelling. Consumed by the future Python/ML backend; the dashboard keeps using data/orders.json."""
import hashlib, json, re, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
from .schema import MEASURES, FILTERS, COLUMNS, EVENTS, EV, PREDICTION_POINTS

def _resolve(df):
    low = {c: c.lower() for c in df.columns}
    out = {}
    for mid, prefix, *_ in MEASURES:
        hit = [c for c in df.columns if low[c].startswith(prefix)]
        out[mid] = hit[0] if hit else None
    return out

def build_processed(df_raw, src_path, out_dir="data/processed"):
    df, out = df_raw.copy(), Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    res = _resolve(df)
    # ---- derived dimensions (same rules as the dashboard) ----
    src = df["Source of Allocation"].astype("string")
    ini = src.str.contains("inital|initial", case=False, na=False)
    df["d_region_code"] = df["Region"].astype("string").str[:3]
    df["d_allocation_group"] = np.where(ini, "Initial", "Additional demand")
    cat = src.str.extract(r"- (T1|T2|RS) -")[0]
    df["d_customer_category"] = cat.where(ini)
    df["d_early"] = df["korai / nem korai rendelés"].astype("string")
    flag = df.get(">48 PO-OC")
    df["d_within_48h_po_oc"] = (flag == 1) if flag is not None else False
    df["d_sales_document_base"] = df["Sales Document"].astype("string").str.split("#").str[0]   # items of one order share this key
    # ---- hours columns (file stores days) ----
    for mid, col in res.items():
        df[f"h_{mid}"] = pd.to_numeric(df[col], errors="coerce") * 24 if col else np.nan
    # ---- parquet-safe object columns ----
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and v != v) else (v if isinstance(v, str) else str(v)))
    df.to_parquet(out / "orders_processed.parquet", index=False)
    # ---- schema.json ----
    meas = []
    for mid, prefix, label, kind, a, b in MEASURES:
        s = df[f"h_{mid}"]
        meas.append(dict(id=mid, source_column=res[mid], hours_column=f"h_{mid}", label=label, kind=kind, start_event=a, end_event=b,
                         n_valid=int(s.notna().sum()), n_negative=int((s < 0).sum()), median_hours=None if s.notna().sum() == 0 else round(float(s.median()), 3)))
    cols = []
    for c in df.columns:
        av, role = COLUMNS.get(c, ("review", "duration_or_helper" if c.startswith("h_") else "unclassified"))
        if c.startswith("h_"):
            m = next(m for m in meas if m["hours_column"] == c); av, role = (m["end_event"] or "review"), "lead_time_measure"
        cols.append(dict(name=c, dtype=str(df[c].dtype), n_missing=int(df[c].isna().sum()), availability=av, role=role, derived=c.startswith(("d_", "h_"))))
    h48 = df["h_po_to_oc"]
    quality = dict(
        unique_sales_document=bool(df["Sales Document"].is_unique), n_unique_base_documents=int(df["d_sales_document_base"].nunique()),
        durations_stored_in="days (converted to hours in h_* columns)",
        flag_gt48_po_oc_means_within_48h=bool(((h48 <= 48) == (flag == 1)).all()) if flag is not None else None,
        lead_time_relevancy_populated=bool(df["Lead time relevancy"].notna().any()) if "Lead time relevancy" in df else False,
        z5_removal_missing_pct=round(float(df["Z5 block removal"].isna().mean() * 100), 1) if "Z5 block removal" in df else None)
    schema = dict(
        dataset=dict(id=hashlib.md5(Path(src_path).read_bytes()).hexdigest()[:12], source_file=Path(src_path).name,
                     built_at=dt.datetime.now().isoformat(timespec="seconds"), n_rows=len(df), parquet="orders_processed.parquet"),
        events=EVENTS, prediction_points=PREDICTION_POINTS,
        filters=[dict(id=k, dashboard_key=v[0], column=v[1], label=v[2]) for k, v in FILTERS.items()],
        measures=meas, columns=cols, quality=quality)
    (out / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}/orders_processed.parquet + schema.json  (dataset id {schema['dataset']['id']})")
    return schema
