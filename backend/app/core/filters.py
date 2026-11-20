"""Dashboard filters as an API contract. Same semantics as the dashboard: AND across filters, exact match on the displayed value."""
from typing import Optional
import pandas as pd
from pydantic import BaseModel

FILTER_COLUMNS = {  # api id -> processed column (mirrors pipeline.schema.FILTERS, also published in schema.json)
    "region": "d_region_code", "company": "Company", "doc_type": "Sales Document type", "allocation_group": "d_allocation_group",
    "source_of_allocation": "Source of Allocation", "pn_group": "PNGroup", "sales_rep": "Sales Rep", "material_group_5": "Material group 5",
    "distribution_channel": "Distribution Channel", "early": "d_early", "month_of_allocation": "Month of Allocation",
    "customer_category": "d_customer_category"}

class FilterSpec(BaseModel):
    region: Optional[list[str]] = None
    company: Optional[list[str]] = None
    doc_type: Optional[list[str]] = None
    allocation_group: Optional[list[str]] = None
    source_of_allocation: Optional[list[str]] = None
    pn_group: Optional[list[str]] = None
    sales_rep: Optional[list[str]] = None
    material_group_5: Optional[list[str]] = None
    distribution_channel: Optional[list[str]] = None
    early: Optional[list[str]] = None
    month_of_allocation: Optional[list[str]] = None
    customer_category: Optional[list[str]] = None
    exclude_negative_durations: bool = False   # same switch as the dashboard; applied per target measure, not per row

def norm(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.map(lambda v: None if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v)))
    return s.astype("string")

def apply_filters(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for fid, col in FILTER_COLUMNS.items():
        vals = getattr(spec, fid)
        if vals:
            mask &= norm(df[col]).isin(vals)
    return df[mask]

def filter_options(df: pd.DataFrame) -> dict:
    return {fid: sorted(norm(df[col]).dropna().unique().tolist()) for fid, col in FILTER_COLUMNS.items()}

MIN_ROWS_FOR_MODELLING = 300   # below this the ML module must warn (configurable)
