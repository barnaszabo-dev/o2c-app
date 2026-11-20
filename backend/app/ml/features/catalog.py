"""Candidate explanatory variables with their availability (leakage guard). A variable is allowed only if it is known at or before the prediction point."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pipeline.schema import COLUMNS
from app.core.dataset import load_schema

CATS = [("region", "d_region_code", "Region"), ("company", "Company", "Company"), ("doc_type", "Sales Document type", "Sales Document type"),
        ("allocation_group", "d_allocation_group", "Allocation group"), ("customer_category", "d_customer_category", "T1/T2/RS"),
        ("source_of_allocation", "Source of Allocation", "Source of Allocation"), ("pn_group", "PNGroup", "PNGroup"),
        ("distribution_channel", "Distribution Channel", "Distribution Channel"), ("material_group_5", "Material group 5", "Material group 5"),
        ("sales_rep", "Sales Rep", "Sales Rep"), ("inco1", "INCO1", "INCO1"), ("item_type", "Item type", "Item type"),
        ("early", "d_early", "Early / Non-early"), ("month_of_allocation", "Month of Allocation", "Month of Allocation"), ("po_weekday", "__po_weekday", "PO weekday")]
NUMS = [("po_hour", "__po_hour", "PO hour of day", "purchase_order"), ("loaded_volume", "Loaded volume", "Loaded volume (actual)", "real_loading")]

def catalog(point: str, include_review: bool = False, target: str = None):
    sc = load_schema(); ev = sc["events"]; pe = sc["prediction_points"][point]["event"]; p = ev.index(pe); out = []
    def add(i, label, typ, col, av):
        if av == "review": ok, why = include_review, (None if include_review else "timing unconfirmed (business review pending)")
        else: ok = ev.index(av) <= p; why = None if ok else f"known only at {av}, after {pe}"
        out.append(dict(id=i, label=label, type=typ, column=col, availability=av, allowed=ok, reason=why, review=av == "review"))
    for i, col, label in CATS: add(i, label, "categorical", col, "purchase_order" if col.startswith("__") else COLUMNS.get(col, ("review", ""))[0])
    for i, col, label, av in NUMS: add(i, label, "numeric", col, av)
    for m in sc["measures"]:
        if m["id"] != target: add("m_" + m["id"], m["label"], "numeric", "h_" + m["id"], m["end_event"] or "review")
    return out
