"""Business schema of the O2C lead-time dataset.
Single source of truth for: process events, lead-time measures, dashboard filters, column availability (leakage guard).
Original Excel column names are NEVER renamed; derived columns carry the prefixes d_ (derived dimension) and h_ (hours)."""

# Process events in chronological order. A value is "known" from the moment its event happens.
EVENTS = ["purchase_order", "sales_order_sap", "erp_order", "scheduling_match",
          "scheduled_status", "z5_removal", "order_confirmation", "real_loading"]
EV = {e: i for i, e in enumerate(EVENTS)}

# id, lowercase header prefix (same matching rule as the dashboard), label, kind, start event, end event
MEASURES = [
 ("po_to_so_sap",        "purchase order date - order created", "PO -> SO created in SAP",              "sequential", "purchase_order",   "sales_order_sap"),
 ("so_to_erp",           "order created in sap",                "SO created -> ERP order (SF)",          "sequential", "sales_order_sap",  "erp_order"),
 ("erp_to_sched_match",  "erp order created in sf -scheduling", "ERP order -> Scheduling match",         "sequential", "erp_order",        "scheduling_match"),
 ("sched_match_to_stat", "scheduling match-scheduled",          "Scheduling match -> Scheduled status",  "sequential", "scheduling_match", "scheduled_status"),
 ("stat_to_z5",          "scheduled status-z5",                 "Scheduled status -> Z5 removal",        "sequential", "scheduled_status", "z5_removal"),
 ("z5_to_oc",            "z5 removal-order",                    "Z5 removal -> Order confirmation sent", "sequential", "z5_removal",       "order_confirmation"),
 ("oc_to_loading",       "order confirmation sent-real",        "Order confirmation -> Real loading",    "sequential", "order_confirmation","real_loading"),
 ("po_to_oc_agg",        "purchase order date >>",              "PO date >> Order confirmation",         "aggregated", "purchase_order",   "order_confirmation"),
 ("erp_to_z5",           "erp order created >> z5",             "ERP order >> Z5 removal",               "aggregated", "erp_order",        "z5_removal"),
 ("erp_to_sm_agg",       "erp order created >> scheduling",     "ERP order >> Scheduling match",         "aggregated", "erp_order",        "scheduling_match"),
 ("sm_to_ss_agg",        "scheduling mathced",                  "Scheduling match >> Scheduled status",  "aggregated", "scheduling_match", "scheduled_status"),
 ("erp_to_oc",           "erp order created >> order",          "ERP order >> Order confirmation",       "aggregated", "erp_order",        "order_confirmation"),
 ("erp_to_price_match",  "erp order created>>price",            "ERP order >> Price match",              "aggregated", "erp_order",        None),   # price-match event position unclear -> review
 ("po_to_loading",       "purchase order date-real",            "PO date -> Real loading (whole days)",  "aggregated", "purchase_order",   "real_loading"),
 ("so_to_oc",            "sales order created ->",              "SO created -> Order confirmation",      "derived",    "sales_order_sap",  "order_confirmation"),
 ("po_to_oc",            "purchase order ->",                   "PO -> Order confirmation",              "derived",    "purchase_order",   "order_confirmation"),
]

# Dashboard filter id (API) -> (dashboard key in index.html, processed column, label)
FILTERS = {
 "region":               ("reg",                   "d_region_code",        "Region"),
 "company":              ("Company",               "Company",              "Company"),
 "doc_type":             ("Sales Document type",   "Sales Document type",  "Sales Document type"),
 "allocation_group":     ("ag",                    "d_allocation_group",   "Allocation group (Initial / Additional demand)"),
 "source_of_allocation": ("Source of Allocation",  "Source of Allocation", "Source of Allocation (raw)"),
 "pn_group":             ("PNGroup",               "PNGroup",              "PNGroup"),
 "sales_rep":            ("Sales Rep",             "Sales Rep",            "Sales Rep"),
 "material_group_5":     ("Material group 5",      "Material group 5",     "Material group 5"),
 "distribution_channel": ("Distribution Channel",  "Distribution Channel", "Distribution Channel"),
 "early":                ("early",                 "d_early",              "Early / Non-early"),
 "month_of_allocation":  ("Month of Allocation",   "Month of Allocation",  "Month of Allocation"),
 "customer_category":    (None,                    "d_customer_category",  "T1 / T2 / RS (initial allocation only)"),
}

# Original column -> (availability event | 'review' | 'identifier', role)
# 'review' = timing relative to the prediction point is not provable from the file -> excluded from features until the business confirms.
COLUMNS = {
 "Sales Document": ("identifier", "identifier"),
 "Region": ("purchase_order", "dimension"), "Company": ("purchase_order", "dimension"),
 "Sold-to party": ("purchase_order", "dimension_high_cardinality"), "Name": ("purchase_order", "dimension_high_cardinality"),
 "Sales Rep": ("purchase_order", "dimension"),
 "Sales Document type": ("sales_order_sap", "dimension"), "INCO1": ("sales_order_sap", "dimension"),
 "Source of Allocation": ("sales_order_sap", "dimension"), "Distribution Channel": ("sales_order_sap", "dimension"),
 "Item type": ("sales_order_sap", "dimension"), "Material group 5": ("sales_order_sap", "dimension"),
 "Pltyp": ("sales_order_sap", "dimension"), "PNGroup": ("sales_order_sap", "dimension"), "Zsourcemat": ("sales_order_sap", "dimension"),
 "Purchase Order Date": ("purchase_order", "timestamp"), "Create Sales Order Item": ("sales_order_sap", "timestamp"),
 "Sales Order Item Created": ("sales_order_sap", "timestamp"), "ERP CreatedDate": ("erp_order", "timestamp"),
 "Schedule match date": ("scheduling_match", "timestamp"), "Price match date": ("review", "timestamp"),
 "Scheduled status": ("scheduled_status", "timestamp"), "Z5 block removal": ("z5_removal", "timestamp"),
 "Order Confirmation Date": ("order_confirmation", "timestamp"), "Loading Date": ("real_loading", "timestamp"),
 "Allocation Date": ("review", "timestamp"), "Month of Allocation": ("review", "dimension"),
 "korai / nem korai rendelés": ("review", "dimension"),
 "diff between sale plan and order": ("review", "numeric"), "Sale Plan": ("review", "numeric"),
 "Field / Event": ("z5_removal", "log"), "Old Value": ("z5_removal", "log"), "New Value": ("z5_removal", "log"),
 "QualityLevel": ("review", "dimension"), "Past due reason": ("review", "text"),
 "Sales comment": ("review", "text"), "Comment": ("review", "text"),
 "Loaded volume": ("real_loading", "numeric"),
 "Oszlop1": ("review", "helper"), "Lead time relevancy": ("review", "metadata"),
 "Date of Allocation - Scheduling match": ("review", "helper"),
 ">48 SO-OC": ("order_confirmation", "outcome_flag"), ">48 PO-OC": ("order_confirmation", "outcome_flag"),
 # derived by pipeline/processed.py
 "d_region_code": ("purchase_order", "dimension"), "d_allocation_group": ("sales_order_sap", "dimension"),
 "d_customer_category": ("sales_order_sap", "dimension"), "d_early": ("review", "dimension"),
 "d_within_48h_po_oc": ("order_confirmation", "outcome_flag"), "d_sales_document_base": ("identifier", "group_key"),
}

# Where a prediction is made. A feature is allowed only if it is known at or before this event.
PREDICTION_POINTS = {
 "at_purchase_order":    {"event": "purchase_order",   "label": "When the purchase order arrives"},
 "at_sales_order":       {"event": "sales_order_sap",  "label": "When the sales order is created in SAP"},
 "at_erp_order":         {"event": "erp_order",        "label": "When the ERP order is created in SF"},
 "at_scheduling_match":  {"event": "scheduling_match", "label": "After the scheduling match"},
}
NON_FEATURE_ROLES = {"identifier", "group_key", "log", "text", "helper", "metadata", "outcome_flag", "dimension_high_cardinality"}
