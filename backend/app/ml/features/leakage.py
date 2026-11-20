"""Leakage guard: which columns may be used as features / targets for a given prediction point.
Rule: a feature must be known at or before the prediction event. Columns whose timing cannot be proven ('review') are excluded by default."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))      # project root -> shared `pipeline` package
from pipeline.schema import NON_FEATURE_ROLES
from app.core.dataset import load_schema

def allowed(point: str, include_review: bool = False) -> dict:
    sc = load_schema(); ev = sc["events"]; pp = sc["prediction_points"][point]; p = ev.index(pp["event"])
    feats, excluded, review = [], [], []
    for c in sc["columns"]:
        av, role, name = c["availability"], c["role"], c["name"]
        if role in NON_FEATURE_ROLES:
            excluded.append(dict(column=name, reason=f"role:{role}"))
        elif av == "review":
            if include_review: feats.append(name)
            else: review.append(dict(column=name, reason="timing unconfirmed - needs business confirmation"))
        elif ev.index(av) <= p:
            feats.append(name)
        else:
            excluded.append(dict(column=name, reason=f"known only at {av} (after {pp['event']})"))
    targets = [m["id"] for m in sc["measures"] if m["end_event"] and ev.index(m["end_event"]) > p]
    return dict(prediction_point=point, event=pp["event"], allowed_features=feats, pending_business_review=review, excluded=excluded, valid_targets=targets)
