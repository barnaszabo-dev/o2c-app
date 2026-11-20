"""Read-only access to the processed O2C dataset produced by build_data.py (pipeline/processed.py)."""
import json, os
from functools import lru_cache
from pathlib import Path
import pandas as pd

DATA_DIR = Path(os.environ.get("O2C_DATA_DIR", Path(__file__).resolve().parents[3] / "data" / "processed"))

@lru_cache
def load_df() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "orders_processed.parquet")

@lru_cache
def load_schema() -> dict:
    return json.loads((DATA_DIR / "schema.json").read_text(encoding="utf-8"))
