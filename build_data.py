"""Excel -> data/orders.json. Eredeti oszlopnevek megmaradnak; az időtartamok napban vannak (a dashboard számol órára)."""
import sys, json, pandas as pd
src = sys.argv[1] if len(sys.argv) > 1 else "data/Lead_Time_between_steps_in_hour.xlsx"
df = pd.read_excel(src, sheet_name=0)            # 1. munkalap = rendelésszintű adat
try:   # ML-előkészítés: feldolgozott adatkészlet + sémaleírás a jövőbeli Python backendnek (a dashboard ettől függetlenül működik)
    from pipeline.processed import build_processed
    build_processed(df, src)
except ImportError as e:
    print("Figyelem: a feldolgozott (parquet) adatkészlet kimarad -", e, "-> pip install -r requirements.txt")
drop = ["Comment","Sales comment","Oszlop1","Schedule match date2","ERP CreatedDate4",
        "Price match date5","Scheduled status9","Date of Allocation - Scheduling match"]
df = df.drop(columns=[c for c in drop if c in df.columns])
for c in df.select_dtypes("datetime").columns:
    df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
for c in df.select_dtypes("float").columns:
    df[c] = df[c].round(5)            # ~1 mp pontosság napban
df = df.astype(object).where(df.notna(), None)
# ellenőrzések
sd = "Sales Document"
print(f"{len(df)} sor, egyedi {sd}: {df[sd].nunique()}")
fl = [c for c in df.columns if c.startswith(">48")]
for c in fl: print(c, "értékek:", df[c].value_counts(dropna=False).to_dict())
payload = {"columns": list(df.columns), "rows": df.values.tolist()}   # oszlopos formátum: kis méret
json.dump(payload, open("data/orders.json","w",encoding="utf-8"), ensure_ascii=False, separators=(",",":"), default=str)
print("-> data/orders.json kész")
