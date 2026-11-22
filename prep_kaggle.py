"""STEP 1: turn the Kaggle ASHRAE Great Energy Predictor III files into data/energy.csv
Needs in data/:  train.csv  and  building_metadata.csv   (from the Kaggle competition)
Output columns:  timestamp, block, building_id, kwh
"""
import pandas as pd

meta = pd.read_csv("data/building_metadata.csv")
# Education buildings, excluding site 0 (its early-2016 electricity readings are mostly zeros)
cand = meta[(meta.primary_use == "Education") & (meta.site_id != 0)].building_id

parts = []
for chunk in pd.read_csv("data/train.csv", chunksize=2_000_000, parse_dates=["timestamp"]):
    parts.append(chunk[(chunk.meter == 0) & chunk.building_id.isin(cand)])  # meter 0 = electricity
df = pd.concat(parts)

stats = df.groupby("building_id")["meter_reading"].agg(n="count", nonzero=lambda s: (s > 0).mean())
good = stats[(stats.n >= 8000) & (stats.nonzero > 0.98)].sort_values(["n"], ascending=False).head(4)
if len(good) < 4:
    raise SystemExit(f"Only {len(good)} suitable buildings found. Loosen the filters in prep_kaggle.py.")

names = dict(zip(sorted(good.index), ["Block A", "Block B", "Block C", "Block D"]))
out = df[df.building_id.isin(names)].copy()
out["block"] = out.building_id.map(names)
out = out.rename(columns={"meter_reading": "kwh"})[["timestamp", "block", "building_id", "kwh"]]
out.sort_values(["timestamp", "block"]).to_csv("data/energy.csv", index=False)
print("Saved data/energy.csv:", len(out), "rows")
print("Block -> Kaggle building_id:", names)
