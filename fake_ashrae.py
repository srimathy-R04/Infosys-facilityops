"""OPTIONAL: creates a small fake file in the same format as the Kaggle ASHRAE data,
so you can test the pipeline before downloading the real files.
It refuses to overwrite real Kaggle files."""
import os
import numpy as np
import pandas as pd

if os.path.exists("data/train.csv"):
    raise SystemExit("data/train.csv already exists. Not overwriting it.")
os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(1)
hours = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
frames, meta = [], []
for b in range(7):
    base = rng.uniform(40, 200)
    working = (hours.hour >= 8) & (hours.hour < 19) & (hours.weekday < 5)
    load = base * (0.4 + 0.6 * working) * (1 + 0.1 * np.sin(2 * np.pi * hours.dayofyear / 365))
    kwh = np.clip(np.asarray(load) + rng.normal(0, base * 0.03, len(hours)), 0.1, None)
    if b == 5:
        kwh[:2400] = 0  # a meter with a long zero stretch (should be skipped)
    frames.append(pd.DataFrame({"building_id": b, "meter": 0, "timestamp": hours, "meter_reading": kwh}))
    meta.append({"site_id": 1, "building_id": b,
                 "primary_use": "Office" if b == 6 else "Education", "square_feet": 50000})
pd.concat(frames).to_csv("data/train.csv", index=False)
pd.DataFrame(meta).to_csv("data/building_metadata.csv", index=False)
print("Created fake data/train.csv and data/building_metadata.csv")
