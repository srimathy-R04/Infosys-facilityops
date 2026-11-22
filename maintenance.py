"""MILESTONE 2: predictive maintenance (simulated temperature and vibration data).
Run:  python maintenance.py
Real alternatives on Kaggle: NASA C-MAPSS, "Machine Predictive Maintenance Classification".
To use one, replace simulate() with code that loads it into the same columns:
asset, kind, day, temperature, vibration, fail_within_7d
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

TODAY = 149
FEATURES = ["temp_dev", "vib_dev", "temp_dev_3d", "vib_dev_3d", "vib_trend"]


def simulate(seed=3):
    """Lifts, generators and HVAC units whose temperature and vibration drift upward before failure."""
    rng = np.random.default_rng(seed)
    kinds = [("Lift", 8, 45, 2.0), ("Generator", 6, 70, 3.5), ("HVAC", 10, 55, 2.5)]
    rows = []
    for kind, n, base_t, base_v in kinds:
        for i in range(n):
            name = f"{kind} {i + 1}"
            bt, bv = base_t + rng.normal(0, 3), base_v + rng.normal(0, 0.3)   # each asset has its own normal
            fail_day = int(rng.integers(30, 175)) if rng.random() < 0.7 else None
            fail_day = {"Lift 3": TODAY + 4, "Lift 6": TODAY + 9, "Generator 2": TODAY + 12,
                        "HVAC 5": TODAY + 2, "HVAC 8": TODAY + 20}.get(name, fail_day)  # a few assets about to fail
            for d in range(TODAY + 1):
                dist = (fail_day - d) if fail_day is not None else 999
                if dist < 0:
                    break                                   # asset failed and was taken out of service
                wear = float(np.clip(1 - dist / 25, 0, 1))  # degradation ramps up over the last 25 days
                rows.append({
                    "asset": name, "kind": kind, "day": d,
                    "temperature": bt + rng.normal(0, 1.0) + 18 * wear ** 1.5,
                    "vibration": bv + rng.normal(0, 0.15) + 2.5 * wear ** 2,
                    "fail_within_7d": int(0 <= dist <= 7),
                    "failed": int(dist == 0),
                })
    return pd.DataFrame(rows)


def add_features(df):
    df = df.sort_values(["asset", "day"]).copy()
    base = df[df.day < 14].groupby("asset")[["temperature", "vibration"]].mean()
    base.columns = ["temp_base", "vib_base"]
    df = df.merge(base, on="asset")
    df["temp_dev"] = df.temperature - df.temp_base
    df["vib_dev"] = df.vibration - df.vib_base
    g = df.groupby("asset")
    df["temp_dev_3d"] = g["temp_dev"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["vib_dev_3d"] = g["vib_dev"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["vib_trend"] = df.groupby("asset")["vib_dev_3d"].transform(lambda s: s.diff(5)).fillna(0)
    return df


def status_label(score):
    return "Excellent" if score >= 80 else "Good" if score >= 60 else "Warning" if score >= 40 else "Critical"


def run():
    df = add_features(simulate())
    train_i, test_i = next(GroupShuffleSplit(test_size=0.3, random_state=0).split(df, groups=df.asset))
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=15, class_weight="balanced", random_state=0)
    model.fit(df.iloc[train_i][FEATURES], df.iloc[train_i].fail_within_7d)
    test = df.iloc[test_i]
    pred = model.predict(test[FEATURES])
    metrics = {"precision": precision_score(test.fail_within_7d, pred, zero_division=0),
               "recall": recall_score(test.fail_within_7d, pred, zero_division=0),
               "f1": f1_score(test.fail_within_7d, pred, zero_division=0)}

    df["risk"] = model.predict_proba(df[FEATURES])[:, 1]                 # chance of failing within 7 days
    latest = df[df.day == TODAY].copy()                                  # assets still running today
    latest["health_score"] = ((1 - latest.risk) * 100).round(0).astype(int)
    latest["status"] = latest.health_score.map(status_label)
    latest["recommendation"] = np.where(
        latest.risk >= 0.5, "Inspect within 3 days",
        np.where(latest.risk >= 0.25, "Schedule inspection within 7 days", "No action needed"))
    status = latest[["asset", "kind", "temperature", "vibration", "risk", "health_score", "status",
                     "recommendation"]].sort_values("health_score").round(2).reset_index(drop=True)

    orders = status[status.recommendation != "No action needed"].copy()
    orders["priority"] = np.where(orders.risk >= 0.5, "High", "Medium")
    orders["work_order"] = ("Check " + orders.asset + ": temperature " + orders.temperature.astype(str)
                            + " C, vibration " + orders.vibration.astype(str) + " mm/s. " + orders.recommendation)
    return status, orders[["asset", "priority", "work_order"]], df, metrics


if __name__ == "__main__":
    status, orders, hist, m = run()
    print("Failure-within-7-days model on unseen assets (SIMULATED data):", {k: round(v, 2) for k, v in m.items()})
    print("\nAsset health today:\n", status.head(8).to_string(index=False))
    print("\nWork orders:\n", orders.to_string(index=False))
    print("\nStatus counts:", status.status.value_counts().to_dict())
