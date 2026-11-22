"""STEP 2: anomaly detection (Isolation Forest + local seasonal baseline) and its evaluation.
Run:  python anomaly.py

Real meter data contains STUCK stretches (the same reading repeated for many hours) and level shifts.
 - Stuck stretches are detected, reported as a data-quality finding and not scored.
 - "Normal" is the same weekday and hour in the NEARBY weeks, built from non-stuck readings only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

STUCK_RUN = 6        # this many identical readings in a row = stuck meter
MIN_MEAN_KWH = 5.0   # blocks below this average hourly use are not scored (noise looks like spikes)


def load(path="data/energy.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values(["block", "timestamp"]).reset_index(drop=True)


def stuck_mask(df, min_run=STUCK_RUN):
    """True for hours inside a run of >= min_run identical readings."""
    out = pd.Series(False, index=df.index)
    for _, g in df.sort_values("timestamp").groupby("block"):
        run_id = (g.kwh != g.kwh.shift()).cumsum()
        out.loc[g.index] = (g.groupby(run_id).kwh.transform("size") >= min_run).values
    return out


def quality(df):
    """Share of hours stuck on a repeated value, per block."""
    s = df["stuck"] if "stuck" in df.columns else stuck_mask(df)
    return (s.groupby(df.block).mean() * 100).round(1).rename("stuck_hours_pct").reset_index()


def _side(s, window, min_weeks, right):
    """Median of the previous (or next, if right=True) `window` weekly values of one weekday-hour slot."""
    x = s.iloc[::-1] if right else s
    r = x.shift(1).rolling(window, min_periods=min_weeks).median()
    return r.iloc[::-1] if right else r


def features(df, side_weeks=8, min_weeks=2, noise_lo=0.7, noise_hi=2.0):
    """'Normal' for an hour = the same block, weekday and hour in the weeks BEFORE and AFTER it (clean readings only).
    The baseline is the higher of the two sides, so a permanent level shift (a new operating regime) is never
    mistaken for a spike, while an isolated spike is high against both sides."""
    f = df.sort_values(["block", "timestamp"]).copy()
    if "stuck" not in f.columns:
        f["stuck"] = stuck_mask(f)
    f["hour"] = f.timestamp.dt.hour
    f["dow"] = f.timestamp.dt.dayofweek
    f["_clean"] = f.kwh.where(~f.stuck)                       # stuck readings carry no information about "normal"
    slot = ["block", "dow", "hour"]
    gc = f.groupby(slot)["_clean"]
    f["_left"] = gc.transform(lambda s: _side(s, side_weeks, min_weeks, False))
    f["_right"] = gc.transform(lambda s: _side(s, side_weeks, min_weeks, True))
    f["baseline"] = np.fmax(f._left, f._right)
    # Noise: local estimate (median absolute deviation of nearby weeks), kept within a band around the block's
    # own typical relative noise, because a few clean weeks can give a wildly wrong local figure.
    f["_res"] = f._clean - f.baseline
    mad = f.groupby(slot)["_res"].transform(
        lambda s: s.abs().rolling(2 * side_weeks + 1, center=True, min_periods=min_weeks + 2).median()) * 1.4826
    rel = (f._res.abs() / f.baseline.clip(lower=0.1)).groupby(f.block).transform("median") * 1.4826
    typical = rel.fillna(0.05).clip(lower=0.03) * f.baseline.clip(lower=0.1)      # kWh, block-typical
    f["scale"] = np.maximum(np.clip(mad.fillna(typical), noise_lo * typical, noise_hi * typical), 0.05)
    f["ratio"] = f.kwh / f.baseline.clip(lower=0.1)
    f["z"] = (f.kwh - f.baseline) / f.scale
    # An hour is scored only with clean evidence on BOTH sides (so a level shift cannot be mistaken for a spike),
    # except in the first and last weeks of the data, where only one side exists.
    both = f._left.notna() & f._right.notna()
    span = f.groupby("block")["timestamp"]
    edge = ((f.timestamp - span.transform("min")) < pd.Timedelta(weeks=side_weeks)) | \
           ((span.transform("max") - f.timestamp) < pd.Timedelta(weeks=side_weeks))
    f["usable"] = f.baseline.notna() & ~f.stuck & (both | edge)
    return f.drop(columns=["_clean", "_left", "_right", "_res"])


def detect(df, contamination=0.04, ratio_thr=1.4, z_thr=3.0, min_mean_kwh=MIN_MEAN_KWH, side_weeks=8):
    f = features(df, side_weeks=side_weeks)
    f["scored"] = f.groupby("block")["kwh"].transform("mean") >= min_mean_kwh     # block level
    f["iso_flag"] = False
    cols = ["hour", "dow", "ratio", "z"]
    for _, g in f[f.scored & f.usable].groupby("block"):       # one model per block
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        iso.fit(g[cols])
        f.loc[g.index, "iso_flag"] = iso.predict(g[cols]) == -1
    f["rule_flag"] = (f.ratio > ratio_thr) & (f.z > z_thr) & f.scored & f.usable
    f["anomaly"] = f.iso_flag & f.rule_flag                    # both methods must agree
    f["reason"] = ""
    m = f.anomaly
    f.loc[m, "reason"] = (f.loc[m, "ratio"].round(1).astype(str) + "x its normal level for "
                          + f.loc[m, "timestamp"].dt.day_name() + " " + f.loc[m, "hour"].astype(str) + ":00")
    return f


def inject(df, frac=0.02, seed=7, lo=1.5, hi=2.5):
    """Plant labelled spikes in hours of a working meter that have enough clean history."""
    d = df.copy()
    f0 = features(d)
    ok = f0.usable & (f0.groupby("block")["kwh"].transform("mean") >= MIN_MEAN_KWH)
    d["stuck"] = f0["stuck"]                                   # stuck hours are fixed from the ORIGINAL data
    rng = np.random.default_rng(seed)
    pool = f0.index[ok]
    idx = rng.choice(pool, int(frac * len(pool)), replace=False)
    d.loc[idx, "kwh"] *= rng.uniform(lo, hi, len(idx))
    d["is_anomaly"] = 0
    d.loc[idx, "is_anomaly"] = 1
    return d


def evaluate(df, seed=7, **kw):
    f = detect(inject(df, seed=seed), **kw)
    f = f[f.scored & f.usable]       # only hours that can be scored
    y = f.is_anomaly
    rows = []
    for name, col in [("Isolation Forest", "iso_flag"), ("Baseline rule", "rule_flag"),
                      ("Both agree", "anomaly"), ("Either", None)]:
        p = (f.iso_flag | f.rule_flag) if col is None else f[col]
        rows.append({"method": name, "accuracy": accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0),
                     "recall": recall_score(y, p), "f1": f1_score(y, p, zero_division=0)})
    return pd.DataFrame(rows).round(3)


def evaluate_by_block(df, seed=7, **kw):
    f = detect(inject(df, seed=seed), **kw)
    rows = []
    for b, g in f.groupby("block"):
        if not g.scored.iloc[0]:
            rows.append({"block": b, "scored": False})
            continue
        g = g[g.usable]
        y, p = g.is_anomaly, g.anomaly
        rows.append({"block": b, "scored": True, "scorable_hours": len(g), "planted_spikes": int(y.sum()),
                     "precision": precision_score(y, p, zero_division=0),
                     "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0)})
    return pd.DataFrame(rows).round(3)


if __name__ == "__main__":
    data = load()
    print("Data quality: share of hours stuck on a repeated reading (not scored):")
    print(quality(data).to_string(index=False))
    print("\nEvaluation on data with PLANTED spikes (1.5x to 2.5x), not real labels:")
    print(evaluate(data).to_string(index=False))
    print("\nPer block:")
    print(evaluate_by_block(data).to_string(index=False))
    flagged = detect(data)
    print("\nAnomalies flagged in the real data:", int(flagged.anomaly.sum()), flagged[flagged.anomaly].groupby("block").size().to_dict())
    print(flagged[flagged.anomaly][["timestamp", "block", "kwh", "reason"]].head(8).to_string(index=False))
