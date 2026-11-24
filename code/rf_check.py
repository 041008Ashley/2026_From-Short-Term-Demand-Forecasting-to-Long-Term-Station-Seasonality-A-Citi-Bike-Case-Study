# Path configuration for the GitHub repository layout.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor

h = pd.read_csv(str(DATA / "network_20stations_hourly_202608-202610.csv"))
h["hour"] = pd.to_datetime(h["hour"])
h = h.sort_values(["station_id", "hour"]).reset_index(drop=True)
h["hod"] = h.hour.dt.hour; h["dow"] = h.hour.dt.dayofweek
g = h.groupby("station_id")
for lag in (1, 2, 3, 6, 24, 168):
    h[f"lag{lag}"] = g.departures.shift(lag)
for w in (24, 168):
    h[f"roll{w}"] = g.departures.transform(lambda s: s.shift(1).rolling(w).mean())
h = h.dropna().reset_index(drop=True)
feat = ["hod", "dow", "lag1", "lag2", "lag3", "lag6", "lag24", "lag168", "roll24", "roll168"]

tr = h[h.hour < "2026-10-01"]; va = h[h.hour >= "2026-10-01"]
print("train", len(tr), "valid", len(va))

rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=1, n_jobs=-1, random_state=0)
rf.fit(tr[feat], tr.departures)
va = va.copy(); va["pred"] = rf.predict(va[feat])

def m(a, p):
    e = a - p
    return dict(MAE=round(np.abs(e).mean(), 3), RMSE=round(np.sqrt((e**2).mean()), 3),
                WAPE=round(np.abs(e).sum()/np.abs(a).sum()*100, 2))

print("RF pooled  Oct1-15 :", m(va.departures.values, va.pred.values))
s = va[va.station_id.astype(str) == "6140.05"]
print("RF st6140.05 only  :", m(s.departures.values, s.pred.values), " n=", len(s))
print("naive st6140.05    :", m(s.departures.values, s.lag168.values))
print("mean dep st6140.05 :", round(s.departures.mean(), 2), " pooled:", round(va.departures.mean(), 2))
