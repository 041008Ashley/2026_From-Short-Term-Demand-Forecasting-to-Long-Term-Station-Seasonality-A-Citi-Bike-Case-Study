# Path configuration for the GitHub repository layout.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

import numpy as np, pandas as pd
from sarima_lib import build, css_eps, fit, path, metrics

df = pd.read_csv(str(DATA / "station_6140.05_hourly_202511-202610.csv"))
df["ts"] = pd.to_datetime(df["hour"]); df = df.sort_values("ts").reset_index(drop=True)
y = df.departures.values.astype(float); n = len(y); s = 168

# per-hour bias across 4 folds
rows = []
for k in range(4):
    cut = n - 168*(4-k)
    m = fit(y[:cut], (1,0,1), (1,1,1), s, n_rand=40)
    f = path(y[:cut], m, 168)
    act = y[cut:cut+168]; hr = df.ts[cut:cut+168].dt.hour.values
    for i in range(168):
        rows.append(dict(h=hr[i], act=act[i], pred=f[i]))
B = pd.DataFrame(rows)
g = B.groupby("h").agg(act=("act","mean"), pred=("pred","mean"))
g["bias_pct"] = (g.pred/g.act-1)*100
print("hour-of-day bias (%, pred vs actual), 4 folds pooled:")
print(g.round(1).to_string())
print("\nmean bias on hours with act>=30: %.1f%%" % (g[g.act>=30].bias_pct.mean()))
print("mean bias on peak hours 7-9,17-19: %.1f%%" % (g.loc[[7,8,9,17,18,19]].bias_pct.mean()))
print("max pred/act ratio at hour 17-18: %.1f%%" % (g.loc[[17,18]].bias_pct.mean()))

# ---- one-step-ahead SARIMA, fixed params, Oct 1-15
cut = int(np.where(df.ts >= pd.Timestamp("2026-10-01"))[0][0])
m = fit(y[:cut], (1,0,1), (1,1,1), s, n_rand=40)
phi, theta, Phi, Theta, sig = m["v"][0], m["v"][1], m["v"][2], m["v"][3], m["sigma"]
layers, _ = build(y[:cut], 0, 1, s); w_full = layers[-1]       # length cut - s
t0 = cut - s
eps = list(css_eps(np.array(w_full[:t0]), [phi], [theta], [Phi], [Theta], s))
w = list(w_full[:t0])
one = {}
for t in range(t0, cut - s):          # observed w available up to cut-s-1
    A = (phi*w[t-1] if t >= 1 else 0.0) + (Phi*w[t-s] if t >= s else 0.0) - (phi*Phi*w[t-1-s] if t >= 1+s else 0.0)
    M = (-theta*eps[t-1] if t >= 1 else 0.0) - (Theta*eps[t-s] if t >= s else 0.0) + (theta*Theta*eps[t-1-s] if t >= 1+s else 0.0)
    wn = A + M
    one[t+s] = wn + y[t]
    w.append(w_full[t]); eps.append(w_full[t]-wn)
idx = np.array(sorted(one)); pred1 = np.array([one[i] for i in idx])
print("\none-step ahead WAPE/MAE on 2026-10-01..10-15 (%d hours):" % len(idx))
print({k: round(v, 3) for k, v in metrics(y[idx], pred1).items()})
print("seasonal naive  :", {k: round(v, 3) for k, v in metrics(y[idx], y[idx-s]).items()})

for H in (24, 168, 336):
    m2 = fit(y[:cut], (1,0,1), (1,1,1), s, n_rand=40)
    f = path(y[:cut], m2, H)
    mm = metrics(y[cut:cut+H], f)
    print("multi-step %3dh  WAPE=%.1f%%  MAE=%.2f  bias=%.2f" % (H, mm["WAPE"], mm["MAE"], mm["bias"]))
