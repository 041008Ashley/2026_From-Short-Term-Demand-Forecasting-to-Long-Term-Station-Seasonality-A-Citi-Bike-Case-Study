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

cut = int(np.where(df.ts >= pd.Timestamp("2026-10-01"))[0][0])
H = 360
m = fit(y[:cut], (1,0,1), (1,1,1), s, n_rand=40)
phi, theta, Phi, Theta = m["v"][:4]
layers, _ = build(y[:cut], 0, 1, s); w = list(layers[-1])   # w[t] -> y[t+s]
eps = list(css_eps(np.array(w), [phi], [theta], [Phi], [Theta], s))

fc, eps_star = [], list(eps)
for t in range(cut-s, cut-s+H):
    A = (phi*w[t-1] if t >= 1 else 0.0) + (Phi*w[t-s] if t >= s else 0.0) - (phi*Phi*w[t-1-s] if t >= 1+s else 0.0)
    M = (-theta*eps_star[t-1] if t >= 1 else 0.0) - (Theta*eps_star[t-s] if t >= s else 0.0) + (theta*Theta*eps_star[t-1-s] if t >= 1+s else 0.0)
    what = A + M
    fc.append(what + y[t])                       # yhat[t+s]
    wact = y[t+s] - y[t]                        # observed differenced value
    w.append(wact); eps_star.append(wact - what)  # update with observed error

pred1 = np.array(fc); act = y[cut:cut+H]
print("one-step ahead 2026-10-01..10-15, fixed params (%d h):" % H)
print("  SARIMA         ", {k: round(v,3) for k,v in metrics(act, pred1).items()})
print("  seasonal naive ", {k: round(v,3) for k,v in metrics(act, y[cut-s:cut-s+H]).items() if k!='na'})
print("  bias by hour (first 24h):", np.round(act[:24]-pred1[:24],1).tolist())

for HH in (24, 168, 336):
    f = path(y[:cut], m, HH)
    mm = metrics(y[cut:cut+HH], f)
    print("multi-step %3dh  WAPE=%.1f%%  MAE=%.2f  bias=%.2f" % (HH, mm["WAPE"], mm["MAE"], mm["bias"]))
