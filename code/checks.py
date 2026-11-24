# Path configuration for the GitHub repository layout.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

import numpy as np, pandas as pd
from sarima_lib import build, css_eps, fit, path, metrics  # reuse

df = pd.read_csv(str(DATA / "station_6140.05_hourly_202511-202610.csv"))
df["ts"] = pd.to_datetime(df["hour"]); df = df.sort_values("ts").reset_index(drop=True)
y = df.departures.values.astype(float); a = df.arrivals.values.astype(float)
n = len(y)

# --- 1. VMR various definitions on the 20-station sample
h = pd.read_csv(str(DATA / "network_20stations_hourly_202608-202610.csv")); h["hour"] = pd.to_datetime(h["hour"])
h["how"] = h.hour.dt.dayofweek * 24 + h.hour.dt.hour
cell = h.groupby(["station_id", "how"]).departures.agg(["var", "mean"])
cell = cell[cell["mean"] > 0]
print("VMR station x hour-of-week cells: mean %.3f median %.3f  (n=%d)" % ((cell["var"]/cell["mean"]).mean(), (cell["var"]/cell["mean"]).median(), len(cell)))
cell2 = h.groupby(["station_id", "hour"]).departures.agg(["var", "mean"]); cell2 = cell2[cell2["mean"] > 0]
print("VMR station x hour-of-day cells: mean %.3f median %.3f" % ((cell2["var"]/cell2["mean"]).mean(), (cell2["var"]/cell2["mean"]).median()))

# --- 2. peak-hour bias, 4 folds
rows = []
for k in range(4):
    cut = n - 168*(4-k)
    m = fit(y[:cut], (1,0,1), (1,1,1), 168, n_rand=40)
    f = path(y[:cut], m, 168)
    act = y[cut:cut+168]; hr = df.ts[cut:cut+168].dt.hour.values
    for nm, mask in [("morning 8-9", np.isin(hr, [8,9])), ("evening 17-18", np.isin(hr, [17,18]))]:
        rows.append(dict(fold=str(df.ts[cut].date()), seg=nm,
                         actual=act[mask].mean(), pred=f[mask].mean(),
                         ratio_pct=(f[mask].mean()/act[mask].mean()-1)*100))
P = pd.DataFrame(rows)
print("\npeak-level bias (mean of 4 folds):")
print(P.groupby("seg")[["actual","pred","ratio_pct"]].mean().round(2).to_string())

# --- 3. Feb 23 (and neighbours) sensitivity: refit with zero block linearly interpolated
mask_bad = df.ts.between("2026-02-22", "2026-02-25")
print("\ndays 2026-02-22..25 dep:", df.loc[mask_bad].groupby(df.ts.dt.date).departures.sum().to_dict())
y2 = y.copy()
idx = np.where(mask_bad)[0]
y2[idx] = np.interp(idx, np.r_[idx[0]-1, idx[-1]+1], np.r_[y[idx[0]-1], y[idx[-1]+1]])
m0 = fit(y, (1,0,1), (1,1,1), 168, n_rand=40)
m1 = fit(y2, (1,0,1), (1,1,1), 168, n_rand=40)
print("as-is   : phi=%.3f theta=%.3f Phi=%.3f Theta=%.3f sigma=%.3f AIC=%.0f" % (*m0["v"], m0["sigma"], m0["aic"]))
print("patched : phi=%.3f theta=%.3f Phi=%.3f Theta=%.3f sigma=%.3f AIC=%.0f" % (*m1["v"], m1["sigma"], m1["aic"]))

# --- 4. one-step-ahead SARIMA on Oct 1-15 (fixed params from data up to Sep 30)
cut = int(np.where(df.ts >= pd.Timestamp("2026-10-01"))[0][0])
m = fit(y[:cut], (1,0,1), (1,1,1), 168, n_rand=40)
v = m["v"]; s = 168
phi, theta, Phi, Theta = v[0], v[1], v[2], v[3]
layers, _ = build(y[:cut], 0, 1, s); w = list(layers[-1])
eps = list(css_eps(np.array(w), [phi], [theta], [Phi], [Theta], s))
one = []
for t in range(len(w), len(y)):
    A = phi*w[t-1] if t >= 1 else 0.0
    A += Phi*w[t-s] if t >= s else 0.0
    A -= phi*Phi*w[t-1-s] if t >= 1+s else 0.0
    M = -theta*eps[t-1] if t >= 1 else 0.0
    M -= Theta*eps[t-s] if t >= s else 0.0
    M += theta*Theta*eps[t-1-s] if t >= 1+s else 0.0
    wn = A + M
    w.append(wn); eps.append(y[t]-wn)          # observed updates
    one.append(wn + y[t-s])                    # undo seasonal differencing
one = np.array(one); act = y[cut:]
print("\nSARIMA one-step ahead 2026-10-01..10-15 (fixed params):")
print({k: round(vv, 3) for k, vv in metrics(act, one).items()})
naive = y[cut-168:len(y)-168]
print("seasonal naive same window: %s" % {k: round(vv,3) for k, vv in metrics(act, naive).items()})

# --- 5. 96h / 168h multi-step on the same window for horizon effect
for H in (24, 168, 360):
    f = path(y[:cut], m, H)
    print("SARIMA %3dh horizon WAPE=%.1f%%  MAE=%.2f" % (H, metrics(y[cut:cut+H], f)["WAPE"], metrics(y[cut:cut+H], f)["MAE"]))
