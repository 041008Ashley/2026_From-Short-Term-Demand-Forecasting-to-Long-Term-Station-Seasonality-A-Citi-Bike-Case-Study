# Path configuration for the GitHub repository layout.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

h = pd.read_csv(str(DATA / "network_20stations_hourly_202608-202610.csv"))
h["hour"] = pd.to_datetime(h["hour"]); h = h.sort_values(["station_id", "hour"]).reset_index(drop=True)
h["hod"] = h.hour.dt.hour; h["dow"] = h.hour.dt.day_of_week; h["how"] = h.dow * 24 + h.hod
for t in ("departures", "arrivals"):
    g = h.groupby("station_id")[t]
    for l in (1, 2, 3, 6, 24, 168): h[f"{t}_l{l}"] = g.shift(l)
    for w in (24, 168): h[f"{t}_r{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).mean())
Fd = ["hod", "dow"] + [f"departures_l{l}" for l in (1,2,3,6,24,168)] + ["departures_r24", "departures_r168"]
Fa = ["hod", "dow"] + [f"arrivals_l{l}" for l in (1,2,3,6,24,168)] + ["arrivals_r24", "arrivals_r168"]
h = h.dropna(subset=Fd+Fa).reset_index(drop=True)

T0 = pd.Timestamp("2026-10-16")
tr, te = h[h.hour < T0], h[h.hour >= T0].copy()
for t, F in (("departures", Fd), ("arrivals", Fa)):
    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(tr[F], tr[t])
    te[f"p_{t}"] = rf.predict(te[F])
te["pn"] = te.p_arrivals - te.p_departures
te["an"] = te.arrivals - te.departures
te["day"] = te.hour.dt.date

# 逐日校准
dd = te.groupby(["day", "station_id"]).agg(p=("pn", "sum"), a=("an", "sum")).reset_index()
k = (dd.p * dd.a).sum() / (dd.p * dd.p).sum()
print("日净流量校准系数 k=%.3f  (k<1 表示预测幅度过分散)" % k)
print("日 MAE: 原始 %.2f | 校准后 %.2f | 零预测 %.2f" % (
    np.abs(dd.p - dd.a).mean(), np.abs(k*dd.p - dd.a).mean(), np.abs(dd.a).mean()))
for nm, col in (("原始", "p"), ("校准后", "pk")):
    if col == "pk": dd["pk"] = k * dd.p
    rho = dd.groupby("day").apply(lambda g: spearmanr(g[col], g.a).correlation, include_groups=False).mean()
    hit5 = dd.groupby("day").apply(lambda g: len(set(g.loc[g[col].abs().nlargest(5).index, "station_id"]) & set(g.loc[g.a.abs().nlargest(5).index, "station_id"]))/5, include_groups=False).mean()
    hit10 = dd.groupby("day").apply(lambda g: len(set(g.loc[g[col].abs().nlargest(10).index, "station_id"]) & set(g.loc[g.a.abs().nlargest(10).index, "station_id"]))/10, include_groups=False).mean()
    print("%-6s 日均 Spearman %.3f  top5 %.2f  top10 %.2f" % (nm, rho, hit5, hit10))
print("\n随机基线：top5 = 0.25, top10 = 0.50")
dd.to_csv(str(OUT / "res_daily_rank.csv"), index=False)
