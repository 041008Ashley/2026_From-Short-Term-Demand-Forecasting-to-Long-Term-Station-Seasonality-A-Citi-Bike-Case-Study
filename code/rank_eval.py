# Path configuration for the GitHub repository layout.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

"""站点失衡排序的可评价性：把 Part 7 从"数值拟合"改成"排序质量"。"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

h = pd.read_csv(str(DATA / "network_20stations_hourly_202608-202610.csv"))
h["hour"] = pd.to_datetime(h["hour"])
h = h.sort_values(["station_id", "hour"]).reset_index(drop=True)
h["hod"] = h.hour.dt.hour; h["dow"] = h.hour.dt.day_of_week

for tgt in ("departures", "arrivals"):
    g = h.groupby("station_id")[tgt]
    for lag in (1, 2, 3, 6, 24, 168):
        h[f"{tgt}_l{lag}"] = g.shift(lag)
    for w in (24, 168):
        h[f"{tgt}_r{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).mean())

F = {"departures": ["hod", "dow"] + [f"departures_l{l}" for l in (1, 2, 3, 6, 24, 168)] + ["departures_r24", "departures_r168"],
     "arrivals":   ["hod", "dow"] + [f"arrivals_l{l}" for l in (1, 2, 3, 6, 24, 168)] + ["arrivals_r24", "arrivals_r168"]}
h = h.dropna(subset=sum(F.values(), [])).reset_index(drop=True)
print("rows after features:", len(h), h.hour.min(), "->", h.hour.max())

TEST0 = pd.Timestamp("2026-10-16")
tr = h[h.hour < TEST0]; te = h[h.hour >= TEST0].copy()
print("refit", len(tr), "test", len(te), "hours", te.hour.nunique())

for tgt in F:
    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
    rf.fit(tr[F[tgt]], tr[tgt])
    te[f"p_{tgt}"] = rf.predict(te[F[tgt]])
print("test WAPE dep %.2f arr %.2f" % (
    np.abs(te.p_departures - te.departures).sum() / te.departures.sum() * 100,
    np.abs(te.p_arrivals - te.arrivals).sum() / te.arrivals.sum() * 100))

te["p_net"] = te.p_arrivals - te.p_departures
te["a_net"] = te.arrivals - te.departures
te["n_net"] = te.arrivals_l168 - te.departures_l168          # seasonal naive net
# static baseline: historical mean |net| per station, from refit window only
hist = tr.assign(net=tr.arrivals - tr.departures).groupby("station_id").net.apply(lambda s: s.abs().mean())
te["static"] = te.station_id.map(hist)

rows = []
for tm, g in te.groupby("hour"):
    if len(g) != 20: continue
    d = dict(hour=tm)
    for nm, col in [("model", "p_net"), ("naive", "n_net"), ("static", "static")]:
        d[f"rho_{nm}"] = spearmanr(g[col], g.a_net).correlation
        d[f"rhoabs_{nm}"] = spearmanr(g[col].abs(), g.a_net.abs()).correlation
        top_pred = set(g.reindex(g[col].abs().sort_values(ascending=False).index[:5].index) if False else g.loc[g[col].abs().nlargest(5).index, "station_id"])
        top_act = set(g.loc[g.a_net.abs().nlargest(5).index, "station_id"])
        d[f"hit_{nm}"] = len(top_pred & top_act) / 5
        d[f"sign_{nm}"] = float(np.mean(np.sign(g[col]) == np.sign(g.a_net)))
    rows.append(d)
M = pd.DataFrame(rows)
print("\n=== 逐小时（20 站横截面，%d 个小时）===" % len(M))
print(M[[c for c in M.columns if c.startswith(("rho_", "rhoabs_", "hit_", "sign_"))]].mean().round(3).to_string())

# daily aggregation
te["day"] = te.hour.dt.date
print("\n=== 按天（预测/实际日净流量，20 站 x 16 天）===")
for nm, col in [("model", "p_net"), ("naive", "n_net")]:
    dd = te.groupby(["day", "station_id"]).agg(p=(col, "sum"), a=("a_net", "sum")).reset_index()
    rho = dd.groupby("day").apply(lambda g: spearmanr(g.p, g.a).correlation, include_groups=False)
    print("%-6s 日均 Spearman=%.3f  日 MAE=%.2f  (日净流量 mean|a|=%.1f)" % (
        nm, rho.mean(), np.abs(dd.p - dd.a).mean(), dd.a.abs().mean()))
print("static 日 Spearman=%.3f" % te.assign(day=te.hour.dt.date).groupby(["day", "station_id"]).static.first().groupby("day").apply(
    lambda s: np.nan, include_groups=False).mean() if False else "")

# 单小时示例（论文里用的那种展示）
last = te[te.hour == te.hour.max()].copy()
last["rank_p"] = last.p_net.abs().rank(ascending=False).astype(int)
last["rank_a"] = last.a_net.abs().rank(ascending=False).astype(int)
print("\n=== 最后一个小时 %s 的 top5 ===" % te.hour.max())
print(last.nsmallest(5, "rank_p")[["station_id", "p_net", "a_net", "rank_p", "rank_a"]].round(2).to_string(index=False))

# 方向一致率（全样本 station-hour）
print("\n全样本方向一致率: model %.3f  naive %.3f" % (
    np.mean(np.sign(te.p_net) == np.sign(te.a_net)), np.mean(np.sign(te.n_net) == np.sign(te.a_net))))
print("净流量 MAE: model %.3f  naive %.3f" % (np.abs(te.p_net - te.a_net).mean(), np.abs(te.n_net - te.a_net).mean()))
M.to_csv(str(OUT / "res_ranking.csv"), index=False)
