# Path configuration for the GitHub repository layout.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sarima_lib import build, css_eps, fit, metrics

h = pd.read_csv(str(DATA / "network_20stations_hourly_202608-202610.csv"))
h["hour"] = pd.to_datetime(h["hour"]); h = h.sort_values(["station_id", "hour"])
h["how"] = h.hour.dt.day_of_week * 24 + h.hour.dt.hour
h["net"] = h.arrivals - h.departures
h["day"] = h.hour.dt.date

TE0 = pd.Timestamp("2026-10-16")
tr_h, te_h = h[h.hour < TE0], h[h.hour >= TE0].copy()

# --- 已算过的 RF 预测无法复用，这里只做"排序"的替代基准：历史上同一周内小时的净流量均值
clim = tr_h.groupby(["station_id", "how"]).net.mean().rename("clim").reset_index()
te_h = te_h.merge(clim, on=["station_id", "how"], how="left")

# 日聚合
def daily(df, col):
    return (df.groupby(["day", "station_id"]).agg(p=(col, "sum"), a=("net", "sum")).reset_index())

print("=== 逐日横截面排序（16 天平均）===")
res = {}
for nm, col in [("气候学（站点×周内小时均值）", "clim")]:
    dd = daily(te_h, col)
    rho = dd.groupby("day").apply(lambda g: spearmanr(g.p, g.a).correlation, include_groups=False).mean()
    hit5 = dd.groupby("day").apply(lambda g: len(set(g.loc[g.p.abs().nlargest(5).index, "station_id"]) & set(g.loc[g.a.abs().nlargest(5).index, "station_id"])) / 5, include_groups=False).mean()
    hit10 = dd.groupby("day").apply(lambda g: len(set(g.loc[g.p.abs().nlargest(10).index, "station_id"]) & set(g.loc[g.a.abs().nlargest(10).index, "station_id"])) / 10, include_groups=False).mean()
    res[nm] = (rho, hit5, hit10)
    print("%-24s Spearman=%.3f  top5=%.2f  top10=%.2f" % (nm, rho, hit5, hit10))

# 逐小时横截面排序用气候学
hh = te_h.copy()
rho = hh.groupby("hour").apply(lambda g: spearmanr(g.clim, g.net).correlation, include_groups=False).mean()
hit5 = hh.groupby("hour").apply(lambda g: len(set(g.loc[g.clim.abs().nlargest(5).index, "station_id"]) & set(g.loc[g.net.abs().nlargest(5).index, "station_id"])) / 5, include_groups=False).mean()
print("\n逐小时横截面：气候学 Spearman=%.3f  top5=%.2f  (随机 baseline top5=0.25)" % (rho, hit5))

# --- arrivals 诊断
df = pd.read_csv(str(DATA / "station_6140.05_hourly_202511-202610.csv"))
df["ts"] = pd.to_datetime(df["hour"]); df = df.sort_values("ts").reset_index(drop=True)
a = df.arrivals.values.astype(float)

def adf(y, p=24):
    y = np.asarray(y, float); dy = np.diff(y)
    X, Y = [], []
    for t in range(p + 1, len(y)):
        X.append([1.0, y[t-1]] + [dy[t-i-1] for i in range(1, p + 1)]); Y.append(dy[t-1])
    X, Y = np.array(X), np.array(Y)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ beta; s2 = e @ e / (len(Y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return beta[1] / se[1]

def kpss(y, lags=24):
    y = np.asarray(y, float); n = len(y); e = y - y.mean()
    eta = np.sum(np.cumsum(e) ** 2) / n ** 2; s = 0.0
    for j in range(1, lags + 1):
        s += 2 * (1 - j / (lags + 1)) * np.sum(e[j:] * e[:-j]) / n
    return eta / (np.sum(e**2) / n + s)

sd = a[168:] - a[:-168]
print("\n=== arrivals 平稳性 ===")
print("原始      ADF=%8.3f  KPSS=%8.3f" % (adf(a), kpss(a)))
print("季节差分   ADF=%8.3f  KPSS=%8.3f" % (adf(sd), kpss(sd)))

m = fit(a, (1, 0, 1), (1, 1, 1), 168, n_rand=40)
v = m["v"]
layers, _ = build(a, 0, 1, 168); w = layers[-1]
eps = css_eps(w, [v[0]], [v[1]], [v[2]], [v[3]], 168)
r = eps - eps.mean(); den = (r * r).sum(); n = len(r)
def lb(h):
    return n * (n + 2) * sum(((r[k:] * r[:-k]).sum() / den) ** 2 / (n - k) for k in range(1, h + 1))
def chi2(df, z=1.6448536):
    return df * (1 - 2 / (9 * df) + z * (2 / (9 * df)) ** 0.5) ** 3
print("\n=== arrivals 残差诊断 ===")
print("phi=%.3f theta=%.3f Phi=%.3f Theta=%.3f sigma=%.3f AIC=%.0f" % (*v, m["sigma"], m["aic"]))
print("ACF(24)=%.3f  ACF(168)=%.3f" % ((r[24:] * r[:-24]).sum() / den, (r[168:] * r[:-168]).sum() / den))
print("LB(24)=%.1f (crit %.1f)  LB(48)=%.1f (crit %.1f)" % (lb(24), chi2(24), lb(48), chi2(48)))
