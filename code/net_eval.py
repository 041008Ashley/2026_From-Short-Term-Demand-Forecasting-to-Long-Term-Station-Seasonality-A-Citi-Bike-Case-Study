# Path configuration for the GitHub repository layout.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

import numpy as np, pandas as pd, json, time

CSV = str(DATA / "station_6140.05_hourly_202511-202610.csv")
df = pd.read_csv(CSV); df["ts"] = pd.to_datetime(df["hour"]); df = df.sort_values("ts").reset_index(drop=True)
y = df.departures.values.astype(float); a = df.arrivals.values.astype(float)
net = a - y

def build(y, d, D, s):
    layers = [np.asarray(y, float)]; steps = []
    for _ in range(d):
        layers.append(np.diff(layers[-1])); steps.append(("d", 1))
    for _ in range(D):
        layers.append(layers[-1][s:] - layers[-1][:-s]); steps.append(("D", s))
    return layers, steps

def css_eps(w, phi, theta, Phi, Theta, s):
    p, q, P, Q = len(phi), len(theta), len(Phi), len(Theta)
    n = len(w); eps = np.zeros(n)
    ar = [(i+1, phi[i]) for i in range(p)] + [((j+1)*s, Phi[j]) for j in range(P)]
    cross = [(i+1+(j+1)*s, phi[i]*Phi[j]) for i in range(p) for j in range(P)]
    ma = [(k+1, theta[k]) for k in range(q)] + [((l+1)*s, Theta[l]) for l in range(Q)]
    mac = [(k+1+(l+1)*s, theta[k]*Theta[l]) for k in range(q) for l in range(Q)]
    for t in range(n):
        A = sum(c*w[t-lag] for lag, c in ar if t >= lag)
        A -= sum(c*w[t-lag] for lag, c in cross if t >= lag)
        M = sum(c*eps[t-lag] for lag, c in ma if t >= lag)
        M -= sum(c*eps[t-lag] for lag, c in mac if t >= lag)
        eps[t] = w[t] - A + M
    return eps

def fit(y, order, sorder, s, n_rand=40, seed=0):
    p, d, q = order; P, D, Q = sorder
    layers, _ = build(y, d, D, s); w = layers[-1]; k = p+q+P+Q
    rng = np.random.default_rng(seed)
    def unpack(v): return list(v[:p]), list(v[p:p+q]), list(v[p+q:p+q+P]), list(v[p+q+P:])
    def sse(v):
        e = css_eps(w, *unpack(v), s); val = float(e @ e)
        return val if np.isfinite(val) else np.inf
    best = None
    for _ in range(n_rand):
        v = rng.uniform(-0.8, 0.8, k); val = sse(v)
        if best is None or val < best[0]: best = (val, v)
    val, v = best
    for step in (0.15, 0.07, 0.03):
        improved = True
        while improved:
            improved = False
            for i in range(k):
                for dv in (-step, step):
                    vv = v.copy(); vv[i] = np.clip(vv[i]+dv, -0.97, 0.97)
                    val2 = sse(vv)
                    if val2 < val: val, v, improved = val2, vv, True
    n_eff = len(w)
    return dict(order=order, sorder=sorder, s=s, v=v, k=k,
                sigma=float(np.sqrt(val/(n_eff-k))), aic=float(n_eff*np.log(val/n_eff)+2*k))

def path(y, m, h, nsim=0, seed=5):
    p, d, q = m["order"]; P, D, Q = m["sorder"]; s = m["s"]; v = m["v"]
    phi, theta = list(v[:p]), list(v[p:p+q]); Phi, Theta = list(v[p+q:p+q+P]), list(v[p+q+P:])
    layers, steps = build(y, d, D, s)
    base = [list(l) for l in layers]
    w = list(layers[-1]); eps = list(css_eps(np.array(w), phi, theta, Phi, Theta, s))
    out = []
    for hh in range(h):
        t = len(w)
        A = sum(phi[i]*w[t-i-1] for i in range(p) if t >= i+1)
        A += sum(Phi[j]*w[t-(j+1)*s] for j in range(P) if t >= (j+1)*s)
        A -= sum(phi[i]*Phi[j]*w[t-i-1-(j+1)*s] for i in range(p) for j in range(P) if t >= i+1+(j+1)*s)
        M = -sum(theta[k]*eps[t-k-1] for k in range(q) if t >= k+1)
        M -= sum(Theta[l]*eps[t-(l+1)*s] for l in range(Q) if t >= (l+1)*s)
        M += sum(theta[k]*Theta[l]*eps[t-k-1-(l+1)*s] for k in range(q) for l in range(Q) if t >= k+1+(l+1)*s)
        e = np.random.default_rng(seed+hh).normal(0, m["sigma"]) if nsim else 0.0
        wn = A + M + e
        w.append(wn); eps.append(wn-A-M)
        val = wn
        for L in range(len(base)-1, 0, -1):
            typ, lag = steps[L-1]
            nv = val + base[L-1][-lag]; base[L-1].append(nv); val = nv
        out.append(val)
    return np.array(out)

def metrics(act, p):
    e = act - p
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e**2))),
                WAPE=float(np.sum(np.abs(e))/np.sum(np.abs(act))*100),
                bias=float(np.mean(e)))

n = len(y); rows = []; netrows = []; foldinfo = []
t0 = time.time()
for k in range(4):
    cut = n - 168*(4-k)
    yd, ya = y[:cut], a[:cut]
    md = fit(yd, (1,0,1), (1,1,1), 168, n_rand=40)
    ma_ = fit(ya, (1,0,1), (1,1,1), 168, n_rand=40)
    fd = path(yd, md, 168); fa = path(ya, ma_, 168)
    actd, acta = y[cut:cut+168], a[cut:cut+168]
    actual_net = acta - actd
    f_net = fa - fd
    naive_d, naive_a = y[cut-168:cut], a[cut-168:cut]
    naive_net = naive_a - naive_d
    # hour-of-week climatology from training only
    tr = df.iloc[:cut].copy(); tr["how"] = tr.ts.dt.dayofweek*24 + tr.ts.dt.hour
    tr["n"] = tr.arrivals - tr.departures
    clim = tr.groupby("how").n.mean()
    fut = df.iloc[cut:cut+168].copy(); fut["how"] = fut.ts.dt.dayofweek*24 + fut.ts.dt.hour
    clim_net = clim.reindex(fut["how"].values).values
    foldinfo.append(str(df.ts[cut].date()))
    for nm, p_ in [("SARIMA", fd), ("seasonal naive", naive_d)]:
        r = metrics(actd, p_); r.update(model=nm, target="departures", fold=str(df.ts[cut].date())); rows.append(r)
    for nm, p_ in [("SARIMA", fa), ("seasonal naive", naive_a)]:
        r = metrics(acta, p_); r.update(model=nm, target="arrivals", fold=str(df.ts[cut].date())); rows.append(r)
    for nm, p_ in [("component difference", f_net), ("seasonal naive", naive_net), ("hour-of-week mean", clim_net)]:
        r = metrics(actual_net, p_); r.update(model=nm, fold=str(df.ts[cut].date()))
        r["corr"] = float(np.corrcoef(actual_net, p_)[0,1])
        r["sign_acc"] = float(np.mean(np.sign(p_) == np.sign(actual_net)))
        netrows.append(r)
    print("fold", df.ts[cut].date(), "done", round(time.time()-t0,1), "s", flush=True)

R = pd.DataFrame(rows); N = pd.DataFrame(netrows)
print("\n=== dep/arr, 4-fold mean ==="); print(R.groupby(["target","model"])[["MAE","RMSE","WAPE","bias"]].mean().round(3).to_string())
print("\n=== net flow, 4-fold mean ==="); print(N.groupby("model")[["MAE","RMSE","WAPE","bias","corr","sign_acc"]].mean().round(3).to_string())
print("\n=== net flow by fold ==="); print(N.pivot(index="model", columns="fold", values="MAE").round(2).to_string())

# direct net-flow SARIMA
R2 = []
for k in range(4):
    cut = n - 168*(4-k)
    mn = fit(net[:cut], (1,0,1), (1,1,1), 168, n_rand=40)
    fn = path(net[:cut], mn, 168)
    act = net[cut:cut+168]
    r = metrics(act, fn); r.update(model="direct SARIMA on net", fold=str(df.ts[cut].date()))
    r["corr"] = float(np.corrcoef(act, fn)[0,1]); r["sign_acc"] = float(np.mean(np.sign(fn)==np.sign(act)))
    R2.append(r)
    print("direct fold", df.ts[cut].date(), "done", round(time.time()-t0,1), flush=True)
D2 = pd.DataFrame(R2)
print("\n=== direct net SARIMA ==="); print(D2[["MAE","RMSE","WAPE","bias","corr","sign_acc"]].mean().round(3).to_string())

# scale of net series
print("\nnet flow: mean %.2f  mean|net| %.2f  sd %.2f  share of |net|<=3: %.1f%%" % (
    net.mean(), np.abs(net).mean(), net.std(), (np.abs(net)<=3).mean()*100))
print("corr(dep,arr) same hour: %.3f" % np.corrcoef(y,a)[0,1])

R.to_csv(str(OUT / "res_dep_arr.csv"), index=False); N.to_csv(str(OUT / "res_net.csv"), index=False); D2.to_csv(str(OUT / "res_net_direct.csv"), index=False)
print("elapsed", round(time.time()-t0,1))
