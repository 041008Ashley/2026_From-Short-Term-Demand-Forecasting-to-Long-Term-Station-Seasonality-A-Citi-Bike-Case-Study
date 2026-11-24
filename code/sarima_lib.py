# Path configuration for the GitHub repository layout.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

import numpy as np

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

