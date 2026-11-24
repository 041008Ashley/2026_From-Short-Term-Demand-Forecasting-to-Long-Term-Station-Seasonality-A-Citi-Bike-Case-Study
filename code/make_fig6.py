# Path configuration for the GitHub repository layout.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})

CSV = str(DATA / "station_6140.05_hourly_202511-202610.csv")
df = pd.read_csv(CSV); df["ts"] = pd.to_datetime(df["hour"])
df = df.sort_values("ts").reset_index(drop=True)
y = df.departures.values.astype(float)

def acf(x, lags):
    x = x - x.mean(); den = (x * x).sum()
    return np.array([1.0] + [(x[l:] * x[:-l]).sum() / den for l in range(1, lags + 1)])

LAGS = 180
a = acf(y, LAGS)
mon = df.groupby(df.ts.dt.month).departures.mean()

print("ACF(24)=%.3f  ACF(48)=%.3f  ACF(168)=%.3f" % (a[24], a[48], a[168]))
print("max in 2..180 at lag %d = %.3f" % (int(np.argmax(a[2:]) + 2), a[2:].max()))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2),
                         gridspec_kw={"width_ratios": [1, 1.55]})

ax = axes[0]
ax.plot(mon.index, mon.values, marker="o", color="#056A96", lw=1.6, ms=5)
ax.set_xlabel("Month"); ax.set_ylabel("Mean departures")
ax.set_title("Annual cycle", fontsize=10, pad=8)
ax.set_xticks(range(1, 13))
ax.grid(axis="y", alpha=.22, lw=.6)
ax.set_ylim(0, max(mon.values) * 1.18)

ax = axes[1]
ax.vlines(range(0, LAGS + 1), 0, a, color="#B85035", lw=.9)
ax.axhline(0, color="#666", lw=.7)
for lag in (24, 168):
    ax.axvline(lag, color="#056A96", lw=.9, ls="--", alpha=.55)
ax.annotate(f"lag 24\n{a[24]:.2f}", xy=(24, a[24]), xytext=(38, 0.92),
            fontsize=8, color="#056A96",
            arrowprops=dict(arrowstyle="-", color="#056A96", lw=.8, alpha=.7))
ax.annotate(f"lag 168\n{a[168]:.2f}", xy=(168, a[168]), xytext=(132, 0.62),
            fontsize=8, color="#056A96",
            arrowprops=dict(arrowstyle="-", color="#056A96", lw=.8, alpha=.7))
ax.set_xlim(-2, LAGS + 2); ax.set_ylim(-0.15, 1.05)
ax.set_xticks(range(0, LAGS + 1, 24))
ax.set_xlabel("Lag (hours)"); ax.set_ylabel("Autocorrelation")
ax.set_title("ACF: peaks at lags 24 and 168", fontsize=10, pad=8)
ax.grid(axis="y", alpha=.22, lw=.6)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(str(OUT / f"fig_annual_acf.{ext}"))
print("saved fig_annual_acf.png / .pdf")
