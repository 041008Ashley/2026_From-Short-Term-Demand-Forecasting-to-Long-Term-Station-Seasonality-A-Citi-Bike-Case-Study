# From Short-Term Demand Forecasting to Long-Term Station Seasonality

**A City Bike Case Study — TUE5206 Group 7 Project**

The Chinese University of Hong Kong · November 2026

---

## Contents

1. [Overview](#1-overview)
2. [Research questions](#2-research-questions)
3. [Data](#3-data)
4. [Methodology](#4-methodology)
5. [Key results](#5-key-results)
6. [Repository structure](#6-repository-structure)
7. [Reproduction guide](#7-reproduction-guide)
8. [Limitations](#8-limitations)
9. [References](#9-references)
10. [Authors](#10-authors)

---

## 1. Overview

Bike-sharing systems need demand information at the level of individual stations. System-wide ridership can stay stable while one station runs out of bicycles and another runs out of available docks. This project studies that mismatch with two linked data windows from the New York City Citi Bike system:

- **Stage 1 — short-term, cross-station forecasting.** A complete station-hour panel is built for 20 fixed, high-activity stations from August to October 2026 (44,160 observations). Weekly-naive, Poisson regression, and Random Forest models are compared under a chronological train / validation / test design.
- **Stage 2 — annual, single-station seasonality.** Station **6140.05**, the busiest station in the three-month window, is studied over 8,760 hourly observations from November 2025 to October 2026. ADF and KPSS tests motivate a 168-hour seasonal difference, and a `SARIMA(1,0,1)(1,1,1)[168]` model is evaluated at four rolling forecast origins.

The two stages connect recent cross-station evidence with a detailed time-series case study, and translate forecast errors and within-day net flows into dock-capacity implications. The full write-up is [`paper/5206_G7project.pdf`](paper/5206_G7project.pdf).

**Key words:** bike sharing · demand forecasting · station-level analysis · seasonality · time-series models · SARIMA · random forest

---

## 2. Research questions

1. Which of the 20 fixed stations has the highest departure and arrival volumes in the three-month network sample?
2. How accurately do weekly-naive, Poisson regression, and Random Forest models forecast station-hour demand under a chronological test design?
3. At station 6140.05, what seasonal structure and stationarity properties appear in the annual hourly series, and how accurate is the resulting SARIMA forecast?
4. What do forecast errors and within-day net flows imply for station-level operational planning?

---

## 3. Data

Source: the public trip-history archives of the [Citi Bike System Data portal](https://www.citibikenyc.com/system-data). Each trip record contains a ride ID, start/end timestamps, start/end station IDs and coordinates, rideable type, and user type. Departures are counted from origin stations; arrivals are counted from destination stations.

| File (in `data/raw/`) | Period | Scope | Unit | Rows |
|---|---|---|---|---|
| `network_20stations_hourly_202608-202610.csv` | 1 Aug – 31 Oct 2026 | 20 fixed stations | station-hour | 44,160 |
| `station_6140.05_hourly_202511-202610.csv` | 1 Nov 2025 – 31 Oct 2026 | station 6140.05 | station-hour | 8,760 |
| `station_6140.05_daily_202511-202610.csv` | 1 Nov 2025 – 31 Oct 2026 | station 6140.05 | station-day | 365 |

All three files share the schema `station_id`, `hour` / `ride_date`, `departures`, `arrivals`, `net_flow`, where `net_flow = arrivals - departures`.

**Volume at a glance**

- Network sample: 825,684 departures and 813,825 arrivals across 92 days.
- Station 6140.05 leads both directions (53,562 departures, 53,623 arrivals), followed by station 6233.04.
- Annual sample at 6140.05: 164,992 departures and 165,327 arrivals (mean 452.03 and 452.95 per day).
- 4,170 zero-departure and 3,968 zero-arrival station-hours in the network sample; these are valid low-activity records and are retained.
- One zero-activity date, **23 February 2026**, is retained and flagged for data-quality review rather than silently removed.

**Important scope note.** The source files describe completed trips, not live bicycle inventory or dock occupancy. The analysis measures observed demand flows; it does not observe how many bicycles or free docks were physically available.

The SQL script [`sql/hourly_station_demand.sql`](sql/hourly_station_demand.sql) documents how the 20-station hourly table is generated in MySQL from the raw trip tables: it creates a dense 2,208-hour calendar, crosses it with a fixed `selected_stations` table, and aggregates departures, arrivals, and net flow. The archived daily SQL file was empty, so daily aggregation is performed directly in pandas.

---

## 4. Methodology

### 4.1 Stage 1: station-hour forecasting (20 stations, three months)

- **Panel construction.** The 20 stations are fixed in advance and are not re-ranked. Crossing them with 2,208 hours yields 44,160 complete station-hour rows; hours with no trips are retained as zeros.
- **Features.** Station ID, hour of day, day of week, short-term lags, and rolling means computed with past-only windows. The first seven days (1–7 August 2026) are used as history.
- **Chronological split.** Training 8 August – 30 September 2026; validation 1–15 October 2026; test 16–31 October 2026.
- **Models.** Weekly-naive baseline (same station, same hour, previous week), Poisson regression (appropriate for count data), and Random Forest (pooled across stations).
- **Metrics.** MAE, RMSE, and WAPE. Over-dispersion (variance-to-mean ratio ≈ 1.96 for departures, ≈ 2.00 for arrivals) is examined alongside the Poisson model.

### 4.2 Stage 2: annual seasonality and SARIMA (station 6140.05)

1. Examine hourly, weekly, monthly, and annual demand patterns.
2. Test stationarity jointly with ADF (unit-root null) and KPSS (stationarity null).
3. Estimate and diagnose a seasonal time-series model.
4. Evaluate 168-hour forecasts at four rolling origins: 4, 11, 18, and 25 October 2026.
5. Re-estimate on the full year, forecast seven days ahead, and analyse net-flow imbalance.

**Stationarity.** The raw hourly departures series fails KPSS (15.841) while ADF rejects a unit root (−5.309). After one seasonal difference at lag 168, both tests support stationarity (ADF −9.496; KPSS 0.327; 5% critical values −2.86 and 0.463).

**Model.** `SARIMA(1,0,1)(1,1,1)[168]`:

```
(1 - phi*B) (1 - Phi*B^168) y_t = (1 + theta*B) (1 + Theta*B^168) eps_t
```

A weekly period of 168 hours preserves weekday–weekend differences; using 24 hours instead raises WAPE from 21.0% to 33.6%. Parameters are estimated by conditional least squares with random search and coordinate descent (coefficient bound 0.97): `phi = 0.760`, `theta = 0.247`, `Phi = 0.041`, `Theta = 0.782`, residual standard deviation 7.24, AIC 34,016. The large seasonal MA coefficient indicates that the model mainly learns how the current week differs from the preceding week. Arrivals are modelled with the same specification.

**Evaluation.** At each origin, parameters are fitted using only earlier observations; a 168-hour forecast is generated and compared with a seasonal-naive benchmark. Metrics: MAE, RMSE, WAPE, and MASE (values below one indicate improvement over the in-sample one-step seasonal-naive error scale).

**Forecast intervals.** The full-sample model is re-fitted and used to forecast 168 hours beyond 31 October 2026. A 95% interval is obtained from 150 simulations that add Gaussian noise to the residuals and propagate it through the horizon.

**Net flow.** Predicted net flow is defined as `net_t = arrivals_t - departures_t`. It is treated as a noisy imbalance signal rather than a highly predictable target in its own right; a direct net-flow model offered only a modest gain.

---

## 5. Key results

### 5.1 Rolling-origin forecast accuracy (168-hour horizon, four origins)

| Model | Target | MAE | RMSE | WAPE (%) | MASE |
|---|---|---:|---:|---:|---:|
| SARIMA | Departures | 5.230 | 7.851 | 21.0 | 0.747 |
| Seasonal naive | Departures | 6.725 | 10.067 | 27.3 | 0.960 |
| SARIMA | Arrivals | 5.460 | — | 22.0 | — |
| Seasonal naive | Arrivals | — | — | 27.3 | — |

For departures, WAPE falls by 6.3 percentage points (a 31% relative reduction); for arrivals, by 5.3 points. In the three-month station-hour experiment, Random Forest is the best model, with test WAPE of **25.2%** for departures and **24.9%** for arrivals.

### 5.2 Error by time of day (averaged over four backtests)

| Period | SARIMA MAE | SARIMA WAPE (%) | Naive MAE | Naive WAPE (%) |
|---|---:|---:|---:|---:|
| Morning peak | 5.29 | 18.64 | 7.69 | 27.11 |
| Off-peak | 7.97 | 21.52 | 10.08 | 27.20 |
| Evening peak | 9.38 | 16.32 | 11.39 | 19.82 |
| Night | 2.34 | 32.67 | 3.06 | 42.71 |

The largest relative gain occurs in the morning peak. Night-time WAPE remains high in percentage terms because counts are small, although its absolute error is the lowest.

### 5.3 Seasonality and within-day reversal

- Monthly activity at 6140.05 ranges from **10,813 events in February 2026** to **37,054 in October 2026**; the annual total is 330,319 events.
- Arrivals dominate the morning peak (08:00–09:00: 14,026 and 14,436 arrivals versus 9,390 and 9,381 departures over the year).
- Departures dominate the evening peak (17:00–18:00: 18,015 and 17,534 departures versus 14,616 and 13,780 arrivals).
- Daily departure and arrival totals are nearly equal, so a near-zero daily net flow hides opposing capacity pressures within the same day.

### 5.4 Seven-day forecast and net flow

| Period | Departures | Arrivals | Net flow |
|---|---:|---:|---:|
| Morning peak | 613 | 896 | +283 |
| Off-peak | 1,890 | 1,811 | −79 |
| Evening peak | 1,249 | 1,042 | −207 |
| Night | 671 | 709 | +38 |
| **Weekly total** | **4,423** | **4,458** | **+35** |

The predicted weekly net flow is only +35 bicycles, but the morning inflow of +283 and evening outflow of −207 imply material intra-day imbalance. The 95% forecast interval widens over the horizon, so capacity plans should include a buffer around the central forecast.

---

## 6. Repository structure

```
.
├── README.md                          # this file
├── paper/
│   └── 5206_G7project.pdf             # final course paper (14 pages)
├── data/
│   └── raw/                           # the three aggregated CSV tables described above
├── sql/
│   └── hourly_station_demand.sql      # MySQL generation of the 20-station hourly panel
├── notebooks/
│   ├── 01_eda_network_20stations.ipynb
│   ├── 02_forecasting_models.ipynb
│   └── 03_sarima_station_6140.05.ipynb
├── code/                              # self-contained verification / sensitivity scripts
│   ├── sarima_lib.py                  # SARIMA fitting, residuals and forecast paths
│   ├── net_eval.py                    # component-difference vs direct net-flow evaluation
│   ├── rank_eval.py                   # cross-station ranking usefulness (hourly / daily)
│   ├── rank_and_arrivals.py           # ranking extension and arrivals diagnostics
│   ├── calib.py                       # over-dispersion and recalibration checks
│   ├── checks.py / checks2.py / checks3.py   # ADF/KPSS, ACF, backtest and data checks
│   ├── rf_check.py                    # random-forest and baseline accuracy checks
│   └── make_fig6.py                   # annual cycle and ACF figure (lags to 180)
├── results/                           # CSV outputs produced by the scripts in code/
├── figures/
│   ├── annual_cycle_acf_lag180.png
│   └── annual_cycle_acf_lag180.pdf
├── presentation/
│   └── group7_presentation_v3.pdf     # group presentation deck
└── references/                        # the five prior papers cited in the project
```

---

## 7. Reproduction guide

### Environment

Python 3.10 or newer is recommended. Required packages:

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn jupyter
```

### Running the analysis

1. **Self-contained scripts.** Every script in `code/` resolves paths relative to this repository (`ROOT/data/raw` for inputs, `ROOT/results` for outputs), so they can be run from any working directory:

   ```bash
   python code/checks.py
   python code/net_eval.py
   python code/rank_eval.py
   python code/make_fig6.py
   ```

   Numerical outputs are written to `results/`; figures are written to `results/` as well.

2. **Group notebooks.** Open Jupyter from the repository root and run the notebooks in numbered order. The notebooks were authored on different group members' machines and still contain absolute input paths (for example `CSV = ...` in notebook 03 and `project = Path(...)` in notebook 02). Update those variables to point at `data/raw/` before running:

   ```python
   from pathlib import Path
   DATA = Path("data/raw")
   # notebook 03
   CSV = DATA / "station_6140.05_hourly_202511-202610.csv"
   # notebook 02
   project = Path.cwd()
   ```

   Notebooks 01 and 02 import local helper modules named `features`, `models`, `evaluate`, and `rebalancing`. Those helper files were not part of the archived materials. If they are unavailable, use the self-contained scripts in `code/` to reproduce the reported diagnostics and robustness checks.

3. **SQL.** `sql/hourly_station_demand.sql` targets MySQL and expects the raw Citi Bike trip tables plus a `selected_stations` table. It uses a recursive CTE (`cte_max_recursion_depth >= 3000`) to build the dense hour calendar.

---

## 8. Limitations

- The network sample covers only three months and 20 fixed stations; patterns may not generalise to the full network or to other cities.
- Station 6140.05 is selected from three-month totals, so its annual profile is not representative of all stations.
- Weather, holidays, special events, service disruptions, station capacity, and live inventory are not observed.
- The annual series contains one zero-activity date (23 February 2026) that should be checked against operating records.
- The SARIMA model uses a single weekly seasonal period. Residual autocorrelation remains at lag 24 (ACF 0.105; Ljung–Box Q(24) = 366.6 versus a 95% critical value of 36.4), and peak hours are still somewhat underestimated.
- Net flow is intrinsically difficult to forecast: near-zero daily totals, frequent sign changes, and cancellation between departures and arrivals limit predictability. Without live inventory, predicted net flow should be read as a planning signal, not an exact surplus or shortage.

---

## 9. References

1. P. Borgnat, C. Robardet, J.-B. Rouquier, P. Abry, E. Fleury, and P. Flandrin, "Shared bicycles in a city: A signal processing and data analysis perspective," *Advances in Complex Systems*, vol. 14, no. 3, pp. 415–438, 2011. doi:10.1142/S0219525911002950. [PDF](references/04_Borgnat_etal_2011_ACS.pdf)
2. S. Y. Y. Hamad, T. Ma, and C. Antoniou, "Analysis and prediction of bikesharing traffic flow — Citi Bike, New York," in *2021 7th International Conference on Models and Technologies for Intelligent Transportation Systems (MT-ITS)*, 2021. doi:10.1109/MTITS49943.2021.9529290. [PDF](references/02_Hamad_etal_2021_MTITS.pdf)
3. L. Lin, Z. He, and S. Peeta, "Predicting station-level hourly demand in a large-scale bike-sharing network: A graph convolutional neural network approach," *Transportation Research Part C: Emerging Technologies*, vol. 97, pp. 258–276, 2018. doi:10.1016/j.trc.2018.10.011. [PDF](references/05_Lin_etal_2018_TRC.pdf)
4. V. E. Sathishkumar and Y. Cho, "Season wise bike sharing demand analysis using random forest algorithm," *Computational Intelligence*, vol. 40, no. 1, p. e12287, 2024. doi:10.1111/coin.12287. [PDF](references/03_Sathishkumar_Cho_2024_ComputationalIntelligence.pdf)
5. Z. Wen, D. Tian, and N. Wu, "Modeling and analyzing the spatiotemporal travel patterns of bike sharing: A case study of Citi Bike in New York," *Sustainability*, vol. 17, no. 1, p. 14, 2025. doi:10.3390/su17010014. [PDF](references/01_Wen_etal_2025_Sustainability.pdf)

---

## 10. Authors

TUE5206 Group 7, The Chinese University of Hong Kong

- LI Chang
- PAN Lingyun
- YUAN Zimeng
- ZHA Tingyu
- ZHANG Zijie
- ZHU Chenyi

Trip data are publicly provided by the Citi Bike System Data portal. This repository is an academic course project; the reference papers remain the property of their respective publishers.
