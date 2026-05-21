# PE Fund Monte Carlo Simulator

A Streamlit web app replicating your Crystal Ball Monte Carlo model for European-style PE fund returns.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run locally
```bash
streamlit run pe_fund_montecarlo.py
```

### 3. Deploy to Streamlit Cloud (free)
1. Push files to a GitHub repo (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `pe_fund_montecarlo.py` → Deploy

---

## What It Does

### Fixed Parameters (sidebar)
| Parameter | Default | Description |
|---|---|---|
| Fund Size | €327.6M | Total committed capital |
| Management Fee | 2% | Annual fee on committed capital |
| Carried Interest | 20% | GP share of profits above hurdle |
| Hurdle Rate | 8% | Preferred return to LPs |
| Investable Capital | 85% | Share of fund deployed in investments |
| Deal Size Std Dev | 20% | Variability of individual investment sizes |

### Stochastic Assumptions (Crystal Ball Equivalents)
These replace Crystal Ball's assumption cells (green background in Excel):

| Variable | Default Distribution | Notes |
|---|---|---|
| **Divestment Multiplier** | Beta-PERT (0.5, 2.0, 5.5) | MoM per portfolio company |
| **Investment Duration** | Uniform [1, 5] years | Holding period per cohort |
| **Market Return (CAGR)** | Normal (7.5%, 2.5%) | For PME benchmark |

### Available Distributions
- **Beta-PERT**: Ideal for bounded estimates (min/mode/max). Matches Crystal Ball's default.
- **Log-Normal**: Right-skewed; good for returns
- **Normal**: Symmetric
- **Uniform**: All values equally likely
- **Triangular**: Simple bounded estimate
- **Fixed**: Deterministic value (sensitivity analysis)

### Output Metrics (Crystal Ball Forecasts)
All cyan-background cells from the Excel model:
- **LP MoM**: Multiple on Money (Total LP distributions / Committed capital)
- **LP IRR**: Internal Rate of Return (LP cash flows)
- **Fund Life**: Year of last distribution
- **PME**: Public Market Equivalent comparison
- **% Beats Market**: Frequency LP MoM > Market MoM

---

## Model Architecture

The app faithfully replicates:
1. **Investment pacing**: 5 cohorts over investment period, sizes drawn from Normal
2. **Management fees**: 2% on committed capital during investment period
3. **Exit timing**: Per-cohort duration drawn stochastically → exit year
4. **Multipliers**: Per-cohort return drawn from chosen distribution
5. **Waterfall**: European-style with hurdle → catch-up → carry split
6. **IRR calculation**: Newton-Raphson on LP cash flow stream

## Next Steps
- Add correlation matrix between multipliers (for portfolio concentration risk)
- Add fund-of-funds overlay
- Export simulation results to CSV
