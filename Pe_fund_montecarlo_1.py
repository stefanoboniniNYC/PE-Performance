"""
PE Fund Monte Carlo Simulator - Streamlit App
All assumptions (structural + stochastic) support full distribution choice.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# âââââââââââââââââââââââââââââââââââââââââââââ
# PAGE CONFIG
# âââââââââââââââââââââââââââââââââââââââââââââ
st.set_page_config(
    page_title="PE Fund Monte Carlo",
    page_icon="ð",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# âââââââââââââââââââââââââââââââââââââââââââââ
# CSS
# âââââââââââââââââââââââââââââââââââââââââââââ
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background: #f6f8fa; }

/* ââ assumption card ââ */
.assump-card {
  background: white;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  padding: 14px 16px 10px;
  margin-bottom: 12px;
}
.assump-title {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .1em; color: #0d1117; margin-bottom: 10px;
  display: flex; align-items: center; gap: 8px;
}
.assump-badge {
  display: inline-block; font-size: 10px; font-weight: 600;
  padding: 1px 7px; border-radius: 3px;
  font-family: 'IBM Plex Mono', monospace; letter-spacing: 0;
  text-transform: none;
}
.badge-fixed   { background:#e6f4ea; color:#1a7f37; border:1px solid #aad7b1; }
.badge-stoch   { background:#ddf4ff; color:#0550ae; border:1px solid #54aeff; }

/* ââ KPI cards ââ */
.metric-card {
  background: white; border: 1px solid #d0d7de;
  border-radius: 8px; padding: 16px 20px; margin-bottom: 8px;
}
.metric-label {
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .1em; color: #656d76; margin-bottom: 4px;
}
.metric-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 24px; font-weight: 600; color: #0d1117;
}
.metric-sub { font-size: 11px; color: #8c959f; margin-top: 3px; }

/* ââ section heading ââ */
.section-hdr {
  font-size: 12px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .12em; color: #0d1117;
  border-bottom: 2px solid #0d1117; padding-bottom: 5px;
  margin: 28px 0 16px;
}

/* ââ percentile table ââ */
.pct-table { width:100%; border-collapse:collapse; font-size:13px; }
.pct-table th {
  background:#0d1117; color:white; padding:8px 12px;
  text-align:left; font-weight:600; font-size:11px;
  text-transform:uppercase; letter-spacing:.08em;
}
.pct-table td {
  padding:7px 12px; border-bottom:1px solid #e6e9ec;
  font-family:'IBM Plex Mono',monospace;
}
.pct-table tr:hover td { background:#f6f8fa; }
.pct-table .hl { background:#fffbdd !important; }

/* preview plot small label */
.prev-label {
  font-size:11px; color:#656d76; text-align:center; margin-top:-4px;
}
</style>
""", unsafe_allow_html=True)


# âââââââââââââââââââââââââââââââââââââââââââââ
# DISTRIBUTION ENGINE
# âââââââââââââââââââââââââââââââââââââââââââââ

DIST_TYPES = ["Fixed", "Triangular", "Beta-PERT", "Uniform", "Normal", "Log-Normal", "Beta"]

DIST_PARAMS = {
    "Fixed":      [("value", "Value")],
    "Triangular": [("min", "Min"), ("mode", "Most Likely"), ("max", "Max")],
    "Beta-PERT":  [("min", "Min"), ("mode", "Most Likely"), ("max", "Max")],
    "Uniform":    [("min", "Min"), ("max", "Max")],
    "Normal":     [("mean", "Mean"), ("std", "Std Dev")],
    "Log-Normal": [("mean", "Mean (arith.)"), ("std", "Std Dev (arith.)")],
    "Beta":       [("alpha", "Alpha (alpha)"), ("beta", "Beta (beta)"),
                   ("min", "Min (scale lo)"), ("max", "Max (scale hi)")],
}

def sample_dist(dist, params, n):
    """Draw n samples from the configured distribution."""
    try:
        if dist == "Fixed":
            return np.full(n, float(params["value"]))

        elif dist == "Triangular":
            lo, mode, hi = params["min"], params["mode"], params["max"]
            lo, hi = min(lo, hi), max(lo, hi)
            mode = np.clip(mode, lo, hi)
            if lo == hi:
                return np.full(n, lo)
            return np.random.triangular(lo, mode, hi, n)

        elif dist == "Beta-PERT":
            lo, mode, hi = params["min"], params["mode"], params["max"]
            lo, hi = min(lo, hi), max(lo, hi)
            mode = np.clip(mode, lo, hi)
            if lo == hi:
                return np.full(n, lo)
            mu = (lo + 4 * mode + hi) / 6.0
            rng = hi - lo
            mu_n = (mu - lo) / rng          # normalised to [0,1]
            s2 = ((mu_n - 0) * (1 - mu_n)) / 7.0
            if s2 <= 0:
                return np.full(n, mode)
            a = mu_n * ((mu_n * (1 - mu_n) / s2) - 1)
            b = (1 - mu_n) * ((mu_n * (1 - mu_n) / s2) - 1)
            a, b = max(a, 0.01), max(b, 0.01)
            return stats.beta.rvs(a, b, loc=lo, scale=rng, size=n)

        elif dist == "Uniform":
            lo, hi = min(params["min"], params["max"]), max(params["min"], params["max"])
            if lo == hi:
                return np.full(n, lo)
            return np.random.uniform(lo, hi, n)

        elif dist == "Normal":
            return np.random.normal(params["mean"], max(params["std"], 1e-9), n)

        elif dist == "Log-Normal":
            mu, sd = params["mean"], max(params["std"], 1e-9)
            if mu <= 0:
                return np.full(n, 0.01)
            sigma2 = np.log(1 + (sd / mu) ** 2)
            mu_ln = np.log(mu) - sigma2 / 2
            return np.random.lognormal(mu_ln, np.sqrt(sigma2), n)

        elif dist == "Beta":
            a, b = max(params["alpha"], 0.01), max(params["beta"], 0.01)
            lo, hi = params["min"], params["max"]
            lo, hi = min(lo, hi), max(lo, hi)
            rng = hi - lo if hi != lo else 1.0
            return stats.beta.rvs(a, b, loc=lo, scale=rng, size=n)

    except Exception:
        return np.full(n, list(params.values())[0])

    return np.full(n, 0.0)


def dist_label(dist, params):
    """One-line human description of the chosen distribution."""
    if dist == "Fixed":
        return f"Fixed = {params['value']:.4g}"
    elif dist in ("Triangular", "Beta-PERT"):
        return f"{dist}({params['min']:.3g}, {params['mode']:.3g}, {params['max']:.3g})"
    elif dist == "Uniform":
        return f"Uniform({params['min']:.3g}, {params['max']:.3g})"
    elif dist in ("Normal", "Log-Normal"):
        return f"{dist}(mu={params['mean']:.3g}, sd={params['std']:.3g})"
    elif dist == "Beta":
        return f"Beta(alpha={params['alpha']:.3g}, beta={params['beta']:.3g}) [{params['min']:.3g}, {params['max']:.3g}]"
    return dist


# âââââââââââââââââââââââââââââââââââââââââââââ
# ASSUMPTION WIDGET
# builds the dist selector + param inputs + inline mini-preview
# returns (dist_type, params_dict)
# âââââââââââââââââââââââââââââââââââââââââââââ

def assumption_widget(key, title, icon,
                      default_dist, defaults_by_dist,
                      fmt="%.3f", preview_scale=1.0,
                      preview_xlabel="Value", is_pct=False):
    """
    Renders a full assumption configurator inside an expander.
    Returns (dist_type_str, params_dict).
    """
    # Badge: Fixed vs Stochastic
    badge_class = "badge-fixed" if default_dist == "Fixed" else "badge-stoch"
    badge_text  = "Fixed" if default_dist == "Fixed" else "Stochastic"

    st.markdown(
        f'<div class="assump-title">'
        f'{icon} {title} &nbsp;'
        f'<span class="assump-badge {badge_class}" id="badge_{key}">{badge_text}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ââ distribution type selector ââ
    dist_idx = DIST_TYPES.index(default_dist) if default_dist in DIST_TYPES else 0
    dist = st.selectbox(
        "Distribution", DIST_TYPES,
        index=dist_idx, key=f"{key}_dist",
        label_visibility="collapsed",
    )

    # ââ parameter inputs (dynamic, based on dist) ââ
    param_defs = DIST_PARAMS[dist]
    dp = defaults_by_dist.get(dist, {})
    params = {}

    cols = st.columns(len(param_defs))
    for i, (pname, plabel) in enumerate(param_defs):
        default_val = float(dp.get(pname, 1.0))
        params[pname] = cols[i].number_input(
            plabel,
            value=default_val,
            key=f"{key}_{pname}",
            format=fmt,
            step=abs(default_val) * 0.05 if default_val != 0 else 0.01,
        )

    # ââ inline mini-preview chart ââ
    try:
        draws = sample_dist(dist, params, 5000) * preview_scale
        if is_pct:
            draws = draws * 100  # show as %

        if dist == "Fixed":
            # single vertical line, no histogram
            fig = go.Figure()
            fig.add_vline(x=float(draws[0]), line_width=2.5, line_color="#58a6ff")
            fig.add_annotation(x=float(draws[0]), y=0.5, text=f"{draws[0]:.3g}",
                               showarrow=False, yref="paper",
                               font=dict(size=13, family="IBM Plex Mono", color="#0d1117"))
        else:
            fig = go.Figure()
            if dist in ("Triangular", "Beta-PERT", "Uniform", "Beta"):
                lo_x, hi_x = draws.min(), draws.max()
            else:
                lo_x = np.percentile(draws, 0.5)
                hi_x = np.percentile(draws, 99.5)
            draws_cl = draws[(draws >= lo_x) & (draws <= hi_x)]

            fig.add_trace(go.Histogram(
                x=draws_cl, nbinsx=60,
                marker_color="#58a6ff", opacity=0.8,
                histnorm="probability density",
            ))
            # P10 / median / P90 lines
            for q, col_line, lbl in [
                (0.10, "#f78166", "P10"),
                (0.50, "#3fb950", "P50"),
                (0.90, "#8957e5", "P90"),
            ]:
                v = np.percentile(draws, q * 100)
                fig.add_vline(x=v, line_dash="dot", line_color=col_line, line_width=1.2,
                              annotation_text=f"{lbl}={v:.3g}",
                              annotation_font_size=9,
                              annotation_position="top right")

        fig.update_layout(
            height=150,
            margin=dict(t=10, b=20, l=30, r=10),
            paper_bgcolor="white", plot_bgcolor="#f6f8fa",
            font=dict(family="IBM Plex Sans", size=10),
            xaxis=dict(title=preview_xlabel + (" (%)" if is_pct else ""), title_font_size=10),
            yaxis=dict(title="", showticklabels=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="prev-label">{dist_label(dist, params)}</div>', unsafe_allow_html=True)
    except Exception:
        st.caption("(preview unavailable)")

    return dist, params


# âââââââââââââââââââââââââââââââââââââââââââââ
# SIMULATION ENGINE  (validated against Excel baseline)
# âââââââââââââââââââââââââââââââââââââââââââââ

def _irr(cfs, guess=0.10, tol=1e-10, maxiter=2000):
    """Newton-Raphson IRR. Returns nan if no solution found."""
    cf = np.asarray(cfs, dtype=float)
    # Quick sign check - need at least one sign change
    if not (np.any(cf > 0) and np.any(cf < 0)):
        return float("nan")
    r = guess
    for _ in range(maxiter):
        t   = np.arange(len(cf))
        f   = np.sum(cf / (1 + r) ** t)
        df  = -np.sum(t * cf / (1 + r) ** (t + 1))
        if abs(df) < 1e-14:
            break
        r2 = r - f / df
        if abs(r2 - r) < tol:
            return r2
        r = r2
    return r


def _mirr(cfs, finance_rate, reinvest_rate):
    """MIRR matching Excel's =MIRR(cfs, finance_rate, reinvest_rate)."""
    cfs = np.asarray(cfs, dtype=float)
    n   = len(cfs)
    pos = np.maximum(cfs, 0.0)
    neg = np.minimum(cfs, 0.0)
    fv  = sum(pos[t] * (1 + reinvest_rate) ** (n - 1 - t) for t in range(n))
    pv  = sum(neg[t] / (1 + finance_rate) ** t           for t in range(n))
    if pv == 0 or fv == 0:
        return float("nan")
    return (fv / abs(pv)) ** (1.0 / (n - 1)) - 1.0


def _fund_life(cap_ret, n=12):
    """
    Fund life = first year where cumulative divested capital equals total invested capital.
    Mirrors Excel row50/C51: XLOOKUP for first year cumsum(cap_ret) == total_invested.
    Falls back to last year with any divestment.
    """
    total_inv = cap_ret.sum()
    cumsum = 0.0
    for yr in range(1, n):
        cumsum += cap_ret[yr]
        if round(cumsum, 2) >= round(total_inv, 2):
            return yr
    # fallback: last year with a divestment
    for yr in range(n - 1, 0, -1):
        if cap_ret[yr] > 0:
            return yr
    return n - 1


def simulate_one(fund_size, mgmt_fee, carry, hurdle, inv_pct,
                 dur_draws, mult_draws, market_ret, inv_std_pct):
    """
    One simulation path.
    Replicates every formula in 'PE returns Baseline WebApp' exactly.
    Returns dict of all performance metrics matching Excel rows 59-81.
    """
    N       = 12    # year indices 0..11  (cols C..N)
    INV_YRS = 5     # investment period years 1-5

    # ââ Row 17: Annual investments (stochastic pacing) ââ
    # Mean = fund_size * inv_pct / 5; each year is Normal(mean, mean*std_pct)
    # Year 5 is forced to deploy whatever is left (mirrors H17 = M7*0.85 - SUM(D17:G17))
    mean_inv = fund_size * inv_pct / INV_YRS
    std_inv  = mean_inv * inv_std_pct
    inv_raw  = np.maximum(0.0, np.random.normal(mean_inv, std_inv, INV_YRS - 1))
    inv_yr5  = max(0.0, fund_size * inv_pct - inv_raw.sum())
    investments = np.append(inv_raw, inv_yr5)   # shape (5,), years 1-5

    # ââ Row 18: Cumulative capital in portfolio (running balance) ââ
    # E18 = D18 + E17 - E28  (investments in minus divestments out)
    # We need row28 (capital returned) first, built below after exits.

    # ââ Row 24/25: Exit years per cohort ââ
    durations  = np.clip(np.round(dur_draws).astype(int), 1, 9)
    exit_years = np.arange(1, INV_YRS + 1) + durations  # shape (5,)

    # ââ Rows 28/29: Capital and gain returned per calendar year ââ
    cap_ret  = np.zeros(N)
    gain_ret = np.zeros(N)
    for i in range(INV_YRS):
        ey  = min(int(exit_years[i]), N - 1)
        cap = investments[i]
        m   = max(0.01, float(mult_draws[i]))
        cap_ret[ey]  += cap
        gain_ret[ey] += cap * (m - 1.0)   # row29: gain = cap*(mult-1), negative if loss

    total_divest = cap_ret + gain_ret   # row28+row29 per year

    # ââ Row 18 (now we can build it) ââ
    row18 = np.zeros(N)
    row18[1] = investments[0]
    for yr in range(2, N):
        inv_yr = investments[yr - 1] if yr <= INV_YRS else 0.0
        row18[yr] = max(0.0, row18[yr - 1] + inv_yr - cap_ret[yr])

    # ââ Row 20: Management fees ââ
    # Years 1-5: 2% of committed capital (fund_size)
    # Years 6+:  mgmt_fee * portfolio_value_previous_year  (IF(prev_row18>0, prev*fee, 0))
    mgmt_fees = np.zeros(N)
    for yr in range(1, INV_YRS + 1):
        mgmt_fees[yr] = mgmt_fee * fund_size
    for yr in range(INV_YRS + 1, N):
        mgmt_fees[yr] = row18[yr - 1] * mgmt_fee if row18[yr - 1] > 0 else 0.0

    # ââ Row 30: Cumulative total divestments ââ
    row30 = np.cumsum(total_divest)

    # ââ Rows 32/33: Hurdle capital & residual ââ
    # D32 = fund_size * (1+hurdle)
    # E32 = D33 * (1+hurdle),  where row33 = max(0, row32 - cap_ret - gain_ret)
    row32 = np.zeros(N)
    row33 = np.zeros(N)
    row32[1] = fund_size * (1.0 + hurdle)
    row33[1] = row32[1]                            # D33 = D32 (no divest in yr1)
    for yr in range(2, N):
        row32[yr] = row33[yr - 1] * (1.0 + hurdle)
        row33[yr] = max(0.0, row32[yr] - total_divest[yr])

    # ââ Row 34: Non-carry distributions to LPs per year ââ
    # IF((cap+gain) < row32, (cap+gain), row32)   [i.e. min of proceeds and hurdle]
    row34 = np.zeros(N)
    for yr in range(1, N):
        if total_divest[yr] > 0:
            row34[yr] = min(total_divest[yr], row32[yr])

    # ââ Row 35: Cumulative non-carry distributions ââ
    row35 = np.cumsum(row34)

    # ââ Row 36: Residual for catch-up and carry ââ
    # IF(row30 > row35, IF((cap+gain)>0, (cap+gain - row34), 0), 0)
    row36 = np.zeros(N)
    for yr in range(1, N):
        if row30[yr] > row35[yr] and total_divest[yr] > 0:
            row36[yr] = total_divest[yr] - row34[yr]

    # ââ Row 38: Catch-up computed amount ââ
    # IF(row36>0, IF(row32>0, (row35 - fund_size) * carry/(1-carry), 0), 0)
    row38 = np.zeros(N)
    for yr in range(1, N):
        if row36[yr] > 0 and row32[yr] > 0:
            row38[yr] = (row35[yr] - fund_size) * (carry / (1.0 - carry))

    # ââ Rows 39/40: Catch-up actual and residual (cumulative state machine) ââ
    row39 = np.zeros(N)
    row40 = np.zeros(N)
    cs36  = np.zeros(N)
    cs38  = np.zeros(N)
    cs39  = np.zeros(N)
    for yr in range(1, N):
        cs36[yr] = cs36[yr - 1] + row36[yr]
        cs38[yr] = cs38[yr - 1] + row38[yr]
        if row36[yr] == 0.0:
            row39[yr] = 0.0
        elif cs36[yr] > cs38[yr]:
            row39[yr] = row40[yr - 1] if row40[yr - 1] > 0 else row38[yr]
        else:
            row39[yr] = cs36[yr]
        cs39[yr]  = cs39[yr - 1] + row39[yr]
        row40[yr] = max(0.0, cs38[yr] - cs39[yr])

    # ââ Rows 41/42: Carry to LP and GP ââ
    row41 = np.zeros(N)   # LP share of carry
    row42 = np.zeros(N)   # GP share of carry
    for yr in range(1, N):
        excess = row36[yr] - row39[yr]
        if excess > 0:
            row41[yr] = excess * (1.0 - carry)
            row42[yr] = excess * carry

    # ââ Row 44: Contributions (calls from LPs) ââ
    contributions = np.zeros(N)
    for yr in range(1, INV_YRS + 1):
        contributions[yr] = investments[yr - 1] + mgmt_fees[yr]
    for yr in range(INV_YRS + 1, N):
        contributions[yr] = mgmt_fees[yr]

    # ââ Row 45: Total fund distributions (LP + GP) ââ
    row45 = np.zeros(N)
    for yr in range(1, N):
        row45[yr] = row34[yr] + row39[yr] + row41[yr] + row42[yr]

    # ââ Row 46: LP distributions only ââ
    # C46 = -fund_size (initial commitment)
    # D46..N46 = row41 + row34  (carry portion to LP + non-carry)
    lp_dist = np.zeros(N)
    lp_dist[0] = -fund_size
    for yr in range(1, N):
        lp_dist[yr] = row41[yr] + row34[yr]

    # ââ Row 47: Annual fund net cash flows ââ
    fund_net = np.zeros(N)
    for yr in range(1, N):
        fund_net[yr] = row45[yr] - contributions[yr]

    # ââ Row 48: Annual net LP cash flows ââ
    net_lp = np.zeros(N)
    net_lp[0] = lp_dist[0]      # -fund_size
    for yr in range(1, N):
        net_lp[yr] = lp_dist[yr] - contributions[yr]

    # ââ Fund life (C51) ââ
    fund_life = _fund_life(cap_ret, N)

    # ââ Performance metrics ââ

    # C53: Total LP distributions
    total_lp_dist = float(lp_dist[1:].sum())

    # C54: Total GP distributions (catch-up + carry)
    total_gp_dist = float((row39 + row42).sum())

    # C59: Fund MoM = cumulative total divest / fund_size
    fund_mom = row30[N - 1] / fund_size if fund_size > 0 else float("nan")

    # C60: Fund IRR Zero = IRR(C45:N45) on total fund distributions
    fund_dist_cf = np.concatenate([[-fund_size], row45[1:]])
    fund_irr_zero = _irr(fund_dist_cf)

    # C61: Fund IRR Calendar = IRR(D47:N47)
    fund_irr_cal = _irr(fund_net[1:])

    # C62: LP MoM = total LP dist / fund_size
    lp_mom = total_lp_dist / fund_size if fund_size > 0 else float("nan")

    # C63: LP IRR Zero = IRR(C46:N46) - distributions only (no subtraction of contributions)
    lp_irr_zero = _irr(lp_dist)

    # C64: LP IRR Calendar = IRR(D48:N48) - net LP CFs, annual (years 1-11)
    lp_irr_cal = _irr(net_lp[1:])

    # C65: MIRR zero = MIRR(C46:M46, 4%, market_ret)  - LP dist years 0-10
    lp_mirr_zero = _mirr(lp_dist[:11], 0.04, market_ret)

    # C66: MIRR calendar = MIRR(D48:N48, 4%, market_ret)
    lp_mirr_cal = _mirr(net_lp[1:], 0.04, market_ret)

    # C67: MoM @ MIRR = (1 + MIRR_zero) ^ fund_life
    mirr_mom = (1.0 + lp_mirr_zero) ** fund_life if not np.isnan(lp_mirr_zero) else float("nan")

    # C69: Fund DPI = total LP dist / total contributions
    total_contrib = float(contributions[1:].sum())
    fund_dpi = total_lp_dist / total_contrib if total_contrib > 0 else float("nan")

    # C72: Market return (passed in as market_ret)

    # C73: PME of committed = fund_size * (1+mkt)^fund_life
    pme_committed = fund_size * (1.0 + market_ret) ** fund_life

    # C74: MoM in public markets = PME / fund_size = (1+mkt)^fund_life
    mkt_mom = pme_committed / fund_size

    # C77/C78: Frequency and excess vs market (raw MoM)
    beats_market_raw  = int(lp_mom > mkt_mom)
    excess_mom_raw    = lp_mom - mkt_mom

    # C77/C78 with MIRR MoM
    beats_market_mirr = int(mirr_mom > mkt_mom) if not np.isnan(mirr_mom) else 0
    excess_mom_mirr   = (mirr_mom - mkt_mom) if not np.isnan(mirr_mom) else float("nan")

    return dict(
        # Core LP metrics
        lp_mom         = lp_mom,
        lp_irr_zero    = lp_irr_zero,
        lp_irr_cal     = lp_irr_cal,
        lp_mirr_zero   = lp_mirr_zero,
        lp_mirr_cal    = lp_mirr_cal,
        mirr_mom       = mirr_mom,
        fund_dpi       = fund_dpi,
        # Fund-level metrics
        fund_mom       = fund_mom,
        fund_irr_zero  = fund_irr_zero,
        fund_irr_cal   = fund_irr_cal,
        # Distributions
        total_lp_dist  = total_lp_dist,
        total_gp_dist  = total_gp_dist,
        # Market / PME
        mkt_mom        = mkt_mom,
        beats_market   = beats_market_raw,
        beats_market_mirr = beats_market_mirr,
        excess_mom     = excess_mom_raw,
        excess_mom_mirr = excess_mom_mirr,
        # Structural
        fund_life      = fund_life,
    )


def run_mc(n_sims, assumption_draws, inv_std_pct, seed):
    """
    Run Monte Carlo. assumption_draws: dict name -> pre-drawn array of length n_sims
    (mult and dur are length n_sims*5, one draw per cohort per sim).
    """
    np.random.seed(seed)
    rows = []
    for i in range(n_sims):
        fs = float(assumption_draws["fund_size"][i])
        mf = float(assumption_draws["mgmt_fee"][i])
        cr = float(assumption_draws["carry"][i])
        hu = float(assumption_draws["hurdle"][i])
        ip = float(np.clip(assumption_draws["inv_pct"][i], 0.01, 0.99))
        mult_arr = np.array([float(assumption_draws["mult"][i * 5 + k]) for k in range(5)])
        dur_arr  = np.array([float(assumption_draws["dur"][i * 5 + k])  for k in range(5)])
        mkt = float(assumption_draws["mkt"][i])
        rows.append(simulate_one(
            fund_size  = max(fs, 1.0),
            mgmt_fee   = np.clip(mf, 0.0, 0.10),
            carry      = np.clip(cr, 0.0, 0.50),
            hurdle     = np.clip(hu, 0.0, 0.30),
            inv_pct    = ip,
            dur_draws  = dur_arr,
            mult_draws = mult_arr,
            market_ret = mkt,
            inv_std_pct = inv_std_pct,
        ))
    return pd.DataFrame(rows)


# âââââââââââââââââââââââââââââââââââââââââââââ
# PAGE HEADER
# âââââââââââââââââââââââââââââââââââââââââââââ

st.markdown("""
<div style="background:#0d1117;padding:22px 32px;border-radius:10px;margin-bottom:24px;">
  <h1 style="color:#58a6ff;font-family:'IBM Plex Mono',monospace;margin:0;font-size:21px;letter-spacing:-.5px;">
    PE Fund Monte Carlo Simulator
  </h1>
  <p style="color:#8b949e;margin:6px 0 0;font-size:13px;">
    European-style waterfall - Catch-up - Every assumption configurable as Fixed or any stochastic distribution
  </p>
</div>
""", unsafe_allow_html=True)

# âââââââââââââââââââââââââââââââââââââââââââââ
# SIMULATION SETTINGS (top bar)
# âââââââââââââââââââââââââââââââââââââââââââââ

cfg1, cfg2 = st.columns([1, 1])
n_sims = cfg1.select_slider("Simulations", options=[500, 1000, 2500, 5000, 10000], value=2500)
seed   = cfg2.number_input(
    "Random Seed",
    value=42, min_value=0,
    help=(
        "Fixes the random number generator so results are reproducible. "
        "Run twice with the same seed -> identical numbers. "
        "Change one assumption and keep the seed fixed -> any output change "
        "is purely from your assumption, not from randomness. "
        "Leave it at 42 unless you want to check stability across different draws."
    ),
)

st.markdown("---")

# âââââââââââââââââââââââââââââââââââââââââââââ
# ASSUMPTION CONFIGURATORS - two columns of cards
# âââââââââââââââââââââââââââââââââââââââââââââ

st.markdown('<div class="section-hdr">Monte Carlo Assumptions</div>', unsafe_allow_html=True)
st.caption("Each assumption can be set as a fixed number or sampled from a distribution every simulation. "
           "The mini-chart updates live as you change parameters.")

col_A, col_B = st.columns(2, gap="large")

# ââ LEFT COLUMN: Fund structure assumptions ââ
with col_A:

    # 1. Fund Size
    with st.container(border=True):
        fs_dist, fs_params = assumption_widget(
            key="fund_size", title="Fund Size (EUR M)", icon="ð°",
            default_dist="Triangular",
            defaults_by_dist={
                "Fixed":      {"value": 300.0},
                "Triangular": {"min": 100.0, "mode": 300.0, "max": 500.0},
                "Beta-PERT":  {"min": 100.0, "mode": 300.0, "max": 500.0},
                "Uniform":    {"min": 100.0, "max": 500.0},
                "Normal":     {"mean": 300.0, "std": 80.0},
                "Log-Normal": {"mean": 300.0, "std": 80.0},
                "Beta":       {"alpha": 2.0, "beta": 2.0, "min": 100.0, "max": 500.0},
            },
            fmt="%.1f", preview_xlabel="EUR M",
        )

    # 2. Management Fee
    with st.container(border=True):
        mf_dist, mf_params = assumption_widget(
            key="mgmt_fee", title="Management Fee", icon="ð",
            default_dist="Fixed",
            defaults_by_dist={
                "Fixed":      {"value": 0.020},
                "Triangular": {"min": 0.015, "mode": 0.020, "max": 0.025},
                "Beta-PERT":  {"min": 0.015, "mode": 0.020, "max": 0.025},
                "Uniform":    {"min": 0.015, "max": 0.025},
                "Normal":     {"mean": 0.020, "std": 0.003},
                "Log-Normal": {"mean": 0.020, "std": 0.003},
                "Beta":       {"alpha": 5.0, "beta": 5.0, "min": 0.01, "max": 0.03},
            },
            fmt="%.4f", preview_xlabel="Fee (decimal)", is_pct=True,
        )

    # 3. Carried Interest
    with st.container(border=True):
        cr_dist, cr_params = assumption_widget(
            key="carry", title="Carried Interest (GP Share)", icon="ðŒ",
            default_dist="Fixed",
            defaults_by_dist={
                "Fixed":      {"value": 0.20},
                "Triangular": {"min": 0.15, "mode": 0.20, "max": 0.25},
                "Beta-PERT":  {"min": 0.15, "mode": 0.20, "max": 0.25},
                "Uniform":    {"min": 0.15, "max": 0.25},
                "Normal":     {"mean": 0.20, "std": 0.03},
                "Log-Normal": {"mean": 0.20, "std": 0.03},
                "Beta":       {"alpha": 5.0, "beta": 5.0, "min": 0.10, "max": 0.30},
            },
            fmt="%.4f", preview_xlabel="Carry", is_pct=True,
        )

    # 4. Hurdle Rate
    with st.container(border=True):
        hu_dist, hu_params = assumption_widget(
            key="hurdle", title="Hurdle Rate (Preferred Return)", icon="ð¯",
            default_dist="Fixed",
            defaults_by_dist={
                "Fixed":      {"value": 0.08},
                "Triangular": {"min": 0.06, "mode": 0.08, "max": 0.10},
                "Beta-PERT":  {"min": 0.05, "mode": 0.08, "max": 0.12},
                "Uniform":    {"min": 0.06, "max": 0.10},
                "Normal":     {"mean": 0.08, "std": 0.01},
                "Log-Normal": {"mean": 0.08, "std": 0.01},
                "Beta":       {"alpha": 5.0, "beta": 5.0, "min": 0.04, "max": 0.15},
            },
            fmt="%.4f", preview_xlabel="Hurdle", is_pct=True,
        )

    # 5. Investable Capital %
    with st.container(border=True):
        ip_dist, ip_params = assumption_widget(
            key="inv_pct", title="Investable Capital (% of Fund)", icon="ð",
            default_dist="Fixed",
            defaults_by_dist={
                "Fixed":      {"value": 0.85},
                "Triangular": {"min": 0.75, "mode": 0.85, "max": 0.92},
                "Beta-PERT":  {"min": 0.70, "mode": 0.85, "max": 0.95},
                "Uniform":    {"min": 0.75, "max": 0.95},
                "Normal":     {"mean": 0.85, "std": 0.04},
                "Log-Normal": {"mean": 0.85, "std": 0.04},
                "Beta":       {"alpha": 8.0, "beta": 2.0, "min": 0.60, "max": 1.00},
            },
            fmt="%.4f", preview_xlabel="Investable %", is_pct=True,
        )

    # 6. Annual Deployment Variability
    with st.container(border=True):
        st.markdown(
            '<div class="assump-title">Annual Deployment Variability'
            ' &nbsp;<span class="assump-badge badge-stoch">Stochastic</span></div>',
            unsafe_allow_html=True,
        )
        inv_std_pct = st.slider(
            "Std dev as % of mean annual deployment",
            min_value=0, max_value=80, value=20, step=5,
            format="%d%%",
            help=(
                "Each year's investment amount is drawn from Normal(mean, mean x this%). "
                "Mean = Fund Size x Investable% / 5. "
                "At 0% all years are identical (flat pacing). "
                "At 20% roughly two-thirds of years land within +/-20% of the mean. "
                "At 40% pacing is highly uneven."
            ),
        ) / 100.0

        # Derive and show the implied mean for context
        # Use the mode/value of fund_size and inv_pct for the illustration
        _fs_ref = fs_params.get("mode", fs_params.get("value", 300.0))
        _ip_ref = ip_params.get("value", ip_params.get("mode", 0.85))
        _mean_ref = _fs_ref * _ip_ref / 5.0
        _lo_ref   = max(0, _mean_ref * (1 - inv_std_pct))
        _hi_ref   = _mean_ref * (1 + inv_std_pct)
        st.caption(
            f"At current fund size / investable% settings: "
            f"mean ~ EUR {_mean_ref:.0f}M/yr,  "
            f"+/-1 std dev range ~ EUR {_lo_ref:.0f}M to EUR {_hi_ref:.0f}M"
        )

        # Mini preview: show +/-1sd band on a normal curve
        _draws = np.maximum(0, np.random.normal(_mean_ref, _mean_ref * inv_std_pct, 5000)) \
                 if inv_std_pct > 0 else np.full(5000, _mean_ref)
        _fig = go.Figure()
        _fig.add_trace(go.Histogram(
            x=_draws, nbinsx=50,
            marker_color="#f0883e", opacity=0.8,
            histnorm="probability density",
        ))
        _fig.add_vline(x=_mean_ref, line_dash="dash", line_color="#3fb950",
                       annotation_text=f"Mean EUR{_mean_ref:.0f}M",
                       annotation_font_size=9, annotation_position="top right")
        _fig.add_vline(x=_lo_ref, line_dash="dot", line_color="#8c959f",
                       annotation_text=f"-1sd EUR{_lo_ref:.0f}M",
                       annotation_font_size=9, annotation_position="top left")
        _fig.add_vline(x=_hi_ref, line_dash="dot", line_color="#8c959f",
                       annotation_text=f"+1sd EUR{_hi_ref:.0f}M",
                       annotation_font_size=9, annotation_position="top right")
        _fig.update_layout(
            height=150,
            margin=dict(t=10, b=20, l=30, r=10),
            paper_bgcolor="white", plot_bgcolor="#f6f8fa",
            font=dict(family="IBM Plex Sans", size=10),
            xaxis=dict(title="Annual deployment (EUR M)", title_font_size=10),
            yaxis=dict(title="", showticklabels=False),
            showlegend=False,
        )
        st.plotly_chart(_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div class="prev-label">Normal(mean, mean x {inv_std_pct*100:.0f}%), clipped at 0</div>',
            unsafe_allow_html=True
        )


# ââ RIGHT COLUMN: Stochastic return/market assumptions ââ
with col_B:

    # 6. Divestment Multiplier
    with st.container(border=True):
        mu_dist, mu_params = assumption_widget(
            key="mult", title="Divestment Multiplier (MoM per Deal)", icon="ð",
            default_dist="Beta-PERT",
            defaults_by_dist={
                "Fixed":      {"value": 2.1},
                "Triangular": {"min": 0.5, "mode": 2.0, "max": 5.5},
                "Beta-PERT":  {"min": 0.5, "mode": 2.0, "max": 5.5},
                "Uniform":    {"min": 0.8, "max": 4.5},
                "Normal":     {"mean": 2.1, "std": 0.9},
                "Log-Normal": {"mean": 2.1, "std": 0.9},
                "Beta":       {"alpha": 2.0, "beta": 3.0, "min": 0.2, "max": 7.0},
            },
            fmt="%.3f", preview_xlabel="MoM (x)",
        )

    # 7. Investment Duration
    with st.container(border=True):
        du_dist, du_params = assumption_widget(
            key="dur", title="Investment Duration (years per deal)", icon="[dur]",
            default_dist="Uniform",
            defaults_by_dist={
                "Fixed":      {"value": 3.0},
                "Triangular": {"min": 1.0, "mode": 3.0, "max": 7.0},
                "Beta-PERT":  {"min": 1.0, "mode": 3.0, "max": 7.0},
                "Uniform":    {"min": 1.0, "max": 5.0},
                "Normal":     {"mean": 3.5, "std": 1.2},
                "Log-Normal": {"mean": 3.5, "std": 1.2},
                "Beta":       {"alpha": 2.0, "beta": 2.0, "min": 1.0, "max": 9.0},
            },
            fmt="%.2f", preview_xlabel="Years",
        )

    # 8. Market Return (PME benchmark)
    with st.container(border=True):
        mk_dist, mk_params = assumption_widget(
            key="mkt", title="Market Return CAGR (PME benchmark)", icon="ð",
            default_dist="Normal",
            defaults_by_dist={
                "Fixed":      {"value": 0.075},
                "Triangular": {"min": 0.03, "mode": 0.075, "max": 0.15},
                "Beta-PERT":  {"min": 0.02, "mode": 0.075, "max": 0.15},
                "Uniform":    {"min": 0.03, "max": 0.13},
                "Normal":     {"mean": 0.075, "std": 0.025},
                "Log-Normal": {"mean": 0.075, "std": 0.025},
                "Beta":       {"alpha": 3.0, "beta": 5.0, "min": 0.0, "max": 0.20},
            },
            fmt="%.4f", preview_xlabel="Annual return", is_pct=True,
        )


# âââââââââââââââââââââââââââââââââââââââââââââ
# RUN BUTTON
# âââââââââââââââââââââââââââââââââââââââââââââ

st.markdown("---")
run_col, info_col = st.columns([1, 5])
run_btn = run_col.button("Run Simulation", type="primary", use_container_width=True)
info_col.markdown(
    f"<span style='font-size:13px;color:#656d76;'>"
    f"Will run <b>{n_sims:,}</b> scenarios - every assumption drawn independently per path</span>",
    unsafe_allow_html=True,
)

# âââââââââââââââââââââââââââââââââââââââââââââ
# SIMULATION + RESULTS
# âââââââââââââââââââââââââââââââââââââââââââââ

if run_btn or "mc_results" in st.session_state:

    if run_btn:
        with st.spinner(f"Sampling distributions and running {n_sims:,} scenarios..."):
            np.random.seed(int(seed))

            # Pre-draw all assumption samples
            # scalar assumptions: n_sims draws each
            # mult & dur: n_sims x 5 draws (one per cohort per sim)
            draws = {
                "fund_size": sample_dist(fs_dist, fs_params, n_sims),
                "mgmt_fee":  sample_dist(mf_dist, mf_params, n_sims),
                "carry":     sample_dist(cr_dist, cr_params, n_sims),
                "hurdle":    sample_dist(hu_dist, hu_params, n_sims),
                "inv_pct":   sample_dist(ip_dist, ip_params, n_sims),
                "mult":      sample_dist(mu_dist, mu_params, n_sims * 5),
                "dur":       sample_dist(du_dist, du_params, n_sims * 5),
                "mkt":       sample_dist(mk_dist, mk_params, n_sims),
            }

            df = run_mc(n_sims, draws, float(inv_std_pct), int(seed))
        st.session_state["mc_results"] = df
        st.session_state["mc_meta"] = {
            "fs": (fs_dist, fs_params), "mf": (mf_dist, mf_params),
            "cr": (cr_dist, cr_params), "hu": (hu_dist, hu_params),
            "ip": (ip_dist, ip_params), "mu": (mu_dist, mu_params),
            "du": (du_dist, du_params), "mk": (mk_dist, mk_params),
        }
    else:
        df = st.session_state["mc_results"]

    # ââ KPI row ââ
    st.markdown('<div class="section-hdr">Simulation Results</div>', unsafe_allow_html=True)

    def s(col): return df[col].dropna().replace([np.inf, -np.inf], np.nan).dropna()

    mom        = s("lp_mom")
    irr_zero   = s("lp_irr_zero") * 100
    irr_cal    = s("lp_irr_cal") * 100
    mirr_zero  = s("lp_mirr_zero") * 100
    mirr_cal   = s("lp_mirr_cal") * 100
    mirr_mom_s = s("mirr_mom")
    fund_mom_s = s("fund_mom")
    fund_dpi_s = s("fund_dpi")
    fl_vals    = s("fund_life")
    beats_pct  = df["beats_market"].mean() * 100

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    def kpi(col, label, value, sub=""):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True
        )

    kpi(k1, "LP MoM (C62)",      f"{mom.median():.2f}x",
        f"P10: {mom.quantile(.1):.2f}  P90: {mom.quantile(.9):.2f}")
    kpi(k2, "LP IRR Zero (C63)", f"{irr_zero.median():.1f}%",
        f"P10: {irr_zero.quantile(.1):.1f}%  P90: {irr_zero.quantile(.9):.1f}%")
    kpi(k3, "LP IRR Cal (C64)",  f"{irr_cal.median():.1f}%",
        f"P10: {irr_cal.quantile(.1):.1f}%  P90: {irr_cal.quantile(.9):.1f}%")
    kpi(k4, "MIRR Zero (C65)",   f"{mirr_zero.median():.1f}%",
        f"P10: {mirr_zero.quantile(.1):.1f}%  P90: {mirr_zero.quantile(.9):.1f}%")
    kpi(k5, "MoM @ MIRR (C67)",  f"{mirr_mom_s.median():.2f}x",
        f"P10: {mirr_mom_s.quantile(.1):.2f}  P90: {mirr_mom_s.quantile(.9):.2f}")
    kpi(k6, "% Beats Market",    f"{beats_pct:.0f}%", "LP MoM > PME MoM")

    # ââ charts ââ
    def hist_fig(series, color, title, xlabel, vlines=None, nbins=80):
        cl = series.replace([np.inf,-np.inf], np.nan).dropna()
        lo, hi = np.percentile(cl, 0.5), np.percentile(cl, 99.5)
        cl = cl[(cl >= lo) & (cl <= hi)]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=cl, nbinsx=nbins,
                                   marker_color=color, opacity=0.85,
                                   histnorm="probability density"))
        if vlines:
            for x_val, dash, ann, pos in vlines:
                fig.add_vline(x=x_val, line_dash=dash, line_color="#f78166",
                              annotation_text=ann, annotation_position=pos,
                              annotation_font_size=10)
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Density",
                          height=300, margin=dict(t=45, b=35, l=35, r=10),
                          paper_bgcolor="white", plot_bgcolor="#f6f8fa",
                          font=dict(family="IBM Plex Sans"), showlegend=False)
        return fig

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(hist_fig(mom, "#58a6ff", "LP MoM (C62)", "MoM (x)",
            [(mom.median(), "dash", f"Median {mom.median():.2f}x", "top right"),
             (1.0, "dot", "1.0x breakeven", "top left")]),
            use_container_width=True)
    with c2:
        st.plotly_chart(hist_fig(irr_zero, "#3fb950", "LP IRR Zero (C63)", "IRR (%)",
            [(irr_zero.median(), "dash", f"Median {irr_zero.median():.1f}%", "top right")]),
            use_container_width=True)
    with c3:
        st.plotly_chart(hist_fig(mirr_mom_s, "#8957e5", "MoM @ MIRR (C67)", "MoM (x)",
            [(mirr_mom_s.median(), "dash", f"Median {mirr_mom_s.median():.2f}x", "top right"),
             (1.0, "dot", "1.0x", "top left")]),
            use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        sub = df.sample(min(1500, len(df)))
        mv  = max(float(sub["mkt_mom"].max()), float(sub["lp_mom"].max())) + 0.3
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=sub["mkt_mom"], y=sub["lp_mom"], mode="markers",
            marker=dict(color=sub["beats_market"].map({1:"#3fb950",0:"#f78166"}),
                        size=4, opacity=0.5)))
        fig_sc.add_trace(go.Scatter(x=[0, mv], y=[0, mv], mode="lines",
            line=dict(color="#8c959f", dash="dash", width=1)))
        fig_sc.update_layout(title="LP MoM vs PME MoM  (green = PE wins)",
            xaxis_title="Market MoM", yaxis_title="LP MoM",
            height=300, margin=dict(t=45, b=35, l=35, r=10),
            paper_bgcolor="white", plot_bgcolor="#f6f8fa",
            font=dict(family="IBM Plex Sans"), showlegend=False)
        st.plotly_chart(fig_sc, use_container_width=True)

    with c5:
        s_mom = np.sort(mom.values)
        cdf   = np.arange(1, len(s_mom)+1) / len(s_mom)
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(x=s_mom, y=cdf*100, mode="lines",
                                     line=dict(color="#58a6ff", width=2.5)))
        for pv, pc, pl in [(.10,"#f78166","P10"),(.50,"#3fb950","P50"),(.90,"#8957e5","P90")]:
            v = np.quantile(s_mom, pv)
            fig_cdf.add_vline(x=v, line_dash="dot", line_color=pc,
                              annotation_text=f"{pl}: {v:.2f}x",
                              annotation_font_size=9, annotation_position="top right")
        fig_cdf.update_layout(title="LP MoM - CDF",
            xaxis_title="LP MoM (x)", yaxis_title="Cumulative %",
            height=300, margin=dict(t=45, b=35, l=35, r=10),
            paper_bgcolor="white", plot_bgcolor="#f6f8fa",
            font=dict(family="IBM Plex Sans"), showlegend=False)
        st.plotly_chart(fig_cdf, use_container_width=True)

    # ââ Full percentile table ââ
    st.markdown('<div class="section-hdr">Percentile Statistics - All Forecasts</div>',
                unsafe_allow_html=True)

    pcts = [5, 10, 25, 50, 75, 90, 95]
    def prow(col, fmt="{:.3f}", scale=1.0):
        vals = df[col].replace([np.inf,-np.inf], np.nan).dropna() * scale
        return [fmt.format(np.percentile(vals, p)) for p in pcts]

    tbl = {
        "Pct":                  [f"P{p}" for p in pcts],
        "Fund MoM (C59)":       prow("fund_mom"),
        "Fund IRR Zero% (C60)": prow("fund_irr_zero", "{:.1f}", 100),
        "Fund IRR Cal% (C61)":  prow("fund_irr_cal",  "{:.1f}", 100),
        "LP MoM (C62)":         prow("lp_mom"),
        "LP IRR Zero% (C63)":   prow("lp_irr_zero",   "{:.1f}", 100),
        "LP IRR Cal% (C64)":    prow("lp_irr_cal",    "{:.1f}", 100),
        "MIRR Zero% (C65)":     prow("lp_mirr_zero",  "{:.1f}", 100),
        "MIRR Cal% (C66)":      prow("lp_mirr_cal",   "{:.1f}", 100),
        "MoM@MIRR (C67)":       prow("mirr_mom"),
        "Fund DPI (C69)":       prow("fund_dpi"),
        "PME MoM (C74)":        prow("mkt_mom"),
        "Fund Life (C51)":      prow("fund_life", "{:.0f}"),
    }
    tbl_df = pd.DataFrame(tbl)

    rows_html = ""
    for _, row in tbl_df.iterrows():
        cls = ' class="hl"' if row["Pct"] == "P50" else ""
        rows_html += f"<tr{cls}>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"

    st.markdown(
        f'<table class="pct-table"><thead><tr>'
        + "".join(f"<th>{c}</th>" for c in tbl_df.columns)
        + f"</tr></thead><tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True
    )

    # ââ Summary stats ââ
    st.markdown('<div class="section-hdr">Summary Statistics</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    def stat_card(col, title, kv):
        rows = "".join(
            f"<tr><td style='color:#656d76;font-size:12px;padding:4px 8px'>{k}</td>"
            f"<td style='font-family:IBM Plex Mono;font-size:13px;font-weight:600;"
            f"padding:4px 8px'>{v}</td></tr>"
            for k, v in kv.items()
        )
        col.markdown(
            f"<div style='background:white;border:1px solid #d0d7de;border-radius:8px;"
            f"padding:16px'><div style='font-weight:700;font-size:11px;text-transform:uppercase;"
            f"letter-spacing:.1em;color:#0d1117;margin-bottom:10px'>{title}</div>"
            f"<table style='width:100%'>{rows}</table></div>",
            unsafe_allow_html=True
        )

    stat_card(s1, "LP Returns", {
        "Mean MoM (C62)":       f"{mom.mean():.3f}x",
        "Median MoM":           f"{mom.median():.3f}x",
        "Std Dev MoM":          f"{mom.std():.3f}x",
        "Sharpe-like (C80)":    f"{mom.mean()/mom.std():.2f}",
        "Mean IRR Zero (C63)":  f"{irr_zero.mean():.1f}%",
        "Mean IRR Cal (C64)":   f"{irr_cal.mean():.1f}%",
        "Mean MIRR Zero (C65)": f"{mirr_zero.mean():.1f}%",
        "Mean MIRR Cal (C66)":  f"{mirr_cal.mean():.1f}%",
        "P(Loss < 1x)":         f"{(mom < 1).mean()*100:.1f}%",
    })
    stat_card(s2, "Market Comparison", {
        "% Beats Mkt MoM (C77)":   f"{beats_pct:.1f}%",
        "% Beats Mkt MIRR (C77d)": f"{df['beats_market_mirr'].mean()*100:.1f}%",
        "Median Excess MoM (C78)": f"{s('excess_mom').median():.3f}x",
        "Median Excess MIRR":      f"{s('excess_mom_mirr').median():.3f}x",
        "Median PME MoM (C74)":    f"{s('mkt_mom').median():.3f}x",
        "Mean Fund Life (C51)":    f"{fl_vals.mean():.1f} yrs",
    })
    stat_card(s3, "Fund & Config", {
        "Mean Fund MoM (C59)":  f"{fund_mom_s.mean():.3f}x",
        "Mean Fund DPI (C69)":  f"{fund_dpi_s.mean():.3f}x",
        "N simulations":        f"{n_sims:,}",
        "Fund size dist":       fs_dist,
        "Multiplier dist":      mu_dist,
        "Duration dist":        du_dist,
    })

    st.markdown("---")
    st.markdown(
        "<p style='font-size:11px;color:#8c959f'>"
        "All metrics (C59-C81) replicate the Excel model exactly. "
        "IRR Zero = from fund inception (C46:N46). "
        "IRR Calendar = from first call (D48:N48). "
        "MIRR uses 4% finance rate and realised market return as reinvestment rate.</p>",
        unsafe_allow_html=True
    )
