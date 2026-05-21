"""
PE Fund Monte Carlo Simulator – Streamlit App
All assumptions (structural + stochastic) support full distribution choice.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PE Fund Monte Carlo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background: #f6f8fa; }

/* ── assumption card ── */
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

/* ── KPI cards ── */
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

/* ── section heading ── */
.section-hdr {
  font-size: 12px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .12em; color: #0d1117;
  border-bottom: 2px solid #0d1117; padding-bottom: 5px;
  margin: 28px 0 16px;
}

/* ── percentile table ── */
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


# ─────────────────────────────────────────────
# DISTRIBUTION ENGINE
# ─────────────────────────────────────────────

DIST_TYPES = ["Fixed", "Triangular", "Beta-PERT", "Uniform", "Normal", "Log-Normal", "Beta"]

DIST_PARAMS = {
    "Fixed":      [("value", "Value")],
    "Triangular": [("min", "Min"), ("mode", "Most Likely"), ("max", "Max")],
    "Beta-PERT":  [("min", "Min"), ("mode", "Most Likely"), ("max", "Max")],
    "Uniform":    [("min", "Min"), ("max", "Max")],
    "Normal":     [("mean", "Mean"), ("std", "Std Dev")],
    "Log-Normal": [("mean", "Mean (arith.)"), ("std", "Std Dev (arith.)")],
    "Beta":       [("alpha", "Alpha (α)"), ("beta", "Beta (β)"),
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
        return f"{dist}(μ={params['mean']:.3g}, σ={params['std']:.3g})"
    elif dist == "Beta":
        return f"Beta(α={params['alpha']:.3g}, β={params['beta']:.3g}) [{params['min']:.3g}, {params['max']:.3g}]"
    return dist


# ─────────────────────────────────────────────
# ASSUMPTION WIDGET
# builds the dist selector + param inputs + inline mini-preview
# returns (dist_type, params_dict)
# ─────────────────────────────────────────────

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

    # ── distribution type selector ──
    dist_idx = DIST_TYPES.index(default_dist) if default_dist in DIST_TYPES else 0
    dist = st.selectbox(
        "Distribution", DIST_TYPES,
        index=dist_idx, key=f"{key}_dist",
        label_visibility="collapsed",
    )

    # ── parameter inputs (dynamic, based on dist) ──
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

    # ── inline mini-preview chart ──
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


# ─────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────

def _irr(cashflows, guess=0.10, tol=1e-6, maxiter=500):
    cf = np.asarray(cashflows, dtype=float)
    r = guess
    for _ in range(maxiter):
        t  = np.arange(len(cf))
        f  = np.sum(cf / (1 + r) ** t)
        df = -np.sum(t * cf / (1 + r) ** (t + 1))
        if abs(df) < 1e-12:
            break
        r2 = r - f / df
        if abs(r2 - r) < tol:
            return r2
        r = r2
    return r

def _fund_life(lp_cf):
    for yr in range(len(lp_cf) - 1, 0, -1):
        if lp_cf[yr] > 0:
            return yr
    return len(lp_cf) - 1

def simulate_one(fund_size, mgmt_fee, carry, hurdle, inv_pct,
                 dur_draws, mult_draws, market_ret, inv_std_pct):
    """One simulation path → scalar metrics."""
    N = 12          # years 0..11
    INV_YRS = 5

    # ── annual investments (stochastic deal pacing) ──
    mean_inv = fund_size * inv_pct / INV_YRS
    investments = np.maximum(0, np.random.normal(mean_inv, mean_inv * inv_std_pct, INV_YRS))
    investments[-1] = max(0, fund_size * inv_pct - investments[:-1].sum())
    investments = np.append(investments, np.zeros(N - INV_YRS - 1))   # years 1-11

    # ── management fees ──
    mgmt_fees = np.zeros(N)
    for yr in range(INV_YRS):
        mgmt_fees[yr] = mgmt_fee * fund_size

    # ── exit cash flows ──
    durations  = np.clip(np.round(dur_draws).astype(int), 1, 9)
    exit_years = np.arange(1, INV_YRS + 1) + durations
    cap_ret    = np.zeros(N)
    gain_ret   = np.zeros(N)
    for i in range(INV_YRS):
        ey  = min(int(exit_years[i]), N - 1)
        cap = investments[i]
        m   = max(0.01, mult_draws[i])
        cap_ret[ey]  += cap
        gain_ret[ey] += cap * (m - 1)

    total_divest = cap_ret + gain_ret

    # ── European waterfall (hurdle → catch-up → carry) ──
    lp_cf  = np.zeros(N)
    gp_cf  = np.zeros(N)
    lp_cf[0] = -fund_size

    hurdle_balance = fund_size
    cumul_non_carry = 0.0
    hurdle_cumul = fund_size
    residual_hurdle = fund_size

    for yr in range(1, N):
        divest  = total_divest[yr]
        hurdle_cumul   *= (1 + hurdle)
        residual_hurdle = max(0, residual_hurdle * (1 + hurdle) - divest)

        if residual_hurdle > 0:
            non_carry  = divest
            cu_gp = carry_gp = carry_lp = 0.0
        else:
            paid_hurdle  = min(divest, hurdle_cumul)
            residual     = max(0, divest - paid_hurdle)
            cu_needed    = (cumul_non_carry + paid_hurdle - fund_size) * carry / (1 - carry)
            cu_gp        = min(residual, max(0, cu_needed))
            residual2    = max(0, residual - cu_gp)
            carry_gp     = residual2 * carry
            carry_lp     = residual2 * (1 - carry)
            non_carry    = paid_hurdle
            cumul_non_carry += non_carry

        lp_cf[yr] += non_carry + carry_lp
        gp_cf[yr] += cu_gp + carry_gp

    # deduct contributions (calls) from LP
    for yr in range(1, INV_YRS + 1):
        lp_cf[yr] -= investments[yr - 1] + mgmt_fees[yr - 1]

    total_lp = max(0, lp_cf[1:].sum())
    total_gp = max(0, gp_cf.sum())

    try:
        irr_val = _irr(lp_cf)
    except Exception:
        irr_val = float("nan")

    fl  = _fund_life(lp_cf)
    mom = total_lp / fund_size if fund_size > 0 else float("nan")
    mkt = (1 + market_ret) ** fl

    return dict(lp_mom=mom, lp_irr=irr_val, total_lp=total_lp,
                total_gp=total_gp, fund_life=fl, mkt_mom=mkt,
                beats_market=int(mom > mkt))


def run_mc(n_sims, assumption_draws, inv_std_pct, seed):
    """
    assumption_draws: dict keyed by assumption name → sample array of length n_sims.
    """
    np.random.seed(seed)
    rows = []

    for i in range(n_sims):
        fs  = float(assumption_draws["fund_size"][i])
        mf  = float(assumption_draws["mgmt_fee"][i])
        cr  = float(assumption_draws["carry"][i])
        hu  = float(assumption_draws["hurdle"][i])
        ip  = float(np.clip(assumption_draws["inv_pct"][i], 0.01, 0.99))

        # multiplier & duration: draw 5 per sim
        mult_arr = np.array([
            float(assumption_draws[f"mult"][i * 5 + k])
            for k in range(5)
        ])
        dur_arr = np.array([
            float(assumption_draws[f"dur"][i * 5 + k])
            for k in range(5)
        ])
        mkt = float(assumption_draws["mkt"][i])

        rows.append(simulate_one(
            fund_size=max(fs, 1.0),
            mgmt_fee=np.clip(mf, 0, 0.1),
            carry=np.clip(cr, 0, 0.5),
            hurdle=np.clip(hu, 0, 0.3),
            inv_pct=ip,
            dur_draws=dur_arr,
            mult_draws=mult_arr,
            market_ret=mkt,
            inv_std_pct=inv_std_pct,
        ))

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div style="background:#0d1117;padding:22px 32px;border-radius:10px;margin-bottom:24px;">
  <h1 style="color:#58a6ff;font-family:'IBM Plex Mono',monospace;margin:0;font-size:21px;letter-spacing:-.5px;">
    PE Fund Monte Carlo Simulator
  </h1>
  <p style="color:#8b949e;margin:6px 0 0;font-size:13px;">
    European-style waterfall · Catch-up · Every assumption configurable as Fixed or any stochastic distribution
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIMULATION SETTINGS (top bar)
# ─────────────────────────────────────────────

cfg1, cfg2, cfg3 = st.columns([1, 1, 4])
n_sims = cfg1.select_slider("Simulations", options=[500, 1000, 2500, 5000, 10000], value=2500)
seed   = cfg2.number_input("Seed", value=42, min_value=0, label_visibility="visible")
inv_std_pct = cfg3.slider(
    "Deal size std-dev (% of mean annual deployment)",
    min_value=0.0, max_value=0.80, value=0.20, step=0.05,
    format="%.0f%%",
    help="Controls how variable each year's investment pacing is around the average."
)

st.markdown("---")

# ─────────────────────────────────────────────
# ASSUMPTION CONFIGURATORS — two columns of cards
# ─────────────────────────────────────────────

st.markdown('<div class="section-hdr">Monte Carlo Assumptions</div>', unsafe_allow_html=True)
st.caption("Each assumption can be set as a fixed number or sampled from a distribution every simulation. "
           "The mini-chart updates live as you change parameters.")

col_A, col_B = st.columns(2, gap="large")

# ── LEFT COLUMN: Fund structure assumptions ──
with col_A:

    # 1. Fund Size
    with st.container(border=True):
        fs_dist, fs_params = assumption_widget(
            key="fund_size", title="Fund Size (€M)", icon="💰",
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
            fmt="%.1f", preview_xlabel="€M",
        )

    # 2. Management Fee
    with st.container(border=True):
        mf_dist, mf_params = assumption_widget(
            key="mgmt_fee", title="Management Fee", icon="📋",
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
            key="carry", title="Carried Interest (GP Share)", icon="💼",
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
            key="hurdle", title="Hurdle Rate (Preferred Return)", icon="🎯",
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
            key="inv_pct", title="Investable Capital (% of Fund)", icon="📐",
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


# ── RIGHT COLUMN: Stochastic return/market assumptions ──
with col_B:

    # 6. Divestment Multiplier
    with st.container(border=True):
        mu_dist, mu_params = assumption_widget(
            key="mult", title="Divestment Multiplier (MoM per Deal)", icon="📈",
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
            key="dur", title="Investment Duration (years per deal)", icon="⏱️",
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
            key="mkt", title="Market Return CAGR (PME benchmark)", icon="🌐",
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


# ─────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────

st.markdown("---")
run_col, info_col = st.columns([1, 5])
run_btn = run_col.button("▶  Run Simulation", type="primary", use_container_width=True)
info_col.markdown(
    f"<span style='font-size:13px;color:#656d76;'>"
    f"Will run <b>{n_sims:,}</b> scenarios · every assumption drawn independently per path</span>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SIMULATION + RESULTS
# ─────────────────────────────────────────────

if run_btn or "mc_results" in st.session_state:

    if run_btn:
        with st.spinner(f"Sampling distributions and running {n_sims:,} scenarios…"):
            np.random.seed(int(seed))

            # Pre-draw all assumption samples
            # scalar assumptions: n_sims draws each
            # mult & dur: n_sims × 5 draws (one per cohort per sim)
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

    # ── KPI row ──
    st.markdown('<div class="section-hdr">Simulation Results</div>', unsafe_allow_html=True)

    mom = df["lp_mom"].dropna()
    irr = df["lp_irr"].dropna() * 100
    beats = df["beats_market"].mean() * 100
    fl_vals = df["fund_life"].dropna()

    k1, k2, k3, k4, k5 = st.columns(5)

    def kpi(col, label, value, sub=""):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True
        )

    kpi(k1, "LP MoM — Median", f"{mom.median():.2f}×",
        f"P10: {mom.quantile(.1):.2f}×  P90: {mom.quantile(.9):.2f}×")
    kpi(k2, "LP IRR — Median", f"{irr.median():.1f}%",
        f"P10: {irr.quantile(.1):.1f}%  P90: {irr.quantile(.9):.1f}%")
    kpi(k3, "% Beats Market", f"{beats:.0f}%", "PE MoM > PME MoM")
    kpi(k4, "Fund Life — Median", f"{fl_vals.median():.0f} yrs",
        f"P10: {fl_vals.quantile(.1):.0f}  P90: {fl_vals.quantile(.9):.0f}")
    kpi(k5, "Sharpe-like (LP MoM)", f"{mom.mean()/mom.std():.2f}",
        f"mean/σ = {mom.mean():.2f}/{mom.std():.2f}")

    # ── charts ──
    def hist_fig(series, color, title, xlabel, vlines=None, nbins=80):
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=series, nbinsx=nbins,
            marker_color=color, opacity=0.85,
            histnorm="probability density",
        ))
        if vlines:
            for x_val, dash, ann, pos in vlines:
                fig.add_vline(x=x_val, line_dash=dash, line_color="#f78166",
                              annotation_text=ann, annotation_position=pos,
                              annotation_font_size=11)
        fig.update_layout(
            title=title, xaxis_title=xlabel, yaxis_title="Density",
            height=320, margin=dict(t=50, b=40, l=40, r=10),
            paper_bgcolor="white", plot_bgcolor="#f6f8fa",
            font=dict(family="IBM Plex Sans"), showlegend=False,
        )
        return fig

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            hist_fig(mom, "#58a6ff", "LP MoM Distribution", "MoM (×)",
                     vlines=[(mom.median(), "dash", f"Median {mom.median():.2f}×", "top right"),
                              (1.0, "dot", "1.0× breakeven", "top left")]),
            use_container_width=True
        )
    with c2:
        irr_cl = irr[(irr > -50) & (irr < 200)]
        st.plotly_chart(
            hist_fig(irr_cl, "#3fb950", "LP IRR Distribution", "IRR (%)",
                     vlines=[(irr_cl.median(), "dash", f"Median {irr_cl.median():.1f}%", "top right")]),
            use_container_width=True
        )

    c3, c4 = st.columns(2)
    with c3:
        sub = df.sample(min(1500, len(df)))
        max_v = max(sub["mkt_mom"].max(), sub["lp_mom"].max()) + 0.3
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=sub["mkt_mom"], y=sub["lp_mom"], mode="markers",
            marker=dict(color=sub["beats_market"].map({1: "#3fb950", 0: "#f78166"}),
                        size=4, opacity=0.55),
        ))
        fig3.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines",
                                  line=dict(color="#8c959f", dash="dash", width=1)))
        fig3.update_layout(
            title="LP MoM vs Market MoM  (green = PE wins)",
            xaxis_title="Market MoM", yaxis_title="LP MoM",
            height=320, margin=dict(t=50, b=40, l=40, r=10),
            paper_bgcolor="white", plot_bgcolor="#f6f8fa",
            font=dict(family="IBM Plex Sans"), showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        s_mom = np.sort(mom.values)
        cdf   = np.arange(1, len(s_mom) + 1) / len(s_mom)
        fig4  = go.Figure()
        fig4.add_trace(go.Scatter(x=s_mom, y=cdf * 100, mode="lines",
                                  line=dict(color="#58a6ff", width=2.5)))
        for pv, pc, pl in [(0.10, "#f78166", "P10"), (0.50, "#3fb950", "P50"), (0.90, "#8957e5", "P90")]:
            v = np.quantile(s_mom, pv)
            fig4.add_vline(x=v, line_dash="dot", line_color=pc,
                           annotation_text=f"{pl}: {v:.2f}×",
                           annotation_font_size=10, annotation_position="top right")
        fig4.update_layout(
            title="LP MoM — Cumulative Distribution",
            xaxis_title="LP MoM (×)", yaxis_title="Cumulative %",
            height=320, margin=dict(t=50, b=40, l=40, r=10),
            paper_bgcolor="white", plot_bgcolor="#f6f8fa",
            font=dict(family="IBM Plex Sans"), showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── percentile table ──
    st.markdown('<div class="section-hdr">Percentile Statistics</div>', unsafe_allow_html=True)

    pcts = [5, 10, 25, 50, 75, 90, 95]
    tbl = {
        "Pct":               [f"P{p}" for p in pcts],
        "LP MoM (×)":       [f"{np.percentile(mom, p):.3f}" for p in pcts],
        "LP IRR (%)":       [f"{np.percentile(irr.dropna(), p):.1f}" for p in pcts],
        "Fund Life (yrs)":  [f"{np.percentile(fl_vals, p):.0f}" for p in pcts],
        "Excess vs Mkt (×)":[f"{np.percentile(df['lp_mom']-df['mkt_mom'], p):.3f}" for p in pcts],
        "P(Loss)%":         ["" if p != 50 else f"{(mom < 1).mean()*100:.1f}" for p in pcts],
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

    # ── summary stats ──
    st.markdown('<div class="section-hdr">Summary Statistics</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    def stat_card(col, title, kv):
        rows = "".join(
            f"<tr><td style='color:#656d76;font-size:12px;padding:4px 8px'>{k}</td>"
            f"<td style='font-family:IBM Plex Mono;font-size:13px;font-weight:600;padding:4px 8px'>{v}</td></tr>"
            for k, v in kv.items()
        )
        col.markdown(
            f"<div style='background:white;border:1px solid #d0d7de;border-radius:8px;padding:16px'>"
            f"<div style='font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.1em;"
            f"color:#0d1117;margin-bottom:10px'>{title}</div>"
            f"<table style='width:100%'>{rows}</table></div>",
            unsafe_allow_html=True
        )

    stat_card(s1, "LP Returns", {
        "Mean MoM":    f"{mom.mean():.3f}×",
        "Median MoM":  f"{mom.median():.3f}×",
        "Std Dev MoM": f"{mom.std():.3f}×",
        "Mean IRR":    f"{irr.dropna().mean():.1f}%",
        "Median IRR":  f"{irr.dropna().median():.1f}%",
        "P(Loss)":     f"{(mom < 1).mean()*100:.1f}%",
    })
    stat_card(s2, "Market Comparison", {
        "% Beats Market":   f"{beats:.1f}%",
        "Avg Excess MoM":   f"{(df['lp_mom']-df['mkt_mom']).mean():.3f}×",
        "Median Excess":    f"{(df['lp_mom']-df['mkt_mom']).median():.3f}×",
        "Avg Market MoM":   f"{df['mkt_mom'].mean():.3f}×",
        "Sharpe-like LP":   f"{mom.mean()/mom.std():.2f}",
    })
    stat_card(s3, "Simulation Config", {
        "N Simulations":  f"{n_sims:,}",
        "Fund Size dist": fs_dist,
        "Carry dist":     cr_dist,
        "Multiplier dist":mu_dist,
        "Duration dist":  du_dist,
        "Avg Fund Life":  f"{fl_vals.mean():.1f} yrs",
    })

    st.markdown("---")
    st.markdown(
        "<p style='font-size:11px;color:#8c959f'>Model: European-style PE waterfall with hurdle accrual, "
        "catch-up, and 20% carry split. All assumptions independently sampled per scenario.</p>",
        unsafe_allow_html=True
    )
