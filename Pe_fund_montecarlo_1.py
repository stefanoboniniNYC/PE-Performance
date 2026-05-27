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
    page_icon="[MoM]",
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

DIST_TYPES = ["Fixed", "Triangular", "Beta-PERT", "Uniform", "Normal", "Log-Normal", "Beta",
              "Custom Discrete"]

# Custom Discrete uses up to 6 value/probability pairs (v1..v6, p1..p6).
# Unused slots: set prob to 0. Values are discrete outcomes; probs must sum to 1.
DIST_PARAMS = {
    "Fixed":           [("value", "Value")],
    "Triangular":      [("min", "Min"), ("mode", "Most Likely"), ("max", "Max")],
    "Beta-PERT":       [("min", "Min"), ("mode", "Most Likely"), ("max", "Max")],
    "Uniform":         [("min", "Min"), ("max", "Max")],
    "Normal":          [("mean", "Mean"), ("std", "Std Dev")],
    "Log-Normal":      [("mean", "Mean (arith.)"), ("std", "Std Dev (arith.)")],
    "Beta":            [("alpha", "Alpha (alpha)"), ("beta", "Beta (beta)"),
                        ("min", "Min (scale lo)"), ("max", "Max (scale hi)")],
    "Custom Discrete": [("v1","Val 1"),("p1","Prob 1"),
                        ("v2","Val 2"),("p2","Prob 2"),
                        ("v3","Val 3"),("p3","Prob 3"),
                        ("v4","Val 4"),("p4","Prob 4"),
                        ("v5","Val 5"),("p5","Prob 5"),
                        ("v6","Val 6"),("p6","Prob 6")],
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

        elif dist == "Custom Discrete":
            # Extract value/prob pairs; skip slots where prob<=0
            vals  = [params[f"v{i}"] for i in range(1, 7)]
            probs = [max(0.0, params[f"p{i}"]) for i in range(1, 7)]
            total = sum(probs)
            if total <= 0:
                return np.full(n, vals[0])
            probs = [p / total for p in probs]   # normalise to sum=1
            # Keep only non-zero probability slots
            active_vals  = [v for v, p in zip(vals, probs) if p > 0]
            active_probs = [p for p in probs if p > 0]
            return np.random.choice(active_vals, p=active_probs, size=n).astype(float)

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
    elif dist == "Custom Discrete":
        pairs = [(params[f"v{i}"], params[f"p{i}"]) for i in range(1,7) if params[f"p{i}"] > 0]
        return "Custom: " + ", ".join(f"{v:.4g}={p*100:.0f}%" for v, p in pairs)
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

    if dist == "Custom Discrete":
        # Render as two rows: val/prob pairs side by side
        st.caption("Enter each outcome value and its probability (probs are auto-normalised to sum to 1).")
        for row_i in range(2):          # row 0 = slots 1-3, row 1 = slots 4-6
            slot_cols = st.columns(6)   # 3 pairs = 6 columns per row
            for slot in range(3):
                idx = row_i * 3 + slot + 1   # 1..6
                pname_v = f"v{idx}"; pname_p = f"p{idx}"
                plabel_v = f"Val {idx}"; plabel_p = f"Prob {idx}"
                default_v = float(dp.get(pname_v, float(idx)))
                default_p = float(dp.get(pname_p, 0.0))
                params[pname_v] = slot_cols[slot*2].number_input(
                    plabel_v, value=default_v, key=f"{key}_{pname_v}",
                    format=fmt, step=1.0)
                params[pname_p] = slot_cols[slot*2+1].number_input(
                    plabel_p, value=default_p, key=f"{key}_{pname_p}",
                    format="%.2f", step=0.05, min_value=0.0, max_value=1.0)
        # Show normalised probs and mean
        raw_probs = [params[f"p{i}"] for i in range(1,7)]
        total_p = sum(raw_probs)
        if total_p > 0:
            norm_probs = [p/total_p for p in raw_probs]
            vals = [params[f"v{i}"] for i in range(1,7)]
            mean_val = sum(v*p for v,p in zip(vals,norm_probs))
            active = [(v,p) for v,p in zip(vals,norm_probs) if p > 0]
            st.caption(f"Mean = {mean_val:.2f} | "
                       + "  ".join(f"{v:.4g}: {p*100:.0f}%" for v,p in active))
    else:
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
            # single vertical line: set a tight window around the value
            v0 = float(draws[0])
            pad = max(abs(v0) * 0.3, 0.5)
            fig = go.Figure()
            fig.add_vline(x=v0, line_width=2.5, line_color="#58a6ff")
            fig.add_annotation(x=v0, y=0.5, text=f"{v0:.3g}",
                               showarrow=False, yref="paper",
                               font=dict(size=13, family="IBM Plex Mono", color="#0d1117"))
            x_range = [v0 - pad, v0 + pad]
        elif dist == "Custom Discrete":
            # Bar chart showing exact probabilities (not sampled histogram)
            pairs = [(params[f"v{i}"], params[f"p{i}"]) for i in range(1,7) if params[f"p{i}"] > 0]
            total_p = sum(p for _, p in pairs)
            bar_x = [v * (100 if is_pct else 1) for v, _ in pairs]
            bar_y = [p / total_p * 100 for _, p in pairs]  # as %
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=bar_x, y=bar_y,
                marker_color="#58a6ff", opacity=0.85,
                marker_line=dict(color="#1a5fa8", width=0.8),
                text=[f"{y:.0f}%" for y in bar_y],
                textposition="outside", textfont=dict(size=9),
            ))
            pad = max((max(bar_x)-min(bar_x))*0.1, 0.3) if len(bar_x) > 1 else 0.5
            x_range = [min(bar_x)-pad, max(bar_x)+pad]
        else:
            fig = go.Figure()
            # Clip to data range (use min/max for bounded dists, percentiles for unbounded)
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

            # x-axis: centre on the data spread with a small pad, never forced to start at 0
            spread = hi_x - lo_x
            pad    = spread * 0.08
            x_range = [lo_x - pad, hi_x + pad]

        fig.update_layout(
            height=150,
            margin=dict(t=10, b=20, l=30, r=10),
            paper_bgcolor="white", plot_bgcolor="#f6f8fa",
            font=dict(family="IBM Plex Sans", size=10),
            xaxis=dict(
                title=preview_xlabel + (" (%)" if is_pct else ""),
                title_font_size=10,
                range=x_range,
            ),
            yaxis=dict(title="", showticklabels=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="prev-label">{dist_label(dist, params)}</div>', unsafe_allow_html=True)
    except Exception:
        st.caption("(preview unavailable)")

    return dist, params


# âââââââââââââââââââââââââââââââââââââââââââââ
# SIMULATION ENGINE  (vectorised across all N sims simultaneously)
# âââââââââââââââââââââââââââââââââââââââââââââ
# All arrays have shape (S,) for scalars or (S, N) for time-series,
# where S = number of simulations and N = 12 (years 0..11).
# The year-loop runs only 11 times (not S times), giving ~50x speedup.

N_YRS   = 12   # calendar years 0..11
INV_YRS = 5    # investment period


def _irr_vec(cfs, guess=0.10, tol=1e-8, maxiter=200):
    """
    Vectorised Newton-Raphson IRR for a batch of cash-flow series.
    cfs : (S, T) array  -- each row is one simulation's cash-flow stream
    returns: (S,) array of IRR values (nan where no solution)
    """
    S, T = cfs.shape
    t_idx = np.arange(T, dtype=float)
    r = np.full(S, guess)
    valid = (np.any(cfs > 0, axis=1)) & (np.any(cfs < 0, axis=1))

    for _ in range(maxiter):
        disc  = (1.0 + r[:, None]) ** t_idx[None, :]        # (S, T)
        f     = np.sum(cfs / disc,         axis=1)           # (S,)
        df    = -np.sum(t_idx * cfs / (disc * (1.0 + r[:, None])), axis=1)
        safe  = np.abs(df) > 1e-14
        step  = np.where(safe, f / df, 0.0)
        r_new = r - step
        converged = np.abs(r_new - r) < tol
        r = r_new
        if np.all(converged | ~valid):
            break

    result = np.where(valid, r, np.nan)
    return result


def _mirr_vec(cfs, finance_rate, reinvest_rate):
    """
    Vectorised MIRR matching Excel's =MIRR().
    cfs : (S, T) array
    returns: (S,) array
    """
    S, T = cfs.shape
    t_idx = np.arange(T, dtype=float)
    pos = np.maximum(cfs, 0.0)
    neg = np.minimum(cfs, 0.0)
    # Future value of positive flows reinvested at reinvest_rate
    fv_factors = (1.0 + reinvest_rate) ** (T - 1 - t_idx)   # (T,)
    fv = np.sum(pos * fv_factors[None, :], axis=1)            # (S,)
    # Present value of negative flows discounted at finance_rate
    pv_factors = (1.0 + finance_rate) ** t_idx                # (T,)
    pv = np.sum(neg / pv_factors[None, :], axis=1)            # (S,)
    valid = (fv > 0) & (pv < 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(valid, (fv / np.abs(pv)) ** (1.0 / (T - 1)) - 1.0, np.nan)
    return result


def run_mc(n_sims, assumption_draws, inv_std_pct, seed):
    """
    Fully vectorised Monte Carlo engine.
    All S simulations computed simultaneously using NumPy broadcasting.
    The only loops are over N_YRS=12 calendar years (sequential dependencies)
    and T=11 Newton-Raphson iterations -- never over S simulations.
    """
    np.random.seed(seed)
    S = n_sims

    # ââ Unpack pre-drawn assumptions ââââââââââââââââââââââââââââââââââââââ
    fs  = np.maximum(assumption_draws["fund_size"],             1.0)     # (S,)
    mf  = np.clip(assumption_draws["mgmt_fee"],   0.0,  0.10)           # (S,)
    cr  = np.clip(assumption_draws["carry"],       0.0,  0.50)          # (S,)
    hu  = np.clip(assumption_draws["hurdle"],      0.0,  0.30)          # (S,)
    ip  = np.clip(assumption_draws["inv_pct"],     0.01, 0.99)          # (S,)
    mkt = assumption_draws["mkt"]                                        # (S,)
    # mult and dur: (S, 5) -- one per cohort per sim
    mult = np.maximum(assumption_draws["mult"].reshape(S, 5), 0.01)     # (S,5)
    dur  = np.clip(np.round(
              assumption_draws["dur"].reshape(S, 5)).astype(int), 1, 9) # (S,5)

    # ââ Row 17: Annual investments (S, 5) ââââââââââââââââââââââââââââââââ
    mean_inv = (fs * ip / INV_YRS)[:, None]                  # (S,1)
    std_inv  = mean_inv * inv_std_pct
    inv_raw  = np.maximum(0.0,
                   np.random.normal(mean_inv, std_inv,
                                    size=(S, INV_YRS - 1)))  # (S,4)
    inv_yr5  = np.maximum(0.0,
                   fs * ip - inv_raw.sum(axis=1))            # (S,)
    inv = np.concatenate([inv_raw, inv_yr5[:, None]], axis=1) # (S,5)

    # ââ Rows 28/29: Scatter exits onto calendar year axis ââââââââââââââââ
    # exit_yr[s, c] = cohort c invested in year (c+1), exits at year (c+1)+dur[s,c]
    cohort_inv_yr = np.arange(1, INV_YRS + 1)[None, :]       # (1,5)
    exit_yr = np.clip(cohort_inv_yr + dur, 1, N_YRS - 1)     # (S,5)

    # cap_ret[s, yr] and gain_ret[s, yr]: (S, N_YRS)
    cap_ret  = np.zeros((S, N_YRS))
    gain_ret = np.zeros((S, N_YRS))
    for c in range(INV_YRS):
        ey = exit_yr[:, c]                    # (S,) -- exit year for cohort c
        cap = inv[:, c]                        # (S,)
        gain = cap * (mult[:, c] - 1.0)       # (S,)
        # Scatter: add to the appropriate calendar year for each sim
        np.add.at(cap_ret,  (np.arange(S), ey), cap)
        np.add.at(gain_ret, (np.arange(S), ey), gain)

    total_divest = cap_ret + gain_ret          # (S, N_YRS)

    # ââ Row 18: Running portfolio balance ââââââââââââââââââââââââââââââââ
    row18 = np.zeros((S, N_YRS))
    row18[:, 1] = inv[:, 0]
    for yr in range(2, N_YRS):
        inv_yr_col = inv[:, yr - 1] if yr <= INV_YRS else 0.0
        row18[:, yr] = np.maximum(0.0, row18[:, yr-1] + inv_yr_col - cap_ret[:, yr])

    # ââ Row 20: Management fees ââââââââââââââââââââââââââââââââââââââââââ
    mgmt_fees = np.zeros((S, N_YRS))
    for yr in range(1, INV_YRS + 1):
        mgmt_fees[:, yr] = mf * fs
    for yr in range(INV_YRS + 1, N_YRS):
        mgmt_fees[:, yr] = np.where(row18[:, yr-1] > 0, row18[:, yr-1] * mf, 0.0)

    # ââ Row 30: Cumulative total divestments âââââââââââââââââââââââââââââ
    row30 = np.cumsum(total_divest, axis=1)    # (S, N_YRS)

    # ââ Rows 32/33: Hurdle capital & residual ââââââââââââââââââââââââââââ
    # Sequential recurrence over years -- unavoidable, but only 11 iterations
    row32 = np.zeros((S, N_YRS))
    row33 = np.zeros((S, N_YRS))
    row32[:, 1] = fs * (1.0 + hu)
    row33[:, 1] = row32[:, 1]
    for yr in range(2, N_YRS):
        row32[:, yr] = row33[:, yr-1] * (1.0 + hu)
        row33[:, yr] = np.maximum(0.0, row32[:, yr] - total_divest[:, yr])

    # ââ Row 34: Non-carry LP distributions ââââââââââââââââââââââââââââââ
    has_divest = total_divest > 0              # (S, N_YRS)
    row34 = np.where(has_divest,
                     np.minimum(total_divest, row32), 0.0)    # (S, N_YRS)

    # ââ Row 35: Cumulative non-carry âââââââââââââââââââââââââââââââââââââ
    row35 = np.cumsum(row34, axis=1)           # (S, N_YRS)

    # ââ Row 36: Residual for catch-up and carry ââââââââââââââââââââââââââ
    above_hurdle = row30 > row35               # (S, N_YRS)
    row36 = np.where(above_hurdle & has_divest,
                     total_divest - row34, 0.0)               # (S, N_YRS)

    # ââ Row 38: Catch-up computed ââââââââââââââââââââââââââââââââââââââââ
    cu_ratio = cr / (1.0 - cr)                 # (S,)
    row38 = np.where(
        (row36 > 0) & (row32 > 0),
        (row35 - fs[:, None]) * cu_ratio[:, None],
        0.0
    )                                           # (S, N_YRS)

    # ââ Rows 39/40: Catch-up actual (state machine, sequential) âââââââââ
    row39 = np.zeros((S, N_YRS))
    row40 = np.zeros((S, N_YRS))
    cs36  = np.zeros((S, N_YRS))
    cs38  = np.zeros((S, N_YRS))
    cs39  = np.zeros((S, N_YRS))
    for yr in range(1, N_YRS):
        cs36[:, yr] = cs36[:, yr-1] + row36[:, yr]
        cs38[:, yr] = cs38[:, yr-1] + row38[:, yr]
        has_res = row36[:, yr] > 0
        use_prev40 = has_res & (cs36[:, yr] > cs38[:, yr]) & (row40[:, yr-1] > 0)
        use_r38    = has_res & (cs36[:, yr] > cs38[:, yr]) & (row40[:, yr-1] <= 0)
        use_cs36   = has_res & (cs36[:, yr] <= cs38[:, yr])
        row39[:, yr] = (np.where(use_prev40, row40[:, yr-1], 0.0)
                      + np.where(use_r38,    row38[:, yr],   0.0)
                      + np.where(use_cs36,   cs36[:, yr],    0.0))
        cs39[:, yr] = cs39[:, yr-1] + row39[:, yr]
        row40[:, yr] = np.maximum(0.0, cs38[:, yr] - cs39[:, yr])

    # ââ Rows 41/42: Carry split ââââââââââââââââââââââââââââââââââââââââââ
    excess = np.maximum(0.0, row36 - row39)    # (S, N_YRS)
    row41  = excess * (1.0 - cr[:, None])      # LP carry  (S, N_YRS)
    row42  = excess * cr[:, None]              # GP carry  (S, N_YRS)

    # ââ Row 44: Contributions ââââââââââââââââââââââââââââââââââââââââââââ
    contributions = np.zeros((S, N_YRS))
    for yr in range(1, INV_YRS + 1):
        contributions[:, yr] = inv[:, yr-1] + mgmt_fees[:, yr]
    for yr in range(INV_YRS + 1, N_YRS):
        contributions[:, yr] = mgmt_fees[:, yr]

    # ââ Rows 45/46/47/48 âââââââââââââââââââââââââââââââââââââââââââââââââ
    row45 = row34 + row39 + row41 + row42                   # total fund dist
    lp_dist       = np.zeros((S, N_YRS))
    lp_dist[:, 0] = -fs
    lp_dist[:, 1:] = (row41 + row34)[:, 1:]                # LP distributions

    fund_net = np.zeros((S, N_YRS))
    fund_net[:, 1:] = row45[:, 1:] - contributions[:, 1:]

    net_lp       = np.zeros((S, N_YRS))
    net_lp[:, 0] = -fs
    net_lp[:, 1:] = lp_dist[:, 1:] - contributions[:, 1:]

    # ââ Fund life (C51): first yr cumulative cap_ret >= total invested âââ
    total_inv_per_sim = cap_ret.sum(axis=1)                  # (S,)
    cum_cap = np.cumsum(cap_ret[:, 1:], axis=1)              # (S,11)
    # For each sim: first col where cumsum >= total_inv (rounded to 2dp)
    reached = np.round(cum_cap, 2) >= np.round(total_inv_per_sim[:, None], 2)
    # argmax gives first True; if never True, falls back to last divestment year
    first_reached = np.where(reached.any(axis=1),
                              reached.argmax(axis=1) + 1,     # +1 because we sliced from yr1
                              np.maximum(1, (cap_ret > 0).cumsum(axis=1).argmax(axis=1)))
    fund_life = first_reached.astype(float)                  # (S,)

    # ââ IRR / MIRR (vectorised) ââââââââââââââââââââââââââââââââââââââââââ
    # C60: Fund IRR Zero = IRR([-fs, row45_yr1..yr11])
    fund_dist_cf = np.concatenate([-fs[:, None], row45[:, 1:]], axis=1)  # (S,12)
    fund_irr_zero = _irr_vec(fund_dist_cf)

    # C61: Fund IRR Calendar = IRR(fund_net yr1..yr11)
    fund_irr_cal = _irr_vec(fund_net[:, 1:])

    # C63: LP IRR Zero = IRR(lp_dist yr0..yr11)
    lp_irr_zero = _irr_vec(lp_dist)

    # C64: LP IRR Calendar = IRR(net_lp yr1..yr11)
    lp_irr_cal = _irr_vec(net_lp[:, 1:])

    # C65: MIRR zero = MIRR(lp_dist yr0..yr10, 4%, mkt)
    # mkt varies per sim -- compute per-sim using broadcasting
    # We need per-sim reinvest rate, so compute manually (vectorised)
    def mirr_vec_perrate(cfs, finance_rate, reinvest_rates):
        """MIRR with per-sim reinvest rate."""
        S2, T = cfs.shape
        t_idx2 = np.arange(T, dtype=float)
        pos = np.maximum(cfs, 0.0)
        neg = np.minimum(cfs, 0.0)
        # fv: each sim uses its own reinvest_rate
        fv = np.sum(pos * (1.0 + reinvest_rates[:, None]) ** (T - 1 - t_idx2[None, :]),
                    axis=1)
        pv_factors = (1.0 + finance_rate) ** t_idx2
        pv = np.sum(neg / pv_factors[None, :], axis=1)
        valid = (fv > 0) & (pv < 0)
        with np.errstate(invalid='ignore', divide='ignore'):
            res = np.where(valid, (fv / np.abs(pv)) ** (1.0 / (T - 1)) - 1.0, np.nan)
        return res

    lp_mirr_zero = mirr_vec_perrate(lp_dist[:, :11], 0.04, mkt)   # C65
    lp_mirr_cal  = mirr_vec_perrate(net_lp[:, 1:],   0.04, mkt)   # C66

    # ââ Scalar metrics ââââââââââââââââââââââââââââââââââââââââââââââââââââ
    total_lp_dist = lp_dist[:, 1:].sum(axis=1)                     # (S,)
    total_contrib = contributions[:, 1:].sum(axis=1)                # (S,)

    lp_mom       = np.where(fs > 0, total_lp_dist / fs, np.nan)    # C62
    fund_mom     = np.where(fs > 0, row30[:, -1] / fs, np.nan)     # C59
    fund_dpi     = np.where(total_contrib > 0,
                            total_lp_dist / total_contrib, np.nan) # C69
    mirr_mom     = (1.0 + lp_mirr_zero) ** fund_life               # C67
    mkt_mom      = (1.0 + mkt) ** fund_life                        # C74

    beats_market      = (lp_mom > mkt_mom).astype(float)
    beats_market_mirr = (mirr_mom > mkt_mom).astype(float)
    excess_mom        = lp_mom - mkt_mom
    excess_mom_mirr   = mirr_mom - mkt_mom

    return pd.DataFrame({
        "lp_mom":           lp_mom,
        "lp_irr_zero":      lp_irr_zero,
        "lp_irr_cal":       lp_irr_cal,
        "lp_mirr_zero":     lp_mirr_zero,
        "lp_mirr_cal":      lp_mirr_cal,
        "mirr_mom":         mirr_mom,
        "fund_dpi":         fund_dpi,
        "fund_mom":         fund_mom,
        "fund_irr_zero":    fund_irr_zero,
        "fund_irr_cal":     fund_irr_cal,
        "total_lp_dist":    total_lp_dist,
        "total_gp_dist":    (row39 + row42).sum(axis=1),
        "mkt_mom":          mkt_mom,
        "beats_market":     beats_market,
        "beats_market_mirr":beats_market_mirr,
        "excess_mom":       excess_mom,
        "excess_mom_mirr":  excess_mom_mirr,
        "fund_life":        fund_life,
    })


# âââââââââââââââââââââââââââââââââââââââââââââ
# PAGE HEADER
# âââââââââââââââââââââââââââââââââââââââââââââ

st.markdown("""
<div style="background:#0d1117;padding:22px 32px;border-radius:10px;margin-bottom:24px;">
  <h1 style="color:#58a6ff;font-family:'IBM Plex Mono',monospace;margin:0;font-size:21px;letter-spacing:-.5px;">
    PE Fund Monte Carlo Simulator
  </h1>
  <p style="color:#8b949e;margin:6px 0 0;font-size:13px;">
    European-style waterfall | Catch-up | Every assumption configurable as Fixed or stochastic
  </p>
</div>
""", unsafe_allow_html=True)

# âââââââââââââââââââââââââââââââââââââââââââââ
# SIMULATION SETTINGS (top bar)
# âââââââââââââââââââââââââââââââââââââââââââââ

cfg1, cfg2 = st.columns([1, 1])
n_sims = cfg1.select_slider("Simulations", options=[500, 1000, 2500, 5000, 10000, 25000, 50000, 100000], value=10000)
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
with col_A:
    st.markdown('<p style="font-size:18px;font-weight:700;color:#0d1117;margin:0 0 12px;">Fund Structure Assumptions</p>', unsafe_allow_html=True)
with col_B:
    st.markdown('<p style="font-size:18px;font-weight:700;color:#0d1117;margin:0 0 12px;">Investment & Market Assumptions</p>', unsafe_allow_html=True)

# ââ LEFT COLUMN: Fund structure assumptions ââ
with col_A:

    # 1. Fund Size
    with st.container(border=True):
        fs_dist, fs_params = assumption_widget(
            key="fund_size", title="Fund Size ($M)", icon="[Fund]",
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
            fmt="%.1f", preview_xlabel="$M",
        )

    # 2. Management Fee
    with st.container(border=True):
        mf_dist, mf_params = assumption_widget(
            key="mgmt_fee", title="Management Fee", icon="[Fee]",
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
            fmt="%.4f", preview_xlabel="Fee (e.g. 0.02 = 2%)", is_pct=True,
        )

    # 3. Carried Interest
    with st.container(border=True):
        cr_dist, cr_params = assumption_widget(
            key="carry", title="Carried Interest (GP Share)", icon="[Carry]",
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
            fmt="%.4f", preview_xlabel="Carry (e.g. 0.20 = 20%)", is_pct=True,
        )

    # 4. Hurdle Rate
    with st.container(border=True):
        hu_dist, hu_params = assumption_widget(
            key="hurdle", title="Hurdle Rate (Preferred Return)", icon="[Hurdle]",
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
            fmt="%.4f", preview_xlabel="Hurdle (e.g. 0.08 = 8%)", is_pct=True,
        )

    # 5. Investable Capital %
    with st.container(border=True):
        ip_dist, ip_params = assumption_widget(
            key="inv_pct", title="Investable Capital (% of Fund)", icon="[Inv%]",
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
            fmt="%.4f", preview_xlabel="Inv. capital (e.g. 0.85 = 85%)", is_pct=True,
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
            f"mean ~ ${_mean_ref:.0f}M/yr,  "
            f"+/-1 std dev range ~ ${_lo_ref:.0f}M to ${_hi_ref:.0f}M"
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
                       annotation_text=f"Mean ${_mean_ref:.0f}M",
                       annotation_font_size=9, annotation_position="top right")
        _fig.add_vline(x=_lo_ref, line_dash="dot", line_color="#8c959f",
                       annotation_text=f"-1sd ${_lo_ref:.0f}M",
                       annotation_font_size=9, annotation_position="top left")
        _fig.add_vline(x=_hi_ref, line_dash="dot", line_color="#8c959f",
                       annotation_text=f"+1sd ${_hi_ref:.0f}M",
                       annotation_font_size=9, annotation_position="top right")
        _fig.update_layout(
            height=150,
            margin=dict(t=10, b=20, l=30, r=10),
            paper_bgcolor="white", plot_bgcolor="#f6f8fa",
            font=dict(family="IBM Plex Sans", size=10),
            xaxis=dict(title="Annual deployment ($M)", title_font_size=10),
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
            key="mult", title="Divestment Multiplier (MoM per Deal)", icon="[MoM]",
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
            key="dur", title="Investment Duration (years per deal)", icon="[Dur]",
            default_dist="Custom Discrete",
            defaults_by_dist={
                "Fixed":           {"value": 3.0},
                "Triangular":      {"min": 1.0, "mode": 3.0, "max": 7.0},
                "Beta-PERT":       {"min": 1.0, "mode": 3.0, "max": 7.0},
                "Uniform":         {"min": 1.0, "max": 5.0},
                "Normal":          {"mean": 3.5, "std": 1.2},
                "Log-Normal":      {"mean": 3.5, "std": 1.2},
                "Beta":            {"alpha": 2.0, "beta": 2.0, "min": 1.0, "max": 9.0},
                # Baseline: 1yr=5%, 2yr=10%, 3yr=30%, 4yr=30%, 5yr=20%, 6yr=5%
                "Custom Discrete": {"v1":1.0,"p1":0.05, "v2":2.0,"p2":0.10,
                                    "v3":3.0,"p3":0.30, "v4":4.0,"p4":0.30,
                                    "v5":5.0,"p5":0.20, "v6":6.0,"p6":0.05},
            },
            fmt="%.2f", preview_xlabel="Years",
        )

    # 8. Market Return (PME benchmark)
    with st.container(border=True):
        mk_dist, mk_params = assumption_widget(
            key="mkt", title="Market Return CAGR (PME benchmark)", icon="[Mkt]",
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

# Build a fingerprint of all current parameters so we can detect stale cache
_current_fingerprint = str({
    "n_sims": n_sims, "seed": seed, "inv_std_pct": inv_std_pct,
    "fs": (fs_dist, sorted(fs_params.items())),
    "mf": (mf_dist, sorted(mf_params.items())),
    "cr": (cr_dist, sorted(cr_params.items())),
    "hu": (hu_dist, sorted(hu_params.items())),
    "ip": (ip_dist, sorted(ip_params.items())),
    "mu": (mu_dist, sorted(mu_params.items())),
    "du": (du_dist, sorted(du_params.items())),
    "mk": (mk_dist, sorted(mk_params.items())),
})
_cached_fingerprint = st.session_state.get("mc_fingerprint", None)
_results_stale = (_cached_fingerprint != _current_fingerprint)

if run_btn or ("mc_results" in st.session_state and not _results_stale):

    if run_btn:
        with st.spinner(f"Sampling distributions and running {n_sims:,} scenarios..."):
            np.random.seed(int(seed))

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

        # ââ DEBUG: show exactly what was passed to the engine ââ
        with st.expander("DEBUG - Parameters used in this run (remove after diagnosis)", expanded=True):
            st.write(f"**Multiplier dist:** `{mu_dist}`")
            st.write(f"**Multiplier params:** `{mu_params}`")
            st.write(f"**Duration dist:** `{du_dist}`")
            st.write(f"**Duration params:** `{du_params}`")
            mult_sample = draws["mult"]
            dur_sample  = draws["dur"]
            st.write(f"**Mult draws** â mean: `{mult_sample.mean():.3f}`  P10: `{np.percentile(mult_sample,10):.3f}`  P50: `{np.percentile(mult_sample,50):.3f}`  P90: `{np.percentile(mult_sample,90):.3f}`  P(mult<1): `{(mult_sample<1).mean()*100:.1f}%`")
            unique_dur, counts = np.unique(np.round(dur_sample).astype(int), return_counts=True)
            dur_freq = {int(k): f"{v/len(dur_sample)*100:.1f}%" for k,v in zip(unique_dur, counts)}
            st.write(f"**Dur draws** â mean: `{dur_sample.mean():.2f}`  frequencies: `{dur_freq}`")
            st.write(f"**GP=0 frequency:** `{(df['total_gp_dist']==0).mean()*100:.1f}%`")
        st.session_state["mc_results"]     = df
        st.session_state["mc_fingerprint"] = _current_fingerprint
        st.session_state["mc_meta"] = {
            "fs": (fs_dist, fs_params), "mf": (mf_dist, mf_params),
            "cr": (cr_dist, cr_params), "hu": (hu_dist, hu_params),
            "ip": (ip_dist, ip_params), "mu": (mu_dist, mu_params),
            "du": (du_dist, du_params), "mk": (mk_dist, mk_params),
        }
    else:
        df = st.session_state["mc_results"]

    # Warn user if displayed results are from different parameters
    if _results_stale and not run_btn:
        st.warning("Parameters have changed since the last run. Click Run Simulation to update results.", icon="!")

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
        # Derive a darker shade for the bar outline
        marker_line_colors = {
            "#58a6ff": "#1a5fa8",   # blue
            "#3fb950": "#1a7f37",   # green
            "#8957e5": "#5a2db5",   # purple
            "#f0883e": "#b85e1a",   # orange
            "#f78166": "#b84c37",   # red-orange
        }
        line_color = marker_line_colors.get(color, "rgba(0,0,0,0.3)")
        fig.add_trace(go.Histogram(x=cl, nbinsx=nbins,
                                   marker_color=color, opacity=0.85,
                                   histnorm="probability density",
                                   marker_line=dict(color=line_color, width=0.5)))
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

    gp_dist_s = s("total_gp_dist")

    c1, c2, c3, c4 = st.columns(4)
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
    with c4:
        st.plotly_chart(hist_fig(gp_dist_s, "#f0883e", "GP Distributions (C54)", "$M",
            [(gp_dist_s.median(), "dash", f"Median ${gp_dist_s.median():.1f}M", "top right"),
             (0.0, "dot", "0 (no carry)", "top left")]),
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
        "GP Dist $M (C54)":     prow("total_gp_dist", "{:.1f}"),
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
        "Mean GP Dist (C54)":   f"${gp_dist_s.mean():.1f}M",
        "Median GP Dist (C54)": f"${gp_dist_s.median():.1f}M",
        "P(GP = 0)":            f"{(gp_dist_s == 0).mean()*100:.1f}%",
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
