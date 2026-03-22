"""
Valuing Market Entry Pricing Strategies:
A Switch Option Analysis based on Bass Diffusion Dynamics
Master Thesis - Martin Bischof, FHV Vorarlberg University of Applied Sciences

Streamlit Application for Interactive Simulation
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from model import (
    run_monte_carlo,
    compute_scenario_stats,
    SEGMENT_LABELS,
    SEGMENT_SHARES,
    ARPU_DEFAULTS,
    CM_DEFAULTS,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Switch Option Analysis | Bass Diffusion",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; letter-spacing: -0.02em; }

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    line-height: 1.15;
    color: #0f1117;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 2rem;
}
.metric-card {
    background: #f8f9fa;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #1a1a2e;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #0f1117;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.72rem;
    color: #9ca3af;
    margin-top: 0.2rem;
}
.option-a { border-left-color: #2563eb; }
.option-b { border-left-color: #dc2626; }
.option-c { border-left-color: #059669; }
.section-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #9ca3af;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}
.callout {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-size: 0.78rem;
    color: #78350f;
    margin-bottom: 1rem;
}
.callout-info {
    background: #eff6ff;
    border-color: #93c5fd;
    color: #1e3a8a;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Switch Option Analysis<br><i>based on Bass Diffusion Dynamics</i></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Market Entry Pricing Strategies · Valuation Framework · Monte Carlo Simulation</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PARAMETER INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    # ── SIMULATION ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Simulation</div>', unsafe_allow_html=True)
    n_sims = st.slider("Monte Carlo Paths (N)", 500, 5000, 1500, 500)
    T = st.slider("Time Horizon (Years)", 3, 10, 5)
    discount_rate = st.slider("Discount Rate r (%)", 2.0, 20.0, 8.0, 0.5) / 100

    # ── MARKET POTENTIAL ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Market Potential (SOM)</div>', unsafe_allow_html=True)
    M_A = st.number_input("SOM Option A (customers)", 100, 5000, 745)
    M_B = st.number_input("SOM Option B (customers)", 100, 5000, 994)
    N_domestic = st.number_input("Domestic market size", 100, 5000, 689)

    # ── DIFFUSION PARAMETERS ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Diffusion Parameters</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Option A (Standard)")
        p_A_mean = st.number_input("p_A mean", 0.001, 0.2, 0.02, 0.001, format="%.3f")
        q_A_mean = st.number_input("q_A mean", 0.01, 0.8, 0.30, 0.01, format="%.2f")
        C_A_mean = st.number_input("C_A mean (%)", 1.0, 50.0, 12.0, 1.0) / 100
    with col2:
        st.caption("Option B (Fighter)")
        p_B_mean = st.number_input("p_B mean", 0.001, 0.2, 0.05, 0.001, format="%.3f")
        q_B_mean = st.number_input("q_B mean", 0.01, 0.8, 0.45, 0.01, format="%.2f")
        C_B_mean = st.number_input("C_B mean (%)", 1.0, 50.0, 22.0, 1.0) / 100

    sigma_pct = st.slider("Parameter Volatility σ (%)", 5, 40, 20) / 100

    # ── FINANCIALS ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Financial Parameters (CHF)</div>', unsafe_allow_html=True)
    capex = st.number_input("CapEx (CHF)", 0, 1_000_000, 150_000, 10_000)
    opex_annual = st.number_input("OpEx / year (CHF)", 0, 500_000, 60_000, 5_000)

    st.caption("ARPU per segment (CHF/year)")
    arpu_A_vals = [st.number_input(f"ARPU_A Seg {s}", 100, 500_000, v, 100) for s, v in zip(SEGMENT_LABELS, [100_000, 15_000, 2_000])]
    arpu_B_vals = [st.number_input(f"ARPU_B Seg {s}", 100, 500_000, v, 100) for s, v in zip(SEGMENT_LABELS, [80_000, 11_000, 1_500])]

    st.caption("CM per segment (CHF/year)")
    cm_A_vals  = [st.number_input(f"CM_A Seg {s}", 100, 200_000, v, 100) for s, v in zip(SEGMENT_LABELS, [25_000, 4_500, 800])]
    cm_B_vals  = [st.number_input(f"CM_B Seg {s}", 100, 200_000, v, 100) for s, v in zip(SEGMENT_LABELS, [16_000, 2_500, 400])]

    # ── CANNIBALIZATION ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Cannibalization</div>', unsafe_allow_html=True)
    kappa = st.slider("κ (cannibalization rate)", 0.0, 0.3, 0.05, 0.005, format="%.3f")

    # ── SWITCH OPTION ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Switch Option (C)</div>', unsafe_allow_html=True)
    switch_trigger_pct = st.slider("Bellman Trigger: W_B growth < x% p.a.", -50, 50, 0) / 100
    grandfathering = st.checkbox("Enable Grandfathering (Soft Switch)", value=True)
    churn_shock = st.slider("Hard-Switch Churn Shock (%)", 0, 80, 35) / 100

    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# RUN MODEL
# ─────────────────────────────────────────────────────────────────────────────
params = dict(
    T=T,
    M_A=M_A, M_B=M_B, N_domestic=N_domestic,
    p_A=p_A_mean, q_A=q_A_mean, C_A=C_A_mean,
    p_B=p_B_mean, q_B=q_B_mean, C_B=C_B_mean,
    sigma=sigma_pct,
    arpu_A=arpu_A_vals, arpu_B=arpu_B_vals,
    cm_A=cm_A_vals, cm_B=cm_B_vals,
    kappa=kappa,
    capex=capex, opex=opex_annual,
    discount_rate=discount_rate,
    switch_trigger=switch_trigger_pct,
    grandfathering=grandfathering,
    churn_shock=churn_shock,
)

if "results" not in st.session_state or run_btn:
    with st.spinner("Running Monte Carlo simulation…"):
        results = run_monte_carlo(n_sims=n_sims, **params)
    st.session_state["results"] = results
    st.session_state["params"] = params

results = st.session_state["results"]
stats   = compute_scenario_stats(results)

# ─────────────────────────────────────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "👥 Customer Base",
    "💰 Net Value Contribution",
    "🔀 Switch Option",
    "🔬 Sensitivity",
])

COLOR = {"A": "#2563eb", "B": "#dc2626", "C": "#059669"}

def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### ENPV Summary — All Strategies")

    # KPI cards
    c1, c2, c3 = st.columns(3)

    def fmt_chf(v):
        if abs(v) >= 1e6:
            return f"CHF {v/1e6:.2f}M"
        return f"CHF {v/1e3:.1f}K"

    for col, label, key, cls in [
        (c1, "Option A · Standard",  "A", "option-a"),
        (c2, "Option B · Fighter",   "B", "option-b"),
        (c3, "Option C · Switch",    "C", "option-c"),
    ]:
        s = stats[key]
        col.markdown(f"""
        <div class="metric-card {cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{fmt_chf(s['enpv_mean'])}</div>
            <div class="metric-sub">
                P5: {fmt_chf(s['enpv_p5'])} &nbsp;·&nbsp; P95: {fmt_chf(s['enpv_p95'])}<br>
                Prob. positive: <b>{s['prob_positive']:.0%}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    # ENPV distribution
    fig_dist = go.Figure()
    for key, name in [("A","Option A"), ("B","Option B"), ("C","Option C")]:
        enpv_vals = results[key]["enpv_series"]
        fig_dist.add_trace(go.Histogram(
            x=enpv_vals / 1e3,
            name=name,
            opacity=0.65,
            nbinsx=60,
            marker_color=COLOR[key],
        ))
    fig_dist.update_layout(
        barmode="overlay",
        title="ENPV Distribution (CHF '000)",
        xaxis_title="ENPV (CHF '000)",
        yaxis_title="Frequency",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
    )
    fig_dist.add_vline(x=0, line_dash="dot", line_color="black", opacity=0.4)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Option value table
    st.markdown("#### Option Value Decomposition")
    rows = []
    for key, name in [("A","Option A"), ("B","Option B"), ("C","Option C (Switch)")]:
        s = stats[key]
        rows.append({
            "Strategy": name,
            "Static NPV (mean)": fmt_chf(s["npv_mean"]),
            "Option Value": fmt_chf(s["enpv_mean"] - s["npv_mean"]),
            "ENPV (mean)": fmt_chf(s["enpv_mean"]),
            "Std Dev": fmt_chf(s["enpv_std"]),
            "P(ENPV > 0)": f"{s['prob_positive']:.1%}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)

    st.markdown("""
    <div class="callout callout-info">
    <b>ENPV = Static NPV + Option Value</b>&nbsp; (Trigeorgis, 1996) — The Option Value quantifies
    the monetary worth of managerial flexibility (the ability to switch strategies). A positive
    option value justifies tolerating a lower static NPV when starting with Option B.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – CUSTOMER BASE
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Synthesized Bass Diffusion — Active Customer Base N(t)")

    years = list(range(1, T + 1))
    fig_cust = go.Figure()

    for key, name in [("A","Option A"), ("B","Option B"), ("C","Option C")]:
        ac_mean = results[key]["ac_mean"]
        ac_p5   = results[key]["ac_p5"]
        ac_p95  = results[key]["ac_p95"]

        fig_cust.add_trace(go.Scatter(
            x=years, y=ac_p95, mode="lines",
            line=dict(width=0), showlegend=False,
            fillcolor=hex_rgba(COLOR[key], 0.12),
        ))
        fig_cust.add_trace(go.Scatter(
            x=years, y=ac_p5, mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=hex_rgba(COLOR[key], 0.12),
            showlegend=False,
        ))
        fig_cust.add_trace(go.Scatter(
            x=years, y=ac_mean, mode="lines+markers",
            name=name,
            line=dict(color=COLOR[key], width=2.5),
            marker=dict(size=6),
        ))

    fig_cust.update_layout(
        xaxis_title="Year",
        yaxis_title="Active Customers",
        height=400,
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cust, use_container_width=True)

    # Cannibalization overlay
    st.markdown("#### Cannibalization Dynamics — K(t) from Domestic Market")
    fig_kann = go.Figure()
    for key, name in [("B","Option B"), ("C","Option C")]:
        fig_kann.add_trace(go.Scatter(
            x=years, y=results[key]["k_mean"],
            mode="lines+markers", name=f"{name} — K(t)",
            line=dict(color=COLOR[key], width=2, dash="dash"),
        ))
    fig_kann.add_hline(
        y=N_domestic, line_dash="dot", line_color="gray",
        annotation_text="Domestic Market Ceiling", annotation_position="top left"
    )
    fig_kann.update_layout(
        xaxis_title="Year", yaxis_title="Cannibalized Customers",
        height=320, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
    )
    st.plotly_chart(fig_kann, use_container_width=True)

    # Segment breakdown table
    st.markdown("#### Segment Breakdown (Year 5 Median)")
    seg_rows = []
    for key, name in [("A","Option A"), ("B","Option B"), ("C","Option C")]:
        for j, seg in enumerate(SEGMENT_LABELS):
            seg_rows.append({
                "Strategy": name,
                "Segment": seg,
                "Share of Base": f"{SEGMENT_SHARES[j]:.0%}",
                "Active Customers (median)": int(results[key]["ac_seg_median"][-1][j]),
                "ARPU (CHF)": (arpu_A_vals if key == "A" else arpu_B_vals)[j],
                "CM (CHF)": (cm_A_vals if key in ("A","C") else cm_B_vals)[j],
            })
    st.dataframe(pd.DataFrame(seg_rows), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 – NET VALUE CONTRIBUTION
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### W(t) — Net Value Contribution per Period")

    fig_w = go.Figure()
    for key, name in [("A","Option A"), ("B","Option B"), ("C","Option C")]:
        w_mean = results[key]["w_mean"]
        w_p5   = results[key]["w_p5"]
        w_p95  = results[key]["w_p95"]

        fig_w.add_trace(go.Scatter(
            x=years, y=[v/1e3 for v in w_p95], mode="lines",
            line=dict(width=0), showlegend=False, fillcolor=hex_rgba(COLOR[key], 0.12),
        ))
        fig_w.add_trace(go.Scatter(
            x=years, y=[v/1e3 for v in w_p5], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor=hex_rgba(COLOR[key], 0.12), showlegend=False,
        ))
        fig_w.add_trace(go.Scatter(
            x=years, y=[v/1e3 for v in w_mean],
            mode="lines+markers", name=name,
            line=dict(color=COLOR[key], width=2.5), marker=dict(size=6),
        ))

    fig_w.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.3)
    fig_w.update_layout(
        xaxis_title="Year", yaxis_title="W(t) [CHF '000]",
        height=400, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_w, use_container_width=True)

    # Cumulative discounted NPV path
    st.markdown("#### Cumulative Discounted NPV Path")
    fig_cum = go.Figure()
    for key, name in [("A","Option A"), ("B","Option B"), ("C","Option C")]:
        cum = results[key]["cum_npv_mean"]
        fig_cum.add_trace(go.Scatter(
            x=[0] + years, y=[0] + [v/1e3 for v in cum],
            mode="lines+markers", name=name,
            line=dict(color=COLOR[key], width=2.5), marker=dict(size=5),
        ))
    fig_cum.add_hline(y=-capex/1e3, line_dash="dot", line_color="gray",
                      annotation_text="−CapEx", annotation_position="top left")
    fig_cum.add_hline(y=0, line_color="black", opacity=0.3)
    fig_cum.update_layout(
        xaxis_title="Year", yaxis_title="Cum. Discounted NPV [CHF '000]",
        height=380, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Margin erosion
    st.markdown("#### Margin Erosion due to Cannibalization (ΔCM, CHF '000)")
    fig_dcm = go.Figure()
    for key, name in [("B","Option B"), ("C","Option C")]:
        fig_dcm.add_trace(go.Bar(
            x=[f"Y{y}" for y in years],
            y=[v/1e3 for v in results[key]["dcm_mean"]],
            name=name, marker_color=COLOR[key], opacity=0.8,
        ))
    fig_dcm.update_layout(
        barmode="group", xaxis_title="Year",
        yaxis_title="ΔCM Total [CHF '000]",
        height=300, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
    )
    st.plotly_chart(fig_dcm, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 – SWITCH OPTION
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Switch Option Dynamics — Bellman Logic")

    col_sw1, col_sw2 = st.columns([2, 1])

    with col_sw1:
        # Switch timing distribution
        switch_times = results["C"]["switch_times"]
        switch_times_clean = [t for t in switch_times if t is not None]

        fig_sw = go.Figure()
        if switch_times_clean:
            fig_sw.add_trace(go.Histogram(
                x=switch_times_clean, nbinsx=T,
                marker_color=COLOR["C"], opacity=0.8,
                name="Switch at year t",
            ))
            never_pct = 1 - len(switch_times_clean) / len(switch_times)
            fig_sw.add_annotation(
                text=f"Never switched: {never_pct:.1%}",
                xref="paper", yref="paper", x=0.98, y=0.95,
                showarrow=False, font=dict(size=11, color="#6b7280"),
                bgcolor="white", bordercolor="#e5e7eb", borderwidth=1,
            )
        fig_sw.update_layout(
            title="Distribution of Optimal Switch Timing (t_bellman)",
            xaxis_title="Year of Switch", yaxis_title="Frequency",
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
            height=350, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        )
        st.plotly_chart(fig_sw, use_container_width=True)

    with col_sw2:
        st.markdown('<div class="section-label">Switch Option Value</div>', unsafe_allow_html=True)
        ov = stats["C"]["enpv_mean"] - stats["B"]["enpv_mean"]
        st.markdown(f"""
        <div class="metric-card option-c">
            <div class="metric-label">Option Value (C vs B)</div>
            <div class="metric-value">{fmt_chf(ov)}</div>
            <div class="metric-sub">Additional value from<br>strategic flexibility</div>
        </div>""", unsafe_allow_html=True)

        median_switch = np.median(switch_times_clean) if switch_times_clean else None
        st.markdown(f"""
        <div class="metric-card option-c">
            <div class="metric-label">Median t_bellman</div>
            <div class="metric-value">Year {f"{median_switch:.1f}" if median_switch else "—"}</div>
            <div class="metric-sub">Optimal switch timing</div>
        </div>""", unsafe_allow_html=True)

        gran_label = "Grandfathering ON" if grandfathering else "Hard Switch"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Switch Mode</div>
            <div class="metric-value" style="font-size:1.1rem">{gran_label}</div>
            <div class="metric-sub">Churn shock: {churn_shock:.0%}</div>
        </div>""", unsafe_allow_html=True)

    # Scenario comparison: C1 vs C2
    st.markdown("#### ENPV by Switch Scenario")
    scenario_labels = ["Option A\n(Standard)", "Option B\n(Fighter)", "Option C\n(Switch)"]
    enpv_means = [stats[k]["enpv_mean"] / 1e3 for k in ("A", "B", "C")]
    enpv_p5    = [stats[k]["enpv_p5"] / 1e3 for k in ("A", "B", "C")]
    enpv_p95   = [stats[k]["enpv_p95"] / 1e3 for k in ("A", "B", "C")]
    err_lo = [m - p5 for m, p5 in zip(enpv_means, enpv_p5)]
    err_hi = [p95 - m for m, p95 in zip(enpv_means, enpv_p95)]

    fig_bar = go.Figure(go.Bar(
        x=scenario_labels,
        y=enpv_means,
        error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo),
        marker_color=[COLOR["A"], COLOR["B"], COLOR["C"]],
        width=0.4,
        text=[fmt_chf(v * 1e3) for v in enpv_means],
        textposition="outside",
    ))
    fig_bar.add_hline(y=0, line_color="black", opacity=0.4, line_dash="dot")
    fig_bar.update_layout(
        yaxis_title="ENPV [CHF '000]", height=380,
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Path-dependent cashflows post-switch
    st.markdown("#### Post-Switch Cash Flow — Cohort Comparison")
    fig_cohort = go.Figure()
    for key, name, style in [("B","Option B (pre-switch)","dash"), ("A","Option A (steady)","solid"), ("C","Option C (switch path)","dot")]:
        fig_cohort.add_trace(go.Scatter(
            x=years, y=[v/1e3 for v in results[key]["w_mean"]],
            mode="lines", name=name,
            line=dict(color=COLOR[key], width=2, dash=style),
        ))
    fig_cohort.update_layout(
        xaxis_title="Year", yaxis_title="W(t) [CHF '000]",
        height=320, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cohort, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 – SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Sensitivity Analysis — Key Value Drivers")

    st.markdown('<div class="callout">Tornado chart shows the marginal impact of ±20% variation in each parameter on mean ENPV of Option C.</div>', unsafe_allow_html=True)

    # ── Tornado chart ────────────────────────────────────────────────────────
    from model import sensitivity_tornado
    tornado_data = sensitivity_tornado(base_params=params, n_sims=400)

    labels = [d["param"] for d in tornado_data]
    lo     = [d["delta_lo"] / 1e3 for d in tornado_data]
    hi     = [d["delta_hi"] / 1e3 for d in tornado_data]

    fig_tornado = go.Figure()
    for i, (l, h, lab) in enumerate(zip(lo, hi, labels)):
        color_lo = "#dc2626" if l < 0 else "#059669"
        color_hi = "#059669" if h > 0 else "#dc2626"
        fig_tornado.add_trace(go.Bar(
            name="-20%", y=[lab], x=[l], orientation="h",
            marker_color=color_lo, showlegend=(i == 0),
            legendgroup="-20%",
        ))
        fig_tornado.add_trace(go.Bar(
            name="+20%", y=[lab], x=[h], orientation="h",
            marker_color=color_hi, showlegend=(i == 0),
            legendgroup="+20%",
        ))

    fig_tornado.update_layout(
        barmode="relative",
        title="Tornado Chart: ΔENPV_C [CHF '000] for ±20% Parameter Shift",
        xaxis_title="ΔENPV [CHF '000]",
        height=420,
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_tornado.add_vline(x=0, line_color="black", opacity=0.5)
    st.plotly_chart(fig_tornado, use_container_width=True)

    # ── κ sweep ──────────────────────────────────────────────────────────────
    st.markdown("#### ENPV vs. Cannibalization Rate κ")
    kappa_range = np.linspace(0, 0.25, 20)
    enpv_kappa_A, enpv_kappa_B, enpv_kappa_C = [], [], []

    sweep_params = dict(params)
    with st.spinner("Running κ sweep…"):
        for k_val in kappa_range:
            sweep_params["kappa"] = k_val
            res_k = run_monte_carlo(n_sims=300, **sweep_params)
            s_k   = compute_scenario_stats(res_k)
            enpv_kappa_A.append(s_k["A"]["enpv_mean"] / 1e3)
            enpv_kappa_B.append(s_k["B"]["enpv_mean"] / 1e3)
            enpv_kappa_C.append(s_k["C"]["enpv_mean"] / 1e3)

    fig_kappa = go.Figure()
    for vals, name, key in [(enpv_kappa_A,"Option A","A"), (enpv_kappa_B,"Option B","B"), (enpv_kappa_C,"Option C","C")]:
        fig_kappa.add_trace(go.Scatter(
            x=kappa_range, y=vals, mode="lines",
            name=name, line=dict(color=COLOR[key], width=2.5),
        ))
    fig_kappa.add_vline(x=kappa, line_dash="dot", line_color="gray",
                        annotation_text=f"Current κ={kappa:.3f}", annotation_position="top right")
    fig_kappa.add_hline(y=0, line_color="black", opacity=0.3)
    fig_kappa.update_layout(
        xaxis_title="Cannibalization Rate κ", yaxis_title="ENPV [CHF '000]",
        height=360, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_kappa, use_container_width=True)

    # ── Churn sweep ───────────────────────────────────────────────────────────
    st.markdown("#### ENPV vs. Churn Rate C (Option B baseline)")
    churn_range = np.linspace(0.05, 0.45, 20)
    enpv_churn_B, enpv_churn_C = [], []
    sweep_params2 = dict(params)
    with st.spinner("Running churn sweep…"):
        for c_val in churn_range:
            sweep_params2["C_B"] = c_val
            sweep_params2["kappa"] = kappa
            res_c = run_monte_carlo(n_sims=300, **sweep_params2)
            s_c   = compute_scenario_stats(res_c)
            enpv_churn_B.append(s_c["B"]["enpv_mean"] / 1e3)
            enpv_churn_C.append(s_c["C"]["enpv_mean"] / 1e3)

    fig_churn = go.Figure()
    for vals, name, key in [(enpv_churn_B,"Option B","B"), (enpv_churn_C,"Option C","C")]:
        fig_churn.add_trace(go.Scatter(
            x=churn_range * 100, y=vals, mode="lines",
            name=name, line=dict(color=COLOR[key], width=2.5),
        ))
    fig_churn.add_vline(x=C_B_mean * 100, line_dash="dot", line_color="gray",
                        annotation_text=f"C_B={C_B_mean:.0%}", annotation_position="top right")
    fig_churn.add_hline(y=0, line_color="black", opacity=0.3)
    fig_churn.update_layout(
        xaxis_title="Churn Rate C_B (%)", yaxis_title="ENPV [CHF '000]",
        height=340, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_churn, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size:0.7rem; color:#9ca3af; font-family:'DM Mono',monospace;">
Master Thesis · Martin Bischof · FHV Vorarlberg University of Applied Sciences · 2026<br>
Bass (1969) · Trigeorgis (1996) · Dixit & Pindyck (1994) · Kulatilaka & Trigeorgis (1994) · Homburg et al. (2005)
</div>
""", unsafe_allow_html=True)
