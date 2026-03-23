"""
Switch Option Analysis — Streamlit App
Masterarbeit: Martin Bischof, FHV Vorarlberg University of Applied Sciences
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
.formula-box {
    background: #1e1e2e;
    color: #cdd6f4;
    border-radius: 6px;
    padding: 1rem 1.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">Switch Option Analysis<br>'
    '<i>based on Bass Diffusion Dynamics</i></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Market Entry Pricing Strategies · '
    'Valuation Framework · Monte Carlo Simulation</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
COLOR = {"A": "#2563eb", "B": "#dc2626", "C": "#059669"}

def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def fmt_chf(v):
    if abs(v) >= 1e6:
        return f"CHF {v/1e6:.2f}M"
    return f"CHF {v/1e3:.1f}K"

PLOT_STYLE = dict(paper_bgcolor="white", plot_bgcolor="#f8f9fa")
LEGEND_H   = dict(orientation="h", yanchor="bottom", y=1.02)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PARAMETER INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    st.markdown('<div class="section-label">Simulation</div>', unsafe_allow_html=True)
    n_sims        = st.slider("Monte Carlo Paths (N)", 500, 5000, 1500, 500)
    T             = st.slider("Time Horizon (Years)", 3, 10, 5)
    discount_rate = st.slider("Discount Rate r (%)", 2.0, 20.0, 8.0, 0.5) / 100

    st.markdown('<div class="section-label">Market Potential (SOM)</div>', unsafe_allow_html=True)
    M_A        = st.number_input("SOM Option A (customers)", 100, 5000, 745)
    M_B        = st.number_input("SOM Option B (customers)", 100, 5000, 994)
    N_domestic = st.number_input("Domestic market size N^dom", 100, 5000, 689)

    st.markdown('<div class="section-label">Diffusion Parameters</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Option A (Standard)")
        p_A_mean = st.number_input("p_A", 0.001, 0.2, 0.02, 0.001, format="%.3f")
        q_A_mean = st.number_input("q_A", 0.01, 0.8, 0.30, 0.01, format="%.2f")
        C_A_mean = st.number_input("C_A (%)", 1.0, 50.0, 12.0, 1.0) / 100
    with col2:
        st.caption("Option B (Fighter)")
        p_B_mean = st.number_input("p_B", 0.001, 0.2, 0.05, 0.001, format="%.3f")
        q_B_mean = st.number_input("q_B", 0.01, 0.8, 0.45, 0.01, format="%.2f")
        C_B_mean = st.number_input("C_B (%)", 1.0, 50.0, 22.0, 1.0) / 100

    sigma_pct = st.slider("Parameter Volatility σ (%)", 5, 40, 20) / 100

    st.markdown('<div class="section-label">Financial Parameters (CHF)</div>', unsafe_allow_html=True)
    capex       = st.number_input("CapEx (CHF)", 0, 1_000_000, 150_000, 10_000)
    opex_annual = st.number_input("OpEx / year (CHF)", 0, 500_000, 60_000, 5_000)

    st.caption("CM per segment (CHF/year)")
    cm_A_vals = [
        st.number_input(f"CM_A Seg {s}", 100, 200_000, v, 100)
        for s, v in zip(SEGMENT_LABELS, [25_000, 4_500, 800])
    ]
    cm_B_vals = [
        st.number_input(f"CM_B Seg {s}", 100, 200_000, v, 100)
        for s, v in zip(SEGMENT_LABELS, [16_000, 2_500, 400])
    ]

    st.caption("ARPU per segment (CHF/year)")
    arpu_A_vals = [
        st.number_input(f"ARPU_A Seg {s}", 100, 500_000, v, 100)
        for s, v in zip(SEGMENT_LABELS, [100_000, 15_000, 2_000])
    ]
    arpu_B_vals = [
        st.number_input(f"ARPU_B Seg {s}", 100, 500_000, v, 100)
        for s, v in zip(SEGMENT_LABELS, [80_000, 11_000, 1_500])
    ]

    st.markdown('<div class="section-label">Kannibalisierung</div>', unsafe_allow_html=True)
    kappa = st.slider("κ (Kannibalisierungsrate)", 0.0, 0.3, 0.05, 0.005, format="%.3f")

    st.markdown('<div class="section-label">Switch Option (C) — Bellman</div>', unsafe_allow_html=True)
    st.caption(
        "S_BA_soft: Grandfathering-Kosten als Anteil des jährl. CM-Volumens.\n"
        "S_BA_hard: wird endogen als churn_shock × AC × CM berechnet."
    )
    switch_trigger_pct = st.slider(
        "S_BA_soft: Grandfathering-Kosten (%  Jahres-CM)", 0, 20, 2
    ) / 100
    grandfathering = st.checkbox("C2: Grandfathering (Soft Switch)", value=True)
    churn_shock    = st.slider("C1: Hard-Switch Churn Shock (%)", 0, 80, 35) / 100

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
    with st.spinner("Monte Carlo Simulation läuft…"):
        results = run_monte_carlo(n_sims=n_sims, **params)
    st.session_state["results"] = results
    st.session_state["params"]  = params

results = st.session_state["results"]
stats   = compute_scenario_stats(results)
years   = list(range(1, T + 1))

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "👥 Customer Base",
    "💰 Net Value Contribution",
    "🔀 Switch Option (Bellman)",
    "🔬 Sensitivity",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### ENPV Summary — All Strategies")

    c1, c2, c3 = st.columns(3)
    for col, label, key, cls in [
        (c1, "Option A · Standard", "A", "option-a"),
        (c2, "Option B · Fighter",  "B", "option-b"),
        (c3, "Option C · Switch",   "C", "option-c"),
    ]:
        s = stats[key]
        col.markdown(f"""
        <div class="metric-card {cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{fmt_chf(s['enpv_mean'])}</div>
            <div class="metric-sub">
                P5: {fmt_chf(s['enpv_p5'])} &nbsp;·&nbsp; P95: {fmt_chf(s['enpv_p95'])}<br>
                Prob. positiv: <b>{s['prob_positive']:.0%}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    # ENPV Distribution
    fig_dist = go.Figure()
    for key, name in [("A", "Option A"), ("B", "Option B"), ("C", "Option C")]:
        fig_dist.add_trace(go.Histogram(
            x=results[key]["enpv_series"] / 1e3,
            name=name, opacity=0.65, nbinsx=60,
            marker_color=COLOR[key],
        ))
    fig_dist.update_layout(
        barmode="overlay",
        title="ENPV Distribution (CHF '000)",
        xaxis_title="ENPV (CHF '000)", yaxis_title="Frequency",
        legend=LEGEND_H, height=380, **PLOT_STYLE,
    )
    fig_dist.add_vline(x=0, line_dash="dot", line_color="black", opacity=0.4)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Option Value Tabelle
    st.markdown("#### Option Value Decomposition")
    rows = []
    for key, name in [("A", "Option A"), ("B", "Option B"), ("C", "Option C (Switch)")]:
        s = stats[key]
        rows.append({
            "Strategy":         name,
            "Static NPV (mean)": fmt_chf(s["npv_mean"]),
            "Option Value":      fmt_chf(s["enpv_mean"] - s["npv_mean"]),
            "ENPV (mean)":       fmt_chf(s["enpv_mean"]),
            "Std Dev":           fmt_chf(s["enpv_std"]),
            "P(ENPV > 0)":       f"{s['prob_positive']:.1%}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)

    st.markdown("""
    <div class="callout callout-info">
    <b>ENPV = Static NPV + Option Value</b> (Trigeorgis, 1996) — Der Option Value quantifiziert
    den monetären Wert der Managementflexibilität. Ein positiver Option Value rechtfertigt einen
    niedrigeren Static NPV bei Start mit Option B.
    </div>
    """, unsafe_allow_html=True)

    # Bar Chart Vergleich
    enpv_means = [stats[k]["enpv_mean"] / 1e3 for k in ("A", "B", "C")]
    enpv_p5    = [stats[k]["enpv_p5"]   / 1e3 for k in ("A", "B", "C")]
    enpv_p95   = [stats[k]["enpv_p95"]  / 1e3 for k in ("A", "B", "C")]
    fig_bar = go.Figure(go.Bar(
        x=["Option A\n(Standard)", "Option B\n(Fighter)", "Option C\n(Switch)"],
        y=enpv_means,
        error_y=dict(
            type="data", symmetric=False,
            array=[p95 - m for m, p95 in zip(enpv_means, enpv_p95)],
            arrayminus=[m - p5 for m, p5 in zip(enpv_means, enpv_p5)],
        ),
        marker_color=[COLOR["A"], COLOR["B"], COLOR["C"]],
        width=0.4,
        text=[fmt_chf(v * 1e3) for v in enpv_means],
        textposition="outside",
    ))
    fig_bar.add_hline(y=0, line_color="black", opacity=0.4, line_dash="dot")
    fig_bar.update_layout(
        yaxis_title="ENPV [CHF '000]", height=380,
        showlegend=False, **PLOT_STYLE,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – CUSTOMER BASE
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Synthesized Bass Diffusion — Aktiver Bestand N(t)")
    st.markdown("""
    <div class="formula-box">
    S_i(τ) = (p_i + q_i · [AC(τ−1) + N^dom] / [M_i + N^dom]) · (M_i − A_i(τ−1))<br>
    AC_i(t) = Σ_{τ} S_i(τ) · (1 − C_i)^{t−τ}
    </div>
    """, unsafe_allow_html=True)

    fig_cust = go.Figure()
    for key, name in [("A", "Option A"), ("B", "Option B"), ("C", "Option C")]:
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
            line=dict(width=0), fill="tonexty",
            fillcolor=hex_rgba(COLOR[key], 0.12), showlegend=False,
        ))
        fig_cust.add_trace(go.Scatter(
            x=years, y=ac_mean, mode="lines+markers",
            name=name, line=dict(color=COLOR[key], width=2.5),
            marker=dict(size=6),
        ))
    fig_cust.update_layout(
        xaxis_title="Jahr", yaxis_title="Aktive Kunden",
        height=400, legend=LEGEND_H, **PLOT_STYLE,
    )
    st.plotly_chart(fig_cust, use_container_width=True)

    # Kannibalisierung
    st.markdown("#### Kannibalisierungs-Dynamik — K^actual(t)")
    st.markdown("""
    <div class="formula-box">
    K_j^actual(t) = min(κ · AC(t), N_j^domestic)
    </div>
    """, unsafe_allow_html=True)
    fig_kann = go.Figure()
    for key, name in [("B", "Option B"), ("C", "Option C")]:
        fig_kann.add_trace(go.Scatter(
            x=years, y=results[key]["k_mean"],
            mode="lines+markers", name=f"{name} — K(t)",
            line=dict(color=COLOR[key], width=2, dash="dash"),
        ))
    fig_kann.add_hline(
        y=N_domestic, line_dash="dot", line_color="gray",
        annotation_text="N^domestic Ceiling", annotation_position="top left",
    )
    fig_kann.update_layout(
        xaxis_title="Jahr", yaxis_title="Kannibalisierte Kunden",
        height=320, **PLOT_STYLE,
    )
    st.plotly_chart(fig_kann, use_container_width=True)

    # Segment-Tabelle
    st.markdown("#### ABC-Kundensegmentierung (Median Jahr T)")
    seg_rows = []
    for key, name in [("A", "Option A"), ("B", "Option B"), ("C", "Option C")]:
        last_seg = results[key]["ac_seg_median"][-1]
        for j, seg in enumerate(SEGMENT_LABELS):
            seg_rows.append({
                "Strategie":              name,
                "Segment":                seg,
                "Anteil":                 f"{SEGMENT_SHARES[j]:.0%}",
                "Aktive Kunden (Median)": int(last_seg[j]),
                "ARPU (CHF)":             (arpu_A_vals if key == "A" else arpu_B_vals)[j],
                "CM (CHF)":               (cm_A_vals if key in ("A", "C") else cm_B_vals)[j],
            })
    st.dataframe(pd.DataFrame(seg_rows), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 – NET VALUE CONTRIBUTION
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### W(t) — Net Value Contribution")
    st.markdown("""
    <div class="formula-box">
    W(t) = Σ_j [AC_{i,j} · CM_j^digital − K_{i,j}^actual · ΔCM_j] − OpEx_t<br>
    W_{B→A}(t) = Σ_j [AC_1·CM^switch + AC_2·CM^A − K·ΔCM^switch] − OpEx
    </div>
    """, unsafe_allow_html=True)

    fig_w = go.Figure()
    for key, name in [("A", "Option A"), ("B", "Option B"), ("C", "Option C")]:
        w_mean = results[key]["w_mean"]
        w_p5   = results[key]["w_p5"]
        w_p95  = results[key]["w_p95"]
        fig_w.add_trace(go.Scatter(
            x=years, y=[v / 1e3 for v in w_p95], mode="lines",
            line=dict(width=0), showlegend=False,
            fillcolor=hex_rgba(COLOR[key], 0.12),
        ))
        fig_w.add_trace(go.Scatter(
            x=years, y=[v / 1e3 for v in w_p5], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor=hex_rgba(COLOR[key], 0.12), showlegend=False,
        ))
        fig_w.add_trace(go.Scatter(
            x=years, y=[v / 1e3 for v in w_mean],
            mode="lines+markers", name=name,
            line=dict(color=COLOR[key], width=2.5), marker=dict(size=6),
        ))
    fig_w.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.3)
    fig_w.update_layout(
        xaxis_title="Jahr", yaxis_title="W(t) [CHF '000]",
        height=400, legend=LEGEND_H, **PLOT_STYLE,
    )
    st.plotly_chart(fig_w, use_container_width=True)

    # Kumulierter NPV
    st.markdown("#### Kumulierter diskontierter NPV-Pfad")
    fig_cum = go.Figure()
    for key, name in [("A", "Option A"), ("B", "Option B"), ("C", "Option C")]:
        cum = results[key]["cum_npv_mean"]
        fig_cum.add_trace(go.Scatter(
            x=[0] + years, y=[0] + [v / 1e3 for v in cum],
            mode="lines+markers", name=name,
            line=dict(color=COLOR[key], width=2.5), marker=dict(size=5),
        ))
    fig_cum.add_hline(
        y=-capex / 1e3, line_dash="dot", line_color="gray",
        annotation_text="−CapEx", annotation_position="top left",
    )
    fig_cum.add_hline(y=0, line_color="black", opacity=0.3)
    fig_cum.update_layout(
        xaxis_title="Jahr", yaxis_title="Kum. Disk. NPV [CHF '000]",
        height=380, legend=LEGEND_H, **PLOT_STYLE,
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Margen-Erosion
    st.markdown("#### Margen-Erosion durch Kannibalisierung (ΔCM, CHF '000)")
    fig_dcm = go.Figure()
    for key, name in [("B", "Option B"), ("C", "Option C")]:
        fig_dcm.add_trace(go.Bar(
            x=[f"Y{y}" for y in years],
            y=[v / 1e3 for v in results[key]["dcm_mean"]],
            name=name, marker_color=COLOR[key], opacity=0.8,
        ))
    fig_dcm.update_layout(
        barmode="group", xaxis_title="Jahr",
        yaxis_title="ΔCM [CHF '000]",
        height=300, **PLOT_STYLE,
    )
    st.plotly_chart(fig_dcm, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 – SWITCH OPTION (BELLMAN)
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Switch Option — Bellman-Rückwärtsrekursion")

    st.markdown("""
    <div class="formula-box">
    Bellman-Bedingung (exakt aus der Arbeit):<br>
    &nbsp;&nbsp;F(T)   = max( V_BA(T) − S_{B,A},  W_B(T) )<br>
    &nbsp;&nbsp;F(t)   = max( V_BA(t) − S_{B,A},  W_B(t) + F(t+1)/(1+r) )<br><br>
    Switch wenn: V_BA(t) − S_{B,A}  ≥  W_B(t) + F(t+1)/(1+r)<br><br>
    ENPV_Switch = −CapEx + Σ_{t &lt; t*} W_B/(1+r)^t<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ V_BA(t*) · (1+r)^{−t*}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;− S_{B,A}/(1+r)^{t*}
    </div>
    """, unsafe_allow_html=True)

    col_sw1, col_sw2 = st.columns([2, 1])

    switch_times       = results["C"]["switch_times"]
    switch_times_clean = [t for t in switch_times if t is not None]

    with col_sw1:
        # Switch-Timing Verteilung
        fig_sw = go.Figure()
        if switch_times_clean:
            fig_sw.add_trace(go.Histogram(
                x=switch_times_clean, nbinsx=T,
                marker_color=COLOR["C"], opacity=0.85,
                name="Switch in Jahr t",
            ))
            never_pct = 1 - len(switch_times_clean) / len(switch_times)
            fig_sw.add_annotation(
                text=f"Nie geswitcht: {never_pct:.1%}",
                xref="paper", yref="paper", x=0.98, y=0.95,
                showarrow=False, font=dict(size=11, color="#6b7280"),
                bgcolor="white", bordercolor="#e5e7eb", borderwidth=1,
            )
        else:
            fig_sw.add_annotation(
                text="Kein Switch optimal (S_BA > V_BA − W_B für alle t)",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=13, color="#6b7280"),
            )
        fig_sw.update_layout(
            title="Verteilung optimaler Switch-Zeitpunkt t_bellman",
            xaxis_title="Jahr des Switches", yaxis_title="Häufigkeit",
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
            height=350, **PLOT_STYLE,
        )
        st.plotly_chart(fig_sw, use_container_width=True)

    with col_sw2:
        st.markdown('<div class="section-label">Option Value</div>', unsafe_allow_html=True)

        ov = stats["C"]["enpv_mean"] - stats["B"]["enpv_mean"]
        st.markdown(f"""
        <div class="metric-card option-c">
            <div class="metric-label">Option Value (C − B)</div>
            <div class="metric-value">{fmt_chf(ov)}</div>
            <div class="metric-sub">Monetärer Wert der<br>strategischen Flexibilität</div>
        </div>""", unsafe_allow_html=True)

        median_switch = np.median(switch_times_clean) if switch_times_clean else None
        st.markdown(f"""
        <div class="metric-card option-c">
            <div class="metric-label">Median t_bellman</div>
            <div class="metric-value">Jahr {f"{median_switch:.1f}" if median_switch else "—"}</div>
            <div class="metric-sub">Optimaler Switch-Zeitpunkt</div>
        </div>""", unsafe_allow_html=True)

        switch_rate = len(switch_times_clean) / max(len(switch_times), 1)
        gran_label  = "C2: Grandfathering" if grandfathering else "C1: Hard Switch"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Switch-Modus</div>
            <div class="metric-value" style="font-size:1.05rem">{gran_label}</div>
            <div class="metric-sub">
                Switch-Rate: {switch_rate:.1%}<br>
                Churn Shock: {churn_shock:.0%}
            </div>
        </div>""", unsafe_allow_html=True)

    # W_B vs V_BA — Bellman-Entscheidungsgrundlage
    st.markdown("#### W_B(t) vs V_BA(t) — Bellman-Entscheidungsgrundlage (Mittelwert)")
    if "W_B_mean" in results["C"] and "V_BA_mean" in results["C"]:
        fig_bell = go.Figure()
        fig_bell.add_trace(go.Scatter(
            x=years, y=[v / 1e3 for v in results["C"]["W_B_mean"]],
            mode="lines+markers", name="W_B(t) — Option B Cashflow",
            line=dict(color=COLOR["B"], width=2.5), marker=dict(size=7),
        ))
        fig_bell.add_trace(go.Scatter(
            x=years, y=[v / 1e3 for v in results["C"]["V_BA_mean"]],
            mode="lines+markers", name="V_BA(t) — Post-Switch NPV",
            line=dict(color=COLOR["C"], width=2.5, dash="dash"), marker=dict(size=7),
        ))
        # F(t) Continuation Value
        if "F_mean" in results["C"]:
            fig_bell.add_trace(go.Scatter(
                x=years, y=[v / 1e3 for v in results["C"]["F_mean"]],
                mode="lines", name="F(t) — Bellman Continuation Value",
                line=dict(color="#7c3aed", width=1.5, dash="dot"),
            ))
        # S_BA Linie (approximiert)
        mean_cm_B = float(np.dot(SEGMENT_SHARES, np.array(cm_B_vals, dtype=float)))
        fig_bell.update_layout(
            xaxis_title="Jahr",
            yaxis_title="Wert [CHF '000]",
            title="Switch-Trigger: V_BA(t) − S_BA  ≥  W_B(t) + F(t+1)/(1+r)",
            height=400, legend=LEGEND_H, **PLOT_STYLE,
        )
        st.plotly_chart(fig_bell, use_container_width=True)
        st.markdown("""
        <div class="callout callout-info">
        <b>Interpretation:</b> In Perioden wo V_BA(t) − S_{B,A} &gt; W_B(t) + F(t+1)/(1+r),
        ist der Switch optimal (t_bellman). V_BA(t) ist der diskontierte NPV aller Post-Switch
        Cashflows W_{B→A}(τ), τ ≥ t — berechnet mit korrekter Kohortenabschmelzung und
        neuem A-Kundenaufbau.
        </div>
        """, unsafe_allow_html=True)

    # Post-Switch Cashflow Vergleich
    st.markdown("#### Post-Switch Cashflow — Kohortenvergleich W(t)")
    fig_post = go.Figure()
    for key, name, style in [
        ("B", "Option B (ohne Switch)",  "dash"),
        ("A", "Option A (steady state)", "solid"),
        ("C", "Option C (Switch-Pfad)",  "dot"),
    ]:
        fig_post.add_trace(go.Scatter(
            x=years, y=[v / 1e3 for v in results[key]["w_mean"]],
            mode="lines", name=name,
            line=dict(color=COLOR[key], width=2, dash=style),
        ))
    fig_post.add_hline(y=0, line_color="black", opacity=0.3, line_dash="dot")
    fig_post.update_layout(
        xaxis_title="Jahr", yaxis_title="W(t) [CHF '000]",
        height=340, legend=LEGEND_H, **PLOT_STYLE,
    )
    st.plotly_chart(fig_post, use_container_width=True)

    # ENPV Zerlegung
    st.markdown("#### ENPV-Zerlegung — Pre-Switch (B) vs Post-Switch (B→A)")
    if "W_B_mean" in results["C"] and "V_BA_mean" in results["C"]:
        disc = np.array([1.0 / (1.0 + discount_rate) ** t for t in range(1, T + 1)])
        W_B_mean   = results["C"]["W_B_mean"]
        V_BA_mean  = results["C"]["V_BA_mean"]

        # Median t_bellman
        t_med = int(round(median_switch)) if median_switch else T + 1
        t_idx = min(t_med - 1, T - 1)

        pre_phase  = [float(W_B_mean[t] * disc[t]) / 1e3 if t < t_idx else 0 for t in range(T)]
        post_phase = [0.0] * T
        if t_idx < T:
            post_phase[t_idx] = float(V_BA_mean[t_idx] * disc[t_idx]) / 1e3

        fig_decomp = go.Figure()
        fig_decomp.add_trace(go.Bar(
            x=[f"Y{y}" for y in years], y=pre_phase,
            name="Phase B (diskontiert)", marker_color=COLOR["B"], opacity=0.8,
        ))
        fig_decomp.add_trace(go.Bar(
            x=[f"Y{y}" for y in years], y=post_phase,
            name="V_BA(t*) diskontiert", marker_color=COLOR["C"], opacity=0.8,
        ))
        fig_decomp.add_hline(y=0, line_color="black", opacity=0.3)
        fig_decomp.update_layout(
            barmode="stack",
            title=f"ENPV-Zerlegung (bei Median t_bellman = Jahr {t_med if median_switch else '—'})",
            yaxis_title="Disk. Wert [CHF '000]",
            height=320, legend=LEGEND_H, **PLOT_STYLE,
        )
        st.plotly_chart(fig_decomp, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 – SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Sensitivitätsanalyse — Key Value Drivers")

    st.markdown("""
    <div class="callout">
    Tornado Chart: Marginaler Einfluss von ±20% Parametervariation auf den
    mittleren ENPV von Option C.
    </div>
    """, unsafe_allow_html=True)

    # Tornado
    from model import sensitivity_tornado
    tornado_data = sensitivity_tornado(base_params=params, n_sims=400)

    labels = [d["param"]    for d in tornado_data]
    lo     = [d["delta_lo"] / 1e3 for d in tornado_data]
    hi     = [d["delta_hi"] / 1e3 for d in tornado_data]

    fig_tornado = go.Figure()
    for i, (l, h, lab) in enumerate(zip(lo, hi, labels)):
        fig_tornado.add_trace(go.Bar(
            name="−20%", y=[lab], x=[l], orientation="h",
            marker_color="#dc2626" if l < 0 else "#059669",
            showlegend=(i == 0), legendgroup="−20%",
        ))
        fig_tornado.add_trace(go.Bar(
            name="+20%", y=[lab], x=[h], orientation="h",
            marker_color="#059669" if h > 0 else "#dc2626",
            showlegend=(i == 0), legendgroup="+20%",
        ))
    fig_tornado.add_vline(x=0, line_color="black", opacity=0.5)
    fig_tornado.update_layout(
        barmode="relative",
        title="Tornado Chart: ΔENPV_C [CHF '000] bei ±20% Parametershift",
        xaxis_title="ΔENPV [CHF '000]",
        height=420, legend=LEGEND_H, **PLOT_STYLE,
    )
    st.plotly_chart(fig_tornado, use_container_width=True)

    # κ-Sweep
    st.markdown("#### ENPV vs. Kannibalisierungsrate κ")
    kappa_range = np.linspace(0.0, 0.25, 20)
    enpv_kA, enpv_kB, enpv_kC = [], [], []
    sweep_p = dict(params)
    with st.spinner("κ-Sweep läuft…"):
        for k_val in kappa_range:
            sweep_p["kappa"] = float(k_val)
            res_k = run_monte_carlo(n_sims=300, **sweep_p)
            s_k   = compute_scenario_stats(res_k)
            enpv_kA.append(s_k["A"]["enpv_mean"] / 1e3)
            enpv_kB.append(s_k["B"]["enpv_mean"] / 1e3)
            enpv_kC.append(s_k["C"]["enpv_mean"] / 1e3)

    fig_kappa = go.Figure()
    for vals, name, key in [
        (enpv_kA, "Option A", "A"),
        (enpv_kB, "Option B", "B"),
        (enpv_kC, "Option C", "C"),
    ]:
        fig_kappa.add_trace(go.Scatter(
            x=kappa_range, y=vals, mode="lines",
            name=name, line=dict(color=COLOR[key], width=2.5),
        ))
    fig_kappa.add_vline(
        x=kappa, line_dash="dot", line_color="gray",
        annotation_text=f"κ={kappa:.3f}", annotation_position="top right",
    )
    fig_kappa.add_hline(y=0, line_color="black", opacity=0.3)
    fig_kappa.update_layout(
        xaxis_title="Kannibalisierungsrate κ",
        yaxis_title="ENPV [CHF '000]",
        height=360, legend=LEGEND_H, **PLOT_STYLE,
    )
    st.plotly_chart(fig_kappa, use_container_width=True)

    # Churn-Sweep
    st.markdown("#### ENPV vs. Churn-Rate C_B (Option B)")
    churn_range = np.linspace(0.05, 0.45, 20)
    enpv_cB, enpv_cC = [], []
    sweep_p2 = dict(params)
    with st.spinner("Churn-Sweep läuft…"):
        for c_val in churn_range:
            sweep_p2["C_B"]   = float(c_val)
            sweep_p2["kappa"] = kappa
            res_c2 = run_monte_carlo(n_sims=300, **sweep_p2)
            s_c2   = compute_scenario_stats(res_c2)
            enpv_cB.append(s_c2["B"]["enpv_mean"] / 1e3)
            enpv_cC.append(s_c2["C"]["enpv_mean"] / 1e3)

    fig_churn = go.Figure()
    for vals, name, key in [(enpv_cB, "Option B", "B"), (enpv_cC, "Option C", "C")]:
        fig_churn.add_trace(go.Scatter(
            x=churn_range * 100, y=vals, mode="lines",
            name=name, line=dict(color=COLOR[key], width=2.5),
        ))
    fig_churn.add_vline(
        x=C_B_mean * 100, line_dash="dot", line_color="gray",
        annotation_text=f"C_B={C_B_mean:.0%}", annotation_position="top right",
    )
    fig_churn.add_hline(y=0, line_color="black", opacity=0.3)
    fig_churn.update_layout(
        xaxis_title="Churn-Rate C_B (%)",
        yaxis_title="ENPV [CHF '000]",
        height=340, legend=LEGEND_H, **PLOT_STYLE,
    )
    st.plotly_chart(fig_churn, use_container_width=True)

    # S_BA Sweep (Switching Costs)
    st.markdown("#### ENPV Option C vs. Switching Cost S_{B,A}")
    sba_range   = np.linspace(0.0, 0.20, 15)
    enpv_sba_C  = []
    sweep_p3 = dict(params)
    with st.spinner("S_BA-Sweep läuft…"):
        for s_val in sba_range:
            sweep_p3["switch_trigger"] = float(s_val)
            res_s = run_monte_carlo(n_sims=300, **sweep_p3)
            s_s   = compute_scenario_stats(res_s)
            enpv_sba_C.append(s_s["C"]["enpv_mean"] / 1e3)

    fig_sba = go.Figure()
    fig_sba.add_trace(go.Scatter(
        x=sba_range * 100, y=enpv_sba_C, mode="lines+markers",
        name="Option C", line=dict(color=COLOR["C"], width=2.5),
        marker=dict(size=6),
    ))
    fig_sba.add_hline(
        y=stats["B"]["enpv_mean"] / 1e3, line_dash="dot",
        line_color=COLOR["B"],
        annotation_text="ENPV_B (Referenz)", annotation_position="top right",
    )
    fig_sba.add_vline(
        x=switch_trigger_pct * 100, line_dash="dot", line_color="gray",
        annotation_text=f"S_BA_soft={switch_trigger_pct:.0%}", annotation_position="top left",
    )
    fig_sba.update_layout(
        xaxis_title="S_BA_soft (% Jahres-CM-Volumen)",
        yaxis_title="ENPV_C [CHF '000]",
        title="Kritischer S_{B,A}: Wo dominiert C nicht mehr über B?",
        height=340, **PLOT_STYLE,
    )
    st.plotly_chart(fig_sba, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size:0.7rem; color:#9ca3af; font-family:'DM Mono',monospace;">
Masterarbeit · Martin Bischof · FHV Vorarlberg University of Applied Sciences · 2026<br>
Bass (1969) · Trigeorgis (1996) · Dixit &amp; Pindyck (1994) ·
Kulatilaka &amp; Trigeorgis (1994) · Homburg et al. (2005)
</div>
""", unsafe_allow_html=True)
