"""
model.py — Core computation engine for Switch Option Analysis
Based on:
  - Synthesized Bass Diffusion Model (Bass 1969, extended)
  - Real Options / Switch Option (Trigeorgis 1996, Kulatilaka & Trigeorgis 1994)
  - Bellman dynamic programming (Dixit & Pindyck 1994)
  - Monte Carlo Simulation
"""

import numpy as np
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEGMENT_LABELS = ["A", "B", "C"]
SEGMENT_SHARES = np.array([0.20, 0.30, 0.50])   # % of total customers per segment
ARPU_DEFAULTS  = np.array([100_000, 15_000, 2_000])
CM_DEFAULTS    = np.array([25_000, 4_500, 800])


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-PATH SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_path(
    T: int,
    M: float,       # SOM
    N_domestic: float,
    p: float, q: float, C: float,
    arpu: np.ndarray,   # shape (3,) per segment
    cm: np.ndarray,     # shape (3,)
    kappa: float,
    capex: float,
    opex: float,
    discount_rate: float,
    mode: str = "A",   # "A", "B"
    # Switch-specific:
    switch_trigger: float = 0.0,
    grandfathering: bool = True,
    churn_shock: float = 0.35,
    p_A: float = None, q_A: float = None, C_A: float = None,
    M_A: float = None,
    arpu_A: np.ndarray = None, cm_A: np.ndarray = None,
) -> dict:
    """
    Simulate a single stochastic path for ONE strategy.

    Returns dict with per-period arrays:
        ac[t]  — active customers (total)
        k[t]   — cannibalized customers (active)
        w[t]   — Net Value Contribution W(t)
        dcm[t] — total margin erosion from cannibalization
        switch_time — year at which Bellman switch was triggered (None if never, only for mode='C')
    """
    seg_n = len(SEGMENT_SHARES)

    # --- cohort tracking (one entry per customer cohort vintage year) --------
    # cohort[tau] = number of customers still active from year tau
    cohort_organic = np.zeros(T + 1)   # indexed by acquisition year

    # cumulative adopters A_i(t) — exhausts market potential
    cum_adopters = 0.0

    # cannibalization stock (active)
    K_active = 0.0
    cum_K     = 0.0   # cumulative ever-cannibalized (for the cap with N_domestic)

    # output arrays
    ac_arr  = np.zeros(T)
    k_arr   = np.zeros(T)
    w_arr   = np.zeros(T)
    dcm_arr = np.zeros(T)

    # For Option C: second cohort (post-switch, under Option A params)
    cohort_organic_2  = np.zeros(T + 1)
    cum_adopters_2    = 0.0
    switched          = False
    switch_time_val   = None

    # --- helper: segment distribution ----------------------------------------
    def split_by_segment(n_customers):
        return np.floor(n_customers * SEGMENT_SHARES).astype(float)

    # --- per-period simulation ------------------------------------------------
    for t in range(1, T + 1):

        # ── 1. Survive existing cohorts ──────────────────────────────────────
        # Apply churn to all previous cohorts (cohort from year tau loses C fraction)
        for tau in range(1, t):
            cohort_organic[tau] *= (1 - C)

        # ── 2. New acquisitions this period (Synthesized Bass) ────────────────
        AC_total_prev = sum(cohort_organic[1:t]) if t > 1 else 0.0
        available_market = M - cum_adopters
        if available_market < 0:
            available_market = 0.0

        # Effective imitation including cross-channel customers (domestic baseline)
        q_eff = q * (1 - C)
        base_influence = AC_total_prev + N_domestic
        new_acq = (p + q_eff * AC_total_prev / max(M, 1)) * available_market
        new_acq = max(0.0, min(new_acq, available_market))

        cohort_organic[t] = new_acq
        cum_adopters += new_acq

        # ── 3. Cannibalization ───────────────────────────────────────────────
        AC_total = AC_total_prev + new_acq
        delta_K  = kappa * AC_total
        K_active = K_active * (1 - C) + delta_K
        cum_K   += delta_K
        K_actual  = min(K_active, N_domestic)   # cap at domestic market size

        # ── 4. Option C — check Bellman switch trigger ────────────────────────
        if mode == "C" and not switched:
            # Proxy: W_B growth trigger (use W_B of this vs prior period as proxy)
            # We apply trigger if AC growth rate drops below threshold
            if t >= 2:
                ac_prev = ac_arr[t - 2] if t >= 2 else 1
                growth = (AC_total - ac_prev) / max(ac_prev, 1)
                if growth <= switch_trigger:
                    switched      = True
                    switch_time_val = t

                    # Hard switch: apply churn shock to existing base
                    if not grandfathering:
                        for tau in range(1, t + 1):
                            cohort_organic[tau] *= (1 - churn_shock)

        # ── 5. Post-switch new acquisitions (cohort 2 under Option A params) ──
        new_acq_2 = 0.0
        if mode == "C" and switched:
            # second cohort uses A-parameters
            for tau in range(1, t):
                cohort_organic_2[tau] *= (1 - C_A)
            avail_2 = M_A - cum_adopters_2
            if avail_2 < 0:
                avail_2 = 0.0
            AC_total_2_prev = sum(cohort_organic_2[1:t])
            q_A_eff = q_A * (1 - C_A)
            new_acq_2 = (p_A + q_A_eff * AC_total_2_prev / max(M_A, 1)) * avail_2
            new_acq_2 = max(0.0, min(new_acq_2, avail_2))
            cohort_organic_2[t] = new_acq_2
            cum_adopters_2 += new_acq_2

        # ── 6. Active customer stock ──────────────────────────────────────────
        ac_organic_1 = sum(cohort_organic[1:t + 1])
        ac_organic_2 = sum(cohort_organic_2[1:t + 1]) if mode == "C" else 0.0
        ac_total_cur = ac_organic_1 + ac_organic_2

        # ── 7. Financial layer ─────────────────────────────────────────────────
        # Segment split
        ac_seg_1 = split_by_segment(ac_organic_1)
        ac_seg_2 = split_by_segment(ac_organic_2)
        k_seg    = split_by_segment(K_actual)

        W_t = 0.0
        dcm_t = 0.0

        for j in range(seg_n):
            # CM for cohort 1
            if mode == "C" and switched:
                cm_1j = cm_A[j] if not grandfathering else cm[j]   # grandfathering keeps old CM
            else:
                cm_1j = cm[j]

            # CM for cohort 2 (always Option A)
            cm_2j = cm_A[j] if (mode == "C" and switched) else 0.0

            # Delta CM (margin erosion per cannibalized customer)
            delta_cm_j = max(0.0, cm[j] - cm_A[j]) if mode == "C" and switched else cm[j] - cm[j]
            if mode in ("B", "C"):
                # domestic CM vs digital CM difference
                cm_domestic_j = cm_A[j] if cm_A is not None else cm[j]
                delta_cm_j = max(0.0, cm_domestic_j - cm[j])

            revenue_j = ac_seg_1[j] * cm_1j + ac_seg_2[j] * cm_2j
            cannibal_j = k_seg[j] * delta_cm_j

            W_t   += revenue_j - cannibal_j
            dcm_t += cannibal_j

        W_t  -= opex
        # Switching cost at transition year (instantaneous churn loss monetized)
        if mode == "C" and switched and switch_time_val == t and not grandfathering:
            W_t -= churn_shock * ac_organic_1 * np.mean(cm)

        # ── Store results ──────────────────────────────────────────────────────
        ac_arr[t - 1]  = ac_total_cur
        k_arr[t - 1]   = K_actual
        w_arr[t - 1]   = W_t
        dcm_arr[t - 1] = dcm_t

    # Segment breakdown for output (year-end snapshot)
    ac_seg_snapshot = [
        split_by_segment(sum(cohort_organic[1:t + 1]) + (sum(cohort_organic_2[1:t + 1]) if mode == "C" else 0))
        for t in range(1, T + 1)
    ]

    return {
        "ac":        ac_arr,
        "k":         k_arr,
        "w":         w_arr,
        "dcm":       dcm_arr,
        "switch_time": switch_time_val,
        "ac_seg":    ac_seg_snapshot,
    }


def discounted_npv(w_arr: np.ndarray, capex: float, discount_rate: float) -> float:
    T = len(w_arr)
    disc = np.array([(1 + discount_rate) ** (-t) for t in range(1, T + 1)])
    return -capex + np.sum(w_arr * disc)


def cumulative_npv(w_arr: np.ndarray, capex: float, discount_rate: float) -> np.ndarray:
    T = len(w_arr)
    disc = np.array([(1 + discount_rate) ** (-t) for t in range(1, T + 1)])
    return np.cumsum(w_arr * disc) - capex


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    n_sims: int,
    T: int,
    M_A: float, M_B: float, N_domestic: float,
    p_A: float, q_A: float, C_A: float,
    p_B: float, q_B: float, C_B: float,
    sigma: float,
    arpu_A, arpu_B, cm_A, cm_B,
    kappa: float,
    capex: float, opex: float,
    discount_rate: float,
    switch_trigger: float = 0.0,
    grandfathering: bool = True,
    churn_shock: float = 0.35,
) -> dict:
    """
    Run N Monte Carlo simulations for all three strategies.
    Returns per-strategy arrays of ENPV, NPV, customer base, W(t), K(t), ΔCM(t).
    """
    arpu_A = np.array(arpu_A, dtype=float)
    arpu_B = np.array(arpu_B, dtype=float)
    cm_A   = np.array(cm_A, dtype=float)
    cm_B   = np.array(cm_B, dtype=float)

    store = {k: {
        "enpv": [], "npv": [],
        "ac":   np.zeros((n_sims, T)),
        "k":    np.zeros((n_sims, T)),
        "w":    np.zeros((n_sims, T)),
        "dcm":  np.zeros((n_sims, T)),
        "switch_times": [],
        "ac_seg": [[None] * T for _ in range(n_sims)],
    } for k in ("A", "B", "C")}

    rng = np.random.default_rng(42)

    for i in range(n_sims):
        # Stochastic parameter draws (log-normal perturbation)
        def draw(mu, s=sigma):
            return float(np.clip(rng.lognormal(np.log(mu), s), mu * 0.1, mu * 5))

        p_a = draw(p_A); q_a = draw(q_A); c_a = float(np.clip(draw(C_A), 0.01, 0.6))
        p_b = draw(p_B); q_b = draw(q_B); c_b = float(np.clip(draw(C_B), 0.01, 0.7))
        k_i = float(np.clip(draw(kappa) if kappa > 0 else 0, 0.0, 0.4))

        # ── Option A ──────────────────────────────────────────────────────────
        res_a = simulate_path(
            T=T, M=M_A, N_domestic=N_domestic,
            p=p_a, q=q_a, C=c_a,
            arpu=arpu_A, cm=cm_A,
            kappa=0.0,   # minimal cannibalization for standard conditions
            capex=capex, opex=opex, discount_rate=discount_rate,
            mode="A",
        )
        enpv_a = discounted_npv(res_a["w"], capex, discount_rate)
        store["A"]["enpv"].append(enpv_a)
        store["A"]["npv"].append(enpv_a)  # static = no option value here
        store["A"]["ac"][i] = res_a["ac"]
        store["A"]["k"][i]  = res_a["k"]
        store["A"]["w"][i]  = res_a["w"]
        store["A"]["dcm"][i] = res_a["dcm"]
        for t_idx in range(T):
            store["A"]["ac_seg"][i][t_idx] = res_a["ac_seg"][t_idx]

        # ── Option B ──────────────────────────────────────────────────────────
        res_b = simulate_path(
            T=T, M=M_B, N_domestic=N_domestic,
            p=p_b, q=q_b, C=c_b,
            arpu=arpu_B, cm=cm_B,
            kappa=k_i,
            capex=capex, opex=opex, discount_rate=discount_rate,
            mode="B",
        )
        enpv_b = discounted_npv(res_b["w"], capex, discount_rate)
        store["B"]["enpv"].append(enpv_b)
        store["B"]["npv"].append(enpv_b)
        store["B"]["ac"][i] = res_b["ac"]
        store["B"]["k"][i]  = res_b["k"]
        store["B"]["w"][i]  = res_b["w"]
        store["B"]["dcm"][i] = res_b["dcm"]
        for t_idx in range(T):
            store["B"]["ac_seg"][i][t_idx] = res_b["ac_seg"][t_idx]

        # ── Option C (Switch) ─────────────────────────────────────────────────
        res_c = simulate_path(
            T=T, M=M_B, N_domestic=N_domestic,
            p=p_b, q=q_b, C=c_b,
            arpu=arpu_B, cm=cm_B,
            kappa=k_i,
            capex=capex, opex=opex, discount_rate=discount_rate,
            mode="C",
            switch_trigger=switch_trigger,
            grandfathering=grandfathering,
            churn_shock=churn_shock,
            p_A=p_a, q_A=q_a, C_A=c_a,
            M_A=M_A,
            arpu_A=arpu_A, cm_A=cm_A,
        )
        enpv_c  = discounted_npv(res_c["w"], capex, discount_rate)
        # Static NPV = deterministic B path; option value = ENPV_C - NPV_B
        store["C"]["enpv"].append(enpv_c)
        store["C"]["npv"].append(enpv_b)   # static baseline is Option B
        store["C"]["ac"][i] = res_c["ac"]
        store["C"]["k"][i]  = res_c["k"]
        store["C"]["w"][i]  = res_c["w"]
        store["C"]["dcm"][i] = res_c["dcm"]
        store["C"]["switch_times"].append(res_c["switch_time"])
        for t_idx in range(T):
            store["C"]["ac_seg"][i][t_idx] = res_c["ac_seg"][t_idx]

    # ── Aggregate statistics ─────────────────────────────────────────────────
    result = {}
    for key in ("A", "B", "C"):
        s = store[key]
        result[key] = {
            "enpv_series": np.array(s["enpv"]),
            "npv_series":  np.array(s["npv"]),
            "ac_mean":     np.mean(s["ac"], axis=0),
            "ac_p5":       np.percentile(s["ac"], 5, axis=0),
            "ac_p95":      np.percentile(s["ac"], 95, axis=0),
            "k_mean":      np.mean(s["k"], axis=0),
            "w_mean":      np.mean(s["w"], axis=0),
            "w_p5":        np.percentile(s["w"], 5, axis=0),
            "w_p95":       np.percentile(s["w"], 95, axis=0),
            "dcm_mean":    np.mean(s["dcm"], axis=0),
            "cum_npv_mean": cumulative_npv(np.mean(s["w"], axis=0), capex, discount_rate),
            "switch_times": s["switch_times"],
            # median segment breakdown per year
            "ac_seg_median": [
                np.median(
                    [s["ac_seg"][i][t] for i in range(n_sims)],
                    axis=0,
                )
                for t in range(T)
            ],
        }

    return result


def compute_scenario_stats(results: dict) -> dict:
    stats = {}
    for key in ("A", "B", "C"):
        enpv = results[key]["enpv_series"]
        npv  = results[key]["npv_series"]
        stats[key] = {
            "enpv_mean":     float(np.mean(enpv)),
            "enpv_std":      float(np.std(enpv)),
            "enpv_p5":       float(np.percentile(enpv, 5)),
            "enpv_p95":      float(np.percentile(enpv, 95)),
            "npv_mean":      float(np.mean(npv)),
            "prob_positive": float(np.mean(enpv > 0)),
        }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY / TORNADO
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_tornado(base_params: dict, n_sims: int = 400, delta: float = 0.20) -> list:
    """
    For each key parameter, compute ΔENPV_C when shifted ±delta (20%) from base.
    Returns list sorted by total range (largest impact first).
    """
    base_res  = run_monte_carlo(n_sims=n_sims, **base_params)
    base_enpv = float(np.mean(base_res["C"]["enpv_series"]))

    sensitivity_params = [
        ("p_B", "p_B (Innovation)"),
        ("q_B", "q_B (Imitation)"),
        ("C_B", "C_B (Churn)"),
        ("kappa", "κ (Cannibalization)"),
        ("discount_rate", "r (Discount Rate)"),
        ("M_B", "M_B (Market Potential)"),
    ]

    tornado = []
    for param_key, label in sensitivity_params:
        base_val = base_params.get(param_key)
        if base_val is None or base_val == 0:
            continue

        results_lo, results_hi = {}, {}

        for sign, store in [(-delta, results_lo), (+delta, results_hi)]:
            p_mod = dict(base_params)
            p_mod[param_key] = base_val * (1 + sign)
            # Clip churn to valid range
            if param_key in ("C_A", "C_B"):
                p_mod[param_key] = float(np.clip(p_mod[param_key], 0.01, 0.85))
            res = run_monte_carlo(n_sims=n_sims, **p_mod)
            enpv_val = float(np.mean(res["C"]["enpv_series"]))
            store["enpv"] = enpv_val

        delta_lo = results_lo["enpv"] - base_enpv
        delta_hi = results_hi["enpv"] - base_enpv
        tornado.append({
            "param":    label,
            "delta_lo": delta_lo,
            "delta_hi": delta_hi,
            "range":    abs(delta_hi - delta_lo),
        })

    tornado.sort(key=lambda x: x["range"], reverse=True)
    return tornado
