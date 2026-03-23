"""
model.py — Switch Option Analysis (exakte Bellman-Implementierung)
Masterarbeit: Martin Bischof, FHV Vorarlberg University of Applied Sciences

Bellman Switch-Bedingung (exakt):
  F(T)   = max(W_{B->A}(T) - S_{B,A},  W_B(T))
  F(t)   = max(W_{B->A}(t) - S_{B,A},  W_B(t) + F(t+1)/(1+r))

W_{B->A}(t) = Cashflow wenn in t geswitcht wird:
  = Bestandskunden(t) * CM^switch + Neue-A-Kunden(t) * CM^A
  - K(t) * DeltaCM^switch - OpEx

ENPV_Switch = -CapEx + sum_{t<t*} W_B/(1+r)^t
                     + sum_{t>=t*} W_{B->A}/(1+r)^t
                     - S_{B,A}/(1+r)^{t*}
"""

import numpy as np

SEGMENT_LABELS = ["A", "B", "C"]
SEGMENT_SHARES = np.array([0.20, 0.30, 0.50])
ARPU_DEFAULTS  = np.array([100_000, 15_000, 2_000])
CM_DEFAULTS    = np.array([25_000, 4_500, 800])


# ---------------------------------------------------------------------------
# BASS DIFFUSION HELPERS
# ---------------------------------------------------------------------------

def bass_new_adopters(AC_prev, cum_adopters, M, N_domestic, p, q):
    available = max(0.0, M - cum_adopters)
    if available <= 0:
        return 0.0
    new_acq = (p + q * (AC_prev + N_domestic) / max(M + N_domestic, 1)) * available
    return max(0.0, min(new_acq, available))


def active_stock_from_cohorts(cohorts, t):
    return float(np.sum(cohorts[1:t + 1]))


def churn_cohorts(cohorts, C, t):
    for tau in range(1, t):
        cohorts[tau] *= (1.0 - C)


# ---------------------------------------------------------------------------
# BASIS-PFAD: Simuliert einen vollstaendigen A- oder B-Pfad
# ---------------------------------------------------------------------------

def simulate_base_path(T, M, N_domestic, p, q, C, cm, cm_domestic, kappa, opex):
    """W(t) = sum_j [AC_j * CM_j - K_j * DeltaCM_j] - OpEx"""
    cohorts      = np.zeros(T + 1)
    cum_adopters = 0.0
    K_active     = 0.0
    delta_cm     = np.maximum(0.0, cm_domestic - cm)

    ac_arr  = np.zeros(T)
    k_arr   = np.zeros(T)
    w_arr   = np.zeros(T)
    dcm_arr = np.zeros(T)
    ac_segs = []

    for t in range(1, T + 1):
        churn_cohorts(cohorts, C, t)
        AC_prev = active_stock_from_cohorts(cohorts, t - 1) if t > 1 else 0.0
        new_acq = bass_new_adopters(AC_prev, cum_adopters, M, N_domestic, p, q)
        cohorts[t]    = new_acq
        cum_adopters += new_acq
        AC_t = active_stock_from_cohorts(cohorts, t)

        K_active = K_active * (1.0 - C) + kappa * AC_t
        K_actual = min(K_active, N_domestic)

        ac_seg   = AC_t * SEGMENT_SHARES
        k_seg    = K_actual * SEGMENT_SHARES
        W_t      = float(np.dot(ac_seg, cm)) - float(np.dot(k_seg, delta_cm)) - opex

        ac_arr[t - 1]  = AC_t
        k_arr[t - 1]   = K_actual
        w_arr[t - 1]   = W_t
        dcm_arr[t - 1] = float(np.dot(k_seg, delta_cm))
        ac_segs.append(ac_seg.copy())

    return {"ac": ac_arr, "k": k_arr, "w": w_arr, "dcm": dcm_arr, "ac_seg": ac_segs}


# ---------------------------------------------------------------------------
# W_{B->A}(t) BERECHNUNG
# Simuliert den vollstaendigen Pfad AB Switch-Zeitpunkt t_sw vorwaerts
# und gibt den Cashflow in t_sw zurueck (= "instantaner Post-Switch-Wert")
# PLUS den diskontierten Wert aller Folgeperioden ab t_sw+1.
# ---------------------------------------------------------------------------

def compute_W_BA_full(
    t_sw, T, r,
    # B-Kohorte zum Switch-Zeitpunkt (State)
    cohorts_B_at_sw, K_active_at_sw, C_B,
    # A-Parameter fuer neue Kunden
    M_A, N_domestic, p_A, q_A, C_A,
    cm_A, cm_switch,   # cm_switch = cm_A (Hard) oder cm_B (Grandfathering)
    cm_domestic, kappa,
    churn_shock, grandfathering,
    opex,
):
    """
    Berechnet den diskontierten Barwert aller Cashflows W_{B->A}(tau) fuer tau >= t_sw,
    wenn der Switch in t_sw ausgefuehrt wird.
    Wird fuer die Bellman-Bewertung pro Switch-Zeitpunkt benoetigt.
    """
    delta_cm_switch = np.maximum(0.0, cm_domestic - cm_switch)

    # Kohorte 1: B-Bestand nach Churn-Schock (Hard Switch)
    coh_B = cohorts_B_at_sw.copy()
    if not grandfathering:
        for tau in range(1, t_sw + 1):
            coh_B[tau] *= (1.0 - churn_shock)

    # Kohorte 2: neue A-Kunden ab t_sw
    coh_A2  = np.zeros(T + 1)
    cum_A2  = 0.0
    K_act   = K_active_at_sw

    W_sum = 0.0
    for t in range(t_sw, T + 1):
        # B-Bestand weiter abschmelzen
        churn_cohorts(coh_B, C_B, t)
        AC_B1 = active_stock_from_cohorts(coh_B, t)

        # A-Neuakquisitionen
        churn_cohorts(coh_A2, C_A, t)
        AC_A2_prev = active_stock_from_cohorts(coh_A2, t - 1) if t > 1 else 0.0
        new_A = bass_new_adopters(AC_A2_prev, cum_A2, M_A, N_domestic, p_A, q_A)
        coh_A2[t]  = new_A
        cum_A2    += new_A
        AC_A2 = active_stock_from_cohorts(coh_A2, t)

        AC_total = AC_B1 + AC_A2
        K_act    = K_act * (1.0 - C_B) + kappa * AC_total
        K_actual = min(K_act, N_domestic)

        ac1_seg = AC_B1  * SEGMENT_SHARES
        ac2_seg = AC_A2  * SEGMENT_SHARES
        k_seg   = K_actual * SEGMENT_SHARES

        # W_{B->A}(t) = sum[AC1*CM^switch + AC2*CM^A - K*DeltaCM^switch] - OpEx
        W_t = (float(np.dot(ac1_seg, cm_switch))
             + float(np.dot(ac2_seg, cm_A))
             - float(np.dot(k_seg, delta_cm_switch))
             - opex)

        disc   = 1.0 / (1.0 + r) ** (t - t_sw + 1)   # relativ zu t_sw diskontiert
        W_sum += W_t * disc

    return W_sum


# ---------------------------------------------------------------------------
# BELLMAN RUECKWAERTSREKURSION
# F(T)   = max(V_BA(T) - S,  W_B(T))
# F(t)   = max(V_BA(t) - S,  W_B(t) + F(t+1)/(1+r))
# wobei V_BA(t) = NPV aller Cashflows wenn Switch in t ausgefuehrt wird
# ---------------------------------------------------------------------------

def bellman_backward(V_BA, W_B, S_BA, r, T):
    """
    V_BA[t-1] = diskontierter Wert aller Post-Switch-Cashflows ab t (relativ zu t)
    W_B[t-1]  = Cashflow Option B in Periode t

    Bellman:
      F(T) = max(V_BA(T) - S, W_B(T))
      F(t) = max(V_BA(t) - S, W_B(t) + F(t+1)/(1+r))
    """
    F      = np.zeros(T)
    switch = np.zeros(T, dtype=bool)

    F[T - 1]      = max(V_BA[T - 1] - S_BA, W_B[T - 1])
    switch[T - 1] = (V_BA[T - 1] - S_BA) >= W_B[T - 1]

    for t in range(T - 2, -1, -1):
        sv = V_BA[t] - S_BA
        cv = W_B[t] + F[t + 1] / (1.0 + r)
        F[t]      = max(sv, cv)
        switch[t] = sv >= cv

    return F, switch


# ---------------------------------------------------------------------------
# SWITCH-PFAD SIMULATION
# ---------------------------------------------------------------------------

def simulate_switch_path(
    T, M_B, N_domestic, p_B, q_B, C_B, cm_B, cm_domestic, kappa,
    M_A, p_A, q_A, C_A, cm_A,
    S_BA_hard, S_BA_soft, grandfathering, churn_shock,
    opex, discount_rate,
):
    S_BA      = S_BA_soft if grandfathering else S_BA_hard
    cm_switch = cm_B if grandfathering else cm_A   # C2: Bestandskunden behalten cm_B

    # -- Schritt 1: Vollstaendiger B-Pfad + State-Snapshots --------------------
    cohorts_B    = np.zeros(T + 1)
    cum_adop_B   = 0.0
    K_active_B   = 0.0

    W_B        = np.zeros(T)
    ac_B       = np.zeros(T)
    k_B        = np.zeros(T)
    # State-Snapshots fuer Bellman
    coh_snapshots = []   # cohorts_B nach Periode t
    K_snapshots   = []   # K_active_B nach Periode t

    delta_cm_B = np.maximum(0.0, cm_domestic - cm_B)

    for t in range(1, T + 1):
        churn_cohorts(cohorts_B, C_B, t)
        AC_prev = active_stock_from_cohorts(cohorts_B, t - 1) if t > 1 else 0.0
        new_B   = bass_new_adopters(AC_prev, cum_adop_B, M_B, N_domestic, p_B, q_B)
        cohorts_B[t]  = new_B
        cum_adop_B   += new_B
        AC_t = active_stock_from_cohorts(cohorts_B, t)
        K_active_B = K_active_B * (1.0 - C_B) + kappa * AC_t
        K_actual   = min(K_active_B, N_domestic)

        ac_seg = AC_t * SEGMENT_SHARES
        k_seg  = K_actual * SEGMENT_SHARES
        W_B[t - 1] = (float(np.dot(ac_seg, cm_B))
                    - float(np.dot(k_seg, delta_cm_B))
                    - opex)
        ac_B[t - 1] = AC_t
        k_B[t - 1]  = K_actual
        coh_snapshots.append(cohorts_B.copy())
        K_snapshots.append(K_active_B)

    # -- Schritt 2: V_BA(t) fuer jeden potenziellen Switch-Zeitpunkt ----------
    # V_BA(t) = NPV aller W_{B->A}(tau) fuer tau >= t, diskontiert auf t
    V_BA = np.zeros(T)
    for t_sw in range(1, T + 1):
        V_BA[t_sw - 1] = compute_W_BA_full(
            t_sw=t_sw, T=T, r=discount_rate,
            cohorts_B_at_sw=coh_snapshots[t_sw - 1],
            K_active_at_sw=K_snapshots[t_sw - 1],
            C_B=C_B,
            M_A=M_A, N_domestic=N_domestic, p_A=p_A, q_A=q_A, C_A=C_A,
            cm_A=cm_A, cm_switch=cm_switch,
            cm_domestic=cm_domestic, kappa=kappa,
            churn_shock=churn_shock, grandfathering=grandfathering,
            opex=opex,
        )

    # -- Schritt 3: Bellman-Rueckwaertsrekursion -> t_bellman ------------------
    F, switch_opt = bellman_backward(V_BA, W_B, S_BA, discount_rate, T)
    switch_indices = np.where(switch_opt)[0]
    t_bellman = int(switch_indices[0]) + 1 if len(switch_indices) > 0 else None

    # -- Schritt 4: Tatsaechlicher Pfad vorwaerts ------------------------------
    coh_B2    = np.zeros(T + 1)
    cum_B2    = 0.0
    K_act2    = 0.0
    coh_A2    = np.zeros(T + 1)
    cum_A2    = 0.0
    switched  = False
    delta_cm_switch = np.maximum(0.0, cm_domestic - cm_switch)

    ac_arr  = np.zeros(T)
    k_arr   = np.zeros(T)
    w_arr   = np.zeros(T)
    dcm_arr = np.zeros(T)
    sc_arr  = np.zeros(T)
    ac_segs = []

    for t in range(1, T + 1):
        if not switched:
            churn_cohorts(coh_B2, C_B, t)
            AC_prev = active_stock_from_cohorts(coh_B2, t - 1) if t > 1 else 0.0
            new_B = bass_new_adopters(AC_prev, cum_B2, M_B, N_domestic, p_B, q_B)
            coh_B2[t]  = new_B
            cum_B2    += new_B
            AC_t = active_stock_from_cohorts(coh_B2, t)
            K_act2 = K_act2 * (1.0 - C_B) + kappa * AC_t
            K_actual = min(K_act2, N_domestic)

            if t_bellman is not None and t == t_bellman:
                switched = True
                if not grandfathering:
                    for tau in range(1, t + 1):
                        coh_B2[tau] *= (1.0 - churn_shock)
                    AC_t = active_stock_from_cohorts(coh_B2, t)
                    sc_arr[t - 1] = S_BA_hard
                else:
                    sc_arr[t - 1] = S_BA_soft

            ac_seg = AC_t * SEGMENT_SHARES
            k_seg  = K_actual * SEGMENT_SHARES
            W_t = (float(np.dot(ac_seg, cm_B))
                 - float(np.dot(k_seg, delta_cm_B))
                 - opex - sc_arr[t - 1])
            dcm_t = float(np.dot(k_seg, delta_cm_B))
            ac_arr[t - 1]  = AC_t
            k_arr[t - 1]   = K_actual
            w_arr[t - 1]   = W_t
            dcm_arr[t - 1] = dcm_t
            ac_segs.append(ac_seg.copy())

        else:
            churn_cohorts(coh_B2, C_B, t)
            AC_B1 = active_stock_from_cohorts(coh_B2, t)

            churn_cohorts(coh_A2, C_A, t)
            AC_A2_prev = active_stock_from_cohorts(coh_A2, t - 1) if t > 1 else 0.0
            new_A = bass_new_adopters(AC_A2_prev, cum_A2, M_A, N_domestic, p_A, q_A)
            coh_A2[t]  = new_A
            cum_A2    += new_A
            AC_A2 = active_stock_from_cohorts(coh_A2, t)

            AC_total = AC_B1 + AC_A2
            K_act2 = K_act2 * (1.0 - C_B) + kappa * AC_total
            K_actual = min(K_act2, N_domestic)

            ac1_seg = AC_B1  * SEGMENT_SHARES
            ac2_seg = AC_A2  * SEGMENT_SHARES
            k_seg   = K_actual * SEGMENT_SHARES

            W_t = (float(np.dot(ac1_seg, cm_switch))
                 + float(np.dot(ac2_seg, cm_A))
                 - float(np.dot(k_seg, delta_cm_switch))
                 - opex)
            dcm_t = float(np.dot(k_seg, delta_cm_switch))

            ac_arr[t - 1]  = AC_B1 + AC_A2
            k_arr[t - 1]   = K_actual
            w_arr[t - 1]   = W_t
            dcm_arr[t - 1] = dcm_t
            ac_segs.append(((AC_B1 + AC_A2) * SEGMENT_SHARES).copy())

    return {
        "ac": ac_arr, "k": k_arr, "w": w_arr, "dcm": dcm_arr,
        "switch_time": t_bellman,
        "W_B": W_B, "V_BA": V_BA, "F": F, "switch_opt": switch_opt,
        "ac_seg": ac_segs,
    }


# ---------------------------------------------------------------------------
# DISCOUNTED NPV
# ---------------------------------------------------------------------------

def discounted_npv(w_arr, capex, r):
    disc = np.array([1.0 / (1.0 + r) ** t for t in range(1, len(w_arr) + 1)])
    return -capex + float(np.dot(w_arr, disc))


def switch_enpv(W_B, V_BA, S_BA, t_bellman, capex, r, T):
    """
    ENPV_Switch = -CapEx + sum_{t<t*} W_B(t)/(1+r)^t
                          + V_BA(t*)/(1+r)^{t*-1}  [bereits intern diskontiert]
                          - S_{B,A}/(1+r)^{t*}
    """
    disc = np.array([1.0 / (1.0 + r) ** t for t in range(1, T + 1)])
    if t_bellman is None:
        return -capex + float(np.dot(W_B, disc))
    t_idx    = t_bellman - 1
    # Pre-Switch: W_B diskontiert
    npv_pre  = float(np.dot(W_B[:t_idx], disc[:t_idx]))
    # Post-Switch: V_BA(t*) ist bereits der NPV aller Post-Switch-CFs relativ zu t*
    # -> noch mit disc[t_idx] auf t=0 diskontieren
    npv_post = V_BA[t_idx] * disc[t_idx]
    # Switching Cost in t*
    npv_cost = S_BA * disc[t_idx]
    return -capex + npv_pre + npv_post - npv_cost


def cumulative_npv(w_arr, capex, r):
    disc = np.array([1.0 / (1.0 + r) ** t for t in range(1, len(w_arr) + 1)])
    return np.cumsum(w_arr * disc) - capex


# ---------------------------------------------------------------------------
# MONTE CARLO ENGINE
# ---------------------------------------------------------------------------

def run_monte_carlo(
    n_sims, T,
    M_A, M_B, N_domestic,
    p_A, q_A, C_A,
    p_B, q_B, C_B,
    sigma,
    arpu_A, arpu_B, cm_A, cm_B,
    kappa, capex, opex, discount_rate,
    switch_trigger=0.02,
    grandfathering=True,
    churn_shock=0.35,
):
    cm_A        = np.array(cm_A, dtype=float)
    cm_B        = np.array(cm_B, dtype=float)
    cm_domestic = cm_A.copy()

    store = {k: {
        "enpv": [], "npv": [],
        "ac":   np.zeros((n_sims, T)),
        "k":    np.zeros((n_sims, T)),
        "w":    np.zeros((n_sims, T)),
        "dcm":  np.zeros((n_sims, T)),
        "switch_times": [],
        "ac_seg": [[None] * T for _ in range(n_sims)],
        "W_B_paths": [], "V_BA_paths": [], "F_paths": [],
    } for k in ("A", "B", "C")}

    rng = np.random.default_rng(42)

    for i in range(n_sims):
        def draw(mu, s=sigma):
            return float(np.clip(rng.lognormal(np.log(max(mu, 1e-9)), s),
                                 mu * 0.1, mu * 5.0))

        p_a = draw(p_A); q_a = draw(q_A); c_a = float(np.clip(draw(C_A), 0.01, 0.6))
        p_b = draw(p_B); q_b = draw(q_B); c_b = float(np.clip(draw(C_B), 0.01, 0.7))
        k_i = float(np.clip(draw(kappa) if kappa > 1e-9 else 0.0, 0.0, 0.4))

        # Option A
        res_a = simulate_base_path(T, M_A, N_domestic, p_a, q_a, c_a,
                                    cm_A, cm_domestic, 0.0, opex)
        enpv_a = discounted_npv(res_a["w"], capex, discount_rate)
        store["A"]["enpv"].append(enpv_a)
        store["A"]["npv"].append(enpv_a)
        store["A"]["ac"][i]  = res_a["ac"]
        store["A"]["k"][i]   = res_a["k"]
        store["A"]["w"][i]   = res_a["w"]
        store["A"]["dcm"][i] = res_a["dcm"]
        for t_idx in range(T):
            store["A"]["ac_seg"][i][t_idx] = res_a["ac_seg"][t_idx]

        # Option B
        res_b = simulate_base_path(T, M_B, N_domestic, p_b, q_b, c_b,
                                    cm_B, cm_domestic, k_i, opex)
        enpv_b = discounted_npv(res_b["w"], capex, discount_rate)
        store["B"]["enpv"].append(enpv_b)
        store["B"]["npv"].append(enpv_b)
        store["B"]["ac"][i]  = res_b["ac"]
        store["B"]["k"][i]   = res_b["k"]
        store["B"]["w"][i]   = res_b["w"]
        store["B"]["dcm"][i] = res_b["dcm"]
        for t_idx in range(T):
            store["B"]["ac_seg"][i][t_idx] = res_b["ac_seg"][t_idx]

        # Option C — Switch Option mit exakter Bellman-Rueckwaertsrekursion
        mean_cm_B = float(np.dot(SEGMENT_SHARES, cm_B))
        AC_B_avg  = float(np.mean(res_b["ac"]))
        S_BA_hard = churn_shock * AC_B_avg * mean_cm_B
        # switch_trigger = Anteil des Jahres-CM als Grandfathering-Kosten (default ~2%)
        S_BA_soft = abs(switch_trigger) * AC_B_avg * mean_cm_B

        res_c = simulate_switch_path(
            T=T, M_B=M_B, N_domestic=N_domestic,
            p_B=p_b, q_B=q_b, C_B=c_b,
            cm_B=cm_B, cm_domestic=cm_domestic, kappa=k_i,
            M_A=M_A, p_A=p_a, q_A=q_a, C_A=c_a, cm_A=cm_A,
            S_BA_hard=S_BA_hard, S_BA_soft=S_BA_soft,
            grandfathering=grandfathering, churn_shock=churn_shock,
            opex=opex, discount_rate=discount_rate,
        )

        S_BA   = S_BA_soft if grandfathering else S_BA_hard
        enpv_c = switch_enpv(res_c["W_B"], res_c["V_BA"], S_BA,
                              res_c["switch_time"], capex, discount_rate, T)

        store["C"]["enpv"].append(enpv_c)
        store["C"]["npv"].append(enpv_b)
        store["C"]["ac"][i]  = res_c["ac"]
        store["C"]["k"][i]   = res_c["k"]
        store["C"]["w"][i]   = res_c["w"]
        store["C"]["dcm"][i] = res_c["dcm"]
        store["C"]["switch_times"].append(res_c["switch_time"])
        store["C"]["W_B_paths"].append(res_c["W_B"])
        store["C"]["V_BA_paths"].append(res_c["V_BA"])
        store["C"]["F_paths"].append(res_c["F"])
        for t_idx in range(T):
            store["C"]["ac_seg"][i][t_idx] = res_c["ac_seg"][t_idx]

    result = {}
    for key in ("A", "B", "C"):
        s = store[key]
        result[key] = {
            "enpv_series":   np.array(s["enpv"]),
            "npv_series":    np.array(s["npv"]),
            "ac_mean":       np.mean(s["ac"], axis=0),
            "ac_p5":         np.percentile(s["ac"], 5,  axis=0),
            "ac_p95":        np.percentile(s["ac"], 95, axis=0),
            "k_mean":        np.mean(s["k"], axis=0),
            "w_mean":        np.mean(s["w"], axis=0),
            "w_p5":          np.percentile(s["w"], 5,  axis=0),
            "w_p95":         np.percentile(s["w"], 95, axis=0),
            "dcm_mean":      np.mean(s["dcm"], axis=0),
            "cum_npv_mean":  cumulative_npv(np.mean(s["w"], axis=0), capex, discount_rate),
            "switch_times":  s["switch_times"],
            "ac_seg_median": [
                np.median([s["ac_seg"][i][t] for i in range(n_sims)], axis=0)
                for t in range(T)
            ],
        }
        if key == "C" and s["W_B_paths"]:
            result[key]["W_B_mean"]  = np.mean(s["W_B_paths"],  axis=0)
            result[key]["V_BA_mean"] = np.mean(s["V_BA_paths"], axis=0)
            result[key]["F_mean"]    = np.mean(s["F_paths"],     axis=0)
    return result


def compute_scenario_stats(results):
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


def sensitivity_tornado(base_params, n_sims=400, delta=0.20):
    base_res  = run_monte_carlo(n_sims=n_sims, **base_params)
    base_enpv = float(np.mean(base_res["C"]["enpv_series"]))
    sensitivity_params = [
        ("p_B",           "p_B (Innovation)"),
        ("q_B",           "q_B (Imitation)"),
        ("C_B",           "C_B (Churn)"),
        ("kappa",         "kappa (Kannibalisierung)"),
        ("discount_rate", "r (Diskontierungszins)"),
        ("M_B",           "M_B (Marktpotenzial)"),
        ("churn_shock",   "Churn-Schock (C1)"),
    ]
    tornado = []
    for param_key, label in sensitivity_params:
        base_val = base_params.get(param_key)
        if base_val is None or abs(base_val) < 1e-9:
            continue
        res_lo, res_hi = {}, {}
        for sign, rd in [(-delta, res_lo), (+delta, res_hi)]:
            p_mod = dict(base_params)
            p_mod[param_key] = base_val * (1.0 + sign)
            for ck, lo, hi in [("C_A", 0.01, 0.85), ("C_B", 0.01, 0.85),
                                ("churn_shock", 0.0, 0.9), ("kappa", 0.0, 0.5)]:
                if param_key == ck:
                    p_mod[param_key] = float(np.clip(p_mod[param_key], lo, hi))
            res = run_monte_carlo(n_sims=n_sims, **p_mod)
            rd["enpv"] = float(np.mean(res["C"]["enpv_series"]))
        tornado.append({
            "param":    label,
            "delta_lo": res_lo["enpv"] - base_enpv,
            "delta_hi": res_hi["enpv"] - base_enpv,
            "range":    abs(res_hi["enpv"] - res_lo["enpv"]),
        })
    tornado.sort(key=lambda x: x["range"], reverse=True)
    return tornado
