"""
=============================================================================
Valuing Market Entry Pricing Strategies:
A Switch Option Analysis based on Bass Diffusion Dynamics
=============================================================================
Author: Martin Bischof
Framework Implementation: Three-Layer Model
  1. Strategic Layer  – Real Option Architecture (Option A / B / C)
  2. Customer Layer   – Synthesized Bass Diffusion Model with Churn & Cohorts
  3. Financial Layer  – Net Value Contribution (W(t)), NPV, ENPV

References matched strictly to PDF Document "260328_Masterarbeit_V2.pdf"
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1.  DATA STRUCTURES & FINANCIAL PARAMETERS
# =============================================================================

@dataclass
class CustomerSegment:
    """ABC customer segmentation (Section 3.4.2)."""
    name: str          
    share: float       
    arpu: float        
    cm_domestic: float # Eq. 9: Contribution Margin in Domestic Market
    cm_entry: float    # Eq. 10: Contribution Margin in Entry Market

    @property
    def delta_cm(self) -> float:
        """Eq. 11: Structural Margin Erosion"""
        return self.cm_domestic - self.cm_entry

@dataclass
class DiffusionParams:
    """Parameters for the Synthesized Bass Diffusion Model."""
    label: str
    p: float          # Coefficient of Innovation
    q: float          # Coefficient of Imitation 
    churn_schedule: List[float]  # \delta_i(k): Time-varying churn
    m: int            # Market saturation limit = SOM 
    kappa: float      # Cannibalization rate 
    
    def churn(self, tenure: int, noise_multiplier: float = 1.0) -> float:
        """Return churn rate \delta_i(k) with optional MCS noise."""
        idx = min(tenure - 1, len(self.churn_schedule) - 1)
        return min(1.0, max(0.0, self.churn_schedule[idx] * noise_multiplier))

@dataclass
class ModelConfig:
    """Global simulation configuration."""
    T: int = 5                    # Horizon 2 
    n_simulations: int = 5000     
    discount_rate: float = 0.08   
    capex: float = 150_000.0      
    opex_annual: float = 60_000.0 
    n_domestic: int = 200         # N_domestic (Eq. 15 limit)
    sigma_p: float = 0.30         
    sigma_q: float = 0.20         
    sigma_churn: float = 0.15     
    sigma_kappa: float = 0.25     
    burnt_earth_shock: float = 0.35  # S_{B,A} (Eq. 19/21 phase transition)
    grandfathering: bool = False     # Scenario C2 logic
    switch_year: Optional[int] = None 

# =============================================================================
# 2.  SEGMENTS & PRESETS
# =============================================================================

DEFAULT_SEGMENTS: List[CustomerSegment] = [
    CustomerSegment("A", share=0.20, arpu=100_000, cm_domestic=25_000, cm_entry=20_000),
    CustomerSegment("B", share=0.30, arpu=15_000,  cm_domestic=4_500,  cm_entry=3_500),
    CustomerSegment("C", share=0.50, arpu=2_000,   cm_domestic=800,    cm_entry=600),
]

OPTION_A_PARAMS = DiffusionParams(
    label="Option A (Standard)", p=0.025, q=0.32,
    churn_schedule=[0.10, 0.08, 0.03, 0.03, 0.03], 
    m=745, kappa=0.02
)

OPTION_B_PARAMS = DiffusionParams(
    label="Option B (Fighter)", p=0.050, q=0.55,
    churn_schedule=[0.20, 0.18, 0.15, 0.15, 0.15], 
    m=994, kappa=0.08 
)

# =============================================================================
# 3.  SYNTHESIZED BASS DIFFUSION ENGINE (Eq. 3-8 & 13-15)
# =============================================================================

class BassEngine:
    def __init__(self, params: DiffusionParams, config: ModelConfig):
        self.params = params
        self.config = config

    def simulate(self, T: int, noise: Optional[Dict] = None) -> Dict:
        p_noise = 1 + (noise["p"] if noise else 0)
        q_noise = 1 + (noise["q"] if noise else 0)
        k_noise = 1 + (noise["kappa"] if noise else 0)
        c_noise = 1 + (noise["churn"] if noise else 0)

        p = max(0, self.params.p * p_noise)
        q = max(0, self.params.q * q_noise)
        kappa = max(0, self.params.kappa * k_noise)
        m = self.params.m
        n_dom_max = self.config.n_domestic

        acq_hist = []               
        AC_organic = 0.0            # N_i^{organic}(t) (Eq. 6)
        B_cumulative = 0.0          # B_i(t) (Eq. 8)
        K_cumul = 0.0               # N_{C,i,j}(t) (Eq. 14)
        K_actual = 0.0              # N_{c,i,j}^{actual}(t) (Eq. 15)

        results = {
            "t": [], "acquisitions": [], "AC_organic": [],
            "K_actual": [], "AC_total": []
        }

        for t in range(1, T + 1):
            AC_total_prev = AC_organic + K_actual

            # --- Organic Acquisitions (Eq. 7 & Eq. 3 Logic) ---
            retention_multiplier = (1.0 - self.params.churn(1, c_noise))
            rate = p + (q * retention_multiplier * (AC_total_prev / (m + n_dom_max)))
            available = max(0.0, m - AC_organic - B_cumulative)
            S_tau = rate * available
            acq_hist.append(S_tau)

            # --- Active Customers Cohort Survival (Eq. 6) ---
            AC_organic = 0.0
            for tau_idx, acq in enumerate(acq_hist):
                tau = tau_idx + 1
                tenure = t - tau + 1
                if tenure <= 0: continue
                
                survival = 1.0
                for k in range(1, tenure + 1):
                    survival *= (1.0 - self.params.churn(k, c_noise))
                AC_organic += acq * survival
            
            # --- Cannibalization (Eq. 13 - 15) ---
            delta_K = kappa * AC_total_prev if t > 1 else 0.0 # Eq. 13
            K_cumul = K_cumul * (1 - self.params.churn(1, c_noise)) + delta_K # Eq. 14
            K_actual = min(K_cumul, n_dom_max) # Eq. 15 

            results["t"].append(t)
            results["acquisitions"].append(S_tau)
            results["AC_organic"].append(AC_organic)
            results["K_actual"].append(K_actual)
            results["AC_total"].append(AC_organic + K_actual)

        return results

# =============================================================================
# 4.  FINANCIAL LAYER (Eq. 16)
# =============================================================================

def calculate_W_t(AC_organic: float, K_actual: float, segments: List[CustomerSegment], is_option_B: bool = False, is_grandfathered: bool = False) -> float:
    """Aggregated Net Value Contribution W(t) - Eq. 16"""
    W = 0.0
    for seg in segments:
        n_organic = AC_organic * seg.share
        n_cannib = K_actual * seg.share
        
        cm_digital = seg.cm_entry
        if is_option_B or is_grandfathered:
            cm_digital = seg.cm_entry * 0.60 # Fighter pricing logic
            
        W += (n_organic * cm_digital) - (n_cannib * seg.delta_cm)
    return W

def npv_static(cash_flows: List[float], capex: float, r: float) -> float:
    """Static NPV - Eq. 16 (Rappaport, 1986)."""
    return -capex + sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows, 1))

# =============================================================================
# 5.  SWITCH OPTION ENGINE (Eq. 17-21)
# =============================================================================

class SwitchOptionEngine:
    def __init__(self, config: ModelConfig, segments: List[CustomerSegment]):
        self.config = config
        self.segments = segments
        self.engine_B = BassEngine(OPTION_B_PARAMS, config)
        self.engine_A = BassEngine(OPTION_A_PARAMS, config)

    def generate_switch_cashflows(self, t_bellman: int, noise: Optional[Dict]) -> List[float]:
        """Generates the cash flow path W_{B->A}(t) given a specific switch year."""
        T = self.config.T
        res_B = self.engine_B.simulate(T, noise)
        res_A = self.engine_A.simulate(T, noise)
        opex = self.config.opex_annual
        gran = self.config.grandfathering
        c_noise = 1 + (noise["churn"] if noise else 0) if noise else 1.0

        W_switch = []
        AC1_organic = 0.0
        K_total = 0.0

        for t in range(1, T + 1):
            t_idx = t - 1
            if t < t_bellman:
                AC1_organic = res_B["AC_organic"][t_idx]
                K_total = res_B["K_actual"][t_idx]
                cf = calculate_W_t(AC1_organic, K_total, self.segments, is_option_B=True) - opex
                W_switch.append(cf)
            elif t == t_bellman:
                if not gran:
                    AC1_organic *= (1 - self.config.burnt_earth_shock) 
                
                K_total = min(K_total, self.config.n_domestic)
                cf = calculate_W_t(AC1_organic, K_total, self.segments, is_option_B=False, is_grandfathered=gran) - opex
                W_switch.append(cf)
            else:
                tenure_b = t - t_bellman + 1
                AC1_organic *= (1 - OPTION_B_PARAMS.churn(tenure_b, c_noise))
                
                idx_a = t - t_bellman - 1
                AC2_organic = res_A["AC_organic"][min(idx_a, T - 1)]
                
                cf_legacy = calculate_W_t(AC1_organic, K_total, self.segments, is_option_B=False, is_grandfathered=gran)
                cf_new = calculate_W_t(AC2_organic, 0, self.segments, is_option_B=False) 
                
                W_switch.append(cf_legacy + cf_new - opex)
                
        return W_switch

    def simulate_path(self, noise: Optional[Dict] = None) -> Dict:
        T = self.config.T
        r = self.config.discount_rate
        capex = self.config.capex
        opex = self.config.opex_annual

        res_A = self.engine_A.simulate(T, noise)
        res_B = self.engine_B.simulate(T, noise)
        
        W_A = [calculate_W_t(res_A["AC_organic"][t], res_A["K_actual"][t], self.segments, is_option_B=False) - opex for t in range(T)]
        W_B = [calculate_W_t(res_B["AC_organic"][t], res_B["K_actual"][t], self.segments, is_option_B=True) - opex for t in range(T)]

        t_bell = self.config.switch_year
        if t_bell is None:
            t_bell = T + 1 
            for t in range(1, T):
                cf_wait = self.generate_switch_cashflows(t + 1, noise)
                val_wait = W_B[t-1] + sum(cf_wait[s] / (1+r)**(s-t+1) for s in range(t, T))
                
                cf_now = self.generate_switch_cashflows(t, noise)
                val_now = sum(cf_now[s] / (1+r)**(s-t) for s in range(t-1, T))
                
                if val_now > val_wait:
                    t_bell = t
                    break

        if t_bell <= T:
            W_switch = self.generate_switch_cashflows(t_bell, noise)
        else:
            W_switch = W_B.copy()

        enpv = npv_static(W_switch, capex, r)
        npv_B = npv_static(W_B, capex, r)

        return {
            "t_bellman": t_bell if t_bell <= T else np.nan,
            "W_switch": W_switch,
            "W_A": W_A,
            "W_B": W_B,
            "ENPV_switch": enpv,
            "NPV_A": npv_static(W_A, capex, r),
            "NPV_B": npv_B,
            "option_value": max(0, enpv - npv_B),
            "AC_A_final": res_A["AC_organic"][-1],
            "AC_B_final": res_B["AC_organic"][-1],
            "K_B_final": res_B["K_actual"][-1]
        }

# =============================================================================
# 6.  MONTE CARLO SIMULATION ENGINE
# =============================================================================

class MonteCarloEngine:
    def __init__(self, config: ModelConfig, segments: List[CustomerSegment]):
        self.config = config
        self.segments = segments
        self.switch_engine = SwitchOptionEngine(config, segments)

    def _sample_noise(self, rng: np.random.Generator) -> Dict:
        return {
            "p":     rng.normal(0, self.config.sigma_p),
            "q":     rng.normal(0, self.config.sigma_q),
            "kappa": rng.normal(0, self.config.sigma_kappa),
            "churn": rng.normal(0, self.config.sigma_churn),
        }

    def run(self, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        N = self.config.n_simulations
        records = []
        
        for _ in range(N):
            noise = self._sample_noise(rng)
            path = self.switch_engine.simulate_path(noise=noise)
            
            records.append({
                "NPV_A": path["NPV_A"],
                "NPV_B": path["NPV_B"],
                "ENPV_C": path["ENPV_switch"],
                "option_value": path["option_value"],
                "t_bellman": path["t_bellman"],
                "AC_A_final": path["AC_A_final"],
                "AC_B_final": path["AC_B_final"],
                "K_B_final": path["K_B_final"],
            })
            
        return pd.DataFrame(records)

# =============================================================================
# 7.  RESULTS, VISUALIZATION & SENSITIVITY
# =============================================================================

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print(f"  MONTE CARLO RESULTS  –  N = {len(df)}")
    print("=" * 65)
    cols = ["NPV_A", "NPV_B", "ENPV_C", "option_value"]
    labels = {
        "NPV_A":        "NPV  Option A (Standard)",
        "NPV_B":        "NPV  Option B (Fighter)",
        "ENPV_C":       "ENPV Option C (Switch)",
        "option_value": "Switch Option Value",
    }
    for col in cols:
        s = df[col]
        print(f"\n  {labels[col]}")
        print(f"    Mean    : CHF {s.mean():>12,.0f}")
        print(f"    Median  : CHF {s.median():>12,.0f}")
        print(f"    Std Dev : CHF {s.std():>12,.0f}")
        print(f"    P5 / P95: CHF {s.quantile(.05):>10,.0f}  /  CHF {s.quantile(.95):>10,.0f}")
        print(f"    P(>0)   : {(s > 0).mean():.1%}")

    print("\n  Optimal Switch Year (t_bellman)")
    valid_bellman = df["t_bellman"].dropna()
    if not valid_bellman.empty:
        print(f"    Mode    : Year {int(valid_bellman.mode()[0])}")
        print(f"    Mean    : Year {valid_bellman.mean():.1f}")
    else:
        print("    No switches triggered.")
    print("=" * 65)

def sensitivity_analysis(config: ModelConfig, segments: List[CustomerSegment]) -> pd.DataFrame:
    print("\n  Running Sensitivity Analysis (OAT) …")
    base_cfg = ModelConfig(**config.__dict__)

    def run_single(cfg):
        mc = MonteCarloEngine(cfg, segments)
        return mc.run(seed=0)["ENPV_C"].mean()

    base_val = run_single(base_cfg)
    results = []

    perturbations = {
        "p (+30%)":   ("sigma_p",   0.60),
        "q (+30%)":   ("sigma_q",   0.50),
        "kappa (+50%)": ("sigma_kappa", 0.60),
        "churn (+30%)": ("sigma_churn", 0.50),
        "burnt_earth (+50%)": ("burnt_earth_shock", 0.525),
        "discount +2pp": ("discount_rate", 0.10),
    }

    for label, (attr, val) in perturbations.items():
        cfg_mod = ModelConfig(**config.__dict__)
        setattr(cfg_mod, attr, val)
        perturbed = run_single(cfg_mod)
        delta = perturbed - base_val
        results.append({
            "Parameter": label,
            "ENPV_C (CHF)": perturbed,
            "Delta vs Base (CHF)": delta,
        })

    df_sens = pd.DataFrame(results).sort_values("Delta vs Base (CHF)")
    return df_sens

def plot_results(df_mc: pd.DataFrame, deterministic: Dict, df_sens: pd.DataFrame, config: ModelConfig) -> plt.Figure:
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    COLORS = {"A": "#2563EB", "B": "#DC2626", "C": "#059669", "OV": "#7C3AED"}
    years = list(range(1, config.T + 1))

    # [0,0] Distributions
    ax0 = fig.add_subplot(gs[0, 0])
    for key, col, lbl in [("A", COLORS["A"], "Option A"), ("B", COLORS["B"], "Option B"), ("C", COLORS["C"], "Option C")]:
        data = df_mc[f"NPV_{key}"] if key != "C" else df_mc["ENPV_C"]
        ax0.hist(data / 1e6, bins=50, alpha=0.5, color=col, label=lbl, density=True)
    ax0.axvline(0, color="black", lw=1.0, ls=":")
    ax0.set_title("NPV/ENPV Distribution", fontsize=10, fontweight="bold")
    ax0.set_xlabel("CHF Millions", fontsize=8)
    ax0.legend(fontsize=7)

    # [0,1] Cash Flows
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(years, [cf / 1e3 for cf in deterministic["W_A"]], color=COLORS["A"], marker="o", label="Option A", lw=2)
    ax1.plot(years, [cf / 1e3 for cf in deterministic["W_B"]], color=COLORS["B"], marker="s", label="Option B", lw=2)
    ax1.plot(years, [cf / 1e3 for cf in deterministic["W_switch"]], color=COLORS["C"], marker="^", label="Option C", lw=2, ls="--")
    bell = deterministic.get("t_bellman", None)
    if pd.notna(bell) and bell <= config.T:
        ax1.axvline(bell, color=COLORS["OV"], ls=":", lw=1.5, label=f"t_bellman = {bell}")
    ax1.axhline(0, color="black", lw=0.8, ls=":")
    ax1.set_title("Deterministic Cash-Flow W(t)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Year", fontsize=8)
    ax1.set_ylabel("TCHF", fontsize=8)
    ax1.legend(fontsize=7)

    # [0,2] Option Value
    ax2 = fig.add_subplot(gs[0, 2])
    ov = df_mc["option_value"] / 1e3
    ax2.hist(ov, bins=50, color=COLORS["OV"], alpha=0.7, density=True)
    ax2.axvline(ov.mean(), color="black", lw=1.8, ls="--", label=f"Mean: {ov.mean():.0f} TCHF")
    ax2.set_title("Switch Option Value (ENPV_C − NPV_B)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("TCHF", fontsize=8)
    ax2.legend(fontsize=7)

    # [1,0] Customer Base
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(df_mc["AC_A_final"], bins=40, alpha=0.5, color=COLORS["A"], label="AC Option A", density=True)
    ax3.hist(df_mc["AC_B_final"], bins=40, alpha=0.5, color=COLORS["B"], label="AC Option B", density=True)
    ax3.set_title("Active Customer Base – Year 5", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Active Customers", fontsize=8)
    ax3.legend(fontsize=7)

    # [1,1] Switch Year
    ax4 = fig.add_subplot(gs[1, 1])
    bell_vals = df_mc["t_bellman"].dropna().value_counts().sort_index()
    if not bell_vals.empty:
        ax4.bar(bell_vals.index, bell_vals.values / len(df_mc) * 100, color=COLORS["OV"], alpha=0.8)
    ax4.set_title("Optimal Switch Year", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Year", fontsize=8)
    ax4.set_ylabel("Frequency (%)", fontsize=8)
    ax4.set_xticks(range(1, config.T + 2))

    # [1,2] Tornado Sensitivity
    ax5 = fig.add_subplot(gs[1, 2])
    df_s = df_sens.sort_values("Delta vs Base (CHF)")
    colors_bar = [COLORS["B"] if v < 0 else COLORS["A"] for v in df_s["Delta vs Base (CHF)"]]
    ax5.barh(df_s["Parameter"], df_s["Delta vs Base (CHF)"] / 1e3, color=colors_bar, alpha=0.8)
    ax5.axvline(0, color="black", lw=0.8)
    ax5.set_title("Sensitivity – ENPV_C vs Base", fontsize=10, fontweight="bold")
    ax5.set_xlabel("ΔENPV_C (TCHF)", fontsize=8)

    fig.suptitle("Market Entry Pricing Strategy Valuation (Bischof, 2026)", fontsize=12, fontweight="bold", y=1.01)
    return fig

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    config = ModelConfig(n_simulations=5000)
    
    print("\n" + "=" * 65)
    print("  SWITCH OPTION MODEL – BASS DIFFUSION VALUATION FRAMEWORK")
    print("  Bischof (2026) · FHV Vorarlberg University of Applied Sciences")
    print("=" * 65)

    print("\n  [1/3] Running Deterministic Baseline …")
    switch_eng = SwitchOptionEngine(config, DEFAULT_SEGMENTS)
    deterministic = switch_eng.simulate_path(noise=None)
    
    print(f"\n  [2/3] Running Monte Carlo Simulation (N={config.n_simulations}) …")
    mc_engine = MonteCarloEngine(config, DEFAULT_SEGMENTS)
    df_mc = mc_engine.run(seed=42)
    print_summary(df_mc)
    
    df_sens = sensitivity_analysis(config, DEFAULT_SEGMENTS)
    
    print("\n  [3/3] Generating Visualisation …")
    fig = plot_results(df_mc, deterministic, df_sens, config)
    fig.savefig("switch_option_results.png", dpi=150, bbox_inches="tight")
    print("        Saved → switch_option_results.png")
    print("\n  Done. Now get to writing Chapter 4.")
