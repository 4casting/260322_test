# Switch Option Analysis — Bass Diffusion Dynamics
## Master Thesis: Martin Bischof, FHV Vorarlberg University of Applied Sciences

---

### Struktur

```
market_entry_model/
├── app.py           # Streamlit Dashboard (UI + Plots)
├── model.py         # Kern-Modell: Bass Diffusion, Monte Carlo, Bellman-Logik
├── requirements.txt
└── README.md
```

---

### Installation & Start

```bash
cd market_entry_model
pip install -r requirements.txt
streamlit run app.py
```

---

### Modell-Übersicht

#### Strategische Layer (Real Option Architecture)
- **Option A** — Standard Conditions (Differenzierungsstrategie): niedriges p/q, hohe Margen, geringer Churn
- **Option B** — Fighter Strategy (Penetrationsstrategie): hohes p/q, niedrige Margen, hoher Churn + Kannibalisierung
- **Option C** — Switch Option: Startet mit B, wechselt auf A sobald Bellman-Bedingung erfüllt

#### Kunden-Layer (Synthesized Bass Model)
```
N(t) = N(t-1)·(1−C) + (p + q·(1−C)·N(t-1)/M) · (M − A(t-1))
```
Erweiterungen:
- Churn-Term `(1−C)` als Retention-Komponente
- Produkt-Innovations-Zyklen (Cohort i)
- "Burnt Earth": Verbrauchtes Marktpotenzial irreversibel

#### Finanz-Layer (Net Value Contribution)
```
W(t) = Σ_j [AC_j(t)·CM_j − K_j(t)·ΔCM_j] − OpEx
ENPV = −CapEx + Σ_t W(t)/(1+r)^t
```
- ABC-Kundensegmentierung (j=A,B,C)
- Kannibalisierung beschränkt auf Heimatmarkt-Größe

#### Switch Option (Bellman-Logik)
```
F(t) = max[W_{B→A}(t) − S_{B,A},  W_B(t) + 1/(1+r)·E[F(t+1)]]
```
- Trigger: Wachstumsrate < Schwellenwert
- C1 (Hard Switch): Alle Kunden auf neuen Preis → Churn-Schock
- C2 (Grandfathering): Bestandskunden behalten alte Konditionen

#### Monte Carlo Simulation
- Stochastische Parameterziehung (Log-Normal mit σ)
- N Simulationspfade (default 1.500)
- ENPV = E[Σ diskontierte Cashflows]

---

### Parameter-Benchmarks (Sultan, Farley & Lehmann 1990)
| Parameter | Benchmark | Option A | Option B |
|-----------|-----------|----------|----------|
| p (Innovation) | ~0.03 | niedrig | hoch |
| q (Imitation) | ~0.38 | moderat | sehr hoch |
| C (Churn) | ~15% | niedrig | hoch |

---

### Literatur
- Bass, F.M. (1969). A new product growth model for consumer durables.
- Trigeorgis, L. (1996). Real Options: Managerial Flexibility and Strategy.
- Dixit & Pindyck (1994). Investment under Uncertainty.
- Kulatilaka & Trigeorgis (1994). The general flexibility to switch.
- Homburg et al. (2005). Do satisfied customers really pay more?
- Wolk & Ebling (2010). Multi-channel price differentiation.
