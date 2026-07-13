# 15 · K-Means Clustering — Batch Reactor Operating Mode Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/KMeans_Runaway_Risk/blob/main/15_KMeans_Runaway_Risk.ipynb)

> *"Unsupervised learning doesn't find what you're looking for. It finds what's actually there."*

---

## 🎯 The Problem Nobody Sees Coming

The process historian recorded everything.

Temperature: 96°C — below the 105°C alarm threshold. Pressure: 3.8 bar — below the 5.0 bar cutoff. Cooling flow: 52 L/min — low, but not zero. No alarm fires. No operator intervenes. And thirty minutes later, the batch is a write-off — or worse.

This is the paradox of threshold-based process control in exothermic batch reactors: **each individual sensor can look acceptable while the combination is already telling you something catastrophic is forming.** A cooling flow that would be fine at 80°C becomes critically insufficient at 96°C. A temperature that seems manageable with 90 L/min of cooling is a runaway risk at 50 L/min. The danger doesn't live in any single variable. It lives in the *relationship* between them — in the multivariate pattern that no single alarm is designed to detect.

K-Means clustering operates exactly where rule-based systems fail. It doesn't evaluate sensors one at a time. It sees all eight process variables simultaneously, as a point in an eight-dimensional space, and asks: **which neighborhood does this batch belong to?**

This project applies K-Means to **1,872 batch production records** from a chemical plant reactor historian spanning three production campaigns. No labels. No predefined rules. No assumptions about what constitutes a "bad" batch. The algorithm is given raw process data and asked to find the natural groupings — the distinct operating regimes that the reactor itself generates. What it finds are four modes: one safe, one inefficient, one symptomatic of a failing cooling system, and one that demands immediate intervention.

> ✅ **This project is completely free.** Full dataset and simulator included.
> If this helped you, check out the rest of the portfolio at
> [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com).

---

## 📊 Dataset

**File:** `reactor_batch.csv` — 1,872 batch production records from a chemical plant process historian spanning three production campaigns.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `temp_max_c` | float | °C | Peak temperature reached during the batch |
| `pressure_max_bar` | float | bar | Maximum pressure recorded per batch |
| `agitation_rpm` | float | rpm | Stirrer / agitator speed during reaction |
| `conc_a_initial_pct` | float | % | Initial concentration of reactant A |
| `conc_b_initial_pct` | float | % | Initial concentration of reactant B |
| `cooling_flow_l_min` | float | L/min | Cooling jacket flow rate |
| `reaction_time_min` | float | min | Total batch duration |
| `final_conversion_pct` | float | % | Reactant conversion at batch end |
| `cluster_true` | int | — | Ground-truth operating mode (held out — validation only) |

> ⚠️ `cluster_true` is **not used during training.** K-Means operates in fully unsupervised mode — the algorithm sees only the 8 process variables.

**Data Origin — Real-World Source Systems:**

| Feature | Source System | Instrument |
|---------|--------------|------------|
| `temp_max_c` | DCS (Distributed Control System) | Type-K thermocouple in reactor jacket |
| `pressure_max_bar` | DCS | Piezoelectric pressure transmitter |
| `agitation_rpm` | VFD feedback loop | Agitator drive tachometer |
| `conc_a_initial_pct`, `conc_b_initial_pct` | MES / Batch recipe server | Pre-batch dosing records |
| `cooling_flow_l_min` | DCS | Magnetic flow meter on cooling line |
| `reaction_time_min` | MES / Batch execution system | Start/stop timestamps |
| `final_conversion_pct` | PAT / LIMS | In-line NIR or GC end-of-batch analysis |

**Key EDA Findings:**

- Temperature and pressure are strongly correlated (r ≈ +0.81): higher temperature drives pressure buildup — the classical exothermic signature. This is the pair most associated with runaway risk.
- Cooling flow correlates negatively with temperature (r ≈ −0.79): insufficient cooling allows heat accumulation. When this variable collapses, the others accelerate.
- Reaction time is shorter when temperature is higher — consistent with Arrhenius kinetics. Batches in the runaway cluster average 60 minutes vs. 105 minutes for the slow reaction cluster.

---

## 🤖 Model

### Why K-Means, not a classifier?

The instinct when a process engineer sees this problem is to write rules: "If temp > 100°C AND cooling < 55 L/min, flag for intervention." That instinct is wrong — not because rules are useless, but because **the rules are derived from the clusters, not the other way around.** Before you can write a good rule, you need to know what the operating modes actually look like in your data.

K-Means answers the prior question: how many distinct behavioral regimes does this reactor actually exhibit, and what are their multivariate signatures? The answer to that question — discovered from data without labels — is what enables the rule writer, the process engineer, and the alarm system designer to work from reality rather than intuition.

The choice of K-Means specifically comes from three properties that match this domain: it is deterministic with fixed initialization (critical for regulatory traceability in chemical manufacturing), it produces hard assignments that map cleanly to operational modes, and its centroids in physical units are directly interpretable by process engineers who already think in terms of temperature setpoints and flow rates.

### Preprocessing

K-Means minimizes Euclidean distances in the feature space. With variables spanning different magnitudes — °C, bar, rpm, L/min — the algorithm would implicitly weight high-magnitude variables. `StandardScaler` brings all eight features to zero mean and unit variance before clustering.

There is no train/test split in this pipeline. The full dataset is used to discover cluster structure. Quality is assessed through internal validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz), a clusterability test (Hopkins statistic), a stability analysis across 10 random seeds, and external validation against the held-out ground truth.

### Choosing k = 4

The Elbow method and Silhouette score were computed for k = 2 to 9. Both agreed at k = 4 — the elbow breaks sharply and Silhouette peaks at 0.4396 before declining. The alignment of two independent criteria confirms the structure is real.

---

## 📈 Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | 0.3331 | Moderate — operating modes are separable but with some boundary overlap |
| **Calinski-Harabasz Index** | 1140.80 | High — strong inter-cluster separation despite boundary noise |
| **Davies-Bouldin Index** | 1.1712 | Near 1.0 — clusters are compact with limited overlap at the mode boundaries |
| **Hopkins Statistic** | 0.6657 | > 0.50 — confirmed non-random cluster structure (data is not uniformly distributed) |
| **Stability (Silhouette std)** | 0.0000 | Perfectly deterministic across 10 random seeds — K-Means converges to the same solution |
| **ARI vs Ground Truth** | 0.9020 | Strong — K-Means recovered the 4 operating modes with ~10% boundary misassignment |
| **NMI vs Ground Truth** | 0.8813 | High mutual information between discovered clusters and expert ground truth |

ARI = 0.9020 reflects a realistic unsupervised result on multi-campaign industrial data. The algorithm recovered the four operating modes with high fidelity, but boundary batches — particularly those transitioning between the poor heat transfer and aggressive reaction regimes — were assigned to the wrong cluster in approximately 10% of cases. This is not a model failure; it is an accurate description of what happens when gradual process drift erodes the gap between operating modes. The four-mode structure is real and actionable. The 10% boundary zone is where operator judgment and engineering limits must supplement statistical assignment.

---

## 🔍 Cluster Profiles (Operating Modes)

| Cluster | Mode | Batches | Temp | Pressure | Cooling | Rxn Time | Conversion |
|---------|------|---------|------|----------|---------|----------|------------|
| C0 | Slow Reaction / Low Yield | 451 (24.1%) | 69°C | 2.01 bar | 99 L/min | 106 min | 85% |
| C1 | Poor Heat Transfer | 398 (21.3%) | 91°C | 3.45 bar | 45 L/min | 84 min | 90% |
| C2 | Normal Operation | 621 (33.2%) | 80°C | 2.51 bar | 89 L/min | 81 min | 95% |
| C3 | Aggressive / Runaway Risk | 402 (21.5%) | 99°C | 4.38 bar | 61 L/min | 62 min | 98% |

**C0 — Slow Reaction / Low Yield:** The reactor is under-driven. Temperature is too low (68°C), the cooling system is working overtime (101 L/min), and reactant conversion is stuck at 85%. These batches are wasting time and energy without producing the yield the recipe was designed for.

**C1 — Poor Heat Transfer:** Temperature has climbed to 92°C but the cooling jacket is delivering only 44 L/min — less than half of what C2 runs normally. This is the silent failure mode: no individual alarm fires, but the heat balance is degrading. Fouled heat exchanger surfaces or a restricted cooling line will eventually push this cluster toward C3.

**C2 — Normal Operation (reference mode):** 80°C, 2.48 bar, 90 L/min cooling, 80-minute run, 95% conversion. This is the target. Any batch classification workflow should use this cluster as the SPC baseline.

**C3 — Aggressive / Runaway Risk:** Temperature at 100°C, pressure at 4.43 bar, agitation at 381 rpm, and cooling reduced to 60 L/min. The Arrhenius effect is visible: these batches finish 25 minutes faster than normal precisely because the reaction is accelerating. High conversion (98%) makes this cluster look productive — it isn't. It is a process that has exceeded its safe operating envelope.

---

## 🗂️ Repository Structure

```
KMeans_Runaway_Risk/
├── 15_KMeans_Runaway_Risk.ipynb   # Full educational notebook
├── reactor_batch.csv              # Complete 1,872-record dataset
├── app.py                         # Streamlit batch classifier simulator
├── requirements.txt
└── README.md
```

> ✅ **This project is completely free.** Full dataset and simulator included.
> If this helped you, check out the rest of the portfolio at
> [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab (no installation):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/KMeans_Runaway_Risk/blob/main/15_KMeans_Runaway_Risk.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/LozanoLsa/KMeans_Runaway_Risk.git
cd KMeans_Runaway_Risk
pip install -r requirements.txt
jupyter notebook 15_KMeans_Runaway_Risk.ipynb
```

**Option 3 — Streamlit Simulator:**

```bash
streamlit run app.py
```

---

## 💡 Key Learnings

1. **The danger lives in the combination, not the reading.** Each individual sensor in the runaway cluster sits below its alarm threshold. It is only in the eight-dimensional process space that C3 separates cleanly from C2. This is the fundamental reason multivariate clustering adds value over threshold-based control.

2. **Hopkins > 0.75 means the data is asking to be clustered.** Not all industrial datasets have genuine cluster structure. Uniform or random data will produce spurious clusters regardless of what the Elbow method suggests. Always test clusterability before committing to K-Means.

3. **Elbow and Silhouette should agree.** When both independent criteria point to the same k, the cluster count is grounded in the data structure — not an artifact of the method. If they disagree, investigate whether the dataset truly has that many modes or whether the signal is weaker than it appears.

4. **ARI = 0.9020 is the honest number.** The ~10% of boundary misassignments are not random errors — they are batches where two operating modes overlap in process space, typically during transition periods between campaigns. The four-cluster structure is real. The boundary zone is real too. Both are operationally meaningful.

5. **The centroid table is the deliverable, not the cluster assignment.** A cluster label means nothing to a process engineer. What matters is the profile — 92°C, 44 L/min cooling, 85 min — because that is the language of the DCS screen, the maintenance ticket, and the corrective action. The model's value is in making the centroid interpretable, not in the mathematics that produced it.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
