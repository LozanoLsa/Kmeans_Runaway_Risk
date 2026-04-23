"""
app.py — Reactor Batch Clustering Dashboard
LozanoLsa · Project 15 · K-Means Clustering · 2026 · FREE PROJECT

Algorithm: K-Means (k=4) + PCA + StandardScaler
Domain: Chemical Processing — Batch Operating Mode Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K-Means · Reactor Batch Clustering",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── FULL CSS INJECTION ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #080c12;
    --surface:  #0e1420;
    --card:     #121922;
    --card2:    #161f2e;
    --border:   #1e2d45;
    --accent:   #f97316;
    --accent2:  #fb923c;
    --danger:   #f87171;
    --warn:     #fbbf24;
    --ok:       #4ade80;
    --blue:     #60a5fa;
    --purp:     #a78bfa;
    --text:     #c8d8f0;
    --muted:    #4e6a8a;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 1.8rem 2.4rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; border: 2px solid var(--accent2) !important; box-shadow: 0 0 8px rgba(249,115,22,0.5) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important; font-size: 0.65rem !important; color: var(--accent2) !important; background: var(--card) !important; border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

[data-testid="stSelectbox"] > div > div { background: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--fm) !important; font-size: 0.78rem !important; border-radius: 3px !important; }

[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-top: 2px solid var(--accent) !important; padding: 1rem 1.1rem 0.9rem !important; border-radius: 3px !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.18em !important; color: var(--muted) !important; font-weight: 400 !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important; font-size: 1.7rem !important; font-weight: 600 !important; color: var(--accent2) !important; line-height: 1.1 !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; gap: 0 !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; color: var(--muted) !important; padding: 0.5rem 1.2rem !important; border: none !important; border-radius: 0 !important; background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--accent2) !important; background: rgba(249,115,22,0.06) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }

[data-testid="stAlert"] { border-radius: 2px !important; font-family: var(--fm) !important; font-size: 0.75rem !important; border: none !important; }
[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; background: var(--card2) !important; color: var(--muted) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; background: var(--card) !important; }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.08em !important; }
h1, h2, h3 { font-family: var(--fh) !important; color: var(--text) !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

.lsa-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 0.2rem; }
.lsa-project-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 4px; }
.lsa-title { font-family: var(--fh); font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.lsa-tagline { font-family: var(--fs); font-style: italic; font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
.lsa-chip { display: inline-block; background: rgba(249,115,22,0.1); border: 1px solid rgba(249,115,22,0.3); color: var(--accent2); font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-chip-free { display: inline-block; background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3); color: #4ade80; font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH     = "reactor_batch.csv"
DATA_PATH_ALT = "15_KMeans_Runaway_Risk/reactor_batch.csv"
RANDOM_STATE  = 42
N_CLUSTERS    = 4

FEATURES = [
    "temp_max_c", "pressure_max_bar", "agitation_rpm",
    "conc_a_initial_pct", "conc_b_initial_pct",
    "cooling_flow_l_min", "reaction_time_min", "final_conversion_pct",
]
FEAT_LABELS = {
    "temp_max_c":           "Peak Temperature (°C)",
    "pressure_max_bar":     "Max Pressure (bar)",
    "agitation_rpm":        "Agitation Speed (rpm)",
    "conc_a_initial_pct":   "Reactant A Concentration (%)",
    "conc_b_initial_pct":   "Reactant B Concentration (%)",
    "cooling_flow_l_min":   "Cooling Flow (L/min)",
    "reaction_time_min":    "Reaction Time (min)",
    "final_conversion_pct": "Final Conversion (%)",
}
CLUSTER_NAMES = {
    0: "Slow Reaction / Low Yield",
    1: "Poor Heat Transfer",
    2: "Normal Operation",
    3: "Aggressive / Runaway Risk",
}
CLUSTER_ICONS = {0: "⚠️", 1: "🔴", 2: "✅", 3: "🚨"}

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
C_BG     = "#080c12"
C_CARD   = "#121922"
C_ORANGE = "#f97316"
C_ORANGE2= "#fb923c"
C_DANGER = "#f87171"
C_OK     = "#4ade80"
C_BLUE   = "#60a5fa"
C_PURP   = "#a78bfa"
C_WARN   = "#fbbf24"
C_TEXT   = "#c8d8f0"
C_MUTED  = "#4e6a8a"

# One distinct color per cluster
CLUSTER_COLORS = {0: C_WARN, 1: C_PURP, 2: C_OK, 3: C_DANGER}

def dark_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)
    ax.title.set_color(C_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2d45")
    return fig, ax

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for path in [DATA_PATH, DATA_PATH_ALT]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    st.error("reactor_batch.csv not found. Place the file in the same folder as app.py.")
    st.stop()

@st.cache_resource
def fit_model(df):
    X = df[FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans   = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    labels   = kmeans.fit_predict(X_scaled)
    pca      = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca    = pca.fit_transform(X_scaled)
    return scaler, kmeans, labels, X_scaled, X_pca, pca

df = load_data()
scaler, kmeans, labels, X_scaled, X_pca, pca = fit_model(df)
df_vis = df.copy()
df_vis["cluster"] = labels
df_vis["cluster_name"] = [CLUSTER_NAMES[c] for c in labels]

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #15 · K-Means Clustering · Batch Reactor · Unsupervised</div>
    <div class="lsa-title">Four Operating Modes — No Labels Required</div>
    <div class="lsa-tagline">K-Means finds the natural boundaries between normal, slow, heat-starved, and runaway batches from raw process data.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">K-MEANS  k=4</span>
        <span class="lsa-chip">8 FEATURES</span>
        <span class="lsa-chip">PCA 2D</span>
        <span class="lsa-chip">SILHOUETTE · CH · DB · ARI · NMI</span>
        <span class="lsa-chip-free">FREE PROJECT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
sil = silhouette_score(X_scaled, labels)
ch  = calinski_harabasz_score(X_scaled, labels)
db  = davies_bouldin_score(X_scaled, labels)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Batches",   f"{len(df):,}",    "Production records")
k2.metric("Silhouette",      f"{sil:.4f}",      "Cluster separation quality")
k3.metric("Calinski-Harabasz", f"{ch:.0f}",     "Higher = better defined")
k4.metric("Davies-Bouldin",  f"{db:.4f}",       "Lower = better separation")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "ELBOW & SILHOUETTE", "BATCH CLASSIFIER", "CLUSTER PROFILES", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Dataset overview — batch reactor production records</div>',
                unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown('<div class="lsa-section">// Raw dataset sample</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=320)
    with col_right:
        st.markdown('<div class="lsa-section">// Descriptive statistics</div>', unsafe_allow_html=True)
        desc = df[FEATURES].describe().T[["mean", "std", "min", "max"]].round(2)
        st.dataframe(desc, use_container_width=True, height=320)

    st.divider()
    st.markdown('<div class="lsa-section">// Feature correlation heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    corr = df[FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
        center=0, vmin=-1, vmax=1, linewidths=0.3, linecolor="#1e2d45",
        annot_kws={"size": 9, "color": C_TEXT}, ax=ax,
        xticklabels=[FEAT_LABELS.get(f, f) for f in FEATURES],
        yticklabels=[FEAT_LABELS.get(f, f) for f in FEATURES],
    )
    ax.set_xticklabels(ax.get_xticklabels(), color=C_TEXT, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), color=C_TEXT, fontsize=8, rotation=0)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("Strong positive: temp ↔ pressure (0.88) · Strong negative: temp ↔ cooling flow (−0.75).")

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Optimal cluster count — k = 4</div>',
                unsafe_allow_html=True)
    st.caption("Elbow method and Silhouette analysis both converge on k = 4 — four distinct operating modes.")

    K_RANGE   = range(2, 10)
    inertias, sil_scores = [], []
    for k in K_RANGE:
        km_k  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        lab_k = km_k.fit_predict(X_scaled)
        inertias.append(km_k.inertia_)
        sil_scores.append(silhouette_score(X_scaled, lab_k))

    ks = list(K_RANGE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(C_BG)
    for ax in [ax1, ax2]:
        ax.set_facecolor(C_CARD)
        ax.tick_params(colors=C_MUTED)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")

    ax1.plot(ks, inertias, "o-", color=C_ORANGE, lw=2.5, ms=8)
    ax1.axvline(x=4, color=C_DANGER, ls="--", lw=1.8, label="Optimal k = 4")
    ax1.set_xlabel("Number of Clusters (k)", color=C_MUTED)
    ax1.set_ylabel("Inertia (WSS)", color=C_MUTED)
    ax1.set_title("Elbow Curve", color=C_TEXT, fontweight="bold")
    ax1.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.15, color="#1e2d45")
    ax1.xaxis.label.set_color(C_MUTED); ax1.yaxis.label.set_color(C_MUTED)

    bar_colors = [C_DANGER if k == 4 else C_ORANGE for k in ks]
    bars = ax2.bar(ks, sil_scores, color=bar_colors, alpha=0.85, edgecolor="#1e2d45", lw=0.5)
    for bar, s in zip(bars, sil_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{s:.3f}", ha="center", va="bottom", fontsize=9, color=C_TEXT)
    ax2.set_xlabel("Number of Clusters (k)", color=C_MUTED)
    ax2.set_ylabel("Silhouette Score", color=C_MUTED)
    ax2.set_title("Silhouette Score vs. k", color=C_TEXT, fontweight="bold")
    ax2.set_ylim(0, 0.58); ax2.set_xticks(ks)
    ax2.grid(axis="y", alpha=0.15, color="#1e2d45")
    ax2.xaxis.label.set_color(C_MUTED); ax2.yaxis.label.set_color(C_MUTED)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// PCA 2D projection — cluster assignments</div>',
                unsafe_allow_html=True)
    fig2, ax = dark_fig(10, 5.5)
    for c in range(N_CLUSTERS):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=CLUSTER_COLORS[c], label=f"C{c}: {CLUSTER_NAMES[c]}",
                   alpha=0.45, s=16, edgecolors="none")
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c="white", s=200, zorder=6, edgecolors="black", lw=2, marker="D")
    for c, (cx, cy) in enumerate(centroids_pca):
        ax.annotate(f"C{c}", (cx, cy), fontsize=9, fontweight="bold",
                    ha="center", va="center", color="black")
    var_exp = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
    ax.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)
    ax.grid(True, alpha=0.1, color="#1e2d45")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True); plt.close()
    st.caption("Diamond markers = cluster centroids. Four operating modes form well-separated regions in PCA space.")

    st.divider()
    st.markdown('<div class="lsa-section">// Validation metrics summary</div>', unsafe_allow_html=True)
    try:
        ari = adjusted_rand_score(df["cluster_true"], labels)
        nmi = normalized_mutual_info_score(df["cluster_true"], labels)
    except KeyError:
        ari = nmi = float("nan")

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, name, val, cap in zip(
        [m1, m2, m3, m4, m5],
        ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "ARI", "NMI"],
        [f"{sil:.4f}", f"{ch:.0f}", f"{db:.4f}",
         f"{ari:.4f}" if not np.isnan(ari) else "N/A",
         f"{nmi:.4f}" if not np.isnan(nmi) else "N/A"],
        ["0.25–0.65 = good", "Higher = better", "< 1 is good", "Perfect = 1.0", "Perfect = 1.0"],
    ):
        col.metric(name, val, cap)
    st.caption("ARI = Adjusted Rand Index · NMI = Normalized Mutual Information · compared against ground truth labels.")

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="lsa-section">// Real-time batch classifier</div>', unsafe_allow_html=True)
    st.caption("Enter process parameters recorded at the end of a batch run. K-Means assigns it to one of four operating modes.")

    # ── Result on LEFT, controls on RIGHT ─────────────────────────────────────
    col_out, col_inp = st.columns([3, 2])

    with col_inp:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid var(--accent);border-radius:2px;
                    padding:1rem 1.2rem;margin-bottom:14px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Process parameters</div>
        </div>
        """, unsafe_allow_html=True)
        temp    = st.slider("Peak Temperature (°C)",        55.0, 120.0,  80.0, 0.5)
        pres    = st.slider("Max Pressure (bar)",             1.0,   8.0,   2.5, 0.1)
        agit    = st.slider("Agitation Speed (rpm)",        100.0, 500.0, 300.0, 5.0)
        conc_a  = st.slider("Reactant A Concentration (%)",  20.0,  60.0,  35.0, 0.5)
        conc_b  = st.slider("Reactant B Concentration (%)",  20.0,  60.0,  35.0, 0.5)
        cool    = st.slider("Cooling Flow (L/min)",          20.0, 130.0,  90.0, 1.0)
        time_rx = st.slider("Reaction Time (min)",           40.0, 150.0,  80.0, 1.0)
        conv    = st.slider("Final Conversion (%)",          70.0, 100.0,  95.0, 0.5)

    # Predict
    row   = np.array([[temp, pres, agit, conc_a, conc_b, cool, time_rx, conv]])
    X_row = scaler.transform(row)
    pred  = int(kmeans.predict(X_row)[0])
    color = CLUSTER_COLORS[pred]
    icon  = CLUSTER_ICONS[pred]
    name  = CLUSTER_NAMES[pred]
    dists = np.linalg.norm(kmeans.cluster_centers_ - X_row, axis=1)
    probs = 1 / (dists + 1e-9); probs = probs / probs.sum()

    with col_out:
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-left:3px solid {color};border-radius:2px;
                        padding:1.6rem 1.8rem;margin-bottom:16px;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:0.6rem;">Classification Result</div>
                <div style="font-size:2rem;margin-bottom:4px;">{icon}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;
                            font-weight:700;color:{color};line-height:1.2;">
                    Cluster {pred}<br>
                    <span style="font-size:0.95rem;">{name}</span>
                </div>
                <div style="font-family:var(--fm);font-size:0.68rem;color:var(--muted);
                            margin-top:10px;">
                    Nearest centroid distance: {dists[pred]:.3f} (scaled units)
                </div>
            </div>''',
            unsafe_allow_html=True
        )

        st.markdown('<div class="lsa-section">// Confidence by cluster</div>', unsafe_allow_html=True)
        for c in range(N_CLUSTERS):
            bar_w   = int(probs[c] * 100)
            bar_col = CLUSTER_COLORS[c]
            bold    = "font-weight:700;" if c == pred else ""
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;
                            font-family:var(--fm);font-size:0.68rem;color:var(--text);
                            {bold}margin-bottom:3px;">
                    <span>C{c}: {CLUSTER_NAMES[c]}</span>
                    <span>{probs[c]*100:.1f}%</span>
                </div>
                <div style="background:#1e2d45;border-radius:3px;height:7px;">
                    <div style="background:{bar_col};width:{bar_w}%;
                                height:7px;border-radius:3px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="lsa-section" style="margin-top:14px;">// Recommended actions</div>',
                    unsafe_allow_html=True)
        PLAYBOOK = {
            0: ["Verify reactant concentrations and dosing sequence.",
                "Check reactor heating system — temperature below setpoint.",
                "Reduce cooling flow if heat generation is insufficient.",
                "Inspect agitator — low mixing slows reaction kinetics."],
            1: ["Inspect cooling jacket for fouling or flow restriction.",
                "Verify cooling water supply pressure and temperature.",
                "Reduce reactant dosing rate if jacket is underperforming.",
                "Schedule heat exchanger maintenance inspection."],
            2: ["All process variables within optimal range.",
                "Maintain current setpoints and continue monitoring.",
                "Use this batch as SPC reference baseline."],
            3: ["INCREASE cooling flow to maximum immediately.",
                "Reduce or halt reactant dosing to limit heat generation.",
                "Verify safety interlock and emergency vent status.",
                "Notify process engineer and safety officer.",
                "Initiate emergency protocol if temp exceeds 105°C."],
        }
        for i, action in enumerate(PLAYBOOK[pred], 1):
            prefix = "🚨 " if pred == 3 and i == 1 else f"{i}. "
            st.markdown(f"""
            <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                        padding:4px 0;border-bottom:1px solid #1e2d4530;line-height:1.6;">
                <span style="color:{color};font-weight:600;">{prefix}</span>{action}
            </div>""", unsafe_allow_html=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="lsa-section">// Cluster characterization — operating mode profiles</div>',
                unsafe_allow_html=True)
    st.caption("Each cluster = a distinct operating regime. Centroid table shows the characteristic process signature.")

    centroids_orig = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=FEATURES,
        index=[f"C{i}: {CLUSTER_NAMES[i]}" for i in range(N_CLUSTERS)]
    ).round(2)
    centroids_orig.columns = [FEAT_LABELS.get(f, f) for f in FEATURES]
    st.markdown('<div class="lsa-section">// Cluster centroids (original engineering units)</div>',
                unsafe_allow_html=True)
    st.dataframe(centroids_orig, use_container_width=True)
    st.caption("Each row = the mean process fingerprint of that operating mode.")

    st.divider()
    st.markdown('<div class="lsa-section">// Normalized centroid heatmap — relative feature intensity</div>',
                unsafe_allow_html=True)
    centroid_norm = (centroids_orig - centroids_orig.min()) / (centroids_orig.max() - centroids_orig.min())
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    sns.heatmap(
        centroid_norm, annot=True, fmt=".2f",
        cmap=sns.color_palette("RdYlGn_r", as_cmap=True),
        linewidths=0.5, linecolor="#1e2d45", vmin=0, vmax=1,
        annot_kws={"size": 10}, cbar_kws={"label": "Normalized [0–1]"}, ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), color=C_TEXT, fontsize=9, rotation=20, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), color=C_TEXT, fontsize=9, rotation=0)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("0 = lowest value across clusters · 1 = highest. Runaway Risk cluster (C3) dominates temp, pressure, and conversion.")

    st.divider()
    st.markdown('<div class="lsa-section">// Radar fingerprint — operating mode profiles</div>',
                unsafe_allow_html=True)
    radar_cols   = ["temp_max_c", "pressure_max_bar", "agitation_rpm",
                    "cooling_flow_l_min", "reaction_time_min", "final_conversion_pct"]
    radar_labels = ["Temp", "Pressure", "Agitation", "Cooling\nFlow", "Rxn\nTime", "Conversion"]
    N = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    centroid_radar    = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
    radar_norm_vals   = centroid_radar[radar_cols].copy()
    for col in radar_cols:
        lo, hi = radar_norm_vals[col].min(), radar_norm_vals[col].max()
        radar_norm_vals[col] = (radar_norm_vals[col] - lo) / (hi - lo)

    fig2, axes = plt.subplots(1, 4, figsize=(16, 4.5), subplot_kw=dict(polar=True))
    fig2.patch.set_facecolor(C_BG)
    for i, ax in enumerate(axes):
        ax.set_facecolor(C_CARD)
        vals  = radar_norm_vals.iloc[i][radar_cols].tolist() + [radar_norm_vals.iloc[i][radar_cols[0]]]
        color = CLUSTER_COLORS[i]
        ax.plot(angles, vals, color=color, lw=2.5)
        ax.fill(angles, vals, color=color, alpha=0.22)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, color=C_TEXT, fontsize=8.5)
        ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels([])
        ax.spines["polar"].set_color("#1e2d45")
        ax.grid(color="#1e2d45", alpha=0.4)
        ax.set_title(f"C{i}: {CLUSTER_NAMES[i]}", color=color,
                     fontsize=9, fontweight="bold", pad=14)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True); plt.close()
    st.caption("Each axis = relative position of that cluster on that feature (0 = min · 1 = max).")

    st.divider()
    st.markdown('<div class="lsa-section">// Cluster size distribution</div>', unsafe_allow_html=True)
    size_cols = st.columns(4)
    for c in range(N_CLUSTERS):
        n   = int((labels == c).sum())
        pct = n / len(labels) * 100
        size_cols[c].markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-top:3px solid {CLUSTER_COLORS[c]};border-radius:3px;
                    padding:1rem;text-align:center;">
            <div style="font-family:var(--fm);font-size:1.8rem;font-weight:700;
                        color:{CLUSTER_COLORS[c]};">{n}</div>
            <div style="font-family:var(--fm);font-size:0.62rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.1em;">{pct:.1f}% of batches</div>
            <div style="font-family:var(--fm);font-size:0.65rem;color:var(--text);
                        margin-top:4px;">C{c}: {CLUSTER_NAMES[c]}</div>
        </div>""", unsafe_allow_html=True)

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Operational playbook — response protocol by operating mode</div>',
                unsafe_allow_html=True)
    st.caption("Each K-Means cluster maps to a specific operational scenario with a defined corrective protocol.")

    action_plan = [
        {"cluster":2, "color":C_OK,     "name":"Normal Operation",
         "signature":"Temp ~80°C · Pressure ~2.5 bar · Cooling ~90 L/min · Conversion ~95%",
         "frequency":"35% of batches", "risk":"Low",
         "actions":["Maintain current process setpoints.",
                    "Use as SPC baseline for control chart limits.",
                    "Document as reference batch for process qualification."]},
        {"cluster":0, "color":C_WARN,   "name":"Slow Reaction / Low Yield",
         "signature":"Temp ~68°C · Pressure ~2.0 bar · Cooling ~101 L/min · Conversion ~85%",
         "frequency":"25% of batches", "risk":"Medium",
         "actions":["Verify reactant concentrations and feed dosing sequence.",
                    "Check reactor heating system — temperature below setpoint.",
                    "Reduce cooling flow to allow temperature to reach target.",
                    "Inspect agitator seal and impeller for mechanical issues."]},
        {"cluster":1, "color":C_PURP,   "name":"Poor Heat Transfer",
         "signature":"Temp ~92°C · Pressure ~3.5 bar · Cooling ~44 L/min · Conversion ~90%",
         "frequency":"20% of batches", "risk":"High",
         "actions":["Inspect cooling jacket for scale/fouling buildup.",
                    "Verify cooling water supply flow and inlet temperature.",
                    "Reduce reactant dosing rate while cooling is degraded.",
                    "Schedule predictive maintenance for heat exchanger surfaces.",
                    "Check control valve response on cooling loop."]},
        {"cluster":3, "color":C_DANGER, "name":"Aggressive / Runaway Risk",
         "signature":"Temp ~100°C · Pressure ~4.4 bar · Cooling ~60 L/min · Conversion ~98%",
         "frequency":"20% of batches", "risk":"Critical",
         "actions":["Increase cooling flow to maximum capacity immediately.",
                    "Reduce or halt reactant dosing to limit heat generation.",
                    "Verify safety interlock system and emergency vent status.",
                    "Initiate emergency cooling protocol if temp exceeds 105°C.",
                    "Notify process engineer and safety officer.",
                    "Log event in process safety management system."]},
    ]

    for mode in action_plan:
        with st.expander(f"Cluster {mode['cluster']} — {mode['name']}", expanded=True):
            left, right = st.columns([2, 1])
            with left:
                st.markdown(f"""
                <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                            margin-bottom:10px;line-height:1.6;">
                    <strong style="color:var(--text);">Process Signature:</strong>
                    {mode['signature']}
                </div>""", unsafe_allow_html=True)
                for i, act in enumerate(mode["actions"], 1):
                    prefix = "🚨 " if mode["cluster"] == 3 and i == 1 else f"{i}. "
                    st.markdown(f"""
                    <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                                padding:3px 0;border-bottom:1px solid #1e2d4530;line-height:1.6;">
                        <span style="color:{mode['color']};font-weight:600;">{prefix}</span>{act}
                    </div>""", unsafe_allow_html=True)
            with right:
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);
                            border-top:3px solid {mode['color']};border-radius:2px;
                            padding:1rem;text-align:center;">
                    <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                                text-transform:uppercase;letter-spacing:.1em;">Frequency</div>
                    <div style="font-family:var(--fm);font-size:1.2rem;font-weight:700;
                                color:{mode['color']};margin:4px 0;">{mode['frequency']}</div>
                    <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                                text-transform:uppercase;letter-spacing:.1em;margin-top:8px;">Risk Level</div>
                    <div style="font-family:var(--fm);font-size:1.1rem;font-weight:700;
                                color:{mode['color']};">{mode['risk']}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:2px;
                padding:1rem 1.3rem;text-align:center;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Free project</div>
        <div style="font-family:var(--fm);font-size:0.68rem;color:var(--muted);line-height:1.7;">
            Full dataset + simulator included. Check the rest of the portfolio at
            <span style="color:#f97316;">lozanolsa.gumroad.com</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Reactor Batch Clustering · Project 15 · v2.0
</div>
""", unsafe_allow_html=True)
