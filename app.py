import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
import streamlit as st
from catboost import CatBoostClassifier
from scipy.stats import norm

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* ---- palette ---- */
    :root {
        --grad-green : #10b981;
        --drop-red   : #ef4444;
        --neutral    : #6b7280;
        --bg-card    : #f9fafb;
        --border     : #e5e7eb;
        --text-main  : #111827;
        --text-muted : #6b7280;
        --accent     : #4f46e5;
    }

    /* ---- base ---- */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }

    /* ---- hero header ---- */
    .hero {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        color: white;
    }
    .hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 .5rem 0; letter-spacing: -.5px; }
    .hero p  { font-size: 1rem; opacity: .8; margin: 0; }

    /* ---- section card ---- */
    .section-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.25rem;
    }
    .section-title {
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }

    /* ---- predict button ---- */
    div[data-testid="stFormSubmitButton"] button {
        background: #4f46e5;
        color: white;
        border-radius: 10px;
        padding: .65rem 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: background .2s;
    }
    div[data-testid="stFormSubmitButton"] button:hover { background: #4338ca; }

    /* ---- result gauge wrapper ---- */
    .gauge-row {
        display: flex;
        gap: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .gauge-card {
        flex: 1;
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        text-align: center;
    }
    .gauge-card.grad {
        background: #d1fae5;
        border: 2px solid #10b981;
    }
    .gauge-card.drop {
        background: #fee2e2;
        border: 2px solid #ef4444;
    }
    .gauge-pct {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: .25rem;
    }
    .gauge-card.grad .gauge-pct { color: #059669; }
    .gauge-card.drop .gauge-pct { color: #dc2626; }
    .gauge-label {
        font-size: .9rem;
        font-weight: 600;
        color: var(--text-main);
    }
    .gauge-sub {
        font-size: .78rem;
        color: var(--text-muted);
        margin-top: .2rem;
    }

    /* ---- verdict badge ---- */
    .verdict {
        border-radius: 10px;
        padding: .85rem 1.25rem;
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .verdict.safe    { background: #d1fae5; color: #065f46; border: 1.5px solid #10b981; }
    .verdict.warning { background: #fef3c7; color: #92400e; border: 1.5px solid #f59e0b; }
    .verdict.danger  { background: #fee2e2; color: #991b1b; border: 1.5px solid #ef4444; }

    /* ---- factor table ---- */
    .factor-row {
        display: flex;
        align-items: center;
        gap: .75rem;
        padding: .55rem 0;
        border-bottom: 1px solid var(--border);
    }
    .factor-rank {
        font-size: .75rem;
        font-weight: 700;
        color: var(--text-muted);
        width: 1.4rem;
        text-align: center;
        flex-shrink: 0;
    }
    .factor-name {
        font-size: .88rem;
        font-weight: 500;
        color: var(--text-main);
        flex: 1;
    }
    .factor-bar-wrap {
        width: 120px;
        height: 8px;
        background: #e5e7eb;
        border-radius: 99px;
        overflow: hidden;
        flex-shrink: 0;
    }
    .factor-bar-fill {
        height: 100%;
        border-radius: 99px;
    }
    .factor-pct {
        font-size: .82rem;
        font-weight: 700;
        width: 3.5rem;
        text-align: right;
        flex-shrink: 0;
    }
    .factor-direction {
        font-size: .75rem;
        width: 5.5rem;
        text-align: right;
        flex-shrink: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading model …")
def load_assets():
    model = CatBoostClassifier()
    model.load_model("artifacts/catboost_m4_model.cbm")

    context_db   = pd.read_csv("artifacts/major_context_db.csv")
    encoder_maps = json.load(open("artifacts/encoder_maps.json"))
    m4_feat_cols = json.load(open("artifacts/m4_feature_cols.json"))
    adm_stats    = json.load(open("artifacts/admission_score_stats.json"))

    return model, context_db, encoder_maps, m4_feat_cols, adm_stats


model, context_db, encoder_maps, M4_FEATURE_COLS, admission_score_stats = load_assets()

ADMISSION_SCORE_CONFIG: dict[str, dict] = {
    "National High School Graduation Exam": {
        "min_val": 0.0,   "max_val": 30.0,   "default": 24.0,  "step": 0.01,
        "label_suffix": "(0 – 30)",
    },
    "Priority Admission": {
        "min_val": 0.0,   "max_val": 30.0,   "default": 26.0,  "step": 0.01,
        "label_suffix": "(0 – 30)",
    },
    "Combined Transcript & National Exam Admission": {
        "min_val": 0.0,   "max_val": 30.0,   "default": 25.0,  "step": 0.01,
        "label_suffix": "(0 – 30)",
    },
    "Direct Admission": {
        "min_val": 0.0,   "max_val": 30.0,   "default": 27.0,  "step": 0.01,
        "label_suffix": "(0 – 30)",
    },
    "Special Admission Consideration in 2021": {
        "min_val": 0.0,   "max_val": 30.0,   "default": 25.0,  "step": 0.01,
        "label_suffix": "(0 – 30)",
    },
    "Admission Evaluation": {
        "min_val": 0.0,   "max_val": 100.0,  "default": 87.0,  "step": 0.01,
        "label_suffix": "(0 – 100, percentage)",
    },
    "SAT": {
        "min_val": 400.0, "max_val": 1600.0, "default": 1000.0, "step": 10.0,
        "label_suffix": "(400 – 1600)",
    },
    "V - ACT": {
        "min_val": 400.0, "max_val": 1200.0, "default": 800.0,  "step": 1.0,
        "label_suffix": "(400 – 1200)",
    },
}

ADMISSION_OPTIONS = list(ADMISSION_SCORE_CONFIG.keys())
REGION_OPTIONS    = ["Southeast", "Highlands", "Mekong Delta", "North Central",
                     "Central", "Red River Delta", "Northeast"]
HS_TYPE_OPTIONS   = ["Public", "Gifted", "Private"]
SCHOLARSHIP_MAP   = {"No Scholarship": 0, "Partial": 1, "Full-ride": 2}
TOP_N_FACTORS     = 8

CLEAN_NAMES = {
    "EntranceScore_Std":      "Entrance Score (Standardised)",
    "mean_score_std":         "Cohort Mean Score",
    "std_score_std":          "Cohort Score Spread",
    "female_ratio":           "Female Ratio in Major",
    "mean_scholarship_type":  "Avg Scholarship Level",
    "priority_ratio":         "Priority Student Ratio",
    "mean_lang_score":        "Cohort Avg Lang Score",
    "region_entropy":         "Regional Diversity",
    "admission_entropy":      "Admission Channel Diversity",
    "hs_type_entropy":        "High School Type Diversity",
    "HasPriorityScore":       "Has Priority Score",
    "LanguageCertiScore":     "Language Certificate Score",
    "HighSchoolType":         "High School Type",
    "ScholarshipType":        "Scholarship Type",
    "SchoolYear":             "School Year",
    "CreditsRequired":        "Credits Required",
    "CourseRequired":         "Courses Required",
    "FirstYearCreditsLoad":   "First-Year Credit Load",
    "MathIntensive":          "Math-Intensive Programme",
    "IsSTEM":                 "STEM Programme",
    "ScoreDev":               "Score Deviation from Cohort",
    "ScoreZ":                 "Score Z-Score in Cohort",
    "ScholarshipSelect":      "Scholarship × Cohort Score",
    "PrioritySelect":         "Priority × Cohort Score",
    "ScorePercentile":        "Score Percentile in Major",
    "LangScoreDev":           "Lang Score vs Cohort Avg",
    "Age":                    "Age at Admission",
    "Gender":                 "Gender",
    "Admission":              "Admission Channel",
    "Region":                 "Home Region",
}


def clean(name: str) -> str:
    if name.startswith("Major_"):
        return name.replace("Major_", "Major: ")
    return CLEAN_NAMES.get(name, name.replace("_", " ").title())


# Feature

def standardise_score(raw_score: float, admission: str) -> float:
    """Standardise entrance exam score per admission channel (using train stats)."""
    stats = admission_score_stats.get(admission, admission_score_stats["__global__"])
    return (raw_score - stats["mean"]) / stats["std"]


def build_m4_vector(
    gender, admission, region, hs_type,
    age, lang_score, has_priority,
    scholarship_str, entrance_raw, major,
) -> pd.DataFrame:
    """Assemble the full M4 feature vector in the exact column order the model expects."""

    #  Encode categoricals 
    gender_enc   = encoder_maps["Gender"].get(gender, -1)
    admission_enc = encoder_maps["Admission"].get(admission, -1)
    region_enc   = encoder_maps["Region"].get(region, -1)
    hs_enc       = encoder_maps["HighSchoolType"].get(hs_type, -1)
    scholarship  = SCHOLARSHIP_MAP[scholarship_str]

    entrance_std = standardise_score(entrance_raw, admission)

    # Major context lookup
    row = context_db[context_db["Major"] == major].iloc[0].to_dict()

    mean_s   = row["mean_score_std"]
    std_s    = row["std_score_std"] if row["std_score_std"] != 0 else 1.0

    # Build base dict 
    feat = {
        "Gender":           gender_enc,
        "Admission":        admission_enc,
        "EntranceScore_Std": entrance_std,
        "Region":           region_enc,
        "HighSchoolType":   hs_enc,
        "Age":              age,
        "LanguageCertiScore": lang_score,
        "HasPriorityScore": int(has_priority),
        "ScholarshipType":  scholarship,
        # curriculum
        "CreditsRequired":     row["CreditsRequired"],
        "CourseRequired":      row["CourseRequired"],
        "FirstYearCreditsLoad": row["FirstYearCreditsLoad"],
        "MathIntensive":       row["MathIntensive"],
        "IsSTEM":              row["IsSTEM"],
        # distribution context
        "mean_score_std":        mean_s,
        "std_score_std":         std_s,
        "female_ratio":          row["female_ratio"],
        "mean_scholarship_type": row["mean_scholarship_type"],
        "priority_ratio":        row["priority_ratio"],
        "mean_lang_score":       row["mean_lang_score"],
        "region_entropy":        row["region_entropy"],
        "admission_entropy":     row["admission_entropy"],
        "hs_type_entropy":       row["hs_type_entropy"],
        # interaction features
        "ScoreDev":        entrance_std - mean_s,
        "ScoreZ":          (entrance_std - mean_s) / std_s,
        "ScholarshipSelect": scholarship * mean_s,
        "PrioritySelect":   int(has_priority) * mean_s,
        "ScorePercentile": norm.cdf((entrance_std - mean_s) / std_s),
        "LangScoreDev":    lang_score - row["mean_lang_score"],
    }

    # Handle One-Hot Major columns if present in M4
    for col in M4_FEATURE_COLS:
        if col.startswith("Major_"):
            feat[col] = 1 if col == f"Major_{major}" else 0

    # Ensure column order exactly matches training
    df_out = pd.DataFrame([feat])
    for col in M4_FEATURE_COLS:
        if col not in df_out.columns:
            df_out[col] = 0
    df_out = df_out[M4_FEATURE_COLS]

    return df_out


# Shap 
@st.cache_resource(show_spinner="Preparing explainer …")
def get_explainer():
    return shap.TreeExplainer(model)


def compute_factor_table(input_df: pd.DataFrame, top_n: int = TOP_N_FACTORS):
    explainer   = get_explainer()
    shap_vals   = explainer.shap_values(input_df)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    sv = shap_vals.flatten()           
    total_abs = np.abs(sv).sum()

    records = []
    for i, col in enumerate(input_df.columns):
        records.append({
            "feature":   col,
            "name":      clean(col),
            "shap":      float(sv[i]),
            "abs_shap":  abs(float(sv[i])),
            "pct":       abs(float(sv[i])) / total_abs * 100 if total_abs > 0 else 0.0,
            "direction": "↑ Risk" if sv[i] > 0 else "↓ Risk",
            "positive":  sv[i] > 0,
        })

    df_factors = (
        pd.DataFrame(records)
        .sort_values("abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return df_factors


# Result rendering 
def render_gauge(prob_dropout: float):
    prob_grad = 1.0 - prob_dropout
    pct_d = f"{prob_dropout * 100:.1f}%"
    pct_g = f"{prob_grad    * 100:.1f}%"
    if prob_dropout < 0.35:
        verdict_cls  = "safe"
        verdict_text = "🟢 Low Risk — Student shows a strong graduation profile."
    elif prob_dropout < 0.60:
        verdict_cls  = "warning"
        verdict_text = "🟡 Moderate Risk — Some vulnerability indicators present."
    else:
        verdict_cls  = "danger"
        verdict_text = "🔴 High Risk — Multiple dropout risk factors detected."

    st.markdown(
        f"""
        <div class="gauge-row">
          <div class="gauge-card grad">
            <div class="gauge-pct">{pct_g}</div>
            <div class="gauge-label">Likely to Graduate</div>
            <div class="gauge-sub">Probability of completion</div>
          </div>
          <div class="gauge-card drop">
            <div class="gauge-pct">{pct_d}</div>
            <div class="gauge-label">Dropout Risk</div>
            <div class="gauge-sub">Probability of leaving</div>
          </div>
        </div>
        <div class="verdict {verdict_cls}">{verdict_text}</div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bar(prob_dropout: float):
    """Horizontal stacked bar showing graduate vs dropout split."""
    pct_d = prob_dropout * 100
    pct_g = 100 - pct_d

    fig, ax = plt.subplots(figsize=(8, 0.9))
    ax.barh(0, pct_g, color="#10b981", height=0.55)
    ax.barh(0, pct_d, left=pct_g, color="#ef4444", height=0.55)
    ax.set_xlim(0, 100)
    ax.axis("off")

    ax.text(pct_g / 2,         0, f"Graduate {pct_g:.1f}%", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    ax.text(pct_g + pct_d / 2, 0, f"Dropout {pct_d:.1f}%",  ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

    fig.patch.set_alpha(0)
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_factor_table(df_factors: pd.DataFrame):
    """Render the top-factor breakdown as an HTML table with inline bars."""
    max_pct = df_factors["pct"].max()

    rows_html = ""
    for i, row in df_factors.iterrows():
        bar_w   = int(row["pct"] / max_pct * 100) if max_pct > 0 else 0
        color   = "#ef4444" if row["positive"] else "#10b981"
        dir_css = f"color:{color}; font-weight:600;"
        rows_html += f"""
        <div class="factor-row">
          <div class="factor-rank">{i+1}</div>
          <div class="factor-name">{row['name']}</div>
          <div class="factor-bar-wrap">
            <div class="factor-bar-fill" style="width:{bar_w}%; background:{color};"></div>
          </div>
          <div class="factor-pct">{row['pct']:.1f}%</div>
          <div class="factor-direction" style="{dir_css}">{row['direction']}</div>
        </div>
        """

    st.markdown(rows_html, unsafe_allow_html=True)


def render_factor_chart(df_factors: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, max(3, len(df_factors) * 0.6)))

    colors = ["#ef4444" if p else "#10b981" for p in df_factors["positive"]]
    y_pos  = np.arange(len(df_factors))

    ax.barh(y_pos, df_factors["pct"], color=colors, height=0.55, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_factors["name"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to model output (%)", fontsize=9)
    ax.set_title("Top Dropout Risk Factors", fontsize=10, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5)

    # Legend patches
    red_patch   = mpatches.Patch(color="#ef4444", label="↑ Increases dropout risk")
    green_patch = mpatches.Patch(color="#10b981", label="↓ Reduces dropout risk")
    ax.legend(handles=[red_patch, green_patch], fontsize=8, loc="lower right",
              framealpha=0.9)

    fig.patch.set_facecolor("#f9fafb")
    ax.set_facecolor("#f9fafb")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# Layout
st.markdown(
    """
    <div class="hero">
      <h1>🎓 Student Dropout Risk Predictor</h1>
      <p>Enter enrolment details below. The model automatically incorporates
      major-specific curriculum and peer-cohort context to estimate dropout probability.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="section-title">Student Information</div>', unsafe_allow_html=True)

pre_c1, pre_c2 = st.columns(2)
with pre_c1:
    major     = st.selectbox("Major", sorted(context_db["Major"].tolist()))
with pre_c2:
    admission = st.selectbox("Admission Channel", ADMISSION_OPTIONS)

score_cfg = ADMISSION_SCORE_CONFIG[admission]

st.caption(
    f"ℹ️  **{admission}** — expected score range: "
    f"**{score_cfg['min_val']:.0f} – {score_cfg['max_val']:.0f}** "
    f"{score_cfg['label_suffix']}"
)

with st.form("prediction_form"):
    entrance_raw = st.number_input(
        f"Entrance Exam Score {score_cfg['label_suffix']}",
        min_value=score_cfg["min_val"],
        max_value=score_cfg["max_val"],
        value=score_cfg["default"],
        step=score_cfg["step"],
        help=(
            f"Score range for '{admission}': "
            f"{score_cfg['min_val']} – {score_cfg['max_val']}. "
            "Standardised against peers in the same admission channel before prediction."
        ),
    )

    st.markdown(
        "<div style='font-size:.7rem;font-weight:700;letter-spacing:.1em;"
        "text-transform:uppercase;color:#6b7280;margin:.75rem 0 .4rem'>Personal Details</div>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        gender      = st.selectbox("Gender", ["Female", "Male"])
        region      = st.selectbox("Home Region", REGION_OPTIONS)
    with col2:
        hs_type     = st.selectbox("High School Type", HS_TYPE_OPTIONS)
        age         = st.number_input("Age at Admission", 16, 50, 18)
    with col3:
        scholarship  = st.selectbox("Scholarship", list(SCHOLARSHIP_MAP.keys()))
        ielts_bands  = [round(x * 0.5, 1) for x in range(0, 19)]  # 0.0 → 9.0
        lang_score   = st.selectbox(
            "Language Certificate Score (IELTS band)",
            options=ielts_bands,
            index=0,
            format_func=lambda x: "0.0 — No certificate" if x == 0.0 else str(x),
            help="IELTS-equivalent band score. Select 0.0 if the student holds no "
                 "language certificate. Bands run 0.0 – 9.0 in 0.5 steps.",
        )
        has_priority = st.checkbox("Has Priority Score")

    submitted = st.form_submit_button("Predict Dropout Risk →", type="primary")


# Prediction & Result

if submitted:
    with st.spinner("Running M4 model …"):
        input_df     = build_m4_vector(
            gender, admission, region, hs_type,
            age, lang_score, has_priority,
            scholarship, entrance_raw, major,
        )
        prob_dropout = float(model.predict_proba(input_df)[0][1])
        df_factors   = compute_factor_table(input_df, top_n=TOP_N_FACTORS)

    st.markdown("---")

    # Outcome probability 
    st.subheader("Predicted Outcome")
    render_gauge(prob_dropout)
    render_probability_bar(prob_dropout)

    st.markdown("---")

    # Factor analysis
    st.subheader(f"Top {TOP_N_FACTORS} Contributing Factors")
    st.caption(
        "Percentage = share of total model output attributed to each factor. "
        "Red = increases dropout risk · Green = reduces dropout risk."
    )

    col_tbl, col_chart = st.columns([1, 1])
    with col_tbl:
        render_factor_table(df_factors)
    with col_chart:
        render_factor_chart(df_factors)

    with st.expander("ℹ️  Major curriculum & cohort context used in prediction"):
        ctx = context_db[context_db["Major"] == major].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Credits Required",    int(ctx["CreditsRequired"]))
        c2.metric("Courses Required",    int(ctx["CourseRequired"]))
        c3.metric("1st-Year Credit Load", int(ctx["FirstYearCreditsLoad"]))
        c4, c5, c6 = st.columns(3)
        c4.metric("STEM Programme",       "Yes" if ctx["IsSTEM"] else "No")
        c5.metric("Math-Intensive",       "Yes" if ctx["MathIntensive"] else "No")
        c6.metric("Cohort Female Ratio",  f"{ctx['female_ratio']*100:.0f}%")
        c7, c8, c9 = st.columns(3)
        c7.metric("Cohort Mean Score",    f"{ctx['mean_score_std']:.2f} σ")
        c8.metric("Cohort Avg Lang Score",f"{ctx['mean_lang_score']:.2f}")
        c9.metric("Priority Student Ratio", f"{ctx['priority_ratio']*100:.0f}%")