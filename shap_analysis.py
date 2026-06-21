import os
import json
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from catboost import CatBoostClassifier

# Import your configurations and preprocessing functions
from config import target_col, entrance_cols, CAT_FEATURE_INDICES
from ablation import create_ablation_matrices

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
TARGET_MODEL   = 'CatBoost'
TIERS          = ['M0', 'M1', 'M2', 'M3', 'M4'] # Set to 4 tiers as requested
TOP_N_FEATURES = 15
SAVE_FIGURES   = True
FIG_DPI        = 150

# Reconstruct cat_col_names from your config
cat_col_names = [entrance_cols[i] for i in CAT_FEATURE_INDICES]

if os.path.exists("tuned_hyperparameters.json"):
    with open("tuned_hyperparameters.json", 'r') as f:
        tuned_params_cache = json.load(f)
    print(f"[OK] Successfully loaded parameters.")
else:
    raise FileNotFoundError("Parameter cache not found! Please check the file path.")

# --- SHAP COLLECTION FUNCTION ---
def collect_shap_values(tier, processed_folds, create_ablation_matrices, target_col, cat_col_names, params_cache):
    shap_list   = []
    X_test_list = []
    feature_names = None

    print(f"\n  Collecting SHAP values — CatBoost / Tier {tier} ...")

    for fold_idx, fold_info in enumerate(processed_folds):
        train_df = fold_info['train']
        test_df  = fold_info['test']
        y_train  = train_df[target_col]
        
        # Calculate dynamic class weights for sample_weights
        n_grad = (y_train == 0).sum()
        n_drop = (y_train == 1).sum()
        dynamic_spw = float(n_grad / n_drop)

        fold_matrices = create_ablation_matrices(train_df, test_df)
        X_train = fold_matrices[tier]['X_train'].copy()
        X_test  = fold_matrices[tier]['X_test'].copy()

        if feature_names is None:
            feature_names = list(X_train.columns)

        # Prepare Categorical Features for CatBoost (requires strings)
        for col in cat_col_names:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype(int).astype(str)
                X_test[col]  = X_test[col].astype(int).astype(str)
        
        cat_idx = [X_train.columns.get_loc(c) for c in cat_col_names if c in X_train.columns]
        sw = np.where(y_train == 1, dynamic_spw, 1.0)
        fit_params = {'cat_features': cat_idx, 'sample_weight': sw}

        # Initialize base model
        clf = CatBoostClassifier(random_seed=42, verbose=0)

        # FAST TRACK: Apply saved parameters
        cache_key = f"CatBoost_{tier}_fold_{fold_idx}"
        if cache_key in params_cache:
            best_params = params_cache[cache_key]
            clf.set_params(**best_params)
        else:
            print(f"    [WARNING] No cache found for {cache_key}. Using default CatBoost params.")

        # Blazing fast single fit (No GridSearch)
        clf.fit(X_train, y_train, **fit_params)

        # Extract SHAP Values
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        
        # CatBoost SHAP returns a list for binary classification sometimes, we need the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_list.append(shap_values)
        
        # ====================================================================
        # THE FIX PART 1: Convert X_test back to floats strictly for SHAP plots
        # This prevents the stringified categories from rendering as grey
        # ====================================================================
        X_test_numeric = X_test.astype(float)
        X_test_list.append(X_test_numeric.reset_index(drop=True))

    all_shap_values = np.vstack(shap_list)
    all_X_test      = pd.concat(X_test_list, axis=0).reset_index(drop=True)

    return all_shap_values, all_X_test, feature_names


# --- FEATURE CLEANING ---
def clean_feature_name(name):
    replacements = {
        'EntranceScore_Std':  'Entrance Score',
        'mean_score_std':     'Major Mean Score',
        'std_score_std':      'Major Score Std',
        'female_ratio':       'Gender Ratio',
        'mean_scholarship_type':  'Mean Scholarship Type',
        'priority_ratio':     'Priority Ratio',
        'mean_lang_score':    'Mean Lang Score',
        'region_entropy':     'Region Diversity',
        'admission_entropy':  'Admission Diversity',
        'hs_type_entropy':    'High School Type Diversity',
        'HasPriorityScore':   'Has Priority Score',
        'LanguageCertiScore': 'Language Cert Score',
        'HighSchoolType':     'High School Type',
        'ScholarshipType':    'Scholarship Type',
        'SchoolYear':         'School Year',
        'CreditsRequired':    'Credits Required',
        'CourseRequired':     'Courses Required',
        'FirstYearCreditsLoad': 'First Year Credit Load',
        'MathIntensive':      'Math Intensive',
        'IsSTEM':             'Is STEM',
        'ScoreDev':           'Score Deviation',
        'ScoreZ':             'Score Z-Score',
        'ScholarshipSelect':  'Scholarship Interaction',
        'PrioritySelect':     'Priority Interaction',
        'ScorePercentile':    'Score Percentile',
        'LangScoreDev':       'Language Score Deviation'
    }

    if name.startswith('Major_'):
        return name.replace('Major_', 'Major: ')
    return replacements.get(name, name)


# --- VISUALIZATION FUNCTION ---
def plot_tier_beeswarm(shap_vals, X_df, tier_name, top_n=TOP_N_FEATURES):
    """Generates a standalone beeswarm plot for a specific tier."""
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]
    top_names  = [clean_feature_name(X_df.columns[i]) for i in top_idx]
    
    shap_subset = shap_vals[:, top_idx]
    X_subset    = X_df.iloc[:, top_idx].copy()
    X_subset.columns = top_names

    # ====================================================================
    # THE FIX PART 2: Double check dtype enforcement before plotting
    # ====================================================================
    X_subset = X_subset.astype(float)

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)
    shap.summary_plot(
        shap_subset, X_subset,
        max_display=top_n,
        show=False,
        plot_size=None
    )
    ax.set_title(
        f'CatBoost SHAP Feature Importance — Tier {tier_name}\n'
        f'Colour = feature value (red=high, blue=low)',
        fontsize=11, fontweight='bold'
    )
    ax.set_xlabel('SHAP value (impact on dropout probability)', fontsize=9)
    plt.tight_layout()

    if SAVE_FIGURES:
        fname = f'shap_beeswarm_catboost_{tier_name.lower()}.png'
        plt.savefig(fname, dpi=FIG_DPI, bbox_inches='tight')
        print(f"  Saved plot to Drive: {fname}")
    plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
# Ensure processed_folds is passed into this script from your main pipeline
# For example, wrap this in a function run_all_shap(processed_folds) or execute after generation

def execute_shap(processed_folds):
    print("\n" + "=" * 70)
    print("BLOCK 4 — FAST SHAP ANALYSIS (CACHED PARAMETERS)")
    print("=" * 70)

    tier_results = {}

    # 1. Collect SHAP values for ALL tiers
    for tier in TIERS:
        shap_vals, X_test_df, _ = collect_shap_values(
            tier                   = tier,
            processed_folds        = processed_folds,
            create_ablation_matrices = create_ablation_matrices,
            target_col             = target_col,
            cat_col_names          = cat_col_names,
            params_cache           = tuned_params_cache
        )
        tier_results[tier] = (shap_vals, X_test_df)

    print(f"\n  → Generating visual reports for CatBoost...")

    # 2. Output a plot for EVERY tier
    for tier in TIERS:
        plot_tier_beeswarm(tier_results[tier][0], tier_results[tier][1], tier)

    print("\n" + "=" * 70)
    print("BLOCK 4 COMPLETE")
    print("=" * 70)