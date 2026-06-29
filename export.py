"""
Export.py — Production export script for Student Dropout Predictor
Trains the final CatBoost M4 model and saves all artefacts needed by App.py
"""

import os
import json
import ast
import pandas as pd
import numpy as np
from collections import Counter
from catboost import CatBoostClassifier

from config import target_col, entrance_cols, CAT_FEATURE_INDICES
from preprocessing import prepare_data_and_anova, generate_folds
from ablation import create_ablation_matrices

ARTIFACTS_DIR = "artifacts"

def _build_cat_col_names():
    return [entrance_cols[i] for i in CAT_FEATURE_INDICES]


def export_m4_production():

    print("M4 CatBoost Production Model")
 

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Load & preprocess
    print("\nLoading data and running preprocessing pipeline …")
    df  = pd.read_csv("data/student_data.csv")
    df1 = pd.read_csv("data/major.csv")

    df              = prepare_data_and_anova(df)
    processed_folds = generate_folds(df, df1)

    
    fold_info = processed_folds[0]
    train_df  = fold_info["train"]
    y_train   = train_df[target_col]

    print("\nExporting OrdinalEncoder categories …")
    encoder = fold_info["encoder"]


    from config import categorical_features
    encoder_maps = {}
    for feat, cats in zip(categorical_features, encoder.categories_):
        encoder_maps[feat] = {str(cat): int(idx) for idx, cat in enumerate(cats)}

    with open(os.path.join(ARTIFACTS_DIR, "encoder_maps.json"), "w") as fh:
        json.dump(encoder_maps, fh, indent=2)
    print("  Saved encoder_maps.json")

    # Export major context lookup table
    print("\nBuilding Major context database …")
    context_cols = [
        "Major",
        # distribution context 
        "mean_score_std", "std_score_std",
        "female_ratio", "mean_scholarship_type",
        "priority_ratio", "mean_lang_score",
        "region_entropy", "admission_entropy", "hs_type_entropy",
        # curriculum features 
        "CreditsRequired", "CourseRequired", "FirstYearCreditsLoad",
        "MathIntensive", "IsSTEM",
    ]
    major_context_db = (
        train_df[context_cols]
        .drop_duplicates(subset=["Major"])
        .reset_index(drop=True)
    )
    major_context_db.to_csv(os.path.join(ARTIFACTS_DIR, "major_context_db.csv"), index=False)
    print(f" Saved major_context_db.csv  ({len(major_context_db)} majors)")

    major_score_stats = (
        train_df.groupby("Major")["EntranceScore_Std"]
        .apply(list)
        .reset_index()
        .rename(columns={"EntranceScore_Std": "scores"})
    )
    major_score_stats.to_json(os.path.join(ARTIFACTS_DIR, "major_score_stats.json"), orient="records")
    print(" Saved major_score_stats.json")

   
    print("\nTraining M4 CatBoost model …")
    cat_col_names = _build_cat_col_names()
    fold_matrices = create_ablation_matrices(train_df, fold_info["test"])
    X_train_m4    = fold_matrices["M4"]["X_train"].copy()

    current_cat_indices = []
    for col in cat_col_names:
        if col in X_train_m4.columns:
            X_train_m4[col] = X_train_m4[col].astype(int).astype(str)
            current_cat_indices.append(X_train_m4.columns.get_loc(col))

    n_grad = (y_train == 0).sum()
    n_drop = (y_train == 1).sum()
    ratio  = float(n_grad / n_drop)
    sw     = np.where(y_train == 1, ratio, 1.0)

    m4_feature_cols = list(X_train_m4.columns)
    with open(os.path.join(ARTIFACTS_DIR, "m4_feature_cols.json"), "w") as fh:
        json.dump(m4_feature_cols, fh, indent=2)
    print(f"Saved m4_feature_cols.json  ({len(m4_feature_cols)} features)")

    model = CatBoostClassifier(random_seed=42, verbose=0)

    PARAMS_FILE = "tuned_hyperparameters.json"
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as fh:
            tuned_params_cache = json.load(fh)

        m4_params_list = [
            str(v)
            for k, v in tuned_params_cache.items()
            if k.startswith("CatBoost_M4_fold_")
        ]
        if m4_params_list:
            best_params = ast.literal_eval(
                Counter(m4_params_list).most_common(1)[0][0]
            )
            print(f"  Loaded optimal M4 params: {best_params}")
            model.set_params(**best_params)
        else:
            print("  Warning: no M4 params in cache — using defaults.")
    else:
        print("  Warning: JSON cache not found — using defaults.")

    model.fit(
        X_train_m4,
        y_train,
        cat_features=current_cat_indices,
        sample_weight=sw,
    )
    model.save_model(os.path.join(ARTIFACTS_DIR, "catboost_m4_model.cbm"))
    print("Saved catboost_m4_model.cbm")

    print("\nComputing global score stats for UI")
    global_stats = {}
    for admission in train_df["Admission"].unique():
        mask = train_df["Admission"] == admission
        m    = float(train_df.loc[mask, "EntranceExamScore"].mean())
        s    = float(train_df.loc[mask, "EntranceExamScore"].std())
        s    = s if pd.notnull(s) and s != 0 else 1.0
        global_stats[str(admission)] = {"mean": m, "std": s}

    # Fallback global
    global_stats["__global__"] = {
        "mean": float(train_df["EntranceExamScore"].mean()),
        "std":  float(train_df["EntranceExamScore"].std()),
    }

    with open(os.path.join(ARTIFACTS_DIR, "admission_score_stats.json"), "w") as fh:
        json.dump(global_stats, fh, indent=2)
    print(" Saved admission_score_stats.json")
    print("Export complete")



if __name__ == "__main__":
    export_m4_production()