import os
import json
import numpy as np
import pandas as pd
import warnings
from collections import Counter
from scipy.stats import t
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

from ablation import create_ablation_matrices
from config import target_col, candidate_models, tuning_setup, entrance_cols, CAT_FEATURE_INDICES

warnings.filterwarnings("ignore")

N_SPLITS  = 5
N_REPEATS = 3
J = N_SPLITS * N_REPEATS
n_test_n_train_ratio = 1.0 / (N_SPLITS - 1)
nb_correction_factor = (1.0 / J) + n_test_n_train_ratio

def get_corrected_se(scores):
    var_raw = np.var(scores, ddof=1)
    return np.sqrt(var_raw * nb_correction_factor)

def run_model_selection(processed_folds):
    print("\nModel Selection Experiment")
    print("-" * 60)

    cat_col_names = [entrance_cols[i] for i in CAT_FEATURE_INDICES]
    selection_results = {name: {'auc': [], 'recall': [], 'precision': [], 'f1': []} for name in candidate_models}
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for fold_info in processed_folds:
        train_df = fold_info['train']
        y_train  = train_df[target_col]
        n_grad = (y_train == 0).sum()
        n_drop = (y_train == 1).sum()
        ratio  = n_grad / n_drop
        cw     = {0: 1.0, 1: ratio}
        sample_weights = np.where(y_train == 1, ratio, 1.0)

        fold_matrices = create_ablation_matrices(train_df, train_df)
        X_train_sel   = fold_matrices['M0']['X_train']

        for name, clf in candidate_models.items():
            inner_aucs, inner_recs, inner_precs, inner_f1s = [], [], [], []

            for inner_tr_idx, inner_val_idx in inner_cv.split(X_train_sel, y_train):
                current_model = clone(clf)
                if name == 'XGBoost':
                    current_model.set_params(scale_pos_weight=float(ratio))
                elif name == 'LightGBM':
                    current_model.set_params(class_weight=cw)
                elif name == 'CatBoost':
                    current_model.set_params(class_weights=[1.0, float(ratio)])
                elif name in ('Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest'):
                    if name in ['Logistic Regression', 'SVM']:
                        current_model.set_params(clf__class_weight=cw)
                    else:
                        current_model.set_params(class_weight=cw)

                X_in_train = X_train_sel.iloc[inner_tr_idx].copy()
                X_in_val   = X_train_sel.iloc[inner_val_idx].copy()
                y_in_train = y_train.iloc[inner_tr_idx]
                y_in_val   = y_train.iloc[inner_val_idx]

                if name == 'AdaBoost':
                    sw_inner = sample_weights[inner_tr_idx]
                    current_model.fit(X_in_train, y_in_train, sample_weight=sw_inner)
                elif name == 'CatBoost':
                    for col in cat_col_names:
                        X_in_train[col] = X_in_train[col].astype(int).astype(str)
                        X_in_val[col]   = X_in_val[col].astype(int).astype(str)
                    current_model.fit(X_in_train, y_in_train, cat_features=CAT_FEATURE_INDICES)
                else:
                    current_model.fit(X_in_train, y_in_train)

                y_pred_proba = current_model.predict_proba(X_in_val)[:, 1]
                y_pred_class = current_model.predict(X_in_val)

                inner_aucs.append(roc_auc_score(y_in_val, y_pred_proba))
                inner_recs.append(recall_score(y_in_val, y_pred_class, zero_division=0))
                inner_precs.append(precision_score(y_in_val, y_pred_class, zero_division=0))
                inner_f1s.append(f1_score(y_in_val, y_pred_class, zero_division=0))

            selection_results[name]['auc'].append(np.mean(inner_aucs))
            selection_results[name]['recall'].append(np.mean(inner_recs))
            selection_results[name]['precision'].append(np.mean(inner_precs))
            selection_results[name]['f1'].append(np.mean(inner_f1s))

    print("\nMODEL SELECTION RESULTS")
    print("Mean +/- Std over 3 repeats (each repeat = mean of 5 folds)")
    print("=" * 80)
    print(f"{'Model':<22} | {'ROC-AUC':<16} | {'Recall':<16} | {'Precision':<16} | {'F1-Score':<16}")
    print("-" * 80)

    for name in candidate_models:
        r_aucs, r_recs, r_precs, r_f1s = [], [], [], []
        for r in range(N_REPEATS):
            s = r * N_SPLITS
            e = s + N_SPLITS
            r_aucs.append(np.mean(selection_results[name]['auc'][s:e]))
            r_recs.append(np.mean(selection_results[name]['recall'][s:e]))
            r_precs.append(np.mean(selection_results[name]['precision'][s:e]))
            r_f1s.append(np.mean(selection_results[name]['f1'][s:e]))

        print(f"{name:<22} | {np.mean(r_aucs):.3f} +/- {np.std(r_aucs):.3f}   | "
              f"{np.mean(r_recs):.3f} +/- {np.std(r_recs):.3f}   | "
              f"{np.mean(r_precs):.3f} +/- {np.std(r_precs):.3f}   | "
              f"{np.mean(r_f1s):.3f} +/- {np.std(r_f1s):.3f}")
    print("=" * 80)

def run_nested_cv_ablation(processed_folds):
    print("\n" + "=" * 85)
    print("NESTED CV TUNING & ABLATION STUDY")
    print("=" * 85)
    print("Loading hyperparameters from JSON if available to save compute.\n")

    models_to_test = ['M0', 'M1', 'M2', 'M3', 'M4']
    cat_col_names = [entrance_cols[i] for i in CAT_FEATURE_INDICES]

    PARAMS_FILE = "tuned_hyperparameters.json"
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            tuned_params_cache = json.load(f)
    else:
        tuned_params_cache = {}

    for algo_name, setup in tuning_setup.items():
        print("\n" + "=" * 85)
        print(f"ALGORITHM: {algo_name.upper()}")
        print("=" * 85)

        results = {model: {'auc': [], 'precision': [], 'recall': [], 'f1': []} for model in models_to_test}
        tier_best_params = {model: [] for model in models_to_test}

        for fold_idx, fold_info in enumerate(processed_folds):
            train_df = fold_info['train']
            test_df  = fold_info['test']
            y_train  = train_df[target_col]
            y_test   = test_df[target_col]

            n_grad_fold = (y_train == 0).sum()
            n_drop_fold = (y_train == 1).sum()
            dynamic_scale_pos_w = n_grad_fold / n_drop_fold
            dynamic_class_weight = {0: 1.0, 1: dynamic_scale_pos_w}

            fold_matrices = create_ablation_matrices(train_df, test_df)

            for model_name in models_to_test:
                X_train_tier = fold_matrices[model_name]['X_train']
                X_test_tier  = fold_matrices[model_name]['X_test']
                clf = clone(setup['base_clf'])

                if algo_name == 'XGBoost':
                    clf.set_params(scale_pos_weight=dynamic_scale_pos_w)
                elif algo_name == 'CatBoost':
                    pass
                else:
                    clf.set_params(class_weight=dynamic_class_weight)

                fit_params = {}
                if algo_name == 'CatBoost':
                    X_train_cb = X_train_tier.copy()
                    X_test_cb  = X_test_tier.copy()
                    for col in cat_col_names:
                        if col in X_train_cb.columns:
                            X_train_cb[col] = X_train_cb[col].astype(int).astype(str)
                            X_test_cb[col]  = X_test_cb[col].astype(int).astype(str)
                    X_train_tier = X_train_cb
                    X_test_tier  = X_test_cb
                    current_cat_indices = [X_train_tier.columns.get_loc(c) for c in cat_col_names if c in X_train_tier.columns]
                    sw = np.where(y_train == 1, float(dynamic_scale_pos_w), 1.0)
                    fit_params = {'cat_features': current_cat_indices, 'sample_weight': sw}

                cache_key = f"{algo_name}_{model_name}_fold_{fold_idx}"
                
                # Check JSON Cache for Hyperparameters
                if cache_key in tuned_params_cache:
                    best_params = tuned_params_cache[cache_key]
                    clf.set_params(**best_params)
                    if fit_params:
                        clf.fit(X_train_tier, y_train, **fit_params)
                    else:
                        clf.fit(X_train_tier, y_train)
                    best_tier_clf = clf
                    tier_best_params[model_name].append(str(best_params))
                else:
                    # Run Tuning if not found
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    grid_search = GridSearchCV(
                        estimator=clf,
                        param_grid=setup['grid'],
                        scoring='f1',
                        cv=inner_cv,
                        n_jobs=-1
                    )
                    
                    if fit_params:
                        grid_search.fit(X_train_tier, y_train, **fit_params)
                    else:
                        grid_search.fit(X_train_tier, y_train)

                    best_tier_clf = grid_search.best_estimator_
                    
                    # Save to cache dynamically
                    tuned_params_cache[cache_key] = grid_search.best_params_
                    with open(PARAMS_FILE, 'w') as f:
                        json.dump(tuned_params_cache, f, indent=4)
                        
                    tier_best_params[model_name].append(str(grid_search.best_params_))

                # Outer CV evaluation
                y_pred_proba = best_tier_clf.predict_proba(X_test_tier)[:, 1]
                y_pred_class = best_tier_clf.predict(X_test_tier)

                results[model_name]['auc'].append(roc_auc_score(y_test, y_pred_proba))
                results[model_name]['precision'].append(precision_score(y_test, y_pred_class, zero_division=0))
                results[model_name]['recall'].append(recall_score(y_test, y_pred_class, zero_division=0))
                results[model_name]['f1'].append(f1_score(y_test, y_pred_class, zero_division=0))

        # Printing results for this algorithm
        print(f"{'Tier':<5} | {'ROC-AUC':<18} | {'Recall ↑':<18} | {'Precision':<18} | {'F1-Score':<18}")
        print("-" * 85)

        for model_name in models_to_test:
            mean_auc  = np.mean(results[model_name]['auc'])
            mean_rec  = np.mean(results[model_name]['recall'])
            mean_prec = np.mean(results[model_name]['precision'])
            mean_f1   = np.mean(results[model_name]['f1'])
            se_auc  = get_corrected_se(results[model_name]['auc'])
            se_rec  = get_corrected_se(results[model_name]['recall'])
            se_prec = get_corrected_se(results[model_name]['precision'])
            se_f1   = get_corrected_se(results[model_name]['f1'])

            print(f"{model_name:<5} | {mean_auc:.3f} +/- {se_auc:.3f}   | {mean_rec:.3f} +/- {se_rec:.3f}   | {mean_prec:.3f} +/- {se_prec:.3f}   | {mean_f1:.3f} +/- {se_f1:.3f}")

        print("-" * 85)
        print("Most Frequent Hyperparameters Chosen (Baseline vs Full Model):")
        m0_params = Counter(tier_best_params['M0']).most_common(1)[0]
        m4_params = Counter(tier_best_params['M4']).most_common(1)[0]
        print(f"  M0 (Baseline) : {m0_params[0]} (Chosen {m0_params[1]}/15 folds)")
        print(f"  M4 (Full)     : {m4_params[0]} (Chosen {m4_params[1]}/15 folds)")

        print("-" * 85)
        print(f"Statistical Significance: M4 vs Baselines for {algo_name}")
        scores_m4 = results['M4']['f1']

        for baseline in ['M0', 'M1', 'M2', 'M3']:
            scores_base = results[baseline]['f1']
            diffs = np.array(scores_m4) - np.array(scores_base)
            var_diff = np.var(diffs, ddof=1)

            if var_diff == 0:
                p_value = 1.0
            else:
                t_stat = np.mean(diffs) / np.sqrt(var_diff * nb_correction_factor)
                p_value = (1 - t.cdf(abs(t_stat), df=J - 1)) * 2

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            diff_mean = np.mean(scores_m4) - np.mean(scores_base)
            sign = "+" if diff_mean > 0 else ""

            print(f"  M4 vs {baseline:<2} | Diff: {sign}{diff_mean:.3f} | p-value: {p_value:.4f} ({sig})")