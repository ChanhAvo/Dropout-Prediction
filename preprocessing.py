import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from config import categorical_features, target_col

def calculate_entropy(series):
    value_counts = series.value_counts(normalize=True).values
    return entropy(value_counts, base=2)

def prepare_data_and_anova(df):
    df[target_col] = df[target_col].map({'Graduated': 0, 'Drop out': 1})
    schol_map = {'No Scholarship': 0, 'Partial': 1, 'Full-ride': 2}
    df['ScholarshipType']= df['ScholarshipType'].map(schol_map)
    
    major_groups = [
        df.loc[df['Major'] == m, target_col].values
        for m in df['Major'].unique()
    ]

    # One-way ANOVA
    f_stat, p_val = stats.f_oneway(*major_groups)

    # Extract variance components
    grand_mean  = df[target_col].mean()
    n_total     = len(df)
    n_groups    = df['Major'].nunique()
    group_sizes = df['Major'].value_counts().reindex(df['Major'].unique()).values

    SS_between = sum(
        n * (df.loc[df['Major'] == m, target_col].mean() - grand_mean) ** 2
        for n, m in zip(group_sizes, df['Major'].unique())
    )
    SS_total = ((df[target_col] - grand_mean) ** 2).sum()
    SS_within = SS_total - SS_between

    df_between = n_groups - 1
    df_within  = n_total - n_groups

    MS_between = SS_between / df_between
    MS_within  = SS_within  / df_within

    n0 = (n_total - (group_sizes ** 2).sum() / n_total) / (n_groups - 1)
    var_between = (MS_between - MS_within) / n0
    var_within  = MS_within

    icc = var_between / (var_between + var_within)
    icc = max(0.0, icc)

    print(f"[ICC] Intraclass Correlation Coefficient = {icc:.4f} ({icc*100:.1f}%)")
    print(f"      F({df_between}, {df_within}) = {f_stat:.3f}, p = {p_val:.4f}")
    
    assert set(df[target_col].unique()).issubset({0, 1}), (
        "Status column must be binary integer {0, 1}. "
        "Check raw data encoding before proceeding."
    )
    assert df[target_col].isnull().sum() == 0, "Status column contains NaN values."
    print(f"Status encoding confirmed: 1=Dropout ({(df[target_col]==1).sum()}), "
          f"0=Graduate ({(df[target_col]==0).sum()})")
    
    return df

def generate_folds(df, df1):
    df['Stratify_Key'] = df['Major'].astype(str) + "_" + df[target_col].astype(str)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    cv_splits = list(cv.split(df, df['Stratify_Key']))

    processed_folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        train = df.iloc[train_idx].copy()
        test  = df.iloc[test_idx].copy()

        train_major_counts = train['Major'].value_counts()
        test_major_counts  = test['Major'].value_counts()
        tiny_in_test       = test_major_counts[test_major_counts < 3]
        if not tiny_in_test.empty:
            print(f"  [Fold {fold_idx}] WARNING — majors with <3 test samples: "
                  f"{tiny_in_test.to_dict()}")

        fold_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        train[categorical_features] = fold_encoder.fit_transform(train[categorical_features])
        test[categorical_features] = fold_encoder.transform(test[categorical_features])

        # Standardize EntranceExamScore
        test['EntranceScore_Std'] = np.nan
        admission_stats = {}
        for admission in train['Admission'].unique():
            mask = train['Admission'] == admission
            m    = train.loc[mask, 'EntranceExamScore'].mean()
            s    = train.loc[mask, 'EntranceExamScore'].std()
            s    = s if pd.notnull(s) and s != 0 else 1.0
            admission_stats[admission] = (m, s)
            train.loc[mask, 'EntranceScore_Std'] = (train.loc[mask, 'EntranceExamScore'] - m) / s

        for admission, (m, s) in admission_stats.items():
            mask = test['Admission'] == admission
            if mask.any():
                test.loc[mask, 'EntranceScore_Std'] = (test.loc[mask, 'EntranceExamScore'] - m) / s

        global_train_mean = train['EntranceExamScore'].mean()
        global_train_std  = train['EntranceExamScore'].std()
        global_train_std  = global_train_std if global_train_std != 0 else 1.0
        test['EntranceScore_Std'] = test['EntranceScore_Std'].fillna(
            (test['EntranceExamScore'] - global_train_mean) / global_train_std
        )

        # Major Context Construction
        gender_categories  = fold_encoder.categories_[categorical_features.index('Gender')].tolist()
        female_encoded_val = gender_categories.index('Female')

        major_context_list = []
        for major in train['Major'].unique():
            major_students = train[train['Major'] == major]

            mean_score = major_students['EntranceScore_Std'].mean()
            std_score  = major_students['EntranceScore_Std'].std()
            std_score  = std_score if pd.notnull(std_score) and std_score != 0 else 1.0

            female_ratio      = (major_students['Gender'] == female_encoded_val).mean()
            scholarship_ratio = major_students['ScholarshipType'].mean()
            priority_ratio    = major_students['HasPriorityScore'].mean()
            lang_cert_ratio   = major_students['LanguageCertiScore'].mean()

            region_entropy    = calculate_entropy(major_students['Region'])
            admission_entropy = calculate_entropy(major_students['Admission'])
            hs_type_entropy   = calculate_entropy(major_students['HighSchoolType'])

            major_context_list.append({
                'Major':                 major,
                'mean_score_std':        mean_score,
                'std_score_std':         std_score,
                'female_ratio':          female_ratio,
                'mean_scholarship_type': scholarship_ratio,
                'priority_ratio':        priority_ratio,
                'mean_lang_score':       lang_cert_ratio,
                'region_entropy':        region_entropy,
                'admission_entropy':     admission_entropy,
                'hs_type_entropy':       hs_type_entropy
            })

        major_context_df = pd.DataFrame(major_context_list)
        major_context_df = major_context_df.merge(df1, on='Major', how='left')
        train = train.merge(major_context_df, on='Major', how='left')
        test  = test.merge(major_context_df,  on='Major', how='left')

        # Interaction Features
        for df_ref, ref_name in [(train, 'train'), (test, 'test')]:
            df_ref['ScoreDev']          = (df_ref['EntranceScore_Std'] - df_ref['mean_score_std'])
            df_ref['ScoreZ']            = (df_ref['ScoreDev']/ df_ref['std_score_std'])
            df_ref['ScholarshipSelect'] = df_ref['ScholarshipType'] * df_ref['mean_score_std']
            df_ref['PrioritySelect']    = (df_ref['HasPriorityScore'] * df_ref['mean_score_std'])
            df_ref['LangScoreDev']      = df_ref['LanguageCertiScore'] - df_ref['mean_lang_score']

            if ref_name == 'train':
                train = df_ref
            else:
                test = df_ref
                
        train['ScorePercentile'] = np.nan
        test['ScorePercentile'] = np.nan

        for major in train['Major'].unique():
            tr_mask = train['Major'] == major
            tr_scores = train.loc[tr_mask, 'EntranceScore_Std'].values
            tr_pcts = np.array([(tr_scores < s).mean() for s in tr_scores])
            train.loc[tr_mask, 'ScorePercentile'] = tr_pcts
            te_mask = test['Major'] == major
            if te_mask.any():
                te_scores = test.loc[te_mask, 'EntranceScore_Std'].values
                te_pcts = np.array([(tr_scores < s).mean() for s in te_scores])
                test.loc[te_mask, 'ScorePercentile'] = te_pcts

        if test['ScorePercentile'].isna().any():
            all_train_scores = train['EntranceScore_Std'].values
            nan_mask = test['ScorePercentile'].isna()
            te_scores_fallback = test.loc[nan_mask, 'EntranceScore_Std'].values
            fallback_pcts = np.array([(all_train_scores < s).mean() for s in te_scores_fallback])
            test.loc[nan_mask, 'ScorePercentile'] = fallback_pcts

        processed_folds.append({
            'fold_idx': fold_idx,
            'train':    train,
            'test':     test,
            'encoder':  fold_encoder
        })

    print(f"\nSuccessfully processed {len(processed_folds)} folds.")
    return processed_folds