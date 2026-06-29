import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Feature Definitions
categorical_features = ['Gender', 'Admission', 'Region', 'HighSchoolType']
numerical_features   = ['EntranceExamScore', 'Age', 'LanguageCertiScore', 'SchoolYear']
binary_features      = ['HasPriorityScore']
target_col = 'Status'

entrance_cols = [
    'Gender', 'Admission', 'EntranceScore_Std', 'Region',
    'HighSchoolType', 'Age', 'LanguageCertiScore',
    'HasPriorityScore', 'ScholarshipType'
]

curriculum_cols = [
    'CreditsRequired', 'CourseRequired', 'FirstYearCreditsLoad',
    'MathIntensive', 'IsSTEM'
]

dist_cols = [
    'mean_score_std', 'std_score_std',
    'female_ratio', 'mean_scholarship_type',
    'priority_ratio', 'mean_lang_score', 'region_entropy',
    'admission_entropy', 'hs_type_entropy'
]

interaction_cols = [
    'ScoreDev', 'ScoreZ', 'ScholarshipSelect', 'PrioritySelect',
    'ScorePercentile', 'LangScoreDev'
]

CAT_FEATURE_INDICES = [
    entrance_cols.index('Gender'),
    entrance_cols.index('Admission'),
    entrance_cols.index('Region'),
    entrance_cols.index('HighSchoolType'),
]

# Base Models for Block 2
candidate_models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Naive Bayes': GaussianNB(priors=[0.5, 0.5]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=42))
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                   n_estimators=100, max_depth=4, learning_rate=0.1,
                                   random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=100, depth=4, learning_rate=0.1,
                                    random_seed=42, verbose=0)
}

# Tuning grids 
tuning_setup = {
    'Random Forest': {
        'base_clf': RandomForestClassifier(random_state=42),
        'grid': {'n_estimators': [50, 100], 'max_depth': [3, 4, 5, 6], 'min_samples_split': [2, 5, 10]}
    },
    'XGBoost': {
        'base_clf': xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42),
        'grid': {'n_estimators': [50, 100], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1]}
    },
    'LightGBM': {
        'base_clf': LGBMClassifier(random_state=42, verbose=-1),
        'grid': {'n_estimators': [50, 100], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1]}
    },
    'CatBoost': {
        'base_clf': CatBoostClassifier(random_seed=42, verbose=0),
        'grid': {'iterations': [50, 100], 'depth': [3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1]}
    }
}