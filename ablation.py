import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from config import entrance_cols, curriculum_cols, dist_cols, interaction_cols

def create_ablation_matrices(train, test):
    X_train_entrance = train[entrance_cols].copy()
    X_test_entrance  = test[entrance_cols].copy()

    X_train_curr     = train[curriculum_cols].copy()
    X_test_curr      = test[curriculum_cols].copy()

    X_train_dist     = train[dist_cols].copy()
    X_test_dist      = test[dist_cols].copy()

    X_train_interact = train[interaction_cols].copy()
    X_test_interact  = test[interaction_cols].copy()

    ohe_major       = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)
    train_major_arr = ohe_major.fit_transform(train[['Major']])
    test_major_arr  = ohe_major.transform(test[['Major']])
    ohe_cols        = [f"Major_{cat}" for cat in ohe_major.categories_[0]]
    
    train_major_ohe = pd.DataFrame(train_major_arr, columns=ohe_cols, index=train.index)
    test_major_ohe  = pd.DataFrame(test_major_arr, columns=ohe_cols, index=test.index)

    ablation_data = {
        'M0': {
            'X_train': X_train_entrance,
            'X_test':  X_test_entrance
        },
        'M1': {
            'X_train': pd.concat([X_train_entrance, train_major_ohe], axis=1),
            'X_test':  pd.concat([X_test_entrance,  test_major_ohe],  axis=1)
        },
        'M2': {
            'X_train': pd.concat([X_train_entrance, X_train_curr], axis=1),
            'X_test':  pd.concat([X_test_entrance,  X_test_curr],  axis=1)
        },
        'M3': {
            'X_train': pd.concat([X_train_entrance, X_train_curr, X_train_dist], axis=1),
            'X_test':  pd.concat([X_test_entrance,  X_test_curr,  X_test_dist],  axis=1)
        },
        'M4': {
            'X_train': pd.concat([X_train_entrance, X_train_curr, X_train_dist, X_train_interact], axis=1),
            'X_test':  pd.concat([X_test_entrance,  X_test_curr, X_test_dist,  X_test_interact],  axis=1)
        },
    }
    return ablation_data