import pandas as pd
from preprocessing import prepare_data_and_anova, generate_folds
from evaluation import run_model_selection, run_nested_cv_ablation
from shap_analysis import execute_shap
def main():
    print("Loading datasets...")
    df = pd.read_csv('data/student_data.csv')
    df1 = pd.read_csv('data/major.csv')
    
    # Preprocessing & ANOVA
    df = prepare_data_and_anova(df)
    
    # Stratified Folds and Context Generation
    processed_folds = generate_folds(df, df1)
    
    # Run Model Selection 
    run_model_selection(processed_folds)
    
    # Nested CV & Ablation 
    run_nested_cv_ablation(processed_folds)
    
    execute_shap(processed_folds)

if __name__ == "__main__":
    main()