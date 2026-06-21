import pandas as pd
from preprocessing import prepare_data_and_anova, generate_folds
from evaluation import run_model_selection, run_nested_cv_ablation
from shap_analysis import execute_shap
def main():
    # Replace these paths with your actual data ingestion sources
    print("Loading datasets...")
    df = pd.read_csv('data/student_data.csv')
    df1 = pd.read_csv('data/major.csv')
    
    
    # Step 1: Preprocessing & ANOVA
    df = prepare_data_and_anova(df)
    
    # Step 2: Stratified Folds and Context Generation
    processed_folds = generate_folds(df, df1)
    
    # Step 3: Run Model Selection (Block 2)
    run_model_selection(processed_folds)
    
    # Step 4: Nested CV & Ablation with JSON Caching (Block 2.5 & 3)
    run_nested_cv_ablation(processed_folds)
    
    execute_shap(processed_folds)

if __name__ == "__main__":
    main()