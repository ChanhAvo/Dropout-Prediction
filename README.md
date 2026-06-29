# **Student Dropout Prediction System**

A machine learning system that predicts student dropout risk using machine learning algorithms with contextual major-level and interaction features. The pipeline includes data preprocessing, model selection, nested cross-validation, ablation study, SHAP explainability, and a Streamlit web application.


## **Project Structure**

```
DropoutPrediction/
├── data/
│   ├── student_data.csv
│   └── major.csv
├── artifacts/
│   ├── catboost_m4_model.cbm
│   ├── encoder_maps.json
│   ├── m4_feature_cols.json
│   ├── major_context_db.csv
│   ├── major_score_stats.json
│   └── admission_score_stats.json
├── ablation.py
├── app.py
├── config.py
├── evaluation.py
├── export.py
├── main.py
├── preprocessing.py
├── shap_analysis.py
├── tuned_hyperparameters.json
└── requirements.txt
```                

## **File Descriptions**

### **Input Data**

| File | Description |
| ----- | ----- |
| data/student\_data.csv | Raw student records containing entrance exam scores, demographics, scholarship type, and graduation/dropout status |
| data/major.csv | Curriculum metadata per major: credits required, course load, STEM flag, math intensity, etc |

### **Source Files**

#### **config.py \- Central Configuration**

Defines all feature column groups used across the pipeline:

* categorical\_features: Gender, Admission, Region, HighSchoolType (ordinal encoded)  
* entrance\_cols: 9 base features from application records (M0 baseline)  
* curriculum\_cols: 5 curriculum features from major.csv (added in M2)  
* dist\_cols: 9 major-level distributional features computed during folding (added in M3)  
* interaction\_cols: 6 engineered interaction features (added in M4)  
* CAT\_FEATURE\_INDICES: Integer indices of categoricals within entrance\_cols, used by CatBoost  
* candidate\_models: All 9 baseline classifiers for Block 2 model selection  
* tuning\_setup: Hyperparameter grids for Block 3 nested CV (Random Forest, XGBoost, LightGBM, CatBoost)

#### **preprocessing.py \- Data Preparation and Fold Generation**

* Encodes the target column (Status) as binary: Graduated: 0, Drop out: 1  
* Ordinal-encodes ScholarshipType   
* Runs one-way ANOVA across majors and computes the Intraclass Correlation Coefficient (ICC) to quantify how much dropout rate varies by major  
* Creates 15 folds using RepeatedStratifiedKFold (5 splits × 3 repeats), stratified by Major × Status combinations  
* For each fold, independently fits an OrdinalEncoder on the training split to avoid data leakage  
* Standardises EntranceExamScore per admission type using train-only statistics  
* Computes a major context feature block from training data only: mean/std of entrance scores, gender ratio, scholarship ratio, priority ratio, language score, and diversity entropies (region, admission type, high school type)  
* Merges major context with curriculum data from major.csv  
* Engineers interaction features: ScoreDev, ScoreZ, ScholarshipSelect, PrioritySelect, LangScoreDev, ScorePercentile  
* Returns a list of 15 fold dictionaries, each containing train, test, and encoder

#### **ablation.py \- Feature Set Construction**

* Builds 5 incremental feature matrices (M0–M4) for ablation study:  
* Returns a dictionary keyed by tier name, each containing X\_train and X\_test DataFrames

#### **evaluation.py \- Model Evaluation**

* Evaluates all 9 candidate classifiers from config.py using inner 3-fold CV within each outer fold  
* Applies class-weight balancing (calculated per fold from train class ratio)  
* Reports mean ± std of AUC, Recall, Precision, F1 across 3 repeats  
* Runs nested CV for each algorithm in tuning\_setup across all 5 tiers (M0–M4)  
* Uses GridSearchCV (inner 3-fold, scoring=F1) for hyperparameter tuning  
* JSON caching: saves best hyperparameters to tuned\_hyperparameters.json after each fold; reloads on subsequent runs to avoid redundant computation  
* Reports mean ± corrected SE (Nadeau-Bengio correction) for all metrics per tier  
* Prints most-frequent hyperparameter configurations for M0 and M4  
* Performs paired t-tests (corrected) comparing M4 vs each baseline tier

#### **shap\_analysis.py \- Explainability**

* Re-trains CatBoost on each fold using cached hyperparameters from tuned\_hyperparameters.json  
* Computes SHAP values using shap.TreeExplainer for all 15 folds per tier  
* Aggregates SHAP values and test sets across folds  
* Generates and saves beeswarm plots for top 15 features per tier:  
  * shap\_beeswarm\_catboost\_m0.png through shap\_beeswarm\_catboost\_m4.png  
* Feature names are cleaned to human-readable labels before plotting

#### **export.py \- Production Model Export**

* Loads and preprocesses data using the same pipeline as training  
* Uses Fold 0 as the representative training partition  
* Saves 5 artefacts required by app.py

| Files | Description |
| ----- | ----- |
| catboost\_m4\_model.cbm | Trained CatBoost M4 model in CatBoost binary format |
| encoder\_maps.json | Maps raw string values → integer codes for each categorical feature |
| major\_context\_db.csv | One row per major with all context and curriculum features |
| major\_score\_stats.json | Per-major lists of standardised entrance scores (for ScorePercentile) |
| admission\_score\_stats.json | Mean and std of entrance scores per admission type (for standardisation) |

* Loads optimal M4 hyperparameters from tuned\_hyperparameters.json (most frequent across folds)

#### **main.py \- Full Training Pipeline Entry Point**

Orchestrates the complete research pipeline in sequence:

1. Load student\_data.csv and major.csv  
2. prepare\_data\_and\_anova():  encode targets, run ANOVA/ICC  
3. generate\_folds():  create 15 stratified folds with all features  
4. run\_model\_selection(): compare 9 classifiers  
5. run\_nested\_cv\_ablation(): tune and ablate CatBoost across M0–M4  
6. execute\_shap(): generate SHAP beeswarm plots

**Note:** main.py is for research and training only. To deploy the web application, run export.py first, then app.py.

#### **app.py \- Streamlit Web Application**

Provides a browser-based inference interface:

* Loads catboost\_m4\_model.cbm, encoder\_maps.json, major\_context\_db.csv, major\_score\_stats.json, and admission\_score\_stats.json  
* Accepts user input for student attributes (major, entrance score, gender, scholarship, etc.)  
* Replicates the full M4 feature engineering pipeline at inference time (standardisation, context lookup, interaction features)  
* Outputs dropout probability and classification result

## **Development Environment**

* **Python version:** 3.9 or higher  
* **Operating System:** Windows 10/11, macOS, or Linux  
* **RAM:** Minimum 8 GB recommended (nested CV with 15 folds is memory-intensive)

## **Installation**

### **Step 1: Extract the project**

git clone https://github.com/\<your\_repo\>/Dropout-Prediction.git  
cd DropoutPrediction


### **Step 2: Create a virtual environment**

\# Create virtual environment  
python \-m venv .venv

\# Activate (macOS / Linux)  
source .venv/bin/activate

\# Activate (Windows)  
.venv\\Scripts\\activate

### **Step 3: Install dependencies**

pip install \-r requirements.txt

If you encounter LightGBM or CatBoost build errors on macOS, install via conda: conda install \-c conda-forge lightgbm catboost

## **How to Run**

### **A: Run the Web Application** 

The trained model artefacts are already included in the submission. You can launch the app directly:

streamlit run app.py

Then open http://localhost:8501 in your browser.

### **B: Re-train the Model from Scratch**

Follow these steps in order:

#### **Step 1: Run the full training and evaluation pipeline**

python main.py

This will:

* Preprocess the data and print ANOVA/ICC results  
* Generate 15 stratified folds  
* Run Block 2 model selection across 9 classifiers  
* Run Block 3 nested CV and ablation for M0–M4 (this step is slow; \~30 minutes on first run)  
* Save hyperparameters to tuned\_hyperparameters.json progressively  
* Generate SHAP beeswarm plots (shap\_beeswarm\_catboost\_m\*.png)

**Subsequent runs are fast** \- if tuned\_hyperparameters.json already exists, all 15 × 4 GridSearch calls are skipped and cached parameters are loaded instead.

#### **Step 2: Export the production model**

python export.py

This retrains CatBoost M4 using the best hyperparameters from the cache and saves all 5 artefacts required by app.py.

#### **Step 3: Launch the web application**

streamlit run app.py

## **Demo**

https://github.com/user-attachments/assets/a8b5f725-7b8c-444b-8e77-4c1d2dbf86b2


## **Troubleshooting**

**FileNotFoundError: tuned\_hyperparameters.json** Run python main.py first. This file is generated during Block 3 nested CV.

**FileNotFoundError: catboost\_m4\_model.cbm** Run python export.py after main.py completes. The model file is generated by the export step.

**SHAP plots not generated** SHAP analysis runs as the final step in main.py. Ensure main.py completes fully and tuned\_hyperparameters.json contains CatBoost M4 entries.

**CatBoost categorical feature errors** CatBoost requires categorical columns to be passed as strings. This conversion is handled automatically in evaluation.py, shap\_analysis.py, and export.py. Do not modify the data types manually before passing to the pipeline.
