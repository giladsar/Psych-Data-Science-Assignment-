

# 🧠 Depression Prediction — Final Solution (Psych Data Science Lab @ BGU, 2025)

This repository contains my **final working solution** submitted to the **Kaggle Depression Prediction** competition  
([competition page](https://www.kaggle.com/competitions/depression-prediction)).

The entire pipeline — from data preprocessing and feature engineering to model training, evaluation, and submission generation —  
is implemented in a single Quarto file:

> **`Psych-Data-Science-Assignment-.qmd`**

---

## 🎯 Objective

Predict participants’ **depression scores (CES-D scale: 0–60)** from:
- Linguistic data from a **Free Association Task** (10 chains × 10 words)
- Demographic information and basic lexical/semantic features

The goal was to model how **chains of spontaneous thought** relate to depressive symptomatology.

---

## ⚙️ Technical Overview

The `.qmd` file reproduces the **entire end-to-end solution** used for the Kaggle submission:

### 1. Data Preparation
- Loads Kaggle training and test datasets.
- Cleans and merges text and demographic data.
- Handles missing values and encodes categorical predictors.
- Normalizes numeric features and aligns variable types across datasets.

### 2. Feature Engineering
- Extracts lexical, semantic, and sentiment features from word associations.
- Computes embedding-based similarity metrics between prompt and responses.
- Aggregates features across association chains (mean, variance, cosine distance).
- Adds demographic features (age, native language, gender, etc.).

### 3. Modeling
- Trains several regression models:
  - **Random Forest**
  - **XGBoost**
- Uses cross-validation for model comparison and tuning.
- Evaluates performance using **RMSE**, **MAE**, and **R²**.

### 4. Final Submission
- The best-performing model (after tuning and CV) is retrained on the full training set.
- Generates `predictions.csv` in the exact format required by Kaggle.
- Results correspond to the **final private leaderboard submission**.

---

## 🗂 Repository Structure

```

.  
├── Psych-Data-Science-Assignment-.qmd # Full, reproducible solution (run this file only)  
├── data/ # Train/test data (not uploaded per Kaggle policy)  
├── predictions/ # Output predictions (Kaggle submission files)  
├── rds/ # Serialized R objects (models, processed data)  
├── references/ # Literature and supporting notes  
├── .gitignore  
└── README.md

```

> ⚠️ Only `Psych-Data-Science-Assignment-.qmd` needs to be run.  

---

## 🧮 How to Run

### Requirements
- **R ≥ 4.3**
- **Quarto** (for rendering `.qmd`)

### Install dependencies
```r
install.packages(c(
  "tidyverse", "tidymodels", "data.table", 
  "quanteda", "text", "embed", "xgboost", 
  "randomForest", "glmnet", "rmarkdown"
))
```

### Run / Render

1. Place the Kaggle data files inside `data/`  
    (e.g., `train.csv`, `test.csv`).
    
2. Open `Psych-Data-Science-Assignment-.qmd` in RStudio.
    
3. Run interactively **or** render:
    
    ```bash
    quarto render Psych-Data-Science-Assignment-.qmd
    ```
    

* * *

📊 Results
----------

* Achieved **competitive leaderboard performance** using optimized ensemble models.
    
* Final model balanced interpretability and accuracy, emphasizing linguistic coherence and affective tone as key predictors.
    
* Outputs include:
    
    * **Feature importance plots**
        
    * **Cross-validation metrics**
        
    * **Final Kaggle submission file** (`predictions.csv`)
        

* * *

🧠 Insights
-----------

This project demonstrates that:

* Spontaneous thought patterns encode measurable markers of mental health.
    
* Machine learning can capture **semantic drift** and **affective cues** predictive of depression.
    
* Interpretable ML methods help bridge psychological theory and predictive modeling.
    

* * *

📘 Citation
-----------

> Sarusi, G. (2025). _Depression Prediction: Final Kaggle Solution._  
> Psych Data Science Lab, Ben-Gurion University of the Negev.

* * *

✉️ Contact
----------

**Author:** Gilad Sarusi  
**Affiliation:** Psych Data Science Lab, Ben-Gurion University  
**GitHub:** [@giladsar](https://github.com/giladsar)

* * *

> “From free associations to depression scores —  
> this solution turns spontaneous thought into structured prediction.”


