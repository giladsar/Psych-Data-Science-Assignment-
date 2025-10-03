

Psych Data Science Assignment @ BGU (2025)
==========================================

**Course:** Psych Data Science Lab  
**Competition:** Kaggle — _Depression Prediction_  https://www.kaggle.com/competitions/depression-prediction

**Main entry point:** **`Psych-Data-Science-Assignment-.qmd`** (run only this file)

> Everything—data prep, feature engineering, modeling, evaluation, and submission—is contained in the Quarto file above.

Repository Layout
-----------------

```
.
├── Psych-Data-Science-Assignment-.qmd   # Single runnable analysis/report
├── data/                                # Train/test data
├── predictions/                         # Generated Kaggle submissions (CSV)
├── rds/                                 # Serialized objects (models, processed data)
├── references/                          # Notes, papers, citations
├── .gitignore
└── README.md
```

What the pipeline does (at a glance)
------------------------------------

* Loads and cleans the provided datasets
    
* Engineers linguistic & demographic features for depression prediction
    
* Trains and compares ML models (e.g., tree-based + baselines)
    
* Evaluates via cross-validation
    
* Exports **`predictions.csv`** in the required Kaggle format
    
* Renders a self-contained report with figures and tables
    

> This description reflects the repository structure and your stated goal for the Kaggle task. If you paste the model section from the `.qmd`, I’ll list the exact algorithms, hyperparameters, and metrics.

How to run
----------

### 1) Prerequisites

* **R ≥ 4.3**
    
* **Quarto** installed (CLI or via RStudio)
    

### 2) Install R packages

```r
install.packages(c(
  "tidyverse", "tidymodels", "data.table",
  "quanteda", "text", "embed",
  "xgboost", "randomForest", "glmnet",
  "rmarkdown"
))
```

> If the `.qmd` uses other packages (e.g., `vip`, `yardstick`, `readr`, `janitor`), add them to the list—happy to pin versions if you share your session info.

### 3) Provide data

* Place the competition data in `./data/` with the filenames expected by the `.qmd` (train/test).
    
* Keep raw data out of Git if competition rules restrict sharing.
    

### 4) Run / Render

* **Interactive (RStudio):** open `Psych-Data-Science-Assignment-.qmd` and run.
    
* **Render to HTML/PDF:**
    
    ```bash
    quarto render Psych-Data-Science-Assignment-.qmd
    ```
    
    (Use HTML unless your environment has LaTeX for PDF.)
    

Outputs
-------

* **`predictions/`**: Kaggle-ready CSVs (e.g., `predictions.csv`)
    
* **Rendered report** (same directory as `.qmd`): includes EDA figures, feature importances, CV metrics, and final leaderboard notes.
    
* **`rds/`**: optional serialized models / processed data for reproducibility.
    

Reproducibility notes
---------------------

* The `.qmd` should set a seed and isolate any preprocessing that must be fit **only on training** (to avoid leakage).
    
* Keep all file paths **relative** to the project root.
    
* If you want, I can add a minimal `renv` lockfile/section to snapshot package versions.
    

Citation
--------

> Sarusi, G. (2025). _Psych Data Science Assignment: Depression Prediction using Free Association Data._  
> Ben-Gurion University of the Negev.

Acknowledgments
---------------

* Kaggle _Depression Prediction_ competition.
    
* Course staff and peers at BGU’s Psych Data Science Lab.
    

* * *

### Why this structure / wording?

* It reflects your repo’s actual layout (main `.qmd` + `data/`, `predictions/`, `rds/`, `references/`).
    
* It makes the single-file entry explicit and hides legacy `.R` scripts.
    
* It’s ready for recruiters/TA’s: quick “How to run”, outputs, and reproducibility.
    

If you paste the top of the `.qmd` (YAML header + model/training chunks), I’ll immediately swap the generic parts for your exact models, metrics (MAE/RMSE/R²), figures, and the precise submission column names.
