# depression_data_science_1.R
#
# This script implements the depression score prediction pipeline for the Kaggle
# "Psych Data Science Lab @ BGU 2025" competition.  It extends the baseline
# code by adding a 70/30 initial split and cross‑validated model tuning.
# The pipeline builds four text‑derived features:
#   1. Negative affect: rolling cosine similarity of word embeddings to a negative lexicon.
#   2. Absolutist language: rolling cosine similarity of word embeddings to an absolutist lexicon.
#   3. Creativity/association: average Euclidean distance between adjacent words within each prompt block.
#   4. Theme diversity: average Euclidean distance from the prompt word to its block words.
# These features are joined to demographic variables and used to train and evaluate
# two models: ridge regression and XGBoost.  Hyperparameters are tuned with
# 10‑fold cross‑validation on the training split, and the final models are fit on
# the full training data.  Predictions for the held‑out validation set and
# external test set are exported with the required format (columns `ID` and
# `matrixScore`).

library(stringr)
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(tidymodels)
library(glmnet)
library(xgboost)
library(quanteda)
library(embed)
library(embedplyr)
library(zoo)

# Prefer tidymodels versions of functions like select_best
tidymodels_prefer()

set.seed(64)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
# Read the competition data.  The training file contains CES‑D depression scores
# (matrixScore) and 100 free association words per participant in long format.
# The test file contains the same structure but lacks matrixScore.  We collapse
# the long format into one row per participant with a single string of 100 words.

train_raw <- read_csv("ds_train_full.csv") %>%
  unite("race", starts_with("race_"), sep = ";", na.rm = TRUE, remove = TRUE)

test_raw <- read_csv("ds_test_full.csv") %>%
  unite("race", starts_with("race_"), sep = ";", na.rm = TRUE, remove = TRUE)

# Collapse each participant's 100 words into a single text string
train_docs <- train_raw %>%
  group_by(ID) %>%
  summarise(text = paste(word, collapse = " "), .groups = "drop")

train_data <- train_raw %>%
  distinct(ID, Age, Native_speaker, sex, school, Income, race, matrixScore) %>%
  left_join(train_docs, by = "ID")

test_docs <- test_raw %>%
  group_by(ID) %>%
  summarise(text = paste(word, collapse = " "), .groups = "drop")

test_data <- test_raw %>%
  distinct(ID, Age, Native_speaker, sex, school, Income, race) %>%
  left_join(test_docs, by = "ID")

# ---------------------------------------------------------------------------
# 2. Load embeddings and define lexica
# ---------------------------------------------------------------------------
# We use the GoogleNews word2vec embeddings (300 dimensions).  Only the tokens
# appearing in the training data are loaded to save memory.  The negative and
# absolutist lexica are defined as character vectors.  Averages of their
# embeddings (DDR) will represent the respective constructs.

# Build vocabulary from training tokens
train_corp <- corpus(train_docs, docid_field = "ID", text_field = "text")
train_tokens <- tokens(train_corp, remove_punct = TRUE) %>% tokens_tolower()
train_dfm <- dfm(train_tokens)
tokens_ready <- featnames(train_dfm)

# Load pretrained embeddings; restrict to training vocabulary for efficiency
word2vec_mod <- load_embeddings(
  "GoogleNews.vectors.negative300",
  words  = tokens_ready,
  format = "rds"
)

# Negative affect lexicon (30 words)
negative_words <- c(
  "sad","unhappy","miserable","hopeless","worthless","helpless",
  "lonely","guilty","ashamed","tired","exhausted","tearful",
  "cry","pain","hurt","hate","afraid","scared","anxious",
  "nervous","worried","gloomy","depressed","despair","failure",
  "useless","empty","broken","angry","frustrated"
)

# Absolutist lexicon (30 words)
absolutist_words <- c(
  "always","never","everyone","noone","nobody","everybody","everything","nothing",
  "only","all","none","must","cannot","should","totally","completely","absolutely",
  "entirely","entire","forever","definitely","constantly","endless","perfect",
  "impossible","utterly","sure","certain","every","ever"
)

# Compute Distributed Dictionary Representations (DDRs) by averaging embeddings
negative_ddr    <- predict(word2vec_mod, negative_words) |> average_embedding()
absolutist_ddr  <- predict(word2vec_mod, absolutist_words) |> average_embedding()

# ---------------------------------------------------------------------------
# 3. Helper functions for text features
# ---------------------------------------------------------------------------

PROMPTS <- c( #create a vector with the prompt word names
  "log","orb","extension","synthesizer","treadmill",
  "valet","toss","testimony","mow","forge"
)


# Rolling DDR curve: computes cosine similarity between rolling averages of word
# embeddings and a DDR vector.  Returns a numeric vector of length equal to the
# number of tokens in the input (with NA padding at the edges).
rolling_ddr_curve <- function(txt, ddr_vec, width = 4) {
  toks <- tokens(
    txt,
    what = "word",
    remove_punct = TRUE,
    remove_symbols = TRUE,
    remove_url = TRUE
  ) |> as.character() |> tolower()
  if (!length(toks)) return(numeric(0))
  # Rolling average of embeddings
  E_roll <- predict(word2vec_mod, toks, .keep_missing = TRUE) |>
    zoo::rollapply(
      width    = width,
      FUN      = mean,
      na.rm    = TRUE,
      by.column = TRUE,
      align    = "center"
    )
  sims <- E_roll |>
    as.embeddings() |>
    get_sims(list(score = ddr_vec), method = "cosine_squished") |>
    dplyr::pull(score)
  if (length(sims) < length(toks)) {
    pad <- length(toks) - length(sims)
    left_pad <- floor(pad / 2)
    sims <- c(rep(NA_real_, left_pad), sims, rep(NA_real_, pad - left_pad))
  }
  sims
}

# Tokenize the first 100 words (10 prompts * 10 words) and split into blocks
tokenize_100 <- function(txt) {
  quanteda::tokens(
    txt, what = "word",
    remove_punct = TRUE, remove_symbols = TRUE, remove_url = TRUE
  ) |> as.character() |> tolower() |>
    (
      function(w) {
        w[seq_len(min(length(w), length(PROMPTS) * 10))]
      }
    )()
}

split_into_10x10 <- function(words, prompts = PROMPTS, chunk_size = 10) {
  if (length(words) == 0) {
    out <- replicate(length(prompts), character(0), simplify = FALSE)
    names(out) <- prompts
    return(out)
  }
  idx <- ceiling(seq_along(words) / chunk_size)
  chunks <- split(words, idx)
  if (length(chunks) < length(prompts)) {
    for (i in (length(chunks) + 1):length(prompts)) chunks[[i]] <- character(0)
  }
  names(chunks) <- prompts
  chunks
}

# Creativity feature: average Euclidean distance between consecutive embeddings within each prompt block.
creativity_from_text <- function(txt) {
  words  <- tokenize_100(txt)
  blocks <- split_into_10x10(words, PROMPTS, 10)
  per_prompt <- purrr::map_dbl(names(blocks), function(pr) {
    chain <- c(pr, blocks[[pr]])
    if (length(chain) < 2) return(NA_real_)
    E <- predict(word2vec_mod, chain, .keep_missing = TRUE)
    if (nrow(E) < 2) return(NA_real_)
    d <- vapply(1:(nrow(E) - 1), function(i) {
      as.embeddings(E[i, , drop = FALSE]) |>
        get_sims(list(next_word = E[i + 1, ]), method = "euclidean") |>
        dplyr::pull(next_word)
    }, numeric(1))
    mean(d, na.rm = TRUE)
  })
  out <- mean(per_prompt, na.rm = TRUE)
  if (is.nan(out)) NA_real_ else out
}

# Theme diversity feature: average Euclidean distance from the prompt word to each word in its block.
theme_diversity_from_text <- function(txt) {
  words  <- tokenize_100(txt)
  blocks <- split_into_10x10(words, PROMPTS, 10)
  per_prompt <- purrr::map_dbl(names(blocks), function(pr) {
    w <- blocks[[pr]]
    if (length(w) == 0) return(NA_real_)
    E <- predict(word2vec_mod, c(pr, w), .keep_missing = TRUE)
    if (nrow(E) < 2) return(NA_real_)
    d <- as.embeddings(E[-1, , drop = FALSE]) |>
      get_sims(list(prompt_vec = E[1, ]), method = "euclidean") |>
      dplyr::pull(prompt_vec)
    mean(d, na.rm = TRUE)
  })
  out <- mean(per_prompt, na.rm = TRUE)
  if (is.nan(out)) NA_real_ else out
}

# Compute text features for a data frame with columns ID and text.
compute_features <- function(doc_tbl) {
  doc_tbl %>%
    transmute(
      ID,
      neg_ddr_roll_mean = map_dbl(
        text,
        ~ {
          v <- rolling_ddr_curve(.x, negative_ddr)
          if (!length(v)) NA_real_ else mean(v, na.rm = TRUE)
        }
      ),
      abs_ddr_roll_mean = map_dbl(
        text,
        ~ {
          v <- rolling_ddr_curve(.x, absolutist_ddr)
          if (!length(v)) NA_real_ else mean(v, na.rm = TRUE)
        }
      ),
      theme_diversity_euclid  = map_dbl(text, theme_diversity_from_text),
      creativity_assoc_euclid = map_dbl(text, creativity_from_text)
    )
}

# ---------------------------------------------------------------------------
# 4. Feature engineering for train and test
# ---------------------------------------------------------------------------

train_feats <- compute_features(train_docs)
test_feats  <- compute_features(test_docs)

train_data <- train_data %>%
  left_join(train_feats, by = "ID")

test_data <- test_data %>%
  left_join(test_feats, by = "ID")

# ---------------------------------------------------------------------------
# 5. Initial split and cross‑validation
# ---------------------------------------------------------------------------
# Create a 70/30 split of the training data for model assessment.  The training
# split is used for cross‑validated hyperparameter tuning and model fitting.
train_data <- train_data %>% 
  Income = if_else(is.na(Income), "$90,000 to $99,999", Income)
 %>% 
  mutate(
    # מין
    sex = factor(sex),
    
    # השכלה - בסדר עולה
    school = factor(
      school, 
      levels = education_levels, 
      ordered = TRUE
    ),
    
    # הכנסה - בסדר עולה
    Income = factor(
      Income, 
      levels = income_range_levels, 
      ordered = TRUE
    ),
    
    # Native speaker
    Native_speaker = factor(Native_speaker)
  ) %>% 
  # הוספת עמודה מספרית להכנסה (midpoint)
  mutate(
    Income_num = case_when(
      Income == "Less than $10,000"      ~ 5000,
      Income == "$10,000 to $19,999"     ~ 15000,
      Income == "$20,000 to $29,999"     ~ 25000,
      Income == "$30,000 to $39,999"     ~ 35000,
      Income == "$40,000 to $49,999"     ~ 45000,
      Income == "$50,000 to $59,999"     ~ 55000,
      Income == "$60,000 to $69,999"     ~ 65000,
      Income == "$70,000 to $79,999"     ~ 75000,
      Income == "$80,000 to $89,999"     ~ 85000,
      Income == "$90,000 to $99,999"     ~ 95000,
      Income == "$100,000 to $149,999"   ~ 125000,
      Income == "$150,000 or more"       ~ 175000,
      TRUE ~ NA_real_
    )
  )

split <- initial_split(train_data, prop = 0.7, strata = matrixScore)
train_split <- training(split)
validation_split <- testing(split)

# Ten‑fold cross‑validation for tuning on the training split
cv_folds <- vfold_cv(train_split, v = 10, strata = matrixScore)

# ---------------------------------------------------------------------------
# 6. Set up Ridge regression pipeline
# ---------------------------------------------------------------------------

ridge_recipe <- recipe(
  matrixScore ~ Age + Native_speaker + sex + school + Income + race +
    neg_ddr_roll_mean + abs_ddr_roll_mean +
    theme_diversity_euclid + creativity_assoc_euclid,
  data = train_data
) %>%
  # don’t model on raw text or prompt columns
  step_rm(any_of(c("ID", "text", PROMPTS))) %>%
  
  # ensure Age is numeric (if it was read as character)
  step_mutate(Age = as.numeric(Age)) %>%
  
  # handle low-variance issues early
  step_nzv(all_predictors()) %>%
  step_zv(all_predictors()) %>%
  
  # make unseen/missing levels safe for new data
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "UNKNOWN") %>%
  step_interact(~ 
                  # Demographic × Demographic
                  Age:Income +
                  Age:sex +
                  Age:race +
                  Income:sex +
                  Income:race +
                  sex:race 
  ) %>%
  # now dummy-encode factors (after interactions so they expand properly)
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # numeric hygiene & scaling for glmnet
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())


ridge_spec <- linear_reg(mode = "regression", engine = "glmnet", penalty = tune(), mixture = 0)

ridge_workflow <- workflow() %>%
  add_model(ridge_spec) %>%
  add_recipe(ridge_recipe)

ridge_grid <- grid_regular(penalty(range = c(-2, 7)), levels = 20)

# Metric set: Pearson correlation; rmse and rsq for reference
metric_fns <- metric_set(yardstick::rmse, yardstick::rsq)

# Tune ridge with cross‑validation
ridge_tuned <- tune_grid(
  ridge_workflow,
  resamples = cv_folds,
  grid = ridge_grid,
  metrics = metric_fns
)

best_ridge <- select_best(ridge_tuned, metric = "rsq")

# Fit on full training split
ridge_final_wf <- finalize_workflow(ridge_workflow, best_ridge)
ridge_fit_split <- fit(ridge_final_wf, data = train_split)

# ---------------------------------------------------------------------------
# 7. Set up XGBoost pipeline
# ---------------------------------------------------------------------------

boost_recipe <- recipe(
  matrixScore ~ Age + Native_speaker + sex + school + Income + race +
    neg_ddr_roll_mean + abs_ddr_roll_mean +
    theme_diversity_euclid + creativity_assoc_euclid,
  data = train_split
) %>%
  step_rm(any_of(c("ID", "text", PROMPTS))) %>%
  step_mutate(Age = as.numeric(Age), Income = as.numeric(Income)) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "UNKNOWN") %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors())

boost_spec <- boost_tree(
  mode = "regression",
  engine = "xgboost",
  tree_depth    = tune(),
  learn_rate    = 0.1,
  trees         = tune(),
  min_n         = 5,
  loss_reduction = 0,
  mtry          = 5,
  sample_size   = 1
)

boost_workflow <- workflow() %>%
  add_model(boost_spec) %>%
  add_recipe(boost_recipe)

boost_grid <- expand.grid(
  tree_depth = c(1, 3, 5),
  trees      = c(100, 500, 1000)
)

boost_tuned <- tune_grid(
  boost_workflow,
  resamples = cv_folds,
  grid      = boost_grid,
  metrics   = metric_fns
)

(best_boost <- select_best(boost_tuned, metric = "rsq"))

boost_final_wf <- finalize_workflow(boost_workflow, best_boost)
boost_fit_split <- fit(boost_final_wf, data = train_split)

# ---------------------------------------------------------------------------
# 8. Evaluate on hold‑out validation split
# ---------------------------------------------------------------------------

ridge_val_pred <- predict(ridge_fit_split, new_data = validation_split) %>%
  bind_cols(validation_split %>% select(matrixScore)) %>%
  rename(pred = .pred)

boost_val_pred <- predict(boost_fit_split, new_data = validation_split) %>%
  bind_cols(validation_split %>% select(matrixScore)) %>%
  rename(pred = .pred)

(ridge_val_cor <- cor(ridge_val_pred$matrixScore, ridge_val_pred$pred, use = "complete.obs"))
(boost_val_cor <- cor(boost_val_pred$matrixScore, boost_val_pred$pred, use = "complete.obs"))

cat("Ridge validation correlation:", ridge_val_cor, "\n")
cat("XGBoost validation correlation:", boost_val_cor, "\n")

# ---------------------------------------------------------------------------
# 9. Fit final models on full training data and predict test
# ---------------------------------------------------------------------------

ridge_final <- fit(ridge_final_wf, data = train_data)
boost_final <- fit(boost_final_wf, data = train_data)

pred_test_ridge <- augment(ridge_final, new_data = test_data) %>%
  transmute(ID, matrixScore = .pred) %>%
  arrange(ID)

pred_test_xgboost <- augment(boost_final, new_data = test_data) %>%
  transmute(ID, matrixScore = .pred) %>%
  arrange(ID)

# Write prediction files with correct header
write_csv(pred_test_ridge, "predictions_test_ridge.csv")
write_csv(pred_test_xgboost, "predictions_test_xgboost.csv")

cat("Prediction files written: predictions_test_ridge.csv and predictions_test_xgboost.csv\n")
