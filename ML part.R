skim(training_set)
hist(training_set$Age)
View(training_set)
df_reduced <- training_set %>%
  mutate(
    Income = if_else(is.na(Income), "$90,000 to $99,999", Income) #its the avarge 
  ) %>% 
  mutate(
    sex = factor(sex),
    school = factor(
      school, 
      levels = education_levels, 
      ordered = TRUE
    ),
    Income = factor(
      Income, 
      levels = income_range_levels, 
      ordered = TRUE
    ),
    Native_speaker = factor(Native_speaker)
  ) %>% 
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
  ) %>% 
  select(
    -c(
      race_1, race_2, race_3, race_4, race_5, race_6, 
      race_6_TEXT, word, wordId, Native_speaker, school,Income
    )
  ) %>% 
  filter(q_num < 2, 
         cycle < 2)%>% 
  select(
    -c(q_num,cycle))

training_set_clean <- df_reduced %>%
  select(1:8, 19:20)
skim(training_set_clean)
View(training_set_clean)
set.seed(42)
splits <- initial_split(training_set_clean, prop = 0.7, strata = matrixScore)
dep.train <- training(splits)
dep.test <- testing(splits)


dep.tune_splits <- vfold_cv(dep.train, v = 5) 
dep.comp_splits <- vfold_cv(dep.test, v = 10)
mset_reg <- metric_set(rsq, mae)

rec <- recipe(matrixScore ~ .,
              data = dep.train) %>% 
  step_rm("ID") 

rf_spec <- rand_forest(
  mode = "regression", engine = "randomForest",
  mtry = tune(),
  trees = 75
)
rf_wf <- workflow(preprocessor = rec, spec = rf_spec)

rf_grid <- expand_grid(
  mtry = c(1, 2, 3, 4, 5, 6, 7, 8)
)

rf_tuner <- tune_grid(rf_wf,
                      resamples = dep.tune_splits,
                      grid = rf_grid,
                      metrics = mset_reg,
                      control = control_grid(verbose = TRUE)
)
autoplot(rf_tuner)


tree_num <- select_by_one_std_err(rf_tuner, mtry, metric = "mae")

rf_fit <- rf_wf |>
  finalize_workflow(parameters = tree_num) |> 
  fit(data = dep.train)
rf_fit

augment(rf_fit, new_data = dep.test) |> mset_reg(matrixScore, .pred)

rf_eng <- extract_fit_engine(rf_fit)

vip::vip(rf_eng, method = "model", num_features = tree_num$mtry) +
  theme(legend.position = "bottom")

# -----------------------------------------------------------------------
rec2 <- recipe(matrixScore ~ ., data = dep.train) |>
  step_rm(ID) |>
  step_dummy(all_factor_predictors(), one_hot = TRUE) %>% 
  step_zv(all_predictors()) %>% 
  step_center(all_numeric_predictors()) |> 
  step_scale(all_numeric_predictors())

enet_spec <- linear_reg(
  mode = "regression", engine = "glmnet", 
  penalty = tune(), mixture = tune()
)

enet_wf <- workflow(preprocessor = rec2, spec = enet_spec)
enet_grid <- grid_regular(
  penalty(range = c(-4, 7)),
  mixture(),
  
  levels = c(15, 11)
)


enet_tuned <- tune_grid(
  enet_wf,
  resamples = dep.tune_splits,
  grid = enet_grid,
  # metrics = mset_reg,
  control = control_grid(verbose = TRUE)
)


autoplot(enet_tuned) + 
  scale_x_continuous(transform = scales::transform_log(),
                     breaks = scales::breaks_log(n = 5),
                     labels = scales::label_number(big.mark = ",")) + 
  theme(axis.text.x = element_text(angle = 20))


best_enet <- select_best(enet_tuned, metric = "rmse")
best_enet
enet_fit <- enet_wf |> 
  finalize_workflow(parameters = best_enet) |> 
  fit(data = dep.train)


augment(enet_fit, new_data = dep.test) |> mset_reg(matrixScore, .pred)


plot_glmnet_coef <- function(mod, s = 0, show_intercept = FALSE) {
  b <- glmnet::coef.glmnet(mod, s = c(s, 0), exact = FALSE) |> 
    as.matrix() |> as.data.frame() |> 
    tibble::rownames_to_column("Coef")
  
  if (isFALSE(show_intercept)) {
    b <- b |> dplyr::filter(Coef != "(Intercept)")
  }
  
  
  ggplot2::ggplot(b, ggplot2::aes(Coef, s1)) + 
    ggplot2::geom_hline(yintercept = 0) + 
    ggplot2::geom_point(ggplot2::aes(shape = s1 == 0), fill = "red", size = 2, 
                        show.legend = c(shape = TRUE)) + 
    ggplot2::scale_shape_manual(NULL, 
                                breaks = c(FALSE, TRUE), values = c(16, 24),
                                labels = c("none-0", "0"), 
                                limits = c(FALSE, TRUE)) + 
    ggplot2::scale_x_discrete(guide = ggplot2::guide_axis(angle = 30)) + 
    ggplot2::coord_cartesian(ylim = range(b[,-1])) + 
    ggplot2::labs(y = "Coef", x = NULL) + 
    ggplot2::ggtitle(bquote(lambda==.(s)))
}

enet_eng <- extract_fit_engine(enet_fit)
plot_glmnet_coef(enet_eng, s = best_enet$penalty)

vip::vip(enet_fit, method = "model", lambda = best_enet$penalty, 
         num_features = 20,
         mapping = aes(fill = Sign)) + 
  theme(legend.position = "right")


dep.test_rf.pred <- augment(rf_fit, dep.test)
dep.test_enet.pred <- augment(enet_fit, dep.test)

bind_rows(
  "rf" = dep.test_rf.pred, 
  "enet" = dep.test_enet.pred,
  .id = "Model"
) |> 
  group_by(Model) |> 
  mset_reg(matrixScore, .pred) #i didnt know how to write this part whith out the bind rows from the original code sorry its wired
p_dat <- bind_rows(
  "rf"   = dep.test_rf.pred   %>% mutate(resid = abs(matrixScore - .pred)),
  "enet" = dep.test_enet.pred %>% mutate(resid = abs(matrixScore - .pred)),
  .id = "Model"
)

ggplot(p_dat, aes(.pred, matrixScore, color = resid)) +
  geom_point(alpha = 0.7, size = 1.2) +
  geom_abline(slope = 1, intercept = 0) +
  coord_obs_pred() +
  facet_wrap(~ Model, ncol = 2) +
  labs(title = "Observed vs Predicted (test set)", x = "Predicted", y = "Observed") +
  scale_color_viridis_c(name = "|residual|") +
  theme(legend.position = "bottom")


rf_resamps <- fit_resamples(rf_fit,
                            resamples = dep.comp_splits,
                            metrics = mset_reg)

enet_resamps <- fit_resamples(enet_fit,
                              resamples = dep.comp_splits,
                              metrics = mset_reg)

ensemble_metrics <- bind_rows(
  "rf"   = collect_metrics(rf_resamps,   summarize = FALSE),
  "enet" = collect_metrics(enet_resamps, summarize = FALSE),
  .id = "Model"
) |>
  mutate(Model = factor(Model, levels = c("rf", "enet")))

ensemble_summary <- ensemble_metrics %>%
  group_by(Model, .metric) %>%
  summarise(mean = mean(.estimate), std_err = sd(.estimate), .groups = 'drop')

print(ensemble_summary)


