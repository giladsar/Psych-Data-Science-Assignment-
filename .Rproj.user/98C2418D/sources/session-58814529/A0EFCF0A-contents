
## Pre-Processing

#Packages loading:
library(tidyverse)
library(tidymodels)
library(finetune) # tuning hyper parameter
library(randomForest) # for rand forest 
library(xgboost) # booster
library(leaps)
library(vip) 

Tinder_df <- read.csv("./data/tinder-data/Tinder_Data_v3_Clean_Edition.csv")
#this data frame is containsing 1209 users and properties that define this user: numer of matchs, the median length of a conversation, number of swiping 'like' etc.

#First and foremost, we would get read of bad instances in our dataset. specifically those user whose hasn't been active at all 





Tinder_filtered <- Tinder_df  |> 
  
   
  #Omit users that didn't open the app at all, and those how haven't engaged in it 
  
  filter(sum_app_opens > 0 & swipe_likes >0 & swipe_passes > 0)  |> 
  
  # Or those users who never engaged in a conversation
  
  filter(medianConversationLength > 0)  |> 
  
  # Factor conversions

  mutate(instagram = as.factor(instagram),
         spotify = as.factor(spotify),
         gender= as.factor(gender)
         ) |> 
  
  #Creating an heterosexual column for easier interpretation. 
  
    mutate( hetero_sx =  as.factor(ifelse(
   (gender == "M" & interestedIn == "F") | (gender == "F" & interestedIn == "M"),
   1,
   0
  ))) |> 
  select(-c(interestedIn))  |> 
  
  #------ Dealing with weird outliers 

  # Some value in  'ageFilterMax' are set to 1000 (???), which is clearly unrealistic - will change it to range from 1 to 100. 
  # We'll correct these by replacing 1000 with the next-highest valid age. (assuming this is an arbitrary error [ while parsing the data maybe] rather than a invalid user. 
  
  
# Review of other user attributes (variable) of those instances supports this assumption)
  
  mutate(ageFilterMax = replace(
    ageFilterMax,
    ageFilterMax == max(ageFilterMax, na.rm = TRUE),
    sort(unique(ageFilterMax), decreasing = TRUE)[2]
  )) |> 
  
  #The same goes for our outcome variable 'nrOfGhostingsAfterInitialMessage'
  
    mutate(nrOfGhostingsAfterInitialMessage = replace(
    nrOfGhostingsAfterInitialMessage,
    nrOfGhostingsAfterInitialMessage == max(nrOfGhostingsAfterInitialMessage,
                                            na.rm = TRUE),
    sort(unique(nrOfGhostingsAfterInitialMessage),
         decreasing = TRUE)[2]
  ))


# there's still some outliers in our outcome variable. 
hist(Tinder_filtered$nrOfGhostingsAfterInitialMessage)

#lets try the Z score outliers detections - above 3 Standard deviation

z_score_no_ghst<- scale(Tinder_filtered$nrOfGhostingsAfterInitialMessage)

outliers <- Tinder_filtered$nrOfGhostingsAfterInitialMessage[abs(z_score_no_ghst) > 3]

length(outliers)

# only 23 instances are above/below 3 standard deviation from the average mean, I would regard those as outliers and therefore remove them 

Tinder_filtered <- Tinder_filtered |> 
  mutate(z_ghost = z_score_no_ghst) |>
  filter(abs(z_ghost) <= 3) |> 
  select(-z_ghost)
  
  

# Since my models of Interest are not an OLS/GKM ones, and given that I do want to interpret the results of such modeling, I would not conduct any dimension reduction method. 
# My models of choice would be Boosting and Random Forest. 



#Adding 'engineer' feature! - the Cosine squish to engineer in jobTitle

library(text)
library(quanteda)
library(embedplyr)
#First things First, Is to make our corpus, done using the jobTitle column.

 sum(Tinder_filtered$jobTitle == "unknown")
#[1] 576 
 
 nrow(Tinder_filtered) -  sum(Tinder_filtered$jobTitle == "unknown")
#[1] 335
 
 #That leaves us with only 335 informative users that hat a job Title 
 

 
 
 #------- Pre-processing---------- 
 
Tinder_filtered_for_txt <- Tinder_filtered |> 
   # Filtering out the 'unknown' users

  filter(jobTitle != "unknown")  |> 
   # Assuming that users with 'False' jobTitle are those that manually chose 'false' indicating that they are unemployed 
   mutate(jobTitle = ifelse(jobTitle == "False",
                 "unemployed",
                 jobTitle))
 
# Creating the corpus
Tinder_job_corpus <- Tinder_filtered_for_txt |>
  corpus(docid_field = "X_id",
         text_field="jobTitle")

 
# Convert to Tokens

# * The tokens are lower-cased! 

Tinder_job_tokens <- tokens(Tinder_job_corpus,
                         remove_punct = TRUE,
                         remove_symbols = T) |> 
  tokens_tolower()

# ------------ Should we Stem or Lemmas? 

# Since we're working with job titles—short phrases rather than full sentences with contextual information—we're less likely to encounter grammatical word variations (like verbs ending in "ing" or "ed").
# Therefore, I don’t see a strong reason to apply stemming or lemmatization in this case.


# --- Preperation to Word-Embedding 

Tinder_job_tokens_dfm <- dfm(Tinder_job_tokens)

#Getting the features from jobTitle. aka unique word.
Tinder_jobtitle_features <- featnames(Tinder_job_tokens_dfm)

# ----- Word Embedding\


#Since that our corpus is not Rich textually speaking we would use a rather small-traiend-model: glove.6B.50d - trained on Wikipedia 2014 + Giga word 5 

Tinder_jobTitle_WEmodel <- load_embeddings(
  "glove.6B.50d",
  words = Tinder_jobtitle_features,
  format = "rds",
  dir= "./rds")


# ./rds/glove.6B.50d.rds were created 

#Using this model, we would embedd our corpus (Tinder job titles):
Tinder_embbd_GloVe <- Tinder_job_tokens_dfm |> 
textstat_embedding(Tinder_jobTitle_WEmodel)

missing_words <- setdiff(Tinder_jobtitle_features, rownames(Tinder_jobTitle_WEmodel))

# 28 instance were dismissed - jobTitle was a typo. For example: "programmerare", "salesperon", "rechtsreferendar".


# ----------Cosine Squished ------

#Getting the directorial position of the word "Engineer"
engineer_embedding <- predict(Tinder_jobTitle_WEmodel,
                                "engineer") 


  
Tinder_distantce_engineer_WEGloVe <- Tinder_embbd_GloVe |> 
  get_sims(
    dim_1:dim_50, 
    list(engineer = engineer_embedding),
    method = "cosine_squished"
    )
  


#Let's add this distnace to our dataset and conduct the modeling! 

Tinder_filtered_for_txt_with_cosine <- Tinder_filtered_for_txt |> 
  merge(Tinder_distantce_engineer_WEGloVe,
        by.x ="X_id",
        by.y = "doc_id") |> 
  
  #Also, omitting the instances where jobTitle is typo - the only NA in 'engineer' 
  filter(!is.na(engineer))


#That's leaving us with 321 users! 



# Modeling

## Data Splitting (70%):
set.seed(130597)
splits <- initial_split(Tinder_filtered_for_txt_with_cosine,
                        prop = 0.7)
Tinder.train <- training(splits)
Tinder.test <- testing(splits)

# Cross Validation and Comparisons sets

# Bootstraps Spliting
Tinder_validation_sets <- bootstraps(
  Tinder.train,
  times = 50,
  strata = NULL,
  apparent = F
)

Tinder_comparison_sets <- bootstraps(
  Tinder.test,
  times = 50,
  strata = NULL,
  apparent = F
)

#the predictions would be evaluated using R2 and MAE

mset_reg <- metric_set(rsq, mae)

# ---------------- Recipe-------------------

# let's define the models R

# Predicting number of ghosting after initial message:

rec <- recipe(nrOfGhostingsAfterInitialMessage ~ .,
              data = Tinder.train) |> 
   #for our purposes we would omit some demographics that are not of high interests for us,   since we would like to examine patterns across nationalities and times.
  
   #after visually inspecting the data frame I can say the the 'education' is not valid. I  found quite a few professionals with no education (physician fro example) 
  
  #I will omit jobTitle because I it seems that the textual analysis tool would be more sufficient in capturing the complexities of this variable  
  step_rm(country,
          cityName,
          createDate,
          education,
          jobTitle,
          X_id,
          birthDate) |> 
  
  
  #dealing with factors 
  step_dummy(spotify,
             gender,
             instagram,
             hetero_sx) 
  
  

# -------------------------  Random Forest ------------------

rf_spec <- rand_forest(
  mode = "regression",
  engine = "randomForest",
  
  #we will tune these hyper-parameters
  
  mtry = tune(),
  trees = tune(),
  min_n = 5
)

rf_wf <- workflow(preprocessor = rec, spec = rf_spec)


rf_grid <-  grid_regular(
  
  #Considering to computation time I would reduce the max number of predictors to 10 
  mtry(range = c(1, 10)),
  trees(range = c(100, 1000)),
  levels = 5
)


 # rf_tuner <- tune_grid(rf_wf,
 #                      resamples = Tinder_validation_sets,
 #                      grid = rf_grid,
 #                     metrics = mset_reg)
 # saveRDS(rf_tuner, file = "./rds/rf_tuner.rds")

 rf_tuner <- readRDS("./rds/rf_tuner.rds")

# ----- scheming and ploting 

autoplot(rf_tuner)

# Suprisingly rsq decrease above 3 predictors, which raise so concerns until I realise tht I'm not in the domain of linear regression. 

#mtry = 1 -> 3: MAE decreases sharply,keeps decreasing beyond, stabilize around mtry =7 

#rsq = 3 -> there's no doubt that 3 is the best option across nearly all trees number.



best_of_trees<- select_by_one_std_err(rf_tuner,
                                      trees,
                                      metric = "rsq")
# tree= 100
#mtry= 3

best_of_predictors<- select_by_one_std_err(rf_tuner,
                                           mtry,
                                           metric = "rsq")
#tree = 100
#mtry = 3


# Note to Self- Always check the Y-axis scale before interpreting visual differences. What appears to be a large gap between options might actually be insignificant/


# Therefore, the selected hyperparameters are:
#tree = 100
#mtry = 3

hp_rnd_best <- tibble(
  trees = 100,
  mtry = 3
)



##------- Training and Fitting -------

rf_fit <- rf_wf |>
  finalize_workflow(parameters = hp_rnd_best) |> 
  fit(data = Tinder.train)
rf_fit 

rf_pred <- predict(rf_fit, new_data = Tinder.train) |> 
  bind_cols(Tinder.train)

mae(rf_pred, truth = nrOfGhostingsAfterInitialMessage, estimate = .pred)
#1 mae     standard         6.19

rsq(rf_pred, truth = nrOfGhostingsAfterInitialMessage, estimate = .pred)
# 1 rsq     standard       0.940

# Performance on training set: 
          # MAE: 6.19
          # R2: 0.94

## ------Fit the Test Bootstrap --------


rf_test_comp <- fit_resamples(rf_fit,
                            resamples = Tinder_comparison_sets,
                            metrics = mset_reg)

#Wait should this be done on the comparisons sets WITHOUT(!) training the model on the training set????

rf_test_comp_op2 <- rf_wf |> 
  finalize_workflow(parameters = hp_rnd_best) |> 
  fit_resamples(resamples = Tinder_comparison_sets,
                metrics = mset_reg)




#--------------------------  Booster  -----------------------
boost_spec <- boost_tree(
  mode = "regression",
  engine = "xgboost",
  
  ## Complexity
  tree_depth = tune(), # [1, Inf] limits the depth of each tree
  min_n = 5, # [1, Inf] don't split if you get less obs in a node
  
  
  loss_reduction = 0, # [0, Inf] node splitting regularization - no need for regulation since I will keep tree depth at 1 
  
  ## Gradient
  learn_rate = 0.1, # [0, 1] learning rate
  trees = tune(),
  
  ## Randomness
  mtry = 3,  # to help me comparing with Random Forest model and since it is close to the theoretical recommendation when it come to regression sqrt(p)
  sample_size = tune() # [0, 1] proportion of random data to use in each tree
)

booster_wf <- workflow(preprocessor = rec,
                       spec = boost_spec)

booester_grid <-  expand_grid(
  trees= seq(100, 2000, by = 100),
  sample_size =  seq(0, 1, by = 0.1),
  tree_depth= c(1,2,3)
  )

# Tune 

# booster_tuner <- tune_grid(booster_wf,
#                       resamples = Tinder_validation_sets,
#                       grid = booester_grid,
#                       metrics = mset_reg)
# saveRDS(booster_tuner,"./rds/booster_tuner")
booster_tuner <- readRDS("./rds/booster_tuner")
autoplot(booster_tuner)




# It seems that depth= 1 is the worst! So lets comapred depth =2 and depth =3


select_by_one_std_err(booster_tuner,sample_size, metric ="rsq")
select_by_one_std_err(booster_tuner,trees, metric ="rsq")


#Based on this, it appears that the optimal tree depth is 3. 

#it seems that best trees and sample_size both indicating 

#   trees tree_depth sample_size .config               
#   <dbl>      <dbl>       <dbl> <chr>                 
# 1   200          3         0.6 Preprocessor1_Model582


hp_boost_best <- tibble(
  trees = 200,
  tree_depth = 3,
  sample_size = 0.6
)

##------- Training and Fitting -------

boost_fit <- booster_wf |>
  finalize_workflow(parameters = hp_boost_best) |> 
  fit(data = Tinder.train)
boost_fit 

boost_pred<- predict(boost_fit, Tinder.train) |> 
  bind_cols(Tinder.train)

 mae(boost_pred,
     truth =nrOfGhostingsAfterInitialMessage ,
     estimate = .pred)
 rsq(boost_pred,
     truth =nrOfGhostingsAfterInitialMessage ,
     estimate = .pred)


# Performance on training set: 

# MAE:   3.04
# R2: 0.99 
 
 ## ------Fit the Test Bootstrap --------

booster_test_comp <- fit_resamples(boost_fit,
                            resamples = Tinder_comparison_sets,
                            metrics = mset_reg)

 
#Same Here,  should this done on the comparisons sets WITHOUT(!) training the model of the training set?

booster_test_comp_op2 <-  booster_wf |>
  finalize_workflow(parameters = hp_boost_best) |> 
  fit_resamples(resamples = Tinder_comparison_sets,
                metrics = mset_reg)


#--------------- Comparison  ---------------- 
Tinder.test_rndf.pred <- augment(rf_fit, Tinder.test)
Tinder.test_boost.pred <- augment(boost_fit, Tinder.test)

library(knitr)


test_set_performance_boost_rnf <-bind_rows(
  "rf" = Tinder.test_rndf.pred,
  "boosting" = Tinder.test_boost.pred, 
  
  .id = "Model"
) |> 
  group_by(Model) |> 
  mset_reg(nrOfGhostingsAfterInitialMessage, .pred)

kable(test_set_performance_boost_rnf,
      caption = "Performance Metrics",
      digits = 3)

#   Model    .metric .estimator .estimate
#   <chr>    <chr>   <chr>          <dbl>
# boosting	rsq	      standard	0.657
# rf	      rsq	      standard	0.576
# boosting	mae	      standard	12.8
# rf      	mae	      standard	14.5


# On average, Random Forest predicts the number of ghosting instances with an error of ±13.6, whereas Boosting’s error is ±12 - when is comes to MAE boosting is better

#R2 of the boosting model indicating that the models predictions accounts for around 68% of the variance in numbers of ghosting messages while the Random Forest model's performances reach only 66% .


# Winner - Booster! 

# comparison using Bootstraps



Booster_rnds_metrics_com <- bind_rows(
  "rf" = collect_metrics(rf_test_comp_op2, summarize = FALSE),
  "boosting" = collect_metrics(booster_test_comp_op2, summarize = FALSE),
  
  .id = "Model"
) |>
  mutate(
    Model = factor(Model, levels = c("rf", "boosting"))
  )


Booster_rnds_metrics_com |>
  group_by(id, .metric) |>
  mutate(
    best_model = case_match(.metric,
                            "mae" ~ Model[which.min(.estimate)],
                            "rsq" ~ Model[which.max(.estimate)])
  ) |>
  ggplot(aes(Model, .estimate, color = Model)) +
  facet_wrap(facets = vars(.metric), scales = "free") +
  stat_summary(size = 1, 
               position = position_nudge(0.1), 
               show.legend = FALSE) + 
  geom_point() +
  geom_line(aes(group = id, color = best_model))




#Visually, Boosting appears to have smaller variance in MAE compared to Random Forest.
# However, it's hard to determine whether Random Forest has more bootstrap samples where it outperforms Boosting on MAE.

# In contrast, when looking at R², although the variance is similar between models, it is hard to tell if Boosting outperforms Random Forest in a greater number of samples.


#When considering standard deviation:

Booster_rnds_metrics_com |>
  filter(.metric == "rsq") |>
  pivot_wider(names_from = "Model", values_from = ".estimate") |>
  mutate(
    diff = rf - boosting
  ) |>
  summarise(
    mean_diff = mean(diff),
    std_err = sd(diff),
    .lower = mean_diff - 2 * std_err,
    .upper = mean_diff + 2 * std_err
  )

# R2

#   mean_diff std_err .lower .upper
#       <dbl>   <dbl>  <dbl>  <dbl>
#    0.00350  0.0991 -0.195  0.202

#mae

  # mean_diff std_err .lower .upper
  #     <dbl>   <dbl>  <dbl>  <dbl>
  #    0.381    1.56  -2.74   3.50

#The differences in performance is not significance and is bound within the uncertainty area




#------------------- VIP ---------------------



library(DALEX)

library(DALEXtra)

# In this assignment I wont be using Shap since it only explain one. prediction at a time, and I'm more fond of the model-agnostic's Permutation-based variable importance logic.


# ------------------------Random Forest

# We first need to setup an explainer:
rndf_xplnr <- explain(rf_fit, label = "Random Forest",
                     data = select(Tinder.train, -nrOfGhostingsAfterInitialMessage),
                     y = Tinder.train$nrOfGhostingsAfterInitialMessage)

rnd_vi_perm <- model_parts(rndf_xplnr, 
                       B = 20, # Number of permutations
                       variables = NULL) # specify to only compute for some
plot(rnd_vi_perm, bar_width = 4)

# Based on the permutation-based variable importance (calculated using the increase in RMSE when each variable is shuffled), it appears that no_of_matchs is the most important predictor.

# This finding is intuitive: one cannot ignore messages if they haven’t received any matches—just like in sports, where the number of shots made depends on the number of shots taken. Without attempts, there can be no successes.

#Unfortunately, permutation-based variable importance doesn’t  provide directional insights into how the variables influence the predictions. For example, even though permuting user_age increases the RMSE, I can’t tell whether this is because younger users are more likely to ghost others, or the opposite. The method tells me that the variable matters, but not how it affects the outcome.


#Using model_profile to asses the direction of the contribution change across variables values :

plot(model_profile(rndf_xplnr,
                   variables = "no_of_matches"
                   ))

#the same goes for the other top 3 variables: no_of_msgs_received and no_of_days.

plot(model_profile(rndf_xplnr,
                   variables = "no_of_msgs_received"
                   ))

# no_of_days- it seems that the ghosting tendency have a weird trend when it comes to the number of days using the platform.

plot(model_profile(rndf_xplnr,
                   variables = "nrOfConversations"
                   ))








#------------------- Boosting

booster_xplnr <- explain(boost_fit, label = "Boosting",
                     data = select(Tinder.train, -nrOfGhostingsAfterInitialMessage),
                     y = Tinder.train$nrOfGhostingsAfterInitialMessage)


#Variable importance 

boost_vi_perm <- model_parts(booster_xplnr, 
                       B = 20, # Number of permutations
                       variables = NULL) # specify to only compute for some
plot(boost_vi_perm, bar_width = 4)

#the variable that contributed the most for the model performnce is, again, no_of_matches. However,the other 2 top variable changes to medianConversationLengthInDays and percentOfOneMesssageConversations

#medianConversationLengthInDays is pretty interesting, let׳s examine the relation between this variable to y:



plot(model_profile(booster_xplnr,
                   variables = "percentOfOneMessageConversations"
                   ))

#so there's a negative monotonic trend which indicating when raising hte percentage of one message convos the corresponding averages prediction decrease

#WHAT? - it looks that very deviants instances make the plot no so informative. let's see what happen if we bound the x axis 

plotRange_percentage_instance<-sum(Tinder.train$medianConversationLengthInDays<1)/nrow(Tinder.train) *100

#96.875% of the users in the training set has medianConversationLengthInDays<1
plot(model_profile(booster_xplnr,
                   variables = "medianConversationLengthInDays"
                   )) + xlim(0, 1) 

#Theoretically it make sense the the relation is negative monotonic one, not answering other first message is cloesly related to having 0 count of conversation - ensuming that the a day of conversation starts when the first REPLY is being sent 


#--------------- Lasso Rigression ------------------

lasso_spec <- linear_reg(
  mode = "regression", engine = "glmnet", 
  penalty = tune(), mixture = 1
)

lasso_wf <- workflow(preprocessor = rec, spec = lasso_spec)

#Setting the grid for the penalization 

lasso_grid <- grid_regular(
  penalty(range = c(-2, 7)),
  
  levels = 20
)

# Tune the model
lasso_tuned <- tune_grid(
  lasso_wf,
  resamples = Tinder_validation_sets,
  grid = lasso_grid,
  metrics = mset_reg
)



autoplot(lasso_tuned) + 
  scale_x_continuous(transform = scales::transform_log(),
                     breaks = scales::breaks_log(n = 10),
                     labels = scales::label_number(big.mark = ",")) + 
  theme(axis.text.x = element_text(angle = 20))
#I have a hinge that maybe I should include smaller values


lasso_regrid <- grid_regular(
  penalty(range = c(-4, 7)),
  
  levels = 40
)

lasso_retuned <- tune_grid(
  lasso_wf,
  resamples = Tinder_validation_sets,
  grid = lasso_regrid,
  metrics = mset_reg
)


autoplot(lasso_retuned) + 
  scale_x_continuous(transform = scales::transform_log(),
                     breaks = scales::breaks_log(n = 10),
                     labels = scales::label_number(big.mark = ",")) + 
  theme(axis.text.x = element_text(angle = 20))


(penalty_hp <- select_by_one_std_err(lasso_retuned,
                                    penalty,
                                    metric = 'rsq'))

#it's seems that no penalty is needed- However,  I would go with the first tune results for the penalty to take action -

(penalty_hp <- select_best(lasso_retuned, metric = "rsq"))


#   0.464 Preprocessor1_Model14

#now let's fit ! 

##------- Training and Fitting -------
library(glmnet)

lasso_fit <- lasso_wf |>
  finalize_workflow(parameters = penalty_hp) |> 
  fit(data = Tinder.train)
lasso_fit 

lasso_pred<- predict(lasso_fit, Tinder.train) |> 
  bind_cols(Tinder.train)

mae(lasso_pred,
    truth =nrOfGhostingsAfterInitialMessage ,
    estimate = .pred)


rsq(lasso_pred,
    truth =nrOfGhostingsAfterInitialMessage ,
    estimate = .pred)



# Performance on training set: 



# MAE:    12.5
# R2: 0.699 

#based on the training set it seems that the rsq is smalls then of the esemble trees model,
#this makes sense because it penalize the training set model to fit better OOB sets 

#let's check the performance on test set: 


Tinder.test_lasso.pred <- augment(lasso_fit, Tinder.test) 


lasso_results <-mset_reg(Tinder.test_lasso.pred,
           nrOfGhostingsAfterInitialMessage,
           .pred)

kable(lasso_results,
      caption = "Performance Metrics",
      digits = 3)

kable(Tinder.test_lasso.pred)

#not even close to assemble trees results :

# MAE  =    16.854
# R2 =      0.607


#using Mattan's plot function to see what going on: which variable has been zero-ed
plot_glmnet_coef <- function(mod, s = 0, show_intercept = FALSE) {
  b <- glmnet::coef.glmnet(mod, s = c(s, 0), exact = FALSE) |> 
    as.matrix() |> as.data.frame() |> 
    tibble::rownames_to_column("Coef")
  
  if (isFALSE(show_intercept)) {
    b <- b |> filter(Coef != "(Intercept)")
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
lasso_eng <- extract_fit_engine(lasso_fit)
coef(lasso_eng, s = penalty_hp$penalty)   

plot_glmnet_coef(lasso_eng, s= penalty_hp$penalty)
