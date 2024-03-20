## PACOTES ##

library(tidyverse)	# manipulacao de dados
library(tidymodels) # ferramentas de ML
library(stacks)     # stacking
library(tictoc)     # registrar o tempo de execução de comandos



## DADOS ##

df<- read.csv("winequality_red.csv") %>% 
  data.frame()

glimpse(df)



##### SPLIT TRAIN/TEST/VALIDATION #####

set.seed(0)
split<- initial_split(df, strata=quality)

df.train<- training(split)
df.test<- testing(split)

folds<- vfold_cv(df.train, v=3, strata=quality)




##### PRÉ-PROCESSAMENTO #####

receita<- recipe(quality ~ . , data = df.train) %>%
  step_filter_missing(all_predictors(),threshold = 0.4) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())




##### MODELOS #####

model_knn<- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("regression")

model_net<- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

model_rfo<- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression")




##### WORKFLOW #####

wf_knn<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_knn)

wf_net<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_net)

wf_rfo<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_rfo)



##### TUNAGEM DE HIPERPARAMETROS - BAYESIAN SEARCH #####

## KNN

tic()
tune_knn<- tune_bayes(wf_knn,
                      resamples = folds,
                      initial = 10,
                      #control = control_stack_bayes(),
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(rmse),
                      param_info = parameters(neighbors(range=c(1,30)))
)
toc()
# 15.89 sec elapsed




## NET - ELASTIC NET

tic()
tune_net<- tune_bayes(wf_net,
                      resamples = folds,
                      initial = 10,
                      #control = control_stack_bayes(),
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(rmse),
                      param_info = parameters(penalty(range=c(-10,10)),
                                              mixture(range=c(0,1)))
)
toc()
# 40.75 sec elapsed




## RFO - RANDOM FOREST

tic()
tune_rfo<- tune_bayes(wf_rfo,
                      resamples = folds,
                      initial = 10,
                      #control = control_stack_bayes(),
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(rmse),
                      param_info = parameters(mtry(range=c(1,10)),
                                              trees(range=c(50,10000)),
                                              min_n(range=c(1,40)))
)
toc()
# 1206 sec elapsed (~20min)





##### PREPARANDO STACKING #####

stack_ensemble_data<- stacks() %>% 
  add_candidates(tune_knn) %>% 
  add_candidates(tune_net) %>% 
  add_candidates(tune_rfo)

stack_ensemble_data





##### AJUSTANDO STACKING #####

set.seed(0)
stack_ensemble_model<- stack_ensemble_data %>% 
  blend_predictions(penalty = 10^(-9:-1),
                    mixture = 1, # 0=RIDGE; 1=LASSO
                    control = control_grid(),
                    non_negative = FALSE,
                    metric = metric_set(rmse))

autoplot(stack_ensemble_model)
autoplot(stack_ensemble_model,type = "weights")

stack_ensemble_model




##### FINALIZANDO O MODELO #####

stack_ensemble_model<- stack_ensemble_model %>% 
  fit_members()

stack_ensemble_model




##### FINALIZANDO MODELOS INDIVIDUAIS #####

wf_train_knn<- wf_knn %>% finalize_workflow(select_best(tune_knn)) %>% fit(df.train)
wf_train_net<- wf_net %>% finalize_workflow(select_best(tune_net)) %>% fit(df.train)
wf_train_rfo<- wf_rfo %>% finalize_workflow(select_best(tune_rfo)) %>% fit(df.train)




### VALIDATION ###

# PREDIZENDO DADOS TESTE

pred.knn<- predict(wf_train_knn, df.test)
pred.net<- predict(wf_train_net, df.test)
pred.rfo<- predict(wf_train_rfo, df.test)
pred.stc<- predict(stack_ensemble_model, df.test)

predicao<- data.frame(df.test$quality,
                      pred.knn,
                      pred.net,
                      pred.rfo,
                      pred.stc)
colnames(predicao)<- c("y","knn","net","rfo","stc")
head(predicao)

RMSE<- cbind(rmse(predicao,y,knn)$.estimate,
             rmse(predicao,y,net)$.estimate,
             rmse(predicao,y,rfo)$.estimate,
             rmse(predicao,y,stc)$.estimate)

colnames(RMSE)<- c("knn","net","rfo","stc")
RMSE


