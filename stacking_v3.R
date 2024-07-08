## PACOTES ##

library(tidyverse)	# manipulacao de dados
library(tidymodels) # ferramentas de ML
library(extrasteps) # complemento para pre-processamento dos dados
library(plsmod)     # necessario para usar modelo pls
library(stacks)     # stacking
library(tictoc)     # registrar o tempo de execução de comandos
library(janitor)    # limpeza de dados



##### CARREGANDO/LIMPANDO OS DADOS #####

df<- read.csv("winequality_red.csv") %>% 
  data.frame()

df<- df %>% 
  dplyr::rename("y"="quality")

glimpse(df)






##### SPLIT TRAIN/TEST/VALIDATION #####

set.seed(0)
split<- initial_split(df, prop=0.8, strata=y)

df_train<- training(split)    # usado para cross-validation
df_test<- testing(split)      # usado para verificar desempenho

folds<- vfold_cv(df_train, v=10, strata=y)




##### PRÉ-PROCESSAMENTO #####

receita<- recipe(y ~ . , data = df_train) %>%
  #step_rm(...) %>%                                           # variaveis removidas
  step_filter_missing(all_predictors(), threshold = 0.3) %>%  # variaveis +30% de faltantes
  step_zv(all_predictors()) %>%                               # variaveis sem variabilidade
  #step_YeoJohnson(all_numeric_predictors()) %>%              # normalizar variaveis
  #step_impute_knn(all_predictors()) %>%                      # imputando faltantes
  #step_naomit() %>%                                          # deletando faltantes
  step_normalize(all_numeric_predictors()) %>%                # padronizar variaveis
  #step_robust(all_numeric_predictors()) %>%                  # padronizacao robusta
  #step_corr(all_numeric_predictors(),threshold=tune()) %>%   # removendo variaveis correlacionadas
  #step_other(all_nominal_predictors(),threshold=tune()) %>%  # cria a categoria "outros"
  #step_novel(all_nominal_predictors()) %>%                   # novas categorias
  step_dummy(all_nominal_predictors())                        # variaveis dummy


receita_pls<- recipe(y ~ . , data = df_train) %>%
  #step_rm(...) %>%                                           # variaveis removidas
  step_filter_missing(all_predictors(), threshold = 0.3) %>%  # variaveis +30% de faltantes
  step_zv(all_predictors()) %>%                               # variaveis sem variabilidade
  #step_YeoJohnson(all_numeric_predictors()) %>%              # normalizar variaveis
  #step_impute_knn(all_predictors()) %>%                      # imputando faltantes
  #step_naomit() %>%                                          # deletando faltantes
  step_normalize(all_numeric_predictors()) %>%                # padronizar variaveis
  #step_robust(all_numeric_predictors()) %>%                  # padronizacao robusta
  #step_corr(all_numeric_predictors(),threshold=tune()) %>%   # removendo variaveis correlacionadas
  #step_other(all_nominal_predictors(),threshold=tune()) %>%  # cria a categoria "outros"
  #step_novel(all_nominal_predictors()) %>%                   # novas categorias
  step_dummy(all_nominal_predictors()) %>%                    # variaveis dummy
  step_pls(all_numeric_predictors(),
           outcome="y",
           num_comp=tune(),
           predictor_prop = tune())                           # reducao de dimensao





##### MODELOS #####

model_knn<- nearest_neighbor(neighbors = tune(),
                             dist_power = tune()) %>%
  set_engine("kknn") %>%
  set_mode("regression")


model_pls<- parsnip::pls(num_comp = tune(),
                         predictor_prop = tune()) %>%
  set_engine("mixOmics") %>%
  set_mode("regression")


model_net<- linear_reg(penalty = tune(),
                       mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")


model_rfo<- rand_forest(mtry = tune(),
                        trees = 10000,
                        min_n = tune()) %>%
  set_engine("ranger") %>%
  set_mode("regression")


model_xgb<- boost_tree(mtry = tune(),
                       trees = 10000,
                       min_n = tune(),
                       loss_reduction = tune(),
                       learn_rate = tune(),
                       stop_iter = 100) %>%
  set_engine("xgboost") %>%
  set_mode("regression")


model_svm<- svm_rbf(cost = tune(),
                    margin = tune(),
                    rbf_sigma = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("regression")


# model_mlp <- mlp(epochs = 50,
#                  hidden_units = tune(),
#                  dropout = tune(),
#                  learn_rate = tune(),
#                  activation = "relu") %>%
#   set_engine("brulee", stop_iter=5) %>%
#   set_mode("regression")


model_mlp <- mlp(epochs = 50,
                 hidden_units = tune(),
                 penalty = tune(),
                 activation = "relu") %>%
  set_engine("nnet") %>%
  set_mode("regression")




##### WORKFLOW #####

wf_knn<- workflow() %>%
  add_recipe(receita_pls) %>%
  add_model(model_knn)

wf_pls<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_pls)

wf_net<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_net)

wf_rfo<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_rfo)

wf_xgb<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_xgb)

wf_svm<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_svm)

wf_mlp<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_mlp)




##### TUNAGEM DE HIPERPARAMETROS - BAYESIAN SEARCH #####

## QUANTIDADES AUXILIARES

df_receita<- df_train %>% prep() %>% juice()
n<- df_receita %>% nrow()
k<- df_receita %>% ncol()



## KNN - K NEAREST NEIGHBORS

tic()
tune_knn<- tune_bayes(wf_knn,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(rmse),
                      param_info = parameters(neighbors(range=c(1,min(200,trunc(0.25*n)))),
                                              dist_power(range=c(1,2)),
                                              num_comp(range=c(1,min(100,trunc(0.75*k)))),
                                              predictor_prop(range=c(0,1)))
)
toc()
# 223.33 sec elapsed (~ 4 min)




## PLS - PARTIAL LEAST SQUARE

tic()
tune_pls<- tune_bayes(wf_pls,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(rmse),
                      param_info = parameters(num_comp(range=c(1,min(100,trunc(0.75*k)))),
                                              predictor_prop(range=c(0,1)))
)
toc()
# 101.03 sec elapsed (~ 2 min)




## NET - ELASTIC NET

tic()
tune_net<- tune_bayes(wf_net,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(rmse),
                      param_info = parameters(penalty(range=c(-10,5)),
                                              mixture(range=c(0,1)))
)
toc()
# 31.97 sec elapsed (~ 0.5 min)




## RFO - RANDOM FOREST

tic()
tune_rfo<- tune_bayes(wf_rfo,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(rmse),
                      param_info = parameters(mtry(range=c(1,trunc(0.9*k))),
                                              min_n(range=c(1,min(200,trunc(0.25*n)))))
)
toc()
# 1354.19 sec elapsed




## XGB - XGBOOSTING

tic()
tune_xgb<- tune_bayes(wf_xgb,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(rmse),
                      param_info = parameters(mtry(range=c(1,trunc(0.9*k))),
                                              min_n(range=c(1,min(200,trunc(0.25*n)))),
                                              loss_reduction(range=c(-10,5)),
                                              learn_rate(range=c(-10,0)))
)
toc()
# 1659.64 sec elapsed




## SVM - SUPPORT VECTOR MACHINE

tic()
tune_svm<- tune_bayes(wf_svm,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(rmse),
                      param_info = parameters(cost(range=c(-10,5)),
                                              svm_margin(range=c(0,0.5)),
                                              rbf_sigma(range=c(-10,5)))
)
toc()
# 63.94 sec elapsed




## MLP - MULTILAYER PERCEPTRON

tic()
tune_mlp<- tune_bayes(wf_mlp,
                      resamples = folds,
                      initial = 10,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(rmse),
                      param_info = parameters(hidden_units(range=c(8,2048)),
                                              penalty(range=c(-10,5)))
                      
)
toc()
# 1739.82 sec elapsed (~ 29 min)




## VISUALIZANDO OS MELHORES MODELOS (BEST RMSE)

show_best(tune_knn, metric="rmse", n=3)
show_best(tune_gam, metric="rmse", n=3)
show_best(tune_pls, metric="rmse", n=3)
show_best(tune_net, metric="rmse", n=3)
show_best(tune_rfo, metric="rmse", n=3)
show_best(tune_xgb, metric="rmse", n=3)
show_best(tune_svm, metric="rmse", n=3)
show_best(tune_mlp, metric="rmse", n=3)





##### PREPARANDO STACKING #####

stack_ensemble_data<- stacks() %>% 
  add_candidates(tune_knn, name="KNN") %>%
  add_candidates(tune_pls, name="PLS") %>% 
  add_candidates(tune_net, name="Elastic_Net") %>% 
  add_candidates(tune_rfo, name="Random_Forest") %>% 
  add_candidates(tune_xgb, name="XGB") %>% 
  add_candidates(tune_svm, name="SVM") %>% 
  add_candidates(tune_mlp, name="MLP")

stack_ensemble_data
stack_ensemble_data_best



##### AJUSTANDO STACKING #####

set.seed(0)
stack_ensemble_model<- stack_ensemble_data %>% 
  blend_predictions(penalty = exp(-15:5),
                    mixture = seq(0,1,by=0.05), # 0=RIDGE; 1=LASSO
                    control = control_grid(save_pred=TRUE,
                                           save_workflow=TRUE),
                    non_negative = TRUE,
                    metric = metric_set(rmse))

#autoplot(stack_ensemble_model)
autoplot(stack_ensemble_model,type = "weights") + ggtitle("")

stack_ensemble_model$penalty




##### FINALIZANDO O MODELO #####

stack_ensemble_trained<- stack_ensemble_model %>% 
  fit_members()

stack_ensemble_trained





##### FINALIZANDO MODELOS INDIVIDUAIS #####

wf_knn_trained<- wf_knn %>% finalize_workflow(select_best(tune_knn,metric="mae")) %>% fit(df_train)
wf_pls_trained<- wf_pls %>% finalize_workflow(select_best(tune_pls,metric="mae")) %>% fit(df_train)
wf_net_trained<- wf_net %>% finalize_workflow(select_best(tune_net,metric="mae")) %>% fit(df_train)
wf_rfo_trained<- wf_rfo %>% finalize_workflow(select_best(tune_rfo,metric="mae")) %>% fit(df_train)
wf_xgb_trained<- wf_xgb %>% finalize_workflow(select_best(tune_xgb,metric="mae")) %>% fit(df_train)
wf_svm_trained<- wf_svm %>% finalize_workflow(select_best(tune_svm,metric="mae")) %>% fit(df_train)
wf_mlp_trained<- wf_mlp %>% finalize_workflow(select_best(tune_mlp,metric="mae")) %>% fit(df_train)




## SALVANDO OS MODELOS

# saveRDS(wf_knn_trained,"wf_knn_trained.rds")
# saveRDS(wf_dcr_trained,"wf_dcr_trained.rds")
# saveRDS(wf_pls_trained,"wf_pls_trained.rds")
# saveRDS(wf_net_trained,"wf_net_trained.rds")
# saveRDS(wf_rfo_trained,"wf_rfo_trained.rds")
# saveRDS(wf_xgb_trained,"wf_xgb_trained.rds")
# saveRDS(wf_svm_trained,"wf_svm_trained.rds")
# saveRDS(wf_mlp_trained,"wf_mlp_trained.rds")
# #saveRDS(wf_tbn_trained,"wf_tbn_trained.rds")
# saveRDS(stack_ensemble_model,"stack_ensemble_model.rds")
# saveRDS(stack_ensemble_trained,"stack_ensemble_trained.rds")
# saveRDS(stack_ensemble_model_best,"stack_ensemble_model_best.rds")
# saveRDS(stack_ensemble_trained_best,"stack_ensemble_trained_best.rds")



## CARREGANDO OS MODELOS SALVOS

# wf_knn_trained<- readRDS("wf_knn_trained.rds")
# wf_dcr_trained<- readRDS("wf_dcr_trained.rds")
# wf_pls_trained<- readRDS("wf_pls_trained.rds")
# wf_net_trained<- readRDS("wf_net_trained.rds")
# wf_rfo_trained<- readRDS("wf_rfo_trained.rds")
# wf_xgb_trained<- readRDS("wf_xgb_trained.rds")
# wf_svm_trained<- readRDS("wf_svm_trained.rds")
# wf_mlp_trained<- readRDS("wf_mlp_trained.rds")
# #wf_tbn_trained<- readRDS("wf_tbn_trained.rds")
# stack_ensemble_model<- readRDS("stack_ensemble_model.rds")
# stack_ensemble_trained<- readRDS("stack_ensemble_trained.rds")
# stack_ensemble_model_best<- readRDS("stack_ensemble_model_best.rds")
# stack_ensemble_trained_best<- readRDS("stack_ensemble_trained_best.rds")





### VALIDATION ###

# PREDIZENDO DADOS TESTE

pred_knn<- wf_knn_trained %>% predict(df_test)
pred_gam<- wf_gam_trained %>% predict(df_test)
pred_pls<- wf_pls_trained %>% predict(df_test)
pred_net<- wf_net_trained %>% predict(df_test)
pred_rfo<- wf_rfo_trained %>% predict(df_test)
pred_xgb<- wf_xgb_trained %>% predict(df_test)
pred_svm<- wf_svm_trained %>% predict(df_test)
pred_mlp<- wf_mlp_trained %>% predict(df_test)
pred_stc<- stack_ensemble_trained %>% predict(df_test)
pred_stc_best<- stack_ensemble_trained_best %>% predict(df_test)

df_pred<- cbind.data.frame(df_test$y,
                           pred_knn,
                           pred_gam,
                           pred_pls,
                           pred_net,
                           pred_rfo,
                           pred_xgb,
                           pred_svm,
                           pred_mlp,
                           pred_stc,
                           pred_stc_best)

colnames(df_pred)<- c("y",
                      "knn",
                      "gam",
                      "pls",
                      "net",
                      "rfo",
                      "xgb",
                      "svm",
                      "mlp",
                      "stc",
                      "stc_best")

df_pred %>% head()    # VISUALIZANDO PROBABILIDADES




#####  VERIFICANDO MEDIDAS DE PREDICAO  #####

# MEDIDAS

medidas<- rbind(
  df_pred %>% metrics(y,knn) %>% dplyr::select(.estimate) %>% t(),
  df_pred %>% metrics(y,gam) %>% dplyr::select(.estimate) %>% t(),
  df_pred %>% metrics(y,pls) %>% dplyr::select(.estimate) %>% t(),
  df_pred %>% metrics(y,net) %>% dplyr::select(.estimate) %>% t(),
  df_pred %>% metrics(y,rfo) %>% dplyr::select(.estimate) %>% t(),
  df_pred %>% metrics(y,xgb) %>% dplyr::select(.estimate) %>% t(),
  df_pred %>% metrics(y,svm) %>% dplyr::select(.estimate) %>% t(),
  df_pred %>% metrics(y,mlp) %>% dplyr::select(.estimate) %>% t(),
  #df_pred %>% metrics(y,tbn) %>% dplyr::select(.estimate) %>% t(),
  df_pred %>% metrics(y,stc) %>% dplyr::select(.estimate) %>% t(),
  df_pred %>% metrics(y,stc_best) %>% dplyr::select(.estimate) %>% t())

colnames(medidas)<- c("rmse","rsq","mae")
rownames(medidas)<- c("knn",
                      "gam",
                      "pls",
                      "net",
                      "rfo",
                      "xgb",
                      "svm",
                      "mlp",
                      #"tbn",
                      "stc",
                      "stc_best")
medidas




#####  FEATURE/VARIABLE IMPORTANCE  #####


## STACKING

tic()
explainer_stc<- explain_tidymodels(model=stack_ensemble_trained_best,
                                   data=dplyr::select(df_train,-y),
                                   y=df_train$y,
                                   #label="Stacking")
                                   label="")
toc()


tic()
vi_stc<- model_parts(explainer_stc,
                     type="variable_importance")
toc()
# 50075.43 sec elapsed (~ 14 hr)

# saveRDS(vi_stc,"vi_stc.rds")
# vi_stc<- readRDS("vi_stc.rds")



vi_stc %>% as.data.frame() %>% 
  dplyr::filter(variable!="_full_model_",
                variable!="_baseline_") %>% 
  dplyr::rename(importance = dropout_loss) %>% 
  dplyr::select(-c(permutation,label)) %>% 
  group_by(variable) %>% 
  summarise(importance = mean(importance)) %>% 
  dplyr::arrange(desc(importance))

vi_stc %>% plot(show_boxplots=FALSE,
                title="Variable Importance",
                subtitle="",
                max_vars=10)


tic()
profile_stc<- explainer_stc %>% model_profile()
toc()


profile_stc %>% plot(geom = "aggregates")
profile_stc %>% plot(geom = "profiles")
profile_stc %>% plot(geom = "points")

profile_stc %>% plot(geom = "aggregates",
                     #title="",
                     subtitle="")

#explainer_stc %>% model_profile(variables=c("alcohol","sulphates")) %>% plot(geom = "profiles")
#explainer_stc %>% model_profile(variables=c("alcohol","sulphates")) %>% plot(geom = "points")


profile_stc$agr_profiles %>% 
  data.frame() %>% 
  filter(X_vname_ %in% c("alcohol")) %>% 
  ggplot(aes(x = X_x_,
             y = X_yhat_)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), se = FALSE) +
  #geom_smooth(method = "loess", se = FALSE) + 
  #geom_line(data = spline_int, aes(x = x, y = y)) +
  #ylab("ROA estimado") +
  #xlab("SIZE") +
  #ylim(c(5,9)) +
  theme_bw()


df_alcohol<- profile_stc$agr_profiles %>% 
  data.frame() %>% 
  dplyr::filter(X_vname_ %in% c("alcohol")) %>% 
  dplyr::rename(y=X_yhat_) %>% 
  dplyr::rename(alcohol=X_x_) %>% 
  dplyr::select(c(y,alcohol))

df_alcohol %>% 
  ggplot(aes(x = alcohol,y = y)) +
  geom_point() +
  #geom_smooth(method = "lm", formula = y ~ poly(x, 3), se = FALSE) +
  #geom_smooth(method = "loess", se = FALSE) + 
  geom_smooth(formula = y ~ s(x, k = 6), method = "gam", se = FALSE) + 
  ylab("Probabilidade de Good") +
  xlab("alcohol") +
  ggtitle("Ceteris Paribus profile") + 
  theme_bw() 


