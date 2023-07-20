options(repos = c(CRAN = "http://cran.rstudio.com"))

install.packages("tidyverse", dependencies = TRUE)
install.packages("tidymodels", dependencies = TRUE)
install.packages("rsample", dependencies = TRUE)
install.packages("corrr", dependencies = TRUE)
install.packages("knitr", dependencies = TRUE)
install.packages("kableExtra", dependencies = TRUE)
install.packages("GGally", dependencies = TRUE)
install.packages("robustbase", dependencies = TRUE)
install.packages("reshape", dependencies = TRUE)
install.packages("dplyr", dependencies = TRUE)
install.packages("ggcorrplot", dependencies = TRUE)

# Carga de liberías
library(tidyverse)
library(tidymodels)
library(rsample)
library(corrr)
library(knitr)
library(kableExtra)
library(GGally)
library(robustbase)
library(reshape)
library(dplyr)
library(ggcorrplot)
library(plyr)
library(caret)
library(caTools)
library(ROCR)

require("pROC")
require("MLmetrics")
require("data.table")
require("lightgbm")

rm( list=ls() )  #remove all objects
gc()             #garbage collection

#LECTURA DE LOS DATOS
#data <- read.csv("C:/Users/programadorweb4/Documents/m_d_m/tt1/tt1/train.csv", 
#row.names = 'id');
#data <- read.csv("C:/Users/programadorweb4/Documents/m_d_m/tt1/tt1/train.csv");
data <- fread("C:/Users/programadorweb4/Documents/m_d_m/tt1/tt1/train.csv", 
              stringsAsFactors = TRUE);

#columnas con las que se va a entrenar
campos_buenos <- setdiff(colnames(data), c("id", "Response"))
campos_buenos

#split de training y test
set.seed(491)
nrow(data)
sample <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.7, 0.3))
dtrain_temp <- data[sample, ]
dtest <- data[!sample, ]
sample2 <- sample(c(TRUE, FALSE), nrow(dtrain_temp), replace=TRUE, prob=c(0.7, 0.3))
dtrain <- dtrain_temp[sample2, ]
dvalid <- dtrain_temp[!sample2]

#formato que necesita lightGBM
cat_feats <- c("Gender", "Driving_License", "Region_Code", "Previously_Insured", 
               "Vehicle_Age", "Vehicle_Damage", "Policy_Sales_Channel")
dtrain_f <- lgb.Dataset(data = data.matrix(dtrain[, campos_buenos, with=FALSE]), 
                      label = dtrain[, Response])
lgb.Dataset.set.categorical(dtrain_f, cat_feats)
dvalid_f <- lgb.Dataset(data = data.matrix(dvalid[, campos_buenos, with=FALSE]), 
                        label = dvalid[, Response])
lgb.Dataset.set.categorical(dvalid_f, cat_feats)
dtest_f <- lgb.Dataset.create.valid(dtrain_f, 
                      data = data.matrix(dtest[, campos_buenos, with=FALSE]), 
                      label = dtest[, Response])
lgb.Dataset.set.categorical(dtest_f, cat_feats)


#generar el modelo
#define parameters
params = list(
  objective = "binary",
  metric = "auc",
  is_unbalance = TRUE,
  seed = 491
)

#validation data
valids = list(test = dvalid_f)

#train model
modelo  <- lgb.train( data= dtrain_f,
                      param= params,
                      nrounds = 100, #alias del num_iterations
                      valids,
                      early_stopping_rounds = 10,
                      categorical_feature = cat_feats
)

print(modelo$best_score)
print(modelo$best_iter)

#prediction
dtest_pred <- data.matrix( dtest[, campos_buenos, with=FALSE ])
pred = predict(modelo, dtest_pred)

roc_obj <- roc(dtest$Response, pred)

# Obtener los valores de F1 para diferentes umbrales
coordenadas <- coords(roc_obj, "best", ret=c("threshold", "f1"))

# Obtener el umbral y valor máximo de F1
umbral_max_f1 <- coordenadas$threshold
umbral_max_f1

predicciones <- ifelse(pred >= umbral_max_f1, 1, 0)

# Calcular el valor de F1
f1 <- F1_Score(predicciones, dtest$Response)
f1

confusionMatrix(table(predicciones, dtest$Response))


#umbral de máximo accuracy
ROCR_pred_test <- prediction(pred, dtest$Response)
ROCR_perf_test <- ROCR::performance(ROCR_pred_test,'tpr','fpr')
plot(ROCR_perf_test,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))
cost_perf = ROCR::performance(ROCR_pred_test, "cost")
#para reducir los FN a costo de incrementar los FP, obtiene un mejor accuracy
umbral_max_acc <- ROCR_pred_test@cutoffs[[1]][which.min(cost_perf@y.values[[1]])] 
umbral_max_acc

predicciones_2 <- ifelse(pred >= umbral_max_acc, 1, 0)

confusionMatrix(table(predicciones_2, dtest$Response))

tb_importancia  <-  as.data.table( lgb.importance(modelo) ) 
tb_importancia


#entrenando un nuevo modelo con los hps obtenidos de la bo del exp1:
  #200 iters
  #objetivo: auc
  #hps básicos

params = list(
  objective = "binary",
  metric = "auc",
  is_unbalance = TRUE,
  max_depth = -1,
  min_gain_to_split = 0.0,
  lambda_l1= 0.0,
  lambda_l2= 0.0,
  learning_rate = 0.0153081329445143,
  feature_fraction = 0.48186684122117,
  min_data_in_leaf = 23,
  num_leaves = 34,
  seed = 491
)

#validation data
valids = list(test = dvalid_f)

#train model
modelo  <- lgb.train( data= dtrain_f,
                      param= params,
                      nrounds = 543, #alias del num_iterations
                      valids,
                      early_stopping_rounds = as.integer(50 + 5/0.0153081329445143),
                      categorical_feature = cat_feats
)

print(modelo$best_score)
print(modelo$best_iter)

pred2 = predict(modelo, dtest_pred)

roc_obj <- roc(dtest$Response, pred2)

# Obtener los valores de F1 para diferentes umbrales
coordenadas <- coords(roc_obj, "best", ret=c("threshold", "f1"))

# Obtener el umbral y valor máximo de F1
umbral_max_f1_2 <- coordenadas$threshold
umbral_max_f1_2

predicciones_2 <- ifelse(pred >= umbral_max_f1_2, 1, 0)

# Calcular el valor de F1
f1 <- F1_Score(predicciones_2, dtest$Response)
f1

confusionMatrix(table(predicciones_2, dtest$Response))


#umbral de máximo accuracy
ROCR_pred_test <- prediction(pred2, dtest$Response)
ROCR_perf_test <- ROCR::performance(ROCR_pred_test,'tpr','fpr')
plot(ROCR_perf_test, colorize=TRUE, print.cutoffs.at=seq(0.1,by=0.1))
cost_perf = ROCR::performance(ROCR_pred_test, "cost")
#para reducir los FN a costo de incrementar los FP, obtiene un mejor accuracy
umbral_max_acc_2 <- ROCR_pred_test@cutoffs[[1]][which.min(cost_perf@y.values[[1]]) + 1] 
umbral_max_acc_2

predicciones_3 <- ifelse(pred >= umbral_max_acc_2, 1, 0)

confusionMatrix(table(predicciones_3, dtest$Response))

tb_importancia  <-  as.data.table( lgb.importance(modelo) ) 
tb_importancia
