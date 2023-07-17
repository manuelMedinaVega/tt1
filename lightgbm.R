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
dtrain <- data[sample, ]
dtest <- data[!sample, ]

#formato que necesita lightGBM
cat_feats <- c("Gender", "Driving_License", "Region_Code", "Previously_Insured", 
               "Vehicle_Age", "Vehicle_Damage", "Policy_Sales_Channel")
dtrain_f <- lgb.Dataset(data = data.matrix(dtrain[, campos_buenos, with=FALSE]), 
                      label = dtrain[, Response])
lgb.Dataset.set.categorical(dtrain_f, cat_feats)
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
valids = list(test = dtest_f)

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
#pred

#obtener el mejor umbral
ROCR_pred_test <- prediction(pred, dtest$Response)
ROCR_perf_test <- performance(ROCR_pred_test,'tpr','fpr')
plot(ROCR_perf_test,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))
cost_perf = performance(ROCR_pred_test, "cost")
#para reducir los FN a costo de incrementar los FP, obtiene un mejor accuracy
threshold <- ROCR_pred_test@cutoffs[[1]][which.min(cost_perf@y.values[[1]])] 
threshold
preds = data.table(id=dtest[,id], target=predict(modelo, dtest_pred))
colnames(preds)[1] = "id"

preds[, class := ifelse(target > threshold, 1, 0)]
#preds

#predicho
preds[class == 0, .N]
preds[class == 1, .N]

#real
dtest[Response == 0, .N]
dtest[Response == 1, .N]

confusion_matrix <- table(PredictedValue = preds$class, ActualValue = dtest$Response)
confusion_matrix

accuracy <- sum(preds$class == dtest$Response) / nrow(dtest)
accuracy

results <- data.table("model" = "basic", "auc" = modelo$best_score, "accuracy" = accuracy)
results

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
  learning_rate = 0.0053211843396746,
  feature_fraction = 0.544331366497683,
  min_data_in_leaf = 71,
  num_leaves = 40,
  seed = 491
)

#validation data
valids = list(test = dtest_f)

#train model
modelo  <- lgb.train( data= dtrain_f,
                      param= params,
                      nrounds = 1984, #alias del num_iterations
                      valids,
                      early_stopping_rounds = as.integer(50 + 5/0.0053211843396746),
                      categorical_feature = cat_feats
)

print(modelo$best_score)
print(modelo$best_iter)

#prediction
dtest_pred <- data.matrix( dtest[, campos_buenos, with=FALSE ])
pred = predict(modelo, dtest_pred)
#pred

#obtener el mejor umbral
ROCR_pred_test <- prediction(pred, dtest$Response)
ROCR_perf_test <- performance(ROCR_pred_test,'tpr','fpr')
plot(ROCR_perf_test,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))
cost_perf = performance(ROCR_pred_test, "cost")
#para reducir los FN a costo de incrementar los FP, obtiene un mejor accuracy
threshold <- ROCR_pred_test@cutoffs[[1]][which.min(cost_perf@y.values[[1]])] 
threshold
preds = data.table(id=dtest[,id], target=predict(modelo, dtest_pred))
colnames(preds)[1] = "id"

preds[, class := ifelse(target > threshold, 1, 0)]
#preds

#predicho
preds[class == 0, .N]
preds[class == 1, .N]

#real
dtest[Response == 0, .N]
dtest[Response == 1, .N]

confusion_matrix <- table(PredictedValue = preds$class, ActualValue = dtest$Response)
confusion_matrix

accuracy <- sum(preds$class == dtest$Response) / nrow(dtest)
accuracy

newResults <- data.table("model" = "exp1", "auc" = modelo$best_score, "accuracy" = accuracy)
results <- rbindlist(list(results, newResults))
results




