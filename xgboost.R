# XGBoost  sabor original ,  cambiando algunos de los parametros

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("xgboost")
require("caret")
require("pROC")
require("MLmetrics")

#cargo el dataset donde voy a entrenar
data  <- fread("C:/Users/programadorweb4/Documents/m_d_m/tt1/tt1/train.csv", stringsAsFactors= TRUE)
data <- data %>%
  mutate(Gender = as.factor(Gender), 
         Driving_License = as.factor(Driving_License),
         Region_Code = as.factor(Region_Code), 
         Previously_Insured = as.factor(Previously_Insured), 
         Vehicle_Age = as.factor(Vehicle_Age),
         Vehicle_Damage = as.factor(Vehicle_Damage),
         Policy_Sales_Channel = as.factor(Policy_Sales_Channel))

set.seed(491)
nrow(data)
sample <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.7, 0.3))
dtrain <- data[sample, ]
dtest <- data[!sample, ]


data_numeric <- dtrain %>% select(c(Age, Annual_Premium, Vintage))
Gender <- model.matrix(~Gender-1, dtrain)
Driving_License <- model.matrix(~Driving_License-1, dtrain)
Region_Code <- model.matrix(~Region_Code-1, dtrain)
Previously_Insured <- model.matrix(~Previously_Insured-1, dtrain)
Vehicle_Age <- model.matrix(~Vehicle_Age-1, dtrain)
Vehicle_Damage <- model.matrix(~Vehicle_Damage-1, dtrain)
Policy_Sales_Channel <- model.matrix(~Policy_Sales_Channel-1, dtrain)

data_numeric <- cbind(data_numeric, Gender, Driving_License, Region_Code,
                      Previously_Insured, Vehicle_Age, Vehicle_Damage, 
                      Policy_Sales_Channel)
data_matrix <- data.matrix(data_numeric)

#dejo los datos en el formato que necesita XGBoost
dtrain_f <- xgb.DMatrix(data = data_matrix, 
                        label = dtrain[, Response])

negative_cases <- dtrain[Response == 0, .N]
negative_cases
positive_cases <- dtrain[Response == 1, .N]
positive_cases

#genero el modelo con los parametros por default
modelo  <- xgb.train( data= dtrain_f,
                      param= list( objective=       "binary:logistic",
                                   max_depth=           6,#profundidad máxima de un árbo, default=6
                                   min_child_weight=    1,#suma mínima de pesos de la instancia (hessian) necesaria en un hijo, default=1
                                   eta=                 0.3,#alias de learning_rate, default=0.3
                                   colsample_bytree=    1.0,#subsample ratio of columns de cada arbol, default=1
                                   gamma=               0.0,#alias de min_split_loss, Minimum loss reduction required to make a further partition on a leaf node of the tree, default=0
                                   alpha=               0.0,#L1 regularization
                                   lambda=              0.0,#L2 regularization
                                   subsample=           1.0,#subsample ratio de las instancias de entrenamiento
                                   scale_pos_weight=    negative_cases/positive_cases,#controla el balance de pesos positivo y negativo, útil para clases desbalanceadas
                                   eval_metric='auc'
                     ),
                      #base_score= mean( getinfo(dtrain, "label")),
                      nrounds= 34
)

data_numeric <- dtest %>% select(c(Age, Annual_Premium, Vintage))
Gender <- model.matrix(~Gender-1, dtest)
Driving_License <- model.matrix(~Driving_License-1, dtest)
Region_Code <- model.matrix(~Region_Code-1, dtest)
Previously_Insured <- model.matrix(~Previously_Insured-1, dtest)
Vehicle_Age <- model.matrix(~Vehicle_Age-1, dtest)
Vehicle_Damage <- model.matrix(~Vehicle_Damage-1, dtest)
Policy_Sales_Channel <- model.matrix(~Policy_Sales_Channel-1, dtest)

data_numeric <- cbind(data_numeric, Gender, Driving_License, Region_Code,
                      Previously_Insured, Vehicle_Age, Vehicle_Damage, 
                      Policy_Sales_Channel)
data_matrix <- data.matrix(data_numeric)

#dejo los datos en el formato que necesita XGBoost
dtest_f <- xgb.DMatrix(data = data_matrix, 
                        label = dtest[, Response])

#aplico el modelo a los datos nuevos
pred = predict(modelo, dtest_f)

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


ROCR_pred_test <- prediction(pred, dtest$Response)
ROCR_perf_test <- performance(ROCR_pred_test,'tpr','fpr')
plot(ROCR_perf_test,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))
cost_perf = performance(ROCR_pred_test, "cost")
#para reducir los FN a costo de incrementar los FP, obtiene un mejor accuracy
threshold <- ROCR_pred_test@cutoffs[[1]][which.min(cost_perf@y.values[[1]])] 
threshold
preds = data.table(id=dtest[,id], target=predict(modelo, dtest_f))
colnames(preds)[1] = "id"

preds[, class := ifelse(target > umbral_max_f1, 1, 0)]
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

importance_matrix <- xgb.importance(model = modelo)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

