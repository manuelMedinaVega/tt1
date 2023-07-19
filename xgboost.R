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
data_labels <- data %>% select(Response)

data_numeric <- data %>% select(c(Age, Annual_Premium, Vintage))
Gender <- model.matrix(~Gender-1, data)
Driving_License <- model.matrix(~Driving_License-1, data)
Region_Code <- model.matrix(~Region_Code-1, data)
Previously_Insured <- model.matrix(~Previously_Insured-1, data)
Vehicle_Age <- model.matrix(~Vehicle_Age-1, data)
Vehicle_Damage <- model.matrix(~Vehicle_Damage-1, data)
Policy_Sales_Channel <- model.matrix(~Policy_Sales_Channel-1, data)

data_numeric <- cbind(data_numeric, Gender, Driving_License, Region_Code,
                      Previously_Insured, Vehicle_Age, Vehicle_Damage, 
                      Policy_Sales_Channel)
data_matrix <- data.matrix(data_numeric)

set.seed(491)
nrow(data_matrix)
sample <- sample(c(TRUE, FALSE), nrow(data_matrix), replace=TRUE, prob=c(0.7, 0.3))
dtrain <- data_matrix[sample, ]
dtest <- data_matrix[!sample, ]
train_labels <- data_labels[sample, ]
test_labels <- data_labels[!sample, ]

#dejo los datos en el formato que necesita XGBoost
dtrain_f <- xgb.DMatrix(data = dtrain, 
                        label = train_labels$Response)
dtest_f <- xgb.DMatrix(data = dtest, 
                       label = test_labels$Response)

negative_cases <- train_labels[Response == 0, .N]
negative_cases
positive_cases <- train_labels[Response == 1, .N]
positive_cases

#MODELO CON PARÁMETROS POR DEFAULT
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
                      nrounds= 100
)

#aplico el modelo a los datos nuevos
pred = predict(modelo, dtest_f)

roc_obj <- roc(test_labels$Response, pred)

# Obtener los valores de F1 para diferentes umbrales
coordenadas <- coords(roc_obj, "best", ret=c("threshold", "f1"))

# Obtener el umbral y valor máximo de F1
umbral_max_f1 <- coordenadas$threshold
umbral_max_f1

predicciones <- ifelse(pred >= umbral_max_f1, 1, 0)

# Calcular el valor de F1
f1 <- F1_Score(predicciones, test_labels$Response)
f1

confusionMatrix(table(predicciones, test_labels$Response))

#maximizando accuracy
ROCR_pred_test <- prediction(pred, test_labels$Response)
ROCR_perf_test <- ROCR::performance(ROCR_pred_test, 'tpr', 'fpr')
plot(ROCR_perf_test, colorize=TRUE, print.cutoffs.at=seq(0.1,by=0.1))
cost_perf = ROCR::performance(ROCR_pred_test, "cost")
#para reducir los FN a costo de incrementar los FP, obtiene un mejor accuracy
umbral_max_acc <- ROCR_pred_test@cutoffs[[1]][which.min(cost_perf@y.values[[1]])] 
umbral_max_acc

predicciones_2 <- ifelse(pred >= umbral_max_acc, 1, 0)

confusionMatrix(table(predicciones_2, test_labels$Response))

importance_matrix <- xgb.importance(model = modelo)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)


#CON HPS DE BO
fganancia_logistic_xgboost   <- function( probs, datos) 
{
  etiquetas = getinfo(datos, "label")
  roc_obj <- roc(etiquetas, probs)
  
  # Obtener los valores de F1 para diferentes umbrales
  coordenadas <- coords(roc_obj, "best", ret=c("threshold", "f1"))
  
  # Obtener el umbral y valor máximo de F1
  umbral_max_f1 <- coordenadas$threshold
  
  predicciones <- ifelse(probs >= umbral_max_f1, 1, 0)
  
  # Calcular el valor de F1
  f1 <- F1_Score(predicciones, etiquetas)
  
  return(  list("metric" = "ganancia", "value" = f1 ) )
}
modelo  <- xgb.train( data= dtrain_f,
                      param= list( objective=       "binary:logistic",
                                   max_depth=           6,#profundidad máxima de un árbo, default=6
                                   min_child_weight=    9,#suma mínima de pesos de la instancia (hessian) necesaria en un hijo, default=1
                                   eta=                 0.954883581841952,#alias de learning_rate, default=0.3
                                   colsample_bytree=    0.612642310194921,#subsample ratio of columns de cada arbol, default=1
                                   gamma=               86,#alias de min_split_loss, Minimum loss reduction required to make a further partition on a leaf node of the tree, default=0
                                   alpha=               0.0,#L1 regularization
                                   lambda=              0.0,#L2 regularization
                                   subsample=           0.333725101807572,#subsample ratio de las instancias de entrenamiento
                                   scale_pos_weight=    negative_cases/positive_cases,#controla el balance de pesos positivo y negativo, útil para clases desbalanceadas
                                   eval_metric=fganancia_logistic_xgboost
                      ),
                      #base_score= mean( getinfo(dtrain, "label")),
                      nrounds= 11
)

#aplico el modelo a los datos nuevos
pred_2 = predict(modelo, dtest_f)

roc_obj_2 <- roc(test_labels$Response, pred_2)

# Obtener los valores de F1 para diferentes umbrales
coordenadas_2 <- coords(roc_obj_2, "best", ret=c("threshold", "f1"))

# Obtener el umbral y valor máximo de F1
umbral_max_f1_2 <- coordenadas_2$threshold
umbral_max_f1_2

predicciones_3 <- ifelse(pred_2 >= umbral_max_f1_2, 1, 0)

# Calcular el valor de F1
f1 <- F1_Score(predicciones_3, test_labels$Response)
f1

confusionMatrix(table(predicciones_3, test_labels$Response))
