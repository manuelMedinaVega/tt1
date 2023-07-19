# Este script esta pensado para correr en Google Cloud
#    8 vCPU
#   16 GB memoria RAM
#  256 GB disco
# eliminar la vm cuando termine de correr

# Optimizacion Bayesiana de hiperparametros de  xgboost, con el metodo TRADICIONAL de los hiperparametros originales de xgboost
# 5-fold cross validation
# la probabilidad de corte es un hiperparametro

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("rlist")

require("xgboost")

#paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")
require("caret")
require("pROC")
require("MLmetrics")
require("dplyr")


kBO_iter  <- 100   #cantidad de iteraciones de la Optimizacion Bayesiana

#Aqui se cargan los hiperparametros
hs <- makeParamSet( 
  makeNumericParam("eta",              lower=  0.01 , upper=    1.0),   #equivalente a learning rate
  makeNumericParam("colsample_bytree", lower=  0.2  , upper=    1.0),   #equivalente a feature_fraction
  makeIntegerParam("min_child_weight", lower=  0L   , upper=   50L),    #groseramente equivalente a  min_data_in_leaf
  makeIntegerParam("max_depth",        lower=  2L   , upper=   50L),    #profundidad del arbol, NO es equivalente a num_leaves
  makeIntegerParam("gamma",            lower=  0L   , upper=   100L),    
  makeNumericParam("subsample",        lower= 0.25, 1)
)

ksemilla_azar  <- 491  #Aqui poner la propia semilla

#------------------------------------------------------------------------------
#graba a un archivo los componentes de lista
#para el primer registro, escribe antes los titulos

loguear  <- function( reg, arch=NA, folder="./exp/", ext=".txt", verbose=TRUE )
{
  archivo  <- arch
  if( is.na(arch) )  archivo  <- paste0(  folder, substitute( reg), ext )
  
  if( !file.exists( archivo ) )  #Escribo los titulos
  {
    linea  <- paste0( "fecha\t", 
                      paste( list.names(reg), collapse="\t" ), "\n" )
    
    cat( linea, file=archivo )
  }
  
  linea  <- paste0( format(Sys.time(), "%Y%m%d %H%M%S"),  "\t",     #la fecha y hora
                    gsub( ", ", "\t", toString( reg ) ),  "\n" )
  
  cat( linea, file=archivo, append=TRUE )  #grabo al archivo
  
  if( verbose )  cat( linea )   #imprimo por pantalla
}
#------------------------------------------------------------------------------
#esta funcion calcula internamente la ganancia de la prediccion probs

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
#------------------------------------------------------------------------------
#esta funcion solo puede recibir los parametros que se estan optimizando
#el resto de los parametros se pasan como variables globales, la semilla del mal ...

EstimarGanancia_xgboost  <- function( x )
{
  gc()  #libero memoria
  
  #llevo el registro de la iteracion por la que voy
  GLOBAL_iteracion  <<- GLOBAL_iteracion + 1
  
  kfolds  <- 5   # cantidad de folds para cross validation
  
  negative_cases <- dtrain[Response == 0, .N]
  #negative_cases
  positive_cases <- dtrain[Response == 1, .N]
  #positive_cases
  
  #otros hiperparmetros, que por ahora dejo en su valor default
  param_basicos  <- list( #gamma=                0.0,  #por ahora, lo dejo fijo, equivalente a  min_gain_to_split
    alpha=                0.0,  #por ahora, lo dejo fijo, equivalente a  lambda_l1
    lambda=               0.0,  #por ahora, lo dejo fijo, equivalente a  lambda_l2
    #subsample=            1.0,  #por ahora, lo dejo fijo
    tree_method=       "auto",  #por ahora lo dejo fijo, pero ya lo voy a cambiar a "hist"
    grow_policy=  "depthwise",  #ya lo voy a cambiar a "lossguide"
    max_bin=            256,    #por ahora fijo
    max_leaves=           0,    #ya lo voy a cambiar
    scale_pos_weight=     negative_cases/positive_cases   #por ahora, lo dejo fijo
  )
  
  param_completo  <- c( param_basicos, x )
  
  set.seed( 491 )
  modelocv  <- xgb.cv( objective= "binary:logistic",
                       data= dtrain_f,
                       feval= fganancia_logistic_xgboost,
                       disable_default_eval_metric= TRUE,
                       maximize= TRUE,
                       stratified= TRUE,     #sobre el cross validation
                       nfold= kfolds,        #folds del cross validation
                       nrounds= 9999,        #un numero muy grande, lo limita early_stopping_rounds
                       early_stopping_rounds= as.integer(50 + 5/x$eta),
                       base_score= mean( getinfo(dtrain_f, "label")),  
                       param= param_completo,
                       verbose= -100
  )
  
  #obtengo la ganancia
  ganancia   <- unlist( modelocv$evaluation_log[ , test_ganancia_mean] )[ modelocv$best_iter ] 
  
  ganancia_normalizada  <- ganancia* kfolds     #normailizo la ganancia
  
  #el lenguaje R permite asignarle ATRIBUTOS a cualquier variable
  attr(ganancia_normalizada ,"extras" )  <- list("nrounds"= modelocv$best_iter)  #esta es la forma de devolver un parametro extra
  
  param_completo$nrounds <- modelocv$best_iter  #asigno el mejor nrounds
  param_completo["early_stopping_rounds"]  <- NULL     #elimino de la lista el componente  "early_stopping_rounds"
  
  #logueo 
  xx  <- param_completo
  xx$ganancia  <- ganancia_normalizada   #le agrego la ganancia
  xx$iteracion <- GLOBAL_iteracion
  loguear( xx, arch= klog )
  
  return( ganancia )
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Aqui empieza el programa

#Aqui se debe poner la carpeta de la computadora local
setwd("~/buckets/b1/")   #Establezco el Working Directory

#cargo el dataset donde voy a entrenar el modelo
data  <- fread("./datasets/train.csv")
data <- data %>%
  mutate(Gender = as.factor(Gender), 
         Driving_License = as.factor(Driving_License),
         Region_Code = as.factor(Region_Code), 
         Previously_Insured = as.factor(Previously_Insured), 
         Vehicle_Age = as.factor(Vehicle_Age),
         Vehicle_Damage = as.factor(Vehicle_Damage),
         Policy_Sales_Channel = as.factor(Policy_Sales_Channel))

#creo la carpeta donde va el experimento
# HT  representa  Hiperparameter Tuning
dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( "./exp/HT_xgboost_exp1/", showWarnings = FALSE )
setwd("./exp/HT_xgboost_exp1/")   #Establezco el Working Directory DEL EXPERIMENTO


#en estos archivos quedan los resultados
kbayesiana  <- "HT_xgboost_exp1.RDATA"
klog        <- "HT_xgboost_exp1.txt"


GLOBAL_iteracion  <- 0   #inicializo la variable global

#si ya existe el archivo log, traigo hasta donde llegue
if( file.exists(klog) )
{
  tabla_log  <- fread( klog )
  GLOBAL_iteracion  <- nrow( tabla_log )
}

set.seed(491)
sample <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.7, 0.3))
dtrain <- data[sample, ]
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


#Aqui comienza la configuracion de la Bayesian Optimization
funcion_optimizar  <- EstimarGanancia_xgboost   #la funcion que voy a maximizar

configureMlr( show.learner.output= FALSE)

#configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
#por favor, no desesperarse por lo complejo
obj.fun  <- makeSingleObjectiveFunction(
  fn=       funcion_optimizar, #la funcion que voy a maximizar
  minimize= FALSE,   #estoy Maximizando la ganancia
  noisy=    TRUE,
  par.set=  hs,     #definido al comienzo del programa
  has.simple.signature = FALSE   #paso los parametros en una lista
)

ctrl  <- makeMBOControl( save.on.disk.at.time= 600,  save.file.path= kbayesiana)  #se graba cada 600 segundos
ctrl  <- setMBOControlTermination(ctrl, iters= kBO_iter )   #cantidad de iteraciones
ctrl  <- setMBOControlInfill(ctrl, crit= makeMBOInfillCritEI() )

#establezco la funcion que busca el maximo
surr.km  <- makeLearner("regr.km", predict.type= "se", covtype= "matern3_2", control= list(trace= TRUE))

#inicio la optimizacion bayesiana
if( !file.exists( kbayesiana ) ) {
  run  <- mbo(obj.fun, learner= surr.km, control= ctrl)
} else {
  run  <- mboContinue( kbayesiana )   #retomo en caso que ya exista
}

