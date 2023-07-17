# Optimizacion Bayesiana de hiperparametros de  lightgbm, con el metodo 
# TRADICIONAL de los hiperparametros originales de lightgbm
# 5-fold cross validation

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("rlist")

require("lightgbm")

#paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")

options(error = function() { 
  traceback(20); 
  options(error = NULL); 
  stop("exiting after script error") 
})


#Aqui se cargan los hiperparametros
hs <- makeParamSet( 
  makeNumericParam("learning_rate",    lower=    0.005, upper=    0.3),
  makeNumericParam("feature_fraction", lower=    0.3  , upper=    1.0),# % de sampleo de columnas, default = 1.0
  makeIntegerParam("min_data_in_leaf", lower=    0L   , upper=  3000L),# num mínimo de data en una hoja, default = 20
  makeIntegerParam("num_leaves",       lower=    16L  , upper=  1024L)# num max de hojas en cada weak learner, default = 31
  #makeIntegerParam("bagging_freq",     lower=    100  , upper=   1000),# frecuencia para el bagging, realiza bagging cada bagging_freq iteraciones
  #makeNumericParam("bagging_fraction", lower=    0.5  , upper=    1.0),# % de filas usadas por el arbol en cada iteración
  #makeIntegerParam("max_bin",          lower=    20   , upper=    255)
)

#defino los parametros de la corrida, en una lista, la variable global  PARAM
#  muy pronto esto se leera desde un archivo formato .yaml
PARAM  <- list()

PARAM$experimento  <- "HT_auc_basic_params_2000_iters_metric_f1"

PARAM$input$dataset       <- "./datasets/train.csv"

PARAM$trainingstrategy$undersampling  <-  1.0 
PARAM$trainingstrategy$semilla_azar   <- 491

PARAM$hyperparametertuning$iteraciones <- 2000
PARAM$hyperparametertuning$xval_folds  <- 5

PARAM$hyperparametertuning$semilla_azar  <- 491

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

fganancia_logistic_lightgbm  <- function( probs, datos) 
{
  labels = getinfo(datos,"label")
  F1 = get.max_f1(probs,labels)[1]
  return( list( "name"= "ganancia", 
                "value"=  F1,
                "higher_better"= TRUE ) )
}

#------------------------------------------------------------------------------
#esta funcion solo puede recibir los parametros que se estan optimizando
#el resto de los parametros se pasan como variables globales, la semilla del mal ...

EstimarGanancia_lightgbm  <- function( x )
{
  gc()  #libero memoria
  
  #llevo el registro de la iteracion por la que voy
  GLOBAL_iteracion  <<- GLOBAL_iteracion + 1
  
  kfolds  <- PARAM$hyperparametertuning$xval_folds   # cantidad de folds para cross validation
  
  param_basicos  <- list( objective= "binary",
                          metric= "custom",
                          is_unbalance = TRUE,
                          first_metric_only= TRUE,
                          boost_from_average= TRUE,
                          feature_pre_filter= FALSE,
                          verbosity= -100,
                          max_depth=  -1,         # profundidad máxima de cada árbol, default = -1 (no limitar), por ahora lo dejo fijo
                          #max_bin=255,            # número máximo de bins para discretizar valores de un feature, default=255
                          min_gain_to_split= 0.0, # minimal gain para hacer un split, default = 0.0, por ahora, lo dejo fijo
                          lambda_l1= 0.0,         #por ahora, lo dejo fijo
                          lambda_l2= 0.0,         #por ahora, lo dejo fijo
                          num_iterations= 9999,   #un numero muy grande, lo limita early_stopping_rounds
                          force_row_wise= TRUE,   #para que los alumnos no se atemoricen con tantos warning
                          seed= PARAM$hyperparametertuning$semilla_azar
  )
  
  #el parametro discolo, que depende de otro
  param_variable  <- list(  early_stopping_rounds= as.integer(50 + 5/x$learning_rate) )
  
  param_completo  <- c( param_basicos, param_variable, x )
  
  set.seed( PARAM$hyperparametertuning$semilla_azar )
  modelocv  <- lgb.cv( data= dtrain,
                       eval= fganancia_logistic_lightgbm,
                       stratified= TRUE, #sobre el cross validation
                       nfold= kfolds,    #folds del cross validation
                       param= param_completo,
                       verbose= -100
  )
  
  #obtengo la ganancia
  ganancia  <- unlist(modelocv$record_evals$valid$ganancia$eval)[ modelocv$best_iter ]
  
  param_completo$num_iterations <- modelocv$best_iter  #asigno el mejor num_iterations
  param_completo["early_stopping_rounds"]  <- NULL     #elimino de la lista el componente  "early_stopping_rounds"
  
  #Voy registrando la importancia de variables
  if( ganancia >  GLOBAL_gananciamax )
  {
    GLOBAL_gananciamax  <<- ganancia
    modelo  <- lgb.train( data= dtrain,
                          param= param_completo,
                          verbose= -100
    )
    
    tb_importancia  <- as.data.table( lgb.importance(modelo ) )
    archivo_importancia  <- paste0( "impo_", GLOBAL_iteracion,".txt")
    fwrite( tb_importancia,
            file= archivo_importancia,
            sep= "\t" )
  }
  
  
  #el lenguaje R permite asignarle ATRIBUTOS a cualquier variable
  attr(ganancia ,"extras" )  <- list("num_iterations"= modelocv$best_iter)  #esta es la forma de devolver un parametro extra
  
  #logueo 
  xx  <- param_completo
  xx$ganancia  <- ganancia   #le agrego la ganancia
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
dataset  <- fread( PARAM$input$dataset, stringsAsFactors = TRUE )

#creo la carpeta donde va el experimento
# HT  representa  Hiperparameter Tuning
dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( paste0( "./exp/", PARAM$experimento, "/"), showWarnings = FALSE )
setwd( paste0( "./exp/", PARAM$experimento, "/") )   #Establezco el Working Directory DEL EXPERIMENTO

#en estos archivos quedan los resultados
kbayesiana  <- paste0( PARAM$experimento, ".RDATA" )
klog        <- paste0( PARAM$experimento, ".txt" )


GLOBAL_iteracion  <- 0   #inicializo la variable global
GLOBAL_gananciamax <- -1 #inicializo la variable global

#si ya existe el archivo log, traigo hasta donde llegue
if( file.exists(klog) )
{
  tabla_log  <- fread( klog )
  GLOBAL_iteracion  <- nrow( tabla_log )
  GLOBAL_gananciamax  <- tabla_log[ , max( ganancia ) ]
}


#los campos que se van a utilizar
campos_buenos <- setdiff(colnames(dataset), c("id", "Response"))
GLOBAL_cat_feats <- c("Gender", "Driving_License", "Region_Code", "Previously_Insured", 
               "Vehicle_Age", "Vehicle_Damage", "Policy_Sales_Channel")
set.seed( PARAM$trainingstrategy$semilla_azar )
sample <- sample(c(TRUE, FALSE), nrow(dataset), replace=TRUE, prob=c(0.7, 0.3))
dtrain <- dataset[sample, ]
dtest <- dataset[!sample, ]

#dejo los datos en el formato que necesita LightGBM
dtrain  <- lgb.Dataset( data= data.matrix(dtrain[, campos_buenos, with=FALSE]),
                        label= dtrain[, Response],
                        free_raw_data= FALSE  )
lgb.Dataset.set.categorical(dtrain, GLOBAL_cat_feats)


#Aqui comienza la configuracion de la Bayesian Optimization
funcion_optimizar  <- EstimarGanancia_lightgbm   #la funcion que voy a maximizar

configureMlr( show.learner.output= FALSE)

#configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
obj.fun  <- makeSingleObjectiveFunction(
  fn=       funcion_optimizar, #la funcion que voy a maximizar
  minimize= FALSE,   #estoy Maximizando la ganancia
  noisy=    TRUE,
  par.set=  hs,     #definido al comienzo del programa
  has.simple.signature = FALSE   #paso los parametros en una lista
)

ctrl  <- makeMBOControl( save.on.disk.at.time= 600,  save.file.path= kbayesiana)  #se graba cada 600 segundos
ctrl  <- setMBOControlTermination(ctrl, iters= PARAM$hyperparametertuning$iteraciones )   #cantidad de iteraciones
ctrl  <- setMBOControlInfill(ctrl, crit= makeMBOInfillCritEI() )

#establezco la funcion que busca el maximo
surr.km  <- makeLearner("regr.km", predict.type= "se", covtype= "matern3_2", control= list(trace= TRUE))

#inicio la optimizacion bayesiana
if( !file.exists( kbayesiana ) ) {
  run  <- mbo(obj.fun, learner= surr.km, control= ctrl)
} else {
  run  <- mboContinue( kbayesiana )   #retomo en caso que ya exista
}


quit( save="no" )

